"""Stage 2 - merge FBref and Transfermarkt into one player-season table.

Output: data/processed/player_seasons.parquet, one row per player-season with
stats, position group, age, league, minutes, market value and contract length.
Everything downstream reads this file and nothing downstream touches a network.
"""

from __future__ import annotations

import argparse
import logging
import re
import zipfile

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config, fetch_fbref_archive, fetch_tm_contracts, name_matching, paths

logger = logging.getLogger(__name__)

IDENTITY_COLUMNS = ("league", "season", "team", "player", "pos", "nation", "age", "born")


def to_numeric(df: pd.DataFrame, exclude: tuple[str, ...] = IDENTITY_COLUMNS) -> pd.DataFrame:
        """Coerce every stat column to a number.

        FBref's HTML tables arrive as strings, with thousands separators on the
        larger counting stats ("1,234" progressive-pass distance). Under pandas 3
        those stay as the new string dtype and silently refuse to divide, so this
        has to happen before any per-90 arithmetic.
        """
        out = df.copy()
        for col in out.columns:
                if col in exclude:
                        continue
                series = out[col]
                if series.dtype == "object" or str(series.dtype).startswith("str"):
                        series = series.astype("string").str.replace(",", "", regex=False)
                out[col] = pd.to_numeric(series, errors="coerce")
        return out


def parse_fbref_age(value: object) -> float:
        """FBref renders age as `years-days`, e.g. `25-123`.

            >>> parse_fbref_age("25-123")
            25.33...
            >>> parse_fbref_age("31")
            31.0
        """
        if value is None or (isinstance(value, float) and pd.isna(value)):
                return np.nan
        text = str(value).strip()
        if match := re.fullmatch(r"(\d+)-(\d+)", text):
                return int(match.group(1)) + int(match.group(2)) / 365.25
        try:
                return float(text)
        except ValueError:
                return np.nan


def load_fbref() -> pd.DataFrame:
        # Live FBref table with its empty advanced-stat cells filled from the
        # Wayback Machine, when that stage has run. See fetch_fbref_archive.
        df = fetch_fbref_archive.load_player_table().reset_index()

        rename = {
                "standard__flat__pos": "pos",
                "standard__flat__age": "age_raw",
                "standard__flat__born": "born",
                "standard__flat__nation": "nation",
                "standard__playing_time__min": "minutes",
                "standard__playing_time__90s": "nineties",
                "standard__playing_time__mp": "matches_played",
                "standard__playing_time__starts": "starts",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

        for required in ("pos", "minutes", "born"):
                if required not in df.columns:
                        raise SystemExit(
                                f"Expected column {required!r} missing from the FBref table. "
                                "Run `python run_pipeline.py --stage inspect-columns` and check "
                                "reports/fbref_columns.txt against the rename map in src/merge.py."
                        )

        df["age"] = df["age_raw"].map(parse_fbref_age) if "age_raw" in df else np.nan
        df["born"] = pd.to_numeric(df["born"], errors="coerce")
        df = to_numeric(df, exclude=IDENTITY_COLUMNS + ("age_raw", "pos"))

        if "nineties" not in df.columns or df["nineties"].isna().all():
                df["nineties"] = df["minutes"] / 90.0

        if config.EXCLUDE_GOALKEEPERS:
                before = len(df)
                df = df[~df["pos"].fillna("").str.contains("GK")]
                logger.info("Dropped %s goalkeeper rows", before - len(df))

        logger.info("FBref player-seasons: %s rows", len(df))
        return df


def attach_team_possession(df: pd.DataFrame) -> pd.DataFrame:
        """Join team possession %, used for the possession adjustment in features."""
        try:
                teams = fetch_fbref_archive.load_team_table().reset_index()
        except FileNotFoundError:
                logger.warning("No team stats file; possession adjustment will be skipped.")
                df["team_possession"] = np.nan
                return df

        poss_cols = [c for c in teams.columns if c.endswith("__poss")]
        if not poss_cols:
                logger.warning("No possession column in team stats; adjustment skipped.")
                df["team_possession"] = np.nan
                return df

        teams = teams[["league", "season", "team", poss_cols[0]]].rename(
                columns={poss_cols[0]: "team_possession"}
        )
        teams["team_possession"] = pd.to_numeric(teams["team_possession"], errors="coerce")
        merged = df.merge(teams, on=["league", "season", "team"], how="left")
        logger.info(
                "Attached team possession to %.1f%% of rows",
                100 * merged["team_possession"].notna().mean(),
        )
        return merged


def attach_market_values(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
        """Join Transfermarkt identity, then the season-appropriate market value."""
        linked = matches[matches["decision"].isin(["accepted", "override"])]
        linked = linked[["fbref_player", "fbref_born", "tm_player_id", "tm_name", "tm_sub_position", "decision"]]

        df = df.merge(
                linked.rename(columns={"decision": "match_decision"}),
                left_on=["player", "born"],
                right_on=["fbref_player", "fbref_born"],
                how="left",
        ).drop(columns=["fbref_player", "fbref_born"])

        players = pd.read_parquet(paths.TM_PLAYERS)
        df = confirm_by_club(df, players)
        attrs = [
                c for c in ("player_id", "date_of_birth", "foot", "height_in_cm",
                                        "country_of_citizenship", "contract_expiration_date")
                if c in players.columns
        ]
        df = df.merge(
                players[attrs].rename(columns={"player_id": "tm_player_id"}),
                on="tm_player_id",
                how="left",
        )

        # Market value as at mid-season, not as at today.
        valuations = pd.read_parquet(paths.TM_VALUATIONS)
        df["season_midpoint"] = df["season"].map(name_matching.season_midpoint)

        value_rows: list[pd.DataFrame] = []
        for season, chunk in df.groupby("season", sort=False):
                target = name_matching.season_midpoint(season)
                snapshot = (
                        valuations.assign(gap=(valuations["date"] - target).abs())
                        .sort_values("gap")
                        .drop_duplicates("player_id")
                        .rename(columns={
                                "market_value_in_eur": "market_value_eur",
                                "date": "valuation_date",
                        })[["player_id", "market_value_eur", "valuation_date"]]
                )
                value_rows.append(
                        chunk.merge(
                                snapshot, left_on="tm_player_id", right_on="player_id", how="left"
                        ).drop(columns=["player_id"])
                )
        df = pd.concat(value_rows, ignore_index=True)

        df = attach_contracts(df)

        # Prefer the exact Transfermarkt date of birth over FBref's age string.
        if "date_of_birth" in df.columns:
                precise = (df["season_midpoint"] - df["date_of_birth"]).dt.days / 365.25
                df["age"] = precise.where(precise.notna(), df["age"])

        logger.info(
                "Market value present on %.1f%% of player-seasons",
                100 * df["market_value_eur"].notna().mean(),
        )
        return df


def load_club_seasons() -> pd.DataFrame:
        """Minutes per (Transfermarkt player, season end year, club). Cached.

        Built from the appearances table in the transfermarkt-datasets zip. A
        match in July or later counts toward the season ending the next year.
        """
        if paths.TM_CLUB_SEASONS.exists():
                return pd.read_parquet(paths.TM_CLUB_SEASONS)
        with zipfile.ZipFile(paths.TM_ZIP) as zf:
                name = next(n for n in zf.namelist() if n.rsplit("/", 1)[-1].startswith("appearances.csv"))
                with zf.open(name) as fh:
                        app = pd.read_csv(
                                fh, usecols=["player_id", "player_club_id", "date", "minutes_played"],
                                compression="gzip" if name.endswith(".gz") else None,
                        )
        app["date"] = pd.to_datetime(app["date"], errors="coerce")
        app = app.dropna(subset=["date"])
        app["season_end"] = app["date"].dt.year + (app["date"].dt.month >= 7).astype(int)
        out = app.groupby(["player_id", "season_end", "player_club_id"], as_index=False)["minutes_played"].sum()
        out.to_parquet(paths.TM_CLUB_SEASONS, index=False)
        logger.info("Wrote %s (%s player-club-seasons)", paths.TM_CLUB_SEASONS, len(out))
        return out


#: How far the best same-club candidate must out-score the next one before a
#: surname-only match is trusted.
SURNAME_MARGIN = 15


def _same_surname(fbref_name: str, tm_name: str) -> bool:
        """Do the two names share a family-name token (anything after the first)?

        Catches nickname vs full-name pairs that the name scorer rates below
        NAME_MATCH_ACCEPT - "andy robertson" / "andrew robertson", "max kilman"
        / "maximilian kilman", "obite n dicka" / "evan ndicka" - while rejecting
        pairs that share only a first name ("dario fuentes" / "dario benavides").
        """
        fb, tm = fbref_name.split(), tm_name.split()
        if len(fb) < 2 or len(tm) < 2:
                return False
        fb_family = fb[1:] + ["".join(fb[1:])]
        tm_family = tm[1:] + ["".join(tm[1:])]
        return any(fuzz.ratio(a, b) >= 85 for a in fb_family for b in tm_family)


def confirm_by_club(df: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
        """Check every name match against where the player actually played.

        The name join keys on (name, birth year) alone, and football is full of
        namesakes born a year apart: two Adama Traorés, two Vitinhas, Igor
        Julio and Igor Vekic. Transfermarkt's appearances table says which
        club each player played for in each season, so a match is confirmed
        only if the Transfermarkt player appeared for this row's club.

        Each FBref club-season is first mapped to a Transfermarkt club by
        majority vote among its players whose names match almost exactly. Then
        per row (so namesakes at different clubs are told apart):

          * confirmed   - the matched player played for that club that season
          * corrected   - he did not, but a same-club player within the birth
                          year tolerance scores >= NAME_MATCH_ACCEPT, or shares
                          a surname and clearly beats every other candidate
                          (see _same_surname); use him
          * recovered   - the row had no match, and such a player exists
          * unconfirmed - no such player, but the names match almost exactly
                          (>= 90); kept, since the gap is usually in the
                          appearances data (Hertha 2022-23)
          * unlinked    - no such player and the names differ too; the match
                          is dropped rather than trusted ("Jose Luis Garcia
                          Vaya" had been linked to a different Jose Garcia)
          * override    - manual overrides are never second-guessed

        Every row that is not plainly confirmed is written to
        paths.MATCH_CLUB_CHECK for review.
        """
        club_seasons = load_club_seasons()
        tm = players.set_index("player_id")
        tm_norm = tm["name"].map(name_matching.normalise_name)

        df = df.copy()
        season_end = df["season"].map(name_matching.season_end_year)
        names = df["player"].map(name_matching.normalise_name)
        ids = pd.to_numeric(df["tm_player_id"], errors="coerce")
        close = pd.Series([
                pd.notna(t) and fuzz.ratio(n, tm_norm.get(int(t), "")) >= 90 for n, t in zip(names, ids)
        ], index=df.index)

        votes = (
                pd.DataFrame({"league": df["league"], "season": df["season"], "team": df["team"],
                              "player_id": ids, "season_end": season_end})[close]
                .merge(club_seasons, on=["player_id", "season_end"])
                .groupby(["league", "season", "team", "player_club_id"]).size().rename("n").reset_index()
                .sort_values("n").drop_duplicates(["league", "season", "team"], keep="last")
                .set_index(["league", "season", "team"])["player_club_id"]
        )
        clubs = [votes.get(key) for key in zip(df["league"], df["season"], df["team"])]
        played = set(zip(club_seasons["player_id"], club_seasons["season_end"], club_seasons["player_club_id"]))
        roster = club_seasons.groupby(["season_end", "player_club_id"])["player_id"].agg(list)

        status, report = [], []
        for pos, (idx, club) in enumerate(zip(df.index, clubs)):
                current = ids.iat[pos]
                if df.at[idx, "match_decision"] == "override":
                        status.append("override")
                        continue
                if club is None or pd.isna(club):
                        status.append("no_club")
                        continue
                if pd.notna(current) and (int(current), season_end.iat[pos], club) in played:
                        status.append("confirmed")
                        continue

                born = df.at[idx, "born"]
                pool = [
                        p for p in roster.get((season_end.iat[pos], club), [])
                        if p in tm.index and pd.notna(born) and pd.notna(tm.at[p, "birth_year"])
                        and abs(tm.at[p, "birth_year"] - born) <= config.BIRTH_YEAR_TOLERANCE
                ]
                scored = sorted(
                        ((fuzz.token_set_ratio(names.iat[pos], tm_norm[p]), fuzz.ratio(names.iat[pos], tm_norm[p]), p) for p in pool),
                        reverse=True,
                )
                by_name = bool(scored) and scored[0][0] >= config.NAME_MATCH_ACCEPT
                if by_name or (scored and _same_surname(names.iat[pos], tm_norm[scored[0][2]])
                               and scored[0][0] - (scored[1][0] if len(scored) > 1 else 0) >= SURNAME_MARGIN):
                        new = scored[0][2]
                        label = "corrected" if pd.notna(current) else "recovered"
                        report.append({"league": df.at[idx, "league"], "season": df.at[idx, "season"],
                                       "team": df.at[idx, "team"], "player": df.at[idx, "player"], "born": born,
                                       "status": label, "via": "name" if by_name else "surname",
                                       "old_tm_name": df.at[idx, "tm_name"],
                                       "new_tm_name": tm.at[new, "name"], "new_tm_player_id": new})
                        df.at[idx, "tm_player_id"] = new
                        df.at[idx, "tm_name"] = tm.at[new, "name"]
                        df.at[idx, "tm_sub_position"] = tm.at[new, "sub_position"]
                        status.append(label)
                else:
                        if pd.isna(current):
                                status.append("unmatched")
                                continue
                        label = "unconfirmed" if close.iat[pos] else "unlinked"
                        report.append({"league": df.at[idx, "league"], "season": df.at[idx, "season"],
                                       "team": df.at[idx, "team"], "player": df.at[idx, "player"], "born": born,
                                       "status": label, "via": None, "old_tm_name": df.at[idx, "tm_name"],
                                       "new_tm_name": None, "new_tm_player_id": None})
                        if label == "unlinked":
                                df.loc[idx, ["tm_player_id", "tm_name", "tm_sub_position"]] = np.nan
                        status.append(label)

        df["club_check"] = status
        pd.DataFrame(report).to_csv(paths.MATCH_CLUB_CHECK, index=False)
        logger.info("Club check on the name join: %s", df["club_check"].value_counts().to_dict())
        return df.drop(columns=["match_decision"])


def attach_contracts(df: pd.DataFrame) -> pd.DataFrame:
        """Contract years remaining at each season's midpoint.

        Uses the dated observations from fetch_tm_contracts, so a 2023-24 row
        gets the contract that was running in January 2024 rather than the one
        the player holds today. Rows with no trustworthy observation stay NaN;
        the value model imputes them. Without the contracts stage's output this
        falls back to today's contract and says so, because that number is
        biased upward for every past season.
        """
        if paths.TM_CONTRACTS.exists():
                observations = pd.read_parquet(paths.TM_CONTRACTS)
                dated = fetch_tm_contracts.contract_at_midpoint(
                        df, observations, fetch_tm_contracts.load_transfers()
                )
                df = df.join(dated)
                end = df["contract_end"]
                by_season = df.groupby("season")["contract_end"].apply(lambda s: s.notna().mean())
                logger.info(
                        "Contract at season midpoint found for %s",
                        ", ".join(f"{s} {v:.0%}" for s, v in by_season.items()),
                )
        else:
                logger.warning(
                        "%s missing: using CURRENT contracts, which overstate years remaining "
                        "for past seasons. Run `python run_pipeline.py --stage fetch-tm-contracts`.",
                        paths.TM_CONTRACTS.name,
                )
                end = df.get("contract_expiration_date", pd.Series(pd.NaT, index=df.index))
        df["contract_years_remaining"] = (
                (end - df["season_midpoint"]).dt.days / 365.25
        ).clip(lower=0, upper=6)
        return df


def assign_position_group(df: pd.DataFrame) -> pd.DataFrame:
        """Transfermarkt sub_position first, FBref pos as a flagged fallback."""
        primary = df["tm_sub_position"].map(config.SUB_POSITION_TO_GROUP)
        fallback = df["pos"].map(config.FBREF_POS_TO_GROUP)

        df["position_group"] = primary.where(primary.notna(), fallback)
        df["position_source"] = np.where(
                primary.notna(), "transfermarkt", np.where(fallback.notna(), "fbref_fallback", "unknown")
        )

        unknown = int(df["position_group"].isna().sum())
        if unknown:
                logger.info("Dropping %s rows with no assignable position group", unknown)
        df = df[df["position_group"].notna()]

        logger.info(
                "Position source: %s",
                df["position_source"].value_counts().to_dict(),
        )
        logger.info("Group sizes: %s", df["position_group"].value_counts().to_dict())
        return df


def run(force: bool = False) -> pd.DataFrame:
        paths.ensure_dirs()
        if paths.PLAYER_SEASONS.exists() and not force:
                logger.info("%s exists; use --force to rebuild.", paths.PLAYER_SEASONS)
                return pd.read_parquet(paths.PLAYER_SEASONS)

        fbref = load_fbref()
        fbref = attach_team_possession(fbref)

        if not paths.TM_PLAYERS.exists():
                raise SystemExit("Run `python run_pipeline.py --stage fetch-transfermarkt` first.")
        tm = pd.read_parquet(paths.TM_PLAYERS)

        distinct = fbref[["player", "born"]].drop_duplicates()
        logger.info("Matching %s distinct FBref players against %s Transfermarkt players",
                                len(distinct), len(tm))
        matches = name_matching.match_players(distinct, tm)
        matches.to_parquet(paths.NAME_MATCHES, index=False)
        name_matching.write_reports(matches)

        df = attach_market_values(fbref, matches)
        df = assign_position_group(df)

        # A unique, stable key for every row - used as the app's lookup and as the
        # grouping key for cross-validation.
        df["player_season_id"] = (
                df["player"].str.replace(r"\s+", "_", regex=True)
                + "__" + df["season"].astype(str)
                + "__" + df["team"].astype(str).str.replace(r"\s+", "_", regex=True)
        )

        df.to_parquet(paths.PLAYER_SEASONS, index=False)
        logger.info("Wrote %s (%s rows, %s columns)", paths.PLAYER_SEASONS, *df.shape)
        return df


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
