"""Stage 2 - merge FBref and Transfermarkt into one player-season table.

Output: data/processed/player_seasons.parquet, one row per player-season with
stats, position group, age, league, minutes, market value and contract length.
Everything downstream reads this file and nothing downstream touches a network.
"""

from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from . import config, name_matching, paths

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
        if not paths.FBREF_WIDE.exists():
                raise SystemExit("Run `python run_pipeline.py --stage fetch-fbref` first.")
        df = pd.read_parquet(paths.FBREF_WIDE).reset_index()

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
        if not paths.FBREF_TEAM.exists():
                logger.warning("No team stats file; possession adjustment will be skipped.")
                df["team_possession"] = np.nan
                return df

        teams = pd.read_parquet(paths.FBREF_TEAM).reset_index()
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
        linked = linked[["fbref_player", "fbref_born", "tm_player_id", "tm_name", "tm_sub_position"]]

        df = df.merge(
                linked,
                left_on=["player", "born"],
                right_on=["fbref_player", "fbref_born"],
                how="left",
        ).drop(columns=["fbref_player", "fbref_born"])

        players = pd.read_parquet(paths.TM_PLAYERS)
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

        if "contract_expiration_date" in df.columns:
                df["contract_years_remaining"] = (
                        (df["contract_expiration_date"] - df["season_midpoint"]).dt.days / 365.25
                ).clip(lower=0, upper=6)
        else:
                df["contract_years_remaining"] = np.nan

        # Prefer the exact Transfermarkt date of birth over FBref's age string.
        if "date_of_birth" in df.columns:
                precise = (df["season_midpoint"] - df["date_of_birth"]).dt.days / 365.25
                df["age"] = precise.where(precise.notna(), df["age"])

        logger.info(
                "Market value present on %.1f%% of player-seasons",
                100 * df["market_value_eur"].notna().mean(),
        )
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
