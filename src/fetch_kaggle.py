"""Stage 1d - fill the archive's gaps from a Kaggle copy of FBref.

The Wayback Machine holds FBref's miscellaneous page for only two leagues in
2024-25, so aerial duels and ball recoveries were missing for La Liga, Ligue 1
and Serie A. hubertsidorowicz's "Football Players Stats (2024-2025)" on Kaggle
is a full FBref scrape of the Big 5 taken before the Opta stats were removed,
and carries those columns for every league:

    https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025

Download it by hand (Kaggle needs an account) and unzip it into
data/raw/kaggle/2425/. The raw file stays out of git; this stage's output,
data/interim/fbref_kaggle_player_seasons.parquet, is committed, so a fresh
clone does not need the download.

**It only fills gaps.** A cell the live table or the archive already has is
never overwritten (see fetch_fbref_archive.load_player_table).

**It is checked before it is trusted.** Rows are matched to the live table
the same way the archive is (fetch_fbref_archive.align_to_live), then compared
with the archive column by column wherever both have a value: the filled
columns against the archived Premier League and Bundesliga misc pages, and
interceptions, tackles won and clearances against the archived defense page
in every league. The agreement rates go to reports/fbref_kaggle_check.csv; if
any falls below MIN_AGREEMENT the stage refuses to use the copy.

Its 2025-26 counterpart was updated after the removal and has only the basic
stats, so it adds nothing and is not used.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from . import fetch_fbref_archive, paths

logger = logging.getLogger(__name__)

#: Season key -> the full Kaggle CSV for it, under paths.KAGGLE_DIR.
KAGGLE_FILES = {
        "2425": "2425/players_data-2024_2025.csv",
}

#: Kaggle's "Comp" values -> this project's league keys.
LEAGUES = {
        "eng Premier League": "ENG-Premier League",
        "es La Liga": "ESP-La Liga",
        "it Serie A": "ITA-Serie A",
        "de Bundesliga": "GER-Bundesliga",
        "fr Ligue 1": "FRA-Ligue 1",
}

#: Kaggle columns this stage fills, renamed to the archive's column names. The
#: CSV joins every FBref table side by side and suffixes repeated headers with
#: the table's name, so the aerial "Lost" is "Lost_stats_misc" - plain "Lost"
#: is the defense table's challenges lost.
FILL_COLUMNS = {
        "Won": "misc__aerial_duels__won",
        "Lost_stats_misc": "misc__aerial_duels__lost",
        "Won%": "misc__aerial_duels__won_pct",
        "Recov": "misc__performance__recov",
}

#: Kaggle columns read only to check the copy against the archived defense
#: page, which has Opta-era values for all five leagues. (Not against the live
#: table: its basic stats come from FBref's post-Opta provider and differ.)
CHECK_COLUMNS = {
        "Int": "defense__flat__int",
        "TklW": "defense__tackles__tklw",
        "Clr": "defense__flat__clr",
}

#: Identity columns align_to_live uses to re-find renamed players.
IDENTITY_COLUMNS = {
        "Born": "misc__flat__born",
        "90s_stats_misc": "misc__flat__90s",
}

#: Below this share of agreeing values, a column is treated as a different source.
MIN_AGREEMENT = 0.95


def read_season(season: str) -> pd.DataFrame | None:
        """One Kaggle CSV as a frame indexed like the live table, or None if absent."""
        path = paths.KAGGLE_DIR / KAGGLE_FILES[season]
        if not path.exists():
                return None
        raw = pd.read_csv(path, low_memory=False)
        columns = {**IDENTITY_COLUMNS, **FILL_COLUMNS, **CHECK_COLUMNS}
        missing = sorted(set(columns) - set(raw.columns))
        if missing:
                raise SystemExit(f"{path.name} lacks expected columns {missing}; is it the full (not light) file?")

        frame = raw[list(columns)].rename(columns=columns)
        frame = frame.apply(pd.to_numeric, errors="coerce")
        frame["league"] = raw["Comp"].map(LEAGUES)
        frame["season"] = season
        frame["team"] = raw["Squad"].replace(fetch_fbref_archive.TEAMNAME_REPLACEMENTS)
        frame["player"] = raw["Player"]
        frame = frame.dropna(subset=["league", "player"])
        frame = frame.set_index(["league", "season", "team", "player"]).sort_index()
        return frame.loc[~frame.index.duplicated()]


def agreement(kaggle: pd.DataFrame, reference: pd.DataFrame, pairs: dict[str, str]) -> pd.DataFrame:
        """Share of matching values per league and column, where both sides have one."""
        rows = []
        ref = reference.reindex(kaggle.index)
        for column in pairs.values():
                if column not in ref.columns:
                        continue
                ours = pd.to_numeric(ref[column], errors="coerce")
                theirs = kaggle[column]
                both = ours.notna() & theirs.notna()
                same = pd.Series(np.isclose(ours, theirs, atol=0.05), index=kaggle.index) & both
                by_league = pd.DataFrame({"both": both, "same": same}).groupby(level="league").sum()
                for league, counts in by_league.iterrows():
                        if counts["both"]:
                                rows.append({
                                        "league": league,
                                        "column": column,
                                        "compared": int(counts["both"]),
                                        "agree_share": round(counts["same"] / counts["both"], 4),
                                })
        return pd.DataFrame(rows, columns=["league", "column", "compared", "agree_share"])


def build() -> pd.DataFrame | None:
        """Every available Kaggle season, aligned to the live table and checked."""
        live = fetch_fbref_archive.live_table()
        archived = pd.read_parquet(paths.FBREF_ARCHIVE) if paths.FBREF_ARCHIVE.exists() else None

        frames, checks = [], []
        for season in KAGGLE_FILES:
                frame = read_season(season)
                if frame is None:
                        logger.warning("No Kaggle file for %s under %s; skipping.", season, paths.KAGGLE_DIR)
                        continue
                aligned = fetch_fbref_archive.align_to_live(frame, live, "misc")
                if archived is None:
                        raise SystemExit("Run fetch-fbref-archive first: the Kaggle copy is checked against it.")
                check = agreement(aligned, archived, {**CHECK_COLUMNS, **FILL_COLUMNS})
                check.insert(1, "season", season)
                checks.append(check)
                frames.append(aligned)

        if not frames:
                return None

        report = pd.concat(checks, ignore_index=True)
        report.to_csv(paths.FBREF_KAGGLE_CHECK, index=False)
        logger.info("Kaggle vs trusted values:\n%s", report.to_string(index=False))
        weak = report[report["agree_share"] < MIN_AGREEMENT]
        if not weak.empty:
                raise SystemExit(
                        "The Kaggle copy disagrees with data we already trust:\n"
                        f"{weak.to_string(index=False)}\nNot using it; see {paths.FBREF_KAGGLE_CHECK}."
                )

        out = pd.concat(frames)
        return out[list(FILL_COLUMNS.values())]


def run(force: bool = False) -> pd.DataFrame | None:
        """Stage entry point. Optional: without the raw download it only warns."""
        paths.ensure_dirs()
        if paths.FBREF_KAGGLE.exists() and not force:
                logger.info("%s exists; use --force to rebuild.", paths.FBREF_KAGGLE)
                return pd.read_parquet(paths.FBREF_KAGGLE)
        table = build()
        if table is None:
                logger.warning("No Kaggle files found; the archive's gaps stay empty.")
                return None
        table.to_parquet(paths.FBREF_KAGGLE)
        logger.info("Wrote %s (%s player-seasons)", paths.FBREF_KAGGLE, len(table))
        return table


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
