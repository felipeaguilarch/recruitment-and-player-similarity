"""Stage 1a - download FBref player-season and team-season stats.

Run this once. It is the slowest stage in the project by a wide margin: FBref
tolerates roughly one request every three seconds, and soccerdata throttles to
respect that. Five leagues x three seasons x eight stat types is 120 page
fetches, so budget well over an hour on a cold cache.

soccerdata caches the raw HTML under ~/soccerdata (or $SOCCERDATA_DIR), so a
re-run after a crash resumes almost instantly. The output of this stage is
written to data/interim/ and nothing downstream ever calls FBref again.
"""

from __future__ import annotations

import argparse
import logging
import re

import pandas as pd

from . import config, paths
from .fbref_extra import FBrefWide

logger = logging.getLogger(__name__)

# Set False if a future soccerdata release supports every page in
# config.FBREF_STAT_TYPES natively.
USE_EXTRA_READER = True

# FBref column headers that carry player identity rather than a statistic.
IDENTITY_STATS = ("age", "born", "pos", "nation")


def snake(value: object) -> str:
        """Normalise an FBref column header into a stable snake_case token.

        `%` becomes `_pct` before punctuation is stripped, so `SoT%` and `SoT`
        stay distinguishable:

            >>> snake("SoT%"), snake("npxG/Sh"), snake("Def 3rd")
            ('sot_pct', 'npxg_sh', 'def_3rd')
        """
        text = str(value).strip().replace("%", "_pct")
        text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
        return re.sub(r"_+", "_", text).strip("_").lower()


def flatten_columns(df: pd.DataFrame, stat_type: str) -> pd.DataFrame:
        """Flatten FBref's two-level header into `{stat_type}__{group}__{stat}`.

        Fully qualifying the name matters: FBref reuses short stat names across
        pages and across header groups within a page. `Int` appears under both
        `defense__blocks` and `misc__performance`; `xG` appears under both
        `standard__expected` and `standard__per_90_minutes`.

        Unnamed header groups - the flat columns on the left of each table - are
        given the literal group `flat`, so every name has exactly three parts and
        the patterns in config.FEATURES can rely on that shape.
        """
        out = df.copy()
        if not isinstance(out.columns, pd.MultiIndex):
                out.columns = [f"{stat_type}__flat__{snake(c)}" for c in out.columns]
                return out

        names: list[str] = []
        for upper, lower in out.columns:
                group = snake(upper)
                if not group or group.startswith("unnamed"):
                        group = "flat"
                names.append(f"{stat_type}__{group}__{snake(lower)}")
        out.columns = names
        return out


def _reader() -> FBrefWide:
        paths.ensure_dirs()
        cls = FBrefWide if USE_EXTRA_READER else __import__("soccerdata").FBref
        return cls(leagues=config.LEAGUES, seasons=config.SEASONS)


def fetch_player_stats() -> pd.DataFrame:
        """Download every configured stat type and join them on the player index."""
        fbref = _reader()
        frames: list[pd.DataFrame] = []

        for stat_type in config.FBREF_STAT_TYPES:
                logger.info("Fetching FBref player stats: %s", stat_type)
                try:
                        raw = fbref.read_player_season_stats(stat_type=stat_type)
                except Exception as exc:  # noqa: BLE001 - one bad page must not sink the run
                        logger.error(
                                "Could not fetch %r (%s). Continuing without it; the feature "
                                "resolution report will show what went missing.",
                                stat_type,
                                exc,
                        )
                        continue

                flat = flatten_columns(raw, stat_type)

                # Identity columns are identical across stat types. Keep them from
                # `standard` only, so the join does not produce _x/_y suffixes.
                if stat_type != "standard":
                        drop = [c for c in flat.columns if c.rsplit("__", 1)[-1] in IDENTITY_STATS]
                        flat = flat.drop(columns=drop)

                frames.append(flat)
                logger.info("  -> %s rows, %s columns", len(flat), flat.shape[1])

        if not frames:
                raise RuntimeError("No FBref stat types could be fetched.")

        merged = pd.concat(frames, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        logger.info("Merged FBref player table: %s rows, %s columns", *merged.shape)
        return merged


def fetch_team_stats() -> pd.DataFrame:
        """Team-level standard stats, needed only for the possession adjustment.

        Team possession lives on FBref's squad standard table, so this needs no
        widened reader.
        """
        fbref = _reader()
        logger.info("Fetching FBref team stats: standard (for possession)")
        raw = fbref.read_team_season_stats(stat_type="standard")
        return flatten_columns(raw, "standard")


def write_column_report(df: pd.DataFrame) -> None:
        """Dump every flattened column name, grouped by stat type.

        Read this before trusting the feature catalogue. FBref changes table
        headers from time to time; when a feature stops resolving, the fix is a
        pattern edit in src/config.py, and this file tells you what to edit it to.
        """
        paths.FBREF_COLUMNS.parent.mkdir(parents=True, exist_ok=True)
        by_stat: dict[str, list[str]] = {}
        for col in sorted(df.columns):
                by_stat.setdefault(col.split("__", 1)[0], []).append(col)

        with paths.FBREF_COLUMNS.open("w", encoding="utf-8") as fh:
                fh.write("FBref columns actually returned, after flattening.\n")
                fh.write("Edit the patterns in src/config.py FEATURES to match these.\n\n")
                for stat_type, cols in by_stat.items():
                        fh.write(f"### {stat_type}  ({len(cols)} columns)\n")
                        for col in cols:
                                fh.write(f"  {col}\n")
                        fh.write("\n")
        logger.info("Wrote column inventory to %s", paths.FBREF_COLUMNS)


def run(force: bool = False) -> pd.DataFrame:
        paths.ensure_dirs()
        if paths.FBREF_WIDE.exists() and not force:
                logger.info("%s exists; use --force to refetch.", paths.FBREF_WIDE)
                return pd.read_parquet(paths.FBREF_WIDE)

        players = fetch_player_stats()
        players.to_parquet(paths.FBREF_WIDE)
        write_column_report(players)

        teams = fetch_team_stats()
        teams.to_parquet(paths.FBREF_TEAM)

        logger.info("Wrote %s and %s", paths.FBREF_WIDE, paths.FBREF_TEAM)
        return players


def inspect_columns() -> None:
        """Re-generate the column report from whatever is already cached."""
        if not paths.FBREF_WIDE.exists():
                raise SystemExit(
                        "No cached FBref table. Run `python run_pipeline.py --stage fetch-fbref` first."
                )
        write_column_report(pd.read_parquet(paths.FBREF_WIDE))


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true", help="refetch even if output exists")
        parser.add_argument("--inspect", action="store_true", help="rewrite the column report only")
        args = parser.parse_args()
        inspect_columns() if args.inspect else run(force=args.force)
