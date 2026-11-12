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
                group, stat = snake(upper), snake(lower)

                # FBref's identity columns arrive in BOTH shapes, depending on
                # whether soccerdata's standardize_colnames left the name in
                # level 0 or level 1:
                #     ('Age', '')                  -> level 1 empty
                #     ('Unnamed: 5_level_0', 'pos') -> level 0 a placeholder
                # Both are flat columns, so both must land as
                # `{stat_type}__flat__{name}`. Handling only the second shape
                # produced `standard__age__`, which no pattern in config.py
                # matches and which merge.py then fails to find.
                if not stat:
                        group, stat = "flat", group
                elif not group or group.startswith("unnamed"):
                        group = "flat"

                names.append(f"{stat_type}__{group}__{stat}")
        out.columns = names
        return out


def _reader() -> FBrefWide:
        paths.ensure_dirs()
        cls = FBrefWide if USE_EXTRA_READER else __import__("soccerdata").FBref
        return cls(leagues=config.LEAGUES, seasons=config.SEASONS)


#: How many times to rebuild the browser and retry one stat type.
BROWSER_ATTEMPTS = 3


def _read_with_fresh_browser(stat_type: str) -> pd.DataFrame:
        """Fetch one stat type, rebuilding the browser on each attempt.

        FBref sits behind Cloudflare, so soccerdata drives a real Chrome via
        SeleniumBase (FBref inherits BaseSeleniumReader - there is no
        requests-based path). That browser dies fairly often on Windows, and
        when it does you get:

            MaxRetryError: HTTPConnectionPool(host='localhost', port=...)
            ... /session/<id>/source ... WinError 10061 actively refused it

        soccerdata has its own retry ("attempt 1 of 5"), but it only retries
        the REQUEST - it never rebuilds `self._driver`. So all five attempts
        hit the same dead session, and because one reader served every stat
        type, a single crash used to take out the whole run.

        Each attempt here constructs a brand-new reader, and therefore a
        brand-new browser. Pages already downloaded are served from
        soccerdata's cache (~/soccerdata), so a retry costs seconds for the
        work already done and only re-fetches what is genuinely missing.
        """
        last: Exception | None = None
        for attempt in range(1, BROWSER_ATTEMPTS + 1):
                try:
                        return _reader().read_player_season_stats(stat_type=stat_type)
                except Exception as exc:  # noqa: BLE001 - browser death is expected here
                        last = exc
                        if attempt < BROWSER_ATTEMPTS:
                                logger.warning(
                                        "%s failed (attempt %s/%s): %s. Rebuilding the browser.",
                                        stat_type, attempt, BROWSER_ATTEMPTS, type(exc).__name__,
                                )
        raise last  # type: ignore[misc]


def fetch_player_stats() -> pd.DataFrame:
        """Download every configured stat type and join them on the player index."""
        frames: list[pd.DataFrame] = []

        for stat_type in config.FBREF_STAT_TYPES:
                logger.info("Fetching FBref player stats: %s", stat_type)
                try:
                        raw = _read_with_fresh_browser(stat_type)
                except Exception as exc:  # noqa: BLE001 - one bad page must not sink the run
                        logger.error(
                                "Could not fetch %r after %s attempts (%s). Continuing without "
                                "it; the feature resolution report will show what went missing.",
                                stat_type,
                                BROWSER_ATTEMPTS,
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


def parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
        """Make every `object` column writable by pyarrow.

        FBref's tables arrive as HTML text, so a column can hold a mix of
        numeric-looking strings and genuinely non-numeric ones. pyarrow infers
        a type from the leading values and then fails on the first that does
        not fit:

            ArrowInvalid: Could not convert '29-034' with type str:
            tried to convert to int64
            Conversion failed for column standard__age__

        '29-034' is FBref's years-days age format. Storing these as `string`
        keeps the raw text intact for merge.parse_fbref_age() and
        merge.to_numeric() to interpret later - which is the right division of
        labour, since this module's job is to fetch faithfully, not to clean.
        """
        objects = [c for c in df.columns if df[c].dtype == object]
        if objects:
                logger.info("Casting %s object columns to string for parquet", len(objects))
                df = df.astype({c: "string" for c in objects})
        return df


def run(force: bool = False) -> pd.DataFrame:
        paths.ensure_dirs()
        if paths.FBREF_WIDE.exists() and not force:
                logger.info("%s exists; use --force to refetch.", paths.FBREF_WIDE)
                return pd.read_parquet(paths.FBREF_WIDE)

        players = parquet_safe(fetch_player_stats())
        players.to_parquet(paths.FBREF_WIDE)
        write_column_report(players)

        teams = parquet_safe(fetch_team_stats())
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
