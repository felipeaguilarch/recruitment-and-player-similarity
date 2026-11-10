"""Stage 1b - download market values from transfermarkt-datasets.

This project does NOT scrape Transfermarkt. It uses dcaribou's
transfermarkt-datasets, an already-scraped, versioned, maintained release
published as CSVs. That is both kinder to the site and far more reproducible
than a bespoke scraper: anyone cloning this repo gets the same bytes.

Two fields make the detour worth it beyond the market value itself:

  * `sub_position` - "Centre-Back", "Left-Back", "Defensive Midfield" and so
    on. FBref's own position column only gives DF/MF/FW, which cannot separate
    a centre-back from a full-back.
  * `contract_expiration_date` - contract years remaining is one of the
    strongest real drivers of transfer price, and most portfolio projects
    omit it entirely.

KNOWN LIMITATION, checked 2026-11-06: upstream updates are paused. The data is
current to 2026-07-06, `player_valuations` stops at 2026-06-12, and 2026/27
squads are not covered. 2025-26 is therefore the newest season with usable
values - which is why config.SEASONS ends at "2526".
https://github.com/dcaribou/transfermarkt-datasets/discussions/383
"""

from __future__ import annotations

import argparse
import logging
import zipfile

import pandas as pd
import requests

from . import config, paths

logger = logging.getLogger(__name__)

CHUNK = 1 << 20  # 1 MiB


def download_zip(force: bool = False) -> None:
        """Fetch the dataset archive (~241 MB) unless it is already on disk."""
        paths.ensure_dirs()
        if paths.TM_ZIP.exists() and not force:
                size_mb = paths.TM_ZIP.stat().st_size / 1e6
                logger.info("%s already present (%.0f MB); skipping download.", paths.TM_ZIP.name, size_mb)
                return

        logger.info("Downloading %s", config.TM_ZIP_URL)
        headers = {"User-Agent": config.USER_AGENT}
        with requests.get(config.TM_ZIP_URL, stream=True, timeout=120, headers=headers) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("Content-Length", 0))
                done = 0
                with paths.TM_ZIP.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=CHUNK):
                                fh.write(chunk)
                                done += len(chunk)
                                if total and done % (20 * CHUNK) < CHUNK:
                                        logger.info("  %.0f%% (%.0f/%.0f MB)", 100 * done / total, done / 1e6, total / 1e6)
        logger.info("Saved %s (%.0f MB)", paths.TM_ZIP, paths.TM_ZIP.stat().st_size / 1e6)


def extract_tables() -> dict[str, pd.DataFrame]:
        """Unpack only the tables listed in config.TM_TABLES.

        The archive holds twelve tables including `appearances` (1.89M rows) and
        `game_lineups` (3.17M rows). Extracting only what the pipeline reads keeps
        data/raw from ballooning.
        """
        wanted = {name: None for name in config.TM_TABLES}
        frames: dict[str, pd.DataFrame] = {}

        with zipfile.ZipFile(paths.TM_ZIP) as zf:
                members = {member.filename.rsplit("/", 1)[-1]: member for member in zf.infolist()}
                for name in wanted:
                        filename = f"{name}.csv"
                        if filename not in members:
                                available = sorted(n for n in members if n.endswith(".csv"))
                                raise FileNotFoundError(
                                        f"{filename} not in archive. Available: {available}"
                                )
                        logger.info("Extracting %s", filename)
                        with zf.open(members[filename]) as fh:
                                frames[name] = pd.read_csv(fh, low_memory=False)
                        logger.info("  -> %s rows", len(frames[name]))

        return frames


def build_players(players: pd.DataFrame) -> pd.DataFrame:
        """Reduce `players.csv` to the identity and contract fields we use."""
        keep = [
                "player_id",
                "name",
                "last_name",
                "first_name",
                "date_of_birth",
                "sub_position",
                "position",
                "foot",
                "height_in_cm",
                "country_of_citizenship",
                "market_value_in_eur",
                "highest_market_value_in_eur",
                "contract_expiration_date",
                "current_club_name",
                "current_club_domestic_competition_id",
                "last_season",
        ]
        present = [c for c in keep if c in players.columns]
        missing = sorted(set(keep) - set(present))
        if missing:
                logger.warning("players.csv is missing expected columns: %s", missing)

        out = players[present].copy()
        out["date_of_birth"] = pd.to_datetime(out["date_of_birth"], errors="coerce")
        out["birth_year"] = out["date_of_birth"].dt.year
        if "contract_expiration_date" in out.columns:
                out["contract_expiration_date"] = pd.to_datetime(
                        out["contract_expiration_date"], errors="coerce"
                )
        return out


def build_valuations(valuations: pd.DataFrame) -> pd.DataFrame:
        """Reduce `player_valuations.csv` to a dated value history."""
        keep = [c for c in ("player_id", "date", "market_value_in_eur") if c in valuations.columns]
        out = valuations[keep].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "market_value_in_eur"])
        return out.sort_values(["player_id", "date"])


def run(force: bool = False) -> None:
        paths.ensure_dirs()
        if paths.TM_PLAYERS.exists() and paths.TM_VALUATIONS.exists() and not force:
                logger.info("Transfermarkt interim files exist; use --force to rebuild.")
                return

        download_zip(force=force)
        tables = extract_tables()

        players = build_players(tables["players"])
        players.to_parquet(paths.TM_PLAYERS, index=False)
        logger.info("Wrote %s (%s rows)", paths.TM_PLAYERS, len(players))

        valuations = build_valuations(tables["player_valuations"])
        valuations.to_parquet(paths.TM_VALUATIONS, index=False)
        logger.info("Wrote %s (%s rows)", paths.TM_VALUATIONS, len(valuations))

        latest = valuations["date"].max()
        logger.info(
                "Latest valuation in the dataset: %s. Seasons after this are unusable.",
                latest.date() if pd.notna(latest) else "unknown",
        )


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
