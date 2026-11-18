"""Stage 1c - contract end dates as they stood during each season.

transfermarkt-datasets' players table only carries each player's CURRENT
contract (as of July 2026). Measured from a 2023-24 midpoint, that makes a
player who has since signed an extension, or moved club, look as if he had
years more left than he did: on this project's data the current-contract
method put the average 2023-24 player at 4.4 years remaining, against 2.8
from dated observations, and moved 45% of rows by more than three months.

**Where the dated contracts come from.** The same project keeps its raw
scrapes - one JSON record per player profile, per season's squads - in a
public, content-addressed DVC store. Each file was scraped on a particular
day and never touched again, so it is a dated observation of every player's
contract (plus the date he joined his club). data/tm_contract_snapshots.csv
pins seven of them by md5, observed between May 2023 and July 2026. The
observation dates are estimated from the records themselves: a player's
listed age and date of birth bound the scrape day, and across ~8,000 players
those bounds narrow to a few days.

**Choosing an observation for a season.** For a player-season with midpoint
M (1 January of the season's end year), an observation taken on day T is
usable only if

  * the contract it shows was still running at M, and
  * it belongs to the club spell the player was in at M: he had joined
    before M, and no transfer (transfers table, loans included) falls
    between T and M.

Of the usable ones the nearest to M wins; on a tie the earlier one, so the
feature never leans on information from after the season when it can avoid
it. Player-seasons with no usable observation get NaN, not a guess - the
obvious guess for players who moved on a free afterwards (contract ran out on
30 June) was checked against known contracts and was right only 53% of the
time, because free moves in that table include loans.

**Expected error.** Within one club spell a contract changes ~15% of the time
per year (measured between pairs of snapshots), so with a median gap of 4-6
months between T and M, roughly 5-7% of matched rows carry a slightly stale
or slightly early end date.

**Fragility.** The store garbage-collects unreferenced objects: snapshots
older than June 2025 are already gone from it. Downloads are cached under
data/raw/transfermarkt/contracts/ so a clone that ran once keeps working.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import zipfile

import numpy as np
import pandas as pd
import requests

from . import config, name_matching, paths

logger = logging.getLogger(__name__)

#: Public HTTP mirror of transfermarkt-datasets' DVC store (from its .dvc/config).
DVC_STORE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/dvc/files/md5/"

#: The day transfermarkt-datasets' players.csv was current to.
CURRENT_OBSERVED_ON = "2026-07-06"


# --- download ----------------------------------------------------------------

def raw_path(name: str):
        return paths.TM_CONTRACTS_DIR / f"{name}.json.gz"


def download(row: dict, session: requests.Session) -> None:
        """Fetch one pinned snapshot by md5, unless it is already cached."""
        target = raw_path(row["name"])
        if target.exists():
                return
        md5 = row["md5"]
        url = f"{DVC_STORE}{md5[:2]}/{md5[2:]}"
        response = session.get(url, timeout=120)
        if response.status_code == 404:
                raise SystemExit(
                        f"{row['name']} ({md5}) is no longer in the transfermarkt-datasets store. "
                        "Remove it from data/tm_contract_snapshots.csv or restore it from a cached copy."
                )
        response.raise_for_status()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(response.content)
        logger.info("Downloaded %s (%.1f MB)", target.name, len(response.content) / 1e6)


# --- parsing -----------------------------------------------------------------

def parse_tm_date(values: pd.Series) -> pd.Series:
        """Transfermarkt dates in either of the two formats the scrapes use.

        Older scrapes write "Jun 30, 2026", newer ones "30/06/2026"; dates of
        birth may carry a trailing "(23)" age. Anything else ("-", blank)
        becomes NaT.
        """
        text = values.fillna("").astype(str).str.replace(r"\s*\(\d+\)", "", regex=True).str.strip()
        english = pd.to_datetime(text, format="%b %d, %Y", errors="coerce")
        numeric = pd.to_datetime(text, format="%d/%m/%Y", errors="coerce")
        return english.fillna(numeric)


def parse_snapshot(name: str, observed_on: str) -> pd.DataFrame:
        """One scraped squads file to (player_id, joined, contract_end, observed_on, source)."""
        ids, joined, contract = [], [], []
        with gzip.open(raw_path(name), "rt", encoding="utf-8") as fh:
                for line in fh:
                        record = json.loads(line)
                        ids.append(int(record["href"].rstrip("/").rsplit("/", 1)[-1]))
                        joined.append((record.get("joined") or "").strip())
                        contract.append(record.get("contract_expires") or "")
        return pd.DataFrame({
                "player_id": ids,
                "joined": parse_tm_date(pd.Series(joined)),
                "contract_end": parse_tm_date(pd.Series(contract)),
                "observed_on": pd.Timestamp(observed_on),
                "source": name,
        })


def current_contracts() -> pd.DataFrame:
        """players.csv's contract dates, as one more (undated-join) observation."""
        players = pd.read_parquet(paths.TM_PLAYERS, columns=["player_id", "contract_expiration_date"])
        return pd.DataFrame({
                "player_id": players["player_id"].astype(int),
                "joined": pd.NaT,
                "contract_end": pd.to_datetime(players["contract_expiration_date"], errors="coerce"),
                "observed_on": pd.Timestamp(CURRENT_OBSERVED_ON),
                "source": "players.csv",
        })


def load_transfers() -> pd.DataFrame:
        """Dated moves from the transfermarkt-datasets zip, future-dated rows dropped.

        The table lists scheduled loan returns years ahead (2030-06-30); only
        moves that had happened by the data cut-off are evidence of anything.
        """
        with zipfile.ZipFile(paths.TM_ZIP) as zf:
                name = next(n for n in zf.namelist() if n.rsplit("/", 1)[-1].startswith("transfers.csv"))
                with zf.open(name) as fh:
                        transfers = pd.read_csv(
                                fh, usecols=["player_id", "transfer_date"],
                                compression="gzip" if name.endswith(".gz") else None,
                        )
        transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"], errors="coerce")
        cutoff = pd.Timestamp(CURRENT_OBSERVED_ON)
        return transfers[transfers["transfer_date"] <= cutoff].dropna()


# --- choosing an observation -------------------------------------------------

def contract_at_midpoint(
        rows: pd.DataFrame, observations: pd.DataFrame, transfers: pd.DataFrame
) -> pd.DataFrame:
        """Contract end date in force at each row's season midpoint.

        `rows` needs `tm_player_id` and `season`. Returns one row per input row,
        in the same order, with contract_end, contract_observed_on and
        contract_source (all NaN/NaT where nothing qualified). See the module
        docstring for the rules.
        """
        keys = rows[["tm_player_id", "season"]].reset_index(drop=True)
        keys["row"] = np.arange(len(keys))
        keys["midpoint"] = keys["season"].map(name_matching.season_midpoint)
        keys = keys.dropna(subset=["tm_player_id"])
        keys["tm_player_id"] = keys["tm_player_id"].astype(int)

        cand = keys.merge(observations, left_on="tm_player_id", right_on="player_id")
        running = cand["contract_end"] >= cand["midpoint"]
        same_spell = cand["joined"].isna() | (cand["joined"] <= cand["midpoint"])
        cand = cand[running & same_spell].reset_index(drop=True)

        # Drop observations separated from the midpoint by a move.
        cand["lo"] = cand[["observed_on", "midpoint"]].min(axis=1)
        cand["hi"] = cand[["observed_on", "midpoint"]].max(axis=1)
        crossed = cand[["tm_player_id", "lo", "hi"]].reset_index().merge(
                transfers, left_on="tm_player_id", right_on="player_id"
        )
        crossed = crossed.loc[
                (crossed["transfer_date"] > crossed["lo"]) & (crossed["transfer_date"] <= crossed["hi"]),
                "index",
        ]
        cand = cand.drop(index=crossed.unique())

        cand["gap"] = (cand["observed_on"] - cand["midpoint"]).abs()
        cand["after"] = cand["observed_on"] > cand["midpoint"]
        best = cand.sort_values(["row", "gap", "after"]).drop_duplicates("row").set_index("row")

        out = pd.DataFrame(index=pd.RangeIndex(len(rows)))
        out["contract_end"] = best["contract_end"]
        out["contract_observed_on"] = best["observed_on"]
        out["contract_source"] = best["source"]
        out.index = rows.index
        return out


# --- the stage ---------------------------------------------------------------

def build_observations() -> pd.DataFrame:
        """Every dated contract observation: the pinned snapshots plus players.csv."""
        if not paths.TM_CONTRACT_SNAPSHOTS.exists():
                raise SystemExit(f"{paths.TM_CONTRACT_SNAPSHOTS} is missing.")
        if not paths.TM_PLAYERS.exists():
                raise SystemExit("Run `python run_pipeline.py --stage fetch-transfermarkt` first.")
        snapshots = pd.read_csv(paths.TM_CONTRACT_SNAPSHOTS, dtype=str)

        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
        frames = []
        for row in snapshots.to_dict("records"):
                download(row, session)
                frame = parse_snapshot(row["name"], row["observed_on"])
                logger.info("  %s: %s players, observed %s", row["name"], len(frame), row["observed_on"])
                frames.append(frame)
        frames.append(current_contracts())
        observations = pd.concat(frames, ignore_index=True).dropna(subset=["contract_end"])
        return observations


def run(force: bool = False) -> pd.DataFrame:
        """Stage entry point: write paths.TM_CONTRACTS (all dated observations)."""
        paths.ensure_dirs()
        if paths.TM_CONTRACTS.exists() and not force:
                logger.info("%s exists; use --force to rebuild.", paths.TM_CONTRACTS)
                return pd.read_parquet(paths.TM_CONTRACTS)
        observations = build_observations()
        observations.to_parquet(paths.TM_CONTRACTS, index=False)
        logger.info(
                "Wrote %s (%s observations of %s players)",
                paths.TM_CONTRACTS, len(observations), observations["player_id"].nunique(),
        )
        return observations


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
