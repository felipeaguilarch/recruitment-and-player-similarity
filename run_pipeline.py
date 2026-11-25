#!/usr/bin/env python
"""Reproduce every data artefact, model and figure in this repo.

    python run_pipeline.py --stage all           # cold start, slow (see below)
    python run_pipeline.py --stage features --force
    python run_pipeline.py --list

Stages are ordered and each one skips itself if its output already exists, so
a re-run after a crash resumes rather than restarting. Pass --force to rebuild
a stage you have deliberately changed.

Cold-start cost, roughly:
    fetch-fbref          60-120 min   rate-limited to ~1 request / 3 s
    fetch-fbref-archive  5-10 min     archived pages, ~2 s apart
    fetch-kaggle         < 1 min      2024-25 gaps from a hand-downloaded Kaggle CSV
    fetch-transfermarkt  3-10 min     one ~241 MB download
    fetch-tm-contracts   1-2 min      seven ~1-5 MB snapshots
    merge                5-7 min      the fuzzy name join and club check dominate
    features             < 1 min
    roles                < 1 min
    similarity           < 1 min      UMAP is the slow part
    value                < 1 min      cross-validation x 2 models
    figures              < 1 min

Only `fetch-fbref` is genuinely expensive, and soccerdata caches the raw HTML
under ~/soccerdata, so the second cold start is far quicker than the first.
The interim tables are committed, though, so a fresh clone skips every fetch
stage and rebuilds from `merge` onward offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

STAGES: list[tuple[str, str]] = [
        ("fetch-fbref", "Download FBref player and team season stats"),
        ("fetch-fbref-archive", "Recover FBref advanced stats from the Wayback Machine"),
        ("fetch-kaggle", "Fill the archive's 2024-25 gaps from a Kaggle copy of FBref"),
        ("fetch-transfermarkt", "Download transfermarkt-datasets market values"),
        ("fetch-tm-contracts", "Dated contract observations from the raw TM scrapes"),
        ("merge", "Join the two sources into one player-season table"),
        ("features", "Per-90, possession-adjust, percentile-rank"),
        ("roles", "Score every player-season against its group's playing roles"),
        ("similarity", "Fit PCA + cosine kNN per position group"),
        ("value", "Fit the expected-value model, flag undervalued players"),
        ("figures", "Rebuild the README figures in assets/"),
]
STAGE_NAMES = [name for name, _ in STAGES]


def run_stage(name: str, force: bool) -> None:
        if name == "fetch-fbref":
                from src import fetch_fbref
                fetch_fbref.run(force=force)
        elif name == "fetch-fbref-archive":
                from src import fetch_fbref_archive
                fetch_fbref_archive.run(force=force)
        elif name == "fetch-kaggle":
                from src import fetch_kaggle
                fetch_kaggle.run(force=force)
        elif name == "fetch-transfermarkt":
                from src import fetch_transfermarkt
                fetch_transfermarkt.run(force=force)
        elif name == "fetch-tm-contracts":
                from src import fetch_tm_contracts
                fetch_tm_contracts.run(force=force)
        elif name == "merge":
                from src import merge
                merge.run(force=force)
        elif name == "features":
                from src import features
                features.run(force=force)
        elif name == "roles":
                from src import roles
                roles.run(force=force)
        elif name == "similarity":
                from src import similarity
                similarity.run(force=force)
        elif name == "value":
                from src import value_model
                value_model.run(force=force)
        elif name == "figures":
                from src import figures
                figures.run(force=force)
        elif name == "inspect-columns":
                from src import fetch_fbref
                fetch_fbref.inspect_columns()
        else:
                raise SystemExit(f"Unknown stage: {name}")


def main() -> int:
        parser = argparse.ArgumentParser(
                description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument(
                "--stage", default="all",
                choices=["all", *STAGE_NAMES, "inspect-columns"],
                help="stage to run, or 'all' for the full pipeline in order",
        )
        parser.add_argument("--force", action="store_true", help="rebuild even if output exists")
        parser.add_argument("--from-stage", choices=STAGE_NAMES, help="run 'all' starting here")
        parser.add_argument("--list", action="store_true", help="list stages and exit")
        parser.add_argument("--quiet", action="store_true")
        args = parser.parse_args()

        logging.basicConfig(
                level=logging.WARNING if args.quiet else logging.INFO,
                format="%(asctime)s  %(levelname)-7s %(message)s",
                datefmt="%H:%M:%S",
        )

        if args.list:
                width = max(len(n) for n in STAGE_NAMES)
                for name, description in STAGES:
                        print(f"  {name:<{width}}  {description}")
                return 0

        if args.stage == "all":
                start = STAGE_NAMES.index(args.from_stage) if args.from_stage else 0
                to_run = STAGE_NAMES[start:]
        else:
                to_run = [args.stage]

        for name in to_run:
                logging.info("=" * 64)
                logging.info("STAGE: %s", name)
                logging.info("=" * 64)
                began = time.perf_counter()
                try:
                        run_stage(name, force=args.force)
                except SystemExit as exc:
                        logging.error("%s", exc)
                        return 1
                logging.info("Stage %s finished in %.1fs", name, time.perf_counter() - began)

        logging.info("Done.")
        return 0


if __name__ == "__main__":
        sys.exit(main())
