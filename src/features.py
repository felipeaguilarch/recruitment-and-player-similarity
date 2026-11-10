"""Stage 3 - raw FBref columns into model-ready features.  [YOU IMPLEMENT THIS]

Three things happen here, and the middle one is what separates this from a
naive build:

1. Per-90 normalisation, so a 3,000-minute player and a 900-minute player are
   on the same scale.
2. Possession adjustment of defensive actions. A midfielder at a team with 65%
   of the ball makes fewer tackles because the ball is not there to be won.
   Without this, a "best defensive midfielder" ranking is really a ranking of
   players at bad teams.
3. Percentile ranks within position group and season, for the radar. A
   non-technical viewer reads "88th percentile for progressive carries"
   instantly; nobody reads "+1.4 sigma".

Z-scoring is deliberately NOT done here - it belongs inside the sklearn
Pipeline in similarity.py, so the fitted scaler travels with the model and the
app cannot drift out of sync with it.

Feature resolution is pattern-based and reported rather than assumed, because
FBref table headers are the part of this project most likely to drift.

Spec: tests/test_smoke.py covers possession_factor and resolve_features;
tests/test_pipeline_synthetic.py runs the whole stage end to end.
"""

from __future__ import annotations

import argparse
import logging
import re

import numpy as np
import pandas as pd

from . import config, paths

logger = logging.getLogger(__name__)

# Columns carried through to the output alongside the computed features.
ID_COLUMNS = [
        "player_season_id", "player", "team", "league", "season", "position_group",
        "position_source", "age", "minutes", "nineties", "team_possession",
        "market_value_eur", "contract_years_remaining", "tm_player_id", "pos",
        "nation", "foot", "height_in_cm", "country_of_citizenship",
]

# Rows below this many minutes are excluded entirely: percentiles, PCA and the
# value model are all fitted on the surviving pool. Deliberately looser than
# config.MIN_MINUTES so the app's minutes slider has room to move in both
# directions; MIN_MINUTES is the slider's *default*, not the hard floor.
INCLUSION_MIN_MINUTES = config.MIN_MINUTES_SENSITIVITY


def resolve_features(columns: list[str]) -> tuple[dict[str, str], pd.DataFrame]:
        """Map each catalogue feature to a real FBref column name.

        Returns `(resolved, report)` where `resolved` is {feature_name: column}
        and `report` is a frame with one row per catalogue feature. A feature
        that resolves to nothing must be dropped from every position group that
        asked for it, and the report must say so - that is your signal to go fix
        its patterns in config.py against reports/fbref_columns.txt.

        Each config.Feature has `.patterns`, a tuple of regexes tried IN ORDER
        against the flattened column names. First pattern with a hit wins; this
        is how `interceptions` prefers defense__blocks__int over the duplicate
        misc__performance__int.

        TODO 1. Loop config.FEATURES.items(). For each, loop feature.patterns and
                collect [c for c in columns if re.match(pattern, c)]. Break on
                the first pattern that matches anything.
        TODO 2. If several columns match one pattern, pick deterministically
                (sorted(...)[0]) so runs are reproducible.
        TODO 3. Build the report frame with at least: feature, status
                ("ok"/"MISSING"), column, matched_pattern, patterns_tried,
                per90, possession_adjusted.
        TODO 4. Write it to paths.FEATURE_RESOLUTION (mkdir the parent first) and
                logger.warning the list of misses.
        TODO 5. Return (resolved, report) - tests unpack both.
        """
        raise NotImplementedError("features.resolve_features")


def effective_group_specs(resolved: dict[str, str]) -> dict[str, list[str]]:
        """Each group's feature list, minus anything that failed to resolve.

        Use this on the RAW FBref frame, where `resolved` came from
        resolve_features.

        TODO 1. For each config.GROUP_SPECS item, keep only features present in
                `resolved`.
        TODO 2. logger.warning what each group lost, and logger.error if a group
                drops below ~6 usable features - at that point its profile is
                not worth trusting.
        """
        raise NotImplementedError("features.effective_group_specs")


def available_group_specs(columns: list[str]) -> dict[str, list[str]]:
        """Each group's feature list, restricted to columns present in a frame.

        Use this on features.parquet, whose columns are already the catalogue's
        canonical names (`npxg`, `prog_carries`, ...).

        This is NOT the same function as effective_group_specs, and the
        difference has already caused one bug: resolve_features' patterns
        describe *raw FBref* column names, so running them against
        features.parquet matches nothing and silently yields zero features per
        group. Downstream that surfaces as a baffling
        "SimpleImputer: found array with 0 feature(s)".

        TODO: filter each config.GROUP_SPECS[group].features by membership in
              set(columns). No regex involved.
        """
        raise NotImplementedError("features.available_group_specs")


def possession_factor(team_possession: pd.Series) -> pd.Series:
        """Scale defensive actions to a neutral 50% possession baseline.

        A team with 65% possession gives its players only 35% of the game
        defending, so their defensive per-90s should be scaled UP by 50/35 = 1.43.
        A team with 35% possession gets scaled DOWN. Getting this backwards is
        easy and the tests check the direction.

        TODO 1. defensive_share = 100 - team_possession.
        TODO 2. factor = 50 / defensive_share, guarding against divide-by-zero
                (mask shares below ~5 with .where before dividing).
        TODO 3. Unknown possession must fall back to 1.0 (no adjustment), not
                NaN - otherwise you silently drop every player whose team row
                failed to join.
        TODO 4. .clip(lower=0.5, upper=2.0) so one freak row cannot dominate.
        """
        raise NotImplementedError("features.possession_factor")


def build(df: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
        """Compute per-90, possession-adjusted values for every resolved feature.

        TODO 1. Start the output from df[ID_COLUMNS] (intersected with what is
                actually present - not every source has `foot` or `height_in_cm`).
        TODO 2. For each resolved feature: pd.to_numeric(..., errors="coerce"),
                then divide by df["nineties"] if feature.per90, then multiply by
                possession_factor(...) if feature.padj.
        TODO 3. Guard the division: nineties of 0 produces inf, so mask it with
                .where(df["nineties"] > 0) first and .replace([inf, -inf], nan)
                after. An inf will survive all the way into StandardScaler and
                poison an entire position group.
        """
        raise NotImplementedError("features.build")


def add_percentiles(features: pd.DataFrame, specs: dict[str, list[str]]) -> pd.DataFrame:
        """Percentile rank of each feature within its position group and season.

        Rank within season as well as group, so that a league-wide shift in, say,
        pressing volume does not make every 2023-24 player look passive next to
        2025-26.

        Output needs `player_season_id`, `position_group`, and one column per
        feature holding 0-100.

        TODO 1. Loop the groups; slice features to that group.
        TODO 2. chunk.groupby("season", observed=True)[names].rank(pct=True) * 100.
        TODO 3. Re-attach player_season_id and position_group - groupby.rank
                returns only the ranked columns, aligned by index, so use
                .to_numpy() or reset_index carefully. Misaligning here silently
                attributes one player's radar to another.
        TODO 4. pd.concat the per-group frames.
        """
        raise NotImplementedError("features.add_percentiles")


def run(force: bool = False) -> pd.DataFrame:
        """Stage entry point, called by run_pipeline.py.

        TODO 1. paths.ensure_dirs(); early-return the existing parquet unless
                `force`, so a re-run after a crash resumes.
        TODO 2. Read paths.PLAYER_SEASONS (SystemExit with a useful message if
                the merge stage has not run).
        TODO 3. Filter to minutes >= INCLUSION_MIN_MINUTES and log how many rows
                survived.
        TODO 4. resolve_features -> effective_group_specs -> build -> write
                paths.FEATURES.
        TODO 5. add_percentiles -> write paths.PERCENTILES.
        TODO 6. Log per-group player counts and flag any below
                config.MIN_GROUP_SIZE. Thin groups produce meaningless
                neighbours and you want to know now, not from the app.
        """
        raise NotImplementedError("features.run")


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
