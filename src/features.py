"""Stage 3 - raw FBref columns into model-ready features.

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

        Each config.Feature carries `.patterns`, a tuple of regexes tried in
        order against the flattened column names; the first pattern that
        matches anything wins. This is how `interceptions` prefers
        defense__flat__int over the duplicate misc__performance__int. When one
        pattern matches several columns the alphabetically first is taken, so
        runs are reproducible.

        Writes the report to paths.FEATURE_RESOLUTION and logs a warning
        listing every feature that resolved to nothing. A MISSING row is the
        signal to fix that feature's patterns in config.py against
        reports/fbref_columns.txt.

        Returns (resolved, report). `resolved` is {feature_name: column} for
        every feature that matched. `report` has one row per catalogue feature:
        feature, status ("ok" or "MISSING"), column, matched_pattern,
        patterns_tried, per90 and possession_adjusted.
        """
        resolved: dict[str, str] = {}
        rows: list[dict[str,object]] = []

        for feature_name, feature in config.FEATURES.items():
                matched_column = ""
                matched_pattern = ""
                status = "MISSING"

                for pattern in feature.patterns:
                        candidates = sorted(
                                c for c in columns if re.match(pattern, c)
                        )

                        if candidates:
                                matched_column = candidates[0]
                                matched_pattern = pattern
                                status = "ok"
                                resolved[feature_name] = matched_column
                                break

                rows.append({
                        "feature": feature_name,
                        "status": status,
                        "column": matched_column,
                        "matched_pattern": matched_pattern,
                        "patterns_tried": " | ".join(feature.patterns),
                        "per90": feature.per90,
                        "possession_adjusted": feature.padj,
                })

        report = pd.DataFrame(rows,
                              columns=[
                                        "feature",
                                        "status",
                                        "column",
                                        "matched_pattern",
                                        "patterns_tried",
                                        "per90",
                                        "possession_adjusted"
                              ])

        missing = report.loc[report["status"] == "MISSING", "feature"].tolist()

        if missing:
                logger.warning("Unresolved feature patterns: %s", ", ".join(missing))

        paths.FEATURE_RESOLUTION.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(paths.FEATURE_RESOLUTION, index=False)

        return resolved, report


def effective_group_specs(resolved: dict[str, str]) -> dict[str, list[str]]:
        """Each group's feature list, minus anything that failed to resolve.

        Use this on the RAW FBref frame, with `resolved` from resolve_features.
        Logs a warning naming what each group lost, and an error when a group
        falls below six usable features - at that point its profile is too thin
        to trust.

        Returns {group: [feature_name, ...]} in config.GROUP_SPECS order.
        """

        effective: dict[str, list[str]] = {}

        for group, spec in config.GROUP_SPECS.items():

                usable = [feature for feature in spec.features if feature in resolved]
                trash = [feature for feature in spec.features if feature not in resolved]

                if trash:
                        logger.warning(
                                "Group %s lost features: %s", group, ", ".join(trash)
                        )

                if len(usable) < 6:
                        logger.error(
                                "Group %s has only %d usable features after resolution: %s",
                                group, len(usable), usable
                        )

                effective[group] = usable

        return effective


def available_group_specs(columns: list[str]) -> dict[str, list[str]]:
        """Each group's feature list, restricted to columns present in a frame.

        Use this on features.parquet, whose columns are already the catalogue's
        canonical names (`npxg`, `prog_carries`, ...). It is a plain membership
        test, no regex.

        This is NOT the same function as effective_group_specs, and the
        difference has already caused one bug: resolve_features' patterns
        describe *raw FBref* column names, so running them against
        features.parquet matches nothing and silently yields zero features per
        group. Downstream that surfaces as a baffling "SimpleImputer: found
        array with 0 feature(s)".

        Returns {group: [feature_name, ...]}; a group can come back empty.
        """
        available: dict[str, list[str]] = {}

        for group, spec in config.GROUP_SPECS.items():
                usable = [feature for feature in spec.features if feature in columns]
                available[group] = usable

        return available


def possession_factor(team_possession: pd.Series) -> pd.Series:
        """Multiplier that scales defensive actions to a neutral 50% possession.

        A team with 65% possession leaves its players only 35% of the game to
        defend, so their defensive per-90s are scaled UP by 50/35 = 1.43; a
        team with 35% possession is scaled DOWN. The tests check the direction.

        A defensive share of 5% or less is treated as unknown rather than
        divided by, and unknown possession falls back to 1.0 (no adjustment),
        so a player whose team row failed to join is kept instead of turning
        into NaN. The result is clipped to [0.5, 2.0] so one freak row cannot
        dominate.
        """
        defensive_share = 100 - team_possession
        factor = 50 / defensive_share.where(defensive_share > 5, np.nan)
        factor = factor.fillna(1.0).clip(lower=0.5, upper=2.0)
        return factor


def build(df: pd.DataFrame, resolved: dict[str, str]) -> pd.DataFrame:
        """Compute per-90, possession-adjusted values for every resolved feature.

        The output starts from whichever ID_COLUMNS are present in `df`. Each
        resolved column is then coerced to numeric, divided by `nineties` if
        the feature is per-90, and multiplied by possession_factor if it is
        possession-adjusted.

        Rows with zero nineties become NaN rather than inf, and any remaining
        inf is replaced with NaN: a single inf survives all the way into
        StandardScaler and poisons a whole position group.

        Returns one row per input row: the ID columns plus one column per
        feature, named by catalogue feature name rather than FBref column.
        """
        # Intersect rather than index directly: ID_COLUMNS lists everything the
        # output *may* carry, but `foot`, `height_in_cm` and
        # `country_of_citizenship` only exist when the Transfermarkt join
        # supplied them. df[ID_COLUMNS] raises KeyError whenever one is absent.
        output = df[[c for c in ID_COLUMNS if c in df.columns]].copy()

        for feature_name, column in resolved.items():
                feature = config.FEATURES[feature_name]
                series = pd.to_numeric(df[column], errors="coerce")

                if feature.per90:
                        series = series.where(df["nineties"] > 0) / df["nineties"]

                if feature.padj:
                        series *= possession_factor(df["team_possession"])

                series = series.replace([np.inf, -np.inf], np.nan)
                output[feature_name] = series

        return output


def add_percentiles(features: pd.DataFrame, specs: dict[str, list[str]]) -> pd.DataFrame:
        """Percentile rank of each feature within its position group and season.

        Ranking within season as well as group means a league-wide shift in,
        say, pressing volume does not make every 2023-24 player look passive
        next to 2025-26. Ties take the highest rank (method="max"), so the best
        player in a group-season always scores 100.

        Each group's ranks are re-attached to player_season_id positionally,
        after resetting both indexes. Misaligning that step would silently
        attribute one player's radar to another.

        Returns player_season_id, position_group and one 0-100 column per
        feature. A feature a group does not use is NaN for that group's rows;
        groups with no usable features or no rows are skipped.
        """
        frames: list[pd.DataFrame] = []

        for group, names in specs.items():
                usable = [name for name in names if name in features.columns]

                if not usable:
                        continue

                group_rows = features.loc[
                        features["position_group"] == group,
                        ["player_season_id", "position_group", "season", *usable]
                ].copy()

                if group_rows.empty:
                        continue

                ranked = (
                        group_rows.groupby("season", observed=True)[usable]
                        .rank(pct=True, method="max") * 100
                )

                frame = pd.concat(
                        [
                                group_rows[["player_season_id", "position_group"]].reset_index(drop=True),
                                ranked.reset_index(drop=True)
                        ],
                        axis=1
                )

                frames.append(frame)

        if not frames:
                return pd.DataFrame(columns=["player_season_id", "position_group"])

        return pd.concat(frames, ignore_index=True)


def run(force: bool = False) -> pd.DataFrame:
        """Stage entry point, called by run_pipeline.py.

        Returns the cached features parquet unless `force`, so a re-run after a
        crash resumes. Otherwise reads paths.PLAYER_SEASONS, drops
        player-seasons under INCLUSION_MIN_MINUTES, resolves and builds the
        features, and writes paths.FEATURES and paths.PERCENTILES.

        Finally logs each group's player count and warns on any group below
        config.MIN_GROUP_SIZE. Thin groups produce meaningless neighbours, and
        it is better to find out here than in the app.

        Raises SystemExit if the merge stage has not produced its output yet.
        """

        paths.ensure_dirs()

        if paths.FEATURES.exists() and not force:
                logger.info("Using cached features: %s", paths.FEATURES)
                return pd.read_parquet(paths.FEATURES)

        try:
                df = pd.read_parquet(paths.PLAYER_SEASONS)
        except FileNotFoundError as exc:
                raise SystemExit(
                        f"Run 'python run_pipeline.py --stage merge' first"
                ) from exc

        before = len(df)
        df = df.loc[df["minutes"] >= INCLUSION_MIN_MINUTES].copy()
        logger.info(
                "Feature stage: %d/%d player-seasons survived the minutes filter.",
                len(df), before
        )

        resolved, _ = resolve_features(list(df.columns))
        specs = effective_group_specs(resolved)

        features_frame = build(df, resolved)
        features_frame.to_parquet(paths.FEATURES, index=False)

        percentiles = add_percentiles(features_frame, specs)
        percentiles.to_parquet(paths.PERCENTILES, index=False)

        for group in specs:
                group_c = int((features_frame["position_group"] == group).sum())

                logger.info("Group %s: %d players after filtering.", group, group_c)

                if group_c < config.MIN_GROUP_SIZE:
                        logger.warning(
                                "Group %s is below MIN_GROUP_SIZE (%d < %d).",
                                group,
                                group_c,
                                config.MIN_GROUP_SIZE,
                        )

        return features_frame

if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
