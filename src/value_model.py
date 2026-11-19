"""Stage 5 - expected market value and the undervalued shortlist.

The most club-relevant output in the project, and also the easiest to
oversell. Three choices make it defensible:

**Fit on log value.** Market values span roughly EUR 100k to EUR 180m and are
close to log-normal. On raw euros the loss is dominated by a handful of
superstars and every prediction under EUR 5m becomes noise - exactly the
range recruitment cares about.

**Group cross-validation by player.** The same player appears in several
seasons. Plain KFold lets his 2022-23 row train a model that predicts his
2024-25 row, which inflates R-squared into fiction.

**Contract years remaining as a feature.** One of the strongest real drivers
of transfer price, and absent from most portfolio versions of this project.

Two candidates are scored on the same folds - a Ridge baseline with a spline
on age, and HistGradientBoosting - and the lower log-space MAE wins. On the
real data they finish level, so the readable model is the one kept.

What the residual means: Transfermarkt values are crowd-sourced estimates, not
transfer fees. The model predicts *the crowd's valuation*, so a negative
residual says "cheaper than his profile suggests the market should rate him",
not "worth more than he costs".

Outputs:
  artifacts/value_model.joblib            selected model refitted on all rows
  data/processed/value_predictions.parquet  out-of-fold prediction per row
  reports/undervalued_shortlist.csv       top 100 flagged, regular starters
  reports/value_model_metrics.json        CV metrics for both candidates

Spec: the value tests in tests/test_pipeline_synthetic.py.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from . import config, paths

logger = logging.getLogger(__name__)


def build_design_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Assemble the model inputs from the configured feature lists.

        Numeric features are config.VALUE_NUMERIC_FEATURES followed by
        config.VALUE_STAT_FEATURES; categoricals are
        config.VALUE_CATEGORICAL_FEATURES. Anything configured but absent from
        `frame` is skipped with a warning - a feature that silently vanished
        would weaken the model with no other visible symptom. Categoricals are
        cast to the pandas "category" dtype, which HistGradientBoosting needs
        to treat them natively.

        The stat features are one universal list rather than each group's own,
        because PC1 means something different for a centre-back than for a
        winger and per-group scores are not comparable in one pooled
        regression.

        Returns (design, numeric, categorical): the design frame, numeric
        columns first, and the two name lists in that same order.

        Raises ValueError if none of the configured features are present.
        """
        numeric_candidates = (config.VALUE_NUMERIC_FEATURES + config.VALUE_STAT_FEATURES)
        categorical_candidates = config.VALUE_CATEGORICAL_FEATURES
        configured = numeric_candidates + categorical_candidates

        missing = [name for name in configured if name not in frame.columns]
        if missing:
                logger.warning(
                        "Value-model features missing from input and will be skipped: %s",
                        missing)

        numeric = [name for name in numeric_candidates if name in frame.columns]
        categorical = [name for name in categorical_candidates if name in frame.columns]

        if not numeric and not categorical:
                raise ValueError("No configured value-model features are present in the input.")

        design = frame[numeric + categorical].copy()
        for name in categorical:
                design[name] = design[name].astype("category")

        return design, numeric, categorical


def make_ridge(numeric: list[str], categorical: list[str]) -> Pipeline:
        """Ridge regression - the explainable baseline.

        Three column blocks feed the Ridge:

        - age: median-imputed, then a cubic spline with 5 knots. Value peaks
          around 24-26 and falls away after, a shape a straight line cannot
          represent.
        - other numerics: median-imputed, plus a was-missing indicator for
          each column that had gaps, then standardised. The indicator matters
          for contract_years_remaining: about 12% of rows have no dated
          contract, and without it the model reads those as a median-length
          contract and over-flags them as undervalued.
        - categoricals: one-hot, with levels seen fewer than 10 times pooled
          and unseen levels ignored, so a new league at predict time does not
          raise.

        Columns not named in `numeric` or `categorical` are dropped.

        Raises ValueError if both lists are empty.
        """
        transformers = []

        if "age" in numeric:
                age_pipeline = Pipeline(
                        [
                                ("impute", SimpleImputer(strategy="median")),
                                (
                                        "spline",
                                        SplineTransformer(n_knots=5,degree=3,include_bias=False),
                                ),
                        ]
                )
                transformers.append(("age", age_pipeline, ["age"]))

        other_numeric = [name for name in numeric if name != "age"]
        if other_numeric:
                numeric_pipeline = Pipeline(
                        [
                                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                ("scale", StandardScaler()),
                        ]
                )
                transformers.append(("numerics", numeric_pipeline, other_numeric))

        if categorical:
                transformers.append(
                        (
                                "categoricals",
                                OneHotEncoder(handle_unknown="ignore",min_frequency=10),
                                categorical,
                        )
                )

        if not transformers:
                raise ValueError("Ridge requires at least one numeric or categorical feature.")

        transformer = ColumnTransformer(transformers=transformers, remainder="drop")
        return Pipeline([("transformer", transformer), ("regressor", Ridge())])


def make_gbm(numeric: list[str], categorical: list[str]) -> Pipeline:
        """HistGradientBoosting with native categorical handling.

        No one-hot, no scaling and no imputation - the trees handle missing
        values themselves and find the age curve on their own.

        A passthrough ColumnTransformer fixes the column order to numeric
        first, then categorical, so the categorical columns sit at positions
        len(numeric) + i. `categorical_features` is built from those
        positions; get them wrong and categoricals are silently treated as
        continuous numbers. random_state is fixed so the README numbers
        reproduce.

        Raises ValueError if both lists are empty.
        """
        if not numeric and not categorical:
                raise ValueError("GBM requires at least one numeric or categorical feature.")

        transformers = []
        if numeric:
                transformers.append(("numeric", "passthrough", numeric))
        if categorical:
                transformers.append(("categorical", "passthrough", categorical))

        transformer = ColumnTransformer(transformers=transformers,
                remainder="drop",verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        categorical_features = list(range(len(numeric), len(numeric) + len(categorical)))
        regressor = HistGradientBoostingRegressor(categorical_features=categorical_features,
                random_state=42)
        return Pipeline([("transformer", transformer), ("regressor", regressor)])


def euro_error(actual: np.ndarray, predicted_log: np.ndarray, log_target: bool) -> dict:
        """Prediction error in the units a person reads, not log space.

        "MAE of 0.6 in log space" means nothing to a sporting director;
        "median error of 45%" does. Predictions are converted back with
        np.expm1 when `log_target` is true (the target was built with log1p).
        Rows whose actual value is missing, non-finite or not positive are
        left out, since a percentage error against them is undefined.

        Returns a dict of floats:

        - median_absolute_percentage_error: median |predicted - actual| /
          actual, in percent.
        - median_ratio: median predicted / actual. Near 1 means no overall
          bias; below 1 means the model undershoots.
        - median_absolute_error_eur: median |predicted - actual| in euros.

        All three are NaN when no valid rows remain.
        """
        actual = np.asarray(actual, dtype=float).reshape(-1)
        predicted = np.asarray(predicted_log, dtype=float).reshape(-1)
        if log_target:
                predicted = np.expm1(predicted)

        valid = (np.isfinite(actual) & (actual > 0) & np.isfinite(predicted))
        actual = actual[valid]
        predicted = predicted[valid]

        if actual.size == 0:
                return {
                        "median_absolute_percentage_error": np.nan,
                        "median_ratio": np.nan,
                        "median_absolute_error_eur": np.nan,
                }

        absolute_percentage_error = (
                np.abs(predicted - actual) / actual
        ) * 100.0
        ratios = predicted / actual
        absolute_error_eur = np.abs(predicted - actual)

        return {
                "median_absolute_percentage_error": float(np.nanmedian(absolute_percentage_error)),
                "median_ratio": float(np.nanmedian(ratios)),
                "median_absolute_error_eur": float(np.nanmedian(absolute_error_eur)),
        }


def evaluate(design: pd.DataFrame,target: np.ndarray,
        groups: np.ndarray,actual_eur: np.ndarray,
        numeric: list[str],categorical: list[str],) -> dict:
        """Score both candidate models with grouped cross-validation.

        Each model predicts every row out of fold through GroupKFold, with
        min(config.CV_FOLDS, number of players) splits, so no player is ever
        in both the training and the test side of a fold. Rows with a
        non-finite target or a missing group are dropped first.

        Both models are reported, not only the winner - otherwise there is no
        way to see whether the extra complexity of boosting bought anything.

        Returns {"ridge": {...}, "gbm": {...}}, each holding cv_mae_log and
        cv_r2_log (in target space) plus the euro_error fields.

        Raises ValueError if the inputs differ in length or cover fewer than
        two players.
        """
        target = np.asarray(target, dtype=float).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
        actual_eur = np.asarray(actual_eur, dtype=float).reshape(-1)

        if len(design) != target.size or len(design) != groups.size or len(design) != actual_eur.size:
                raise ValueError("Value-model design, target, groups, and actual values must have the same length.")

        valid_row = np.isfinite(target) & pd.notna(groups)
        if not valid_row.all():
                design = design.loc[valid_row].copy()
                target = target[valid_row]
                groups = groups[valid_row]
                actual_eur = actual_eur[valid_row]

        n_unique_groups = len(pd.unique(groups))
        if n_unique_groups < 2:
                raise ValueError("Value-model evaluation needs at least two player groups.")

        cv = GroupKFold(n_splits=min(config.CV_FOLDS, n_unique_groups))
        models = {
                "ridge": make_ridge(numeric, categorical),
                "gbm": make_gbm(numeric, categorical),
        }

        results = {}
        for name, pipeline in models.items():
                predicted_log = cross_val_predict(
                        pipeline,
                        design,
                        target,
                        cv=cv,
                        groups=groups,
                )
                model_result = {
                        "cv_mae_log": float(mean_absolute_error(target, predicted_log)),
                        "cv_r2_log": float(r2_score(target, predicted_log)),
                }
                model_result.update(
                        euro_error(
                                actual_eur,
                                predicted_log,
                                log_target=config.VALUE_LOG_TRANSFORM,
                        )
                )
                results[name] = model_result

        return results


def player_groups(frame: pd.DataFrame) -> np.ndarray:
        """One string key per player, for grouped cross-validation.

        The key is the Transfermarkt id ("tm:123") where the name join found
        one. Otherwise it falls back to the FBref name and birth year
        ("fbref:Name|2001"), or to the player part of player_season_id. Every
        key is a string: GroupKFold sorts the groups, and a mix of numeric ids
        and name strings cannot be sorted.

        Groups are players - never seasons or clubs - so all of a player's
        seasons fall in the same fold.

        Raises SystemExit if the frame has no player identifier at all.
        """
        if "player" in frame.columns:
                born = frame["born"].astype(str) if "born" in frame.columns else ""
                fallback = "fbref:" + frame["player"].astype(str) + "|" + born
        elif "player_season_id" in frame.columns:
                fallback = "fbref:" + frame["player_season_id"].str.rsplit("__", n=1).str[0]
        else:
                fallback = None

        if "tm_player_id" in frame.columns:
                tm_id = pd.to_numeric(frame["tm_player_id"], errors="coerce").astype("Int64")
                keys = "tm:" + tm_id.astype(str)
                if fallback is not None:
                        keys = keys.where(tm_id.notna(), fallback)
                elif tm_id.isna().any():
                        raise SystemExit("Some rows have no tm_player_id and the features "
                                "table has no player column to fall back on.")
        elif fallback is not None:
                keys = fallback
        else:
                raise SystemExit(
                        "The features table has no player identifier for grouped cross-validation."
                )
        return keys.astype(str).to_numpy()


def run(force: bool = False) -> pd.DataFrame:
        """Stage entry point, called by run_pipeline.py.

        Reuses the cached outputs unless `force` is set or one of them is
        missing. Otherwise:

        1. Reads paths.FEATURES and keeps rows with a market value of at least
           config.MIN_MARKET_VALUE_EUR. Stops with a pointer to
           reports/name_match_summary.json if fewer than 100 rows survive, or
           fewer than config.MIN_VALUE_COVERAGE of them - either means the
           name join failed, and debugging the regression would be the wrong
           place to look.
        2. Fits on log1p(value) when config.VALUE_LOG_TRANSFORM, grouped by
           player (see player_groups).
        3. Scores both candidates with evaluate() and keeps the one with the
           lower cv_mae_log.
        4. Predicts every row OUT OF FOLD with the selected model. In-sample
           predictions shrink every residual toward zero and hide exactly the
           players the shortlist exists to surface. The model refitted on all
           rows is saved for the app only.
        5. Adds predicted_value_eur, log_residual (actual minus predicted, so
           negative means cheaper than predicted), value_ratio (predicted /
           actual, which reads as "a 2.7x gap") and undervalued
           (log_residual <= -config.UNDERVALUED_LOG_THRESHOLD).

        Writes paths.VALUE_MODEL, paths.VALUE_PREDICTIONS (every row),
        paths.UNDERVALUED and paths.VALUE_METRICS (both models plus
        "selected_model"). The shortlist is the 100 most undervalued players
        with at least config.MIN_MINUTES - per-90 profiles from fewer minutes
        are too noisy to recommend anyone on, though those rows keep their
        flag in the predictions file.

        Mean residuals by league and by age band sit near zero on the real
        data, so the shortlist is not just one league or one age bracket.
        Read it inside a league and budget filter anyway: the top of it is
        mostly breakthrough youngsters whose Transfermarkt value had not
        caught up yet, which is what the backtest checks.

        Returns the predictions frame.
        """
        paths.ensure_dirs()

        if (
                not force
                and paths.VALUE_PREDICTIONS.exists()
                and paths.VALUE_MODEL.exists()
                and paths.VALUE_METRICS.exists()
        ):
                logger.info("Using cached value predictions: %s", paths.VALUE_PREDICTIONS)
                return pd.read_parquet(paths.VALUE_PREDICTIONS)

        try:
                frame = pd.read_parquet(paths.FEATURES)
        except FileNotFoundError as exc:
                raise SystemExit(f"Missing features file {paths.FEATURES}. "
                        f"Run 'python run_pipeline.py --stage features' first.") from exc

        if "market_value_eur" not in frame.columns:
                raise SystemExit(f"No market_value_eur column in {paths.FEATURES}. "
                        f"Check reports/name_match_summary.json and rerun merge.")

        actual = pd.to_numeric(frame["market_value_eur"], errors="coerce")
        n_features = len(frame)
        frame = frame.loc[actual >= config.MIN_MARKET_VALUE_EUR].copy()
        actual = pd.to_numeric(frame["market_value_eur"], errors="coerce")

        coverage = len(frame) / n_features if n_features else 0.0
        if len(frame) < 100 or coverage < config.MIN_VALUE_COVERAGE:
                raise SystemExit(f"Only {len(frame)} of {n_features} rows ({coverage:.0%}) "
                        f"have a market value of at least {config.MIN_MARKET_VALUE_EUR:,} EUR. "
                        f"The name join probably failed; inspect reports/name_match_summary.json.")

        design, numeric, categorical = build_design_matrix(frame)
        if config.VALUE_USE_PCA_SCORES:
                pca_names = [f"pc{i}" for i in range(1, config.VALUE_N_PCA_SCORES + 1)
                        if f"pc{i}" in frame.columns
                ]
                for name in pca_names:
                        if name not in numeric:
                                numeric.append(name)
                                design[name] = frame[name]

        actual_eur = actual.to_numpy(dtype=float)
        if config.VALUE_LOG_TRANSFORM:
                target = np.log1p(actual_eur)
        else:
                target = actual_eur.copy()

        groups = player_groups(frame)

        metrics = evaluate(design, target, groups, actual_eur, numeric, categorical)
        selected_model = min(metrics, key=lambda name: metrics[name]["cv_mae_log"])
        logger.info("Value model selected %s (CV MAE log=%.4f)",
                selected_model,metrics[selected_model]["cv_mae_log"])

        cv = GroupKFold(n_splits=min(config.CV_FOLDS, len(pd.unique(groups))))
        if selected_model == "ridge":
                pipeline = make_ridge(numeric, categorical)
        else:
                pipeline = make_gbm(numeric, categorical)

        oof_pred = cross_val_predict(pipeline, design, target, cv=cv, groups=groups)
        if config.VALUE_LOG_TRANSFORM:
                predicted_eur = np.expm1(oof_pred)
        else:
                predicted_eur = oof_pred
        predicted_eur = np.maximum(predicted_eur, 0.0)

        predictions = frame.copy()
        predictions["predicted_value_eur"] = predicted_eur
        predictions["log_residual"] = target - oof_pred
        predictions["value_ratio"] = predicted_eur / actual_eur
        predictions["undervalued"] = (predictions["log_residual"] <= -config.UNDERVALUED_LOG_THRESHOLD)

        pipeline.fit(design, target)
        joblib.dump(pipeline, paths.VALUE_MODEL)

        predictions.to_parquet(paths.VALUE_PREDICTIONS, index=False)

        # Per-90 profiles from a few hundred minutes are too noisy to recommend
        # anyone on; those rows keep their flag in the predictions file.
        regular = predictions["minutes"] >= config.MIN_MINUTES
        shortlisted = predictions.loc[predictions["undervalued"] & regular].copy()
        shortlisted = shortlisted.sort_values(["log_residual", "value_ratio"],ascending=[True, False]).head(100)
        shortlisted.to_csv(paths.UNDERVALUED, index=False)

        metrics["selected_model"] = selected_model
        paths.VALUE_METRICS.write_text(json.dumps(metrics, indent=2),encoding="utf-8")

        logger.info("Wrote %d value predictions and %d undervalued shortlist rows",
                len(predictions),len(shortlisted))
        return predictions


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
