"""Stage 5 - expected market value and the undervalued shortlist.
[YOU IMPLEMENT THIS]

The most club-relevant output in the project, and also the easiest to
oversell. Three choices make it defensible:

**Fit on log value.** Market values span roughly EUR 100k to EUR 180m and are
close to log-normal. On raw euros the loss is dominated by a handful of
superstars and every prediction under EUR 5m becomes noise - exactly the
range recruitment cares about.

**Group cross-validation by player.** The same player appears in several
seasons. Plain KFold lets his 2023-24 row train a model that predicts his
2024-25 row, which inflates R-squared into fiction. Try it both ways once, to
see how large the lie is.

**Contract years remaining as a feature.** One of the strongest real drivers
of transfer price, and absent from most portfolio versions of this project.

What the residual actually means: Transfermarkt values are crowd-sourced
estimates, not transfer fees. The model predicts *the crowd's valuation*, so a
negative residual says "cheaper than his profile suggests the market should
rate him", not "worth more than he costs". Keep that sentence in the README.

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
        """Assemble model inputs, dropping anything that did not resolve.

        Return (design, numeric, categorical). Sources in config:
        VALUE_NUMERIC_FEATURES, VALUE_CATEGORICAL_FEATURES, VALUE_STAT_FEATURES.

        Note why VALUE_STAT_FEATURES is one universal list rather than each
        group's own features: PC1 means something different for a centre-back
        than for a winger, so per-group scores are not comparable inside one
        pooled regression.

        TODO 1. Intersect each config list with frame.columns and warn about
                anything missing - a feature that silently vanishes here
                weakens the model with no visible symptom.
        TODO 2. Treat the stat features as numeric (numeric + stats).
        TODO 3. Cast categoricals with .astype("category") - HistGradientBoosting
                needs that to use them natively.
        """
        raise NotImplementedError("value_model.build_design_matrix")


def make_ridge(numeric: list[str], categorical: list[str]) -> Pipeline:
        """Explainable baseline. Always fit a baseline you can read.

        Age needs a SplineTransformer, not a linear term: value peaks somewhere
        around 24-26 and falls away after, and a straight line cannot represent
        that shape at all.

        TODO 1. An age block: SimpleImputer(median) -> SplineTransformer(
                n_knots=5, degree=3, include_bias=False).
        TODO 2. Other numerics: SimpleImputer(median) -> StandardScaler.
        TODO 3. Categoricals: OneHotEncoder(handle_unknown="ignore",
                min_frequency=10) - unseen leagues at predict time must not
                raise.
        TODO 4. ColumnTransformer them together, remainder="drop", then Ridge.
        """
        raise NotImplementedError("value_model.make_ridge")


def make_gbm(numeric: list[str], categorical: list[str]) -> Pipeline:
        """HistGradientBoosting with native categorical handling.

        No one-hot, no scaling, and it finds the age curve on its own.

        TODO 1. A passthrough ColumnTransformer that fixes column ORDER:
                numeric first, then categorical.
        TODO 2. HistGradientBoostingRegressor(categorical_features=[...]) - the
                indices must match the post-transform positions, i.e.
                len(numeric) + i. This is fiddly; get it wrong and categoricals
                are silently treated as continuous numbers.
        TODO 3. Set random_state so your README numbers are reproducible.
        """
        raise NotImplementedError("value_model.make_gbm")


def euro_error(actual: np.ndarray, predicted_log: np.ndarray, log_target: bool) -> dict:
        """Report error the way a person reads it, not only in log space.

        "MAE of 0.42 in log space" means nothing to a sporting director. "Median
        absolute error of 38%" does.

        TODO 1. np.expm1 the predictions if log_target (you used log1p).
        TODO 2. Return median absolute percentage error, median ratio
                (predicted/actual), and median absolute error in euros.
        TODO 3. Use np.nanmedian - some actuals may be zero or missing.
        """
        raise NotImplementedError("value_model.euro_error")


def evaluate(
        design: pd.DataFrame,
        target: np.ndarray,
        groups: np.ndarray,
        actual_eur: np.ndarray,
        numeric: list[str],
        categorical: list[str],
) -> dict:
        """Grouped cross-validation for both candidate models.

        TODO 1. GroupKFold(n_splits=min(config.CV_FOLDS, n_unique_groups)).
        TODO 2. cross_val_predict(pipeline, design, target, cv=cv, groups=groups)
                for both ridge and gbm.
        TODO 3. Return {"ridge": {...}, "gbm": {...}} with cv_mae_log, cv_r2_log
                and the euro_error fields merged in. The tests check both keys
                exist - reporting only the winner hides whether the extra
                complexity bought anything.
        """
        raise NotImplementedError("value_model.evaluate")


def run(force: bool = False) -> pd.DataFrame:
        """Stage entry point, called by run_pipeline.py.

        TODO 1. Read paths.FEATURES; keep rows with a market value at or above
                config.MIN_MARKET_VALUE_EUR. SystemExit with a pointed message
                if very few rows survive - that means the name join failed, and
                you want to be sent back to reports/name_match_summary.json
                rather than debugging the regression.
        TODO 2. target = np.log1p(actual) when config.VALUE_LOG_TRANSFORM.
        TODO 3. groups = the player name/id array. Not the season, not the club.
        TODO 4. evaluate(...), pick the lower cv_mae_log, and log which won.
        TODO 5. Build the shortlist from OUT-OF-FOLD predictions
                (cross_val_predict with the chosen model), not from the model
                refitted on everything. In-sample predictions shrink every
                residual toward zero and hide exactly the players the tool
                exists to surface. Then fit on all data and joblib.dump that for
                the app.
        TODO 6. log_residual = target - oof_pred; negative means cheaper than
                predicted. value_ratio = predicted_eur / actual_eur reads better
                for humans ("2.7x gap").
        TODO 7. undervalued = log_residual <= -config.UNDERVALUED_LOG_THRESHOLD.
        TODO 8. Write paths.VALUE_PREDICTIONS, the top ~100 to paths.UNDERVALUED,
                and metrics to paths.VALUE_METRICS including "selected_model".

        Then go and look at the shortlist. If it is all 33-year-olds in one
        league, the model has learnt league and age and little else; that is a
        finding about the model, and the fix is to read residuals inside a
        league and budget filter rather than globally.
        """
        raise NotImplementedError("value_model.run")


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
