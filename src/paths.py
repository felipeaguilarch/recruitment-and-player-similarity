"""Single source of truth for every path the project reads or writes.

Everything is resolved relative to the repo root, so the pipeline and the
Streamlit app behave the same whether they are launched from the repo root,
from ``notebooks/``, or from Streamlit Community Cloud.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
REPORTS = ROOT / "reports"

# --- raw ---
TM_ZIP = RAW / "transfermarkt-datasets.zip"
TM_DIR = RAW / "transfermarkt"
FBREF_DIR = RAW / "fbref"

# --- interim ---
FBREF_WIDE = INTERIM / "fbref_player_seasons.parquet"
FBREF_TEAM = INTERIM / "fbref_team_seasons.parquet"
TM_PLAYERS = INTERIM / "tm_players.parquet"
TM_VALUATIONS = INTERIM / "tm_valuations.parquet"
NAME_MATCHES = INTERIM / "name_matches.parquet"

# --- processed (committed; the app reads these) ---
PLAYER_SEASONS = PROCESSED / "player_seasons.parquet"
FEATURES = PROCESSED / "features.parquet"
PERCENTILES = PROCESSED / "percentiles.parquet"
SIMILARITY = PROCESSED / "similarity.parquet"
STYLE_MAP = PROCESSED / "style_map.parquet"
VALUE_PREDICTIONS = PROCESSED / "value_predictions.parquet"

# --- model artefacts (committed) ---
SIMILARITY_MODELS = ARTIFACTS / "similarity_models.joblib"
VALUE_MODEL = ARTIFACTS / "value_model.joblib"

# --- human-maintained input ---
NAME_OVERRIDES = DATA / "manual_name_overrides.csv"

# --- reports (committed; evidence that the pipeline behaved) ---
FEATURE_RESOLUTION = REPORTS / "feature_resolution.csv"
FBREF_COLUMNS = REPORTS / "fbref_columns.txt"
MATCH_REVIEW = REPORTS / "name_match_review.csv"
MATCH_SUMMARY = REPORTS / "name_match_summary.json"
PCA_LOADINGS = REPORTS / "pca_loadings.csv"
VALUE_METRICS = REPORTS / "value_model_metrics.json"
PCA_VARIANCE = REPORTS / "pca_explained_variance.csv"
VALIDATION = REPORTS / "similarity_validation.json"
UNDERVALUED = REPORTS / "undervalued_shortlist.csv"


def ensure_dirs() -> None:
        """Create every output directory. Safe to call repeatedly."""
        for path in (RAW, INTERIM, PROCESSED, ARTIFACTS, ASSETS, REPORTS, TM_DIR, FBREF_DIR):
                path.mkdir(parents=True, exist_ok=True)
