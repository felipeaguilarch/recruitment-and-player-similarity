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
TM_CONTRACTS_DIR = TM_DIR / "contracts"
FBREF_DIR = RAW / "fbref"
FBREF_ARCHIVE_DIR = RAW / "fbref_archive"
KAGGLE_DIR = RAW / "kaggle"

# --- interim ---
FBREF_WIDE = INTERIM / "fbref_player_seasons.parquet"
FBREF_TEAM = INTERIM / "fbref_team_seasons.parquet"
FBREF_ARCHIVE = INTERIM / "fbref_archive_player_seasons.parquet"
FBREF_ARCHIVE_BACKBONE = INTERIM / "fbref_archive_backbone.parquet"
FBREF_ARCHIVE_TEAMS = INTERIM / "fbref_archive_team_seasons.parquet"
FBREF_KAGGLE = INTERIM / "fbref_kaggle_player_seasons.parquet"
TM_PLAYERS = INTERIM / "tm_players.parquet"
TM_VALUATIONS = INTERIM / "tm_valuations.parquet"
TM_CONTRACTS = INTERIM / "tm_contract_observations.parquet"
TM_CLUB_SEASONS = INTERIM / "tm_club_seasons.parquet"
TM_TRANSFERS = INTERIM / "tm_transfers.parquet"
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
# Wayback captures used by fetch_fbref_archive, pinned by timestamp.
FBREF_ARCHIVE_SNAPSHOTS = DATA / "fbref_archive_snapshots.csv"
TM_CONTRACT_SNAPSHOTS = DATA / "tm_contract_snapshots.csv"

# --- reports (committed; evidence that the pipeline behaved) ---
FEATURE_RESOLUTION = REPORTS / "feature_resolution.csv"
FBREF_COLUMNS = REPORTS / "fbref_columns.txt"
FBREF_ARCHIVE_COVERAGE = REPORTS / "fbref_archive_coverage.csv"
FBREF_KAGGLE_CHECK = REPORTS / "fbref_kaggle_check.csv"
MATCH_REVIEW = REPORTS / "name_match_review.csv"
MATCH_SUMMARY = REPORTS / "name_match_summary.json"
MATCH_CLUB_CHECK = REPORTS / "name_match_club_check.csv"
PCA_LOADINGS = REPORTS / "pca_loadings.csv"
VALUE_METRICS = REPORTS / "value_model_metrics.json"
PCA_VARIANCE = REPORTS / "pca_explained_variance.csv"
VALIDATION = REPORTS / "similarity_validation.json"
UNDERVALUED = REPORTS / "undervalued_shortlist.csv"


def ensure_dirs() -> None:
        """Create every output directory. Safe to call repeatedly."""
        for path in (RAW, INTERIM, PROCESSED, ARTIFACTS, ASSETS, REPORTS, TM_DIR, TM_CONTRACTS_DIR, FBREF_DIR, FBREF_ARCHIVE_DIR):
                path.mkdir(parents=True, exist_ok=True)
