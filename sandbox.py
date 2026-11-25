"""Interactive sandbox - try one function at a time on synthetic data.

The test suite's `pipeline` fixture runs every stage in one go, which is right
for CI but slow for experimenting with a single function. This builds the same
synthetic frames outside pytest and leaves them in the interactive namespace:

    python -i sandbox.py

    >>> frame.shape
    >>> cb = frame[frame["position_group"] == "CB"].reset_index(drop=True)
    >>> fit_group(cb, specs["CB"])

Everything writes into a temp directory, so nothing here can touch
data/processed or artifacts/.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from src import config, paths  # noqa: E402

# --- redirect every output path into a throwaway directory -----------------

_root = Path(tempfile.mkdtemp(prefix="sandbox_"))
for _sub in ("processed", "artifacts", "reports", "assets"):
        (_root / _sub).mkdir()

_redirect = {
        "PROCESSED": _root / "processed", "ARTIFACTS": _root / "artifacts",
        "REPORTS": _root / "reports", "ASSETS": _root / "assets",
        "PLAYER_SEASONS": _root / "processed/player_seasons.parquet",
        "FEATURES": _root / "processed/features.parquet",
        "PERCENTILES": _root / "processed/percentiles.parquet",
        "SIMILARITY": _root / "processed/similarity.parquet",
        "STYLE_MAP": _root / "processed/style_map.parquet",
        "VALUE_PREDICTIONS": _root / "processed/value_predictions.parquet",
        "ROLES": _root / "processed/roles.parquet",
        "ROLE_SCORES": _root / "processed/role_scores.parquet",
        "SIMILARITY_MODELS": _root / "artifacts/similarity_models.joblib",
        "VALUE_MODEL": _root / "artifacts/value_model.joblib",
        "FEATURE_RESOLUTION": _root / "reports/feature_resolution.csv",
        "PCA_LOADINGS": _root / "reports/pca_loadings.csv",
        "PCA_VARIANCE": _root / "reports/pca_explained_variance.csv",
        "VALIDATION": _root / "reports/similarity_validation.json",
        "VALUE_METRICS": _root / "reports/value_model_metrics.json",
        "UNDERVALUED": _root / "reports/undervalued_shortlist.csv",
        "ROLE_STABILITY": _root / "reports/role_stability.csv",
        "ROLE_TOP_PLAYERS": _root / "reports/role_top_players.csv",
}
for _key, _value in _redirect.items():
        setattr(paths, _key, _value)
paths.ensure_dirs = lambda: None

# --- the same synthetic table the tests use --------------------------------

# `tests/` is not a package, so import the module by file path rather than
# relying on pytest's rootdir insertion.
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
        "_synth", Path(__file__).parent / "tests" / "test_pipeline_synthetic.py"
)
_synth = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_synth)

LEAGUES = _synth.LEAGUES
N_PLAYERS = _synth.N_PLAYERS
SEASONS = _synth.SEASONS
SUB_POSITIONS = _synth.SUB_POSITIONS
concrete_column = _synth.concrete_column


def build_player_seasons() -> pd.DataFrame:
        """Copy of the test fixture's generator, callable outside pytest."""
        rng = np.random.default_rng(20260211)
        rows = []
        for index in range(N_PLAYERS):
                group = config.POSITION_GROUPS[index % len(config.POSITION_GROUPS)]
                for season in SEASONS:
                        minutes = float(rng.integers(650, 3200))
                        rows.append({
                                "player": f"Player {index:03d}",
                                "team": f"Club {index % 24:02d}",
                                "league": LEAGUES[index % len(LEAGUES)],
                                "season": season,
                                "pos": "DF" if group in ("CB", "FB") else "MF",
                                "nation": "eng ENG",
                                "age": float(rng.uniform(17, 36)),
                                "born": 2000 - (index % 15),
                                "minutes": minutes,
                                "nineties": minutes / 90.0,
                                "team_possession": float(rng.uniform(38, 64)),
                                "position_group": group,
                                "position_source": "transfermarkt",
                                "tm_sub_position": SUB_POSITIONS[group],
                                "tm_player_id": 100000 + index,
                                "market_value_eur": float(np.exp(rng.normal(15.5, 1.2))),
                                "contract_years_remaining": float(rng.uniform(0.2, 5.0)),
                                "player_season_id": f"Player_{index:03d}__{season}__Club_{index % 24:02d}",
                        })

        out = pd.DataFrame(rows)
        for name, feature in config.FEATURES.items():
                column = concrete_column(feature.patterns[0])
                if feature.per90:
                        out[column] = rng.gamma(2.0, 8.0, len(out)) * (out["nineties"] / 30.0)
                else:
                        out[column] = rng.uniform(20, 85, len(out))
        out["possession__nonsense__unused"] = 1.0
        return out


from src import features as features_mod  # noqa: E402

build_player_seasons().to_parquet(paths.PLAYER_SEASONS, index=False)
features_mod.run(force=True)

frame = pd.read_parquet(paths.FEATURES)
specs = features_mod.available_group_specs(list(frame.columns))

# Import the stage modules last, so an error in one of them cannot stop the
# frames above from being built.
from src import figures as figures_mod  # noqa: E402,F401
from src import radar as radar_mod  # noqa: E402,F401
from src import roles as roles_mod  # noqa: E402,F401
from src import similarity as similarity_mod  # noqa: E402
from src import value_model as value_mod  # noqa: E402,F401
from src.similarity import (  # noqa: E402,F401
        choose_n_components, fit_group, loadings_frame, neighbours_frame,
        style_map, top_comps, validate,
)

print()
print(f"sandbox ready   outputs -> {_root}")
print(f"  frame   {frame.shape[0]} rows x {frame.shape[1]} cols")
print(f"  specs   {', '.join(f'{g}({len(f)})' for g, f in specs.items())}")
print("  modules similarity_mod, roles_mod, value_mod, radar_mod, figures_mod")
