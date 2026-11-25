"""Run the whole pipeline on synthetic data, with no network.

`fetch-fbref` takes one to two hours on a cold cache. Finding a wiring bug
after that wait is miserable, so this builds a player-season table with the
same *shape* as the real one - FBref-style flattened column names, a plausible
spread of positions, ages, minutes and market values - and pushes it through
features -> roles -> similarity -> value -> figures for real.

It verifies plumbing, not football: that feature resolution finds the columns,
that per-90 and possession adjustment produce finite numbers, that PCA and the
neighbour search agree on their index, that the value model trains and writes a
shortlist, and that every figure renders. Whether the comps are any *good* is
what notebooks/validation.ipynb is for.

    pytest tests/test_pipeline_synthetic.py -q

A stage that still raises NotImplementedError does not error the whole file:
its tests (and those of every stage after it) are SKIPPED with a note saying
which stage is unfinished. Figure builders are checked one by one, so a single
unimplemented figure is skipped while the rest are still verified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config

N_PLAYERS = 420
SEASONS = ("2324", "2425")
LEAGUES = ("ENG-Premier League", "ESP-La Liga", "ITA-Serie A")

#: Stage name -> the stub that stopped it, filled in by the `pipeline` fixture.
UNFINISHED: dict[str, str] = {}

#: Figure builders (figures.FIGURES keys) that still raise NotImplementedError.
#: figures.run() swallows that per figure, so the fixture runs each one itself.
PENDING_FIGURES: set[str] = set()

#: Figure builders that raised anything else, with the error. Recorded rather
#: than raised so a bug in one figure fails only the figures test, not every
#: test that shares the pipeline fixture.
FAILED_FIGURES: dict[str, str] = {}

#: The file each figure builder writes.
FIGURE_FILES = {
        "radar": "radar_example.png",
        "pca_variance": "pca_variance.png",
        "age_curve": "age_curve.png",
        "value_scatter": "value_scatter.png",
        "style_map": "style_map.png",
}


def _run_figures(figures_mod) -> None:
        """Build each figure separately, recording stubs and failures."""
        PENDING_FIGURES.clear()
        FAILED_FIGURES.clear()
        for name, builder in figures_mod.FIGURES.items():
                try:
                        builder()
                except NotImplementedError:
                        PENDING_FIGURES.add(name)
                except Exception as exc:  # noqa: BLE001
                        FAILED_FIGURES[name] = f"{type(exc).__name__}: {exc}"
        if PENDING_FIGURES == set(figures_mod.FIGURES):
                raise NotImplementedError("every figure builder")


def needs(*stages: str) -> None:
        """Skip the calling test if any of these stages is still a stub."""
        for stage in stages:
                if stage in UNFINISHED:
                        pytest.skip(f"{stage} not implemented yet ({UNFINISHED[stage]})")

SUB_POSITIONS = {
        "CB": "Centre-Back",
        "FB": "Left-Back",
        "DM": "Defensive Midfield",
        "CM": "Central Midfield",
        "AM": "Attacking Midfield",
        "W": "Left Winger",
        "ST": "Centre-Forward",
}


def concrete_column(pattern: str) -> str:
        """Turn a feature pattern from config into one column name it matches."""
        return pattern.strip("^$").replace(".*", "flat")


@pytest.fixture(scope="module")
def synthetic_player_seasons() -> pd.DataFrame:
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

        frame = pd.DataFrame(rows)

        # Synthetic raw FBref columns, named so that every catalogue pattern
        # resolves. Scaled to roughly realistic magnitudes: counting stats as
        # season totals, percentages as percentages.
        for name, feature in config.FEATURES.items():
                column = concrete_column(feature.patterns[0])
                if feature.per90:
                        frame[column] = rng.gamma(2.0, 8.0, len(frame)) * (frame["nineties"] / 30.0)
                else:
                        frame[column] = rng.uniform(20, 85, len(frame))

        # A column that resolves nowhere, to prove extras are ignored rather than
        # crashing the resolver.
        frame["possession__nonsense__unused"] = 1.0
        return frame


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory, synthetic_player_seasons):
        """Redirect every output path into a temp dir, then run all four stages."""
        from src import features as features_mod
        from src import figures as figures_mod
        from src import paths as paths_mod
        from src import roles as roles_mod
        from src import similarity as similarity_mod
        from src import value_model as value_mod

        root = tmp_path_factory.mktemp("pipeline")
        for sub in ("processed", "artifacts", "reports", "assets"):
                (root / sub).mkdir()

        redirect = {
                "PROCESSED": root / "processed", "ARTIFACTS": root / "artifacts",
                "REPORTS": root / "reports", "ASSETS": root / "assets",
                "PLAYER_SEASONS": root / "processed/player_seasons.parquet",
                "FEATURES": root / "processed/features.parquet",
                "PERCENTILES": root / "processed/percentiles.parquet",
                "SIMILARITY": root / "processed/similarity.parquet",
                "STYLE_MAP": root / "processed/style_map.parquet",
                "VALUE_PREDICTIONS": root / "processed/value_predictions.parquet",
                "ROLES": root / "processed/roles.parquet",
                "ROLE_SCORES": root / "processed/role_scores.parquet",
                "ROLE_STABILITY": root / "reports/role_stability.csv",
                "ROLE_TOP_PLAYERS": root / "reports/role_top_players.csv",
                "SIMILARITY_MODELS": root / "artifacts/similarity_models.joblib",
                "VALUE_MODEL": root / "artifacts/value_model.joblib",
                "FEATURE_RESOLUTION": root / "reports/feature_resolution.csv",
                "PCA_LOADINGS": root / "reports/pca_loadings.csv",
                "PCA_VARIANCE": root / "reports/pca_explained_variance.csv",
                "VALIDATION": root / "reports/similarity_validation.json",
                "VALUE_METRICS": root / "reports/value_model_metrics.json",
                "UNDERVALUED": root / "reports/undervalued_shortlist.csv",
        }
        saved = {key: getattr(paths_mod, key) for key in redirect}
        for key, value in redirect.items():
                setattr(paths_mod, key, value)
        original_ensure = paths_mod.ensure_dirs
        paths_mod.ensure_dirs = lambda: None

        try:
                synthetic_player_seasons.to_parquet(paths_mod.PLAYER_SEASONS, index=False)
                stages = [
                        ("features", lambda: features_mod.run(force=True)),
                        ("roles", lambda: roles_mod.run(force=True)),
                        ("similarity", lambda: similarity_mod.run(force=True)),
                        ("value", lambda: value_mod.run(force=True)),
                        ("figures", lambda: _run_figures(figures_mod)),
                ]
                UNFINISHED.clear()
                for name, stage in stages:
                        if UNFINISHED:
                                # A later stage reads the earlier one's output.
                                UNFINISHED[name] = f"waits on {next(iter(UNFINISHED))}"
                                continue
                        try:
                                stage()
                        except NotImplementedError as exc:
                                UNFINISHED[name] = str(exc)
                yield paths_mod
        finally:
                for key, value in saved.items():
                        setattr(paths_mod, key, value)
                paths_mod.ensure_dirs = original_ensure


# --- features --------------------------------------------------------------


def test_all_catalogue_features_resolve(pipeline):
        resolution = pd.read_csv(pipeline.FEATURE_RESOLUTION)
        missing = resolution[resolution["status"] == "MISSING"]["feature"].tolist()
        assert not missing, f"patterns in config.FEATURES no longer self-consistent: {missing}"


def test_features_are_finite_and_per90_scaled(pipeline):
        features = pd.read_parquet(pipeline.FEATURES)
        assert len(features) > 0

        per90 = [n for n, f in config.FEATURES.items() if f.per90 and n in features.columns]
        values = features[per90].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        assert finite.size > 0
        # Per-90 counting stats should land in a sane band, not season totals.
        assert finite.max() < 500, "per-90 scaling looks wrong - values too large"


def test_minutes_floor_applied(pipeline):
        from src.features import INCLUSION_MIN_MINUTES

        features = pd.read_parquet(pipeline.FEATURES)
        assert features["minutes"].min() >= INCLUSION_MIN_MINUTES


def test_percentiles_span_zero_to_one_hundred(pipeline):
        percentiles = pd.read_parquet(pipeline.PERCENTILES)
        numeric = percentiles.select_dtypes("number")
        assert numeric.min().min() >= 0
        assert numeric.max().max() <= 100


def test_every_group_has_percentiles(pipeline):
        percentiles = pd.read_parquet(pipeline.PERCENTILES)
        assert set(percentiles["position_group"]) == set(config.POSITION_GROUPS)


# --- roles -----------------------------------------------------------------


def test_every_player_season_gets_a_role_from_its_group(pipeline):
        needs("roles")
        features = pd.read_parquet(pipeline.FEATURES)
        roles = pd.read_parquet(pipeline.ROLES)
        assert set(roles["player_season_id"]) == set(features["player_season_id"])
        for group, rows in roles.groupby("position_group"):
                names = {role.name for role in config.ROLES[group]}
                assert set(rows["primary_role"]) <= names, group
        assert roles["primary_score"].between(0, 100).all()
        assert (roles["primary_score"] >= roles["secondary_score"]).all()


def test_role_reports_written(pipeline):
        needs("roles")
        stability = pd.read_csv(pipeline.ROLE_STABILITY)
        assert "All" in set(stability["position_group"])
        assert len(pd.read_csv(pipeline.ROLE_TOP_PLAYERS)) > 0


# --- similarity ------------------------------------------------------------


def test_similarity_models_cover_every_group(pipeline):
        import joblib

        models = joblib.load(pipeline.SIMILARITY_MODELS)
        assert set(models) == set(config.POSITION_GROUPS)
        for group, model in models.items():
                assert len(model["ids"]) == model["scores"].shape[0], group
                assert config.PCA_MIN_COMPONENTS <= model["pca"].n_components_ <= config.PCA_MAX_COMPONENTS


def test_neighbours_exclude_self_and_rank_from_one(pipeline):
        neighbours = pd.read_parquet(pipeline.SIMILARITY)
        assert not (neighbours["player_season_id"] == neighbours["comp_player_season_id"]).any()
        assert neighbours["rank"].min() == 1


def test_similarity_is_a_bounded_cosine(pipeline):
        neighbours = pd.read_parquet(pipeline.SIMILARITY)
        assert neighbours["similarity"].between(-1.0, 1.0001).all()


def test_comparables_stay_inside_their_position_group(pipeline):
        """A striker must never be returned as a comparable to a centre-back."""
        neighbours = pd.read_parquet(pipeline.SIMILARITY)
        features = pd.read_parquet(pipeline.FEATURES)
        groups = features.set_index("player_season_id")["position_group"]

        joined = neighbours.assign(comp_group=neighbours["comp_player_season_id"].map(groups))
        assert (joined["position_group"] == joined["comp_group"]).all()


def test_neighbour_ranks_follow_descending_similarity(pipeline):
        neighbours = pd.read_parquet(pipeline.SIMILARITY)
        sample = neighbours[neighbours["player_season_id"] == neighbours["player_season_id"].iloc[0]]
        ordered = sample.sort_values("rank")["similarity"].to_numpy()
        assert np.all(np.diff(ordered) <= 1e-9)


def test_loadings_and_style_map_written(pipeline):
        loadings = pd.read_csv(pipeline.PCA_LOADINGS)
        assert set(loadings["position_group"]) == set(config.POSITION_GROUPS)

        style = pd.read_parquet(pipeline.STYLE_MAP)
        assert style[["x", "y"]].notna().all().all()


# --- value model -----------------------------------------------------------


def test_value_predictions_are_positive_and_complete(pipeline):
        needs("value")
        predictions = pd.read_parquet(pipeline.VALUE_PREDICTIONS)
        assert len(predictions) > 0
        assert (predictions["predicted_value_eur"] > 0).all()
        assert predictions["log_residual"].notna().all()


def test_undervalued_flag_matches_its_threshold(pipeline):
        needs("value")
        predictions = pd.read_parquet(pipeline.VALUE_PREDICTIONS)
        expected = predictions["log_residual"] <= -config.UNDERVALUED_LOG_THRESHOLD
        assert (predictions["undervalued"] == expected).all()


def test_value_metrics_report_both_candidate_models(pipeline):
        import json

        needs("value")
        metrics = json.loads(pipeline.VALUE_METRICS.read_text(encoding="utf-8"))
        assert {"ridge", "gbm"} <= set(metrics)
        assert metrics["selected_model"] in {"ridge", "gbm"}
        for name in ("ridge", "gbm"):
                assert "cv_mae_log" in metrics[name]


# --- figures ---------------------------------------------------------------


def test_every_readme_figure_renders(pipeline):
        needs("figures")
        if PENDING_FIGURES:
                print(f"not implemented yet, not checked: {sorted(PENDING_FIGURES)}")
        assert not FAILED_FIGURES, "figure builders raised: " + "; ".join(
                f"{name} -> {error}" for name, error in FAILED_FIGURES.items())
        expected = {f for name, f in FIGURE_FILES.items() if name not in PENDING_FIGURES}
        produced = {p.name for p in pipeline.ASSETS.glob("*.png")}
        assert expected <= produced, f"missing figures: {sorted(expected - produced)}"
        for path in pipeline.ASSETS.glob("*.png"):
                assert path.stat().st_size > 5_000, f"{path.name} looks empty"
