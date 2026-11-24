"""Stage 4 - PCA per position group, then cosine kNN.

Two design decisions here are worth defending in the README, because they are
where most "find similar players" projects go wrong.

**Cosine, not Euclidean.** On standardised PCA scores, cosine distance
compares the *shape* of a profile and largely ignores its magnitude - which is
what "plays like him" means. Euclidean blends style with level, so it returns
the same handful of elite players as comparables for everybody.

**Search in PCA space, plot in UMAP space.** UMAP produces a much nicer 2-D
scatter and is worth having in the app as a style map. It is the wrong tool
for the neighbour search: the embedding deliberately distorts global distances
to preserve local structure, so distances between distant points carry no
meaning and shift on every refit. The map is a picture, not the model.

Outputs:
  artifacts/similarity_models.joblib   fitted pipeline + index per group
  data/processed/similarity.parquet    top-N neighbours per player-season
  data/processed/style_map.parquet     2-D coordinates for the scatter
  reports/pca_loadings.csv             component loadings, for naming styles
  reports/similarity_validation.json   same-player self-consistency check

Spec: tests/test_pipeline_synthetic.py. The index-alignment tests
(test_comparables_stay_inside_their_position_group,
test_neighbours_exclude_self_and_rank_from_one) guard the classic mistake
here: kNN returns positional indices into the fitted matrix, not player ids.
"""

from __future__ import annotations

import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, features as features_mod, paths

logger = logging.getLogger(__name__)


def choose_n_components(scaled: np.ndarray, target: float) -> int:
        """Smallest component count reaching `target` explained variance.

        Picking a round number like "5 components" is arbitrary; picking by
        variance is defensible and lets each position group differ.

        Fits a probe PCA at the ceiling - the smallest of the sample count, the
        feature count and config.PCA_MAX_COMPONENTS - and returns the first
        count whose cumulative explained variance reaches `target`, clipped to
        [config.PCA_MIN_COMPONENTS, ceiling].

        On data with no real structure the target is never reached and the
        ceiling is returned. That is correct behaviour, not a bug - the
        synthetic tests show it.
        """
        ceiling = min(scaled.shape[0], scaled.shape[1], config.PCA_MAX_COMPONENTS)
        probe = PCA(n_components=ceiling).fit(scaled)
        variance = np.cumsum(probe.explained_variance_ratio_)
        n_components = np.searchsorted(variance, target) + 1
        return int(np.clip(n_components, config.PCA_MIN_COMPONENTS, ceiling))


def fit_group(frame: pd.DataFrame, feature_names: list[str]) -> dict:
        """Fit imputer -> scaler -> PCA -> kNN for one position group.

        Features that are entirely NaN are dropped first, with a warning.
        Remaining gaps are filled with the group median before scaling, so a
        player missing one stat stays in the model. The PCA keeps
        choose_n_components components, and the neighbour index asks for
        N_NEIGHBOURS + 1 because every point is its own nearest neighbour.

        `ids` and `players` are taken with .to_numpy(), so they are
        positionally aligned with `scores`: row i of scores is ids[i].
        Everything downstream depends on that invariant.

        Returns a dict with keys prep, pca, nn, feature_names, scores, ids,
        players, explained_variance and explained_variance_ratio.
        `feature_names` is the list after dropping empty features, so it
        matches the PCA's input columns.
        """
        # SimpleImputer silently DROPS columns that are entirely NaN, which would
        # leave feature_names longer than the matrix and mislabel every loading.
        empty = [f for f in feature_names if frame[f].isna().all()]
        if empty:
                logger.warning("Dropping all-NaN features: %s", empty)
                feature_names = [f for f in feature_names if f not in empty]

        prep = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())])

        scaled = prep.fit_transform(frame[feature_names])

        n_components = choose_n_components(scaled, config.PCA_VARIANCE_TARGET)
        pca = PCA(n_components=n_components)
        pca.fit(scaled)
        scores = pca.transform(scaled)

        nn = NearestNeighbors(
                n_neighbors=min(config.N_NEIGHBOURS + 1, len(frame)),
                metric=config.SIMILARITY_METRIC).fit(scores)

        return {
                "prep": prep,
                "pca": pca,
                "nn": nn,
                "feature_names": feature_names,
                "scores": scores,
                "ids": frame["player_season_id"].to_numpy(),
                "players": frame["player"].to_numpy(),
                "explained_variance": pca.explained_variance_,
                "explained_variance_ratio": pca.explained_variance_ratio_
        }


def neighbours_frame(models: dict[str, dict]) -> pd.DataFrame:
        """Top-N comparables per player-season, self excluded.

        Runs one kneighbors query per group over every row at once, skips each
        row's own index, and numbers the survivors from 1. Positional indices
        are mapped back through model["ids"] before storing - the step where
        index bugs hide. `similarity` is 1 - distance, because sklearn's
        "cosine" metric returns cosine DISTANCE.

        `same_player` is flagged rather than removed: the app hides those rows
        from the comparables list, and the notebook reads them. It
        compares player names, so two different players who share a name within
        one group are also flagged.

        Returns position_group, player_season_id, rank, comp_player_season_id,
        similarity and same_player.
        """

        rows = []

        for group, model in models.items():
                distances, indices = model["nn"].kneighbors(model["scores"])

                for i in range(len(model["ids"])):
                        rank = 0

                        for distance, neighbor_index in zip(distances[i], indices[i]):
                                neighbour_i = int(neighbor_index)

                                if neighbour_i == i:
                                        continue

                                rank += 1
                                rows.append({
                                        "position_group": group,
                                        "player_season_id": model["ids"][i],
                                        "rank": rank,
                                        "comp_player_season_id": model["ids"][neighbour_i],
                                        "similarity": 1 - float(distance),
                                        "same_player": (model["players"][i] == model["players"][neighbour_i])
                                })

        return pd.DataFrame(rows, columns=[
                "position_group",
                "player_season_id",
                "rank",
                "comp_player_season_id",
                "similarity",
                "same_player"
        ])


def loadings_frame(models: dict[str, dict]) -> pd.DataFrame:
        """PCA loadings, long format - the input for naming style dimensions.

        The first few components per group are meant to get a football name:
        if PC1 on wingers reads as "carries and shoots a lot" and PC2 as
        "crosses rather than cuts inside", those are dimensions a scout
        recognises.

        Returns position_group, component ("PC1"...), explained_variance_ratio,
        feature and loading. Rows are sorted by group, component number and
        loading descending, so the extremes of each component sit at the top
        and bottom of its block. Components sort numerically, so PC10 follows
        PC9 rather than PC1.
        """
        rows = []

        for group, model in models.items():
                for component_index, (weights, variance_ratio) in enumerate(
                        zip(model["pca"].components_, model["pca"].explained_variance_ratio_),
                        start=1):

                        for feature_name, loading in zip(model["feature_names"], weights):
                                rows.append({
                                        "position_group": group,
                                        "component": f"PC{component_index}",
                                        "explained_variance_ratio": variance_ratio,
                                        "feature": feature_name,
                                        "loading": loading
                                })

        columns = [
                "position_group",
                "component",
                "explained_variance_ratio",
                "feature",
                "loading",
        ]

        loadings = pd.DataFrame(rows, columns=columns)
        loadings["_component_index"] = loadings["component"].str[2:].astype(int)

        return (
                loadings.sort_values(
                        ["position_group", "_component_index", "loading"],
                        ascending=[True, True, False],
                )
                .drop(columns="_component_index")
                .reset_index(drop=True)
        )


def style_map(models: dict[str, dict]) -> pd.DataFrame:
        """2-D coordinates per player-season for the app's scatter.

        Runs UMAP on each group's PCA scores (n_neighbors capped at the group
        size minus one, min_dist 0.1, random_state 42). Without umap-learn
        installed it logs a warning and falls back to the first two PCA
        components, so the app still works.

        These coordinates are a picture, not the model: the neighbour search
        runs in PCA space, because UMAP distorts global distances.

        Returns player_season_id, position_group, x, y and method ("umap" or
        "pca"), so the app can label its axes honestly.
        """
        try:
                from umap import UMAP

        except ImportError:
                use_map = False
                logger.warning("umap-learn not installed, falling back to PCA for the style map")

        else:
                use_map = True

        rows = []

        for group, model in models.items():
                scores = model["scores"]

                if use_map:
                        coordinates = UMAP(
                                n_neighbors=min(15, len(scores) - 1),
                                min_dist=0.1,
                                n_components=2,
                                random_state=42,
                                n_jobs=1,          # a fixed seed forces one thread; say so, and skip the warning
                        ).fit_transform(scores)
                        method = "umap"

                else:
                        coordinates = scores[:, :2]
                        method = "pca"

                for i, (x,y) in zip(model["ids"], coordinates):
                        rows.append({
                                "player_season_id": i,
                                "position_group": group,
                                "x": x,
                                "y": y,
                                "method": method
                        })

        return pd.DataFrame(rows, columns=[
                "player_season_id",
                "position_group",
                "x",
                "y",
                "method"
        ])


def own_season_ranks(models: dict[str, dict], seasons: dict[str, str]) -> pd.DataFrame:
        """Exact rank of each player's own other season among everyone in his group.

        For every player-season whose player has a different season in the
        same group, ranks all other rows of the group by cosine similarity and
        records where the player's own other season lands (1 = closest). With
        several other seasons the best rank is kept, along with the season gap
        it came from. Computed from the PCA scores directly rather than read
        from the stored neighbour list, so a season ranked beyond
        config.N_NEIGHBOURS still counts - as a miss, not as missing.

        "Same player" means the same name within the group, exactly as
        neighbours_frame flags it, so two namesakes in one group are counted
        as one player.

        Returns player_season_id, position_group, rank, group_size and
        season_gap (in seasons).
        """
        rows = []
        for group, model in models.items():
                scores = np.asarray(model["scores"], dtype=float)
                unit = scores / np.linalg.norm(scores, axis=1, keepdims=True).clip(min=1e-12)
                similarity = unit @ unit.T
                np.fill_diagonal(similarity, -np.inf)          # a row is never its own neighbour

                players = pd.Series(model["players"])
                for player, positions in players.groupby(players).groups.items():
                        if len(positions) < 2:
                                continue
                        for i in positions:
                                # Only the player's OTHER seasons. Two rows in the same season
                                # are a mid-season transfer split across clubs, or namesakes.
                                own = seasons[model["ids"][i]]
                                others = [j for j in positions if seasons[model["ids"][j]] != own]
                                if not others:
                                        continue
                                # rank of j = 1 + number of rows strictly closer to i than j
                                ranks = [int((similarity[i] > similarity[i, j]).sum()) + 1 for j in others]
                                best = int(np.argmin(ranks))
                                gap = abs(
                                        int(str(seasons[model["ids"][i]])[2:])
                                        - int(str(seasons[model["ids"][others[best]]])[2:])
                                )
                                rows.append({
                                        "player_season_id": model["ids"][i],
                                        "position_group": group,
                                        "rank": ranks[best],
                                        "group_size": len(players),
                                        "season_gap": gap,
                                })
        return pd.DataFrame(
                rows, columns=["player_season_id", "position_group", "rank", "group_size", "season_gap"]
        )


def validate(models: dict[str, dict], seasons: dict[str, str]) -> dict:
        """Self-consistency: how highly do players rank their own other seasons?

        The one similarity metric here that is not a matter of taste. The same
        player a season later is the best "comparable" there is, so the closer
        he ranks to himself, the more the embedding captured the player rather
        than his team or the noise.

        Uses own_season_ranks, so every player with two seasons in a group is
        scored - including those whose other season falls outside the stored
        neighbour list, which an earlier version silently left out and so
        flattered the model.

        Reports, and writes to paths.VALIDATION:

        - median_rank / top_5_share: the raw rank and the share ranked 1-5.
        - median_rank_pct / top_5pct_share: the rank as a share of the group's
          size, which makes groups of 263 and 851 comparable. A random
          embedding would sit near 50%.
        - by_season_gap: the same, split by how many seasons apart the two
          rows are. A two-season gap is a harder test than an adjacent one.
        - player_seasons_scored.

        Raises ValueError if no player has two seasons in any group.
        """
        ranks = own_season_ranks(models, seasons)
        if ranks.empty:
                raise ValueError("No player has two seasons in the same group.")

        share = ranks["rank"] / ranks["group_size"]

        def summary(rank: pd.Series, pct: pd.Series) -> dict:
                return {
                        "median_rank": float(rank.median()),
                        "top_5_share": float((rank <= 5).mean()),
                        "median_rank_pct": float(pct.median()),
                        "top_5pct_share": float((pct <= 0.05).mean()),
                        "player_seasons_scored": int(len(rank)),
                }

        results = summary(ranks["rank"], share)
        results["by_season_gap"] = {
                str(gap): summary(ranks.loc[rows.index, "rank"], share.loc[rows.index])
                for gap, rows in ranks.groupby("season_gap")
        }

        paths.ensure_dirs()
        paths.VALIDATION.write_text(json.dumps(results, indent=2), encoding="utf-8")

        return results


def run(force: bool = False) -> dict[str, dict]:
        """Stage entry point, called by run_pipeline.py.

        Returns the cached models unless `force`. Otherwise reads
        paths.FEATURES and takes each group's features from
        features_mod.available_group_specs - NOT resolve_features, whose
        patterns describe raw FBref names. Groups below config.MIN_GROUP_SIZE
        are fitted with a warning; groups with fewer rows or features than
        config.PCA_MIN_COMPONENTS are skipped.

        Writes the models (paths.SIMILARITY_MODELS), neighbours
        (paths.SIMILARITY), loadings (paths.PCA_LOADINGS), per-group explained
        variance and component count (paths.PCA_VARIANCE) and the style map
        (paths.STYLE_MAP), then runs validate().

        Raises SystemExit if the features stage has not run, and ValueError if
        no group could be fitted.
        """
        paths.ensure_dirs()

        if paths.SIMILARITY_MODELS.exists() and not force:
                logger.info("Loading existing similarity models from %s", paths.SIMILARITY_MODELS)
                return joblib.load(paths.SIMILARITY_MODELS)

        try:
                frame = pd.read_parquet(paths.FEATURES)
        except FileNotFoundError as exc:
                raise SystemExit(f"Missing features file {paths.FEATURES}. Run stage 3 first.") from exc

        specs = features_mod.available_group_specs(list(frame.columns))
        models = {}

        for group, feature_names in specs.items():
                group_frame = frame.loc[frame["position_group"] == group].reset_index(drop=True)

                if len(group_frame) < config.MIN_GROUP_SIZE:
                        logger.warning("Group %s is below min size %d: only %d rows", group, config.MIN_GROUP_SIZE, len(group_frame))

                if (
                       len(group_frame) < config.PCA_MIN_COMPONENTS
                       or len(feature_names) < config.PCA_MIN_COMPONENTS
                ):
                        logger.warning("Skipping group %s: not enough rows or features to fit PCA.",group)
                        continue

                models[group] = fit_group(group_frame, feature_names)

        if not models:
                raise ValueError("No position groups were large enough to fit a model.")

        neighbours = neighbours_frame(models)
        loadings = loadings_frame(models)
        style = style_map(models)
        variance = pd.DataFrame(
                [
                        {
                                "position_group": group,
                                "explained_variance": float(
                                        model["explained_variance_ratio"].sum()
                                ),
                                "n_components": int(model["pca"].n_components_),
                        }
                        for group, model in models.items()
                ],
                columns=["position_group", "explained_variance", "n_components"],
        )

        joblib.dump(models, paths.SIMILARITY_MODELS)
        neighbours.to_parquet(paths.SIMILARITY, index=False)
        loadings.to_csv(paths.PCA_LOADINGS, index=False)
        variance.to_csv(paths.PCA_VARIANCE, index=False)
        style.to_parquet(paths.STYLE_MAP, index=False)
        validate(models, dict(zip(frame["player_season_id"], frame["season"].astype(str))))

        return models


def top_comps( player_season_id: str, neighbours: pd.DataFrame,
        n: int = 5, exclude_same_player: bool = True,) -> pd.DataFrame:
        """Convenience lookup used by the app and the validation notebook.

        Filters `neighbours` (the output of neighbours_frame) to one
        player-season and, unless `exclude_same_player` is False, drops the
        same player's other seasons. Returns the n rows with the smallest
        `rank`.

        `rank` keeps its original numbering, so after dropping a same-player
        row the ranks can skip (1, 2, 4, ...). An unknown id returns an empty
        frame rather than raising, and n=0 returns no rows.

        Raises ValueError if n is negative.
        """
        if n < 0:
                raise ValueError("n must be non-negative")
        filtered = neighbours.loc[neighbours["player_season_id"] == player_season_id]

        if exclude_same_player:
                filtered = filtered.loc[~filtered["same_player"]]

        return filtered.nsmallest(n, "rank")


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
