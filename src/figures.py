"""Stage 6 - static figures for the README.

Deliberately matplotlib rather than Plotly. Exporting a Plotly figure to PNG
needs kaleido, which is a frequent install headache and one more thing that
can break a deploy; the interactive charts in the app stay Plotly. So the
README figures and the app are drawn by different code - a little duplication
for a pipeline that reproduces on a clean clone.

Conventions shared by every figure:
  * Every axis label names a real unit (EUR m, %, years).
  * The thing to be read is annotated - the peak of the age curve, the
    fair-value diagonal - rather than left for the reader to infer.
  * Figures sit in the README at roughly 600-700px wide, so type is sized to
    stay legible there.
"""

from __future__ import annotations

import argparse
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

matplotlib.use("Agg")  # no display on CI or Streamlit Cloud
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config, paths  # noqa: E402

logger = logging.getLogger(__name__)

INK = "#111614"
MUTED = "#5E6B64"
ACCENT = "#0F6B4A"
FLAG = "#A8481B"
GRID = "#D8DED8"

plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
})


#: One colour per position group, for figures that draw every group at once.
GROUP_COLOURS = {
        "CB": "#0F6B4A", "FB": "#A8481B", "DM": "#3B6EA8", "CM": "#8A6BBE",
        "AM": "#C29A1B", "W": "#5E6B64", "ST": "#C2417A",
}

#: Age bins with fewer player-seasons than this are left off the age curve;
#: a median and IQR over a handful of players is noise, not a curve.
MIN_PLAYERS_PER_AGE = 10

#: Players labelled on the style map, chosen by market value.
STYLE_MAP_LABELS = 8


def _save(fig, name: str) -> None:
        """Write a figure into assets/ and close it."""
        paths.ASSETS.mkdir(parents=True, exist_ok=True)
        target = paths.ASSETS / name
        fig.savefig(target)
        plt.close(fig)
        logger.info("Wrote %s", target)


def _season_label(season) -> str:
        """"2223" -> "2022-23"; anything else is returned unchanged."""
        season = str(season)
        return f"20{season[:2]}-{season[2:]}" if len(season) == 4 and season.isdigit() else season


def _eur_m(value: float, _position=None) -> str:
        """Tick label for a value already in EUR millions: 0.1 -> "€0.1m"."""
        return f"€{value:g}m"


def radar_figure(player_season_id: str | None = None) -> None:
        """Percentile radar for one player - the README's hero image.

        Writes assets/radar_example.png. With no argument the player is the
        highest-minutes player-season in the largest position group (ties
        broken by id, so the default never changes between runs).

        Axes are the group's radar axes from config.GROUP_SPECS, minus any the
        player has no value for. The radial axis is fixed at 0-100 so two
        radars are always comparable, and the title names the reference
        population: percentiles are ranked within position group and season.
        """
        # Percentile ranks explain how the player compares within his
        # position-and-season peer group. Identity and minutes live in FEATURES,
        # so read that table together with PERCENTILES before choosing a default
        # player or looking up an explicitly requested player-season.
        percentiles = pd.read_parquet(paths.PERCENTILES)
        features = pd.read_parquet(paths.FEATURES)

        if player_season_id is None:
                # "Largest group" means the most data-rich peer population.
                # Within that group, the highest-minutes player is the most stable
                # default example rather than a noisy, short-sample outlier.
                largest_group = (
                        features["position_group"]
                        .value_counts()
                        .sort_index()
                        .idxmax()
                )
                candidates = features.loc[
                        features["position_group"] == largest_group
                ].copy()
                if candidates.empty:
                        raise ValueError("No player is available for the default radar")
                # Several players can share the maximum minutes (3,420 = 38 full
                # games); the id tie-break keeps the default stable across runs.
                default_row = candidates.sort_values(
                        ["minutes", "player_season_id"],
                        ascending=[False, True],
                        na_position="last",
                ).iloc[0]
                player_season_id = str(default_row["player_season_id"])

        # Joining on the explicit player_season_id avoids accidental attribute
        # swaps. Prefer the richer identity row from FEATURES, but keep fallbacks
        # for older or partial exports.
        identity_rows = features.loc[
                features["player_season_id"].astype(str) == str(player_season_id)
        ]
        if identity_rows.empty:
                identity_rows = percentiles.loc[
                        percentiles["player_season_id"].astype(str) == str(player_season_id)
                ]
        if identity_rows.empty:
                raise ValueError(f"No player-season found for radar id {player_season_id}")

        identity = identity_rows.iloc[0]
        percentile_rows = percentiles.loc[
                percentiles["player_season_id"].astype(str) == str(player_season_id)
        ]
        if percentile_rows.empty:
                raise ValueError(f"No percentile row found for radar id {player_season_id}")
        profile = percentile_rows.iloc[0]

        group = str(identity["position_group"])
        spec = config.GROUP_SPECS.get(group)
        if spec is None:
                raise ValueError(f"No radar configuration exists for position group {group}")

        # Build the usable axis list once. Each tuple contains a short display
        # label, a source-column name and the player's percentile. Axes missing
        # for this player are removed, as a radar should not imply a known 0.
        usable = []
        for feature_name in spec.radar_axes():
                value = profile.get(feature_name, np.nan)
                if pd.notna(value):
                        feature = config.FEATURES.get(feature_name)
                        label = feature.display() if feature is not None else feature_name.replace("_", " ").title()
                        usable.append((label, float(value)))

        if not usable:
                raise ValueError(f"Player {player_season_id} has no usable radar values")

        labels = [item[0] for item in usable]
        values = [item[1] for item in usable]
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

        # Repeating the first value and angle closes the polygon. Without these
        # duplicate endpoints matplotlib leaves a visual gap at 12 o'clock.
        plot_angles = np.concatenate([angles, angles[:1]])
        plot_values = np.asarray(values + [values[0]], dtype=float)

        fig, ax = plt.subplots(figsize=(7.2, 6.4), subplot_kw={"projection": "polar"})
        ax.plot(plot_angles, plot_values, color=ACCENT, linewidth=2.2)
        ax.fill(plot_angles, plot_values, color=ACCENT, alpha=0.18)
        ax.scatter(plot_angles[:-1], plot_values[:-1], color=ACCENT, s=18, zorder=3)

        ax.set_thetagrids(np.degrees(angles), labels)
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.spines["polar"].set_color(GRID)

        player_name = str(identity.get("player", player_season_id))
        club = str(identity.get("team", ""))
        season = _season_label(identity.get("season", ""))
        club_suffix = f", {club}" if club else ""
        season_suffix = f", {season}" if season else ""
        # features.py ranks percentiles within position group AND season, so
        # the reference population is this season's players in the group.
        population = f"{season} {group}s" if season else f"{group}s"
        ax.set_title(
                f"{player_name}{club_suffix}{season_suffix}\n"
                f"Percentile rank among {population} in the sample",
                pad=24,
        )
        fig.tight_layout()
        _save(fig, "radar_example.png")


def pca_variance_figure() -> None:
        """How many style dimensions each position group needs, and how fast
        its playing style compresses.

        Writes assets/pca_variance.png. For every group a full PCA is refitted
        on the exact matrix the similarity model was fitted on - the stored
        imputer and scaler (paths.SIMILARITY_MODELS) applied to that group's
        rows of paths.FEATURES - and its cumulative explained-variance curve
        is drawn. A dot marks the components the model actually keeps: the
        first count to reach config.PCA_VARIANCE_TARGET.

        A bar per group would sit at 85-87% for every group by construction;
        the curves show what differs - CM's style is captured in 5
        dimensions, CB's and DM's need 9.
        """
        models = joblib.load(paths.SIMILARITY_MODELS)
        features = pd.read_parquet(paths.FEATURES).set_index("player_season_id")

        # Football order first, then any group the config does not list.
        groups = [group for group in config.POSITION_GROUPS if group in models]
        groups += [group for group in models if group not in groups]
        if not groups:
                raise ValueError("No fitted similarity models to draw")

        target = config.PCA_VARIANCE_TARGET * 100
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        longest, lowest = 1, 100.0
        for group in groups:
                model = models[group]
                # Same median-impute + standardise as the fitted model, so the
                # curve is the space the neighbour search actually runs in.
                matrix = model["prep"].transform(
                        features.loc[model["ids"], model["feature_names"]]
                )
                curve = np.cumsum(PCA().fit(matrix).explained_variance_ratio_) * 100
                kept = int(model["pca"].n_components_)
                colour = GROUP_COLOURS.get(group, MUTED)
                ax.plot(np.arange(1, len(curve) + 1), curve, color=colour,
                        linewidth=1.8, label=f"{group} ({kept} kept)")
                ax.scatter([kept], [curve[kept - 1]], color=colour, s=30, zorder=3,
                           edgecolors="white", linewidths=0.7)
                longest = max(longest, len(curve))
                lowest = min(lowest, float(curve[0]))

        ax.axhline(target, color=MUTED, linestyle="--", linewidth=1.0)
        ax.text(longest, target - 1.5, f"{target:.0f}% variance target",
                ha="right", va="top", fontsize=8, color=MUTED)

        ax.set_xlim(1, longest)
        ax.set_xticks(np.arange(2, longest + 1, 2))
        ax.set_ylim(max(0, np.floor(lowest / 10) * 10), 101)
        ax.set_xlabel("Number of PCA components (style dimensions)")
        ax.set_ylabel("Cumulative explained variance (%)")
        ax.set_title("Style dimensions needed per position group\n"
                     "Dot = components kept by the similarity model")
        ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
        fig.tight_layout()
        _save(fig, "pca_variance.png")


def age_curve_figure() -> None:
        """The ageing curve, recovered from data rather than asserted.

        A strong README figure precisely because nobody told the model that value
        peaks in the mid-twenties - it fell out of the fit.

        Writes assets/age_curve.png from paths.VALUE_PREDICTIONS: median
        predicted value per whole year of age, with the interquartile range as
        a band (a median line alone overstates certainty) and the peak marked.
        Ages with fewer than MIN_PLAYERS_PER_AGE player-seasons are dropped.
        """
        predictions = pd.read_parquet(paths.VALUE_PREDICTIONS)
        required = {"age", "predicted_value_eur"}
        missing = required.difference(predictions.columns)
        if missing:
                raise ValueError(f"Value predictions are missing columns: {sorted(missing)}")

        frame = predictions.copy()
        frame["age"] = pd.to_numeric(frame["age"], errors="coerce")
        frame["predicted_value_eur"] = pd.to_numeric(
                frame["predicted_value_eur"], errors="coerce"
        )
        frame = frame.loc[
                frame["age"].between(16, 40)
                & frame["predicted_value_eur"].notna()
                & (frame["predicted_value_eur"] >= 0)
        ].copy()
        if frame.empty:
                raise ValueError("No in-range value predictions are available for the age curve")

        # Single-year bins: [25, 26) holds every player aged 25, and is
        # plotted at x = 25 so the axis reads as "age in completed years".
        bins = np.arange(16, 41)
        frame["age_bin"] = pd.cut(
                frame["age"],
                bins=bins,
                right=False,
                include_lowest=True,
        )
        summary = (
                frame.groupby("age_bin", observed=True)["predicted_value_eur"]
                .agg(
                        median="median",
                        lower=lambda values: values.quantile(0.25),
                        upper=lambda values: values.quantile(0.75),
                        players="size",
                )
                .dropna(subset=["median"])
        )
        # The extremes (age 16, 38+) hold one to seven player-seasons each.
        summary = summary.loc[summary["players"] >= MIN_PLAYERS_PER_AGE]
        if summary.empty:
                raise ValueError("No complete age bins are available")

        ages = np.asarray([interval.left for interval in summary.index], dtype=float)
        median_eur_m = summary["median"].to_numpy(dtype=float) / 1_000_000
        lower_eur_m = summary["lower"].to_numpy(dtype=float) / 1_000_000
        upper_eur_m = summary["upper"].to_numpy(dtype=float) / 1_000_000

        peak_index = int(np.argmax(median_eur_m))
        peak_age = ages[peak_index]
        peak_value = median_eur_m[peak_index]

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.fill_between(
                ages,
                lower_eur_m,
                upper_eur_m,
                color=ACCENT,
                alpha=0.16,
                label="25th-75th percentile",
        )
        ax.plot(ages, median_eur_m, color=ACCENT, linewidth=2.4,
                label="Median predicted value")
        ax.axvline(peak_age, color=FLAG, linestyle="--", linewidth=1.2)
        ax.annotate(
                f"Peak at {peak_age:.0f}: EUR {peak_value:.1f}m",
                xy=(peak_age, peak_value),
                xytext=(12, 14),
                textcoords="offset points",
                color=FLAG,
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": FLAG},
        )

        ax.set_xlim(ages.min() - 0.5, ages.max() + 0.5)
        ax.set_xticks(np.arange(int(ages.min()) + int(ages.min()) % 2, int(ages.max()) + 1, 2))
        ax.set_xlabel("Age (completed years)")
        ax.set_ylabel("Predicted market value (EUR m)")
        ax.set_title("Predicted player value by age")
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        _save(fig, "age_curve.png")


def value_scatter_figure() -> None:
        """Predicted against actual, with the undervalued side distinguished.

        Writes assets/value_scatter.png from paths.VALUE_PREDICTIONS. Both
        axes are log scale in EUR m (values are roughly log-normal; a linear
        scale spends most of the plot on a few superstars) with identical
        limits, so the y = x fair-value diagonal is a true 45-degree line.
        Undervalued players - predicted well above actual - sit above it, past
        a second, dotted line marking the flag threshold.
        """
        predictions = pd.read_parquet(paths.VALUE_PREDICTIONS)
        required = {"market_value_eur", "predicted_value_eur", "undervalued"}
        missing = required.difference(predictions.columns)
        if missing:
                raise ValueError(f"Value predictions are missing columns: {sorted(missing)}")

        frame = predictions.copy()
        for column in ("market_value_eur", "predicted_value_eur"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.loc[
                frame["market_value_eur"].gt(0)
                & frame["predicted_value_eur"].gt(0)
                & frame["market_value_eur"].notna()
                & frame["predicted_value_eur"].notna()
        ].copy()
        if frame.empty:
                raise ValueError("No positive value predictions are available")

        actual = frame["market_value_eur"].to_numpy(dtype=float) / 1_000_000
        predicted = frame["predicted_value_eur"].to_numpy(dtype=float) / 1_000_000
        flagged = frame["undervalued"].fillna(False).astype(bool).to_numpy()

        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        ax.scatter(
                actual[~flagged],
                predicted[~flagged],
                s=18,
                color=ACCENT,
                alpha=0.58,
                edgecolors="none",
                label="Not flagged",
        )
        ax.scatter(
                actual[flagged],
                predicted[flagged],
                s=22,
                color=FLAG,
                alpha=0.82,
                edgecolors="none",
                label="Undervalued",
        )

        # ~12% padding on the log scale so points at the extremes are not
        # cut in half by the axes.
        lower = float(min(actual.min(), predicted.min())) / 1.12
        upper = float(max(actual.max(), predicted.max())) * 1.12
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.xaxis.set_major_formatter(FuncFormatter(_eur_m))
        ax.yaxis.set_major_formatter(FuncFormatter(_eur_m))

        # y = x is the fair-value line. x is the actual value and y the
        # prediction, so a player ABOVE it is valued by the model at more
        # than his market price - the undervalued side.
        diagonal = np.array([lower, upper])
        ax.plot(diagonal, diagonal, color=INK, linewidth=1.2,
                linestyle="--", label="Fair value: actual = predicted")

        # The flag threshold: the model's target is log1p(EUR), and a player
        # is flagged when the prediction beats it by UNDERVALUED_LOG_THRESHOLD,
        # i.e. predicted ~= e^0.5 = 1.65x actual. Drawing it shows the orange
        # region starts here, not at the diagonal.
        ratio = float(np.exp(config.UNDERVALUED_LOG_THRESHOLD))
        ax.plot(diagonal, diagonal * ratio, color=FLAG, linewidth=1.0,
                linestyle=":", label=f"Flag line: predicted = {ratio:.2f}x actual")

        ax.set_xlabel("Actual market value (EUR m, log scale)")
        ax.set_ylabel("Predicted market value (EUR m, log scale)")
        ax.set_title("Predicted value vs actual market value")
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        _save(fig, "value_scatter.png")


def style_map_figure(group: str | None = None) -> None:
        """The 2-D style map - presentation only, never the neighbour search.

        Writes assets/style_map.png from paths.STYLE_MAP: one position group's
        2-D coordinates (the largest group by default), with the
        STYLE_MAP_LABELS most valuable players labelled, one season each.
        The method (UMAP or PCA) is named in the title and
        axes, with the caveat that comparables are found in PCA space - the
        picture is a reading aid, not the model.
        """
        style = pd.read_parquet(paths.STYLE_MAP)
        features = pd.read_parquet(paths.FEATURES)
        required = {"player_season_id", "position_group", "x", "y", "method"}
        missing = required.difference(style.columns)
        if missing:
                raise ValueError(f"Style map is missing columns: {sorted(missing)}")

        if group is None:
                counts = style["position_group"].value_counts()
                if counts.empty:
                        raise ValueError("Style map contains no position groups")
                group = str(counts.idxmax())
        group = str(group)
        frame = style.loc[
                style["position_group"].astype(str) == group
        ].copy()
        if frame.empty:
                raise ValueError(f"No style-map rows found for position group {group}")

        frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        frame = frame.loc[frame["x"].notna() & frame["y"].notna()]
        if frame.empty:
                raise ValueError(f"No finite coordinates found for position group {group}")

        method = str(frame["method"].mode().iloc[0] if not frame["method"].mode().empty
                     else frame["method"].iloc[0]).upper()
        identity = features.set_index("player_season_id")
        frame = frame.join(
                identity[["player", "season", "minutes", "market_value_eur"]], on="player_season_id"
        )
        frame["market_value_eur"] = pd.to_numeric(frame["market_value_eur"], errors="coerce")
        frame = frame.loc[frame["player"].notna()]

        # Label the most valuable players - the names a reader recognises -
        # once each (his most valuable season), with the season shown since
        # every player can appear up to three times. Id tie-break keeps the
        # labelled set identical between runs.
        picks = (
                frame.sort_values(
                        ["market_value_eur", "player_season_id"],
                        ascending=[False, True], na_position="last",
                )
                .drop_duplicates("player")
                .head(STYLE_MAP_LABELS)
        )

        fig, ax = plt.subplots(figsize=(7.2, 5.8))
        ax.scatter(frame["x"], frame["y"], s=18, color=ACCENT, alpha=0.5, edgecolors="none")
        ax.scatter(picks["x"], picks["y"], s=34, color=FLAG, zorder=3,
                   edgecolors="white", linewidths=0.6)

        # The axes of a UMAP (or 2-component PCA) projection have no units,
        # so the tick numbers are hidden; only relative position means anything.
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.set_xlabel(f"{method} dimension 1")
        ax.set_ylabel(f"{method} dimension 2")
        ax.set_title(
                f"{group} style map ({method}), most valuable players labelled\n"
                "Presentation coordinates; comparables are found in PCA space"
        )
        ax.grid(True, color=GRID, linewidth=0.6)
        fig.tight_layout()

        # Place each label at the first of four corners around its point that
        # does not overlap a label already placed (measured in rendered pixels).
        renderer = fig.canvas.get_renderer()
        corners = [((5, 4), "left", "bottom"), ((5, -4), "left", "top"),
                   ((-5, 4), "right", "bottom"), ((-5, -4), "right", "top")]
        placed = []
        for row in picks.itertuples():
                text = f"{row.player} ({_season_label(row.season)})"
                for attempt, ((dx, dy), ha, va) in enumerate(corners):
                        label = ax.annotate(text, (row.x, row.y), xytext=(dx, dy),
                                            textcoords="offset points", ha=ha, va=va,
                                            fontsize=7.5, color=INK)
                        box = label.get_window_extent(renderer).expanded(1.04, 1.2)
                        last = attempt == len(corners) - 1
                        if last or not any(box.overlaps(other) for other in placed):
                                placed.append(box)
                                break
                        label.remove()

        _save(fig, "style_map.png")


FIGURES = {
        "radar": radar_figure,
        "pca_variance": pca_variance_figure,
        "age_curve": age_curve_figure,
        "value_scatter": value_scatter_figure,
        "style_map": style_map_figure,
}


def run(force: bool = False, only: str | None = None) -> None:  # noqa: ARG001
        """Build every figure, or just one.

        Each builder is wrapped so that one failing figure is logged and the
        rest are still written. Figures are cheap, so `force` is accepted for
        the stage interface but they are always rebuilt.
        """
        paths.ensure_dirs()
        targets = {only: FIGURES[only]} if only else FIGURES
        for name, builder in targets.items():
                try:
                        builder()
                except NotImplementedError:
                        logger.warning("Figure %r not implemented yet", name)
                except Exception as exc:  # noqa: BLE001
                        logger.error("Figure %r failed: %s", name, exc)


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--only", choices=sorted(FIGURES), help="build just one figure")
        args = parser.parse_args()
        run(only=args.only)
