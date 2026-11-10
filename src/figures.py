"""Stage 6 - static figures for the README.  [YOU IMPLEMENT THE BUILDERS]

Deliberately matplotlib rather than Plotly. Exporting a Plotly figure to PNG
needs kaleido, which is a frequent install headache and one more thing that
can break a deploy; the interactive charts in the app stay Plotly. So the
README figures and the app are drawn by different code - a little duplication
for a pipeline that reproduces on a clean clone.

The rcParams block below is done for you; it is boilerplate, and having one
consistent look across every figure matters more than how it was set. The
five builders are yours.

General rules worth following in all of them:
  * Every axis label names a real unit. "EUR m", not "value".
  * Annotate the thing you want read - the peak of the age curve, the
    diagonal on the value scatter - rather than leaving the reader to infer.
  * Figures go in the README at roughly 600-700px wide, so keep type legible
    at that size and resist cramming.
"""

from __future__ import annotations

import argparse
import logging

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # no display on CI or Streamlit Cloud
import matplotlib.pyplot as plt  # noqa: E402

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


def _save(fig, name: str) -> None:
        """Write a figure into assets/ and close it. Done for you."""
        paths.ASSETS.mkdir(parents=True, exist_ok=True)
        target = paths.ASSETS / name
        fig.savefig(target)
        plt.close(fig)
        logger.info("Wrote %s", target)


def radar_figure(player_season_id: str | None = None) -> None:
        """Percentile radar for one player - the README's hero image.

        Must write assets/radar_example.png.

        TODO 1. Read paths.PERCENTILES and paths.FEATURES. With no argument,
                pick a sensible default - highest-minutes player in the largest
                group - so the figure is of somebody a reader might recognise.
        TODO 2. Axes come from config.GROUP_SPECS[group].radar_axes(); drop any
                that are NaN for this player.
        TODO 3. Polar subplot: angles = np.linspace(0, 2*pi, n, endpoint=False).
                Close the polygon by appending the first value and first angle,
                or you get a gap between the last and first axis.
        TODO 4. set_ylim(0, 100) - percentiles are the whole point, and an
                auto-scaled radial axis makes two players incomparable.
        TODO 5. Title with player, club, season and what the percentiles are
                against ("vs CBs in the sample"). A radar with no reference
                population is meaningless.
        """
        raise NotImplementedError("figures.radar_figure")


def pca_variance_figure() -> None:
        """How many style dimensions each position group needs.

        Must write assets/pca_variance.png. Reads paths.PCA_VARIANCE.

        TODO 1. Bar per position group, height = explained_variance * 100.
        TODO 2. Annotate each bar with its component count - the interesting
                fact is "CB needs 6 PCs to reach 85%", not the bar height alone.
        TODO 3. axhline at config.PCA_VARIANCE_TARGET * 100, labelled.
        """
        raise NotImplementedError("figures.pca_variance_figure")


def age_curve_figure() -> None:
        """The ageing curve, recovered from data rather than asserted.

        A strong README figure precisely because nobody told the model that value
        peaks in the mid-twenties - it fell out of the fit.

        Must write assets/age_curve.png. Reads paths.VALUE_PREDICTIONS.

        TODO 1. Bin age into single years (pd.cut) over a sane range, ~16-40.
        TODO 2. Median predicted value per bin, plus the interquartile range as
                a fill_between band. A median line alone overstates certainty.
        TODO 3. Mark the peak with an axvline and label the age.
        TODO 4. Plot in EUR millions, not raw euros.
        """
        raise NotImplementedError("figures.age_curve_figure")


def value_scatter_figure() -> None:
        """Predicted against actual, with the undervalued side distinguished.

        Must write assets/value_scatter.png. Reads paths.VALUE_PREDICTIONS.

        TODO 1. Log both axes - values are log-normal and a linear scale wastes
                90% of the plot on a handful of superstars.
        TODO 2. Two scatters: flagged vs not, so the reader sees which side of
                the line matters.
        TODO 3. Draw the y = x diagonal. With it, nobody needs a legend to read
                "below the line is cheap".
        TODO 4. Equal x and y limits, or the diagonal lies.
        """
        raise NotImplementedError("figures.value_scatter_figure")


def style_map_figure(group: str | None = None) -> None:
        """The 2-D style map - presentation only, never the neighbour search.

        Must write assets/style_map.png. Reads paths.STYLE_MAP.

        TODO 1. Scatter one position group's coordinates.
        TODO 2. Annotate a handful of recognisable players (highest minutes) -
                an unlabelled cloud of dots says nothing.
        TODO 3. Read the `method` column and put UMAP or PCA in the axis labels
                and title, plus the caveat that comps are found in PCA space.
                A reader who assumes the picture is the model will misread it.
        """
        raise NotImplementedError("figures.style_map_figure")


FIGURES = {
        "radar": radar_figure,
        "pca_variance": pca_variance_figure,
        "age_curve": age_curve_figure,
        "value_scatter": value_scatter_figure,
        "style_map": style_map_figure,
}


def run(force: bool = False, only: str | None = None) -> None:  # noqa: ARG001
        """Build every figure, or just one.

        Each builder is wrapped so that one failure does not sink the rest -
        useful while you are implementing them one at a time.
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
