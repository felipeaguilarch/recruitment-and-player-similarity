"""Plotly charts for the app.  [YOU IMPLEMENT THIS]

Percentiles, not z-scores: a radial axis fixed to 0-100 means two players
drawn on the same chart are genuinely comparable, and a non-technical viewer
reads "88th percentile" without being told what a standard deviation is.

The colours below are shared with src/figures.py so the README and the app
look like one product. Keep them.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import config

PRIMARY = "#0F6B4A"
COMPARISON = "#A8481B"
GRID = "rgba(128,140,132,0.35)"


def axis_labels(feature_names: list[str]) -> list[str]:
        """Human-readable axis labels for a list of feature names.

        config.FEATURES[name].display() already returns the label; fall back to
        a title-cased version of the raw name for anything not in the catalogue.
        """
        raise NotImplementedError("radar.axis_labels")


def radar(
        percentiles: pd.Series,
        feature_names: list[str],
        name: str,
        comparison: pd.Series | None = None,
        comparison_name: str | None = None,
        height: int = 460,
) -> go.Figure:
        """One or two percentile profiles on a shared 0-100 polar axis.

        `percentiles` is a row from percentiles.parquet; `feature_names` picks
        and orders the axes.

        TODO 1. go.Scatterpolar(r=..., theta=..., fill="toself"). Close the loop
                by appending the first r and first theta, or the polygon leaves
                a gap between the last and first axis.
        TODO 2. Fixed radial range [0, 100]. Letting Plotly auto-scale makes two
                charts silently incomparable, which defeats the whole point.
        TODO 3. The comparison trace should be visually subordinate - dashed
                line, lighter fill - so the primary player reads first.
        TODO 4. hovertemplate saying "<axis>: Nth percentile". The default hover
                shows raw numbers with no context.
        TODO 5. Transparent paper_bgcolor and plot_bgcolor so the chart sits on
                Streamlit's background in both light and dark themes.
        """
        raise NotImplementedError("radar.radar")


def value_scatter(
        predictions: pd.DataFrame,
        highlight_id: str | None = None,
        height: int = 430,
) -> go.Figure:
        """Predicted against actual market value on log axes.

        The diagonal is fair value; anything below it is cheaper than the model
        expects. Nobody needs a legend to read that.

        TODO 1. Grey markers for the filtered pool, with player and club in the
                hover text - an anonymous point cloud is not explorable.
        TODO 2. A dashed y = x line spanning the data range.
        TODO 3. If highlight_id is given, a larger accent-coloured marker with
                the player's name as a text label.
        TODO 4. type="log" on both axes.
        """
        raise NotImplementedError("radar.value_scatter")
