"""Plotly charts for the app.

Percentiles, not z-scores: a radial axis fixed to 0-100 means two players
drawn on the same chart are genuinely comparable, and a non-technical viewer
reads "88th percentile" without being told what a standard deviation is.

Every chart has a transparent background and uses colours that read on both
of Streamlit's themes. PRIMARY and COMPARISON are shared with src/figures.py,
so the README and the app look like one product.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from . import config

PRIMARY = "#0F6B4A"
COMPARISON = "#A8481B"
GRID = "rgba(128,140,132,0.35)"
# Mid-grey, so the fair-value line shows on Streamlit's light AND dark
# backgrounds; a near-black line vanishes in dark mode.
DIAGONAL = "rgba(128,140,132,0.9)"


def ordinal(value: float) -> str:
        """88 -> "88th", 21 -> "21st", 12 -> "12th". Rounds to a whole number.

        Plotly's hovertemplate can format a number but cannot pick its suffix,
        so the hover text is built here and passed to each trace as `text`.
        """
        n = int(round(value))
        suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suffix}"


def axis_labels(feature_names: list[str]) -> list[str]:
        """Human-readable axis labels, in the order given.

        Uses config.FEATURES[name].display(); a name not in the catalogue
        falls back to title case ("touches_att_box" -> "Touches Att Box").
        """
        labels: list[str] = []
        for name in feature_names:
                feature = config.FEATURES.get(name)
                labels.append(feature.display() if feature is not None else name.replace("_", " ").title())
        return labels


def radar(percentiles: pd.Series,feature_names: list[str],name: str,
        comparison: pd.Series | None = None,
        comparison_name: str | None = None,height: int = 460) -> go.Figure:
        """One or two percentile profiles on a shared 0-100 polar axis.

        `percentiles` is a row from percentiles.parquet and `feature_names`
        picks and orders the axes (normally
        config.GROUP_SPECS[group].radar_axes()). Axes the player has no value
        for are dropped, and the comparison is drawn on the primary player's
        remaining axes only.

        The radial range is fixed at 0-100 so two charts are always
        comparable; auto-scaling would make them silently incomparable. Each
        polygon repeats its first point at the end, or it would leave a gap
        between the last axis and the first.

        The primary player is a solid line over a filled area. The comparison
        is a dotted outline with no fill, so the two never look alike and the
        primary reads first. Hovering shows "<axis>: 88th percentile".

        Returns an empty figure titled "No percentile profile available" if
        the player has no value on any axis.
        """
        labels = axis_labels(feature_names)
        values = [percentiles.get(feature, float("nan")) for feature in feature_names]

        usable = [(label, feature, float(value)) for label, feature, value in zip(labels, feature_names, values)
                  if pd.notna(value)]
        if not usable:
                return go.Figure().update_layout(
                        height=height,
                        title="No percentile profile available",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                )

        labels, feature_names_usable, values = zip(*usable)

        plot_labels = list(labels) + [labels[0]]
        plot_values = list(values) + [values[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
                r=plot_values,
                theta=plot_labels,
                fill="toself",
                fillcolor="rgba(15,107,74,0.18)",
                line=dict(color=PRIMARY, width=2.5),
                name=name,
                text=[ordinal(value) for value in plot_values],
                hovertemplate="%{theta}: %{text} percentile<extra>" + name + "</extra>",
        ))

        if comparison is not None:
                comparison_values = [comparison.get(feature, float("nan")) for feature in feature_names_usable]
                paired = [(label, value) for label, value in zip(labels, comparison_values) if pd.notna(value)]
                if paired:
                        comp_labels = [item[0] for item in paired] + [paired[0][0]]
                        comp_values = [float(item[1]) for item in paired] + [float(paired[0][1])]
                        # Dotted outline, no fill: it must never be mistaken for
                        # the primary player's solid, filled shape.
                        fig.add_trace(go.Scatterpolar(
                                r=comp_values,
                                theta=comp_labels,
                                mode="lines+markers",
                                fill="none",
                                line=dict(color=COMPARISON, width=2.5, dash="dot"),
                                marker=dict(color=COMPARISON, size=5),
                                name=comparison_name or "Comparison",
                                text=[ordinal(value) for value in comp_values],
                                hovertemplate="%{theta}: %{text} percentile<extra>"
                                               + (comparison_name or "Comparison") + "</extra>",
                        ))

        fig.update_layout(
                height=height,
                margin=dict(l=45, r=45, t=35, b=25),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                polar=dict(
                        radialaxis=dict(
                                visible=True,
                                range=[0, 100],
                                tickmode="linear",
                                tick0=0,
                                dtick=20,
                                gridcolor=GRID,
                                tickfont=dict(size=10),
                        ),
                        angularaxis=dict(
                                gridcolor=GRID,
                                linecolor=GRID,
                                tickfont=dict(size=11),
                        ),
                        bgcolor="rgba(0,0,0,0)",
                ),
        )
        return fig


def value_scatter(predictions: pd.DataFrame,highlight_id: str | None = None,height: int = 430) -> go.Figure:
        """Predicted against actual market value on log axes.

        The dashed diagonal is fair value. A player above it is valued below
        what the model expects, so cheaper than his profile suggests; nobody
        needs a legend to read that.

        `predictions` is the filtered pool, with market_value_eur and
        predicted_value_eur; `player` and `team` go in the hover text when
        present. Rows without a positive value on both axes are skipped. Both
        axes are log scale with equal scaling, so the diagonal sits at a true
        45 degrees.

        `highlight_id` is a player_season_id, drawn as a larger labelled
        marker. If that player has no prediction (market value under
        config.MIN_MARKET_VALUE_EUR) the chart is drawn without the marker -
        say so in the caller.

        Returns an empty figure titled "Value data unavailable" if either
        value column is missing.
        """
        frame = predictions.copy()
        required = {"market_value_eur", "predicted_value_eur"}
        missing = required - set(frame.columns)
        if missing:
                return go.Figure().update_layout(
                        height=height,
                        title="Value data unavailable",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                )

        frame["market_value_eur"] = pd.to_numeric(frame["market_value_eur"], errors="coerce")
        frame["predicted_value_eur"] = pd.to_numeric(frame["predicted_value_eur"], errors="coerce")
        frame = frame[
                frame["market_value_eur"].gt(0)
                & frame["predicted_value_eur"].gt(0)
        ].copy()

        fig = go.Figure()
        hover_columns = [column for column in ("player", "team") if column in frame.columns]
        if hover_columns:
                hover_text = (
                        "<b>%{customdata[0]}</b>"
                        + (" · %{customdata[1]}" if "team" in hover_columns else "")
                        + "<br>Actual: €%{x:.3s}<br>Model: €%{y:.3s}"
                        + "<extra></extra>"
                )
                customdata = frame[hover_columns].fillna("Unknown").to_numpy()
        else:
                hover_text = "Actual: €%{x:.3s}<br>Model: €%{y:.3s}<extra></extra>"
                customdata = None

        fig.add_trace(go.Scatter(
                x=frame["market_value_eur"],
                y=frame["predicted_value_eur"],
                mode="markers",
                customdata=customdata,
                hovertemplate=hover_text,
                marker=dict(color="rgba(94,107,100,0.48)", size=7),
                name="Player pool",
        ))

        if not frame.empty:
                low = min(frame["market_value_eur"].min(), frame["predicted_value_eur"].min())
                high = max(frame["market_value_eur"].max(), frame["predicted_value_eur"].max())
                low = max(low * 0.8, 1.0)
                high = high * 1.2
                fig.add_trace(go.Scatter(
                        x=[low, high],
                        y=[low, high],
                        mode="lines",
                        line=dict(color=DIAGONAL, width=1.5, dash="dash"),
                        name="Fair value",
                        hoverinfo="skip",
                ))

        if highlight_id is not None and "player_season_id" in frame.columns:
                highlighted = frame.loc[frame["player_season_id"] == highlight_id]
                if not highlighted.empty:
                        row = highlighted.iloc[0]
                        label = str(row["player"]) if "player" in row.index and pd.notna(row["player"]) else "Selected"
                        fig.add_trace(go.Scatter(
                                x=[row["market_value_eur"]],
                                y=[row["predicted_value_eur"]],
                                mode="markers+text",
                                text=[label],
                                textposition="top center",
                                marker=dict(color=PRIMARY, size=13, line=dict(color="white", width=1)),
                                name="Selected",
                                hovertemplate="<b>" + label + "</b><br>Actual: €%{x:.3s}"
                                              "<br>Model: €%{y:.3s}<extra></extra>",
                        ))

        fig.update_layout(
                height=height,
                margin=dict(l=55, r=20, t=20, b=55),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(
                        title="Actual market value (€, log scale)",
                        type="log",
                        gridcolor=GRID,
                        zeroline=False,
                ),
                yaxis=dict(
                        title="Predicted market value (€, log scale)",
                        type="log",
                        gridcolor=GRID,
                        zeroline=False,
                        scaleanchor="x",
                        scaleratio=1,
                ),
        )
        return fig
