"""Recruitment and player-similarity dashboard - Streamlit entry point.

This app is read-only: it loads committed parquet tables and fitted model
artifacts, but never scrapes, retrains or runs the pipeline. The sidebar
controls one shared candidate pool, then the page combines a selected player's
header, percentile radar, nearest comparables, value chart and style map.
"""

from __future__ import annotations

import streamlit as st

# Must be the first Streamlit call in the script.
st.set_page_config(
        page_title="Player Recruitment Tool",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src import config, paths, radar  # noqa: E402

EUR = "€"


# ---------------------------------------------------------------------------
# Loading - done for you
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading player data…")
def load_table(path_str: str) -> pd.DataFrame:
        """st.cache_data returns a COPY per session - correct for data frames,
        because a user mutating one must not affect anybody else."""
        return pd.read_parquet(path_str)


@st.cache_resource(show_spinner="Loading models…")
def load_models(path_str: str):
        """st.cache_resource shares ONE object across all sessions - correct for
        a fitted model, which is read-only and expensive to unpickle."""
        import joblib

        return joblib.load(path_str)


def require_pipeline() -> None:
        """Fail with instructions rather than a stack trace on a fresh clone."""
        needed = {
                "features": paths.FEATURES,
                "percentiles": paths.PERCENTILES,
                "comparables": paths.SIMILARITY,
                "value predictions": paths.VALUE_PREDICTIONS,
        }
        missing = {name: path for name, path in needed.items() if not path.exists()}
        if not missing:
                return

        st.error("The app's data files have not been built yet.")
        st.markdown(
                "Run the pipeline once, then restart the app:\n\n"
                "```bash\npython run_pipeline.py --stage all\n```\n\n"
                "Missing: " + ", ".join(f"`{p.name}`" for p in missing.values())
        )
        st.stop()


require_pipeline()

features = load_table(str(paths.FEATURES))
percentiles = load_table(str(paths.PERCENTILES))
comps = load_table(str(paths.SIMILARITY))
values = load_table(str(paths.VALUE_PREDICTIONS))
style = load_table(str(paths.STYLE_MAP)) if paths.STYLE_MAP.exists() else None

# One frame the whole app reads from. Merging value predictions onto features
# here means no section has to think about where a column came from.
players = features.merge(
        values[["player_season_id", "predicted_value_eur", "log_residual",
                        "value_ratio", "undervalued"]],
        on="player_season_id",
        how="left",
)


def euros(value: float | None) -> str:
        """Format a value the way a recruitment person writes it: €8.5m, €750k."""
        if value is None or pd.isna(value):
                return "—"
        if value >= 1e6:
                return f"{EUR}{value / 1e6:.1f}m"
        return f"{EUR}{value / 1e3:.0f}k"


def season_label(season: str) -> str:
        """"2425" -> "2024-25"."""
        season = str(season)
        return f"20{season[:2]}-{season[2:]}" if len(season) == 4 and season.isdigit() else season


# ---------------------------------------------------------------------------
# 1. Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.title("Filters")

available_groups = [
        group for group in config.POSITION_GROUPS
        if group in set(players["position_group"].dropna())
]
group = st.sidebar.selectbox("Position group", available_groups)

available_seasons = sorted(
        players["season"].dropna().astype(str).unique(),
        reverse=True,
)
season = st.sidebar.selectbox(
        "Season",
        available_seasons,
        format_func=season_label,
        help="The season of the player you search for.",
)

# Similarity neighbours come from every season, and over half of a player's
# top-5 comparables are from a different one - so restricting candidates to
# the searched season would throw most of them away (2023-24 is Premier
# League only and would keep about two).
candidate_seasons = st.sidebar.multiselect(
        "Candidate seasons",
        available_seasons,
        default=available_seasons,
        format_func=season_label,
        help="Seasons the comparables and the value chart may come from.",
)

available_leagues = sorted(players["league"].dropna().unique())
selected_leagues = st.sidebar.multiselect(
        "Leagues",
        available_leagues,
        default=available_leagues,
)

# Numeric filters use finite values so NaN ages or values cannot break a slider.
ages = pd.to_numeric(players["age"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
if ages.empty:
        age_minimum, age_maximum = 16.0, 40.0
else:
        age_minimum = max(16.0, float(np.floor(ages.min())))
        age_maximum = min(40.0, float(np.ceil(ages.max())))
age_range = st.sidebar.slider(
        "Age range",
        min_value=age_minimum,
        max_value=age_maximum,
        value=(age_minimum, age_maximum),
        step=1.0,
)

# The data holds nobody under MIN_MINUTES_SENSITIVITY, so the slider starts
# there rather than at a range that filters nothing.
minutes = pd.to_numeric(players["minutes"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
minutes_floor = int(config.MIN_MINUTES_SENSITIVITY)
minutes_max = int(np.ceil(minutes.max())) if not minutes.empty else 5000
minimum_minutes = st.sidebar.slider(
        "Minimum minutes",
        min_value=minutes_floor,
        max_value=max(minutes_max, minutes_floor + 50),
        value=int(config.MIN_MINUTES),
        step=50,
)

# In EUR millions: a slider labelled "€200000000" is unreadable.
market_values = pd.to_numeric(players["market_value_eur"], errors="coerce").dropna()
budget_max_m = float(np.ceil(market_values.max() / 1e6)) if not market_values.empty else 20.0
budget_max_m = max(budget_max_m, 1.0)
budget_range_m = st.sidebar.slider(
        "Budget",
        min_value=0.0,
        max_value=budget_max_m,
        value=(0.0, budget_max_m),
        step=0.5,
        format="€%.1fm",
        help="Market-value range for candidates. Players with no Transfermarkt "
             "market value are left out.",
)

exclude_fallback = st.sidebar.toggle(
        "Exclude FBref-position fallbacks",
        value=False,
        help="Hide rows whose position group was guessed from the coarse FBref position.",
)


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
        """Return one consistent candidate pool for comparables and value charts."""
        result = frame.copy()
        result["season"] = result["season"].astype(str)

        # An empty league or season selection means no candidates, not all of
        # them - the section messages then say which filter to loosen.
        result = result[
                (result["position_group"] == group)
                & result["season"].isin(candidate_seasons)
                & result["league"].isin(selected_leagues)
        ]

        result = result[
                pd.to_numeric(result["age"], errors="coerce").between(*age_range)
                & pd.to_numeric(result["minutes"], errors="coerce").ge(minimum_minutes)
        ]

        actual_value = pd.to_numeric(result["market_value_eur"], errors="coerce")
        budget_low, budget_high = (bound * 1e6 for bound in budget_range_m)
        result = result[actual_value.between(budget_low, budget_high).fillna(False)]

        if exclude_fallback:
                result = result[result["position_source"] != "fbref_fallback"]

        return result


# The pool is passed to both shortlist sections, so all candidate-facing views
# share exactly the same filter decisions.
pool = apply_filters(players)

st.sidebar.divider()
st.sidebar.caption(
        "Comparables are found with cosine distance on PCA style scores, within "
        "position group. Market values are crowd-sourced Transfermarkt estimates, "
        "not transfer fees."
)


# ---------------------------------------------------------------------------
# 2. Player search and header
# ---------------------------------------------------------------------------
st.title("Player recruitment & similarity")

# Search uses position and season only. Budget, age, minutes and league filters
# describe the replacement pool, not the player being replaced.
search_base = players[
        (players["position_group"] == group)
        & (players["season"].astype(str) == season)].copy()

if search_base.empty:
        st.warning("No players match this position and season. Change the filters.")
        st.stop()

search_base = search_base.sort_values(["player", "team"], kind="stable")
search_labels = {
        str(row["player_season_id"]): (
                f"{row['player']} — {row['team']} "
                f"({euros(row['market_value_eur'])})"
        )
        for _, row in search_base.iterrows()
}
search_ids = list(search_labels)

# Open on the highest-minute searchable player who is also a candidate, so the
# first view has comparables. If the candidate filters exclude everyone, still
# open on somebody: the filters describe the pool, never the searched player.
searchable_candidates = search_base[search_base["player_season_id"].isin(pool["player_season_id"])]
default_source = searchable_candidates if not searchable_candidates.empty else search_base
default_id = str(default_source.nlargest(1, "minutes").iloc[0]["player_season_id"])

# The widget reads its value from this key, so it takes no index= of its own;
# passing both makes Streamlit warn on every rerun.
if st.session_state.get("selected_player_id") not in search_labels:
        st.session_state.selected_player_id = default_id

selected_id = st.selectbox(
        "Search for a player",
        options=search_ids,
        format_func=lambda player_id: search_labels[player_id],
        key="selected_player_id",
)

target = players.loc[players["player_season_id"] == selected_id].iloc[0]

st.subheader(str(target["player"]))
name_caption = f"{target['team']} · {target['league']} · {season_label(target['season'])}"
st.caption(name_caption)


def value_verdict(row: pd.Series) -> tuple[str, str, str]:
        """(short label, pill style, sentence) for one player's value.

        Checked in order: no model estimate first - value_ratio is NaN then,
        and every other branch would wrongly read as "fair value".
        """
        if pd.isna(row["predicted_value_eur"]):
                return ("No estimate", "info",
                        f"No model estimate: Transfermarkt has no market value of at least "
                        f"{euros(config.MIN_MARKET_VALUE_EUR)} for this player-season.")
        ratio = row["value_ratio"]
        # Every row with a prediction has a boolean flag, so bool() is safe here.
        if bool(row["undervalued"]):
                return ("Undervalued", "success",
                        f"Undervalued: the model expects {ratio:.1f}x his market value.")
        if ratio >= 1.15:
                return ("Cheaper than model", "info",
                        f"Cheaper than the model expects ({ratio:.2f}x), but short of the "
                        f"undervalued threshold.")
        if ratio <= 0.85:
                return ("Pricier than model", "warning",
                        f"Valued above what the model expects (it predicts {ratio:.2f}x his market value).")
        return ("Fair value", "info", "Market value is in line with the model's expectation.")


verdict_label, verdict_style, verdict_text = value_verdict(target)

metric_columns = st.columns(5)
metric_columns[0].metric("Age", f"{float(target['age']):.1f}" if pd.notna(target["age"]) else "—")
metric_columns[1].metric("Minutes", f"{int(target['minutes']):,}" if pd.notna(target["minutes"]) else "—")
metric_columns[2].metric("Market value", euros(target["market_value_eur"]))
metric_columns[3].metric("Model expectation", euros(target["predicted_value_eur"]))
metric_columns[4].metric("Value verdict", verdict_label)

# A coloured pill reads at a glance in a way a ratio never will. The copy
# describes the model's view, not a forecast.
{"success": st.success, "info": st.info, "warning": st.warning}[verdict_style](verdict_text)

if str(target["position_source"]) != "transfermarkt":
        st.caption(
                "Position was estimated from FBref's coarse position field; "
                "treat this profile as a fallback rather than a verified sub-position."
        )


# ---------------------------------------------------------------------------
# 3. Comparables  |  4. Radar
# ---------------------------------------------------------------------------
st.divider()
left, right = st.columns([1.15, 1])

target_percentiles = percentiles.loc[
        percentiles["player_season_id"] == selected_id
]
target_percentile_row = target_percentiles.iloc[0] if not target_percentiles.empty else None

player_lookup = players.set_index("player_season_id")
target_comps = comps[
        (comps["player_season_id"] == selected_id)
        & (~comps["same_player"].fillna(False))
].copy()

# Keeping only IDs in the filtered pool makes every shortlist row honour the
# sidebar budget and eligibility settings.
pool_ids = set(pool["player_season_id"].astype(str))
target_comps = target_comps[
        target_comps["comp_player_season_id"].astype(str).isin(pool_ids)
].copy()

# Every remaining comparable is in the pool, so a plain merge finds them all.
comp_details = pool[["player_season_id", "player", "team", "season", "market_value_eur"]].rename(
        columns={"player_season_id": "comp_player_season_id"})
comp_details["comp_player_season_id"] = comp_details["comp_player_season_id"].astype(str)
target_comps["comp_player_season_id"] = target_comps["comp_player_season_id"].astype(str)
target_comps = target_comps.merge(comp_details, on="comp_player_season_id", how="left")
target_comps = target_comps.sort_values("rank").reset_index(drop=True)

comparison_id = None
comparison_row = None
table_selection = st.session_state.get(f"comparable_table_{selected_id}", {})
selected_rows = table_selection.get("selection", {}).get("rows", [])
if target_comps.empty:
        selected_rows = []
if selected_rows:
        selected_index = int(selected_rows[0])
        if 0 <= selected_index < len(target_comps):
                comparison_id = str(
                        target_comps.iloc[selected_index]["comp_player_season_id"]
                )
                if comparison_id in player_lookup.index:
                        comparison_row = player_lookup.loc[comparison_id]

with left:
        st.subheader("Style profile")
        if target_percentile_row is None:
                st.info("No percentile profile is available for the selected player.")
        else:
                comparison_percentiles = (
                        percentiles.loc[percentiles["player_season_id"] == comparison_id]
                        if comparison_id is not None else pd.DataFrame()
                )
                comparison_percentile_row = (
                        comparison_percentiles.iloc[0]
                        if not comparison_percentiles.empty else None
                )
                radar_fig = radar.radar(
                        target_percentile_row,
                        config.GROUP_SPECS[group].radar_axes(),
                        str(target["player"]),
                        comparison=comparison_percentile_row,
                        comparison_name=(
                                str(comparison_row["player"])
                                if comparison_row is not None else None
                        ),
                )
                st.plotly_chart(radar_fig, width="stretch")

with right:
        st.subheader("Closest comparables")
        if target_comps.empty:
                st.info(
                        "No comparable players remain in the filtered pool. "
                        "Loosen the budget, minutes, age, league or candidate-season filters."
                )
        else:
                table = target_comps[["rank", "player", "team", "season", "similarity"]].copy()
                table["season"] = table["season"].map(season_label)
                # In millions: "€45000000" is unreadable in a table cell.
                table["value_m"] = target_comps["market_value_eur"] / 1e6
                table = table.rename(columns={
                        "rank": "Rank",
                        "player": "Player",
                        "team": "Club",
                        "season": "Season",
                        "similarity": "Similarity",
                        "value_m": "Market value",
                })

                st.dataframe(
                        table,
                        width="stretch",
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        key=f"comparable_table_{selected_id}",
                        column_config={
                                "Rank": st.column_config.NumberColumn(format="%d"),
                                "Similarity": st.column_config.ProgressColumn(
                                        "Similarity",
                                        min_value=0,
                                        max_value=1,
                                        format="%.3f",
                                ),
                                "Market value": st.column_config.NumberColumn(
                                        "Market value",
                                        format="€%.1fm",
                                ),
                        },
                )
                st.caption("Click a row to overlay that player on the radar.")


# ---------------------------------------------------------------------------
# 5. Value for money  |  6. Style map
# ---------------------------------------------------------------------------
st.divider()
value_col, map_col = st.columns(2)
with value_col:
        st.subheader("Value for money")
        # The searched player is not always a candidate (his own value may be
        # over the budget), but he should still be marked on the chart.
        scatter_frame = pool
        if selected_id not in pool_ids:
                scatter_frame = pd.concat([pool, players[players["player_season_id"] == selected_id]])
        value_figure = radar.value_scatter(scatter_frame, highlight_id=selected_id)
        st.plotly_chart(value_figure, width="stretch")
        if pd.isna(target["predicted_value_eur"]):
                st.caption(f"{target['player']} has no model estimate, so he is not on the chart.")
        st.caption(
                "Above the dashed line = cheaper than the model expects for his age, "
                "minutes, contract, league and output. Market values are crowd-sourced "
                "Transfermarkt estimates, not transfer fees."
        )
with map_col:
        st.subheader("Style map")
        group_map = style[
                style["position_group"] == group
        ].merge(
                players[["player_season_id", "player", "team"]],
                on="player_season_id",
                how="left",
        ) if style is not None else pd.DataFrame()

        if group_map.empty:
                st.info("No style-map coordinates are available for this position group.")
        else:
                method = str(group_map["method"].dropna().iloc[0]).upper()
                map_figure = go.Figure()
                map_figure.add_trace(go.Scatter(
                        x=group_map["x"],
                        y=group_map["y"],
                        mode="markers",
                        marker=dict(
                                color="rgba(94,107,100,0.38)",
                                size=7,
                                line=dict(width=0),
                        ),
                        customdata=group_map[["player", "team"]].fillna("Unknown").to_numpy(),
                        hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<extra></extra>",
                        name="Players",
                ))

                highlighted_map = group_map[
                        group_map["player_season_id"] == selected_id
                ]
                if not highlighted_map.empty:
                        map_figure.add_trace(go.Scatter(
                                x=highlighted_map["x"],
                                y=highlighted_map["y"],
                                mode="markers+text",
                                text=highlighted_map["player"],
                                textposition="top center",
                                marker=dict(
                                        color=radar.PRIMARY,
                                        size=13,
                                        line=dict(color="white", width=1),
                                ),
                                name="Selected",
                                hovertemplate="<b>%{text}</b><extra></extra>",
                        ))

                map_figure.update_layout(
                        height=430,
                        margin=dict(l=55, r=20, t=20, b=55),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        showlegend=False,
                        xaxis=dict(title=f"{method} dimension 1", gridcolor=radar.GRID, zeroline=False),
                        yaxis=dict(title=f"{method} dimension 2", gridcolor=radar.GRID, zeroline=False),
                )
                st.plotly_chart(map_figure, width="stretch")
                st.caption(
                        f"The axes show {method} coordinates for presentation. "
                        "Comparables are computed in PCA space; this picture is "
                        "not the neighbour-search model."
                )


st.divider()
st.caption(
        f"{len(players):,} player-seasons · {players['league'].nunique()} leagues · "
        "data: FBref via soccerdata, market values via transfermarkt-datasets. "
        "This tool produces a shortlist to scout, not a decision to sign."
)
