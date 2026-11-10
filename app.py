"""Recruitment and player similarity tool - Streamlit entry point.
[YOU BUILD THE SECTIONS]

This app only ever READS. It never scrapes, never refits and never calls
soccerdata. Everything expensive happens in run_pipeline.py and is committed
as parquet and joblib. That single rule is the difference between a link that
opens in two seconds and one that times out while somebody is looking at it.

The loading layer below is done - it is API knowledge, not design. The page
itself is yours. It runs right now: `streamlit run app.py` shows the skeleton
with each unbuilt section marked, so you can fill them in one at a time and
watch the page come together.

Build order that keeps you unblocked:
  1. Sidebar filters          -> you can see the pool shrink
  2. Player search + header   -> you can select somebody
  3. Comparables table        -> the core output
  4. Radar                    -> needs src/radar.py
  5. Value scatter            -> needs the value stage
  6. Style map                -> last; it is the least load-bearing

Design notes worth honouring, because the audience is non-technical:
  * A dashboard is scanned and operated, not read top to bottom. Put the
    summary above the detail.
  * Encode state in form as well as number - a coloured pill for the value
    flag reads at a glance in a way "0.54" never will.
  * Open on a real player, not an empty search box. First impressions are
    the entire point of a deployed link.
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

import pandas as pd  # noqa: E402

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


# ---------------------------------------------------------------------------
# 1. Sidebar filters
# ---------------------------------------------------------------------------
#
# TODO Build the filters recruitment staff actually reach for:
#        position group (st.selectbox over config.POSITION_GROUPS, but only
#          those present in the data), season, leagues (st.multiselect),
#          age range (st.slider with a tuple value), budget ceiling,
#          minimum minutes, and a toggle to exclude players whose
#          position_group came from the FBref fallback.
#      The budget ceiling is the filter that makes this feel like a working
#      tool rather than a demo - it is how a real shortlist gets made.
#      Then write apply_filters(frame) returning the filtered candidate pool.
#      Keep it a function: the pool is used by the comps list AND the value
#      scatter, and they must agree.

st.sidebar.title("Filters")
st.sidebar.info("TODO: build the filters (section 1 in app.py)")

group = config.POSITION_GROUPS[0]
season = str(sorted(players["season"].astype(str).unique())[-1])
pool = players[
        (players["position_group"] == group) & (players["season"].astype(str) == season)
]

st.sidebar.divider()
st.sidebar.caption(
        "Comparables are found with cosine distance on PCA style scores, within "
        "position group. Market values are crowd-sourced Transfermarkt estimates, "
        "not transfer fees."
)


# ---------------------------------------------------------------------------
# 2. Player search and header
# ---------------------------------------------------------------------------
#
# TODO A searchable st.selectbox over "Player — Club" labels, mapping back to
#      player_season_id. Note the player you SEARCH for should not be
#      restricted by the budget/age filters - those describe the candidates
#      you want back, not the player you are replacing.
#
# TODO Header: put the name in st.subheader, NOT st.metric. Metric values are
#      truncated with an ellipsis and mangle anything as long as
#      "Pierre-Emerick Aubameyang".
#      Then a row of st.metric for age, minutes, market value, model
#      expectation, and a coloured pill (st.success / st.warning / st.info)
#      for the value verdict.
#
# TODO Show a caption when position_source != "transfermarkt", so nobody
#      trusts a profile whose position was guessed.

st.title("Player recruitment & similarity")
st.info("TODO: build player search and the header block (section 2 in app.py)")

if pool.empty:
        st.warning("No players in the default pool - run the pipeline first.")
        st.stop()

target = pool.nlargest(1, "minutes").iloc[0]
selected_id = target["player_season_id"]
st.caption(f"Placeholder selection: {target['player']} ({target['team']})")


# ---------------------------------------------------------------------------
# 3. Comparables  |  4. Radar
# ---------------------------------------------------------------------------
#
# TODO 3. Filter `comps` to this player, drop same_player rows, and keep only
#         comparables that are inside the filtered pool - that is what makes
#         the budget ceiling meaningful. Render with st.dataframe and
#         column_config: ProgressColumn for similarity, NumberColumn with a
#         currency format for the values.
#         Pass on_select="rerun", selection_mode="single-row" so clicking a
#         row overlays that player on the radar. Handle the empty case with
#         a message telling the user which filter to loosen.
#
# TODO 4. Radar from percentiles for the selected player, overlaid with the
#         clicked comparable. Axes come from
#         config.GROUP_SPECS[group].radar_axes(). Needs src/radar.py.

st.divider()
left, right = st.columns([1.15, 1])
with left:
        st.subheader("Style profile")
        st.info("TODO: radar chart (section 4)")
with right:
        st.subheader("Closest comparables")
        st.info("TODO: comparables table (section 3)")


# ---------------------------------------------------------------------------
# 5. Value for money  |  6. Style map
# ---------------------------------------------------------------------------
#
# TODO 5. radar.value_scatter over the filtered pool, highlighting the
#         selected player. Caption it with the honest caveat: residuals skew
#         toward cheaper leagues by construction, so read it inside a filter.
#
# TODO 6. Scatter of style_map x/y for this group, highlighting the selected
#         player. Label the axes with the `method` column and say plainly
#         that comparables are computed in PCA space - a reader who assumes
#         the picture is the model will misread it.

st.divider()
value_col, map_col = st.columns(2)
with value_col:
        st.subheader("Value for money")
        st.info("TODO: predicted vs actual scatter (section 5)")
with map_col:
        st.subheader("Style map")
        st.info("TODO: UMAP style map (section 6)")


st.divider()
st.caption(
        f"{len(players):,} player-seasons · {players['league'].nunique()} leagues · "
        "data: FBref via soccerdata, market values via transfermarkt-datasets. "
        "This tool produces a shortlist to scout, not a decision to sign."
)
