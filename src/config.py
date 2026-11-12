"""Every tunable choice in the project lives here.

Edit this file rather than the pipeline modules. The three things most worth
revisiting are LEAGUES, SEASONS and MIN_MINUTES, in that order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

# soccerdata league identifiers. Confirm against the installed version with:
#     python -c "import soccerdata as sd; print(sd.FBref.available_leagues())"
# Start with the big five; add a selling league once the pipeline runs end to
# end, because that is where undervalued targets actually live.
LEAGUES: list[str] = [
        "ENG-Premier League",
        "ESP-La Liga",
        "ITA-Serie A",
        "GER-Bundesliga",
        "FRA-Ligue 1",
        # "NED-Eredivisie",
        # "POR-Primeira Liga",
]

# IMPORTANT CONSTRAINT, checked 2026-11-06:
# transfermarkt-datasets has PAUSED updates. The data is current to
# 2026-07-06, player_valuations stop at 2026-06-12, and 2026/27 squads are not
# covered at all. So 2025-26 is the last season with usable market values.
# Adding "2627" here would produce FBref rows with no value to join to.
# See: https://github.com/dcaribou/transfermarkt-datasets/discussions/383
SEASONS: list[str] = ["2324", "2425", "2526"]

# Minutes floor. 900 = ten full matches. The app exposes this as a slider so a
# user can loosen it to hunt prospects; this is just the pipeline default.
MIN_MINUTES: int = 900

# Re-run validation at this looser floor to measure how much the output
# degrades. The number is worth reporting in the README.
MIN_MINUTES_SENSITIVITY: int = 600

# FBref pages to download. The first five are supported by stock soccerdata
# 1.9.0; the rest need src/fbref_extra.py, which widens the reader.
FBREF_STAT_TYPES: list[str] = [
        "standard",
        "shooting",
        "misc",
        "passing",
        "passing_types",
        "gca",
        "defense",
        "possession",
]

# Goalkeepers are deliberately out of scope. Their stat profile shares almost
# nothing with outfield players and would need its own feature set, its own
# PCA and its own value model. Say so in the README rather than half-doing it.
EXCLUDE_GOALKEEPERS: bool = True


# ---------------------------------------------------------------------------
# Position groups
# ---------------------------------------------------------------------------

POSITION_GROUPS: list[str] = ["CB", "FB", "DM", "CM", "AM", "W", "ST"]

# Transfermarkt `sub_position` -> group. This is the primary mapping, and the
# reason the Transfermarkt join is worth the trouble: FBref's own `pos` column
# only gives DF / MF / FW, which cannot separate a centre-back from a
# full-back - exactly the distinction this tool exists to respect.
SUB_POSITION_TO_GROUP: dict[str, str] = {
        "Centre-Back": "CB",
        "Left-Back": "FB",
        "Right-Back": "FB",
        "Defensive Midfield": "DM",
        "Central Midfield": "CM",
        "Left Midfield": "CM",
        "Right Midfield": "CM",
        "Attacking Midfield": "AM",
        "Left Winger": "W",
        "Right Winger": "W",
        "Second Striker": "ST",
        "Centre-Forward": "ST",
}

# Fallback for players who failed the Transfermarkt join, keyed on FBref's
# coarse `pos`. Deliberately crude - these players are flagged
# `position_source == "fbref_fallback"` so you can exclude them from the
# case study if you would rather not defend the guess.
FBREF_POS_TO_GROUP: dict[str, str] = {
        "DF": "CB",
        "DF,MF": "FB",
        "MF,DF": "DM",
        "MF": "CM",
        "MF,FW": "AM",
        "FW,MF": "W",
        "FW": "ST",
}


# ---------------------------------------------------------------------------
# Feature catalogue
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Feature:
        """One model input.

        `patterns` are regexes tried in order against the flattened FBref column
        names (`{stat_type}__{header_group}__{stat}`, snake_cased). The first hit
        wins. Flattening is fully qualified because FBref reuses short stat names
        across pages - `Int` appears under both `defense__blocks` and
        `misc__performance`.

        Column names are the part of this project most likely to drift when FBref
        changes a table header, which is why resolution is pattern-based and
        reported rather than assumed. Run

            python run_pipeline.py --stage inspect-columns

        to dump what actually came back, then fix the patterns here.
        """

        name: str
        patterns: tuple[str, ...]
        per90: bool = True  # divide by 90s played; False for rates and percentages
        padj: bool = False  # possession-adjust (defensive actions only)
        label: str = ""  # human-readable, used on radar axes

        def display(self) -> str:
                return self.label or self.name.replace("_", " ").title()


def _f(name: str, *patterns: str, per90: bool = True, padj: bool = False, label: str = "") -> Feature:
        return Feature(name=name, patterns=patterns, per90=per90, padj=padj, label=label)


# Registry of every feature any position group can ask for.
FEATURES: dict[str, Feature] = {f.name: f for f in [
        # ---------------------------------------------------------------
        # NOTE (verified 2026-11-11): FBref no longer serves the
        # expected-goals family (xG / npxG / xAG / xA), the progressive
        # COUNTS (PrgC / PrgP / PrgR), aerial duels, ball recoveries or
        # average shot distance. Confirmed by inspecting the raw HTML -
        # those data-stat attributes are absent from the page, not lost in
        # parsing. Understat was the planned source for the xG family but is
        # in maintenance ("We will come back soon").
        #
        # Everything below resolves against the real 137-column table.
        # Where a stat was lost, a documented substitute is used:
        #   PrgC / PrgP  -> progressive DISTANCE (carries / passes)
        #   PrgR         -> passes received (possession__flat__rec)
        #   xG / npxG    -> shot volume + touches in box + SCA
        #   xA / xAG     -> key passes + passes into box
        # ---------------------------------------------------------------

        # --- volume / output ---
        _f("goals", r"^standard__performance__gls$", label="Goals"),
        _f("assists", r"^standard__performance__ast$", label="Assists"),
        _f("non_pen_goals", r"^standard__performance__g_pk$", label="Non-Pen Goals"),
        _f("shots", r"^shooting__standard__sh$", label="Shots"),
        _f("shots_on_target", r"^shooting__standard__sot$", label="Shots on Target"),
        _f("shots_on_target_pct", r"^shooting__standard__sot_pct$", per90=False, label="Shot Accuracy %"),
        _f("goals_per_shot", r"^shooting__standard__g_sh$", per90=False, label="Goals / Shot"),
        _f("goals_per_sot", r"^shooting__standard__g_sot$", per90=False, label="Goals / Shot on Target"),

        # --- progression (distance-based; the counts are gone) ---
        _f("prog_carry_distance", r"^possession__carries__prgdist$", label="Prog. Carry Distance"),
        _f("prog_pass_distance", r"^passing__total__prgdist$", label="Prog. Pass Distance"),
        _f("total_carry_distance", r"^possession__carries__totdist$", label="Carry Distance"),
        _f("carries", r"^possession__carries__carries$", label="Carries"),
        _f("carries_into_final_third", r"^possession__carries__1_3$", label="Carries into Final 3rd"),
        _f("carries_into_box", r"^possession__carries__cpa$", label="Carries into Box"),
        _f("passes_received", r"^possession__flat__rec$", label="Passes Received"),

        # --- passing ---
        _f("passes_attempted", r"^passing__total__att$", label="Passes Attempted"),
        _f("pass_completion_pct", r"^passing__total__cmp_pct$", per90=False, label="Pass Completion %"),
        _f("pass_completion_short_pct", r"^passing__short__cmp_pct$", per90=False, label="Short Pass %"),
        _f("pass_completion_medium_pct", r"^passing__medium__cmp_pct$", per90=False, label="Medium Pass %"),
        _f("pass_completion_long_pct", r"^passing__long__cmp_pct$", per90=False, label="Long Pass %"),
        _f("long_passes_attempted", r"^passing__long__att$", label="Long Passes"),
        _f("key_passes", r"^passing__flat__kp$", label="Key Passes"),
        _f("passes_into_final_third", r"^passing__flat__1_3$", label="Passes into Final 3rd"),
        _f("passes_into_box", r"^passing__flat__ppa$", label="Passes into Box"),
        _f("crosses_into_box", r"^passing__flat__crspa$", label="Crosses into Box"),
        _f("crosses", r"^passing_types__pass_types__crs$", r"^misc__performance__crs$", label="Crosses"),
        _f("through_balls", r"^passing_types__pass_types__tb$", label="Through Balls"),
        _f("switches", r"^passing_types__pass_types__sw$", label="Switches"),
        _f("corners", r"^passing_types__pass_types__ck$", label="Corners"),
        _f("passes_blocked", r"^passing_types__outcomes__blocks$", label="Passes Blocked"),

        # --- creation ---
        _f("sca", r"^gca__sca__sca$", label="Shot-Creating Actions"),
        _f("gca", r"^gca__gca__gca$", label="Goal-Creating Actions"),
        _f("sca_from_take_ons", r"^gca__sca_types__to$", label="SCA from Take-Ons"),
        _f("sca_from_dead_balls", r"^gca__sca_types__passdead$", label="SCA from Dead Balls"),

        # --- carrying / dribbling ---
        _f("take_ons_attempted", r"^possession__take_ons__att$", label="Take-Ons Attempted"),
        _f("take_ons_succeeded", r"^possession__take_ons__succ$", label="Take-Ons Completed"),
        _f("take_on_success_pct", r"^possession__take_ons__succ_pct$", per90=False, label="Take-On Success %"),
        _f("times_tackled_pct", r"^possession__take_ons__tkld_pct$", per90=False, label="Tackled on Take-On %"),
        _f("miscontrols", r"^possession__carries__mis$", label="Miscontrols"),
        _f("dispossessed", r"^possession__carries__dis$", label="Dispossessed"),

        # --- touches by zone ---
        _f("touches", r"^possession__touches__touches$", label="Touches"),
        _f("touches_def_pen", r"^possession__touches__def_pen$", label="Touches Own Box"),
        _f("touches_def_third", r"^possession__touches__def_3rd$", label="Touches Def 3rd"),
        _f("touches_mid_third", r"^possession__touches__mid_3rd$", label="Touches Mid 3rd"),
        _f("touches_att_third", r"^possession__touches__att_3rd$", label="Touches Att 3rd"),
        _f("touches_att_box", r"^possession__touches__att_pen$", label="Touches in Box"),

        # --- defending (possession-adjusted) ---
        _f("tackles", r"^defense__tackles__tkl$", padj=True, label="Tackles"),
        _f("tackles_won", r"^defense__tackles__tklw$", r"^misc__performance__tklw$", padj=True, label="Tackles Won"),
        _f("tackles_def_third", r"^defense__tackles__def_3rd$", padj=True, label="Tackles Def 3rd"),
        _f("tackles_mid_third", r"^defense__tackles__mid_3rd$", padj=True, label="Tackles Mid 3rd"),
        _f("tackles_att_third", r"^defense__tackles__att_3rd$", padj=True, label="Tackles Att 3rd"),
        _f("dribblers_tackled_pct", r"^defense__challenges__tkl_pct$", per90=False, label="Dribblers Tackled %"),
        _f("challenges_attempted", r"^defense__challenges__att$", padj=True, label="Dribblers Challenged"),
        _f("challenges_lost", r"^defense__challenges__lost$", padj=True, label="Challenges Lost"),
        _f("interceptions", r"^defense__flat__int$", r"^misc__performance__int$", padj=True, label="Interceptions"),
        _f("tackles_plus_interceptions", r"^defense__flat__tkl_int$", padj=True, label="Tackles + Int."),
        _f("blocks", r"^defense__blocks__blocks$", padj=True, label="Blocks"),
        _f("shots_blocked", r"^defense__blocks__sh$", padj=True, label="Shots Blocked"),
        _f("passes_blocked_def", r"^defense__blocks__pass$", padj=True, label="Passes Blocked (def)"),
        _f("clearances", r"^defense__flat__clr$", padj=True, label="Clearances"),
        _f("errors", r"^defense__flat__err$", label="Errors"),

        # --- discipline ---
        _f("fouls", r"^misc__performance__fls$", label="Fouls"),
        _f("fouls_drawn", r"^misc__performance__fld$", label="Fouls Drawn"),
        _f("offsides", r"^misc__performance__off$", label="Offsides"),
        _f("yellow_cards", r"^misc__performance__crdy$", label="Yellow Cards"),
]}


@dataclass(frozen=True)
class GroupSpec:
        """Feature set and radar layout for one position group."""

        group: str
        features: list[str]
        # Axes shown on the radar, in order. A subset of `features`, kept to 8-10
        # because twelve axes stop being readable.
        radar: list[str] = field(default_factory=list)

        def radar_axes(self) -> list[str]:
                return self.radar or self.features[:9]


GROUP_SPECS: dict[str, GroupSpec] = {
        "CB": GroupSpec("CB", features=[
                "tackles_won", "interceptions", "blocks", "shots_blocked", "clearances",
                "tackles_def_third", "dribblers_tackled_pct", "challenges_attempted",
                "prog_pass_distance", "prog_carry_distance", "pass_completion_pct",
                "pass_completion_long_pct", "long_passes_attempted",
                "passes_into_final_third", "passes_attempted", "touches_def_third",
                "errors", "fouls",
        ], radar=[
                "tackles_won", "interceptions", "clearances", "blocks",
                "dribblers_tackled_pct", "prog_pass_distance", "prog_carry_distance",
                "pass_completion_pct", "passes_into_final_third",
        ]),
        "FB": GroupSpec("FB", features=[
                "crosses_into_box", "crosses", "prog_carry_distance", "carries_into_final_third",
                "touches_att_third", "key_passes", "passes_into_box", "take_ons_attempted",
                "take_on_success_pct", "tackles_plus_interceptions", "dribblers_tackled_pct",
                "pass_completion_pct", "passes_received", "miscontrols", "tackles_def_third",
                "sca",
        ], radar=[
                "crosses_into_box", "key_passes", "passes_into_box", "prog_carry_distance",
                "touches_att_third", "take_ons_attempted", "tackles_plus_interceptions",
                "dribblers_tackled_pct", "pass_completion_pct",
        ]),
        "DM": GroupSpec("DM", features=[
                "tackles_plus_interceptions", "tackles_def_third", "tackles_mid_third",
                "interceptions", "blocks", "clearances", "passes_attempted",
                "pass_completion_pct", "prog_pass_distance", "switches", "long_passes_attempted",
                "touches_mid_third", "touches_def_third", "fouls", "errors",
                "challenges_lost",
        ], radar=[
                "tackles_plus_interceptions", "interceptions", "tackles_def_third",
                "passes_attempted", "pass_completion_pct", "prog_pass_distance",
                "switches", "touches_mid_third", "fouls",
        ]),
        "CM": GroupSpec("CM", features=[
                "prog_pass_distance", "passes_into_final_third", "passes_into_box", "sca",
                "key_passes", "prog_carry_distance", "carries_into_final_third",
                "take_ons_succeeded", "tackles_plus_interceptions", "touches_mid_third",
                "touches_att_third", "pass_completion_pct", "passes_attempted",
                "passes_received", "miscontrols", "through_balls",
        ], radar=[
                "prog_pass_distance", "passes_into_final_third", "sca", "key_passes",
                "prog_carry_distance", "take_ons_succeeded", "tackles_plus_interceptions",
                "passes_attempted", "pass_completion_pct",
        ]),
        "AM": GroupSpec("AM", features=[
                "key_passes", "sca", "gca", "passes_into_box", "through_balls",
                "take_ons_attempted", "take_ons_succeeded", "touches_att_box",
                "touches_att_third", "shots", "carries_into_box", "prog_carry_distance",
                "passes_received", "miscontrols", "fouls_drawn", "sca_from_take_ons",
        ], radar=[
                "shots", "key_passes", "sca", "passes_into_box", "take_ons_succeeded",
                "touches_att_box", "prog_carry_distance", "carries_into_box",
                "fouls_drawn",
        ]),
        "W": GroupSpec("W", features=[
                "shots", "key_passes", "take_ons_attempted", "take_ons_succeeded",
                "take_on_success_pct", "prog_carry_distance", "carries_into_box",
                "touches_att_box", "touches_att_third", "crosses", "miscontrols",
                "dispossessed", "fouls_drawn", "tackles_att_third", "passes_received",
                "sca",
        ], radar=[
                "shots", "key_passes", "sca", "take_ons_attempted", "take_on_success_pct",
                "prog_carry_distance", "touches_att_box", "crosses", "fouls_drawn",
        ]),
        "ST": GroupSpec("ST", features=[
                "shots", "shots_on_target", "shots_on_target_pct", "goals_per_shot",
                "non_pen_goals", "touches_att_box", "touches_att_third", "key_passes",
                "sca", "passes_received", "fouls_drawn", "miscontrols", "dispossessed",
                "take_ons_attempted", "offsides", "carries_into_box",
        ], radar=[
                "shots", "shots_on_target_pct", "goals_per_shot", "touches_att_box",
                "non_pen_goals", "key_passes", "sca", "passes_received", "fouls_drawn",
        ]),
}


# ---------------------------------------------------------------------------
# Similarity model
# ---------------------------------------------------------------------------

# Keep enough PCA components to explain this share of variance within a group.
# Usually lands between 5 and 8 components. Name the first three per group in
# the README - that narrative is half the value of the project.
PCA_VARIANCE_TARGET: float = 0.85
PCA_MIN_COMPONENTS: int = 3
PCA_MAX_COMPONENTS: int = 10

# Cosine on PCA scores compares the SHAPE of a profile and largely ignores its
# magnitude, which is what "plays like him" means. Euclidean blends style with
# level and will keep returning elite players as comparables for everyone.
SIMILARITY_METRIC: str = "cosine"
N_NEIGHBOURS: int = 25  # stored per player; the app shows the top 5

# A group with fewer than this many qualifying players produces meaningless
# neighbours. The pipeline warns rather than failing.
MIN_GROUP_SIZE: int = 150


# ---------------------------------------------------------------------------
# Value model
# ---------------------------------------------------------------------------

# Market values span roughly EUR 100k to EUR 180m and are close to log-normal.
# Fitting on raw euros lets a handful of superstars dominate the loss and turns
# every prediction under EUR 5m into noise - which is exactly the range that
# matters for recruitment.
VALUE_LOG_TRANSFORM: bool = True

VALUE_NUMERIC_FEATURES: list[str] = [
        "age",
        "minutes",
        "contract_years_remaining",
]

VALUE_CATEGORICAL_FEATURES: list[str] = [
        "league",
        "position_group",
        "season",
]

# Style/output stats fed to the value model. Deliberately a single universal
# set rather than each group's own feature list: PC1 means something different
# for a centre-back than for a winger, so per-group PCA scores are not
# comparable across the one pooled regression. Every stat here is available for
# outfield players in any position.
VALUE_STAT_FEATURES: list[str] = [
        "shots",
        "key_passes",
        "sca",
        "gca",
        "prog_pass_distance",
        "prog_carry_distance",
        "passes_received",
        "pass_completion_pct",
        "passes_into_final_third",
        "tackles_plus_interceptions",
        "interceptions",
        "touches_att_box",
        "take_ons_succeeded",
]

# Optionally append each group's PCA scores (pc1..pc5, zero-padded). Off by
# default for the reason above; flip to True to test whether it helps.
VALUE_USE_PCA_SCORES: bool = False
VALUE_N_PCA_SCORES: int = 5

# Cross-validation must be grouped BY PLAYER. The same player appears in
# several seasons, and letting his 2023-24 row train a model that predicts his
# 2024-25 row inflates R-squared into fiction.
CV_FOLDS: int = 5

# A player is flagged undervalued when actual value sits this far below
# prediction. 0.5 in log space is roughly a 1.65x gap.
UNDERVALUED_LOG_THRESHOLD: float = 0.5

# Drop player-seasons with no market value, or with implausible ones.
MIN_MARKET_VALUE_EUR: int = 50_000


# ---------------------------------------------------------------------------
# Name matching
# ---------------------------------------------------------------------------

# rapidfuzz token_set_ratio thresholds.
NAME_MATCH_ACCEPT: int = 90  # auto-accept at or above
NAME_MATCH_REJECT: int = 80  # auto-reject below
# Scores between the two land in reports/name_match_review.csv for you to
# adjudicate by hand into data/manual_name_overrides.csv.

# Candidate blocking: only compare players whose birth years agree within this
# many years. Blocking is what keeps the match fast AND prevents most false
# positives - it is more important than the scorer choice.
BIRTH_YEAR_TOLERANCE: int = 1


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

# Direct, credential-free download published by the transfermarkt-datasets
# project (verified reachable 2026-11-06, ~241 MB).
TM_ZIP_URL: str = (
        "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.zip"
)

# Tables actually used. The zip contains twelve; unpacking only these keeps
# data/raw manageable.
TM_TABLES: list[str] = ["players", "player_valuations", "competitions", "clubs"]

# Identify yourself when scraping. Replace with your own contact or project URL
# before running anything at volume - a generic agent string on a rate-limited
# site is how you get blocked.
USER_AGENT: str = "player-recruitment-tool/0.1 (portfolio project; +https://github.com/)"
