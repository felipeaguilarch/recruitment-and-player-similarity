"""Stage 3b - playing roles: a best-fit role label for every player-season.

The similarity model describes style continuously, which is right for finding
comparables but is not how a recruitment brief is written. "We need a
deep-lying playmaker" is. This stage puts a role vocabulary on top of the same
data, as a label and a filter - it never feeds the neighbour search.

**How a role is scored.** Each role in config.ROLES lists stats whose high,
low or middling percentile counts in its favour. For every player-season:

1. Each stat is percentile-ranked within position group and season, exactly
   as features.py does for the radar (method="max", so the top player is 100).
2. A "low" stat is flipped (100 - pct) and a "middle" stat scores
   100 - 2 * |pct - 50|, so the median player gets 100 on it.
3. The role's raw score is the mean of its stats' scores, skipping any the
   player has no value for.
4. Raw scores are percentile-ranked again within group and season, so every
   role is on the same 0-100 scale. Without this, a role defined by four
   correlated stats would beat one defined by six independent ones simply
   because its averages spread wider.

The best-scoring role is the label. When the second-best is within
config.ROLE_HYBRID_MARGIN points the player is a hybrid ("Box-to-box /
Midfield playmaker") - midfield style is a spectrum, and forcing a single
label on a player who sits between two roles would be false precision.

**Why scores rather than clusters.** Clustering the same data into roles was
tried first: on all style dimensions the clusters split players at dominant
teams from everyone else (volume, not style), and without the volume
dimension a third of players sat on a boundary and the same player changed
cluster between seasons. Role scores are transparent, use football language
chosen by hand, and degrade gracefully into hybrids.

**Validation.** reports/role_stability.csv: how often a player's role in one
season matches his role in his next, against the agreement expected by
chance. reports/role_top_players.csv: the five highest-scoring regulars
(config.MIN_MINUTES) per role and season, for the eye test.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from . import config, paths
from .value_model import player_groups

logger = logging.getLogger(__name__)

#: Players listed per role and season in reports/role_top_players.csv.
TOP_PLAYERS_PER_ROLE = 5


def add_ratios(features: pd.DataFrame) -> pd.DataFrame:
        """Add the derived stats roles can use beyond the feature catalogue.

        config.ROLE_RATIOS columns are a / (a + b) and config.ROLE_SHARES
        columns part / whole: NaN where the divisor is zero (a player with no
        progression has no pass share) - the role score then skips that stat
        rather than guessing. config.ROLE_POSSESSION_ADJUSTED columns are
        scaled by on_ball_factor.
        """
        out = features.copy()
        for name, (numerator, other, _label) in config.ROLE_RATIOS.items():
                if numerator not in out.columns or other not in out.columns:
                        continue
                top = pd.to_numeric(out[numerator], errors="coerce")
                total = top + pd.to_numeric(out[other], errors="coerce")
                out[name] = (top / total.where(total > 0)).astype(float)
        for name, (part, whole, _label) in config.ROLE_SHARES.items():
                if part not in out.columns or whole not in out.columns:
                        continue
                denom = pd.to_numeric(out[whole], errors="coerce")
                out[name] = (pd.to_numeric(out[part], errors="coerce") / denom.where(denom > 0)).astype(float)
        if "team_possession" in out.columns:
                factor = on_ball_factor(pd.to_numeric(out["team_possession"], errors="coerce"))
                for name, (stat, _label) in config.ROLE_POSSESSION_ADJUSTED.items():
                        if stat in out.columns:
                                out[name] = (pd.to_numeric(out[stat], errors="coerce") * factor).astype(float)
        return out


def on_ball_factor(team_possession: pd.Series) -> pd.Series:
        """Multiplier that scales on-ball volume to a neutral 50% possession.

        The mirror of features.possession_factor: a team with 68% possession
        gives its players more of the ball, so their passes and touches are
        scaled DOWN by 50/68 = 0.74. Possession of 5% or less is treated as
        unknown, unknown falls back to 1.0, and the result is clipped to
        [0.5, 2.0], all as in possession_factor.
        """
        factor = 50 / team_possession.where(team_possession > 5, np.nan)
        return factor.fillna(1.0).clip(lower=0.5, upper=2.0)


def stat_scores(pct: pd.Series, direction: str) -> pd.Series:
        """One stat's contribution to a role, from its 0-100 percentile."""
        if direction == "high":
                return pct
        if direction == "low":
                return 100 - pct
        if direction == "middle":
                return 100 - 2 * (pct - 50).abs()
        raise ValueError(f"Unknown direction {direction!r}")


def score_roles(features: pd.DataFrame) -> pd.DataFrame:
        """Every role score for every player-season, in long format.

        `features` needs player_season_id, position_group, season and the
        stats config.ROLES uses (add_ratios is applied here). Returns
        player_season_id, position_group, season, role, role_order (the
        role's position in its group's list) and score (0-100).

        A stat missing from `features` is logged and left out of its role;
        a role with no usable stats at all is skipped.
        """
        frame = add_ratios(features)
        frames: list[pd.DataFrame] = []

        for group, roles in config.ROLES.items():
                rows = frame.loc[frame["position_group"] == group].reset_index(drop=True)
                if rows.empty:
                        continue
                needed = sorted({stat for role in roles for stat in role.stats()})
                absent = [stat for stat in needed if stat not in rows.columns]
                if absent:
                        logger.warning("%s roles: stats not in the features table, skipped: %s", group, absent)
                present = [stat for stat in needed if stat in rows.columns]

                # Step 1: the same percentile ranking features.py uses for radars.
                pct = (
                        rows[present].apply(pd.to_numeric, errors="coerce")
                        .groupby(rows["season"], observed=True)
                        .rank(pct=True, method="max") * 100
                )

                for order, role in enumerate(roles):
                        parts = [
                                stat_scores(pct[stat], direction)
                                for direction, stats in (("high", role.high), ("low", role.low),
                                                         ("middle", role.middle))
                                for stat in stats if stat in pct.columns
                        ]
                        if not parts:
                                logger.warning("%s role %r has no usable stats; skipped", group, role.name)
                                continue
                        # Steps 2-3: mean of the stat scores, skipping missing values.
                        raw = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
                        # Step 4: re-rank so every role shares one 0-100 scale.
                        score = raw.groupby(rows["season"], observed=True).rank(pct=True, method="max") * 100
                        frames.append(pd.DataFrame({
                                "player_season_id": rows["player_season_id"],
                                "position_group": group,
                                "season": rows["season"].astype(str),
                                "role": role.name,
                                "role_order": order,
                                "score": score.astype(float),
                        }))

        if not frames:
                return pd.DataFrame(columns=["player_season_id", "position_group", "season",
                                             "role", "role_order", "score"])
        return pd.concat(frames, ignore_index=True)


def best_roles(scores: pd.DataFrame, margin: float | None = None) -> pd.DataFrame:
        """Each player-season's best and second-best role, and its label.

        Returns player_season_id, position_group, season, primary_role,
        primary_score, secondary_role, secondary_score, hybrid and
        role_label. Ties go to the role listed first in config.ROLES, so the
        label never depends on row order.
        """
        margin = config.ROLE_HYBRID_MARGIN if margin is None else margin
        ranked = (
                scores.dropna(subset=["score"])
                .sort_values(["player_season_id", "score", "role_order"], ascending=[True, False, True])
        )
        ranked["place"] = ranked.groupby("player_season_id").cumcount()
        first = ranked[ranked["place"] == 0].set_index("player_season_id")
        second = ranked[ranked["place"] == 1].set_index("player_season_id")

        out = first[["position_group", "season"]].copy()
        out["primary_role"] = first["role"]
        out["primary_score"] = first["score"]
        out["secondary_role"] = second["role"].reindex(out.index)
        out["secondary_score"] = second["score"].reindex(out.index)
        out["hybrid"] = (out["primary_score"] - out["secondary_score"]).le(margin).fillna(False)
        out["role_label"] = np.where(
                out["hybrid"],
                out["primary_role"] + " / " + out["secondary_role"].fillna(""),
                out["primary_role"],
        )
        return out.reset_index()


def role_stability(roles: pd.DataFrame, identity: pd.DataFrame) -> pd.DataFrame:
        """How often a player keeps his role from one of his seasons to the next.

        Pairs are a player's consecutive seasons in the data (2022-23 to
        2024-25 counts when he has no 2023-24 row) within the same position
        group; players identified as in value_model.player_groups.

          same_primary   - the two best-fit roles are the same.
          same_any       - the roles overlap once hybrids count both of
                           theirs (a "Box-to-box / Midfield playmaker" who
                           becomes a "Midfield playmaker" counts).
          chance_primary - the same_primary rate expected if roles were
                           assigned at random with the observed role shares:
                           sum over roles of share_first * share_second.

        One row per position group plus an "All" row (pair-weighted).
        """
        frame = roles.merge(identity, on="player_season_id", how="left")
        frame["player_key"] = player_groups(frame)
        frame = frame.sort_values(["player_key", "position_group", "season"])
        nxt = frame.groupby(["player_key", "position_group"]).shift(-1)
        pairs = frame.assign(
                next_primary=nxt["primary_role"],
                next_secondary=nxt["secondary_role"],
                next_hybrid=nxt["hybrid"],
        ).dropna(subset=["next_primary"])

        def roles_of(primary, secondary, hybrid):
                return {primary, secondary} if bool(hybrid) and isinstance(secondary, str) else {primary}

        pairs["same_primary"] = pairs["primary_role"] == pairs["next_primary"]
        pairs["same_any"] = [
                bool(roles_of(a, b, c) & roles_of(d, e, f))
                for a, b, c, d, e, f in zip(
                        pairs["primary_role"], pairs["secondary_role"], pairs["hybrid"],
                        pairs["next_primary"], pairs["next_secondary"], pairs["next_hybrid"],
                )
        ]

        rows = []
        for group in [g for g in config.POSITION_GROUPS if g in set(pairs["position_group"])]:
                sub = pairs[pairs["position_group"] == group]
                first = sub["primary_role"].value_counts(normalize=True)
                second = sub["next_primary"].value_counts(normalize=True)
                chance = float((first * second.reindex(first.index).fillna(0)).sum())
                rows.append({
                        "position_group": group,
                        "pairs": len(sub),
                        "roles": len(config.ROLES.get(group, [])),
                        "same_primary": round(float(sub["same_primary"].mean()), 3),
                        "same_any": round(float(sub["same_any"].mean()), 3),
                        "chance_primary": round(chance, 3),
                })
        table = pd.DataFrame(rows)
        if not table.empty:
                weights = table["pairs"]
                table.loc[len(table)] = {
                        "position_group": "All",
                        "pairs": int(weights.sum()),
                        "roles": np.nan,
                        **{column: round(float(np.average(table[column], weights=weights)), 3)
                           for column in ("same_primary", "same_any", "chance_primary")},
                }
        return table


def top_players(scores: pd.DataFrame, identity: pd.DataFrame, n: int = TOP_PLAYERS_PER_ROLE) -> pd.DataFrame:
        """The n highest-scoring regulars (config.MIN_MINUTES) per role and season."""
        frame = scores.merge(identity, on="player_season_id", how="left")
        frame = frame[pd.to_numeric(frame["minutes"], errors="coerce") >= config.MIN_MINUTES]
        frame = frame.sort_values(
                ["position_group", "role_order", "season", "score", "player_season_id"],
                ascending=[True, True, False, False, True],
        )
        top = frame.groupby(["position_group", "role", "season"], sort=False).head(n)
        top = top.assign(rank=top.groupby(["position_group", "role", "season"]).cumcount() + 1)
        return top[["position_group", "role", "season", "rank", "player", "team", "score"]].reset_index(drop=True)


def run(force: bool = False) -> pd.DataFrame:
        """Stage entry point: write paths.ROLES, paths.ROLE_SCORES and the reports.

        Reads paths.FEATURES. Returns the per-player-season role table.
        """
        paths.ensure_dirs()
        if paths.ROLES.exists() and paths.ROLE_SCORES.exists() and not force:
                logger.info("%s exists; use --force to rebuild.", paths.ROLES)
                return pd.read_parquet(paths.ROLES)
        if not paths.FEATURES.exists():
                raise SystemExit("Run `python run_pipeline.py --stage features` first.")

        features = pd.read_parquet(paths.FEATURES)
        scores = score_roles(features)
        roles = best_roles(scores)

        scores.drop(columns="role_order").to_parquet(paths.ROLE_SCORES, index=False)
        roles.to_parquet(paths.ROLES, index=False)
        logger.info("Wrote %s (%s player-seasons, %.0f%% hybrids)",
                    paths.ROLES, len(roles), 100 * roles["hybrid"].mean())

        identity = features[["player_season_id", "player", "team", "minutes", "tm_player_id"]]
        stability = role_stability(roles, identity)
        stability.to_csv(paths.ROLE_STABILITY, index=False)
        top_players(scores, identity).to_csv(paths.ROLE_TOP_PLAYERS, index=False)
        if not stability.empty:
                overall = stability.iloc[-1]
                logger.info(
                        "Role stability: same role next season %.0f%% (%.0f%% counting hybrids) "
                        "vs %.0f%% by chance, over %d pairs",
                        100 * overall["same_primary"], 100 * overall["same_any"],
                        100 * overall["chance_primary"], overall["pairs"],
                )
        return roles


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
        parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("--force", action="store_true")
        run(force=parser.parse_args().force)
