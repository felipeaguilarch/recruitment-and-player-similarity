"""Tests that run on a clean clone, before any data has been downloaded.

Deliberately not a test suite for the models - those need data. These catch
the mistakes that otherwise cost you an hour into a pipeline run: a feature
listed in a position group but absent from the registry, a season key that
parses to the wrong year, a name normaliser that drops a character it should
keep.

    pytest -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.features import possession_factor, resolve_features
from src.fetch_fbref import snake
from src.merge import parse_fbref_age
from src.name_matching import normalise_name, season_end_year, season_midpoint


# --- config integrity ------------------------------------------------------

def test_every_group_feature_exists_in_registry():
        for group, spec in config.GROUP_SPECS.items():
                unknown = [f for f in spec.features if f not in config.FEATURES]
                assert not unknown, f"{group} lists features missing from FEATURES: {unknown}"


def test_radar_axes_are_a_subset_of_group_features():
        for group, spec in config.GROUP_SPECS.items():
                extra = set(spec.radar) - set(spec.features)
                assert not extra, f"{group} radar axes not in its feature list: {sorted(extra)}"


def test_radar_stays_readable():
        for group, spec in config.GROUP_SPECS.items():
                assert 5 <= len(spec.radar_axes()) <= 10, f"{group} has an unreadable radar"


def test_every_position_group_has_a_spec():
        assert set(config.GROUP_SPECS) == set(config.POSITION_GROUPS)


def test_sub_position_map_targets_real_groups():
        for sub_position, group in config.SUB_POSITION_TO_GROUP.items():
                assert group in config.POSITION_GROUPS, f"{sub_position} -> unknown {group}"


def test_value_model_stats_exist():
        unknown = [f for f in config.VALUE_STAT_FEATURES if f not in config.FEATURES]
        assert not unknown, f"value model asks for unknown features: {unknown}"


def test_name_match_thresholds_are_ordered():
        assert config.NAME_MATCH_REJECT < config.NAME_MATCH_ACCEPT


def test_seasons_stop_before_the_transfermarkt_cutoff():
        """transfermarkt-datasets is frozen at 2026-07-06, so 2026-27 has no values."""
        assert all(season_end_year(s) <= 2026 for s in config.SEASONS)


# --- column flattening -----------------------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [
        ("SoT%", "sot_pct"),
        ("npxG/Sh", "npxg_sh"),
        ("np:G-xG", "np_g_xg"),
        ("Def 3rd", "def_3rd"),
        ("Att Pen", "att_pen"),
        ("Take-Ons", "take_ons"),
        ("1/3", "1_3"),
        ("Per 90 Minutes", "per_90_minutes"),
])
def test_snake(raw, expected):
        assert snake(raw) == expected


def test_percent_survives_flattening():
        """`Cmp` and `Cmp%` must not collapse into the same column name."""
        assert snake("Cmp") != snake("Cmp%")


# --- name matching ---------------------------------------------------------

@pytest.mark.parametrize(("raw", "expected"), [
        ("Sergio Reguilón", "sergio reguilon"),
        ("N'Golo Kanté", "ngolo kante"),
        ("Pierre-Emerick Aubameyang", "pierre emerick aubameyang"),
        ("  Álvaro   Morata ", "alvaro morata"),
        (None, ""),
])
def test_normalise_name(raw, expected):
        assert normalise_name(raw) == expected


@pytest.mark.parametrize(("season", "year"), [
        ("2425", 2025),
        ("2324", 2024),
        ("2024-2025", 2025),
])
def test_season_end_year(season, year):
        assert season_end_year(season) == year


def test_season_midpoint_is_mid_season():
        assert season_midpoint("2425") == pd.Timestamp("2025-01-01")


def test_season_end_year_rejects_nonsense():
        with pytest.raises(ValueError):
                season_end_year("not-a-season")


# --- feature engineering ---------------------------------------------------

def test_parse_fbref_age_handles_years_and_days():
        assert parse_fbref_age("25-123") == pytest.approx(25 + 123 / 365.25)
        assert parse_fbref_age("31") == 31.0
        assert np.isnan(parse_fbref_age(None))
        assert np.isnan(parse_fbref_age("unknown"))


def test_possession_factor_scales_the_right_way():
        """A high-possession team's defenders get scaled UP, not down."""
        factor = possession_factor(pd.Series([35.0, 50.0, 65.0]))
        assert factor.iloc[0] < 1.0   # low possession: plenty of defending to do
        assert factor.iloc[1] == pytest.approx(1.0)
        assert factor.iloc[2] > 1.0   # high possession: fewer chances to tackle


def test_possession_factor_survives_missing_data():
        factor = possession_factor(pd.Series([np.nan, 0.0]))
        assert factor.notna().all()
        assert (factor.between(0.5, 2.0)).all()


@pytest.fixture
def isolated_reports(tmp_path, monkeypatch):
        """Keep resolve_features' report out of the real repo.

        It writes to paths.FEATURE_RESOLUTION as a side effect, so any test
        calling it must redirect first or it leaves a file behind in reports/.
        """
        from src import paths as paths_mod

        monkeypatch.setattr(paths_mod, "REPORTS", tmp_path)
        monkeypatch.setattr(paths_mod, "FEATURE_RESOLUTION", tmp_path / "res.csv")
        return tmp_path


def test_resolve_features_reports_misses(isolated_reports):
        tmp_path = isolated_reports
        resolved, report = resolve_features(["shooting__standard__sh", "irrelevant__flat__x"])

        assert resolved.get("shots") == "shooting__standard__sh"
        assert "interceptions" not in resolved
        assert (report["status"] == "MISSING").any()
        assert (tmp_path / "res.csv").exists()


def test_resolve_features_disambiguates_repeated_stat_names(isolated_reports):
        """`int` appears under both defense__flat and misc__performance.

        Both column names are real - verified against
        data/interim/fbref_player_seasons.parquet - and the catalogue lists
        the defence page first, because that table is the authoritative one
        for defensive actions.
        """
        resolved, _ = resolve_features(
                ["misc__performance__int", "defense__flat__int"]
        )
        assert resolved["interceptions"] == "defense__flat__int"


# --- archived FBref stats --------------------------------------------------

def _live_frame(rows):
        """A tiny live-style FBref table: (team, player, born, 90s, passes)."""
        index = pd.MultiIndex.from_tuples(
                [("ESP-La Liga", "2425", team, player) for team, player, *_ in rows],
                names=["league", "season", "team", "player"],
        )
        return pd.DataFrame(
                {
                        "standard__flat__born": [r[2] for r in rows],
                        "standard__playing_time__90s": [r[3] for r in rows],
                        "passing__total__cmp": pd.array([r[4] for r in rows], dtype="Int64"),
                },
                index=index,
        )


def test_archive_alignment_follows_renamed_clubs_and_players():
        """FBref renamed 'Betis' to 'Real Betis' and some players since the archive."""
        from src.fetch_fbref_archive import align_to_live

        live = _live_frame([
                ("Real Betis", "Isco", 1992, 20.1, None),
                ("Real Betis", "Johnny Cardoso", 2001, 25.3, None),
                ("Real Betis", "Abde Ezzalzouli", 2001, 18.0, None),
                ("Real Betis", "Andy Smith", 1995, 12.4, None),
        ])
        archived = _live_frame([
                ("Betis", "Isco", 1992, 20.1, 900),
                ("Betis", "Johnny Cardoso", 2001, 25.3, 1100),
                ("Betis", "Abde Ezzalzouli", 2001, 18.0, 400),
                ("Betis", "Andrew Smith", 1995, 12.4, 300),
        ]).rename(columns={
                "standard__flat__born": "passing__flat__born",
                "standard__playing_time__90s": "passing__flat__90s",
        })

        aligned = align_to_live(archived, live, "passing")
        assert set(aligned.index) == set(live.index)
        assert aligned.loc[("ESP-La Liga", "2425", "Real Betis", "Andy Smith"), "passing__total__cmp"] == 300


def test_archive_overlay_fills_gaps_without_adding_rows():
        from src.fetch_fbref_archive import overlay

        live = _live_frame([("Arsenal", "A", 2000, 30.0, None), ("Arsenal", "B", 2001, 10.0, 50)])
        archived = _live_frame([("Arsenal", "A", 2000, 30.0, 1200), ("Arsenal", "Ghost", 1999, 5.0, 99)])
        archived = archived[["passing__total__cmp"]].assign(passing__flat__prgp=[40, 1])

        combined = overlay(live, archived)
        assert list(combined.index) == list(live.index)                 # no "Ghost" row
        assert combined.loc[("ESP-La Liga", "2425", "Arsenal", "A"), "passing__total__cmp"] == 1200
        assert combined.loc[("ESP-La Liga", "2425", "Arsenal", "B"), "passing__total__cmp"] == 50
        assert "passing__flat__prgp" in combined.columns                # archive-only stat kept


# --- dated contracts ----------------------------------------------------------

def _obs(player_id, joined, contract_end, observed_on, source):
        return {
                "player_id": player_id,
                "joined": pd.Timestamp(joined) if joined else pd.NaT,
                "contract_end": pd.Timestamp(contract_end),
                "observed_on": pd.Timestamp(observed_on),
                "source": source,
        }


def test_contract_at_midpoint_picks_the_nearest_observation_of_the_same_spell():
        """Midpoint of "2324" is 2024-01-01. Each player tests one rule."""
        from src.fetch_tm_contracts import contract_at_midpoint

        observations = pd.DataFrame([
                # 1: nearest wins (Aug 2023 beats Jul 2024)
                _obs(1, "2020-07-01", "2025-06-30", "2023-08-01", "near"),
                _obs(1, "2020-07-01", "2027-06-30", "2024-07-01", "far"),
                # 2: equal distance either side -> the earlier one (no look-ahead)
                _obs(2, "2020-07-01", "2025-06-30", "2023-12-01", "before"),
                _obs(2, "2020-07-01", "2026-06-30", "2024-02-01", "after"),
                # 3: joined after the midpoint -> a later spell, unusable
                _obs(3, "2024-07-15", "2029-06-30", "2024-08-01", "later_spell"),
                # 4: contract already over at the midpoint -> unusable
                _obs(4, "2019-07-01", "2023-06-30", "2023-05-01", "expired"),
                # 5: undated join, but a transfer between observation and midpoint
                _obs(5, None, "2030-06-30", "2026-07-06", "players.csv"),
                _obs(5, "2021-07-01", "2024-06-30", "2023-07-01", "old_club"),
        ])
        transfers = pd.DataFrame({"player_id": [5], "transfer_date": [pd.Timestamp("2024-07-01")]})
        rows = pd.DataFrame({"tm_player_id": [1, 2, 3, 4, 5, np.nan, 99], "season": ["2324"] * 7},
                            index=[10, 11, 12, 13, 14, 15, 16])

        out = contract_at_midpoint(rows, observations, transfers)

        assert list(out.index) == list(rows.index)
        assert out.loc[10, "contract_source"] == "near"
        assert out.loc[11, "contract_source"] == "before"
        assert pd.isna(out.loc[12, "contract_end"])
        assert pd.isna(out.loc[13, "contract_end"])
        assert out.loc[14, "contract_source"] == "old_club"
        assert out.loc[14, "contract_end"] == pd.Timestamp("2024-06-30")
        assert pd.isna(out.loc[15, "contract_end"])     # no Transfermarkt id
        assert pd.isna(out.loc[16, "contract_end"])     # no observation at all


def test_tm_dates_parse_in_both_scrape_formats():
        from src.fetch_tm_contracts import parse_tm_date

        parsed = parse_tm_date(pd.Series(["Jun 30, 2026", "30/06/2026", "Jul 29, 2001 (21)", "-", None]))
        assert parsed.iloc[0] == parsed.iloc[1] == pd.Timestamp("2026-06-30")
        assert parsed.iloc[2] == pd.Timestamp("2001-07-29")
        assert parsed.iloc[3:].isna().all()


# --- club check on the name join ----------------------------------------------

def test_club_check_swaps_a_namesake_for_the_player_who_was_at_the_club(monkeypatch, tmp_path):
        from src import merge

        monkeypatch.setattr(merge.paths, "MATCH_CLUB_CHECK", tmp_path / "club_check.csv")

        players = pd.DataFrame({
                "player_id": [1, 2, 3, 4, 5, 6, 7],
                "name": ["Igor Vekic", "Igor Julio", "Anna Smith", "Bea Jones", "Cleo Brown", "Dana White", "Eve Black"],
                "birth_year": [1998, 1998, 1995, 1996, 1997, 1994, 1999],
                "sub_position": ["Goalkeeper", "Centre-Back", "Left Winger", "Centre-Forward", "Right-Back", "Centre-Back", "Left-Back"],
        })
        club_seasons = pd.DataFrame({
                "player_id": [2, 3, 4, 5, 6, 1],
                "season_end": [2024] * 6,
                "player_club_id": [10, 10, 10, 10, 10, 99],
                "minutes_played": [900] * 6,
        })
        monkeypatch.setattr(merge, "load_club_seasons", lambda: club_seasons)
        df = pd.DataFrame({
                "league": ["ENG-Premier League"] * 5, "season": ["2324"] * 5, "team": ["Brighton"] * 5,
                "player": ["Igor", "Anna Smith", "Bea Jones", "Cleo Brown", "Dana White"],
                "born": [1998, 1995, 1996, 1997, 1994],
                "tm_player_id": [1.0, 3.0, 4.0, 5.0, 7.0],
                "tm_name": ["Igor Vekic", "Anna Smith", "Bea Jones", "Cleo Brown", "Eve Black"],
                "tm_sub_position": ["Goalkeeper", "Left Winger", "Centre-Forward", "Right-Back", "Left-Back"],
                "match_decision": ["accepted", "accepted", "accepted", "accepted", "override"],
        })

        out = merge.confirm_by_club(df, players)

        assert out.loc[0, "tm_name"] == "Igor Julio"              # namesake swapped
        assert out.loc[0, "tm_sub_position"] == "Centre-Back"
        assert out.loc[0, "club_check"] == "corrected"
        assert (out.loc[1:3, "club_check"] == "confirmed").all()
        assert out.loc[4, "club_check"] == "override"             # never second-guessed
        assert out.loc[4, "tm_name"] == "Eve Black"
        assert "match_decision" not in out.columns
