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
