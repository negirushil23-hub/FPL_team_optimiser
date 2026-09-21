"""Sanity-check unit tests for feature engineering."""
import pandas as pd
import pytest

from src.feature_engineering import add_derived_features, correlated_features, drop_leaky_columns


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "opponent_xG": [1.2, 0.8],
        "opponent_xGC": [1.5, 1.0],
        "team_xG": [1.8, 1.4],
        "team_xGC": [1.0, 1.1],
        "xG": [0.3, 0.1],
        "xA": [0.2, 0.05],
        "transfers_in": [1000, 500],
        "transfers_out": [200, 100],
        "points_scored": [6, 2],
        "goals_scored": [1, 0],
        "assists": [0, 0],
        "bonus": [2, 0],
        "clean_sheets": [0, 0],
        "goals_conceded": [1, 2],
        "own_goals": [0, 0],
        "total_points": [6, 2],
    })


def test_add_derived_features_creates_expected_columns(sample_data):
    result = add_derived_features(sample_data)
    for col in ["opponent_xG_difference", "team_xG_difference", "xG_fraction",
                "xA_fraction", "transfer_activity"]:
        assert col in result.columns
    assert result["transfer_activity"].iloc[0] == 1200
    assert result["opponent_xG_difference"].iloc[0] == pytest.approx(1.2 - 1.5)


def test_drop_leaky_columns_removes_outcome_stats_but_keeps_target(sample_data):
    result = drop_leaky_columns(sample_data)
    assert "goals_scored" not in result.columns
    assert "bonus" not in result.columns
    assert "points_scored" in result.columns


def test_correlated_features_returns_only_strong_correlations(sample_data):
    features = correlated_features(sample_data, "points_scored", threshold=0.5)
    assert isinstance(features, list)
    assert "points_scored" not in features
