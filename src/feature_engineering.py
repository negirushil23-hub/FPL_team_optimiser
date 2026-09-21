"""Derived features and per-position feature selection helpers."""
from __future__ import annotations

import pandas as pd

# Columns representing "actual outcome" (rather than underlying expected-value) 
# that we deliberately drop from the feature set to reduce leakage
ABSOLUTE_FEATURES_TO_DROP = [
    "total_points", "assists", "goals_scored", "bonus", "clean_sheets",
    "goals_conceded", "own_goals", "transfers_out", "transfers_in",
]


def add_derived_features(data: pd.DataFrame) -> pd.DataFrame:
    """Adds net threat and share of team attack features. Safe to call on
    both training data and single-gameweek prediction data."""
    data = data.copy()
    data["opponent_xG_difference"] = data["opponent_xG"] - data["opponent_xGC"]
    data["team_xG_difference"] = data["team_xG"] - data["team_xGC"]
    data["xG_fraction"] = data["xG"] / data["team_xG"]
    data["xA_fraction"] = data["xA"] / data["team_xG"]
    data["transfer_activity"] = data["transfers_in"] + data["transfers_out"]
    # A team's total xG over the form window can occasionally be 0, which
    # turns the fraction features above into 0/0 = NaN or x/0 = inf. Neither
    # is something the models can handle, so treat "no data yet" as 0 share.
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    for col in ["xG_fraction", "xA_fraction"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
    return data


def drop_leaky_columns(data: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in data.columns if c not in ABSOLUTE_FEATURES_TO_DROP or c == "points_scored"]
    return data[keep]


def correlated_features(data: pd.DataFrame, target: str, threshold: float) -> list[str]:
    """Returns feature names whose absolute correlation with target exceeds
    threshold. Mirrors the original notebook's per-position feature
    selection (thresholds of ~0.05-0.1 were used per position there)."""
    corr = data.corr(numeric_only=True)[target].drop(target)
    return corr[corr.abs() > threshold].index.tolist()
