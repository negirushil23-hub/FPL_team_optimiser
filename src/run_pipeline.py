"""End-to-end pipeline: build training data, select+fit a model per
position, predict an upcoming gameweek, and print the optimal team.

Usage:
    python -m src.run_pipeline --gameweek 5 --fixtures-year 2026
"""
from __future__ import annotations

import argparse

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer, StandardScaler

from .data_collection import TrainData
from .feature_engineering import add_derived_features, correlated_features, drop_leaky_columns
from .models import select_best_model
from .optimize_team import get_predictions, print_team, select_best_xi

# Extend this as more historical seasons/gameweeks become available. 
# Kept small here; the original notebook trained on ~37 gameweeks across two seasons.
TRAINING_SOURCES = [
    ("2023-24", range(5, 25)),
    ("2024-25", range(5, 25)),
]

POSITION_CORR_THRESHOLDS = {"GK": 0.06, "DEF": 0.06, "MID": 0.10, "FWD": 0.05}
CANDIDATE_MODELS = [Ridge(), LGBMRegressor()]
CANDIDATE_TRANSFORMERS = [StandardScaler(), PowerTransformer(), None]


def build_training_data() -> pd.DataFrame:
    frames = []
    for season, gws in TRAINING_SOURCES:
        for gw in gws:
            td = TrainData(gw=gw, season=season, form_range=4)
            df = td.df.copy()
            df["gw"] = f"{season}_gw{gw}"  # used later for walk-forward validation
            frames.append(df)
    data = pd.concat(frames)
    data = add_derived_features(data)
    data = drop_leaky_columns(data)
    return data


def train_all_positions(data: pd.DataFrame) -> dict:
    models = {}
    for position, threshold in POSITION_CORR_THRESHOLDS.items():
        pos_data = data[data.position == position].drop(columns=["position", "gw"], errors="ignore")
        features = correlated_features(pos_data, "points_scored", threshold)
        pos_data = pos_data[features + ["points_scored"]]
        models[position] = select_best_model(pos_data, position, CANDIDATE_MODELS, CANDIDATE_TRANSFORMERS)
    return models


def main():
    parser = argparse.ArgumentParser(description="Predict FPL points and suggest a team.")
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--fixtures-year", type=int, default=2026,
                         help="Season start year for fixtures.csv, e.g. 2026 for 2026/27.")
    parser.add_argument("--season", type=str, default="2024-25",
                         help="Season string for pulling this gameweek's player form data.")
    parser.add_argument("--budget", type=float, default=100.0)
    args = parser.parse_args()

    print("Building training data (this pulls gameweek CSVs and may take a minute)...")
    data = build_training_data()

    print("Selecting and fitting best model per position...")
    fitted_models = train_all_positions(data)

    print(f"Predicting gameweek {args.gameweek}...")
    predictions = get_predictions(fitted_models, args.gameweek, args.fixtures_year, args.season)

    print("Solving for optimal XI...")
    team = select_best_xi(predictions, budget=args.budget)
    print_team(team)


if __name__ == "__main__":
    main()
