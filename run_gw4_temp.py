"""
TEMPORARY script: predicts gameweek 4 of the 2026-27 season using a
3-gameweek form window (GW1-3) instead of the usual 4, since GW4 doesn't
have a "gameweek 0" to look back to.

Run fpl_api.py FIRST if you haven't already — it fetches
real GW2/GW3 stats from the official FPL API and caches them locally, since
vaastav's GitHub repo (the usual data source) hadn't published them yet.

Once you're past gameweek 5, go back to using the normal command instead:
    python3 -m src.run_pipeline --gameweek 5 --fixtures-year 2026

You can delete this file at that point, or keep it around for future one-off diagnostics/in case the usual data source is not updated.

Run with:
    python3 run_gw4_temp.py
"""
from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PowerTransformer, StandardScaler

from src.data_collection import PredData, TrainData
from src.feature_engineering import add_derived_features, correlated_features, drop_leaky_columns
from src.models import select_best_model
from src.optimize_team import print_team, select_best_xi

FORM_RANGE = 3               # GW1, GW2, GW3 — the only weeks that exist before GW4
GAMEWEEK_TO_PREDICT = 4
FIXTURES_START_YEAR = 2026   # 2026/27 season
SEASON = "2026-27"           # season string used to pull player gameweek data
BUDGET = 100.0

POSITION_CORR_THRESHOLDS = {"GK": 0.06, "DEF": 0.06, "MID": 0.10, "FWD": 0.05}
CANDIDATE_MODELS = [Ridge(), LGBMRegressor()]
CANDIDATE_TRANSFORMERS = [StandardScaler(), PowerTransformer(), None]

TRAINING_SOURCES = [
    ("2023-24", range(FORM_RANGE + 1, 25)),
    ("2024-25", range(FORM_RANGE + 1, 25)),
]


def build_training_data() -> pd.DataFrame:
    frames = []
    for season, gws in TRAINING_SOURCES:
        for gw in gws:
            df = TrainData(gw=gw, season=season, form_range=FORM_RANGE).df.copy()
            df["gw"] = f"{season}_gw{gw}"
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


def predict_gameweek(fitted_models: dict, gw: int) -> pd.DataFrame:
    pred_source = PredData(gw, FIXTURES_START_YEAR, form_range=FORM_RANGE, season=SEASON).df
    frames = []
    for position, model in fitted_models.items():
        X_pred = pred_source[pred_source.position == position].copy()
        teams = X_pred.team
        values = X_pred.value
        X_pred = add_derived_features(X_pred)
        feature_cols = [c for c in model.data.columns if c != "points_scored"]
        X_pred = X_pred.reindex(columns=feature_cols, fill_value=0).fillna(0)
        prediction = model.pipe.predict(X_pred)
        result = pd.DataFrame(prediction, index=X_pred.index, columns=["xP"])
        result = result.join(values).join(teams)
        result["Pos"] = position
        frames.append(result.reset_index())
    return pd.concat(frames, ignore_index=True)


def main():
    print(f"Building training data with a {FORM_RANGE}-week form window...")
    data = build_training_data()

    print("Selecting and fitting the best model per position...")
    fitted_models = train_all_positions(data)

    print(f"Predicting gameweek {GAMEWEEK_TO_PREDICT}...")
    predictions = predict_gameweek(fitted_models, GAMEWEEK_TO_PREDICT)

    print("Solving for optimal XI...")
    team = select_best_xi(predictions, budget=BUDGET)
    print_team(team)


if __name__ == "__main__":
    main()

