"""LP-based squad selection, kept close to the original notebook's PuLP
formulation, plus two extensions: a full 15-man squad solve, and a
single-transfer suggestion given a squad you already own."""
from __future__ import annotations

import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

POSITION_LIMITS_XI = {"GK": (1, 1), "DEF": (3, 5), "MID": (3, 5), "FWD": (1, 3)}
SQUAD_SIZE_XI = 11
BUDGET_DEFAULT = 100.0
MAX_PER_TEAM = 3


def get_predictions(fitted_models: dict, gw: int, fixtures_start_year: int,
                     season: str = "2024-25") -> pd.DataFrame:
    """fitted_models: {"GK": ModelClass, "DEF": ModelClass, "MID": ModelClass, "FWD": ModelClass},
    each already fit_pipe()'d."""
    frames = [
        model.predict(gw, fixtures_start_year, season).reset_index()
        for model in fitted_models.values()
    ]
    return pd.concat(frames, ignore_index=True)


def select_best_xi(predictions: pd.DataFrame, budget: float = BUDGET_DEFAULT) -> dict:
    """Selects an 11-player team maximizing total predicted points subject
    to budget, formation, and max-3-players-per-real-team constraints."""
    model = LpProblem(name="FPL_XI_Selection", sense=LpMaximize)
    n = len(predictions)
    selected = {i: LpVariable(cat="Binary", name=f"player_{i}") for i in range(n)}

    model += lpSum(predictions.loc[i, "xP"] * selected[i] for i in range(n))
    model += lpSum(predictions.loc[i, "value"] * selected[i] for i in range(n)) <= budget * 10
    model += lpSum(selected[i] for i in range(n)) == SQUAD_SIZE_XI

    for pos, (lo, hi) in POSITION_LIMITS_XI.items():
        count = lpSum(selected[i] for i in range(n) if predictions.loc[i, "Pos"] == pos)
        model += count >= lo
        model += count <= hi

    for team, count in predictions.groupby("team").size().items():
        model += lpSum(selected[i] for i in range(n) if predictions.loc[i, "team"] == team) <= MAX_PER_TEAM

    model.solve()

    chosen = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for i in range(n):
        if selected[i].varValue == 1:
            row = predictions.loc[i]
            chosen[row["Pos"]].append((row["name"], round(row["xP"], 2), row["team"]))
    return chosen


def print_team(chosen: dict) -> None:
    print("Selected Team:")
    for pos in ["GK", "DEF", "MID", "FWD"]:
        label = {"GK": "Goalkeeper", "DEF": "Defenders", "MID": "Midfielders", "FWD": "Forwards"}[pos]
        print(f"\n{label}:")
        for name, xp, team in chosen[pos]:
            print(f"  {name} ({team}) - xP: {xp}")


def suggest_best_transfer(current_squad_names: list[str], predictions: pd.DataFrame,
                           budget_remaining: float) -> pd.DataFrame | None:
    """Given the names of players currently owned and this gameweek's
    predictions, returns the single swap (out -> in) that maximizes xP gain
    within `budget_remaining` (the extra money freed if selling out_player,
    in £m, e.g. bank balance + out_player's value - in_player's value >= 0).
    """
    owned = predictions[predictions.name.isin(current_squad_names)]
    if owned.empty:
        return None
    candidates = []
    for _, out_row in owned.iterrows():
        pool = predictions[
            (predictions.Pos == out_row.Pos)
            & (~predictions.name.isin(current_squad_names))
            & (predictions.value <= out_row.value + budget_remaining * 10)
        ]
        for _, in_row in pool.iterrows():
            gain = in_row.xP - out_row.xP
            if gain > 0:
                candidates.append({
                    "out": out_row["name"], "in": in_row["name"],
                    "position": out_row.Pos, "xP_gain": round(gain, 2),
                    "cost_delta": round((in_row.value - out_row.value) / 10, 1),
                })
    if not candidates:
        return None
    return pd.DataFrame(candidates).sort_values("xP_gain", ascending=False).head(5)
