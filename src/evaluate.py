"""Walk-forward validation and interpretability helpers.

Random K-fold CV (as used in the original notebook) shuffles rows from all
gameweeks together, so a model can effectively "see the future" relative to
what it's being validated on, since a player's rows in nearby gameweeks are
correlated. walk_forward_mae fixes this by training only on gameweeks that
happened before the validation gameweek, closer to how the model would
actually be used week to week.
"""
from __future__ import annotations
from asyncio.windows_utils import pipe

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error


def walk_forward_mae(data: pd.DataFrame, pipe, gw_column: str = "gw",
                      min_train_gws: int = 5) -> tuple[float, pd.DataFrame]:
    """data must include a `gw_column` identifying which gameweek each row
    came from."""
    gameweeks = sorted(data[gw_column].unique())
    records = []
    for gw in gameweeks[min_train_gws:]:
        train = data[data[gw_column] < gw].drop(columns=[gw_column])
        val = data[data[gw_column] == gw].drop(columns=[gw_column])
        if len(train) == 0 or len(val) == 0:
            continue
        X_train, y_train = train.drop(columns=["points_scored"]), train["points_scored"]
        X_val, y_val = val.drop(columns=["points_scored"]), val["points_scored"]
        model = clone(pipe)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        records.append({"gw": gw, "mae": mean_absolute_error(y_val, preds), "n_val": len(val)})
    breakdown = pd.DataFrame(records)
    overall_mae = float(np.average(breakdown["mae"], weights=breakdown["n_val"]))
    return overall_mae, breakdown


def permutation_importance_report(pipe, X: pd.DataFrame, y: pd.Series, n_repeats: int = 20):
    """Model-agnostic feature importance: works for both the linear and
    tree-based candidates used in this project. Use shap.TreeExplainer
    instead for a finer-grained view on the LightGBM/XGBoost models."""
    from sklearn.inspection import permutation_importance
    result = permutation_importance(pipe, X, y, n_repeats=n_repeats,
                                     scoring="neg_mean_absolute_error", random_state=42)
    importances = pd.Series(result.importances_mean, index=X.columns)
    return importances.sort_values(ascending=False)

def shap_summary(pipe, X: pd.DataFrame):
    """Returns a SHAP explainer + values for the fitted tree-based regressor
    inside `pipe`."""
    import shap
    regressor = pipe.named_steps["regressor"]
    # Apply any preceding preprocessing steps so SHAP sees the same features
    # the regressor was actually trained on.
    X_transformed = X
    for name, step in pipe.steps[:-1]:
        X_transformed = step.transform(X_transformed)
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer(X_transformed)
    return explainer, shap_values
