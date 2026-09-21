"""Per-position model fitting, selection, and prediction."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .data_collection import PredData
from .feature_engineering import add_derived_features


@dataclass
class ModelClass:
    """Wraps a (transformer, regressor) pipeline for a single position."""

    data: pd.DataFrame
    position: str
    regressor: object
    transformer: object
    pca: object = None
    pipe: Pipeline | None = field(default=None, init=False)

    def _initialize_pipeline(self) -> Pipeline:
        steps = [("preprocessor", self.transformer)] if self.transformer is not None else []
        if self.pca is not None:
            steps.append(("pca", self.pca))
        steps.append(("regressor", self.regressor))
        return Pipeline(steps=steps)

    def cross_val_score(self, n_splits: int = 10, random_state: int = 42) -> float:
        """Random K-fold MAE. Kept for comparison against walk-forward MAE
        (see evaluate.py): random folds overstate real-world performance
        because nearby gameweeks for the same player are correlated."""
        X = self.data.copy()
        y = X.pop("points_scored")
        X = X.fillna(0)  # safety net: models can't handle stray NaNs
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        pipe = self._initialize_pipeline()
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="neg_mean_absolute_error")
        return -scores.mean()

    def fit_pipe(self) -> None:
        self.pipe = self._initialize_pipeline()
        X = self.data.copy()
        y = X.pop("points_scored")
        X = X.fillna(0)  # safety net: models can't handle stray NaNs
        self.pipe.fit(X, y)

    def predict(self, gw: int, fixtures_start_year: int, season: str = "2024-25") -> pd.DataFrame:
        if self.pipe is None:
            raise RuntimeError("Call fit_pipe() before predict().")
        pred_data = PredData(gw, fixtures_start_year, season=season).df
        X_pred = pred_data[pred_data.position == self.position].copy()
        teams = X_pred.team
        values = X_pred.value
        X_pred = add_derived_features(X_pred)
        feature_cols = [c for c in self.data.columns if c != "points_scored"]
        X_pred = X_pred.reindex(columns=feature_cols, fill_value=0).fillna(0)
        prediction = self.pipe.predict(X_pred)
        result = pd.DataFrame(prediction, index=X_pred.index, columns=["xP"])
        result = result.join(values).join(teams)
        result["Pos"] = self.position
        return result.sort_values("xP", ascending=False)


def select_best_model(data: pd.DataFrame, position: str, models: list, transformers: list) -> ModelClass:
    """Grid-searches {models} x {transformers} on random-KFold MAE and
    returns the ModelClass with the lowest score. See evaluate.py for the
    walk-forward version you should also run before trusting a result.
    """
    best_score = float("inf")
    best_model = None
    for model in models:
        for transformer in transformers:
            model_copy = clone(model)
            transformer_copy = clone(transformer) if transformer is not None else None
            candidate = ModelClass(data, position, model_copy, transformer_copy, None)
            score = candidate.cross_val_score()
            if score < best_score:
                best_score = score
                best_model = candidate
    best_model.fit_pipe()
    print(f"[{position}] best model: {best_model.regressor.__class__.__name__} "
          f"+ {best_model.transformer.__class__.__name__ if best_model.transformer else 'None'} "
          f"(random-CV MAE={best_score:.3f})")
    return best_model
