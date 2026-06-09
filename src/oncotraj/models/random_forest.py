"""Random forest baseline (sklearn). Per PAPER_OUTLINE.md §6.1: 500 trees, depth 8."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .base import OncoTrajModel, register_model


@register_model
class RandomForestModel(OncoTrajModel):
    name = "random_forest"

    def __init__(
        self,
        is_classifier: bool = True,
        n_estimators: int = 500,
        max_depth: int = 8,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        super().__init__()
        self.is_classifier = is_classifier
        common_kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        if is_classifier:
            self._model = RandomForestClassifier(class_weight="balanced", **common_kwargs)
        else:
            self._model = RandomForestRegressor(**common_kwargs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> RandomForestModel:
        self.feature_names_ = list(X.columns)
        self._model.fit(X, y)
        if self.is_classifier:
            self.classes_ = self._model.classes_
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise NotImplementedError("RandomForestModel is in regression mode.")
        self._check_fitted()
        return self._model.predict_proba(X)
