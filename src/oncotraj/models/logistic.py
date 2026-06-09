"""Logistic regression on hand-crafted features (Task A/C) or linear
regression (Task B). Standard scaler in front."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import OncoTrajModel, register_model


@register_model
class LogisticRegressionModel(OncoTrajModel):
    name = "logistic"

    def __init__(
        self,
        is_classifier: bool = True,
        C: float = 1.0,
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        **_: Any,
    ) -> None:
        super().__init__()
        self.is_classifier = is_classifier
        if is_classifier:
            self._pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight),
                    ),
                ]
            )
        else:
            self._pipeline = Pipeline(
                steps=[("scaler", StandardScaler()), ("reg", LinearRegression())]
            )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> LogisticRegressionModel:
        self.feature_names_ = list(X.columns)
        self._pipeline.fit(X, y)
        if self.is_classifier:
            self.classes_ = self._pipeline.named_steps["clf"].classes_
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self._pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise NotImplementedError("LogisticRegressionModel is in regression mode.")
        self._check_fitted()
        return self._pipeline.predict_proba(X)
