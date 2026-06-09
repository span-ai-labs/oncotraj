"""Majority-class / median-baseline model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .base import OncoTrajModel, register_model


@register_model
class MajorityClassModel(OncoTrajModel):
    """Predicts the modal class (classification) or median value (regression).

    For classification, `predict_proba` returns the train-set class frequencies
    broadcast to every row — gives a sensible baseline AUC of 0.5 and a
    calibrated probability that matches the prior.
    """

    name = "majority"

    def __init__(self, is_classifier: bool = True, **_: Any) -> None:
        super().__init__()
        self.is_classifier = is_classifier
        self._prediction: float | str | int | None = None
        self._proba_vector: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MajorityClassModel:
        self.feature_names_ = list(X.columns)
        if self.is_classifier:
            counts = y.value_counts()
            self._prediction = counts.idxmax()
            self.classes_ = np.array(sorted(counts.index))
            freqs = counts.reindex(self.classes_).fillna(0).to_numpy(dtype=float)
            self._proba_vector = freqs / freqs.sum()
        else:
            self._prediction = float(y.median())
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return np.array([self._prediction] * len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise NotImplementedError("MajorityClassModel is in regression mode.")
        self._check_fitted()
        return np.tile(self._proba_vector, (len(X), 1))
