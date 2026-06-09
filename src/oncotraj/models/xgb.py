"""XGBoost baseline. Per PAPER_OUTLINE.md §6.1: 1000 rounds, early stopping."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor

from .base import OncoTrajModel, register_model


@register_model
class XGBoostModel(OncoTrajModel):
    name = "xgboost"

    def __init__(
        self,
        is_classifier: bool = True,
        n_estimators: int = 1000,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        random_state: int = 42,
        **_: Any,
    ) -> None:
        super().__init__()
        self.is_classifier = is_classifier
        cls = XGBClassifier if is_classifier else XGBRegressor
        kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            tree_method="hist",
            n_jobs=-1,
            verbosity=0,
        )
        if is_classifier:
            kwargs["eval_metric"] = "mlogloss"
        self._model = cls(**kwargs)
        self._label_encoder: LabelEncoder | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> XGBoostModel:
        from sklearn.utils.class_weight import compute_sample_weight

        self.feature_names_ = list(X.columns)
        if self.is_classifier:
            # XGBClassifier requires integer-coded labels; preserve original
            # string classes for downstream metric reporting.
            self._label_encoder = LabelEncoder().fit(y)
            y_enc = self._label_encoder.transform(y)
            # P1: class-weighted training to avoid mechanism-class collapse.
            # XGB doesn't expose `class_weight=`; we pass sample-level weights
            # computed from the inverse class frequencies.
            sw = compute_sample_weight(class_weight="balanced", y=y_enc)
            self._model.fit(X, y_enc, sample_weight=sw)
            self.classes_ = self._label_encoder.classes_
        else:
            self._model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        raw = self._model.predict(X)
        if self.is_classifier and self._label_encoder is not None:
            return self._label_encoder.inverse_transform(raw)
        return raw

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise NotImplementedError("XGBoostModel is in regression mode.")
        self._check_fitted()
        return self._model.predict_proba(X)
