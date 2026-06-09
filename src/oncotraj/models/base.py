"""Abstract base class and registry for v1 OncoTraj baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd


class OncoTrajModel(ABC):
    """Unified fit / predict / predict_proba interface.

    Subclasses set `is_classifier` at class scope (True for Tasks A and C,
    False for Task B regression).
    """

    name: ClassVar[str] = ""
    is_classifier: ClassVar[bool] = True

    def __init__(self, **kwargs: Any) -> None:
        self._params: dict[str, Any] = dict(kwargs)
        self._fitted: bool = False
        self.feature_names_: list[str] | None = None
        self.classes_: np.ndarray | None = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> OncoTrajModel:
        """Fit the model. Must set `self._fitted = True` and, for classifiers,
        `self.classes_`."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return label predictions (classifiers) or scalar predictions (regressors)."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities for classifiers. Regressors raise."""
        if not self.is_classifier:
            raise NotImplementedError(f"{type(self).__name__} is a regressor; no predict_proba.")
        raise NotImplementedError(f"{type(self).__name__} did not implement predict_proba.")

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> OncoTrajModel:
        return joblib.load(Path(path))

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} has not been fitted.")


MODEL_REGISTRY: dict[str, type[OncoTrajModel]] = {}


def register_model(cls: type[OncoTrajModel]) -> type[OncoTrajModel]:
    if not cls.name:
        raise ValueError(f"{cls.__name__} must set class attribute `name`.")
    if cls.name in MODEL_REGISTRY:
        raise ValueError(f"Duplicate model registration for {cls.name!r}.")
    MODEL_REGISTRY[cls.name] = cls
    return cls


def get_model(name: str, is_classifier: bool = True, **kwargs: Any) -> OncoTrajModel:
    """Resolve a name from MODEL_REGISTRY and instantiate it.

    `is_classifier` switches Task B regression vs Task A/C classification for
    models that support both (RF, XGBoost).
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}. Registered: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](is_classifier=is_classifier, **kwargs)
