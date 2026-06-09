"""Classical-ML baselines for the v1 OncoTraj tasks (PAPER_OUTLINE.md §6.1).

Five locked baselines per the outline; this module implements the four
non-sequence ones — LSTM and Transformer are out of scope here.

Unified interface (see `base.py`):
- `model.fit(X, y)`
- `model.predict(X)`
- `model.predict_proba(X)` (classifiers only; regressors raise NotImplementedError)
- `model.save(path)` / `Model.load(path)` via joblib

Available models:
- `MajorityClassModel` — predicts the modal class (or median) seen in train.
- `LogisticRegressionModel` — sklearn LogisticRegression + StandardScaler.
- `RandomForestModel` — sklearn RandomForest{Classifier,Regressor}.
- `XGBoostModel` — xgboost.XGB{Classifier,Regressor}.
- `CoxModel` (v1.1) — lifelines CoxPHFitter for Task B (time-to-progression).
"""

# Import order matters: xgboost must load before torch on Python 3.14 to
# avoid a SEGV during XGBClassifier.fit when both libraries are present.
# Tracked upstream; do not reorder without re-verifying the test suite.
from .xgb import XGBoostModel  # noqa: I001
from .base import MODEL_REGISTRY, OncoTrajModel, get_model
from .cox import CoxModel, build_cox_target
from .logistic import LogisticRegressionModel
from .lstm import LSTMModel
from .majority import MajorityClassModel
from .random_forest import RandomForestModel
from .transformer import TransformerModel

__all__ = [
    "MODEL_REGISTRY",
    "CoxModel",
    "LSTMModel",
    "LogisticRegressionModel",
    "MajorityClassModel",
    "OncoTrajModel",
    "RandomForestModel",
    "TransformerModel",
    "XGBoostModel",
    "build_cox_target",
    "get_model",
]
