"""Cox proportional-hazards baseline for Task B (time-to-progression).

OncoTraj v1.1 adds Cox PH alongside the classical regression baselines so
that submitters can rank patients by hazard and report a survival-aware
metric (Harrell's C-index) on the same feature set as the regressors.

Design choices
--------------
- **Lifelines `CoxPHFitter`** is the primary fitter. It accepts a static
  feature matrix plus duration / event columns, supports `predict_partial_
  hazard` and `predict_expectation`, and is the standard reference for
  cohort-level survival modelling.
- **"Time-varying" framing.** A full time-varying Cox via
  `lifelines.CoxTimeVaryingFitter` would require a long-format frame with
  one row per (patient, observation interval). In OncoTraj v1, 740 / 813
  patients have a single ctDNA / tissue draw at or before osimertinib
  start, so the long-format trivially collapses to one row per patient
  and the time-varying fitter reduces to the static fit. The temporal
  signal we *do* have — `vaf_slope` and `time_on_therapy_days` — is
  already encoded into static covariates by the leak-safe feature
  pipeline (`oncotraj.models.features.build_features`). We therefore use
  the static fitter on those features. The leakage guarantee is inherited
  from the upstream pipeline (see `tests/test_taska_feature_leakage.py`).
- **Censoring.** Task B's regression target (`build_target_b`) only
  exists for patients with an observed progression event; Cox needs the
  censored patients too. `build_cox_target` below derives (T, E) for
  every patient with an osimertinib start date: T = days to first
  progression if observed, else days to last follow-up; E = 1 if
  observed, else 0.

Outputs
-------
`predict(X)` returns predicted expected-survival times in days
(suitable for MAE evaluation against `build_target_b`). The model also
exposes `predict_partial_hazard(X)` so the eval harness can compute
Harrell's C-index on the full cohort (event + censored).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

from .base import OncoTrajModel, register_model
from .features import BuiltTables


def build_cox_target(tables: BuiltTables) -> pd.DataFrame:
    """Return per-patient survival target with columns ``T`` and ``E``.

    T = days from osimertinib start to first RECIST progression (E=1)
    OR to last follow-up date (E=0, right-censored). Patients without an
    osimertinib start are excluded.
    """
    treatments = tables.treatments.loc[tables.treatments["is_osimertinib"]].copy()
    if treatments.empty:
        return pd.DataFrame(columns=["T", "E"]).astype({"T": float, "E": int})
    treatments["start_date"] = pd.to_datetime(treatments["start_date"])
    start = treatments.groupby("patient_id")["start_date"].min()

    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist",
        ["patient_id", "event_date"],
    ].copy()
    prog["event_date"] = pd.to_datetime(prog["event_date"])
    first_prog = prog.groupby("patient_id")["event_date"].min()

    pats = tables.patients.set_index("patient_id")
    lfu = pd.to_datetime(pats["last_followup_date"]) if "last_followup_date" in pats else None

    rows = {}
    for pid, s in start.items():
        if pid in first_prog.index:
            t = (first_prog.loc[pid] - s).days
            e = 1
        elif lfu is not None and pid in lfu.index and pd.notna(lfu.loc[pid]):
            t = max(0, (lfu.loc[pid] - s).days)
            e = 0
        else:
            continue
        if pd.notna(t) and t >= 0:
            rows[pid] = {"T": float(t), "E": int(e)}
    return pd.DataFrame.from_dict(rows, orient="index")


@register_model
class CoxModel(OncoTrajModel):
    """Cox PH baseline on the leak-safe feature set.

    See module docstring for the time-varying-vs-static choice and its
    justification given v1's longitudinal sparsity.
    """

    name = "cox"

    def __init__(
        self,
        is_classifier: bool = False,
        penalizer: float = 0.05,
        l1_ratio: float = 0.0,
        **_: Any,
    ) -> None:
        super().__init__()
        # Cox is regression-only in our taxonomy (Task B).
        self.is_classifier = False
        self._fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
        self._duration_col = "_T"
        self._event_col = "_E"
        # We need access to the survival target at fit time; the unified
        # interface only hands us (X, y). We attach the target frame via
        # `set_survival_target` before fit. See `fit_survival` for an
        # ergonomic wrapper used by the cross-source script.
        self._surv_target: pd.DataFrame | None = None

    def set_survival_target(self, surv: pd.DataFrame) -> None:
        """Attach the (T, E) DataFrame the next `fit` should use.

        Required before `fit(X, y)`. `y` is kept on the call signature so
        the model still composes with the rest of the harness; its values
        are ignored.
        """
        if not {"T", "E"}.issubset(surv.columns):
            raise ValueError("surv must have columns 'T' and 'E'")
        self._surv_target = surv

    def fit_survival(self, X: pd.DataFrame, surv: pd.DataFrame) -> CoxModel:
        """Ergonomic wrapper: attach survival target then fit."""
        self.set_survival_target(surv)
        # Reuse y as a dummy series aligned to X.
        return self.fit(X, pd.Series(0.0, index=X.index))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> CoxModel:
        if self._surv_target is None:
            raise RuntimeError(
                "CoxModel.fit needs a survival target. Call set_survival_target(...) "
                "or use fit_survival(X, surv)."
            )
        common = X.index.intersection(self._surv_target.index)
        if len(common) < 10:
            raise RuntimeError(
                f"CoxModel needs at least 10 patients with (X, T, E); got {len(common)}."
            )
        df = X.loc[common].copy()
        df[self._duration_col] = self._surv_target.loc[common, "T"].astype(float).values
        df[self._event_col] = self._surv_target.loc[common, "E"].astype(int).values
        # Drop near-zero-variance columns lifelines would reject anyway.
        keep = [
            c for c in X.columns
            if df[c].nunique(dropna=False) > 1
        ]
        df = df[[*keep, self._duration_col, self._event_col]]
        self.feature_names_ = keep
        self._fitter.fit(
            df,
            duration_col=self._duration_col,
            event_col=self._event_col,
            show_progress=False,
            robust=False,
        )
        self._fitted = True
        return self

    def _aligned(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_fitted()
        # Restrict to the features seen at fit time, in the same order.
        # Missing columns get zeros so prediction does not error on minor
        # schema drift (e.g., an egfr_<class> column absent in a sub-cohort).
        out = pd.DataFrame(index=X.index, columns=self.feature_names_, dtype=float)
        for col in self.feature_names_ or []:
            out[col] = X[col].astype(float) if col in X.columns else 0.0
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted expected-survival times in days.

        Suitable for MAE evaluation against `build_target_b` (which is the
        observed time to first progression on the event-observed subset).
        """
        Xa = self._aligned(X)
        expected = self._fitter.predict_expectation(Xa)
        # lifelines returns a Series indexed by row; align to X.index.
        return np.asarray(expected.reindex(X.index).values, dtype=float)

    def predict_partial_hazard(self, X: pd.DataFrame) -> np.ndarray:
        """Hazard ratio per patient; higher = earlier predicted event.

        Used by the harness to compute Harrell's C-index on the full cohort
        (including censored patients).
        """
        Xa = self._aligned(X)
        ph = self._fitter.predict_partial_hazard(Xa)
        return np.asarray(ph.reindex(X.index).values, dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError("CoxModel is a survival regressor; no predict_proba.")
