"""Build EvalReport from a predictions DataFrame."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import task_a_metrics, task_b_metrics, task_c_metrics

EVAL_SCHEMA_VERSION = "oncotraj-eval/1.1.0"


# ---------------------------------------------------------------------------
# Predictions CSV schema
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = ["patient_id", "split"]

TASK_A_REQUIRED = ["task_a_true", "task_a_pred"]
TASK_A_OPTIONAL = ["task_a_prob"]
TASK_B_REQUIRED = ["task_b_true", "task_b_pred"]
TASK_B_OPTIONAL = ["task_b_event"]
TASK_C_REQUIRED = ["task_c_true", "task_c_pred"]
TASK_C_PROB_PREFIX = "task_c_prob_"

SCHEMA_DOC = """\
predictions.csv schema (any task may be omitted):

  patient_id            str   patient identifier matching the OncoTraj cohort
  split                 str   "train" | "val" | "test"
  task_a_true           0/1   binary truth for Task A
  task_a_pred           0/1   binary prediction
  task_a_prob           0..1  P(progression); optional, enables AUC/Brier/ECE
  task_b_true           float days to resistance event
  task_b_pred           float predicted days to resistance event
  task_b_event          0/1   1 if event observed, 0 if censored (optional)
  task_c_true           str   mechanism class label
  task_c_pred           str   predicted mechanism class label
  task_c_prob_<class>   0..1  per-class probabilities (optional, enables OvR AUC)
"""


def _detect_class_prob_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(TASK_C_PROB_PREFIX))


# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    schema_version: str
    submission_id: str
    generated_at_utc: str
    split: str
    n_predictions: int
    n_unique_patients: int
    tasks_evaluated: list[str]
    task_a: dict | None = None
    task_b: dict | None = None
    task_c: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> EvalReport:
        return cls(**json.loads(raw))

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())
        return path


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def evaluate(
    predictions: pd.DataFrame,
    split: str,
    submission_id: str | None = None,
) -> EvalReport:
    """Evaluate a predictions DataFrame on the given split."""
    missing = [c for c in REQUIRED_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions is missing required columns: {missing}")

    sub = predictions.loc[predictions["split"] == split].copy()
    notes: list[str] = []

    tasks_evaluated: list[str] = []
    task_a_report = None
    task_b_report = None
    task_c_report = None

    if all(c in sub.columns for c in TASK_A_REQUIRED):
        df_a = sub.dropna(subset=TASK_A_REQUIRED)
        if not df_a.empty:
            y_prob = (
                df_a["task_a_prob"].astype(float).to_numpy()
                if "task_a_prob" in df_a.columns and df_a["task_a_prob"].notna().any()
                else None
            )
            task_a_report = task_a_metrics(
                df_a["task_a_true"].astype(int).to_numpy(),
                df_a["task_a_pred"].astype(int).to_numpy(),
                y_prob,
            )
            tasks_evaluated.append("A")
        else:
            notes.append("Task A columns present but all rows had NaN; skipped.")

    if all(c in sub.columns for c in TASK_B_REQUIRED):
        df_b = sub.dropna(subset=TASK_B_REQUIRED)
        if not df_b.empty:
            event = (
                df_b["task_b_event"].astype(int).to_numpy()
                if "task_b_event" in df_b.columns and df_b["task_b_event"].notna().any()
                else None
            )
            task_b_report = task_b_metrics(
                df_b["task_b_true"].astype(float).to_numpy(),
                df_b["task_b_pred"].astype(float).to_numpy(),
                event,
            )
            if event is None:
                notes.append(
                    "Task B: no `task_b_event` column; treating every row as uncensored. "
                    "C-index is computed but assumes no censoring."
                )
            tasks_evaluated.append("B")

    if all(c in sub.columns for c in TASK_C_REQUIRED):
        df_c = sub.dropna(subset=TASK_C_REQUIRED)
        if not df_c.empty:
            prob_cols = _detect_class_prob_columns(df_c)
            y_prob_arr: np.ndarray | None = None
            class_names: list[str] | None = None
            if prob_cols:
                class_names = [c[len(TASK_C_PROB_PREFIX) :] for c in prob_cols]
                y_prob_arr = df_c[prob_cols].astype(float).to_numpy()
            task_c_report = task_c_metrics(
                df_c["task_c_true"].astype(str).tolist(),
                df_c["task_c_pred"].astype(str).tolist(),
                y_prob_arr,
                class_names,
            )
            tasks_evaluated.append("C")

    if not tasks_evaluated:
        raise ValueError(
            "No task columns found. Predictions CSV needs at least one of "
            "(task_a_true, task_a_pred), (task_b_true, task_b_pred), "
            "(task_c_true, task_c_pred)."
        )

    return EvalReport(
        schema_version=EVAL_SCHEMA_VERSION,
        submission_id=submission_id or "anonymous",
        generated_at_utc=datetime.now(UTC).isoformat(),
        split=split,
        n_predictions=len(sub),
        n_unique_patients=int(sub["patient_id"].nunique()),
        tasks_evaluated=tasks_evaluated,
        task_a=task_a_report,
        task_b=task_b_report,
        task_c=task_c_report,
        notes=notes,
    )


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load a predictions CSV with permissive dtype handling."""
    return pd.read_csv(path)
