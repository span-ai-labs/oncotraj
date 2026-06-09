"""Train one baseline on one task and log to MLflow.

Usage:
    python scripts/train_baseline.py \
        --model {majority,logistic,random_forest,xgboost} \
        --task  {A,B,C} \
        --parquet data/processed/oncotraj_v0 \
        --splits  data/processed/oncotraj_v0/_splits.json \
        --output  models/baselines

Tasks:
- A: binary "patient ever progresses" (v1 simplification of the 90-day
  task in the paper outline; absolute prediction times require real
  calendar dates we don't have for FLAURA).
- B: regression on days from osimertinib start to first progression.
- C: classification of dominant resistance mechanism class at progression.

Metrics logged per task:
- A: ROC-AUC, accuracy, Brier score.
- B: MAE, RMSE.
- C: macro F1, weighted F1, accuracy.

MLflow tracking writes to `mlruns/` under the repo root by default.
Each run is logged under the `oncotraj-baselines` experiment.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncotraj.data.splits import SplitManifest  # noqa: E402
from oncotraj.models import get_model  # noqa: E402
from oncotraj.models.features import (  # noqa: E402
    TASK_BUILDERS,
    BuiltTables,
    build_features,
    task_a_cutoff,
)


def _load_tables(parquet_dir: Path) -> BuiltTables:
    return BuiltTables(
        patients=pd.read_parquet(parquet_dir / "patients.parquet"),
        variants=pd.read_parquet(parquet_dir / "variants.parquet"),
        treatments=pd.read_parquet(parquet_dir / "treatments.parquet"),
        outcomes=pd.read_parquet(parquet_dir / "outcomes.parquet"),
    )


def _split_index(manifest: SplitManifest, split: str) -> list[str]:
    return [pid for pid, s in manifest.assignments.items() if s == split]


def _metrics_classification(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None, classes: list
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            if len(classes) == 2:
                pos_idx = list(classes).index(max(classes))
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, pos_idx]))
                metrics["brier"] = float(
                    brier_score_loss((y_true == classes[pos_idx]).astype(int), y_proba[:, pos_idx])
                )
            else:
                metrics["roc_auc_ovr"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", labels=list(classes))
                )
        except ValueError:
            # Single-class val/test slice: AUC/Brier undefined.
            pass
    return metrics


def _metrics_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_one(
    model_name: str,
    task: str,
    parquet_dir: Path,
    splits_path: Path,
    output_dir: Path,
    mlflow_experiment: str = "oncotraj-baselines",
) -> dict:
    if task not in TASK_BUILDERS:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(TASK_BUILDERS)}.")
    task_kind, target_builder = TASK_BUILDERS[task]
    is_classifier = task_kind == "classification"

    tables = _load_tables(parquet_dir)
    manifest = SplitManifest.from_json(splits_path.read_text())

    # Apply leak-safe per-patient cutoff for every task. The prediction time
    # for v1 is osimertinib start, so features must be computed from events
    # on or before that date. See tests/test_taska_feature_leakage.py and
    # paper/STEP12_RESULTS_REVIEW.md P0-1.
    cutoff = task_a_cutoff(tables)
    X = build_features(tables, cutoff_per_patient=cutoff)
    y = target_builder(tables)
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    train_pids = [p for p in _split_index(manifest, "train") if p in common]
    val_pids = [p for p in _split_index(manifest, "val") if p in common]
    test_pids = [p for p in _split_index(manifest, "test") if p in common]

    if not train_pids:
        raise RuntimeError(
            f"Task {task}: no training patients after intersecting with split manifest."
        )

    model = get_model(model_name, is_classifier=is_classifier)
    model.fit(X.loc[train_pids], y.loc[train_pids])

    results: dict[str, dict[str, float]] = {}
    for split_name, pids in (("train", train_pids), ("val", val_pids), ("test", test_pids)):
        if not pids:
            continue
        Xs = X.loc[pids]
        ys = y.loc[pids]
        preds = model.predict(Xs)
        if is_classifier:
            try:
                proba = model.predict_proba(Xs)
            except NotImplementedError:
                proba = None
            classes = list(model.classes_) if model.classes_ is not None else sorted(ys.unique())
            results[split_name] = _metrics_classification(ys, preds, proba, classes)
        else:
            results[split_name] = _metrics_regression(ys, preds)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"task{task}_{model_name}.joblib"
    model.save(model_path)

    # MLflow logging.
    mlflow.set_tracking_uri(f"file://{(REPO_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment(mlflow_experiment)
    run_name = f"task{task}_{model_name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("task", task)
        mlflow.log_param("task_kind", task_kind)
        mlflow.log_param("n_train", len(train_pids))
        mlflow.log_param("n_val", len(val_pids))
        mlflow.log_param("n_test", len(test_pids))
        mlflow.log_param("n_features", X.shape[1])
        for split_name, m in results.items():
            for k, v in m.items():
                mlflow.log_metric(f"{split_name}_{k}", v)
        mlflow.log_artifact(str(model_path), artifact_path="model")

    return {
        "model": model_name,
        "task": task,
        "task_kind": task_kind,
        "n_train": len(train_pids),
        "n_val": len(val_pids),
        "n_test": len(test_pids),
        "n_features": X.shape[1],
        "metrics": results,
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=["majority", "logistic", "random_forest", "xgboost"]
    )
    parser.add_argument("--task", required=True, choices=list(TASK_BUILDERS))
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "baselines")
    parser.add_argument("--experiment", default="oncotraj-baselines")
    args = parser.parse_args()
    result = train_one(
        model_name=args.model,
        task=args.task,
        parquet_dir=args.parquet,
        splits_path=args.splits,
        output_dir=args.output,
        mlflow_experiment=args.experiment,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
