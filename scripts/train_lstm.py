"""Joint-multi-task LSTM training with a small grid search.

Trains one LSTM that has both heads (regression for Task B + classification
for Task C) for each grid point. Selects the best config by validation loss
and persists the best-of-grid models per task.

Grid (override with --hidden-grid / --lr-grid / --epochs):
    hidden_size in {32, 64, 128}
    learning_rate in {1e-3, 1e-4}

MLflow tracking writes to `mlruns/` under experiment `oncotraj-baselines`.

Usage:
    python scripts/train_lstm.py \
        --parquet data/processed/oncotraj_v0 \
        --splits  data/processed/oncotraj_v0/_splits.json \
        --output  models/baselines
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncotraj.data.splits import SplitManifest  # noqa: E402
from oncotraj.models.features import (  # noqa: E402
    BuiltTables,
    build_target_b,
    build_target_c,
)
from oncotraj.models.lstm import (  # noqa: E402
    DEFAULT_EPOCHS,
    LSTMModel,
    best_device,
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


def _evaluate(model: LSTMModel, pids: list[str], y_reg: pd.Series, y_cls: pd.Series) -> dict:
    """Per-split metrics for both heads. Skips patients missing the target."""
    if not pids:
        return {}
    X = pd.DataFrame(index=pids)

    metrics: dict[str, float] = {}

    reg_pids = [p for p in pids if p in y_reg.index]
    if reg_pids:
        model.is_classifier = False
        reg_pred = model.predict(X.loc[reg_pids])
        metrics["mae"] = float(mean_absolute_error(y_reg.loc[reg_pids], reg_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_reg.loc[reg_pids], reg_pred)))

    cls_pids = [p for p in pids if p in y_cls.index]
    if cls_pids:
        model.is_classifier = True
        cls_pred = model.predict(X.loc[cls_pids])
        metrics["accuracy"] = float(accuracy_score(y_cls.loc[cls_pids], cls_pred))
        metrics["macro_f1"] = float(
            f1_score(y_cls.loc[cls_pids], cls_pred, average="macro", zero_division=0)
        )
    return metrics


def grid_search(
    parquet_dir: Path,
    splits_path: Path,
    output_dir: Path,
    hidden_grid: list[int],
    lr_grid: list[float],
    epochs: int,
    mlflow_experiment: str = "oncotraj-baselines",
) -> dict:
    tables = _load_tables(parquet_dir)
    manifest = SplitManifest.from_json(splits_path.read_text())

    y_reg = build_target_b(tables)
    y_cls = build_target_c(tables)
    train_pids = _split_index(manifest, "train")
    val_pids = _split_index(manifest, "val")
    test_pids = _split_index(manifest, "test")

    device = best_device()
    mlflow.set_tracking_uri(f"file://{(REPO_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment(mlflow_experiment)

    grid: list[dict] = []
    best_overall = {"val_combined": float("inf"), "config": None, "result": None}

    output_dir.mkdir(parents=True, exist_ok=True)

    for hidden, lr in itertools.product(hidden_grid, lr_grid):
        run_name = f"lstm_grid_h{hidden}_lr{lr:.0e}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(
                {
                    "model": "lstm",
                    "hidden_size": hidden,
                    "learning_rate": lr,
                    "epochs": epochs,
                    "device": str(device),
                    "n_train": len(train_pids),
                    "n_val": len(val_pids),
                    "n_test": len(test_pids),
                }
            )
            model = LSTMModel(
                hidden_size=hidden,
                learning_rate=lr,
                epochs=epochs,
                variants_df=tables.variants,
                device=device,
            )
            start = time.time()
            model.fit_multitask(
                patient_ids=train_pids,
                variants=tables.variants,
                y_reg=y_reg,
                y_cls=y_cls,
                val_patient_ids=val_pids,
            )
            elapsed = time.time() - start
            mlflow.log_metric("train_time_seconds", elapsed)

            history = getattr(model, "training_history_", [])
            final_val_loss = history[-1]["val_loss"] if history else float("nan")
            mlflow.log_metric("final_val_loss", final_val_loss)

            results = {
                "train": _evaluate(model, train_pids, y_reg, y_cls),
                "val": _evaluate(model, val_pids, y_reg, y_cls),
                "test": _evaluate(model, test_pids, y_reg, y_cls),
            }
            for split_name, m in results.items():
                for k, v in m.items():
                    mlflow.log_metric(f"{split_name}_{k}", v)

            grid_entry = {
                "hidden_size": hidden,
                "learning_rate": lr,
                "elapsed_seconds": elapsed,
                "final_val_loss": final_val_loss,
                "metrics": results,
            }
            grid.append(grid_entry)
            if final_val_loss < best_overall["val_combined"]:
                best_overall = {
                    "val_combined": final_val_loss,
                    "config": {"hidden_size": hidden, "learning_rate": lr},
                    "result": grid_entry,
                }
                # Persist best-so-far in both task modes.
                model.is_classifier = False
                model.save(output_dir / "taskB_lstm.joblib")
                model.is_classifier = True
                model.save(output_dir / "taskC_lstm.joblib")
                mlflow.log_artifact(str(output_dir / "taskB_lstm.joblib"), "model")
                mlflow.log_artifact(str(output_dir / "taskC_lstm.joblib"), "model")

    return {
        "device": str(device),
        "n_grid": len(grid),
        "grid": grid,
        "best": best_overall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "models" / "baselines")
    parser.add_argument("--hidden-grid", nargs="+", type=int, default=[32, 64, 128])
    parser.add_argument("--lr-grid", nargs="+", type=float, default=[1e-3, 1e-4])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--experiment", default="oncotraj-baselines")
    args = parser.parse_args()

    result = grid_search(
        parquet_dir=args.parquet,
        splits_path=args.splits,
        output_dir=args.output,
        hidden_grid=args.hidden_grid,
        lr_grid=args.lr_grid,
        epochs=args.epochs,
        mlflow_experiment=args.experiment,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
