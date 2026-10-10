"""OncoTraj v1.1 — cross-source Task B (time-to-progression) evaluation.

Trains the Task B regressors and the new Cox PH baseline on MSK-CHORD
*train-split* patients only, then evaluates on three target sets:

1. In-distribution: MSK-CHORD test-split patients.
2. Cross-source: the full GENIE BPC NSCLC cohort.
3. Cross-source: the full FLAURA molecular-resistance supplement.

Per (model, train_src, eval_src) tuple we report:
- MAE (days) on event-observed patients, with 1000-bootstrap 95% CI.
- Harrell's C-index on the full target set (event + censored), with
  1000-bootstrap 95% CI.

Results are dropped as JSON into ``eval_reports_v1/`` under the naming
convention ``v11_<model>_<train_src>_<eval_src>.json``. The leaderboard
script picks up these files via its existing schema-version filter; the
narrative cross-source table is appended to ``leaderboard.md`` separately
(see the v1.1 README delta).

Honesty flags raised by this script
-----------------------------------
- FLAURA's Task B target collapses to a constant value (910 days, the
  FLAURA-arm pseudo-PFS used by the harmonization layer because patient-
  level dates are not in the supplement). MAE there reflects the
  systematic offset between MSK-derived predictions and that constant;
  C-index is undefined because no two patients have different true times.
  We still report MAE so the offset is auditable; we explicitly set
  C-index to ``None`` and surface a ``notes`` entry in the JSON.
- GENIE BPC is small (n=34 total, 14 events). CIs are wide. We do not
  hide that.

Run::

    uv run python scripts/train_v11_cross_source.py \
        --parquet data/processed/oncotraj_v1 \
        --splits  data/processed/oncotraj_v1/_splits.json \
        --reports-dir eval_reports_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncotraj.data.splits import SplitManifest  # noqa: E402
from oncotraj.eval.metrics import bootstrap_ci, harrell_c_index  # noqa: E402
from oncotraj.eval.report import EVAL_SCHEMA_VERSION  # noqa: E402
from oncotraj.models import build_cox_target, get_model  # noqa: E402
from oncotraj.models.features import (  # noqa: E402
    BuiltTables,
    build_features,
    build_target_b,
    task_a_cutoff,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

CROSS_SOURCE_PLAN = [
    # (train_src, eval_src, eval_label) — eval_src=None means "the test split of train_src".
    ("MSK_CHORD", "MSK_CHORD", "in_distribution"),
    ("MSK_CHORD", "GENIE_BPC", "cross_source"),
    ("MSK_CHORD", "FLAURA_SUPP", "cross_source"),
]

REGRESSION_MODELS = ["majority", "logistic", "random_forest", "xgboost", "cox"]


def _load_tables(parquet_dir: Path) -> BuiltTables:
    return BuiltTables(
        patients=pd.read_parquet(parquet_dir / "patients.parquet"),
        variants=pd.read_parquet(parquet_dir / "variants.parquet"),
        treatments=pd.read_parquet(parquet_dir / "treatments.parquet"),
        outcomes=pd.read_parquet(parquet_dir / "outcomes.parquet"),
    )


def _fit_model(model_name: str, X: pd.DataFrame, y_b: pd.Series, surv: pd.DataFrame):
    """Fit a Task B model on (X, y_b). Cox needs the (T, E) surv frame too."""
    model = get_model(model_name, is_classifier=False)
    if model_name == "cox":
        model.fit_survival(X, surv.loc[X.index])
    else:
        # Regressors: fit on event-observed subset where y_b is defined.
        common = X.index.intersection(y_b.index)
        model.fit(X.loc[common], y_b.loc[common])
    return model


def _hazard_of(model_name: str, model, X: pd.DataFrame) -> np.ndarray:
    """Per-patient hazard score (higher = earlier predicted progression).

    For day-predicting regressors we use -days as the hazard. For Cox we
    use the partial hazard.
    """
    if model_name == "cox":
        return np.asarray(model.predict_partial_hazard(X), dtype=float)
    return -np.asarray(model.predict(X), dtype=float)


def _expected_days(model_name: str, model, X: pd.DataFrame) -> np.ndarray:
    """Per-patient predicted survival time in days (for MAE)."""
    return np.asarray(model.predict(X), dtype=float)


def _evaluate(
    model_name: str,
    model,
    X_eval: pd.DataFrame,
    y_b_eval: pd.Series,
    surv_eval: pd.DataFrame,
) -> dict:
    """MAE on event-observed; C-index on full event+censored cohort."""
    notes: list[str] = []
    out: dict = {
        "n_eval_full": len(X_eval),
        "n_eval_events": int(surv_eval.loc[X_eval.index, "E"].sum()) if len(X_eval) else 0,
        "n_eval_censored": int((surv_eval.loc[X_eval.index, "E"] == 0).sum()) if len(X_eval) else 0,
    }

    # --- MAE on event-observed subset ---
    obs_idx = X_eval.index.intersection(y_b_eval.index)
    if len(obs_idx) >= 2:
        y_true = y_b_eval.loc[obs_idx].to_numpy(dtype=float)
        y_pred = _expected_days(model_name, model, X_eval.loc[obs_idx])
        out["mae"] = float(mean_absolute_error(y_true, y_pred))
        ci = bootstrap_ci(mean_absolute_error, y_true, y_pred)
        out["mae_ci"] = list(ci) if ci else None
        if y_b_eval.loc[obs_idx].nunique() <= 1:
            notes.append(
                "MAE reported but target is constant on this cohort "
                "(harmonization-time pseudo-PFS); the value reflects a "
                "systematic offset, not predictive accuracy."
            )
    else:
        out["mae"] = None
        out["mae_ci"] = None

    # --- C-index on full event+censored cohort ---
    surv_x = surv_eval.loc[X_eval.index]
    times = surv_x["T"].to_numpy(dtype=float)
    events = surv_x["E"].to_numpy(dtype=int)
    if len(X_eval) >= 2 and len(set(times)) >= 2 and events.sum() >= 1:
        hazards = _hazard_of(model_name, model, X_eval)
        # harrell_c_index treats lower `pred` as higher hazard, so it
        # already lines up for regressors-as-days. For Cox we passed
        # partial hazard (higher = earlier), so we negate to match the
        # function's convention.
        pred_for_c = (
            -hazards if model_name == "cox" else hazards  # already -days for regressors
        )
        out["c_index"] = float(harrell_c_index(times=times, preds=pred_for_c, events=events))
        ci_c = bootstrap_ci(
            lambda T, P, E: harrell_c_index(T, P, E) if len(set(T)) > 1 else float("nan"),
            times,
            pred_for_c,
            events,
        )
        out["c_index_ci"] = list(ci_c) if ci_c else None
    else:
        out["c_index"] = None
        out["c_index_ci"] = None
        notes.append(
            "C-index undefined: target has no inter-patient variation (no comparable pairs)."
        )

    out["notes"] = notes
    return out


def run(parquet_dir: Path, splits_path: Path, reports_dir: Path) -> dict:
    tables = _load_tables(parquet_dir)
    manifest = SplitManifest.from_json(splits_path.read_text())

    cutoff = task_a_cutoff(tables)
    X = build_features(tables, cutoff_per_patient=cutoff)
    y_b = build_target_b(tables)
    surv = build_cox_target(tables)
    src = tables.patients.set_index("patient_id")["source_dataset"]

    assignment = pd.Series(manifest.assignments)

    reports_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    summary: list[dict] = []

    for train_src, eval_src, eval_label in CROSS_SOURCE_PLAN:
        # Training set: train-split patients with source == train_src.
        train_split_idx = assignment[assignment == "train"].index
        train_src_mask = src.reindex(train_split_idx) == train_src
        train_pool = train_split_idx[train_src_mask.fillna(False).to_numpy()]
        train_pool = train_pool.intersection(X.index).intersection(surv.index)
        # Eval set: depends on label.
        if eval_label == "in_distribution":
            test_idx = assignment[assignment == "test"].index
            test_src_mask = src.reindex(test_idx) == eval_src
            eval_pool = test_idx[test_src_mask.fillna(False).to_numpy()]
        else:
            # Cross-source: ENTIRE target cohort (none of these patients
            # were seen at training time since we restrict train to
            # train-split MSK-CHORD).
            eval_pool = src[src == eval_src].index
        eval_pool = eval_pool.intersection(X.index).intersection(surv.index)
        if len(train_pool) < 10 or len(eval_pool) < 2:
            print(
                f"[skip] train={train_src} -> eval={eval_src} ({eval_label}): "
                f"|train|={len(train_pool)}, |eval|={len(eval_pool)}"
            )
            continue

        for model_name in REGRESSION_MODELS:
            print(
                f"[fit ] {model_name:14s} train={train_src} ({len(train_pool):3d}) "
                f"-> eval={eval_src} ({eval_label}, {len(eval_pool):3d})",
                flush=True,
            )
            try:
                model = _fit_model(model_name, X.loc[train_pool], y_b, surv)
                metrics = _evaluate(
                    model_name=model_name,
                    model=model,
                    X_eval=X.loc[eval_pool],
                    y_b_eval=y_b,
                    surv_eval=surv,
                )
            except Exception as exc:
                print(f"  ! failed: {exc}")
                continue

            submission_id = f"v11_{model_name}_{train_src.lower()}_{eval_src.lower()}"
            report = {
                "schema_version": EVAL_SCHEMA_VERSION,
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "submission_id": submission_id,
                "split": "v11_cross_source",
                "n_predictions": int(metrics["n_eval_full"]),
                "n_unique_patients": int(metrics["n_eval_full"]),
                "v11_meta": {
                    "train_source": train_src,
                    "train_n": len(train_pool),
                    "eval_source": eval_src,
                    "eval_label": eval_label,
                    "model": model_name,
                },
                "task_a": None,
                "task_b": {
                    "n": int(metrics["n_eval_full"]),
                    "n_observed": int(metrics["n_eval_events"]),
                    "n_censored": int(metrics["n_eval_censored"]),
                    "mae": metrics["mae"],
                    "mae_ci": metrics["mae_ci"],
                    "c_index": metrics["c_index"],
                    "c_index_ci": metrics["c_index_ci"],
                    "censoring_rate": (
                        metrics["n_eval_censored"] / metrics["n_eval_full"]
                        if metrics["n_eval_full"]
                        else None
                    ),
                },
                "task_c": None,
                "tasks_evaluated": ["B"],
                "notes": metrics["notes"],
            }
            out_path = reports_dir / f"{submission_id}.json"
            out_path.write_text(json.dumps(report, indent=2) + "\n")
            written.append(str(out_path.relative_to(REPO_ROOT)))
            summary.append(
                {
                    "model": model_name,
                    "train_src": train_src,
                    "eval_src": eval_src,
                    "label": eval_label,
                    "n_eval": metrics["n_eval_full"],
                    "n_obs": metrics["n_eval_events"],
                    "mae": metrics["mae"],
                    "mae_ci": metrics["mae_ci"],
                    "c_index": metrics["c_index"],
                    "c_index_ci": metrics["c_index_ci"],
                }
            )

    return {"reports_written": written, "summary": summary}


def _fmt_ci(ci: list[float] | None, digits: int = 0) -> str:
    if ci is None:
        return "—"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def _fmt_val(v: float | None, digits: int = 0) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _print_summary(summary: list[dict]) -> None:
    if not summary:
        return
    print("\n=== v1.1 cross-source Task B summary ===")
    header = (
        f"{'model':14s} {'train':10s} {'eval':10s} {'label':17s} "
        f"{'n':4s} {'obs':4s} {'MAE [95% CI]':>26s} {'C-idx [95% CI]':>26s}"
    )
    print(header)
    print("-" * len(header))
    for row in summary:
        mae_s = (
            f"{_fmt_val(row['mae'], 0):>5}d {_fmt_ci(row['mae_ci'], 0):>20}"
            if row["mae"] is not None
            else f"{'—':>26}"
        )
        c_s = (
            f"{_fmt_val(row['c_index'], 3):>5} {_fmt_ci(row['c_index_ci'], 3):>20}"
            if row["c_index"] is not None
            else f"{'—':>26}"
        )
        print(
            f"{row['model']:14s} {row['train_src']:10s} {row['eval_src']:10s} "
            f"{row['label']:17s} {row['n_eval']:>4d} {row['n_obs']:>4d} {mae_s} {c_s}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet", type=Path, default=REPO_ROOT / "data" / "processed" / "oncotraj_v1"
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "oncotraj_v1" / "_splits.json",
    )
    parser.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "eval_reports_v1")
    args = parser.parse_args()
    result = run(args.parquet, args.splits, args.reports_dir)
    _print_summary(result["summary"])
    print(f"\nWrote {len(result['reports_written'])} report(s) to {args.reports_dir}.")


if __name__ == "__main__":
    main()
