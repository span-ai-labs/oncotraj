"""End-to-end tests for the standalone eval package.

The end-to-end test trains a real logistic baseline, writes its predictions
to a CSV, drives the eval CLI, and verifies the JSON report + leaderboard
markdown are correctly produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncotraj.eval import (
    EVAL_SCHEMA_VERSION,
    EvalReport,
    evaluate,
    expected_calibration_error,
    reliability_bins,
    task_a_metrics,
    task_b_metrics,
    task_c_metrics,
    update_leaderboard,
)
from oncotraj.eval.cli import main as cli_main
from oncotraj.eval.leaderboard import END_MARKER, START_MARKER

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


# ---------------------------------------------------------------------------
# Pure metric tests
# ---------------------------------------------------------------------------


def test_task_a_metrics_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.05, 0.05, 0.95, 0.95]
    m = task_a_metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["brier"] < 0.01


def test_task_a_metrics_random_baseline():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_prob = np.full(1000, 0.5)
    y_pred = (y_prob > 0.5).astype(int)
    m = task_a_metrics(y_true, y_pred, y_prob)
    # Constant-prediction AUC is undefined; sklearn raises -> we report None.
    # But here y_prob has no variance, AUC -> 0.5 (sklearn returns 0.5).
    assert m["roc_auc"] is None or abs(m["roc_auc"] - 0.5) < 1e-9
    assert abs(m["brier"] - 0.25) < 0.05


def test_expected_calibration_error_zero_when_calibrated():
    """If predicted probability equals empirical frequency in every bin, ECE=0."""
    # 100 samples, half with prob=0.2 → 20% positives; half with prob=0.8 → 80% positives.
    y_prob = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    y_true = np.concatenate([np.array([1] * 20 + [0] * 80), np.array([1] * 80 + [0] * 20)])
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    assert ece < 0.01


def test_reliability_bins_drops_empty_buckets():
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.1, 0.9, 0.9])
    bins = reliability_bins(y_true, y_prob, n_bins=10)
    assert len(bins["bin_centers"]) == 2  # only the 0.1 and 0.9 buckets


def test_task_b_metrics_mae_excludes_censored():
    y_true = [100.0, 200.0, 300.0]
    y_pred = [110.0, 190.0, 290.0]
    event = [1, 1, 0]
    m = task_b_metrics(y_true, y_pred, event)
    # MAE over uncensored only: (10 + 10) / 2 = 10.
    assert abs(m["mae"] - 10.0) < 1e-6
    assert m["n_observed"] == 2
    assert m["censoring_rate"] == 1 / 3


def test_task_b_c_index_perfect_ranking():
    """If predicted ranks are consistent with observed times, C=1.

    Convention (see metrics.harrell_c_index docstring): lower predicted
    value = higher hazard = earlier event. So patient with the earliest
    observed time should have the lowest pred.
    """
    y_true = [100.0, 200.0, 300.0, 400.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]
    event = [1, 1, 1, 1]
    m = task_b_metrics(y_true, y_pred, event)
    assert m["c_index"] == 1.0


def test_task_c_metrics_per_class_and_macro_f1():
    y_true = ["A", "A", "B", "B", "C", "C"]
    y_pred = ["A", "A", "B", "C", "C", "C"]
    m = task_c_metrics(y_true, y_pred)
    assert m["accuracy"] == 5 / 6
    assert m["per_class_f1"]["A"] == 1.0
    assert m["per_class_f1"]["C"] == pytest.approx(0.8, abs=1e-9)


def test_task_c_metrics_with_probabilities():
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "B"]
    y_prob = np.array(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ]
    )
    m = task_c_metrics(y_true, y_pred, y_prob, class_names=["A", "B"])
    assert m["roc_auc_ovr"] == 1.0


# ---------------------------------------------------------------------------
# evaluate() / report serialization
# ---------------------------------------------------------------------------


def _toy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "split": "test",
                "task_a_true": 1,
                "task_a_pred": 1,
                "task_a_prob": 0.9,
                "task_b_true": 200.0,
                "task_b_pred": 190.0,
                "task_b_event": 1,
                "task_c_true": "EGFR_C797S",
                "task_c_pred": "EGFR_C797S",
                "task_c_prob_EGFR_C797S": 0.8,
                "task_c_prob_MET_amplification": 0.2,
            },
            {
                "patient_id": "p2",
                "split": "test",
                "task_a_true": 0,
                "task_a_pred": 0,
                "task_a_prob": 0.1,
                "task_b_true": 400.0,
                "task_b_pred": 380.0,
                "task_b_event": 1,
                "task_c_true": "MET_amplification",
                "task_c_pred": "MET_amplification",
                "task_c_prob_EGFR_C797S": 0.2,
                "task_c_prob_MET_amplification": 0.8,
            },
            {
                "patient_id": "p3",
                "split": "test",
                "task_a_true": 1,
                "task_a_pred": 0,
                "task_a_prob": 0.4,
                "task_b_true": 100.0,
                "task_b_pred": 250.0,
                "task_b_event": 0,
                "task_c_true": "EGFR_C797S",
                "task_c_pred": "MET_amplification",
                "task_c_prob_EGFR_C797S": 0.4,
                "task_c_prob_MET_amplification": 0.6,
            },
            {
                "patient_id": "p4",
                "split": "val",
                "task_a_true": 1,
                "task_a_pred": 1,
                "task_a_prob": 0.7,
                "task_b_true": 150.0,
                "task_b_pred": 160.0,
                "task_b_event": 1,
                "task_c_true": "EGFR_C797S",
                "task_c_pred": "EGFR_C797S",
                "task_c_prob_EGFR_C797S": 0.6,
                "task_c_prob_MET_amplification": 0.4,
            },
        ]
    )


def test_evaluate_filters_to_split_and_reports_all_tasks():
    df = _toy_predictions()
    report = evaluate(df, split="test", submission_id="toy")
    assert report.split == "test"
    assert report.n_predictions == 3
    assert set(report.tasks_evaluated) == {"A", "B", "C"}
    assert report.task_a["accuracy"] == pytest.approx(2 / 3)
    assert report.task_b["n_observed"] == 2  # one censored
    assert report.task_c["accuracy"] == pytest.approx(2 / 3)


def test_evaluate_skips_missing_tasks():
    df = _toy_predictions()[["patient_id", "split", "task_a_true", "task_a_pred"]]
    report = evaluate(df, split="test", submission_id="taskA_only")
    assert report.tasks_evaluated == ["A"]
    assert report.task_b is None
    assert report.task_c is None


def test_evaluate_raises_when_no_task_columns_present():
    df = pd.DataFrame([{"patient_id": "p1", "split": "test"}])
    with pytest.raises(ValueError, match="No task columns"):
        evaluate(df, split="test")


def test_eval_report_roundtrips():
    df = _toy_predictions()
    report = evaluate(df, split="test", submission_id="rt")
    raw = report.to_json()
    parsed = EvalReport.from_json(raw)
    assert parsed.submission_id == "rt"
    assert parsed.schema_version == EVAL_SCHEMA_VERSION
    assert parsed.task_a["accuracy"] == report.task_a["accuracy"]


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def test_update_leaderboard_renders_table_and_preserves_prose(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    leaderboard = tmp_path / "leaderboard.md"

    df = _toy_predictions()
    r1 = evaluate(df, split="test", submission_id="model_a")
    r1.write(reports_dir / "model_a.json")
    # A second submission with worse Task A AUC to verify sorting.
    df2 = df.copy()
    df2["task_a_pred"] = (1 - df2["task_a_pred"]).astype(int)
    df2["task_a_prob"] = 1.0 - df2["task_a_prob"]
    r2 = evaluate(df2, split="test", submission_id="model_b")
    r2.write(reports_dir / "model_b.json")

    update_leaderboard(reports_dir, leaderboard)
    text = leaderboard.read_text()
    assert START_MARKER in text and END_MARKER in text
    assert "| `model_a` |" in text
    assert "| `model_b` |" in text
    # model_a should appear before model_b in the table since it has a higher AUC.
    assert text.index("| `model_a` |") < text.index("| `model_b` |")


def test_update_leaderboard_preserves_user_prose(tmp_path):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    leaderboard = tmp_path / "leaderboard.md"
    leaderboard.write_text(
        f"# Custom Title\n\nMy prose here.\n\n{START_MARKER}\nold table\n{END_MARKER}\n\n"
        f"## Trailing section\nMore prose.\n"
    )
    update_leaderboard(reports_dir, leaderboard)
    text = leaderboard.read_text()
    assert "# Custom Title" in text
    assert "Trailing section" in text
    assert "My prose here." in text
    assert "old table" not in text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_writes_json_and_updates_leaderboard(tmp_path, capsys):
    df = _toy_predictions()
    csv_path = tmp_path / "predictions.csv"
    df.to_csv(csv_path, index=False)

    reports_dir = tmp_path / "reports"
    leaderboard = tmp_path / "leaderboard.md"

    exit_code = cli_main(
        [
            "--predictions",
            str(csv_path),
            "--split",
            "test",
            "--submission-id",
            "cli_test",
            "--reports-dir",
            str(reports_dir),
            "--leaderboard",
            str(leaderboard),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Wrote eval report" in out
    assert "Updated leaderboard" in out
    json_path = reports_dir / "cli_test.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["submission_id"] == "cli_test"
    assert leaderboard.exists()
    assert "| `cli_test` |" in leaderboard.read_text()


def test_cli_refresh_only(tmp_path, capsys):
    """--refresh rebuilds the leaderboard without a new prediction CSV."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    leaderboard = tmp_path / "leaderboard.md"

    df = _toy_predictions()
    r = evaluate(df, split="test", submission_id="seeded")
    r.write(reports_dir / "seeded.json")

    exit_code = cli_main(
        [
            "--refresh",
            "--reports-dir",
            str(reports_dir),
            "--leaderboard",
            str(leaderboard),
        ]
    )
    assert exit_code == 0
    assert "| `seeded` |" in leaderboard.read_text()


# ---------------------------------------------------------------------------
# End-to-end: real baseline -> predictions -> oncotraj-eval
# ---------------------------------------------------------------------------


def test_end_to_end_with_logistic_baseline(tmp_path):
    """Train a logistic baseline, dump its predictions, run the eval CLI."""
    import build_dataset

    from oncotraj.data.splits import SplitManifest, make_splits_from_parquet
    from oncotraj.models import LogisticRegressionModel
    from oncotraj.models.features import (
        BuiltTables,
        build_features,
        build_target_a,
    )

    output = tmp_path / "v0"
    build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=tmp_path / "raw",
        papers_root=None,
    )
    splits_path = tmp_path / "splits.json"
    make_splits_from_parquet(parquet_dir=output, output_path=splits_path, seed=42)

    tables = BuiltTables(
        patients=pd.read_parquet(output / "patients.parquet"),
        variants=pd.read_parquet(output / "variants.parquet"),
        treatments=pd.read_parquet(output / "treatments.parquet"),
        outcomes=pd.read_parquet(output / "outcomes.parquet"),
    )
    X = build_features(tables)
    y_a = build_target_a(tables)
    common = X.index.intersection(y_a.index)
    X = X.loc[common]
    y_a = y_a.loc[common]

    manifest = SplitManifest.from_json(splits_path.read_text())
    train_pids = [p for p in X.index if manifest.assignments.get(p) == "train"]
    test_pids = [p for p in X.index if manifest.assignments.get(p) == "test"]

    model = LogisticRegressionModel(is_classifier=True).fit(X.loc[train_pids], y_a.loc[train_pids])
    test_pred = model.predict(X.loc[test_pids])
    test_proba = model.predict_proba(X.loc[test_pids])[:, list(model.classes_).index(1)]

    pred_df = pd.DataFrame(
        {
            "patient_id": test_pids,
            "split": "test",
            "task_a_true": y_a.loc[test_pids].values,
            "task_a_pred": test_pred.astype(int),
            "task_a_prob": test_proba,
        }
    )
    csv = tmp_path / "predictions.csv"
    pred_df.to_csv(csv, index=False)

    reports_dir = tmp_path / "reports"
    leaderboard = tmp_path / "leaderboard.md"
    exit_code = cli_main(
        [
            "--predictions",
            str(csv),
            "--split",
            "test",
            "--submission-id",
            "logistic_taskA",
            "--reports-dir",
            str(reports_dir),
            "--leaderboard",
            str(leaderboard),
        ]
    )
    assert exit_code == 0
    report = json.loads((reports_dir / "logistic_taskA.json").read_text())
    assert "A" in report["tasks_evaluated"]
    assert report["task_a"]["n"] == len(test_pids)
    assert "| `logistic_taskA` |" in leaderboard.read_text()
