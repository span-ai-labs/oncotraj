"""Standalone evaluation package for OncoTraj v1.

Model-agnostic: takes a `predictions.csv` and emits a JSON evaluation
report + optionally updates a markdown leaderboard. Lives independently
of `oncotraj.models` so external models can be evaluated using the
same harness.

Public API:
    from oncotraj.eval import evaluate, EvalReport, update_leaderboard

CLI:
    oncotraj-eval --predictions predictions.csv --split test \\
                  --output report.json [--leaderboard leaderboard.md]
"""

from .leaderboard import update_leaderboard
from .metrics import (
    expected_calibration_error,
    reliability_bins,
    task_a_metrics,
    task_b_metrics,
    task_c_metrics,
)
from .report import EVAL_SCHEMA_VERSION, EvalReport, evaluate

__all__ = [
    "EVAL_SCHEMA_VERSION",
    "EvalReport",
    "evaluate",
    "expected_calibration_error",
    "reliability_bins",
    "task_a_metrics",
    "task_b_metrics",
    "task_c_metrics",
    "update_leaderboard",
]
