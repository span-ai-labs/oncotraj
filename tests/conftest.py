"""Pytest-wide environment setup.

Two interactions cause SEGVs on macOS arm64 if not pre-empted; both are
addressed here before any test imports the offending libraries.

1. MLflow's telemetry thread crashes Python 3.13/3.14 during interpreter
   teardown when multiple test files invoke `mlflow.start_run`. Setting
   `MLFLOW_DISABLE_TELEMETRY=1` keeps the thread from spawning.

2. xgboost and torch each bundle their own OpenMP runtime; when both
   libraries run multi-threaded in the same process they collide and
   SEGV on arm64. Pinning every thread pool to 1 worker is enough to
   stay clear of the collision. The training scripts override these
   env vars when they want real parallelism, so the test-suite
   constraint doesn't propagate to production runs.
"""

from __future__ import annotations

import os

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
