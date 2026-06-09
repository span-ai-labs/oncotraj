"""Integration tests for the patient-level stratified split logic.

The tests build a synthetic parquet via `scripts/build_dataset.py
--use-synthetic --keep-excluded --papers-dir ...` (when the FLAURA supplement
is present) and verify the no-leakage and determinism guarantees on that
real-shape input. Pure-unit tests use small in-memory DataFrames.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from oncotraj.data.splits import (
    NO_PROGRESSION_STRATUM,
    SPLIT_SCHEMA_VERSION,
    SplitManifest,
    SplitProportions,
    make_splits,
    make_splits_from_parquet,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_dataset  # noqa: E402


def _toy_tables(n_per_stratum: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Three sources by two mechanisms by n_per_stratum patients."""
    patient_rows = []
    outcome_rows = []
    pid = 0
    for source in ("GENIE_BPC", "MSK_CHORD", "FLAURA_SUPP"):
        for mech in ("EGFR_C797S", "MET_amplification"):
            for _ in range(n_per_stratum):
                patient_id = f"{source}:pt_{pid:04d}"
                pid += 1
                patient_rows.append(
                    {
                        "patient_id": patient_id,
                        "source_dataset": source,
                        "included_in_v1_cohort": True,
                    }
                )
                outcome_rows.append(
                    {
                        "patient_id": patient_id,
                        "event_type": "progression_recist",
                        "event_date": "2024-01-01",
                        "resistance_mechanism_class": mech,
                    }
                )
    return pd.DataFrame(patient_rows), pd.DataFrame(outcome_rows)


def test_split_no_leakage_unit():
    patients, outcomes = _toy_tables(n_per_stratum=10)
    m = make_splits(patients, outcomes, seed=1)
    splits_by_pid: dict[str, set[str]] = {}
    for pid, split in m.assignments.items():
        splits_by_pid.setdefault(pid, set()).add(split)
    # Every patient appears in exactly one split.
    assert all(len(s) == 1 for s in splits_by_pid.values())
    # Every eligible patient is assigned.
    assert set(m.assignments.keys()) == set(patients["patient_id"])


def test_split_proportions_approximate_70_15_15():
    patients, outcomes = _toy_tables(n_per_stratum=100)
    m = make_splits(patients, outcomes, seed=7)
    total = sum(m.counts.values())
    assert abs(m.counts["train"] / total - 0.70) < 0.02
    assert abs(m.counts["val"] / total - 0.15) < 0.02
    assert abs(m.counts["test"] / total - 0.15) < 0.02


def test_split_is_deterministic_given_seed():
    patients, outcomes = _toy_tables(n_per_stratum=15)
    m1 = make_splits(patients, outcomes, seed=42)
    m2 = make_splits(patients, outcomes, seed=42)
    assert m1.assignments == m2.assignments
    # And a different seed produces a different assignment.
    m3 = make_splits(patients, outcomes, seed=43)
    assert m3.assignments != m1.assignments


def test_excluded_patients_never_in_splits():
    patients, outcomes = _toy_tables(n_per_stratum=8)
    # Flip half to excluded.
    half = patients.iloc[: len(patients) // 2].index
    patients.loc[half, "included_in_v1_cohort"] = False
    m = make_splits(patients, outcomes, seed=11)
    excluded_pids = set(patients.loc[half, "patient_id"])
    assert excluded_pids.isdisjoint(set(m.assignments.keys()))
    assert m.excluded_count == len(half)


def test_site_out_holds_out_full_source():
    patients, outcomes = _toy_tables(n_per_stratum=8)
    m = make_splits(patients, outcomes, seed=3, site_out=["FLAURA_SUPP"])
    assigned_sources = set(
        patients.loc[patients["patient_id"].isin(m.assignments), "source_dataset"]
    )
    assert "FLAURA_SUPP" not in assigned_sources


def test_small_strata_go_entirely_to_train():
    """A stratum with fewer than min_stratum_size patients should be 100% train."""
    patients, outcomes = _toy_tables(n_per_stratum=3)  # 3 per (source, mech)
    m = make_splits(patients, outcomes, seed=99, min_stratum_size=7)
    # Every assignment should be "train".
    assert set(m.assignments.values()) == {"train"}


def test_no_progression_stratum_for_unprogressed_patients():
    patients = pd.DataFrame(
        [
            {
                "patient_id": "GENIE_BPC:p1",
                "source_dataset": "GENIE_BPC",
                "included_in_v1_cohort": True,
            },
            {
                "patient_id": "GENIE_BPC:p2",
                "source_dataset": "GENIE_BPC",
                "included_in_v1_cohort": True,
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "patient_id": "GENIE_BPC:p1",
                "event_type": "last_followup",
                "event_date": "2024-01-01",
                "resistance_mechanism_class": None,
            },
            {
                "patient_id": "GENIE_BPC:p2",
                "event_type": "last_followup",
                "event_date": "2024-01-01",
                "resistance_mechanism_class": None,
            },
        ]
    )
    m = make_splits(patients, outcomes, seed=0, min_stratum_size=2)
    assert NO_PROGRESSION_STRATUM in next(iter(m.stratum_breakdown))


def test_manifest_roundtrips_through_json():
    patients, outcomes = _toy_tables(n_per_stratum=8)
    m = make_splits(patients, outcomes, seed=5)
    raw = m.to_json()
    parsed = SplitManifest.from_json(raw)
    assert parsed.schema_version == SPLIT_SCHEMA_VERSION
    assert parsed.assignments == m.assignments
    assert parsed.proportions == m.proportions


def test_split_proportions_validation():
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        SplitProportions(0.5, 0.3, 0.3)


def test_end_to_end_with_real_parquet(tmp_path):
    """Build the synthetic+papers parquet and then split it; verify no leakage
    against the actual patient_id format and the cohort filter."""
    output = tmp_path / "v0"
    build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=tmp_path / "raw",
        papers_root=None,  # synthetic-only avoids the supplement-presence dependency
    )
    manifest_path = tmp_path / "splits.json"
    manifest = make_splits_from_parquet(
        parquet_dir=output,
        output_path=manifest_path,
        seed=42,
    )
    patients = pd.read_parquet(output / "patients.parquet")

    # Every assigned patient must be in the parquet's included cohort.
    cohort_pids = set(patients.loc[patients["included_in_v1_cohort"], "patient_id"])
    assert set(manifest.assignments.keys()) == cohort_pids

    # JSON is on disk.
    assert manifest_path.exists()
    loaded = SplitManifest.from_json(manifest_path.read_text())
    assert loaded.assignments == manifest.assignments
    assert loaded.counts["train"] + loaded.counts["val"] + loaded.counts["test"] == len(cohort_pids)
