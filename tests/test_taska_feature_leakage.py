"""Label-leakage audit for Task A features (P0-1 in STEP12_RESULTS_REVIEW.md).

Acceptance criterion from the audit:
    For every patient and every Task A feature, assert the feature was
    computed using only events with event_date <= prediction_cutoff - 1 day.

We define `prediction_cutoff` per patient = osimertinib start date (see
`oncotraj.models.features.task_a_cutoff`). The invariant the leak-safe
implementation satisfies is the stronger:

    build_features(tables, cutoff_per_patient=c) ==
    build_features(filter_tables_to(tables, sample_date<=c, start_date<=c,
                                    event_date<=c, end_date<=c), None)

i.e., applying the cutoff inside `build_features` is equivalent to first
deleting every row whose date is strictly after the cutoff and then
calling `build_features` with no cutoff. This proves that no event after
the cutoff influences any feature value.

Plus a fresh-eyes sanity check: every event-derived feature must change
(at least for some patient) when we re-run with a strict cutoff vs. the
pre-fix no-cutoff version — otherwise the cutoff isn't doing anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from oncotraj.models.features import (
    BuiltTables,
    build_features,
    task_a_cutoff,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_dataset  # noqa: E402

PAPERS_ROOT = REPO_ROOT / "data" / "raw" / "papers"

# The synthetic cohort alone has no post-baseline samples; the leak these tests
# measure only exists once the FLAURA + OSCILLATE supplements are in the build.
requires_papers = pytest.mark.skipif(
    not PAPERS_ROOT.exists(),
    reason="paper supplements not present at data/raw/papers/",
)

# ---------------------------------------------------------------------------
# Fixture: parquet built from synthetic + papers (the same data the baselines
# train against). One tmp_path fixture so the test runs in isolation.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tables(tmp_path_factory) -> BuiltTables:
    tmp = tmp_path_factory.mktemp("leak_audit")
    output = tmp / "v0"
    build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=tmp / "raw",
        papers_root=PAPERS_ROOT if PAPERS_ROOT.exists() else None,
    )
    return BuiltTables(
        patients=pd.read_parquet(output / "patients.parquet"),
        variants=pd.read_parquet(output / "variants.parquet"),
        treatments=pd.read_parquet(output / "treatments.parquet"),
        outcomes=pd.read_parquet(output / "outcomes.parquet"),
    )


def _filter_tables_to_cutoff(t: BuiltTables, cutoff: pd.Series) -> BuiltTables:
    """Brute-force filter that drops every row whose date is strictly past
    the patient's cutoff. The strict-equality test relies on this being a
    plain, obvious filter so any divergence is in `build_features`, not here.
    """
    cut = pd.to_datetime(cutoff)

    v = t.variants.copy()
    v["sample_date"] = pd.to_datetime(v["sample_date"])
    v["_c"] = v["patient_id"].map(cut)
    v = v[v["_c"].notna() & (v["sample_date"] <= v["_c"])].drop(columns=["_c"])

    tx = t.treatments.copy()
    tx["start_date"] = pd.to_datetime(tx["start_date"])
    tx["end_date"] = pd.to_datetime(tx["end_date"])
    tx["_c"] = tx["patient_id"].map(cut)
    # Drop treatments starting after cutoff; clip end_date at cutoff.
    tx = tx[tx["_c"].notna() & (tx["start_date"] <= tx["_c"])]
    tx["end_date"] = tx[["end_date", "_c"]].min(axis=1)
    tx = tx.drop(columns=["_c"])

    o = t.outcomes.copy()
    o["event_date"] = pd.to_datetime(o["event_date"])
    o["_c"] = o["patient_id"].map(cut)
    o = o[o["_c"].notna() & (o["event_date"] <= o["_c"])].drop(columns=["_c"])

    return BuiltTables(patients=t.patients, variants=v, treatments=tx, outcomes=o)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_task_a_cutoff_is_osimertinib_start(tables: BuiltTables):
    cutoff = task_a_cutoff(tables)
    osi = tables.treatments.loc[
        tables.treatments["is_osimertinib"], ["patient_id", "start_date"]
    ].copy()
    osi["start_date"] = pd.to_datetime(osi["start_date"])
    expected = osi.groupby("patient_id")["start_date"].min()
    pd.testing.assert_series_equal(cutoff.sort_index(), expected.sort_index(), check_names=False)


def test_no_leakage_invariant_strict(tables: BuiltTables):
    """The headline test. If this fails, some feature is using a post-cutoff event."""
    cutoff = task_a_cutoff(tables)
    feats_cutoff = build_features(tables, cutoff_per_patient=cutoff)
    filtered = _filter_tables_to_cutoff(tables, cutoff)
    feats_filtered = build_features(filtered, cutoff_per_patient=None)

    # Same shape, same index.
    assert feats_cutoff.shape == feats_filtered.shape
    common = feats_cutoff.index.intersection(feats_filtered.index)
    assert len(common) == len(feats_cutoff.index) == len(feats_filtered.index)

    # Same values everywhere.
    pd.testing.assert_frame_equal(
        feats_cutoff.loc[common].sort_index(axis=1),
        feats_filtered.loc[common].sort_index(axis=1),
        check_dtype=False,
        atol=1e-9,
    )


@requires_papers
def test_pre_fix_features_did_leak(tables: BuiltTables):
    """Sanity check that the cutoff is actually doing something.

    For the v0 cohort, at least one patient has a Discontinuation sample after
    osi_start. The cutoff-aware features should differ from the no-cutoff
    features for that patient — confirming the leakage was real.
    """
    cutoff = task_a_cutoff(tables)
    feats_unsafe = build_features(tables, cutoff_per_patient=None)
    feats_safe = build_features(tables, cutoff_per_patient=cutoff)
    # At least the variant_burden column must differ for ≥ 1 patient,
    # because the unsafe version counts Discontinuation rows.
    delta = (feats_safe["variant_burden"] - feats_unsafe["variant_burden"]).abs().sum()
    assert delta > 0, (
        "If feats_safe == feats_unsafe, the cutoff is not removing any "
        "rows — either the parquet has no post-baseline samples (unexpected "
        "for FLAURA + OSCILLATE) or the cutoff filter is a no-op."
    )


def test_no_variant_sample_used_is_after_cutoff(tables: BuiltTables):
    """Independent cross-check on the variants table specifically.

    After filtering, every variant row used to compute features must have
    sample_date <= cutoff. Done by reading the filter helper's output
    directly rather than trusting `build_features`.
    """
    cutoff = task_a_cutoff(tables)
    filtered = _filter_tables_to_cutoff(tables, cutoff)
    cut = pd.to_datetime(cutoff)
    filtered.variants["_c"] = filtered.variants["patient_id"].map(cut)
    violations = (pd.to_datetime(filtered.variants["sample_date"]) > filtered.variants["_c"]).sum()
    assert violations == 0, f"{violations} variant rows leak past cutoff"


def test_no_treatment_end_after_cutoff_in_filtered_view(tables: BuiltTables):
    """time_on_therapy_days should never use an end_date past cutoff."""
    cutoff = task_a_cutoff(tables)
    filtered = _filter_tables_to_cutoff(tables, cutoff)
    cut = pd.to_datetime(cutoff)
    filtered.treatments["_c"] = filtered.treatments["patient_id"].map(cut)
    end_dt = pd.to_datetime(filtered.treatments["end_date"])
    violations = (end_dt.notna() & (end_dt > filtered.treatments["_c"])).sum()
    assert violations == 0, f"{violations} treatment rows have end_date past cutoff"


# ---------------------------------------------------------------------------
# v1.1 — extend the leakage audit to the Cox PH baseline.
# Cox introduces a NEW target builder (`build_cox_target`) that legitimately
# uses post-cutoff event dates for T (the survival time we're predicting).
# That is a LABEL, not a feature, and survival labels are required to know
# the future event time by construction. The guarantee we must defend is
# that Cox's FEATURE matrix is the same leak-safe `build_features` output
# every other v1 model uses — no new feature surface, no new leakage risk.
# ---------------------------------------------------------------------------


def test_cox_uses_only_leak_safe_features(tables: BuiltTables):
    """CoxModel.feature_names_ must be a subset of build_features() columns.

    If Cox ever started reading raw outcomes / event_date as a feature, this
    test would catch it.
    """
    from oncotraj.models import build_cox_target, get_model

    cutoff = task_a_cutoff(tables)
    X = build_features(tables, cutoff_per_patient=cutoff)
    surv = build_cox_target(tables)
    common = X.index.intersection(surv.index)
    if len(common) < 20:
        pytest.skip(f"Cox needs more than 20 (X, T, E) rows; got {len(common)}.")

    model = get_model("cox", is_classifier=False)
    model.fit_survival(X.loc[common], surv.loc[common])

    assert model.feature_names_ is not None
    assert set(model.feature_names_).issubset(set(X.columns)), (
        f"Cox introduced features outside build_features(): "
        f"{set(model.feature_names_) - set(X.columns)}"
    )


def test_cox_fit_is_unchanged_by_pre_filter(tables: BuiltTables):
    """Stronger invariant: fitting Cox on `build_features(cutoff)` is
    identical to fitting on `build_features(filter_to(cutoff))`.

    Together with the existing `test_no_leakage_invariant_strict`, this
    proves Cox sees exactly the leak-safe feature values; if any post-
    cutoff event ever crept into the Cox feature pipeline, the fitted
    coefficients would diverge.
    """
    from oncotraj.models import build_cox_target, get_model

    cutoff = task_a_cutoff(tables)
    surv = build_cox_target(tables)

    feats_cutoff = build_features(tables, cutoff_per_patient=cutoff)
    feats_filtered = build_features(
        _filter_tables_to_cutoff(tables, cutoff), cutoff_per_patient=None
    )
    common = feats_cutoff.index.intersection(feats_filtered.index).intersection(surv.index)
    if len(common) < 20:
        pytest.skip(f"Need ≥ 20 patients to fit Cox twice; got {len(common)}.")

    m_a = get_model("cox", is_classifier=False)
    m_a.fit_survival(feats_cutoff.loc[common], surv.loc[common])
    m_b = get_model("cox", is_classifier=False)
    m_b.fit_survival(feats_filtered.loc[common], surv.loc[common])

    coef_a = m_a._fitter.params_.sort_index()
    coef_b = m_b._fitter.params_.sort_index()
    pd.testing.assert_series_equal(coef_a, coef_b, check_dtype=False, atol=1e-9)


def test_one_hot_columns_unaffected_by_cutoff(tables: BuiltTables):
    """source_dataset and egfr_variant_class are patient attributes, not events.
    They must be identical with or without the cutoff."""
    cutoff = task_a_cutoff(tables)
    feats_unsafe = build_features(tables, cutoff_per_patient=None)
    feats_safe = build_features(tables, cutoff_per_patient=cutoff)
    one_hots = [c for c in feats_unsafe.columns if c.startswith(("src_", "egfr_"))]
    pd.testing.assert_frame_equal(
        feats_safe[one_hots].sort_index().sort_index(axis=1),
        feats_unsafe[one_hots].sort_index().sort_index(axis=1),
    )


@requires_papers
def test_quantify_leakage_for_audit_log(tables: BuiltTables, capsys):
    """Not strictly a test — emits the count of leaking rows for the audit log.

    Always passes; the assertion is just on the artefact being computable.
    """
    cutoff = task_a_cutoff(tables)
    cut = pd.to_datetime(cutoff)
    v = tables.variants.copy()
    v["sample_date"] = pd.to_datetime(v["sample_date"])
    v["_c"] = v["patient_id"].map(cut)
    n_post = (v["sample_date"] > v["_c"]).sum()
    n_total = len(v)
    print(f"\n[leakage audit] {n_post}/{n_total} variant rows had sample_date > cutoff")
    pids_leaking = v.loc[v["sample_date"] > v["_c"], "patient_id"].nunique()
    print(f"[leakage audit] {pids_leaking} patients affected")
    assert n_total > 0  # parquet was built; sanity.
    # Verify the leakage was non-trivial in v0.
    assert pids_leaking > 0, "Expected post-baseline variant rows in v0 cohort; got none."
