"""Hand-crafted feature engineering for the v1 baselines.

Per the user spec for paper #6 (classical ML baselines), features are:
- `latest_vaf`: max VAF across this patient's most recent ctDNA sample.
- `vaf_slope`: per-day change in max EGFR-VAF from earliest to latest sample.
  A simple two-point slope; full linear regression is overkill for the
  patient counts we have today.
- `variant_burden`: total count of variant rows for the patient.
- `time_on_therapy_days`: (treatment.end_date or last_followup_date) -
  treatment.start_date. Negative or missing values clamp to 0.
- One-hot indicators for `source_dataset` and `egfr_variant_class`.

The output is a DataFrame indexed by `patient_id`. Target builders
(`build_target`) produce a `pd.Series` aligned to this index.

Limitations the user already knows about:
- VAF features are weak for synthetic and FLAURA rows (synthetic GENIE/CHORD
  doesn't emit serial draws; FLAURA emits at most baseline + discontinuation).
  When real ctDNA dynamics land via TRACERx or registries, these features
  become useful.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BuiltTables:
    """The four-table view a feature builder operates on."""

    patients: pd.DataFrame
    variants: pd.DataFrame
    treatments: pd.DataFrame
    outcomes: pd.DataFrame


def _latest_vaf_per_patient(variants: pd.DataFrame) -> pd.Series:
    """Max VAF (excluding sentinels) at each patient's most-recent sample."""
    if variants.empty:
        return pd.Series(dtype=float, name="latest_vaf")
    df = variants.loc[variants["vaf"] >= 0.0, ["patient_id", "sample_date", "vaf"]].copy()
    if df.empty:
        return pd.Series(dtype=float, name="latest_vaf")
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    df = df.sort_values(["patient_id", "sample_date"])
    last = df.groupby("patient_id").tail(1).groupby("patient_id")["vaf"].max()
    last.name = "latest_vaf"
    return last


def _vaf_slope_per_patient(variants: pd.DataFrame) -> pd.Series:
    """(latest_vaf - baseline_vaf) / days_between, in 1/day units.

    Computed only on EGFR sensitizing-allele VAFs (gene == EGFR, alteration SNV).
    Zero for patients with only one sample.
    """
    if variants.empty:
        return pd.Series(dtype=float, name="vaf_slope")
    egfr = variants.loc[
        (variants["gene_symbol"] == "EGFR")
        & (variants["alteration_type"] == "SNV")
        & (variants["vaf"] >= 0.0),
        ["patient_id", "sample_date", "vaf"],
    ].copy()
    if egfr.empty:
        return pd.Series(dtype=float, name="vaf_slope")
    egfr["sample_date"] = pd.to_datetime(egfr["sample_date"])
    egfr = egfr.sort_values(["patient_id", "sample_date"])
    grouped = egfr.groupby("patient_id")
    first = grouped.head(1).set_index("patient_id")
    last = grouped.tail(1).set_index("patient_id")
    common = first.index.intersection(last.index)
    if common.empty:
        return pd.Series(dtype=float, name="vaf_slope")
    days = (last.loc[common, "sample_date"] - first.loc[common, "sample_date"]).dt.days
    delta = last.loc[common, "vaf"] - first.loc[common, "vaf"]
    slope = (delta / days.replace(0, np.nan)).fillna(0.0)
    slope.name = "vaf_slope"
    return slope


def _variant_burden(variants: pd.DataFrame) -> pd.Series:
    if variants.empty:
        return pd.Series(dtype=int, name="variant_burden")
    s = variants.groupby("patient_id").size().astype(int)
    s.name = "variant_burden"
    return s


def _time_on_therapy(
    treatments: pd.DataFrame,
    patients: pd.DataFrame,
    cutoff_per_patient: pd.Series | None = None,
) -> pd.Series:
    """Days of osimertinib exposure as of the per-patient prediction cutoff.

    Without a cutoff: end_date or last_followup_date is the effective end.
    With a cutoff: effective_end = min(end_date or last_followup, cutoff).
    Treatments that *start* after cutoff are excluded entirely.
    """
    if treatments.empty or patients.empty:
        return pd.Series(dtype=float, name="time_on_therapy_days")
    osi = treatments.loc[
        treatments["is_osimertinib"], ["patient_id", "start_date", "end_date"]
    ].copy()
    if osi.empty:
        return pd.Series(dtype=float, name="time_on_therapy_days")
    osi["start_date"] = pd.to_datetime(osi["start_date"])
    osi["end_date"] = pd.to_datetime(osi["end_date"])
    lfu = patients.set_index("patient_id")["last_followup_date"].pipe(pd.to_datetime)
    osi = osi.set_index("patient_id")
    effective_end = osi["end_date"].fillna(lfu.reindex(osi.index))
    if cutoff_per_patient is not None:
        cut = pd.to_datetime(cutoff_per_patient).reindex(osi.index)
        # Drop treatments that started AFTER the cutoff entirely.
        mask = osi["start_date"] <= cut
        osi = osi.loc[mask]
        effective_end = effective_end.loc[mask]
        cut = cut.loc[mask]
        effective_end = effective_end.where(effective_end <= cut, cut)
    days = (effective_end - osi["start_date"]).dt.days.clip(lower=0).fillna(0.0)
    days.name = "time_on_therapy_days"
    return days.groupby(level=0).max()


def _apply_cutoff_to_variants(
    variants: pd.DataFrame, cutoff_per_patient: pd.Series
) -> pd.DataFrame:
    """Filter variants to rows whose sample_date is on or before the cutoff.

    Rows for patients missing from `cutoff_per_patient` are dropped. The
    intent is conservative: when the cutoff is unknown, we cannot guarantee
    feature freshness so we exclude.
    """
    if variants.empty:
        return variants
    v = variants.copy()
    v["sample_date"] = pd.to_datetime(v["sample_date"])
    cut = pd.to_datetime(cutoff_per_patient)
    v["_cutoff"] = v["patient_id"].map(cut)
    v = v[v["_cutoff"].notna() & (v["sample_date"] <= v["_cutoff"])]
    return v.drop(columns=["_cutoff"])


def task_a_cutoff(tables: BuiltTables) -> pd.Series:
    """Per-patient prediction-time cutoff for Task A.

    Defined as the patient's first osimertinib `start_date`. Feature
    computation may use any event with `event_date <= cutoff` (i.e., the
    baseline / C1D1 draw and anything prior). Anything strictly after the
    cutoff (Discontinuation / Progression samples, treatment-end-as-
    progression-marker, etc.) is excluded.
    """
    if tables.treatments.empty:
        return pd.Series(dtype="datetime64[ns]", name="cutoff")
    osi = tables.treatments.loc[tables.treatments["is_osimertinib"]]
    s = pd.to_datetime(osi.groupby("patient_id")["start_date"].min())
    s.name = "cutoff"
    return s


# Co-occurring genes used as prognostic features. TP53 co-mutation is the
# best-established poor-prognosis marker for osimertinib PFS in EGFR+ NSCLC;
# RB1/CDKN2A loss and bypass-pathway alterations (MET, ERBB2, PIK3CA) carry
# additional signal. All are computed leak-safely from variants on or before
# the prediction cutoff.
COMUTATION_GENES = ("TP53", "RB1", "CDKN2A", "PIK3CA", "CTNNB1", "MET", "ERBB2", "KRAS", "BRAF")


def _comutation_features(variants: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Per-patient co-mutation presence flags + total co-mutation burden.

    `variants` must already be cutoff-filtered by the caller. EGFR itself is
    excluded from the burden count (it is the defining driver, not a
    co-mutation)."""
    cols = {}
    if variants is None or variants.empty:
        for g in COMUTATION_GENES:
            cols[f"comut_{g}"] = pd.Series(0.0, index=index)
        cols["comut_burden"] = pd.Series(0.0, index=index)
        return pd.DataFrame(cols, index=index)
    for g in COMUTATION_GENES:
        pids = set(variants.loc[variants["gene_symbol"] == g, "patient_id"])
        cols[f"comut_{g}"] = index.to_series().isin(pids).astype(float)
    non_egfr = variants.loc[variants["gene_symbol"] != "EGFR"]
    burden = non_egfr.groupby("patient_id")["gene_symbol"].nunique().reindex(index).fillna(0.0)
    cols["comut_burden"] = burden.astype(float)
    return pd.DataFrame(cols, index=index)


def build_features(
    tables: BuiltTables,
    cutoff_per_patient: pd.Series | None = None,
) -> pd.DataFrame:
    """Assemble the v1 baseline feature matrix indexed by patient_id.

    If `cutoff_per_patient` is provided, every event-derived feature is
    computed using only events with `event_date <= cutoff`. This is the
    leak-safe mode used for Task A (see `task_a_cutoff`).
    """
    patients = tables.patients.set_index("patient_id")
    variants_for_features = (
        _apply_cutoff_to_variants(tables.variants, cutoff_per_patient)
        if cutoff_per_patient is not None
        else tables.variants
    )
    parts = [
        _latest_vaf_per_patient(variants_for_features),
        _vaf_slope_per_patient(variants_for_features),
        _variant_burden(variants_for_features),
        _time_on_therapy(tables.treatments, tables.patients, cutoff_per_patient),
    ]
    feats = pd.concat(parts, axis=1)
    feats = feats.reindex(patients.index)
    # Co-mutation features (leak-safe, computed from the cutoff-filtered variants).
    comut = _comutation_features(variants_for_features, patients.index)
    feats = pd.concat([feats, comut], axis=1)
    # Categorical one-hots — these are inherent patient attributes, not events,
    # so they are always safe to use as features regardless of cutoff.
    one_hots = pd.get_dummies(
        patients[["source_dataset", "egfr_variant_class"]],
        prefix=["src", "egfr"],
        dtype=float,
    )
    feats = pd.concat([feats, one_hots], axis=1)
    feats = feats.fillna(0.0)
    feats.index.name = "patient_id"
    return feats


# ---------------------------------------------------------------------------
# Target builders
# ---------------------------------------------------------------------------


def build_target_a(tables: BuiltTables) -> pd.Series:
    """Task A (deprecated 'ever-progresses' form): 1 iff the patient has any
    progression_recist outcome.

    On a first-line-osimertinib cohort this label is near-degenerate (~95%
    positive) and not learnable. Retained only for backward comparison;
    `build_target_a_landmark` is the form used for the headline result.
    """
    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist", "patient_id"
    ].unique()
    target = pd.Series(0, index=tables.patients["patient_id"].unique(), dtype=int, name="task_a")
    target.loc[target.index.isin(prog)] = 1
    return target


def build_target_a_landmark(tables: BuiltTables, horizon_days: int = 365) -> pd.Series:
    """Task A (landmark form): 1 iff the patient progresses within
    `horizon_days` of osimertinib start, 0 iff confirmed progression-free at
    the landmark, dropped (NaN, excluded) iff censored before the landmark.

    This converts the degenerate ever-progresses label into a balanced,
    well-defined fixed-horizon prediction (default 12 months). The standard
    landmark-analysis convention: a patient with no progression but follow-up
    shorter than the horizon is uninformative and excluded, not labelled 0.
    """
    treatments = tables.treatments.loc[tables.treatments["is_osimertinib"]].copy()
    treatments["start_date"] = pd.to_datetime(treatments["start_date"])
    start = treatments.groupby("patient_id")["start_date"].min()

    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist", ["patient_id", "event_date"]
    ].copy()
    prog["event_date"] = pd.to_datetime(prog["event_date"])
    first_prog = prog.groupby("patient_id")["event_date"].min()

    patients = tables.patients.set_index("patient_id")
    lfu = pd.to_datetime(patients["last_followup_date"])

    labels: dict[str, int] = {}
    for pid in start.index:
        s = start.loc[pid]
        if pid in first_prog.index:
            days_to_prog = (first_prog.loc[pid] - s).days
            labels[pid] = 1 if days_to_prog <= horizon_days else 0
        elif pid in lfu.index:
            fu_days = (lfu.loc[pid] - s).days
            if fu_days >= horizon_days:
                labels[pid] = 0  # confirmed progression-free at the landmark
            # else: censored before the landmark → excluded (not added)
    out = pd.Series(labels, dtype=int, name="task_a_landmark")
    return out


def build_target_b(tables: BuiltTables) -> pd.Series:
    """Task B: days from osimertinib start to first progression. NaN if no PD."""
    treatments = tables.treatments.loc[tables.treatments["is_osimertinib"]].copy()
    treatments["start_date"] = pd.to_datetime(treatments["start_date"])
    start = treatments.groupby("patient_id")["start_date"].min()

    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist", ["patient_id", "event_date"]
    ].copy()
    prog["event_date"] = pd.to_datetime(prog["event_date"])
    first_prog = prog.groupby("patient_id")["event_date"].min()

    common = start.index.intersection(first_prog.index)
    days = (first_prog.loc[common] - start.loc[common]).dt.days.astype(float)
    days.name = "task_b"
    return days


def build_target_c(tables: BuiltTables) -> pd.Series:
    """Task C: dominant mechanism class at first progression."""
    prog = tables.outcomes.loc[
        tables.outcomes["event_type"] == "progression_recist",
        ["patient_id", "event_date", "resistance_mechanism_class"],
    ].copy()
    prog["event_date"] = pd.to_datetime(prog["event_date"])
    prog = prog.sort_values(["patient_id", "event_date"])
    first = prog.groupby("patient_id").head(1).set_index("patient_id")["resistance_mechanism_class"]
    first = first.dropna()
    first.name = "task_c"
    return first


TASK_BUILDERS = {
    "A": ("classification", build_target_a_landmark),  # 12-month landmark (headline form)
    "A_ever": ("classification", build_target_a),  # deprecated ever-progresses form
    "B": ("regression", build_target_b),
    "C": ("classification", build_target_c),
}
