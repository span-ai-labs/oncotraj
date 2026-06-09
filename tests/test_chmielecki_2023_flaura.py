"""Adapter tests for Chmielecki et al. 2023 FLAURA molecular-resistance supplement.

Runs against the real Supplementary Data 1 from the paper if available at
`data/raw/papers/chmielecki_2023_flaura/MOESM3.xlsx`. Skipped when the file is
not present (so CI without the supplement still passes).
"""

from pathlib import Path

import pytest

from oncotraj.parsers.common import validate_tables
from oncotraj.parsers.papers._base import PaperInputs
from oncotraj.parsers.papers.chmielecki_2023_flaura import (
    ENROLLMENT_EPOCH,
    ChmieleckiFlauraAdapter,
    _find_supplement,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "data" / "raw" / "papers" / "chmielecki_2023_flaura"

requires_supplement = pytest.mark.skipif(
    _find_supplement(STUDY_DIR) is None,
    reason="Chmielecki 2023 FLAURA supplement not present at data/raw/papers/chmielecki_2023_flaura/",
)


@requires_supplement
def test_chmielecki_flaura_yields_published_cohort_size():
    tables = ChmieleckiFlauraAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    # Paper's molecular analysis subset for osimertinib arm = 109 patients.
    assert len(tables.patients) == 109
    # SoC arm must not leak in.
    assert tables.patients["patient_id"].str.startswith("FLAURA_SUPP:chmielecki_2023_flaura.").all()


@requires_supplement
def test_chmielecki_flaura_egfr_classification():
    tables = ChmieleckiFlauraAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    classes = tables.patients["egfr_variant_class"].value_counts().to_dict()
    # exon19del and L858R together dominate the FLAURA cohort.
    assert classes.get("exon19del", 0) + classes.get("L858R", 0) >= 100
    # No spurious exon20ins matches (the FLAURA trial excluded exon20ins).
    assert classes.get("exon20ins", 0) == 0
    # Spec invariant I3: any included patient has a known EGFR class.
    included = tables.patients.loc[tables.patients["included_in_v1_cohort"]]
    assert (included["egfr_variant_class"] != "unknown").all()


@requires_supplement
def test_chmielecki_flaura_c797s_calls_are_at_pd_only():
    tables = ChmieleckiFlauraAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    c797s = tables.variants[
        tables.variants["protein_change_hgvs"].fillna("").str.contains("C797")
        & (tables.variants["gene_symbol"] == "EGFR")
    ]
    # The published C797S rate on first-line osimertinib is ~6-7% — sanity check.
    assert 1 <= c797s["is_resistance_call"].sum() <= 20
    # No C797S in baseline samples (per LABELING_GUIDELINES.md §1.1 disqualifier).
    baseline_c797s = c797s[c797s["sample_date"] == ENROLLMENT_EPOCH]
    assert not baseline_c797s["is_resistance_call"].any()


@requires_supplement
def test_chmielecki_flaura_progression_outcomes_have_mechanism_or_unknown():
    tables = ChmieleckiFlauraAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    prog = tables.outcomes[tables.outcomes["event_type"] == "progression_recist"]
    # Every progression row carries a mechanism call (or explicit other_or_unknown).
    assert prog["resistance_mechanism_class"].notna().all()
    # Most patients fall into either C797S or other_or_unknown for a v0 adapter.
    valid_classes = {
        "EGFR_C797S",
        "MET_amplification",
        "HER2_amplification",
        "EGFR_amplification",
        "histologic_transformation",
        "other_or_unknown",
    }
    assert set(prog["resistance_mechanism_class"].unique()).issubset(valid_classes)


@requires_supplement
def test_chmielecki_flaura_schema_validation():
    tables = ChmieleckiFlauraAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    validate_tables(tables, sample_size=50)
