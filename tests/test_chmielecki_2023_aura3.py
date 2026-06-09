"""Adapter tests for Chmielecki et al. 2023 AURA3 supplement."""

from pathlib import Path

import pytest

from oncotraj.parsers.common import validate_tables
from oncotraj.parsers.papers._base import PaperInputs
from oncotraj.parsers.papers.chmielecki_2023_aura3 import (
    ChmieleckiAura3Adapter,
    _find_supplement,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "data" / "raw" / "papers" / "chmielecki_2023_aura3"

requires_supplement = pytest.mark.skipif(
    _find_supplement(STUDY_DIR) is None,
    reason="Chmielecki 2023 AURA3 supplement not present at data/raw/papers/chmielecki_2023_aura3/",
)


@requires_supplement
def test_aura3_yields_published_cohort_size():
    tables = ChmieleckiAura3Adapter(PaperInputs(study_dir=STUDY_DIR)).load()
    # Paper's molecular-analysis subset for the osimertinib arm = 78 patients.
    assert len(tables.patients) == 78
    assert tables.patients["patient_id"].str.startswith("AURA3_SUPP:chmielecki_2023_aura3.").all()


@requires_supplement
def test_aura3_patients_are_all_excluded_from_v1_cohort():
    """AURA3 is the held-out sensitivity slice per PAPER_OUTLINE.md §0.

    Every patient must carry an exclusion_reason; none should land in the v1
    cohort (it would contaminate the training set with prior-TKI patients).
    """
    tables = ChmieleckiAura3Adapter(PaperInputs(study_dir=STUDY_DIR)).load()
    assert tables.patients["included_in_v1_cohort"].sum() == 0
    reasons = set(tables.patients["exclusion_reason"].dropna().unique())
    assert reasons.issubset({"prior_EGFR_TKI", "not_EGFR_sensitizing"})
    assert "prior_EGFR_TKI" in reasons


@requires_supplement
def test_aura3_compound_with_resistance_dominates_classification():
    """AURA3 enrolled T790M-acquired patients; baseline EGFR variants should
    almost always be `compound_with_resistance` (sensitizing + T790M)."""
    tables = ChmieleckiAura3Adapter(PaperInputs(study_dir=STUDY_DIR)).load()
    classes = tables.patients["egfr_variant_class"].value_counts().to_dict()
    assert classes.get("compound_with_resistance", 0) >= 50


@requires_supplement
def test_aura3_c797s_rate_consistent_with_published():
    """Published Chmielecki AURA3 C797S rate is ~15-22%."""
    tables = ChmieleckiAura3Adapter(PaperInputs(study_dir=STUDY_DIR)).load()
    prog = tables.outcomes[tables.outcomes["event_type"] == "progression_recist"]
    c797s_count = (prog["resistance_mechanism_class"] == "EGFR_C797S").sum()
    assert 8 <= c797s_count <= 25


@requires_supplement
def test_aura3_treatments_are_second_line():
    tables = ChmieleckiAura3Adapter(PaperInputs(study_dir=STUDY_DIR)).load()
    assert (tables.treatments["line_of_therapy"] == "second_line").all()
    assert tables.treatments["is_osimertinib"].all()


@requires_supplement
def test_aura3_schema_validation():
    tables = ChmieleckiAura3Adapter(PaperInputs(study_dir=STUDY_DIR)).load()
    validate_tables(tables, sample_size=50)
