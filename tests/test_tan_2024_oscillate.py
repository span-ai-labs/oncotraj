"""Adapter tests for Tan et al. 2024 OSCILLATE."""

from datetime import date
from pathlib import Path

import pytest

from oncotraj.parsers.common import validate_tables
from oncotraj.parsers.papers._base import PaperInputs
from oncotraj.parsers.papers.tan_2024_oscillate import (
    TanOscillateAdapter,
    _find_supplement,
    _parse_gene_and_alteration,
    _split_compound_variants,
)
from oncotraj.schemas import AlterationType

REPO_ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = REPO_ROOT / "data" / "raw" / "papers" / "tan_2024_oscillate"

requires_supplement = pytest.mark.skipif(
    _find_supplement(STUDY_DIR) is None,
    reason="OSCILLATE supplement not present at data/raw/papers/tan_2024_oscillate/",
)


def test_gene_normalisation_handles_erbb2_typo():
    gene, alt = _parse_gene_and_alteration("ERRB2 amplification")
    assert gene == "ERBB2"
    assert alt == AlterationType.CNV_amplification


def test_gene_normalisation_strips_trailing_space():
    gene, alt = _parse_gene_and_alteration("MET amplification ")
    assert gene == "MET"
    assert alt == AlterationType.CNV_amplification


def test_gene_normalisation_handles_bare_snv_gene():
    gene, alt = _parse_gene_and_alteration("EGFR ")
    assert gene == "EGFR"
    assert alt == AlterationType.SNV


def test_compound_variant_split():
    assert _split_compound_variants("p.Arg988Cys, p.Arg970Cys") == ["p.Arg988Cys", "p.Arg970Cys"]
    assert _split_compound_variants("p.Leu858Arg") == ["p.Leu858Arg"]
    assert _split_compound_variants(None) == []
    assert _split_compound_variants(float("nan")) == []


@requires_supplement
def test_oscillate_yields_published_cohort_size():
    tables = TanOscillateAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    assert len(tables.patients) == 47
    assert tables.patients["patient_id"].str.startswith("AURA3_SUPP:tan_2024_oscillate.").all()


@requires_supplement
def test_oscillate_patients_all_excluded():
    tables = TanOscillateAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    assert tables.patients["included_in_v1_cohort"].sum() == 0
    reasons = set(tables.patients["exclusion_reason"].dropna().unique())
    assert reasons.issubset({"prior_EGFR_TKI", "not_EGFR_sensitizing"})


@requires_supplement
def test_oscillate_uses_real_calendar_dates():
    import pandas as pd

    tables = TanOscillateAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    # Trial enrolled patients 2017-2019; dates should be in that envelope (not a placeholder).
    sample_dates = pd.to_datetime(tables.variants["sample_date"]).dropna()
    assert sample_dates.min().date() >= date(2017, 1, 1)
    assert sample_dates.max().date() <= date(2022, 12, 31)


@requires_supplement
def test_oscillate_met_amplification_is_a_dominant_mechanism():
    """OSCILLATE's headline finding is that MET amp emerges in a meaningful
    fraction of patients despite alternating osi/gefitinib."""
    tables = TanOscillateAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    prog = tables.outcomes[tables.outcomes["event_type"] == "progression_recist"]
    met_count = (prog["resistance_mechanism_class"] == "MET_amplification").sum()
    assert met_count >= 5


@requires_supplement
def test_oscillate_compound_variants_become_multiple_rows():
    """`p.Arg988Cys, p.Arg970Cys` in the supplement should yield two variant rows."""
    tables = TanOscillateAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    # Should be strictly more variant rows than input rows (input had ~355).
    assert len(tables.variants) > 355


@requires_supplement
def test_oscillate_schema_validation():
    tables = TanOscillateAdapter(PaperInputs(study_dir=STUDY_DIR)).load()
    validate_tables(tables, sample_size=50)
