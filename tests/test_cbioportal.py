from pathlib import Path

from oncotraj.parsers import cbioportal, synthetic
from oncotraj.parsers.common import validate_tables


def test_cbioportal_parses_synthetic_fixture(tmp_path: Path):
    raw = synthetic.write_synthetic_cbioportal(tmp_path / "cbio", n_patients=20, seed=3)
    tables = cbioportal.parse(raw)

    assert len(tables.patients) == 20
    assert tables.patients["patient_id"].str.startswith("MSK_CHORD:").all()
    assert tables.patients["source_dataset"].eq("MSK_CHORD").all()
    assert tables.patients["included_in_v1_cohort"].all()

    # Some MET amplifications should be flagged as resistance calls.
    met_rows = tables.variants.loc[tables.variants["gene_symbol"] == "MET"]
    if not met_rows.empty:
        assert met_rows["is_resistance_call"].any()
        assert met_rows["resistance_mechanism_class"].dropna().eq("MET_amplification").any()


def test_cbioportal_validates_against_pydantic(tmp_path: Path):
    raw = synthetic.write_synthetic_cbioportal(tmp_path / "cbio", n_patients=12, seed=11)
    tables = cbioportal.parse(raw)
    validate_tables(tables, sample_size=12)


def test_cbioportal_outcomes_have_last_followup(tmp_path: Path):
    raw = synthetic.write_synthetic_cbioportal(tmp_path / "cbio", n_patients=15, seed=5)
    tables = cbioportal.parse(raw)
    pid_with_lfu = set(
        tables.outcomes.loc[tables.outcomes["event_type"] == "last_followup", "patient_id"]
    )
    assert pid_with_lfu == set(tables.patients["patient_id"])
