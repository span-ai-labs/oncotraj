from pathlib import Path

from oncotraj.parsers import genie, synthetic
from oncotraj.parsers.common import validate_tables


def test_genie_parses_synthetic_fixture(tmp_path: Path):
    raw = synthetic.write_synthetic_genie(tmp_path / "genie", n_patients=15, seed=42)
    tables = genie.parse(raw)

    assert len(tables.patients) == 15
    assert tables.patients["patient_id"].str.startswith("GENIE_BPC:").all()
    assert tables.patients["source_dataset"].eq("GENIE_BPC").all()
    assert not tables.variants.empty
    assert not tables.treatments.empty
    assert (tables.outcomes["event_type"] == "last_followup").any()

    # Cohort filter must yield at least one included patient — synthetic fixture
    # always emits osimertinib + EGFR sensitizing for every patient.
    assert tables.patients["included_in_v1_cohort"].all()

    # Every variant row points back to a real patient.
    assert tables.variants["patient_id"].isin(tables.patients["patient_id"]).all()


def test_genie_egfr_variant_classification(tmp_path: Path):
    raw = synthetic.write_synthetic_genie(tmp_path / "genie", n_patients=20, seed=7)
    tables = genie.parse(raw)
    classes = set(tables.patients["egfr_variant_class"].unique())
    # The synthetic fixture samples from {exon19del, L858R, G719X}.
    assert classes.issubset(
        {"exon19del", "L858R", "G719X", "compound_sensitizing", "compound_with_resistance"}
    )


def test_genie_validates_against_pydantic(tmp_path: Path):
    raw = synthetic.write_synthetic_genie(tmp_path / "genie", n_patients=10, seed=1)
    tables = genie.parse(raw)
    validate_tables(tables, sample_size=10)
