"""Framework-level tests for per-study supplement adapters.

No real-study assertions live here — only that the wiring works end-to-end
against the `_example_synthetic` fixture adapter.
"""

from pathlib import Path

from oncotraj.parsers import papers
from oncotraj.parsers.common import validate_tables


def _write_example_csv(study_dir: Path) -> None:
    manual = study_dir / "manual"
    manual.mkdir(parents=True, exist_ok=True)
    (manual / "cohort.csv").write_text(
        "stable_id,sex,smoking_status,diagnosis_date,egfr_hgvsp,osimertinib_start_date,"
        "last_followup_date,vital_status,progression_date\n"
        "pt001,Female,Never,2022-03-01,p.Leu858Arg,2022-04-01,2024-01-15,alive,2023-09-10\n"
        "pt002,Male,Former,2021-11-15,p.Glu746_Ala750del,2021-12-20,2023-06-30,deceased,\n"
        "pt003,Female,Never,2023-02-10,p.Gly719Cys,2023-03-05,2024-05-01,alive,\n"
    )


def test_example_adapter_is_registered():
    assert "_example_synthetic" in papers.registered_study_ids()


def test_load_all_handles_missing_study_dirs(tmp_path: Path):
    # No study subdirs exist — load_all should return empty tables, not raise.
    tables = papers.load_all(tmp_path)
    assert len(tables) == 0


def test_example_adapter_loads_manual_csv(tmp_path: Path):
    _write_example_csv(tmp_path / "_example_synthetic")
    tables = papers.load_all(tmp_path)

    assert len(tables.patients) == 3
    assert tables.patients["patient_id"].str.startswith("FLAURA_SUPP:_example_synthetic.").all()
    assert tables.patients["included_in_v1_cohort"].all()
    assert (tables.variants["gene_symbol"] == "EGFR").all()
    # The patient with a progression_date should produce a progression outcome row.
    assert (tables.outcomes["event_type"] == "progression_recist").sum() == 1


def test_example_adapter_validates_against_pydantic(tmp_path: Path):
    _write_example_csv(tmp_path / "_example_synthetic")
    tables = papers.load_all(tmp_path)
    validate_tables(tables, sample_size=10)


def test_build_script_integration_with_papers_dir(tmp_path: Path):
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import build_dataset

    _write_example_csv(tmp_path / "papers" / "_example_synthetic")

    output = tmp_path / "out.parquet"
    meta = build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=tmp_path / "raw",
        papers_root=tmp_path / "papers",
    )

    # meta["sources"] is derived from the post-cohort patients table's
    # `source_dataset` column, so it reflects what actually landed in the
    # parquet — `FLAURA_SUPP` for the example paper, plus the synthetic
    # GENIE/CHORD sources. Paper study IDs filtered out by the cohort are
    # therefore correctly absent (this is the fix for the stale-meta bug).
    assert "FLAURA_SUPP" in meta["sources"]
    # Synthetic GENIE (20) + synthetic CHORD (30) + 3 from the example paper.
    assert meta["row_counts"]["patients"] == 53
