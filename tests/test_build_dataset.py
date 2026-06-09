import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_dataset  # noqa: E402


def test_build_dataset_end_to_end(tmp_path: Path):
    output = tmp_path / "oncotraj_v0.parquet"
    raw_root = tmp_path / "raw"

    meta = build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=raw_root,
    )

    for name in ("patients", "variants", "treatments", "outcomes"):
        assert (output / f"{name}.parquet").exists(), f"missing {name}.parquet"

    patients = pd.read_parquet(output / "patients.parquet")
    variants = pd.read_parquet(output / "variants.parquet")

    assert meta["row_counts"]["patients"] == len(patients)
    assert meta["row_counts"]["patients"] > 0
    assert patients["included_in_v1_cohort"].all(), (
        "cohort filter should retain only included patients"
    )
    assert set(patients["source_dataset"].unique()) == {"GENIE_BPC", "MSK_CHORD"}

    # FK integrity: every variant patient_id resolves to a patient row.
    assert variants["patient_id"].isin(patients["patient_id"]).all()

    meta_on_disk = json.loads((output / "_meta.json").read_text())
    assert meta_on_disk["schema_version"].startswith("oncotraj-schema/")
