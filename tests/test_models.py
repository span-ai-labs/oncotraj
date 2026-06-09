"""Tests for the v1 baseline models and the train_baseline script."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oncotraj.models import (
    LogisticRegressionModel,
    MajorityClassModel,
    RandomForestModel,
    XGBoostModel,
    get_model,
)
from oncotraj.models.features import (
    BuiltTables,
    build_features,
    build_target_a,
    build_target_b,
    build_target_c,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import build_dataset  # noqa: E402
import train_baseline  # noqa: E402


def _toy_tables() -> BuiltTables:
    patients = pd.DataFrame(
        [
            {
                "patient_id": "GENIE_BPC:p1",
                "source_dataset": "GENIE_BPC",
                "egfr_variant_class": "exon19del",
                "last_followup_date": "2024-01-01",
                "included_in_v1_cohort": True,
            },
            {
                "patient_id": "GENIE_BPC:p2",
                "source_dataset": "GENIE_BPC",
                "egfr_variant_class": "L858R",
                "last_followup_date": "2024-06-01",
                "included_in_v1_cohort": True,
            },
            {
                "patient_id": "GENIE_BPC:p3",
                "source_dataset": "GENIE_BPC",
                "egfr_variant_class": "L858R",
                "last_followup_date": "2024-12-01",
                "included_in_v1_cohort": True,
            },
        ]
    )
    variants = pd.DataFrame(
        [
            {
                "patient_id": "GENIE_BPC:p1",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.20,
                "sample_date": "2022-01-01",
            },
            {
                "patient_id": "GENIE_BPC:p1",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.40,
                "sample_date": "2023-01-01",
            },
            {
                "patient_id": "GENIE_BPC:p2",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.15,
                "sample_date": "2022-06-01",
            },
            {
                "patient_id": "GENIE_BPC:p3",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.30,
                "sample_date": "2023-06-01",
            },
        ]
    )
    treatments = pd.DataFrame(
        [
            {
                "patient_id": "GENIE_BPC:p1",
                "is_osimertinib": True,
                "start_date": "2022-01-01",
                "end_date": "2023-04-01",
            },
            {
                "patient_id": "GENIE_BPC:p2",
                "is_osimertinib": True,
                "start_date": "2022-06-01",
                "end_date": None,
            },
            {
                "patient_id": "GENIE_BPC:p3",
                "is_osimertinib": True,
                "start_date": "2023-06-01",
                "end_date": "2024-09-01",
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "patient_id": "GENIE_BPC:p1",
                "event_type": "progression_recist",
                "event_date": "2023-04-01",
                "resistance_mechanism_class": "EGFR_C797S",
            },
            {
                "patient_id": "GENIE_BPC:p3",
                "event_type": "progression_recist",
                "event_date": "2024-09-01",
                "resistance_mechanism_class": "MET_amplification",
            },
            {
                "patient_id": "GENIE_BPC:p2",
                "event_type": "last_followup",
                "event_date": "2024-06-01",
                "resistance_mechanism_class": None,
            },
        ]
    )
    return BuiltTables(
        patients=patients, variants=variants, treatments=treatments, outcomes=outcomes
    )


def test_features_shape_and_columns():
    tables = _toy_tables()
    feats = build_features(tables)
    assert list(feats.index) == ["GENIE_BPC:p1", "GENIE_BPC:p2", "GENIE_BPC:p3"]
    for col in ("latest_vaf", "vaf_slope", "variant_burden", "time_on_therapy_days"):
        assert col in feats.columns
    # Source/egfr one-hots:
    assert any(c.startswith("src_") for c in feats.columns)
    assert any(c.startswith("egfr_") for c in feats.columns)


def test_target_builders():
    tables = _toy_tables()
    a = build_target_a(tables)
    assert a.loc["GENIE_BPC:p1"] == 1 and a.loc["GENIE_BPC:p2"] == 0
    b = build_target_b(tables)
    # p1: 2022-01-01 → 2023-04-01 = 455 days
    assert b.loc["GENIE_BPC:p1"] == 455
    assert "GENIE_BPC:p2" not in b.index  # no progression event
    c = build_target_c(tables)
    assert c.loc["GENIE_BPC:p1"] == "EGFR_C797S"
    assert c.loc["GENIE_BPC:p3"] == "MET_amplification"


@pytest.mark.parametrize(
    "model_cls",
    [
        MajorityClassModel,
        LogisticRegressionModel,
        RandomForestModel,
        XGBoostModel,
    ],
)
def test_classifier_interface_uniform(model_cls):
    tables = _toy_tables()
    X = build_features(tables)
    y = build_target_a(tables).reindex(X.index)
    model = model_cls(is_classifier=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    proba = model.predict_proba(X)
    assert proba.shape[0] == len(X)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    "model_cls",
    [
        MajorityClassModel,
        RandomForestModel,
        XGBoostModel,
    ],
)
def test_regressor_interface(model_cls):
    tables = _toy_tables()
    X = build_features(tables)
    y = build_target_b(tables)
    X = X.loc[y.index]
    model = model_cls(is_classifier=False)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


def test_registry_lookup():
    m = get_model("logistic", is_classifier=True)
    assert isinstance(m, LogisticRegressionModel)
    with pytest.raises(KeyError):
        get_model("does_not_exist")


def test_model_save_and_load(tmp_path):
    tables = _toy_tables()
    X = build_features(tables)
    y = build_target_a(tables).reindex(X.index)
    model = LogisticRegressionModel(is_classifier=True).fit(X, y)
    path = tmp_path / "lr.joblib"
    model.save(path)
    loaded = LogisticRegressionModel.load(path)
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))


def test_train_baseline_end_to_end(tmp_path, monkeypatch):
    """Build parquet → build splits → train one model → verify MLflow run + saved model."""
    from oncotraj.data.splits import make_splits_from_parquet

    output = tmp_path / "v0"
    build_dataset.build_dataset(
        output_dir=output,
        cohort="egfr_nsclc",
        use_synthetic=True,
        raw_root=tmp_path / "raw",
        papers_root=None,
    )
    splits_path = tmp_path / "splits.json"
    make_splits_from_parquet(parquet_dir=output, output_path=splits_path, seed=42)

    monkeypatch.setattr(train_baseline, "REPO_ROOT", tmp_path)  # isolate mlruns/

    result = train_baseline.train_one(
        model_name="logistic",
        task="A",
        parquet_dir=output,
        splits_path=splits_path,
        output_dir=tmp_path / "models",
    )
    assert result["model"] == "logistic"
    assert result["task"] == "A"
    assert "train" in result["metrics"]
    assert (tmp_path / "models" / "taskA_logistic.joblib").exists()
    assert (tmp_path / "mlruns").exists()
