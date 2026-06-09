"""Tests for the small Transformer baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oncotraj.models.transformer import (
    CLS_EVENT_TYPE,
    EVENT_TYPES,
    GENE_VOCAB,
    OncoTrajTransformer,
    TransformerConfig,
    TransformerModel,
    apply_rope,
    build_event_sequences,
    build_rope_cos_sin,
    count_parameters,
    pad_event_batch,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _toy_tables():
    variants = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.20,
                "sample_date": "2022-01-01",
                "protein_change_hgvs": "p.Leu858Arg",
            },
            {
                "patient_id": "p1",
                "gene_symbol": "EGFR",
                "alteration_type": "SNV",
                "vaf": 0.40,
                "sample_date": "2023-01-01",
                "protein_change_hgvs": "p.Cys797Ser",
            },
            {
                "patient_id": "p2",
                "gene_symbol": "MET",
                "alteration_type": "CNV_amplification",
                "vaf": -1.0,
                "sample_date": "2022-06-01",
                "protein_change_hgvs": None,
            },
        ]
    )
    treatments = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "is_osimertinib": True,
                "start_date": "2022-01-01",
                "end_date": "2023-04-01",
            },
            {
                "patient_id": "p2",
                "is_osimertinib": True,
                "start_date": "2022-06-01",
                "end_date": None,
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "event_type": "progression_recist",
                "event_date": "2023-04-01",
                "resistance_mechanism_class": "EGFR_C797S",
            },
            {
                "patient_id": "p2",
                "event_type": "last_followup",
                "event_date": "2024-01-01",
                "resistance_mechanism_class": None,
            },
        ]
    )
    return variants, treatments, outcomes


def test_event_vocab_includes_required_types():
    for required in ("PAD", "CLS", "variant_snv", "osi_start", "progression"):
        assert required in EVENT_TYPES


def test_build_event_sequences_starts_with_cls():
    v, t, o = _toy_tables()
    seqs = build_event_sequences(["p1", "p2", "p_missing"], v, t, o)
    for pid in ("p1", "p2", "p_missing"):
        assert seqs[pid].event_types[0] == CLS_EVENT_TYPE
        assert seqs[pid].timestamps[0] == 0.0


def test_event_sequence_ordering_is_chronological():
    v, t, o = _toy_tables()
    seqs = build_event_sequences(["p1"], v, t, o)
    ts = seqs["p1"].timestamps
    # Timestamps should be non-decreasing after the CLS token.
    assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))


def test_pad_event_batch_shapes():
    v, t, o = _toy_tables()
    seqs = build_event_sequences(["p1", "p2"], v, t, o)
    batch = pad_event_batch([seqs["p1"], seqs["p2"]])
    assert batch["event_types"].shape[0] == 2
    assert batch["event_types"].dtype == torch.long
    assert batch["values"].dtype == torch.float32
    assert batch["attention_mask"].dtype == torch.bool


def test_rope_cos_sin_shapes_and_apply():
    head_dim = 32
    timestamps = torch.tensor([[0.0, 1.0, 100.0]])
    cos, sin = build_rope_cos_sin(timestamps, head_dim)
    assert cos.shape == (1, 3, head_dim)
    assert sin.shape == (1, 3, head_dim)
    x = torch.randn(1, 3, head_dim)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    # RoPE at timestamp 0 should be the identity (cos=1, sin=0).
    torch.testing.assert_close(out[:, 0], x[:, 0])


def test_transformer_forward_shape_and_param_count():
    cfg = TransformerConfig(d_model=256, n_heads=8, n_layers=4, d_ff=1024, num_classes=3)
    net = OncoTrajTransformer(cfg)
    n_params = count_parameters(net)
    # ~3M-4M params at this size; well below the 10M target but the test
    # cares about correctness, not yield.
    assert n_params > 100_000
    batch_size, seq_len = 2, 5
    out = net(
        event_types=torch.zeros((batch_size, seq_len), dtype=torch.long),
        values=torch.zeros((batch_size, seq_len), dtype=torch.float32),
        timestamps=torch.zeros((batch_size, seq_len), dtype=torch.float32),
        gene_ids=torch.zeros((batch_size, seq_len), dtype=torch.long),
        attention_mask=torch.ones((batch_size, seq_len), dtype=torch.bool),
    )
    assert out["reg_pred"].shape == (batch_size,)
    assert out["cls_logits"].shape == (batch_size, 3)


def test_transformer_param_count_in_target_band_for_default_args():
    """At the script-default config, the model should fall in 10-50M."""
    cfg = TransformerConfig(d_model=384, n_heads=8, n_layers=8, d_ff=1536, num_classes=2)
    net = OncoTrajTransformer(cfg)
    n = count_parameters(net)
    assert 10e6 < n < 50e6, f"Got {n:,} parameters"


def test_transformer_model_classifier_interface():
    v, t, o = _toy_tables()
    pids = ["p1", "p2"] * 4
    variants = pd.concat([v.assign(patient_id=p) for p in pids], ignore_index=True)
    treatments = pd.concat([t.assign(patient_id=p) for p in pids], ignore_index=True)
    outcomes = pd.concat([o.assign(patient_id=p) for p in pids], ignore_index=True)

    X = pd.DataFrame(index=list(set(pids)))
    y_cls = pd.Series(["EGFR_C797S", "other_or_unknown"], index=list(set(pids)), name="task_c")
    model = TransformerModel(
        is_classifier=True,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        epochs=2,
        patience=10,
        device=torch.device("cpu"),
        variants_df=variants,
        treatments_df=treatments,
        outcomes_df=outcomes,
    )
    model.fit(X, y_cls)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    assert preds.shape == (2,)
    assert proba.shape == (2, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_transformer_model_regressor_interface():
    v, t, o = _toy_tables()
    pids = ["p1", "p2"] * 4
    variants = pd.concat([v.assign(patient_id=p) for p in pids], ignore_index=True)
    treatments = pd.concat([t.assign(patient_id=p) for p in pids], ignore_index=True)
    outcomes = pd.concat([o.assign(patient_id=p) for p in pids], ignore_index=True)

    X = pd.DataFrame(index=list(set(pids)))
    y_reg = pd.Series([100.0, 200.0], index=list(set(pids)), name="task_b")
    model = TransformerModel(
        is_classifier=False,
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        epochs=2,
        patience=10,
        device=torch.device("cpu"),
        variants_df=variants,
        treatments_df=treatments,
        outcomes_df=outcomes,
    )
    model.fit(X, y_reg)
    preds = model.predict(X)
    assert preds.shape == (2,)
    with pytest.raises(NotImplementedError):
        model.predict_proba(X)


def test_attention_mask_zeroes_padded_positions():
    """Confirm that padded positions don't contaminate the CLS pooling.

    We construct two identical-CLS sequences with different padding lengths
    and verify the model output is invariant up to numerical error.
    """
    torch.manual_seed(0)
    cfg = TransformerConfig(d_model=32, n_heads=4, n_layers=2, d_ff=64, num_classes=2)
    net = OncoTrajTransformer(cfg).eval()

    seq_short = {
        "event_types": torch.tensor([[EVENT_TYPES["CLS"], EVENT_TYPES["variant_snv"]]]),
        "values": torch.tensor([[0.0, 0.3]]),
        "timestamps": torch.tensor([[0.0, 30.0]]),
        "gene_ids": torch.tensor([[0, GENE_VOCAB["EGFR"]]]),
        "attention_mask": torch.tensor([[True, True]]),
    }
    seq_padded = {
        "event_types": torch.tensor([[EVENT_TYPES["CLS"], EVENT_TYPES["variant_snv"], 0, 0]]),
        "values": torch.tensor([[0.0, 0.3, 0.0, 0.0]]),
        "timestamps": torch.tensor([[0.0, 30.0, 0.0, 0.0]]),
        "gene_ids": torch.tensor([[0, GENE_VOCAB["EGFR"], 0, 0]]),
        "attention_mask": torch.tensor([[True, True, False, False]]),
    }
    with torch.no_grad():
        out_short = net(**seq_short)
        out_padded = net(**seq_padded)

    torch.testing.assert_close(out_short["reg_pred"], out_padded["reg_pred"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        out_short["cls_logits"], out_padded["cls_logits"], atol=1e-5, rtol=1e-5
    )


def test_train_transformer_end_to_end(tmp_path, monkeypatch):
    """Run the training script's `train()` once on the synthetic build."""
    import argparse

    import build_dataset
    import train_transformer

    from oncotraj.data.splits import make_splits_from_parquet

    # MPS hits an empty-placeholder bug on tiny synthetic eval batches; force CPU.
    monkeypatch.setattr(train_transformer, "best_device", lambda: torch.device("cpu"))

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

    args = argparse.Namespace(
        parquet=output,
        splits=splits_path,
        output=tmp_path / "transformer_model",
        d_model=32,
        n_heads=4,
        n_layers=2,
        d_ff=64,
        dropout=0.1,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        push_to_hub=False,
        hub_repo_id="x/y",
        private=True,
        dry_run=False,
    )
    result = train_transformer.train(args)
    assert "n_params" in result
    assert (tmp_path / "transformer_model" / "final" / "pytorch_model.bin").exists()
    assert (tmp_path / "transformer_model" / "final" / "config.json").exists()
