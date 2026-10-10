"""Train the multi-task transformer baseline via Hugging Face Trainer.

Per the spec: train on local CPU/MPS today (so the pipeline is testable
without a GPU); when migrated to ZeroGPU / AutoTrain / A10G, the same
TrainingArguments work because we only depend on Trainer + the model's
forward signature.

Usage:
    # Local run (no hub push):
    python scripts/train_transformer.py \
        --parquet data/processed/oncotraj_v0 \
        --splits  data/processed/oncotraj_v0/_splits.json \
        --output  models/baselines/transformer

    # With hub push (requires write-scope HF_TOKEN):
    HF_TOKEN=$(grep HUGGINGFACE_API_TOKEN .env.local | cut -d= -f2) \
    python scripts/train_transformer.py \
        --parquet data/processed/oncotraj_v0 \
        --splits  data/processed/oncotraj_v0/_splits.json \
        --output  models/baselines/transformer \
        --push-to-hub --hub-repo-id span-ai-labs/oncotraj-tx-small --private

ZeroGPU / Spaces deployment notes
---------------------------------
ZeroGPU's `@spaces.GPU` decorator caps single calls at ~minutes, which is
unsuitable for a multi-epoch training run. The realistic cloud paths are:
- HF AutoTrain (paid) with this repo's code wrapped in an autotrain config.
- A dedicated HF Space using GPU runtime ("Spaces Hardware"), not ZeroGPU.
- HF Inference Endpoints in training mode.
This script is runtime-agnostic; bring your own A10G.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncotraj.data.splits import SplitManifest  # noqa: E402
from oncotraj.models.features import (  # noqa: E402
    BuiltTables,
    build_target_b,
    build_target_c,
)
from oncotraj.models.transformer import (  # noqa: E402
    EVENT_TYPES,
    GENE_VOCAB,
    OncoTrajTransformer,
    TransformerConfig,
    best_device,
    build_event_sequences,
    count_parameters,
    pad_event_batch,
)

# ---------------------------------------------------------------------------
# Dataset + collator for HF Trainer
# ---------------------------------------------------------------------------


class EventDataset(torch.utils.data.Dataset):
    def __init__(self, patient_ids, sequences, y_reg, y_cls, label_encoder):
        self.patient_ids = patient_ids
        self.sequences = sequences
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.label_encoder = label_encoder

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict:
        pid = self.patient_ids[idx]
        return {
            "pid": pid,
            "seq": self.sequences[pid],
            "reg_y": float(self.y_reg.loc[pid]) if pid in self.y_reg.index else 0.0,
            "reg_mask": 1.0 if pid in self.y_reg.index else 0.0,
            "cls_y": int(self.label_encoder.transform([self.y_cls.loc[pid]])[0])
            if pid in self.y_cls.index
            else 0,
            "cls_mask": 1.0 if pid in self.y_cls.index else 0.0,
        }


def collate(batch):
    seqs = [b["seq"] for b in batch]
    padded = pad_event_batch(seqs)
    padded["reg_y"] = torch.tensor([b["reg_y"] for b in batch], dtype=torch.float32)
    padded["reg_mask"] = torch.tensor([b["reg_mask"] for b in batch], dtype=torch.float32)
    padded["cls_y"] = torch.tensor([b["cls_y"] for b in batch], dtype=torch.long)
    padded["cls_mask"] = torch.tensor([b["cls_mask"] for b in batch], dtype=torch.float32)
    return padded


class MultiTaskTransformer(torch.nn.Module):
    """HF Trainer expects forward(**inputs) to return (loss, ...) or a dict.

    Wraps OncoTrajTransformer to compute the joint loss inside forward so
    Trainer can train it without custom compute_loss overrides.
    """

    def __init__(self, base: OncoTrajTransformer):
        super().__init__()
        self.base = base
        self.mse = torch.nn.MSELoss(reduction="none")
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        event_types,
        values,
        timestamps,
        gene_ids,
        attention_mask,
        reg_y=None,
        reg_mask=None,
        cls_y=None,
        cls_mask=None,
    ):
        out = self.base(
            event_types=event_types,
            values=values,
            timestamps=timestamps,
            gene_ids=gene_ids,
            attention_mask=attention_mask,
        )
        # HF Trainer requires loss to be a real tensor for both train and eval.
        if reg_y is not None and cls_y is not None:
            reg_loss = (
                self.mse(out["reg_pred"], reg_y) * reg_mask
            ).sum() / reg_mask.sum().clamp_min(1.0)
            cls_loss = (
                self.ce(out["cls_logits"], cls_y) * cls_mask
            ).sum() / cls_mask.sum().clamp_min(1.0)
            loss = reg_loss + cls_loss
        else:
            loss = torch.zeros((), device=out["reg_pred"].device, dtype=out["reg_pred"].dtype)
        return {"loss": loss, "reg_pred": out["reg_pred"], "cls_logits": out["cls_logits"]}


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def _load_tables(parquet_dir: Path) -> BuiltTables:
    return BuiltTables(
        patients=pd.read_parquet(parquet_dir / "patients.parquet"),
        variants=pd.read_parquet(parquet_dir / "variants.parquet"),
        treatments=pd.read_parquet(parquet_dir / "treatments.parquet"),
        outcomes=pd.read_parquet(parquet_dir / "outcomes.parquet"),
    )


def _split_index(manifest: SplitManifest, split: str) -> list[str]:
    return [pid for pid, s in manifest.assignments.items() if s == split]


def _hf_token() -> str | None:
    """Resolve a HF token from env, then from .env.local as a fallback."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        return token
    env_local = REPO_ROOT / ".env.local"
    if env_local.exists():
        for line in env_local.read_text().splitlines():
            if line.startswith("HUGGINGFACE_API_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def _verify_hub_write_scope(token: str, repo_id: str) -> tuple[bool, str]:
    """Check whether the token can write to the target repo's namespace.

    Tries the cheapest possible probe (whoami) and parses the returned scopes
    if available. We don't actually mutate anything until the user confirms.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        info = api.whoami(token=token)
    except Exception as e:
        return False, f"whoami failed: {e}"
    auth = info.get("auth", {})
    access_token = auth.get("accessToken", {})
    role = access_token.get("role", "unknown")
    fine_grained = access_token.get("fineGrained", {}) if isinstance(access_token, dict) else {}
    if role == "write":
        return True, "token role=write"
    # Fine-grained tokens carry a list of permitted scopes; look for write-repos.
    if fine_grained:
        scopes = fine_grained.get("scoped", []) or fine_grained.get("global", [])
        flat = []
        for s in scopes if isinstance(scopes, list) else []:
            if isinstance(s, dict):
                flat.extend(s.get("permissions", []) or [])
            else:
                flat.append(s)
        if any("write" in str(s).lower() or "repo" in str(s).lower() for s in flat):
            return True, f"fine-grained scopes look writeable for {repo_id}"
        return False, f"fine-grained scopes do not include write: {flat}"
    return False, f"token role={role}; cannot confirm write access. Regenerate with role=write."


def train(args: argparse.Namespace) -> dict:
    parquet_dir = args.parquet
    splits_path = args.splits
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = _load_tables(parquet_dir)
    manifest = SplitManifest.from_json(splits_path.read_text())

    y_reg = build_target_b(tables)
    y_cls = build_target_c(tables)
    train_pids = _split_index(manifest, "train")
    val_pids = _split_index(manifest, "val")
    test_pids = _split_index(manifest, "test")

    from sklearn.preprocessing import LabelEncoder

    # If the parquet has no Task C labels (synthetic-only build), pad with a
    # placeholder class so cross-entropy doesn't degenerate on num_classes=1.
    if y_cls.empty:
        from sklearn.preprocessing import LabelEncoder as _LE

        label_encoder = _LE()
        label_encoder.classes_ = np.array(["__no_label__", "__placeholder__"])
    else:
        label_encoder = LabelEncoder().fit(y_cls)
        if len(label_encoder.classes_) < 2:
            label_encoder.classes_ = np.append(label_encoder.classes_, "__placeholder__")
    num_classes = len(label_encoder.classes_)

    device = best_device()
    config = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        num_classes=num_classes,
    )
    base = OncoTrajTransformer(config)
    model = MultiTaskTransformer(base)
    n_params = count_parameters(base)
    print(f"Model parameters: {n_params:,} (~{n_params / 1e6:.1f}M)")
    if not (10e6 <= n_params <= 50e6):
        print("  WARNING: outside the 10-50M target band.")

    all_pids = list(set(train_pids + val_pids + test_pids))
    sequences = build_event_sequences(all_pids, tables.variants, tables.treatments, tables.outcomes)

    train_ds = EventDataset(train_pids, sequences, y_reg, y_cls, label_encoder)
    val_ds = EventDataset(val_pids, sequences, y_reg, y_cls, label_encoder)

    # transformers 5 dropped `warmup_ratio`; give the same 10% warmup as a step count.
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    targs = TrainingArguments(
        output_dir=str(output_dir / "hf_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-2,
        warmup_steps=math.ceil(0.1 * total_steps),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        # We don't gate on `eval_loss` because the custom forward signature
        # uses reg_y/cls_y rather than `labels`, so Trainer's default metric
        # aggregation doesn't pick it up. Best-of-epoch selection is enforced
        # at the wrapper level (TransformerModel.fit_multitask) for direct use.
        report_to=["none"],
        seed=42,
        push_to_hub=False,  # We push manually after verifying token scope.
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        use_cpu=(device.type == "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate,
    )

    # MLflow 3 refuses the filesystem backend unless asked; `mlruns/` is the
    # documented tracking location here, so opt in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(f"file://{(REPO_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment("oncotraj-baselines")
    run_name = f"transformer_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "transformer",
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": args.d_ff,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "device": str(device),
                "n_train": len(train_pids),
                "n_val": len(val_pids),
                "n_test": len(test_pids),
                "n_params": n_params,
            }
        )
        start = time.time()
        trainer.train()
        elapsed = time.time() - start
        mlflow.log_metric("train_time_seconds", elapsed)

        # Final evaluation on val and test.
        metrics = _evaluate(
            model, sequences, val_pids, y_reg, y_cls, label_encoder, device, args.batch_size
        )
        for k, v in metrics.items():
            mlflow.log_metric(f"val_{k}", v)
        test_metrics = _evaluate(
            model, sequences, test_pids, y_reg, y_cls, label_encoder, device, args.batch_size
        )
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        # Persist the model and its tokeniser config.
        artefact_dir = output_dir / "final"
        artefact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(base.state_dict(), artefact_dir / "pytorch_model.bin")
        (artefact_dir / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "oncotraj-transformer",
                    "d_model": config.d_model,
                    "n_heads": config.n_heads,
                    "n_layers": config.n_layers,
                    "d_ff": config.d_ff,
                    "dropout": config.dropout,
                    "num_classes": num_classes,
                    "classes": list(label_encoder.classes_),
                    "event_types": EVENT_TYPES,
                    "gene_vocab": GENE_VOCAB,
                    "n_params": n_params,
                },
                indent=2,
            )
            + "\n"
        )
        (artefact_dir / "README.md").write_text(_readme(args, n_params, metrics, test_metrics))
        mlflow.log_artifact(str(artefact_dir / "pytorch_model.bin"), "model")
        mlflow.log_artifact(str(artefact_dir / "config.json"), "model")

    result = {
        "n_params": n_params,
        "train_time_seconds": elapsed,
        "val_metrics": metrics,
        "test_metrics": test_metrics,
        "artefact_dir": str(artefact_dir),
    }

    # Optional hub push.
    if args.push_to_hub:
        push_result = _push_to_hub(
            artefact_dir, args.hub_repo_id, private=args.private, dry_run=args.dry_run
        )
        result["hub_push"] = push_result

    return result


def _evaluate(model, sequences, pids, y_reg, y_cls, label_encoder, device, batch_size):
    if not pids:
        return {}
    model.eval()
    model = model.to(device)
    reg_preds, cls_preds, reg_truth, cls_truth = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(pids), batch_size):
            chunk = pids[i : i + batch_size]
            seqs = [sequences[p] for p in chunk]
            batch = pad_event_batch(seqs)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.base(**batch)
            reg_preds.append(out["reg_pred"].cpu().numpy())
            cls_preds.append(out["cls_logits"].argmax(dim=-1).cpu().numpy())
            for p in chunk:
                reg_truth.append(y_reg.loc[p] if p in y_reg.index else None)
                cls_truth.append(y_cls.loc[p] if p in y_cls.index else None)
    reg_preds_arr = np.concatenate(reg_preds)
    cls_preds_arr = np.concatenate(cls_preds)

    metrics: dict[str, float] = {}
    reg_valid = [i for i, t in enumerate(reg_truth) if t is not None]
    if reg_valid:
        y_true = np.array([reg_truth[i] for i in reg_valid])
        y_pred = reg_preds_arr[reg_valid]
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    cls_valid = [i for i, t in enumerate(cls_truth) if t is not None]
    if cls_valid:
        y_true_cls = [cls_truth[i] for i in cls_valid]
        y_pred_cls = label_encoder.inverse_transform(cls_preds_arr[cls_valid])
        metrics["accuracy"] = float(accuracy_score(y_true_cls, y_pred_cls))
        metrics["macro_f1"] = float(
            f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
        )
    return metrics


def _readme(args: argparse.Namespace, n_params: int, val: dict, test: dict) -> str:
    return (
        f"# OncoTraj small transformer (v0)\n\n"
        f"- Parameters: {n_params:,} (~{n_params / 1e6:.1f}M).\n"
        f"- Architecture: {args.n_layers}-layer, d_model={args.d_model}, "
        f"n_heads={args.n_heads}, d_ff={args.d_ff}, RoPE on continuous timestamps.\n"
        f"- Training: HF Trainer, {args.epochs} epochs, lr={args.learning_rate}, "
        f"batch_size={args.batch_size}.\n\n"
        f"## Val metrics\n```\n{json.dumps(val, indent=2)}\n```\n\n"
        f"## Test metrics\n```\n{json.dumps(test, indent=2)}\n```\n\n"
        f"## Caveat\n"
        f"This artefact was trained on the v0 OncoTraj cohort which is dominated by "
        f"FLAURA + synthetic GENIE/CHORD; per-patient sequences average ~2 events. "
        f"Real signal will emerge once serial-ctDNA studies (TRACERx-EVO, registries) "
        f"are merged into the parquet. Do not use for any clinical decision.\n"
    )


def _push_to_hub(artefact_dir: Path, repo_id: str, private: bool, dry_run: bool) -> dict:
    token = _hf_token()
    if not token:
        return {"pushed": False, "reason": "no HF token found in env or .env.local"}

    ok, reason = _verify_hub_write_scope(token, repo_id)
    if not ok:
        return {
            "pushed": False,
            "reason": f"Token does not have write scope: {reason}. "
            "Regenerate a write-scope token at https://huggingface.co/settings/tokens.",
        }

    if dry_run:
        return {
            "pushed": False,
            "reason": "dry-run: token has write scope, would push.",
            "scope": reason,
        }

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)
    create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True, token=token)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(artefact_dir),
        repo_type="model",
        token=token,
    )
    return {"pushed": True, "repo_id": repo_id, "private": private, "scope": reason}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=1536)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo-id", default="span-ai-labs/oncotraj-tx-small")
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify the HF token's write scope without uploading.",
    )
    args = parser.parse_args()

    # Quiet down a noisy HF warning about MPS support.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    result = train(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    # Cleanup HF Trainer's temp checkpoint dirs from prior runs.
    main()
