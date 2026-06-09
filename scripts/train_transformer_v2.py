"""Train the multi-task transformer baseline with explicit, no-silent-failure
training loop. Replaces the older HF-Trainer-driven `train_transformer.py`
for the v0 paper rerun per audit P0-3.

Three fixed configs (small / medium / large d_model). Each run:
- writes a checkpoint per epoch under `models/baselines/transformer_v2/<id>/`
- logs train_loss, val_loss, val_mae_days, val_macro_f1 at every epoch end
- asserts non-NaN loss and bails immediately on divergence
- saves predictions.csv for the eval harness
- applies the same log1p target transform as the LSTM (P0-2) so Task B is
  learnable on raw-day targets.

Usage:
    python scripts/train_transformer_v2.py \\
        --parquet data/processed/oncotraj_v0 \\
        --splits  data/processed/oncotraj_v0/_splits.json \\
        --output  models/baselines/transformer_v2

By default runs ALL three configs sequentially. Use --config {small,medium,
large} to run just one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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
from sklearn.preprocessing import LabelEncoder
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncotraj.data.splits import SplitManifest  # noqa: E402
from oncotraj.models.features import (  # noqa: E402
    BuiltTables,
    build_target_b,
    build_target_c,
)
from oncotraj.models.transformer import (  # noqa: E402
    OncoTrajTransformer,
    TransformerConfig,
    best_device,
    build_event_sequences,
    count_parameters,
    pad_event_batch,
)

# ---------------------------------------------------------------------------
# Three locked configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    name: str
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    learning_rate: float
    epochs: int
    batch_size: int


CONFIGS: dict[str, RunConfig] = {
    "small": RunConfig(
        "small",
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        learning_rate=3e-4,
        epochs=40,
        batch_size=16,
    ),
    "medium": RunConfig(
        "medium",
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        learning_rate=2e-4,
        epochs=40,
        batch_size=16,
    ),
    "large": RunConfig(
        "large",
        d_model=384,
        n_heads=8,
        n_layers=8,
        d_ff=1536,
        learning_rate=1e-4,
        epochs=40,
        batch_size=16,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
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


def _make_batches(
    pids: list[str],
    sequences: dict,
    y_reg: pd.Series,
    y_cls: pd.Series,
    label_encoder: LabelEncoder,
    batch_size: int,
    target_transform: str,
):
    """Yield (batch_dict, reg_y, reg_mask, cls_y, cls_mask) tuples."""
    for i in range(0, len(pids), batch_size):
        chunk = pids[i : i + batch_size]
        seqs = [sequences[p] for p in chunk]
        batch = pad_event_batch(seqs)

        reg_y = torch.zeros(len(chunk), dtype=torch.float32)
        reg_mask = torch.zeros(len(chunk), dtype=torch.float32)
        cls_y = torch.zeros(len(chunk), dtype=torch.long)
        cls_mask = torch.zeros(len(chunk), dtype=torch.float32)
        for j, p in enumerate(chunk):
            if p in y_reg.index:
                raw = float(y_reg.loc[p])
                reg_y[j] = float(np.log1p(raw)) if target_transform == "log1p" else raw
                reg_mask[j] = 1.0
            if p in y_cls.index:
                cls_y[j] = int(label_encoder.transform([y_cls.loc[p]])[0])
                cls_mask[j] = 1.0
        yield chunk, batch, reg_y, reg_mask, cls_y, cls_mask


def _inverse_transform(pred: np.ndarray, target_transform: str) -> np.ndarray:
    if target_transform == "log1p":
        return np.clip(np.expm1(pred), 0.0, None)
    return pred


def _evaluate(
    net,
    sequences,
    pids: list[str],
    y_reg: pd.Series,
    y_cls: pd.Series,
    label_encoder: LabelEncoder,
    device,
    batch_size: int,
    target_transform: str,
) -> dict[str, float]:
    if not pids:
        return {}
    net.eval()
    reg_preds_all: list[np.ndarray] = []
    cls_preds_all: list[np.ndarray] = []
    pids_ordered: list[str] = []
    with torch.no_grad():
        for chunk, batch, *_ in _make_batches(
            pids, sequences, y_reg, y_cls, label_encoder, batch_size, target_transform
        ):
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = net(**batch_dev)
            reg_preds_all.append(out["reg_pred"].cpu().numpy())
            cls_preds_all.append(out["cls_logits"].argmax(dim=-1).cpu().numpy())
            pids_ordered.extend(chunk)
    reg_pred_arr = _inverse_transform(np.concatenate(reg_preds_all), target_transform)
    cls_pred_arr = np.concatenate(cls_preds_all)

    metrics: dict[str, float] = {}
    reg_pids = [p for p in pids_ordered if p in y_reg.index]
    if reg_pids:
        idx = [pids_ordered.index(p) for p in reg_pids]
        y_true_reg = np.array([y_reg.loc[p] for p in reg_pids], dtype=float)
        y_pred_reg = reg_pred_arr[idx]
        metrics["mae"] = float(mean_absolute_error(y_true_reg, y_pred_reg))
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_reg, y_pred_reg)))

    cls_pids = [p for p in pids_ordered if p in y_cls.index]
    if cls_pids:
        idx = [pids_ordered.index(p) for p in cls_pids]
        y_true_cls = [y_cls.loc[p] for p in cls_pids]
        y_pred_cls = label_encoder.inverse_transform(cls_pred_arr[idx])
        metrics["accuracy"] = float(accuracy_score(y_true_cls, y_pred_cls))
        metrics["macro_f1"] = float(
            f1_score(y_true_cls, y_pred_cls, average="macro", zero_division=0)
        )
    net.train()
    return metrics


# ---------------------------------------------------------------------------
# Main training loop (one config)
# ---------------------------------------------------------------------------


def train_one_config(
    config: RunConfig,
    parquet_dir: Path,
    splits_path: Path,
    output_dir: Path,
    target_transform: str = "log1p",
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    tables = _load_tables(parquet_dir)
    manifest = SplitManifest.from_json(splits_path.read_text())
    train_pids = _split_index(manifest, "train")
    val_pids = _split_index(manifest, "val")
    test_pids = _split_index(manifest, "test")

    y_reg = build_target_b(tables)
    y_cls = build_target_c(tables)

    # Pad to ≥2 classes (synthetic builds can produce 0 Task C labels).
    if y_cls.empty:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(["__no_label__", "__placeholder__"])
    else:
        label_encoder = LabelEncoder().fit(y_cls)
        if len(label_encoder.classes_) < 2:
            label_encoder.classes_ = np.append(label_encoder.classes_, "__placeholder__")
    num_classes = len(label_encoder.classes_)

    tx_config = TransformerConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        dropout=0.1,
        num_classes=num_classes,
    )
    device = best_device()
    net = OncoTrajTransformer(tx_config).to(device)
    n_params = count_parameters(net)

    all_pids = list(set(train_pids + val_pids + test_pids))
    sequences = build_event_sequences(all_pids, tables.variants, tables.treatments, tables.outcomes)

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, weight_decay=1e-2)
    mse = nn.MSELoss(reduction="none")
    # P1: class-weighted CE (inverse class frequencies from training labels).
    ce_weight = None
    if not y_cls.empty and num_classes > 1:
        counts = y_cls.value_counts().reindex(label_encoder.classes_).fillna(1.0).to_numpy()
        inv = counts.sum() / (len(counts) * counts)
        ce_weight = torch.tensor(inv, dtype=torch.float32, device=device)
    ce = nn.CrossEntropyLoss(reduction="none", weight=ce_weight)

    run_dir = output_dir / config.name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    mlflow.set_tracking_uri(f"file://{(REPO_ROOT / 'mlruns').as_posix()}")
    mlflow.set_experiment("oncotraj-baselines")
    run_name = f"transformer_v2_{config.name}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    history: list[dict[str, float]] = []
    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    rng = np.random.default_rng(seed)
    start = time.time()

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "model": "transformer_v2",
                "config_name": config.name,
                **{k: v for k, v in asdict(config).items() if k != "name"},
                "n_params": n_params,
                "device": str(device),
                "n_train": len(train_pids),
                "n_val": len(val_pids),
                "n_test": len(test_pids),
                "target_transform": target_transform,
                "seed": seed,
            }
        )

        for epoch in range(config.epochs):
            net.train()
            order = list(train_pids)
            rng.shuffle(order)
            train_loss = 0.0
            n = 0
            for _chunk, batch, reg_y, reg_mask, cls_y, cls_mask in _make_batches(
                order,
                sequences,
                y_reg,
                y_cls,
                label_encoder,
                config.batch_size,
                target_transform,
            ):
                batch_dev = {k: v.to(device) for k, v in batch.items()}
                reg_y_d, reg_mask_d = reg_y.to(device), reg_mask.to(device)
                cls_y_d, cls_mask_d = cls_y.to(device), cls_mask.to(device)
                out = net(**batch_dev)
                reg_loss = (
                    mse(out["reg_pred"], reg_y_d) * reg_mask_d
                ).sum() / reg_mask_d.sum().clamp_min(1.0)
                cls_loss = (
                    ce(out["cls_logits"], cls_y_d) * cls_mask_d
                ).sum() / cls_mask_d.sum().clamp_min(1.0)
                loss = reg_loss + cls_loss
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Non-finite loss at epoch {epoch}: "
                        f"reg_loss={reg_loss.item()} cls_loss={cls_loss.item()}. "
                        "Refusing to continue — see P0-3 acceptance criterion."
                    )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += float(loss.detach().cpu()) * len(_chunk)
                n += len(_chunk)
            train_loss /= max(n, 1)

            # Explicit per-epoch val metrics. Failure to log == silent failure
            # — exactly what P0-3 says we should never accept again.
            val_metrics = _evaluate(
                net,
                sequences,
                val_pids,
                y_reg,
                y_cls,
                label_encoder,
                device,
                config.batch_size,
                target_transform,
            )
            val_loss = val_metrics.get("mae", math.nan)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            # Best-epoch checkpoint.
            ckpt = run_dir / "checkpoints" / f"epoch_{epoch:03d}.pt"
            torch.save(net.state_dict(), ckpt)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

        # Restore best.
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()

        test_metrics = _evaluate(
            net,
            sequences,
            test_pids,
            y_reg,
            y_cls,
            label_encoder,
            device,
            config.batch_size,
            target_transform,
        )
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("train_time_seconds", time.time() - start)

        # Final artefacts.
        final_path = run_dir / "final.pt"
        torch.save(net.state_dict(), final_path)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    **asdict(config),
                    "n_params": n_params,
                    "num_classes": num_classes,
                    "classes": list(label_encoder.classes_),
                },
                indent=2,
            )
            + "\n"
        )

        # Predictions CSV for the eval harness.
        preds_test = _evaluate_predictions(
            net,
            sequences,
            test_pids,
            y_reg,
            y_cls,
            label_encoder,
            device,
            config.batch_size,
            target_transform,
        )
        preds_val = _evaluate_predictions(
            net,
            sequences,
            val_pids,
            y_reg,
            y_cls,
            label_encoder,
            device,
            config.batch_size,
            target_transform,
        )
        pd.concat([preds_test.assign(split="test"), preds_val.assign(split="val")]).to_csv(
            run_dir / "predictions.csv", index=False
        )

    return {
        "config": config.name,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_mae": best_val_loss,
        "test_metrics": test_metrics,
        "history_path": str(run_dir / "history.json"),
    }


def _evaluate_predictions(
    net,
    sequences,
    pids,
    y_reg,
    y_cls,
    label_encoder,
    device,
    batch_size,
    target_transform,
) -> pd.DataFrame:
    """Like _evaluate, but returns the raw per-patient predictions DataFrame."""
    if not pids:
        return pd.DataFrame()
    net.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for chunk, batch, *_ in _make_batches(
            pids, sequences, y_reg, y_cls, label_encoder, batch_size, target_transform
        ):
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = net(**batch_dev)
            reg_p = _inverse_transform(out["reg_pred"].cpu().numpy(), target_transform)
            cls_p = out["cls_logits"].argmax(dim=-1).cpu().numpy()
            cls_labels = label_encoder.inverse_transform(cls_p)
            for j, p in enumerate(chunk):
                row = {"patient_id": p}
                if p in y_reg.index:
                    row["task_b_true"] = float(y_reg.loc[p])
                    row["task_b_pred"] = float(reg_p[j])
                if p in y_cls.index:
                    row["task_c_true"] = str(y_cls.loc[p])
                    row["task_c_pred"] = str(cls_labels[j])
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--config",
        choices=[*list(CONFIGS), "all"],
        default="all",
        help="Which config to run (default: all three).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configs_to_run = list(CONFIGS) if args.config == "all" else [args.config]
    results = []
    for name in configs_to_run:
        cfg = CONFIGS[name]
        print(f"=== running {cfg.name} ({cfg.d_model}-d, {cfg.n_layers}-L) ===")
        result = train_one_config(
            config=cfg,
            parquet_dir=args.parquet,
            splits_path=args.splits,
            output_dir=args.output,
            seed=args.seed,
        )
        results.append(result)
        print(json.dumps(result, indent=2, default=str))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2, default=str) + "\n")


if __name__ == "__main__":
    main()
