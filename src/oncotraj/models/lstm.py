"""Multi-task LSTM baseline.

Per PAPER_OUTLINE.md §6.1: two hidden layers, hidden=256, sequence-aware
non-attention baseline. This module is the v1 scaffold; the dataset's
sequence lengths are short today (FLAURA: 1-2 samples/patient; synthetic
GENIE/CHORD: 1) so the recurrent dynamics will only pay off once real
serial ctDNA (TRACERx, registries) is in.

Architecture:
- Per-timestep input: aggregate per-sample features
  [days_from_first_sample, max_vaf, mean_vaf, variant_count,
   egfr_variant_count, met_variant_count, erbb2_variant_count,
   c797s_present, baseline_flag] (9 features).
- Shared LSTM trunk: 2 layers, hidden_size configurable, batch_first.
- Two heads:
    - regression head -> 1 scalar (Task B, days-to-progression).
    - classification head -> n_classes logits (Task C, mechanism).
- Joint training: combined loss = mse(reg) + cross_entropy(cls), with
  per-patient masking for patients missing one of the two targets.
- Packed sequences via torch.nn.utils.rnn.pack_padded_sequence so
  variable-length per-patient trajectories don't waste compute on padding.

Device: MPS when available on M-series Mac, otherwise CPU. CUDA is
opportunistically used if torch sees it.

Interface: conforms to OncoTrajModel via `LSTMModel(is_classifier=...)`.
At fit time the wrapper auto-detects whether `y` is regression or
classification and trains only the matching head, leaving the other
inactive. For joint training across both heads use the helper
`fit_multitask()` and `predict_taskB()` / `predict_taskC()` directly.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from ..parsers.common import C797S_RE
from .base import OncoTrajModel, register_model

PerSampleFeatureDim = 9
DEFAULT_HIDDEN = 64
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 32
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.1
DEFAULT_PATIENCE = 25


def best_device() -> torch.device:
    """Prefer MPS on Apple silicon, fall back to CPU. CUDA is also OK if seen."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


def build_sequences(variants: pd.DataFrame, patient_ids: Sequence[str]) -> dict[str, np.ndarray]:
    """Build per-patient (T, F) feature sequences from the variants table.

    One sample-event per timestep, ordered by `sample_date`. The chosen
    aggregations are simple and dataset-agnostic; replace with richer
    per-variant tokenisation when serial ctDNA reaches enough density to
    justify it.
    """
    if variants.empty:
        return {pid: np.zeros((1, PerSampleFeatureDim), dtype=np.float32) for pid in patient_ids}

    df = variants.copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"])
    df = df.sort_values(["patient_id", "sample_date"])

    sequences: dict[str, np.ndarray] = {}
    for pid in patient_ids:
        sub = df.loc[df["patient_id"] == pid]
        if sub.empty:
            sequences[pid] = np.zeros((1, PerSampleFeatureDim), dtype=np.float32)
            continue
        first_date = sub["sample_date"].iloc[0]
        rows: list[list[float]] = []
        for sample_date, samp in sub.groupby("sample_date", sort=True):
            vafs = samp["vaf"].where(samp["vaf"] >= 0).dropna()
            max_vaf = float(vafs.max()) if not vafs.empty else 0.0
            mean_vaf = float(vafs.mean()) if not vafs.empty else 0.0
            variant_count = float(len(samp))
            egfr = float((samp["gene_symbol"] == "EGFR").sum())
            met = float((samp["gene_symbol"] == "MET").sum())
            erbb2 = float((samp["gene_symbol"] == "ERBB2").sum())
            c797s_present = float(
                samp["protein_change_hgvs"]
                .fillna("")
                .apply(lambda s: bool(C797S_RE.search(s)))
                .any()
            )
            baseline_flag = float(sample_date == first_date)
            days_from_first = float((sample_date - first_date).days)
            rows.append(
                [
                    days_from_first,
                    max_vaf,
                    mean_vaf,
                    variant_count,
                    egfr,
                    met,
                    erbb2,
                    c797s_present,
                    baseline_flag,
                ]
            )
        sequences[pid] = np.asarray(rows, dtype=np.float32)
    return sequences


# ---------------------------------------------------------------------------
# Dataset + collate
# ---------------------------------------------------------------------------


class PatientSequenceDataset(Dataset):
    """In-memory dataset of (sequence, regression_target, class_target, mask) tuples."""

    def __init__(
        self,
        patient_ids: list[str],
        sequences: dict[str, np.ndarray],
        y_reg: pd.Series | None,
        y_cls: pd.Series | None,
        cls_classes: np.ndarray | None,
        scaler: StandardScaler | None,
        target_transform: str = "log1p",
    ):
        self.patient_ids = patient_ids
        self.sequences = sequences
        self.y_reg = y_reg
        self.y_cls = y_cls
        self.cls_classes = cls_classes
        self.scaler = scaler
        self.target_transform = target_transform
        if cls_classes is not None:
            self._class_index = {c: i for i, c in enumerate(cls_classes)}
        else:
            self._class_index = {}

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        pid = self.patient_ids[idx]
        seq = self.sequences[pid].astype(np.float32)
        if self.scaler is not None:
            seq = self.scaler.transform(seq).astype(np.float32)

        reg_val, reg_mask = 0.0, 0.0
        if self.y_reg is not None and pid in self.y_reg.index:
            raw = float(self.y_reg.loc[pid])
            # log1p-transform the target so MSE doesn't see raw-day magnitudes
            # (700-day errors squared blow up gradient norms). The model
            # output is inverse-transformed back to days at predict time.
            reg_val = float(np.log1p(raw)) if self.target_transform == "log1p" else raw
            reg_mask = 1.0

        cls_idx, cls_mask = 0, 0.0
        if self.y_cls is not None and pid in self.y_cls.index:
            cls_idx = self._class_index.get(self.y_cls.loc[pid], 0)
            cls_mask = 1.0

        return {
            "pid": pid,
            "seq": torch.from_numpy(seq),
            "length": seq.shape[0],
            "reg": torch.tensor(reg_val, dtype=torch.float32),
            "reg_mask": torch.tensor(reg_mask, dtype=torch.float32),
            "cls": torch.tensor(cls_idx, dtype=torch.long),
            "cls_mask": torch.tensor(cls_mask, dtype=torch.float32),
        }


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length sequences for packing; sort by length descending."""
    batch = sorted(batch, key=lambda x: x["length"], reverse=True)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_len = int(lengths.max())
    feat_dim = batch[0]["seq"].shape[-1]
    padded = torch.zeros(len(batch), max_len, feat_dim, dtype=torch.float32)
    for i, b in enumerate(batch):
        padded[i, : b["length"]] = b["seq"]
    return {
        "pids": [b["pid"] for b in batch],
        "seq": padded,
        "lengths": lengths,
        "reg": torch.stack([b["reg"] for b in batch]),
        "reg_mask": torch.stack([b["reg_mask"] for b in batch]),
        "cls": torch.stack([b["cls"] for b in batch]),
        "cls_mask": torch.stack([b["cls_mask"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class OncoLSTMNet(nn.Module):
    """LSTM trunk with two task-specific linear heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.reg_head = nn.Linear(hidden_size, 1)
        self.cls_head = nn.Linear(hidden_size, num_classes)

    def forward(
        self, seq: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # MPS does not support packed-sequence with CPU lengths cleanly; pack
        # always on CPU and let the LSTM handle device transfer.
        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]  # (batch, hidden) — top layer's final hidden state
        return self.reg_head(last).squeeze(-1), self.cls_head(last)


# ---------------------------------------------------------------------------
# Wrapper conforming to OncoTrajModel
# ---------------------------------------------------------------------------


@register_model
class LSTMModel(OncoTrajModel):
    name = "lstm"

    def __init__(
        self,
        is_classifier: bool = True,
        hidden_size: int = DEFAULT_HIDDEN,
        num_layers: int = DEFAULT_NUM_LAYERS,
        dropout: float = DEFAULT_DROPOUT,
        learning_rate: float = DEFAULT_LR,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        patience: int = DEFAULT_PATIENCE,
        random_state: int = 42,
        device: torch.device | None = None,
        variants_df: pd.DataFrame | None = None,
        target_transform: str = "log1p",
        **_: Any,
    ) -> None:
        super().__init__()
        self.is_classifier = is_classifier
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.device = device or best_device()
        # log1p maps raw-day targets (1..3000) into ~7-unit range. Inverse is
        # expm1 at predict time. STEP12_RESULTS_REVIEW.md P0-2: pre-fix LSTM
        # never learned because raw-day MSE gradients saturated.
        self.target_transform = target_transform
        # Variants are a separate input source that the LSTM needs in addition
        # to the X DataFrame the single-task interface provides. The training
        # script attaches the variants DataFrame here via the `variants_df`
        # kwarg or `set_variants()`.
        self._variants: pd.DataFrame | None = variants_df
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._net: OncoLSTMNet | None = None

    def set_variants(self, variants: pd.DataFrame) -> LSTMModel:
        self._variants = variants
        return self

    # -- joint multi-task fit --------------------------------------------
    def fit_multitask(
        self,
        patient_ids: list[str],
        variants: pd.DataFrame,
        y_reg: pd.Series | None,
        y_cls: pd.Series | None,
        val_patient_ids: list[str] | None = None,
    ) -> LSTMModel:
        """Joint training with both heads. Patients missing one target are
        masked from that loss; the LSTM trunk still sees their sequences."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self._variants = variants

        train_seqs = build_sequences(variants, patient_ids)

        all_train_rows = np.concatenate([s for s in train_seqs.values() if s.size > 0], axis=0)
        self._scaler = StandardScaler().fit(all_train_rows)

        cls_classes: np.ndarray | None = None
        if y_cls is not None and not y_cls.empty:
            self._label_encoder = LabelEncoder().fit(y_cls)
            cls_classes = self._label_encoder.classes_
            self.classes_ = cls_classes

        num_classes = max(len(cls_classes) if cls_classes is not None else 0, 1)
        self._net = OncoLSTMNet(
            input_dim=PerSampleFeatureDim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_classes=num_classes,
            dropout=self.dropout,
        ).to(self.device)

        train_ds = PatientSequenceDataset(
            patient_ids=patient_ids,
            sequences=train_seqs,
            y_reg=y_reg,
            y_cls=y_cls,
            cls_classes=cls_classes,
            scaler=self._scaler,
            target_transform=self.target_transform,
        )
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=pad_collate
        )

        val_loader: DataLoader | None = None
        if val_patient_ids:
            val_seqs = build_sequences(variants, val_patient_ids)
            val_ds = PatientSequenceDataset(
                patient_ids=val_patient_ids,
                sequences=val_seqs,
                y_reg=y_reg,
                y_cls=y_cls,
                cls_classes=cls_classes,
                scaler=self._scaler,
                target_transform=self.target_transform,
            )
            val_loader = DataLoader(
                val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate
            )

        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)
        mse = nn.MSELoss(reduction="none")
        # P1: class-weighted CE to avoid mechanism-class collapse (Task C).
        # Weights are inverse-frequency from the training labels, with a
        # uniform fallback when y_cls is missing or single-class.
        ce_weight = None
        if y_cls is not None and not y_cls.empty and cls_classes is not None and len(cls_classes) > 1:
            counts = y_cls.value_counts().reindex(cls_classes).fillna(1.0).to_numpy()
            inv = counts.sum() / (len(counts) * counts)
            ce_weight = torch.tensor(inv, dtype=torch.float32, device=self.device)
        ce = nn.CrossEntropyLoss(reduction="none", weight=ce_weight)

        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
        patience_counter = 0
        self.training_history_: list[dict[str, float]] = []

        for epoch in range(self.epochs):
            self._net.train()
            train_loss = self._run_epoch(train_loader, optimizer, mse, ce, training=True)
            val_loss = (
                self._run_epoch(val_loader, optimizer, mse, ce, training=False)
                if val_loader is not None
                else train_loss
            )
            self.training_history_.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            )
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        self._net.load_state_dict(best_state)
        self._net.eval()
        self._fitted = True
        return self

    def _run_epoch(self, loader, optimizer, mse, ce, training: bool) -> float:
        if loader is None:
            return float("nan")
        total, n = 0.0, 0
        for batch in loader:
            seq = batch["seq"].to(self.device)
            lengths = batch["lengths"]
            reg_y = batch["reg"].to(self.device)
            reg_mask = batch["reg_mask"].to(self.device)
            cls_y = batch["cls"].to(self.device)
            cls_mask = batch["cls_mask"].to(self.device)

            reg_pred, cls_logits = self._net(seq, lengths)

            reg_loss_per = mse(reg_pred, reg_y) * reg_mask
            reg_loss = reg_loss_per.sum() / reg_mask.sum().clamp_min(1.0)
            cls_loss_per = ce(cls_logits, cls_y) * cls_mask
            cls_loss = cls_loss_per.sum() / cls_mask.sum().clamp_min(1.0)
            loss = reg_loss + cls_loss

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += float(loss.detach().cpu()) * seq.size(0)
            n += seq.size(0)
        return total / max(n, 1)

    # -- single-task adapter to the OncoTrajModel interface --------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> LSTMModel:
        """`X` is ignored (only used for its patient_id index). Variants come
        from `self._variants` which the training script attaches."""
        if self._variants is None:
            raise RuntimeError(
                "LSTMModel.fit requires variants. Call set_variants(variants_df) first."
            )
        self.feature_names_ = list(X.columns)
        patient_ids = list(X.index)
        if self.is_classifier:
            self.fit_multitask(patient_ids, self._variants, y_reg=None, y_cls=y)
        else:
            self.fit_multitask(patient_ids, self._variants, y_reg=y, y_cls=None)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        reg, logits = self._infer(X)
        if self.is_classifier:
            idx = logits.argmax(axis=1)
            if self._label_encoder is not None:
                return self._label_encoder.inverse_transform(idx)
            return idx
        return reg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise NotImplementedError("LSTMModel is in regression mode.")
        self._check_fitted()
        _, logits = self._infer(X)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    def _infer(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self._net is not None and self._variants is not None
        patient_ids = list(X.index)
        sequences = build_sequences(self._variants, patient_ids)
        ds = PatientSequenceDataset(
            patient_ids=patient_ids,
            sequences=sequences,
            y_reg=None,
            y_cls=None,
            cls_classes=self.classes_ if self._label_encoder is not None else None,
            scaler=self._scaler,
            target_transform=self.target_transform,
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False, collate_fn=pad_collate)
        regs, all_logits, ordered_pids = [], [], []
        with torch.no_grad():
            for batch in loader:
                seq = batch["seq"].to(self.device)
                reg, logits = self._net(seq, batch["lengths"])
                regs.append(reg.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                ordered_pids.extend(batch["pids"])
        reg_arr = np.concatenate(regs) if regs else np.array([])
        logits_arr = np.concatenate(all_logits) if all_logits else np.array([])
        # The network's regression head outputs in transformed space (log1p
        # of days). Map back to days before returning so downstream metric
        # code sees real-day predictions.
        if self.target_transform == "log1p" and reg_arr.size > 0:
            reg_arr = np.expm1(reg_arr).astype(np.float32)
            # Clamp negatives that can arise from extrapolation; days >= 0.
            reg_arr = np.clip(reg_arr, 0.0, None)
        # Reorder predictions back to the input X's order.
        order_map = {pid: i for i, pid in enumerate(ordered_pids)}
        order = np.array([order_map[pid] for pid in patient_ids])
        return reg_arr[order], logits_arr[order]


# Silence a noisy MPS warning that fires on small batches; harmless.
warnings.filterwarnings("ignore", message=".*Mixed memory format.*", category=UserWarning)
