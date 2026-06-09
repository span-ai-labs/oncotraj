"""Small multi-task Transformer baseline over typed clinical events.

Per PAPER_OUTLINE.md §6.1: ~10M-param encoder, sequence-aware attention
baseline. This is the fifth locked baseline; pairs with the LSTM.

Event representation
--------------------
Each per-patient sequence is a list of typed clinical events:

    Event(event_type, value, timestamp_days, gene_id)

`event_type` enumerates a small fixed vocabulary (variant_snv, variant_cnv_amp,
variant_rearrangement, variant_splice, osi_start, osi_stop, progression,
last_followup, death). `value` is the relevant continuous quantity for the
event (VAF for variant events, 0.0 elsewhere). `timestamp_days` is days from
the earliest event in the patient's sequence. `gene_id` indexes a small gene
vocabulary for variant events; 0 (PAD) for non-variant events.

This is a richer view than the LSTM's per-sample aggregate — every variant
gets its own token instead of being summed across a draw.

Position encoding
-----------------
Rotary Position Embeddings (RoPE) applied to Q and K **per-head, per-token,
parameterised by the actual timestamp_days delta** rather than the integer
position index. The rotation angles for the i-th feature pair are
`timestamp_days * base ** (-2i / d_head)`. This makes attention timestamp-
aware while inheriting RoPE's properties (relative-position invariance,
exact recovery of inner-product on identical timestamps).

Heads
-----
Two task-specific linear heads on a `[CLS]`-style pooled representation:
- regression head -> 1 scalar (Task B, days-to-progression).
- classification head -> n_classes logits (Task C, mechanism).

Trained jointly with masked combined loss; same conventions as the LSTM.

Interface
---------
The model conforms to the OncoTrajModel interface via `TransformerModel`,
the wrapper that builds event sequences from the parquet variants+outcomes+
treatments tables. The underlying `OncoTrajTransformer` (nn.Module) is also
exposed for direct integration with the HF Trainer in `scripts/train_transformer.py`.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch import nn

from ..parsers.common import C797S_RE
from .base import OncoTrajModel, register_model

# ---------------------------------------------------------------------------
# Event vocabulary
# ---------------------------------------------------------------------------

EVENT_TYPES: dict[str, int] = {
    "PAD": 0,
    "CLS": 1,
    "variant_snv": 2,
    "variant_cnv_amp": 3,
    "variant_rearrangement": 4,
    "variant_splice_other": 5,
    "osi_start": 6,
    "osi_stop": 7,
    "progression": 8,
    "last_followup": 9,
    "death": 10,
}
NUM_EVENT_TYPES = len(EVENT_TYPES)

# Top-N genes seen in the FLAURA + GENIE + cBioPortal builds. Anything else
# maps to UNK. Keep this small so `gene_id` embedding stays cheap.
GENE_VOCAB: dict[str, int] = {
    "PAD": 0,
    "UNK": 1,
    "EGFR": 2,
    "TP53": 3,
    "MET": 4,
    "ERBB2": 5,
    "CCND1": 6,
    "CCND2": 7,
    "CCND3": 8,
    "CCNE1": 9,
    "PIK3CA": 10,
    "KRAS": 11,
    "BRAF": 12,
    "APC": 13,
    "PTEN": 14,
    "CTNNB1": 15,
    "FBXW7": 16,
    "RB1": 17,
    "CDKN2A": 18,
    "MYC": 19,
    "GNAS": 20,
}
NUM_GENES = len(GENE_VOCAB)

# Special tokens are also patient-level "events" at the head of the sequence.
CLS_EVENT_TYPE = EVENT_TYPES["CLS"]
PAD_EVENT_TYPE = EVENT_TYPES["PAD"]


# ---------------------------------------------------------------------------
# Sequence construction
# ---------------------------------------------------------------------------


@dataclass
class EventSequence:
    event_types: np.ndarray  # (T,) int64
    values: np.ndarray  # (T,) float32
    timestamps: np.ndarray  # (T,) float32, days from first event
    gene_ids: np.ndarray  # (T,) int64

    @property
    def length(self) -> int:
        return self.event_types.shape[0]


def _gene_id(gene: object) -> int:
    if not isinstance(gene, str):
        return GENE_VOCAB["UNK"]
    return GENE_VOCAB.get(gene.upper(), GENE_VOCAB["UNK"])


def _variant_event_type(alteration_type: object, protein_change: object) -> int:
    """Map a variant row to a single event_type id."""
    if isinstance(protein_change, str) and C797S_RE.search(protein_change):
        return EVENT_TYPES["variant_snv"]
    if isinstance(alteration_type, str):
        s = alteration_type.lower()
        if "snv" in s or "splice" in s:
            return EVENT_TYPES["variant_snv"] if "snv" in s else EVENT_TYPES["variant_splice_other"]
        if "indel" in s:
            return EVENT_TYPES["variant_snv"]
        if "cnv_amplification" in s:
            return EVENT_TYPES["variant_cnv_amp"]
        if "rearrangement" in s or "fusion" in s:
            return EVENT_TYPES["variant_rearrangement"]
    return EVENT_TYPES["variant_splice_other"]


def build_event_sequences(
    patient_ids: list[str],
    variants: pd.DataFrame,
    treatments: pd.DataFrame | None,
    outcomes: pd.DataFrame | None,
) -> dict[str, EventSequence]:
    """Per-patient typed event sequences for the transformer.

    Each sequence is prefixed with a CLS token at timestamp 0 (used for
    pooling). Variants contribute one event each; treatments and outcomes
    contribute one event per row.
    """
    sequences: dict[str, EventSequence] = {}

    if variants is None or variants.empty:
        variants = pd.DataFrame(columns=variants.columns if variants is not None else [])
    if treatments is None:
        treatments = pd.DataFrame()
    if outcomes is None:
        outcomes = pd.DataFrame()

    if not variants.empty:
        variants = variants.copy()
        variants["sample_date"] = pd.to_datetime(variants["sample_date"], errors="coerce")
    if not treatments.empty:
        treatments = treatments.copy()
        treatments["start_date"] = pd.to_datetime(treatments["start_date"], errors="coerce")
        treatments["end_date"] = pd.to_datetime(treatments["end_date"], errors="coerce")
    if not outcomes.empty:
        outcomes = outcomes.copy()
        outcomes["event_date"] = pd.to_datetime(outcomes["event_date"], errors="coerce")

    for pid in patient_ids:
        rows: list[tuple[int, float, pd.Timestamp, int]] = []

        v_sub = (
            variants.loc[variants["patient_id"] == pid] if not variants.empty else pd.DataFrame()
        )
        for _, vrow in v_sub.iterrows():
            event_type = _variant_event_type(
                vrow.get("alteration_type"), vrow.get("protein_change_hgvs")
            )
            value = float(vrow["vaf"]) if vrow["vaf"] >= 0 else 0.0
            rows.append((event_type, value, vrow["sample_date"], _gene_id(vrow.get("gene_symbol"))))

        t_sub = (
            treatments.loc[treatments["patient_id"] == pid]
            if not treatments.empty
            else pd.DataFrame()
        )
        for _, trow in t_sub.iterrows():
            if not bool(trow.get("is_osimertinib", False)):
                continue
            rows.append((EVENT_TYPES["osi_start"], 0.0, trow["start_date"], 0))
            if pd.notna(trow["end_date"]):
                rows.append((EVENT_TYPES["osi_stop"], 0.0, trow["end_date"], 0))

        o_sub = (
            outcomes.loc[outcomes["patient_id"] == pid] if not outcomes.empty else pd.DataFrame()
        )
        for _, orow in o_sub.iterrows():
            et = orow.get("event_type")
            if et == "progression_recist":
                rows.append((EVENT_TYPES["progression"], 0.0, orow["event_date"], 0))
            elif et == "death":
                rows.append((EVENT_TYPES["death"], 0.0, orow["event_date"], 0))
            elif et == "last_followup":
                rows.append((EVENT_TYPES["last_followup"], 0.0, orow["event_date"], 0))

        if not rows:
            # A patient with no events still needs a CLS token.
            sequences[pid] = EventSequence(
                event_types=np.array([CLS_EVENT_TYPE], dtype=np.int64),
                values=np.zeros(1, dtype=np.float32),
                timestamps=np.zeros(1, dtype=np.float32),
                gene_ids=np.zeros(1, dtype=np.int64),
            )
            continue

        rows.sort(key=lambda r: (r[2] is pd.NaT, r[2]))
        first_ts = next((r[2] for r in rows if pd.notna(r[2])), None)

        event_types_l = [CLS_EVENT_TYPE]
        values_l = [0.0]
        timestamps_l = [0.0]
        gene_ids_l = [0]
        for et_id, val, ts, gene in rows:
            event_types_l.append(et_id)
            values_l.append(val)
            if pd.notna(ts) and first_ts is not None:
                timestamps_l.append(float((ts - first_ts).days))
            else:
                timestamps_l.append(0.0)
            gene_ids_l.append(gene)

        sequences[pid] = EventSequence(
            event_types=np.asarray(event_types_l, dtype=np.int64),
            values=np.asarray(values_l, dtype=np.float32),
            timestamps=np.asarray(timestamps_l, dtype=np.float32),
            gene_ids=np.asarray(gene_ids_l, dtype=np.int64),
        )

    return sequences


def pad_event_batch(
    seqs: list[EventSequence], max_len: int | None = None
) -> dict[str, torch.Tensor]:
    """Right-pad a batch of EventSequences to a common length."""
    if not seqs:
        return {}
    length = max(s.length for s in seqs)
    if max_len is not None:
        length = min(length, max_len)

    B = len(seqs)
    event_types = torch.zeros((B, length), dtype=torch.long)
    values = torch.zeros((B, length), dtype=torch.float32)
    timestamps = torch.zeros((B, length), dtype=torch.float32)
    gene_ids = torch.zeros((B, length), dtype=torch.long)
    attention_mask = torch.zeros((B, length), dtype=torch.bool)

    for i, s in enumerate(seqs):
        t = min(s.length, length)
        event_types[i, :t] = torch.from_numpy(s.event_types[:t])
        values[i, :t] = torch.from_numpy(s.values[:t])
        timestamps[i, :t] = torch.from_numpy(s.timestamps[:t])
        gene_ids[i, :t] = torch.from_numpy(s.gene_ids[:t])
        attention_mask[i, :t] = True

    return {
        "event_types": event_types,
        "values": values,
        "timestamps": timestamps,
        "gene_ids": gene_ids,
        "attention_mask": attention_mask,
    }


# ---------------------------------------------------------------------------
# RoPE on continuous timestamps
# ---------------------------------------------------------------------------


def build_rope_cos_sin(
    timestamps: torch.Tensor, head_dim: int, base: float = 10_000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute (cos, sin) tables for RoPE driven by continuous timestamps.

    timestamps: (B, T) float
    Returns (cos, sin) each of shape (B, T, head_dim).
    """
    assert head_dim % 2 == 0, "RoPE requires an even head_dim."
    device = timestamps.device
    half = head_dim // 2
    freq_idx = torch.arange(half, dtype=timestamps.dtype, device=device)
    inv_freq = base ** (-2.0 * freq_idx / head_dim)  # (half,)
    angles = timestamps.unsqueeze(-1) * inv_freq  # (B, T, half)
    cos = angles.cos().repeat_interleave(2, dim=-1)
    sin = angles.sin().repeat_interleave(2, dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = -x2
    out[..., 1::2] = x1
    return out


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, D) ; cos/sin: (..., T, D)"""
    return x * cos + _rotate_half(x) * sin


class RoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.k = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cos_sin(timestamps, self.head_dim)
        cos = cos.unsqueeze(1)  # (B, 1, T, D_h)
        sin = sin.unsqueeze(1)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Key-padding mask (B, T) -> (B, 1, 1, T)
        key_pad = ~attention_mask
        attn_mask = key_pad.unsqueeze(1).unsqueeze(1)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, T, D_h)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out(out)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attn = RoPEMultiheadAttention(d_model, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        timestamps: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), timestamps, attention_mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    num_classes: int = 2  # filled in at fit time from the label encoder
    max_position_days: float = 3650.0  # 10 years; used only for clipping


class OncoTrajTransformer(nn.Module):
    """Small transformer with two task-specific heads.

    Returns a dict {"reg_pred": (B,), "cls_logits": (B, C)}.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.event_emb = nn.Embedding(NUM_EVENT_TYPES, config.d_model, padding_idx=PAD_EVENT_TYPE)
        self.gene_emb = nn.Embedding(NUM_GENES, config.d_model, padding_idx=GENE_VOCAB["PAD"])
        self.value_proj = nn.Linear(1, config.d_model)
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.reg_head = nn.Linear(config.d_model, 1)
        self.cls_head = nn.Linear(config.d_model, config.num_classes)

    def forward(
        self,
        event_types: torch.Tensor,
        values: torch.Tensor,
        timestamps: torch.Tensor,
        gene_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        x = (
            self.event_emb(event_types)
            + self.gene_emb(gene_ids)
            + self.value_proj(values.unsqueeze(-1))
        )
        ts = timestamps.clamp(min=0.0, max=self.config.max_position_days)
        for layer in self.layers:
            x = layer(x, ts, attention_mask)
        x = self.norm(x)
        cls_repr = x[:, 0, :]  # CLS token is always position 0 by construction.
        return {
            "reg_pred": self.reg_head(cls_repr).squeeze(-1),
            "cls_logits": self.cls_head(cls_repr),
        }


def best_device() -> torch.device:
    """Prefer CUDA, then MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# OncoTrajModel-conformant wrapper
# ---------------------------------------------------------------------------


@register_model
class TransformerModel(OncoTrajModel):
    name = "transformer"

    def __init__(
        self,
        is_classifier: bool = True,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        epochs: int = 40,
        batch_size: int = 16,
        weight_decay: float = 1e-2,
        warmup_ratio: float = 0.1,
        patience: int = 10,
        random_state: int = 42,
        device: torch.device | None = None,
        variants_df: pd.DataFrame | None = None,
        treatments_df: pd.DataFrame | None = None,
        outcomes_df: pd.DataFrame | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        self.is_classifier = is_classifier
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.patience = patience
        self.random_state = random_state
        self.device = device or best_device()

        self._variants = variants_df
        self._treatments = treatments_df
        self._outcomes = outcomes_df
        self._label_encoder: LabelEncoder | None = None
        self._net: OncoTrajTransformer | None = None

    def set_tables(
        self,
        variants: pd.DataFrame,
        treatments: pd.DataFrame,
        outcomes: pd.DataFrame,
    ) -> TransformerModel:
        self._variants = variants
        self._treatments = treatments
        self._outcomes = outcomes
        return self

    def fit_multitask(
        self,
        patient_ids: list[str],
        variants: pd.DataFrame,
        treatments: pd.DataFrame,
        outcomes: pd.DataFrame,
        y_reg: pd.Series | None,
        y_cls: pd.Series | None,
        val_patient_ids: list[str] | None = None,
    ) -> TransformerModel:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._variants = variants
        self._treatments = treatments
        self._outcomes = outcomes

        cls_classes: np.ndarray | None = None
        if y_cls is not None and not y_cls.empty:
            self._label_encoder = LabelEncoder().fit(y_cls)
            cls_classes = self._label_encoder.classes_
            self.classes_ = cls_classes
        num_classes = max(len(cls_classes) if cls_classes is not None else 0, 1)

        config = TransformerConfig(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_classes=num_classes,
        )
        self._net = OncoTrajTransformer(config).to(self.device)

        train_seqs = build_event_sequences(patient_ids, variants, treatments, outcomes)
        val_seqs = (
            build_event_sequences(val_patient_ids, variants, treatments, outcomes)
            if val_patient_ids
            else {}
        )

        optimizer = torch.optim.AdamW(
            self._net.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        steps_per_epoch = max(1, math.ceil(len(patient_ids) / self.batch_size))
        total_steps = steps_per_epoch * self.epochs
        warmup_steps = max(1, int(self.warmup_ratio * total_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: (
                min(step / warmup_steps, 1.0)
                * max(0.0, 1.0 - max(0, step - warmup_steps) / max(1, total_steps - warmup_steps))
            ),
        )

        mse = nn.MSELoss(reduction="none")
        ce = nn.CrossEntropyLoss(reduction="none")

        rng = np.random.default_rng(self.random_state)
        best_val = float("inf")
        best_state = {k: v.detach().clone() for k, v in self._net.state_dict().items()}
        patience_counter = 0
        self.training_history_: list[dict[str, float]] = []

        for epoch in range(self.epochs):
            train_order = list(patient_ids)
            rng.shuffle(train_order)
            self._net.train()
            train_loss = 0.0
            n_seen = 0
            for i in range(0, len(train_order), self.batch_size):
                batch_ids = train_order[i : i + self.batch_size]
                batch_seqs = [train_seqs[pid] for pid in batch_ids]
                batch = pad_event_batch(batch_seqs)
                batch = {k: v.to(self.device) for k, v in batch.items()}

                reg_y, reg_mask, cls_y, cls_mask = self._batch_targets(batch_ids, y_reg, y_cls)
                reg_y, reg_mask = reg_y.to(self.device), reg_mask.to(self.device)
                cls_y, cls_mask = cls_y.to(self.device), cls_mask.to(self.device)

                out = self._net(**batch)
                reg_loss = (
                    mse(out["reg_pred"], reg_y) * reg_mask
                ).sum() / reg_mask.sum().clamp_min(1.0)
                cls_loss = (
                    ce(out["cls_logits"], cls_y) * cls_mask
                ).sum() / cls_mask.sum().clamp_min(1.0)
                loss = reg_loss + cls_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                train_loss += float(loss.detach().cpu()) * len(batch_ids)
                n_seen += len(batch_ids)

            train_loss /= max(n_seen, 1)
            val_loss = (
                self._eval_loss(val_seqs, val_patient_ids or [], y_reg, y_cls, mse, ce)
                if val_seqs
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

    def _batch_targets(
        self, batch_ids: list[str], y_reg: pd.Series | None, y_cls: pd.Series | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reg_y = torch.zeros(len(batch_ids), dtype=torch.float32)
        reg_mask = torch.zeros(len(batch_ids), dtype=torch.float32)
        cls_y = torch.zeros(len(batch_ids), dtype=torch.long)
        cls_mask = torch.zeros(len(batch_ids), dtype=torch.float32)
        for i, pid in enumerate(batch_ids):
            if y_reg is not None and pid in y_reg.index:
                reg_y[i] = float(y_reg.loc[pid])
                reg_mask[i] = 1.0
            if y_cls is not None and self._label_encoder is not None and pid in y_cls.index:
                cls_y[i] = int(self._label_encoder.transform([y_cls.loc[pid]])[0])
                cls_mask[i] = 1.0
        return reg_y, reg_mask, cls_y, cls_mask

    def _eval_loss(
        self,
        seqs: dict[str, EventSequence],
        ids: list[str],
        y_reg: pd.Series | None,
        y_cls: pd.Series | None,
        mse: nn.Module,
        ce: nn.Module,
    ) -> float:
        if not ids:
            return float("nan")
        self._net.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(ids), self.batch_size):
                batch_ids = ids[i : i + self.batch_size]
                batch = pad_event_batch([seqs[pid] for pid in batch_ids])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                reg_y, reg_mask, cls_y, cls_mask = self._batch_targets(batch_ids, y_reg, y_cls)
                reg_y, reg_mask = reg_y.to(self.device), reg_mask.to(self.device)
                cls_y, cls_mask = cls_y.to(self.device), cls_mask.to(self.device)
                out = self._net(**batch)
                reg_loss = (
                    mse(out["reg_pred"], reg_y) * reg_mask
                ).sum() / reg_mask.sum().clamp_min(1.0)
                cls_loss = (
                    ce(out["cls_logits"], cls_y) * cls_mask
                ).sum() / cls_mask.sum().clamp_min(1.0)
                total += float((reg_loss + cls_loss).cpu()) * len(batch_ids)
                n += len(batch_ids)
        return total / max(n, 1)

    # --- single-task OncoTrajModel adapter ---------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series) -> TransformerModel:
        if self._variants is None or self._treatments is None or self._outcomes is None:
            raise RuntimeError(
                "TransformerModel.fit requires the variants/treatments/outcomes tables. "
                "Use set_tables(variants, treatments, outcomes) before .fit()."
            )
        self.feature_names_ = list(X.columns)
        patient_ids = list(X.index)
        if self.is_classifier:
            self.fit_multitask(
                patient_ids, self._variants, self._treatments, self._outcomes, None, y
            )
        else:
            self.fit_multitask(
                patient_ids, self._variants, self._treatments, self._outcomes, y, None
            )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        reg, logits = self._infer(X)
        if self.is_classifier:
            idx = logits.argmax(axis=1)
            return (
                self._label_encoder.inverse_transform(idx)
                if self._label_encoder is not None
                else idx
            )
        return reg

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_classifier:
            raise NotImplementedError("TransformerModel is in regression mode.")
        self._check_fitted()
        _, logits = self._infer(X)
        return torch.softmax(torch.from_numpy(logits), dim=1).numpy()

    def _infer(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        assert self._net is not None
        patient_ids = list(X.index)
        seqs = build_event_sequences(patient_ids, self._variants, self._treatments, self._outcomes)
        regs: list[np.ndarray] = []
        all_logits: list[np.ndarray] = []
        self._net.eval()
        with torch.no_grad():
            for i in range(0, len(patient_ids), self.batch_size):
                batch_ids = patient_ids[i : i + self.batch_size]
                batch = pad_event_batch([seqs[pid] for pid in batch_ids])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                out = self._net(**batch)
                regs.append(out["reg_pred"].cpu().numpy())
                all_logits.append(out["cls_logits"].cpu().numpy())
        return (
            np.concatenate(regs) if regs else np.array([]),
            np.concatenate(all_logits) if all_logits else np.array([]),
        )


warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")
