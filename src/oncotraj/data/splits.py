"""Patient-level stratified train/val/test splits for the v1 OncoTraj cohort.

Strata: (`source_dataset`, `resistance_mechanism`). The mechanism is derived
per-patient from their `progression_recist` outcome's `resistance_mechanism_class`;
patients without a PD event fall into a `no_progression` stratum.

Design notes:
- Patient-level. A patient's variants/outcomes/treatments all share the
  patient's split assignment. Trajectories are never split across folds —
  enforced by `_assert_no_leakage` in the integration tests.
- Deterministic. Given the same input parquet and seed, the output manifest
  is byte-identical.
- Excluded patients (`included_in_v1_cohort == False`) are never assigned
  to any split. They are the held-out sensitivity slice (AURA3, OSCILLATE,
  atypical-EGFR rows) and should not contaminate train/val/test.
- Site-out replication. Optional `site_out` argument removes a list of
  `source_dataset` values from the splittable pool before stratifying.
  PAPER_OUTLINE.md §4.4 specifies TRACERx as the v1 site-out cohort.
- Small-stratum policy. A stratum with fewer than `min_stratum_size`
  patients (default 7) is assigned entirely to train. Splitting a
  3-patient stratum 70/15/15 yields 2/0/1 or 2/1/0 — useless for
  evaluation and bad for downstream calibration estimates.

Strata divergence from PAPER_OUTLINE.md §4.4:
PAPER_OUTLINE proposes (`variant_class`, `stage`, `site`) but the locked
metrics (Task C per-mechanism F1) and v1 modeling target make mechanism a
more relevant stratum. The `strata_fields` argument is exposed so the
outline's original strata can be substituted without code changes.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SPLIT_SCHEMA_VERSION = "oncotraj-splits/1.0.0"
NO_PROGRESSION_STRATUM = "no_progression"
DEFAULT_PROPORTIONS = (0.70, 0.15, 0.15)
DEFAULT_MIN_STRATUM_SIZE = 7
DEFAULT_SEED = 42


@dataclass(frozen=True)
class SplitProportions:
    train: float
    val: float
    test: float

    def __post_init__(self):
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")
        for name, p in (("train", self.train), ("val", self.val), ("test", self.test)):
            if not 0.0 < p < 1.0:
                raise ValueError(f"Proportion {name}={p} must be in (0, 1)")


@dataclass
class SplitManifest:
    """JSON-serialisable record of one split run."""

    schema_version: str
    generated_at_utc: str
    seed: int
    strata_fields: list[str]
    proportions: SplitProportions
    counts: dict[str, int]
    excluded_count: int
    site_out: list[str]
    min_stratum_size: int
    input_parquet_sha256: str | None
    stratum_breakdown: dict[str, dict[str, int]]
    assignments: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        d = asdict(self)
        # `SplitProportions` is a frozen dataclass; asdict expands it.
        return json.dumps(d, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, raw: str) -> SplitManifest:
        d = json.loads(raw)
        d["proportions"] = SplitProportions(**d["proportions"])
        return cls(**d)


def _derive_mechanism(patient_id: str, outcomes: pd.DataFrame) -> str:
    """The mechanism stratum for a patient is the class at their PD event.

    A patient with no progression row gets `no_progression`. A patient with
    a progression row but `resistance_mechanism_class=None` gets the same.
    Multiple progression rows (rare) → take the earliest one's mechanism.
    """
    rows = outcomes[
        (outcomes["patient_id"] == patient_id) & (outcomes["event_type"] == "progression_recist")
    ]
    if rows.empty:
        return NO_PROGRESSION_STRATUM
    rows = rows.sort_values("event_date")
    first_mech = rows["resistance_mechanism_class"].iloc[0]
    if first_mech is None or (isinstance(first_mech, float) and pd.isna(first_mech)):
        return NO_PROGRESSION_STRATUM
    return str(first_mech)


def _stratum_key(values: list[str]) -> str:
    """Concatenate strata values into a single bucket key."""
    return "::".join(values)


def _proportional_counts(n: int, props: SplitProportions) -> tuple[int, int, int]:
    """Greedy proportional allocation that guarantees sum == n and every
    bucket gets at least one patient if it's allowed to be non-empty.

    For n < 3, the policy is upstream: small buckets are forced to train.
    """
    train = round(n * props.train)
    val = round(n * props.val)
    test = n - train - val
    # Floating-point rounding can produce negative test; correct by stealing from train.
    if test < 0:
        train += test
        test = 0
    return train, val, test


def _split_one_stratum(
    patient_ids: list[str], props: SplitProportions, rng: random.Random
) -> dict[str, str]:
    """Deterministic shuffle + greedy proportional allocation."""
    sorted_ids = sorted(patient_ids)  # deterministic input order
    rng.shuffle(sorted_ids)
    n = len(sorted_ids)
    n_train, n_val, _ = _proportional_counts(n, props)
    out: dict[str, str] = {}
    for i, pid in enumerate(sorted_ids):
        if i < n_train:
            out[pid] = "train"
        elif i < n_train + n_val:
            out[pid] = "val"
        else:
            out[pid] = "test"
    return out


def make_splits(
    patients: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    proportions: SplitProportions | tuple[float, float, float] = DEFAULT_PROPORTIONS,
    site_out: list[str] | None = None,
    min_stratum_size: int = DEFAULT_MIN_STRATUM_SIZE,
    strata_fields: list[str] | None = None,
    input_parquet_sha256: str | None = None,
) -> SplitManifest:
    """Compute a stratified patient-level split and return its manifest.

    Args:
        patients: `patients.parquet` as a DataFrame. Must have `patient_id`,
            `source_dataset`, `included_in_v1_cohort`.
        outcomes: `outcomes.parquet`. Used to derive per-patient mechanism.
        seed: deterministic seed.
        proportions: train/val/test fractions. Must sum to 1.0.
        site_out: source_dataset values to hold out entirely (no split rows
            emitted for them, treated like excluded patients).
        min_stratum_size: strata smaller than this get assigned all-to-train.
        strata_fields: column names from `patients` plus the literal token
            `"resistance_mechanism"` to use as stratification. Default is
            `["source_dataset", "resistance_mechanism"]`.
        input_parquet_sha256: optional content hash of the source parquet
            for the manifest header. Caller computes this when invoking
            from the CLI.

    Returns:
        SplitManifest with a complete `assignments` dict.
    """
    if isinstance(proportions, tuple):
        proportions = SplitProportions(*proportions)
    if site_out is None:
        site_out = []
    if strata_fields is None:
        strata_fields = ["source_dataset", "resistance_mechanism"]

    # Filter to splittable patients: included AND not in a site-out source.
    eligible = patients.loc[
        patients["included_in_v1_cohort"] & ~patients["source_dataset"].isin(site_out)
    ].copy()
    excluded_count = len(patients) - len(eligible)

    # Derive mechanism per patient if any stratum needs it.
    if "resistance_mechanism" in strata_fields:
        eligible["resistance_mechanism"] = eligible["patient_id"].map(
            lambda pid: _derive_mechanism(pid, outcomes)
        )

    # Build strata buckets.
    buckets: dict[str, list[str]] = {}
    for _, row in eligible.iterrows():
        key = _stratum_key([str(row[f]) for f in strata_fields])
        buckets.setdefault(key, []).append(row["patient_id"])

    assignments: dict[str, str] = {}
    stratum_breakdown: dict[str, dict[str, int]] = {}

    # Iterate strata in deterministic order; each gets its own RNG derived
    # from the global seed and the stratum name, so adding a new stratum
    # doesn't reshuffle the others.
    for key in sorted(buckets):
        pids = buckets[key]
        derived_seed = int(hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()[:16], 16)
        rng = random.Random(derived_seed)

        if len(pids) < min_stratum_size:
            stratum_assignments = {pid: "train" for pid in sorted(pids)}
        else:
            stratum_assignments = _split_one_stratum(pids, proportions, rng)

        assignments.update(stratum_assignments)
        stratum_breakdown[key] = {
            "train": sum(1 for v in stratum_assignments.values() if v == "train"),
            "val": sum(1 for v in stratum_assignments.values() if v == "val"),
            "test": sum(1 for v in stratum_assignments.values() if v == "test"),
            "total": len(pids),
        }

    counts = {
        "train": sum(1 for v in assignments.values() if v == "train"),
        "val": sum(1 for v in assignments.values() if v == "val"),
        "test": sum(1 for v in assignments.values() if v == "test"),
    }

    return SplitManifest(
        schema_version=SPLIT_SCHEMA_VERSION,
        generated_at_utc=datetime.now(UTC).isoformat(),
        seed=seed,
        strata_fields=strata_fields,
        proportions=proportions,
        counts=counts,
        excluded_count=excluded_count,
        site_out=sorted(site_out),
        min_stratum_size=min_stratum_size,
        input_parquet_sha256=input_parquet_sha256,
        stratum_breakdown=stratum_breakdown,
        assignments=dict(sorted(assignments.items())),
    )


def make_splits_from_parquet(
    parquet_dir: Path,
    output_path: Path,
    **kwargs,
) -> SplitManifest:
    """Read the four-table parquet dir and write the split manifest to disk."""
    parquet_dir = Path(parquet_dir)
    patients = pd.read_parquet(parquet_dir / "patients.parquet")
    outcomes = pd.read_parquet(parquet_dir / "outcomes.parquet")

    # Hash the patients file for the manifest header.
    sha = hashlib.sha256((parquet_dir / "patients.parquet").read_bytes()).hexdigest()

    manifest = make_splits(patients, outcomes, input_parquet_sha256=sha, **kwargs)
    Path(output_path).write_text(manifest.to_json())
    return manifest
