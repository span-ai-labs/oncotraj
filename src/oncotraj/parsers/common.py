"""Shared utilities for source-specific parsers.

The four-table contract (patients/variants/treatments/outcomes) is documented
in DATASET_SPEC.md sections 0 through 5. This module owns:
- `ParsedTables`: the four-DataFrame container every parser returns.
- Cohort filtering (EGFR sensitizing + first-line osimertinib).
- EGFR-variant-class derivation from raw HGVSp strings.
- Pydantic-row validation of an emitted DataFrame against the schemas module.

Parsers MUST NOT silently coerce. If a row cannot meet the schema, the parser
either excludes the patient (with `exclusion_reason`) or raises.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import pandas as pd
from pydantic import ValidationError

from oncotraj.schemas import (
    EgfrVariantClass,
    OutcomeRecord,
    PatientRecord,
    TreatmentRecord,
    VariantRecord,
)

EGFR_SENSITIZING_HGVSP = {
    EgfrVariantClass.L858R: re.compile(r"p\.(?:L|Leu)858(?:R|Arg)"),
    EgfrVariantClass.G719X: re.compile(r"p\.(?:G|Gly)719[A-Za-z]+"),
    EgfrVariantClass.L861Q: re.compile(r"p\.(?:L|Leu)861(?:Q|Gln)"),
    EgfrVariantClass.S768I: re.compile(r"p\.(?:S|Ser)768(?:I|Ile)"),
}
# Exon 19 deletions: residue range fully within 746-759 followed by `del` or
# `delins<aa>`. Covers the canonical `E746_A750del`, the long tail of `delins`
# variants (`E746_T751delinsVP`, `L747_P753delinsS`), and the rarer 750s starts.
EXON19DEL_RE = re.compile(
    r"p\.[A-Za-z]+74[6-9]_[A-Za-z]+(?:74[6-9]|75\d)(?:del|delins)|exon\s*19.*del", re.I
)
# Exon 20 insertions: residue range in 760-779 ending with `ins` (not `delins`).
# Negative lookbehind on `del` excludes exon-19 deletion-insertions.
EXON20INS_RE = re.compile(
    r"p\.[A-Za-z]+(?:76\d|77\d)_[A-Za-z]+(?:76\d|77\d)(?<!del)ins|exon\s*20.*ins", re.I
)
C797S_RE = re.compile(r"p\.(?:C|Cys)797(?:S|Ser)")
T790M_RE = re.compile(r"p\.(?:T|Thr)790(?:M|Met)")


@dataclass
class ParsedTables:
    """Four-table output of any source-specific parser."""

    patients: pd.DataFrame
    variants: pd.DataFrame
    treatments: pd.DataFrame
    outcomes: pd.DataFrame

    def __len__(self) -> int:
        return len(self.patients)

    def concat(self, other: ParsedTables) -> ParsedTables:
        return ParsedTables(
            patients=pd.concat([self.patients, other.patients], ignore_index=True),
            variants=pd.concat([self.variants, other.variants], ignore_index=True),
            treatments=pd.concat([self.treatments, other.treatments], ignore_index=True),
            outcomes=pd.concat([self.outcomes, other.outcomes], ignore_index=True),
        )


def make_patient_id(source_prefix: str, stable_id: str) -> str:
    """Build the colon-prefixed patient_id per DATASET_SPEC.md §0."""
    return f"{source_prefix}:{stable_id}"


def new_uuid() -> str:
    return str(uuid.uuid4())


def classify_egfr_variant(hgvsp_strings: list[str]) -> EgfrVariantClass:
    """Pick the EGFR sensitizing class for a patient from their EGFR variant calls.

    Implements DATASET_SPEC.md §3 derivation rule R1 — prefer the strongest
    sensitizing match. Compound calls (sensitizing + resistance) collapse to
    `compound_with_resistance`.
    """
    sensitizing_hits: set[EgfrVariantClass] = set()
    has_resistance = False
    for s in hgvsp_strings:
        if not isinstance(s, str):
            continue
        if EXON19DEL_RE.search(s):
            sensitizing_hits.add(EgfrVariantClass.exon19del)
        for cls, pattern in EGFR_SENSITIZING_HGVSP.items():
            if pattern.search(s):
                sensitizing_hits.add(cls)
        if EXON20INS_RE.search(s):
            sensitizing_hits.add(EgfrVariantClass.exon20ins)
        if C797S_RE.search(s) or T790M_RE.search(s):
            has_resistance = True

    if not sensitizing_hits:
        return EgfrVariantClass.unknown
    if has_resistance:
        return EgfrVariantClass.compound_with_resistance
    if len(sensitizing_hits) > 1:
        return EgfrVariantClass.compound_sensitizing
    return next(iter(sensitizing_hits))


def is_osimertinib(drug_name: str) -> bool:
    if not isinstance(drug_name, str):
        return False
    s = drug_name.strip().lower()
    return s in {"osimertinib", "tagrisso", "azd9291"} or "osimertinib" in s


def validate_tables(tables: ParsedTables, sample_size: int = 50) -> None:
    """Spot-validate emitted rows against Pydantic models.

    Full per-row validation on millions of rows is wasteful for v0; we sample
    `sample_size` rows from each table and bubble up the first ValidationError.
    The schemas themselves remain the canonical contract.
    """
    pairs = [
        (PatientRecord, tables.patients),
        (VariantRecord, tables.variants),
        (TreatmentRecord, tables.treatments),
        (OutcomeRecord, tables.outcomes),
    ]
    for model, df in pairs:
        if df.empty:
            continue
        sample = df.sample(n=min(sample_size, len(df)), random_state=0)
        int_fields = {name for name, info in model.model_fields.items() if info.annotation is int}
        for row in sample.to_dict(orient="records"):
            cleaned: dict = {}
            for k, v in row.items():
                if v is None or (isinstance(v, float | int | type(pd.NaT)) and pd.isna(v)):
                    cleaned[k] = None
                elif k in int_fields and isinstance(v, float):
                    cleaned[k] = int(v)
                else:
                    cleaned[k] = v
            try:
                model.model_validate(cleaned)
            except ValidationError as e:
                raise ValueError(
                    f"Schema validation failed for {model.__name__} row {cleaned}: {e}"
                ) from e
            except TypeError as e:
                raise ValueError(f"Schema TypeError for {model.__name__} row {cleaned}: {e}") from e
