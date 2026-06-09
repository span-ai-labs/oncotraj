"""Tan et al. 2024 — OSCILLATE alternating osimertinib/gefitinib supplement.

Citation: Tan L, Brown C, Mersiades A, et al. "A Phase II trial of alternating
osimertinib and gefitinib therapy in advanced EGFR-T790M positive non-small
cell lung cancer: OSCILLATE." Nat Commun 15 (2024).
DOI: 10.1038/s41467-024-46008-1.  PMID: 38418463.  PMC: PMC10902357.

Cohort treatment: OSCILLATE enrolled T790M+ patients with prior 1L EGFR TKI
exposure who then received alternating osimertinib + gefitinib (not osi
monotherapy). Both conditions force exclusion from the v1 training cohort:
- `prior_EGFR_TKI`: every patient had a prior 1L TKI by inclusion criterion.
- `not_first_line_osimertinib`: alternating regimen, not 1L osi monotherapy.
We emit `prior_EGFR_TKI` as the dominant reason; downstream filters can layer
in the regimen exclusion via `treatments.drug_name` if needed.

Supplement consumed: Supplementary Data 1 (`41467_2024_46008_MOESM5_ESM.xlsx`),
single sheet `Sheet1`. The dataset has REAL calendar dates per visit —
unlike Chmielecki FLAURA/AURA3 — so the time-to-resistance values are usable
for sensitivity analysis once the cohort filter is reversed.

Yield: 47 patients (matches published n).
Quirks handled:
- `Gene` strings: trailing spaces (`"EGFR "`, `"MET amplification "`), the
  `"ERRB2"` typo for ERBB2, and the `"<gene> amplification"` pattern which
  encodes a CNV in the gene column instead of via a separate type field.
- `Variant` strings sometimes compound: `"p.Arg988Cys, p.Arg970Cys"` is
  split into two rows.
- `"no detectable mutations"` is a per-timepoint sentinel — emit the visit
  as a sample but no variant rows.
- `Date` is set on the first row of a (Patient, Timepoint) group; subsequent
  rows in the group are NaN and inherit the group's date by forward-fill.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from oncotraj.schemas import (
    AlterationType,
    Assay,
    EgfrVariantClass,
    Histology,
    LineOfTherapy,
    OncoKBOncogenic,
    OutcomeEventType,
    RecistResponse,
    ResistanceMechanism,
    SampleType,
    Sex,
    SmokingStatus,
    SourceDataset,
    StageAtDiagnosis,
    VitalStatus,
)

from ..common import C797S_RE, ParsedTables, classify_egfr_variant, new_uuid
from . import register
from ._base import PaperAdapter

SHEET_NAME = "Sheet1"
MIN_AMP_PROXY_VAF = 0.0  # OSCILLATE doesn't report CN; amp calls trusted as-is.

AMP_GENE_RE = re.compile(r"^\s*(ERRB2|ERBB2|MET|EGFR|HER2)\s*amplification\s*$", re.IGNORECASE)
GENE_NORMALIZE = {"ERRB2": "ERBB2", "HER2": "ERBB2"}
SENTINEL_NO_MUTATIONS = "no detectable mutations"


def _find_supplement(study_dir: Path) -> Path | None:
    candidates = [
        study_dir / "MOESM5.xlsx",
        study_dir / "41467_2024_46008_MOESM5_ESM.xlsx",
        study_dir / "pdf" / "MOESM5.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(study_dir.glob("**/*MOESM5*.xlsx"))
    return matches[0] if matches else None


def _parse_gene_and_alteration(raw_gene: object) -> tuple[str, AlterationType]:
    """Normalise the supplement's gene-column quirks.

    Returns `(gene_symbol, alteration_type)`. For amp-in-gene encodings
    (e.g. `"MET amplification "`), alteration_type is `CNV_amplification`.
    """
    if not isinstance(raw_gene, str):
        return "UNKNOWN", AlterationType.SNV
    gene = raw_gene.strip()
    amp_match = AMP_GENE_RE.match(gene)
    if amp_match:
        normalised = GENE_NORMALIZE.get(amp_match.group(1).upper(), amp_match.group(1).upper())
        return normalised, AlterationType.CNV_amplification
    gene = GENE_NORMALIZE.get(gene.upper(), gene.upper())
    return gene, AlterationType.SNV


def _split_compound_variants(raw_variant: object) -> list[str]:
    """`"p.Arg988Cys, p.Arg970Cys"` → `["p.Arg988Cys", "p.Arg970Cys"]`."""
    if not isinstance(raw_variant, str):
        return []
    return [v.strip() for v in raw_variant.split(",") if v.strip()]


def _vaf_from_percent(raw: object) -> float:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return -1.0
    if isinstance(raw, str) and raw.strip() == "":
        return -1.0
    try:
        value = float(raw) / 100.0
    except (TypeError, ValueError):
        return -1.0
    if value != value:  # NaN check
        return -1.0
    return max(min(value, 1.0), 0.0)


def _coerce_date(raw: object) -> date | None:
    if isinstance(raw, date):
        return raw
    if hasattr(raw, "date"):
        try:
            return raw.date()
        except Exception:
            return None
    if isinstance(raw, str) and raw.strip():
        try:
            return date.fromisoformat(raw.strip())
        except ValueError:
            return None
    return None


def _is_pd(timepoint: object) -> bool:
    return isinstance(timepoint, str) and timepoint.strip().startswith("Progression")


def _assign_dominant_mechanism(pd_rows: pd.DataFrame) -> ResistanceMechanism | None:
    """Apply H2 → H3: C797S > MET amp > HER2 amp > EGFR amp > other."""
    if pd_rows.empty:
        return None

    for variant in pd_rows.loc[pd_rows["_gene"] == "EGFR", "_variant"].dropna():
        if isinstance(variant, str) and C797S_RE.search(variant):
            return ResistanceMechanism.EGFR_C797S

    amps = pd_rows[pd_rows["_alteration"] == AlterationType.CNV_amplification.value]
    if amps.empty:
        return ResistanceMechanism.other_or_unknown
    mechanism_priority = {
        "MET": ResistanceMechanism.MET_amplification,
        "ERBB2": ResistanceMechanism.HER2_amplification,
        "EGFR": ResistanceMechanism.EGFR_amplification,
    }
    for gene, mech in mechanism_priority.items():
        if (amps["_gene"] == gene).any():
            return mech
    return ResistanceMechanism.other_or_unknown


def _build_variant_row(
    pid: str,
    sample_date: date,
    timepoint: str,
    gene: str,
    alteration: str,
    variant_text: str | None,
    vaf: float,
) -> dict:
    is_c797s = (
        gene == "EGFR"
        and alteration == AlterationType.SNV.value
        and isinstance(variant_text, str)
        and bool(C797S_RE.search(variant_text))
    )
    is_amp = alteration == AlterationType.CNV_amplification.value
    is_resistance = bool((is_c797s or is_amp) and _is_pd(timepoint))
    resistance_class: str | None = None
    if is_resistance:
        if is_c797s:
            resistance_class = ResistanceMechanism.EGFR_C797S.value
        elif gene == "MET":
            resistance_class = ResistanceMechanism.MET_amplification.value
        elif gene == "ERBB2":
            resistance_class = ResistanceMechanism.HER2_amplification.value
        elif gene == "EGFR":
            resistance_class = ResistanceMechanism.EGFR_amplification.value
    is_baseline_driver = (
        gene == "EGFR"
        and alteration == AlterationType.SNV.value
        and timepoint == "Baseline"
        and isinstance(variant_text, str)
        and "C797" not in variant_text
        and "T790" not in variant_text
    )
    return {
        "variant_id": new_uuid(),
        "patient_id": pid,
        "sample_id": f"{pid.split(':', 1)[1]}-{timepoint.replace(' ', '_')}",
        "sample_type": SampleType.ctDNA_plasma.value,
        "sample_date": sample_date,
        "assay": Assay.custom_panel.value,  # Roche AVENIO targeted panel (per MOESM6).
        "gene_symbol": gene,
        "alteration_type": alteration,
        "protein_change_hgvs": variant_text,
        "vaf": vaf,
        "read_depth": -1,
        "oncokb_oncogenic": OncoKBOncogenic.unknown.value,
        "is_germline": False,
        "is_baseline_driver": is_baseline_driver,
        "is_resistance_call": is_resistance,
        "resistance_mechanism_class": resistance_class,
    }


@register
class TanOscillateAdapter(PaperAdapter):
    study_id = "tan_2024_oscillate"
    source_dataset = SourceDataset.AURA3_SUPP  # No OSCILLATE-specific enum yet; AURA3-tier holdout.
    citation = "Tan L et al. Nat Commun 15 (2024). DOI:10.1038/s41467-024-46008-1."

    def load(self) -> ParsedTables:
        supplement = _find_supplement(self.inputs.study_dir)
        if supplement is None:
            return ParsedTables(
                patients=pd.DataFrame(),
                variants=pd.DataFrame(),
                treatments=pd.DataFrame(),
                outcomes=pd.DataFrame(),
            )

        df = pd.read_excel(
            supplement, sheet_name=SHEET_NAME, header=0, engine="openpyxl", dtype=object
        )
        df = df[df["Patient ID"].notna()].copy()
        # Date is set on the first row of each (Patient ID, Timepoint) group; forward-fill within group.
        df["Date"] = df.groupby(["Patient ID", "Timepoint"])["Date"].transform("ffill")

        # Pre-normalise gene/alteration onto helper columns so the mechanism rule can see them.
        norm = df["Gene"].map(_parse_gene_and_alteration)
        df["_gene"] = norm.map(lambda x: x[0])
        df["_alteration"] = norm.map(lambda x: x[1].value)
        df["_variant"] = df["Variant"]

        patients: list[dict] = []
        variants: list[dict] = []
        treatments: list[dict] = []
        outcomes: list[dict] = []

        for raw_pid, group in df.groupby("Patient ID"):
            pid = self.make_patient_id(str(raw_pid))

            baseline = group[group["Timepoint"] == "Baseline"]
            baseline_date = _coerce_date(baseline["Date"].iloc[0]) if not baseline.empty else None
            pd_rows = group[group["Timepoint"].isin({"Progression 1", "Progression 2"})]
            last_pd = pd_rows.sort_values("Timepoint").tail(1)
            last_pd_date = _coerce_date(last_pd["Date"].iloc[0]) if not last_pd.empty else None
            first_pd_date = (
                _coerce_date(pd_rows.sort_values("Timepoint")["Date"].iloc[0])
                if not pd_rows.empty
                else None
            )

            # EGFR class from baseline EGFR SNVs (excluding the T790M resistance allele itself).
            baseline_egfr = baseline[
                (baseline["_gene"] == "EGFR") & (baseline["_alteration"] == AlterationType.SNV)
            ]
            hgvsp_for_class = [
                v
                for v in baseline_egfr["_variant"]
                if isinstance(v, str) and v != SENTINEL_NO_MUTATIONS
            ]
            variant_class = classify_egfr_variant(hgvsp_for_class)

            dominant = _assign_dominant_mechanism(pd_rows)

            # Both inclusion conditions fail: prior TKI (T790M+ enrolment) and not 1L osi monotherapy.
            included = False
            exclusion = "prior_EGFR_TKI"
            if variant_class == EgfrVariantClass.unknown:
                exclusion = "not_EGFR_sensitizing"

            diagnosis_anchor = baseline_date or date(2017, 6, 1)
            last_followup = last_pd_date or baseline_date or diagnosis_anchor

            patients.append(
                {
                    "patient_id": pid,
                    "source_dataset": self.source_dataset.value,
                    "age_at_diagnosis_years": -1,
                    "sex": Sex.unknown.value,
                    "smoking_status": SmokingStatus.unknown.value,
                    "diagnosis_date": diagnosis_anchor,
                    "stage_at_diagnosis": StageAtDiagnosis.IV_NOS.value,
                    "histology": Histology.adenocarcinoma.value,
                    "egfr_variant_class": variant_class.value,
                    "site_id": "OSCILLATE_trial",
                    "vital_status_at_last_followup": VitalStatus.unknown.value,
                    "last_followup_date": last_followup,
                    "included_in_v1_cohort": included,
                    "exclusion_reason": exclusion,
                }
            )

            treatments.append(
                {
                    "treatment_id": new_uuid(),
                    "patient_id": pid,
                    "drug_name": "osimertinib_gefitinib_alternating",
                    "line_of_therapy": LineOfTherapy.second_line.value,
                    "start_date": diagnosis_anchor,
                    "end_date": first_pd_date,
                    "is_osimertinib": True,  # contains osimertinib, even if not monotherapy
                }
            )

            outcomes.append(
                {
                    "outcome_id": new_uuid(),
                    "patient_id": pid,
                    "event_type": OutcomeEventType.last_followup.value,
                    "event_date": last_followup,
                    "recist_response": None,
                    "resistance_mechanism_class": None,
                }
            )
            if first_pd_date is not None:
                outcomes.append(
                    {
                        "outcome_id": new_uuid(),
                        "patient_id": pid,
                        "event_type": OutcomeEventType.progression_recist.value,
                        "event_date": first_pd_date,
                        "recist_response": RecistResponse.progressive_disease.value,
                        "resistance_mechanism_class": dominant.value if dominant else None,
                    }
                )

            for _, raw_row in group.iterrows():
                if raw_row["_gene"] in {None, "NO_DETECTABLE_MUTATIONS"} or (
                    isinstance(raw_row["Gene"], str)
                    and raw_row["Gene"].strip().lower() == SENTINEL_NO_MUTATIONS
                ):
                    continue
                gene = raw_row["_gene"]
                alteration = raw_row["_alteration"]
                sample_date = _coerce_date(raw_row["Date"]) or diagnosis_anchor
                timepoint = str(raw_row["Timepoint"])
                vaf = _vaf_from_percent(raw_row["Allele Fraction (%)"])
                # Compound variant strings ("p.X, p.Y") get one variant row each.
                variant_pieces = _split_compound_variants(raw_row["Variant"]) or [None]
                for piece in variant_pieces:
                    variants.append(
                        _build_variant_row(
                            pid, sample_date, timepoint, gene, alteration, piece, vaf
                        )
                    )

        return ParsedTables(
            patients=pd.DataFrame(patients),
            variants=pd.DataFrame(variants),
            treatments=pd.DataFrame(treatments),
            outcomes=pd.DataFrame(outcomes),
        )
