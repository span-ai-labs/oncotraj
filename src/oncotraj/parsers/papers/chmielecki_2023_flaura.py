"""Chmielecki et al. 2023 — FLAURA molecular-resistance supplement adapter.

Citation: Chmielecki J, Gray JE, Cheng Y, et al. "Candidate mechanisms of
acquired resistance to first-line osimertinib in EGFR-mutated advanced
non-small cell lung cancer." Nat Commun 14, 1070 (2023).
DOI: 10.1038/s41467-023-35961-y.  PMID: 36849494.  PMC: PMC9971254.

Supplement consumed: Supplementary Data 1 (`41467_2023_35961_MOESM3_ESM.xlsx`),
sheet `FLAURA_osimertinib arm`. The `FLAURA_SoC arm` sheet covers comparator
chemo and is excluded (those patients never received osimertinib).

Cohort yield: 109 patients on first-line osimertinib (matches the published
molecular analysis subset). 69 with paired C1D1 + Discontinuation samples.

Hard limitation worth flagging downstream: the supplement does not publish
absolute calendar dates. We use a documented epoch anchor (FLAURA enrollment
start) so the schema's date columns can be populated, but the resulting
time-to-resistance values are nominal, not real. Downstream Task B
(time-to-resistance MAE/C-index) MUST exclude this study from its headline
unless paired with a separate dated source. Marked in `_meta` via the
`paper:chmielecki_2023_flaura` source tag.

Labeling: applies LABELING_GUIDELINES.md §3 hierarchy H1 (skipped — no
histology data) → H2 (EGFR_C797S supersedes off-target) → H3 (highest CN among
MET/ERBB2/EGFR amps). HER2_AMP maps from ERBB2 amplifications.
"""

from __future__ import annotations

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

from ..common import ParsedTables, classify_egfr_variant, new_uuid
from . import register
from ._base import PaperAdapter

OSIMERTINIB_ARM_SHEET = "FLAURA_osimertinib arm"
ENROLLMENT_EPOCH = date(2017, 1, 1)
DATA_CUTOFF = date(2019, 6, 30)
MIN_AMP_CN = 6.0
HER2_GENE = "ERBB2"


def _find_supplement(study_dir: Path) -> Path | None:
    """Resolve the MOESM3 supplement, whether stored under pdf/ or beside it."""
    candidates = [
        study_dir / "MOESM3.xlsx",
        study_dir / "41467_2023_35961_MOESM3_ESM.xlsx",
        study_dir / "pdf" / "MOESM3.xlsx",
        study_dir / "pdf" / "41467_2023_35961_MOESM3_ESM.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(study_dir.glob("**/*MOESM3*.xlsx"))
    return matches[0] if matches else None


def _normalize_hgvsp(raw: object) -> str | None:
    """Supplement strings are bare (`E746_A750del`, `L858R`); spec wants `p.` prefix."""
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s or s == "-":
        return None
    return s if s.startswith("p.") else f"p.{s}"


def _vaf_from_percent(raw: object) -> float:
    if not isinstance(raw, str | float | int):
        return -1.0
    s = str(raw).strip()
    if not s or s == "-":
        return -1.0
    try:
        return max(min(float(s) / 100.0, 1.0), 0.0)
    except ValueError:
        return -1.0


def _cn_value(raw: object) -> float | None:
    if not isinstance(raw, str | float | int):
        return None
    s = str(raw).strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _is_pd_visit(visit: object) -> bool:
    return isinstance(visit, str) and visit.strip() in {"Discontinuation", "Progression"}


def _assign_dominant_mechanism(pd_rows: pd.DataFrame) -> ResistanceMechanism | None:
    """Apply hierarchy H2 → H3 within a single patient's PD rows."""
    if pd_rows.empty:
        return None

    egfr_snvs = pd_rows[(pd_rows["GENE"] == "EGFR") & (pd_rows["VARIANT-TYPE"] == "short-variant")]
    for protein in egfr_snvs["SV-PROTEIN-CHANGE"].dropna():
        if isinstance(protein, str) and "C797" in protein:
            return ResistanceMechanism.EGFR_C797S

    cnas = pd_rows[pd_rows["VARIANT-TYPE"] == "copy-number-alteration"].copy()
    if cnas.empty:
        return ResistanceMechanism.other_or_unknown
    cnas["_cn"] = cnas["CNA-COPY-NUMBER"].map(_cn_value)
    cnas = cnas[cnas["_cn"].fillna(0) >= MIN_AMP_CN]
    if cnas.empty:
        return ResistanceMechanism.other_or_unknown

    candidates: dict[ResistanceMechanism, float] = {}
    for _, row in cnas.iterrows():
        gene = row["GENE"]
        cn = row["_cn"]
        if gene == "MET":
            candidates[ResistanceMechanism.MET_amplification] = max(
                candidates.get(ResistanceMechanism.MET_amplification, 0.0), cn
            )
        elif gene == HER2_GENE:
            candidates[ResistanceMechanism.HER2_amplification] = max(
                candidates.get(ResistanceMechanism.HER2_amplification, 0.0), cn
            )
        elif gene == "EGFR":
            candidates[ResistanceMechanism.EGFR_amplification] = max(
                candidates.get(ResistanceMechanism.EGFR_amplification, 0.0), cn
            )

    if not candidates:
        return ResistanceMechanism.other_or_unknown
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def _build_variant_row(
    pid: str, raw_row: pd.Series, sample_date: date, mechanism: ResistanceMechanism | None
) -> dict:
    visit = raw_row.get("Visit", "")
    variant_type = raw_row.get("VARIANT-TYPE", "")
    gene = str(raw_row.get("GENE", "")).upper() or "UNKNOWN"
    protein = _normalize_hgvsp(raw_row.get("SV-PROTEIN-CHANGE"))

    if variant_type == "short-variant":
        alteration = AlterationType.SNV
        vaf = _vaf_from_percent(raw_row.get("SV-PERCENT-READS"))
    elif variant_type == "copy-number-alteration":
        alteration = AlterationType.CNV_amplification
        vaf = -1.0
    elif variant_type == "rearrangement":
        alteration = AlterationType.rearrangement
        vaf = -1.0
    else:
        alteration = AlterationType.SNV
        vaf = -1.0

    is_c797s = (
        gene == "EGFR"
        and variant_type == "short-variant"
        and isinstance(protein, str)
        and "C797" in protein
    )
    is_baseline_driver = (
        gene == "EGFR"
        and variant_type == "short-variant"
        and isinstance(protein, str)
        and ("C797" not in protein and "T790" not in protein)
    )

    is_resistance_amp = (
        variant_type == "copy-number-alteration"
        and _is_pd_visit(visit)
        and gene in {"MET", HER2_GENE, "EGFR"}
        and (_cn_value(raw_row.get("CNA-COPY-NUMBER")) or 0.0) >= MIN_AMP_CN
    )
    is_resistance = bool((is_c797s and _is_pd_visit(visit)) or is_resistance_amp)

    resistance_class: str | None = None
    if is_resistance:
        if is_c797s:
            resistance_class = ResistanceMechanism.EGFR_C797S.value
        elif gene == "MET":
            resistance_class = ResistanceMechanism.MET_amplification.value
        elif gene == HER2_GENE:
            resistance_class = ResistanceMechanism.HER2_amplification.value
        elif gene == "EGFR":
            resistance_class = ResistanceMechanism.EGFR_amplification.value

    return {
        "variant_id": new_uuid(),
        "patient_id": pid,
        "sample_id": f"{pid.split(':', 1)[1]}-{visit}",
        "sample_type": SampleType.ctDNA_plasma.value,
        "sample_date": sample_date,
        "assay": Assay.FoundationOne_Liquid_CDx.value,
        "gene_symbol": gene,
        "alteration_type": alteration.value,
        "protein_change_hgvs": protein,
        "vaf": vaf,
        "read_depth": -1,
        "oncokb_oncogenic": OncoKBOncogenic.unknown.value,
        "is_germline": False,
        "is_baseline_driver": bool(is_baseline_driver and visit == "C1D1"),
        "is_resistance_call": is_resistance,
        "resistance_mechanism_class": resistance_class,
        # mechanism arg keeps the patient-level call available for callers that need it
        "_dominant_mechanism": mechanism.value if mechanism else None,
    }


@register
class ChmieleckiFlauraAdapter(PaperAdapter):
    study_id = "chmielecki_2023_flaura"
    source_dataset = SourceDataset.FLAURA_SUPP
    citation = "Chmielecki J et al. Nat Commun 14, 1070 (2023). DOI:10.1038/s41467-023-35961-y."

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
            supplement, sheet_name=OSIMERTINIB_ARM_SHEET, header=0, engine="openpyxl", dtype=object
        )
        df = df[df["Patient_ID"].notna()].copy()

        patients: list[dict] = []
        variants: list[dict] = []
        treatments: list[dict] = []
        outcomes: list[dict] = []

        for raw_pid, group in df.groupby("Patient_ID"):
            pid = self.make_patient_id(str(raw_pid))

            baseline_egfr = group[
                (group["Visit"] == "C1D1")
                & (group["GENE"] == "EGFR")
                & (group["VARIANT-TYPE"] == "short-variant")
            ]
            hgvsp_for_class = [
                _normalize_hgvsp(p)
                for p in baseline_egfr["SV-PROTEIN-CHANGE"]
                if isinstance(p, str)
            ]
            # Fall back to PD-sample EGFR variants if no baseline (still a FLAURA enrollee).
            if not hgvsp_for_class:
                pd_egfr = group[
                    group["Visit"].isin({"Discontinuation", "Progression"})
                    & (group["GENE"] == "EGFR")
                    & (group["VARIANT-TYPE"] == "short-variant")
                ]
                hgvsp_for_class = [
                    _normalize_hgvsp(p) for p in pd_egfr["SV-PROTEIN-CHANGE"] if isinstance(p, str)
                ]
            hgvsp_for_class = [h for h in hgvsp_for_class if h]
            variant_class = classify_egfr_variant(hgvsp_for_class)

            pd_rows = group[group["Visit"].isin({"Discontinuation", "Progression"})]
            dominant = _assign_dominant_mechanism(pd_rows)

            included = variant_class != EgfrVariantClass.unknown
            exclusion = None if included else "not_EGFR_sensitizing"

            patients.append(
                {
                    "patient_id": pid,
                    "source_dataset": self.source_dataset.value,
                    "age_at_diagnosis_years": -1,
                    "sex": Sex.unknown.value,
                    "smoking_status": SmokingStatus.unknown.value,
                    "diagnosis_date": ENROLLMENT_EPOCH,
                    "stage_at_diagnosis": StageAtDiagnosis.IV_NOS.value,
                    "histology": Histology.adenocarcinoma.value,
                    "egfr_variant_class": variant_class.value,
                    "site_id": "FLAURA_trial",
                    "vital_status_at_last_followup": VitalStatus.unknown.value,
                    "last_followup_date": DATA_CUTOFF,
                    "included_in_v1_cohort": included,
                    "exclusion_reason": exclusion,
                }
            )

            treatments.append(
                {
                    "treatment_id": new_uuid(),
                    "patient_id": pid,
                    "drug_name": "osimertinib",
                    "line_of_therapy": LineOfTherapy.first_line.value,
                    "start_date": ENROLLMENT_EPOCH,
                    "end_date": DATA_CUTOFF if not pd_rows.empty else None,
                    "is_osimertinib": True,
                }
            )

            outcomes.append(
                {
                    "outcome_id": new_uuid(),
                    "patient_id": pid,
                    "event_type": OutcomeEventType.last_followup.value,
                    "event_date": DATA_CUTOFF,
                    "recist_response": None,
                    "resistance_mechanism_class": None,
                }
            )
            if not pd_rows.empty:
                outcomes.append(
                    {
                        "outcome_id": new_uuid(),
                        "patient_id": pid,
                        "event_type": OutcomeEventType.progression_recist.value,
                        "event_date": DATA_CUTOFF,
                        "recist_response": RecistResponse.progressive_disease.value,
                        "resistance_mechanism_class": dominant.value if dominant else None,
                    }
                )

            for _, raw_row in group.iterrows():
                sample_date = ENROLLMENT_EPOCH if raw_row["Visit"] == "C1D1" else DATA_CUTOFF
                row_dict = _build_variant_row(pid, raw_row, sample_date, dominant)
                row_dict.pop("_dominant_mechanism", None)
                variants.append(row_dict)

        return ParsedTables(
            patients=pd.DataFrame(patients),
            variants=pd.DataFrame(variants),
            treatments=pd.DataFrame(treatments),
            outcomes=pd.DataFrame(outcomes),
        )
