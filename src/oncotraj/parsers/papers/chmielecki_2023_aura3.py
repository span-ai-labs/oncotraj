"""Chmielecki et al. 2023 — AURA3 molecular-resistance supplement adapter.

Citation: Chmielecki J, Mok T, Wu YL, et al. "Analysis of acquired resistance
mechanisms to osimertinib in patients with EGFR-mutated advanced non-small
cell lung cancer from the AURA3 trial." Nat Commun 14, 1071 (2023).
DOI: 10.1038/s41467-023-35962-x.  PMID: 36849516.  PMC: PMC9971022.

Cohort treatment: per DATASET_SPEC.md §10 inclusion/exclusion, AURA3 patients
have prior EGFR TKI exposure (the trial enrolled T790M-acquired post-1L
patients), so they are explicitly excluded from the v1 training cohort with
`exclusion_reason = "prior_EGFR_TKI"`. They are retained in the parquet for
the held-out sensitivity slice analysis per PAPER_OUTLINE.md §0.

Supplement consumed: Supplementary Data 1 (`41467_2023_35962_MOESM2_ESM.xlsx`),
sheet `AURA3_osimertinib arm`. The chemotherapy arm is excluded.
The column carrying the visit type is `Visit_name` (vs. `Visit` in FLAURA).

Yield: 78 patients in the osimertinib arm (matches the published n).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from oncotraj.schemas import (
    EgfrVariantClass,
    Histology,
    LineOfTherapy,
    OutcomeEventType,
    RecistResponse,
    Sex,
    SmokingStatus,
    SourceDataset,
    StageAtDiagnosis,
    VitalStatus,
)

from ..common import ParsedTables, classify_egfr_variant, new_uuid
from . import register
from ._base import PaperAdapter
from .chmielecki_2023_flaura import (
    _assign_dominant_mechanism,
    _build_variant_row,
    _normalize_hgvsp,
)

OSIMERTINIB_ARM_SHEET = "AURA3_osimertinib arm"
# AURA3 enrolled 2014-2015 (Mok 2017 NEJM primary analysis); molecular-resistance
# analysis used the November 2016 data cut + subsequent biomarker follow-up.
ENROLLMENT_EPOCH = date(2015, 1, 1)
DATA_CUTOFF = date(2018, 6, 30)


def _find_supplement(study_dir: Path) -> Path | None:
    candidates = [
        study_dir / "MOESM2.xlsx",
        study_dir / "41467_2023_35962_MOESM2_ESM.xlsx",
        study_dir / "pdf" / "MOESM2.xlsx",
        study_dir / "pdf" / "41467_2023_35962_MOESM2_ESM.xlsx",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(study_dir.glob("**/*MOESM2*.xlsx"))
    return matches[0] if matches else None


@register
class ChmieleckiAura3Adapter(PaperAdapter):
    study_id = "chmielecki_2023_aura3"
    source_dataset = SourceDataset.AURA3_SUPP
    citation = "Chmielecki J et al. Nat Commun 14, 1071 (2023). DOI:10.1038/s41467-023-35962-x."

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
            supplement,
            sheet_name=OSIMERTINIB_ARM_SHEET,
            header=0,
            engine="openpyxl",
            dtype=object,
        )
        df = df[df["Patient_ID"].notna()].copy()
        # Normalise FLAURA-vs-AURA3 column-name drift so shared helpers work.
        df["Visit"] = df["Visit_name"]

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

            # AURA3 enrolled post-1L-TKI patients; this is the per-DATASET_SPEC.md §10
            # "prior_EGFR_TKI" exclusion. Emitted for the sensitivity-slice analysis only.
            included = False
            exclusion = "prior_EGFR_TKI"
            # If the EGFR variant couldn't be classified, layer that reason on top.
            if variant_class == EgfrVariantClass.unknown:
                exclusion = "not_EGFR_sensitizing"

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
                    "site_id": "AURA3_trial",
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
                    "line_of_therapy": LineOfTherapy.second_line.value,
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
