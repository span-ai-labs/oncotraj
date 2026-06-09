"""Reference adapter — used to prove the framework wiring end-to-end.

NOT a real study. study_id is deliberately `_example_synthetic` so any
downstream filter can exclude it. Real-study adapters should mirror this
shape but live in their own module with a non-underscore study_id.

Input convention this adapter demonstrates:
    <papers_root>/_example_synthetic/manual/cohort.csv
    columns: stable_id, sex, smoking_status, diagnosis_date, egfr_hgvsp,
             osimertinib_start_date, last_followup_date, vital_status,
             progression_date (blank if none)
"""

from __future__ import annotations

from datetime import date

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
    SampleType,
    Sex,
    SmokingStatus,
    SourceDataset,
    StageAtDiagnosis,
    VitalStatus,
)

from ..common import ParsedTables, classify_egfr_variant, new_uuid
from . import register
from ._base import PaperAdapter, read_manual_csv


def _as_str(v: object) -> str:
    return v.strip() if isinstance(v, str) else ""


def _parse_sex(v: object) -> Sex:
    return {"male": Sex.male, "female": Sex.female}.get(_as_str(v).lower(), Sex.unknown)


def _parse_smoke(v: object) -> SmokingStatus:
    return {
        "never": SmokingStatus.never,
        "former": SmokingStatus.former,
        "current": SmokingStatus.current,
    }.get(_as_str(v).lower(), SmokingStatus.unknown)


def _parse_vital(v: object) -> VitalStatus:
    return {"alive": VitalStatus.alive, "deceased": VitalStatus.deceased}.get(
        _as_str(v).lower(), VitalStatus.unknown
    )


def _parse_date(v: object) -> date | None:
    s = _as_str(v)
    return date.fromisoformat(s) if s else None


@register
class ExampleSyntheticAdapter(PaperAdapter):
    study_id = "_example_synthetic"
    source_dataset = SourceDataset.FLAURA_SUPP  # arbitrary; never confuse with a real source
    citation = "Not a real study — fixture adapter for framework tests."

    def load(self) -> ParsedTables:
        csvs = self.inputs.manual_csvs()
        if not csvs:
            return ParsedTables(
                patients=pd.DataFrame(),
                variants=pd.DataFrame(),
                treatments=pd.DataFrame(),
                outcomes=pd.DataFrame(),
            )

        frames = [read_manual_csv(p) for p in csvs]
        df = pd.concat(frames, ignore_index=True)

        patients: list[dict] = []
        variants: list[dict] = []
        treatments: list[dict] = []
        outcomes: list[dict] = []

        for _, row in df.iterrows():
            stable = row["stable_id"]
            pid = self.make_patient_id(stable)
            hgvsp = _as_str(row.get("egfr_hgvsp", ""))
            variant_class = classify_egfr_variant([hgvsp])
            dx = _parse_date(row.get("diagnosis_date", ""))
            osi_start = _parse_date(row.get("osimertinib_start_date", ""))
            last_fu = _parse_date(row.get("last_followup_date", ""))
            prog = _parse_date(row.get("progression_date", ""))

            included = variant_class != EgfrVariantClass.unknown and osi_start is not None
            exclusion: str | None
            if not included:
                exclusion = (
                    "not_EGFR_sensitizing"
                    if variant_class == EgfrVariantClass.unknown
                    else "not_first_line_osimertinib"
                )
            else:
                exclusion = None

            patients.append(
                {
                    "patient_id": pid,
                    "source_dataset": self.source_dataset.value,
                    "age_at_diagnosis_years": -1,
                    "sex": _parse_sex(row.get("sex", "")).value,
                    "smoking_status": _parse_smoke(row.get("smoking_status", "")).value,
                    "diagnosis_date": dx or date(2022, 1, 1),
                    "stage_at_diagnosis": StageAtDiagnosis.IV_NOS.value,
                    "histology": Histology.adenocarcinoma.value,
                    "egfr_variant_class": variant_class.value,
                    "site_id": "literature",
                    "vital_status_at_last_followup": _parse_vital(
                        row.get("vital_status", "")
                    ).value,
                    "last_followup_date": last_fu or dx or date(2022, 1, 1),
                    "included_in_v1_cohort": included,
                    "exclusion_reason": exclusion,
                }
            )

            if hgvsp:
                variants.append(
                    {
                        "variant_id": new_uuid(),
                        "patient_id": pid,
                        "sample_id": f"{stable}-baseline",
                        "sample_type": SampleType.ctDNA_plasma.value,
                        "sample_date": dx or date(2022, 1, 1),
                        "assay": Assay.unknown.value,
                        "gene_symbol": "EGFR",
                        "alteration_type": AlterationType.SNV.value,
                        "protein_change_hgvs": hgvsp,
                        "vaf": -1.0,
                        "read_depth": -1,
                        "oncokb_oncogenic": OncoKBOncogenic.unknown.value,
                        "is_germline": False,
                        "is_baseline_driver": True,
                        "is_resistance_call": False,
                        "resistance_mechanism_class": None,
                    }
                )

            if osi_start:
                treatments.append(
                    {
                        "treatment_id": new_uuid(),
                        "patient_id": pid,
                        "drug_name": "osimertinib",
                        "line_of_therapy": LineOfTherapy.first_line.value,
                        "start_date": osi_start,
                        "end_date": None,
                        "is_osimertinib": True,
                    }
                )

            if last_fu:
                outcomes.append(
                    {
                        "outcome_id": new_uuid(),
                        "patient_id": pid,
                        "event_type": OutcomeEventType.last_followup.value,
                        "event_date": last_fu,
                        "recist_response": None,
                        "resistance_mechanism_class": None,
                    }
                )
            if prog:
                outcomes.append(
                    {
                        "outcome_id": new_uuid(),
                        "patient_id": pid,
                        "event_type": OutcomeEventType.progression_recist.value,
                        "event_date": prog,
                        "recist_response": RecistResponse.progressive_disease.value,
                        "resistance_mechanism_class": None,
                    }
                )

        return ParsedTables(
            patients=pd.DataFrame(patients),
            variants=pd.DataFrame(variants),
            treatments=pd.DataFrame(treatments),
            outcomes=pd.DataFrame(outcomes),
        )
