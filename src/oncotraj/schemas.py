"""Pydantic schemas for the four OncoTraj Parquet tables.

Mirrors DATASET_SPEC.md v1.0.0. Coverage in v0.1.0 is the subset of fields
the GENIE + cBioPortal parsers populate. Fields not yet implemented are
intentionally absent; adding them is additive (schema minor bump).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceDataset(StrEnum):
    GENIE_BPC = "GENIE_BPC"
    MSK_CHORD = "MSK_CHORD"
    TRACERX = "TRACERX"
    FLAURA_SUPP = "FLAURA_SUPP"
    AURA3_SUPP = "AURA3_SUPP"
    ELIOS_SUPP = "ELIOS_SUPP"
    FMI_REGISTRY = "FMI_REGISTRY"
    GUARDANT_REGISTRY = "GUARDANT_REGISTRY"


class Sex(StrEnum):
    male = "male"
    female = "female"
    unknown = "unknown"


class SmokingStatus(StrEnum):
    never = "never"
    former = "former"
    current = "current"
    unknown = "unknown"


class StageAtDiagnosis(StrEnum):
    I = "I"  # noqa: E741
    II = "II"
    IIIA = "IIIA"
    IIIB = "IIIB"
    IIIC = "IIIC"
    IVA = "IVA"
    IVB = "IVB"
    IV_NOS = "IV_NOS"
    unknown = "unknown"


class Histology(StrEnum):
    adenocarcinoma = "adenocarcinoma"
    squamous = "squamous"
    adenosquamous = "adenosquamous"
    large_cell = "large_cell"
    NSCLC_NOS = "NSCLC_NOS"
    small_cell = "small_cell"
    other = "other"


class EgfrVariantClass(StrEnum):
    exon19del = "exon19del"
    L858R = "L858R"
    G719X = "G719X"
    L861Q = "L861Q"
    S768I = "S768I"
    exon20ins = "exon20ins"
    compound_sensitizing = "compound_sensitizing"
    compound_with_resistance = "compound_with_resistance"
    unknown = "unknown"


class VitalStatus(StrEnum):
    alive = "alive"
    deceased = "deceased"
    unknown = "unknown"


class ResistanceMechanism(StrEnum):
    EGFR_C797S = "EGFR_C797S"
    MET_amplification = "MET_amplification"
    HER2_amplification = "HER2_amplification"
    EGFR_amplification = "EGFR_amplification"
    histologic_transformation = "histologic_transformation"
    other_or_unknown = "other_or_unknown"


class SampleType(StrEnum):
    tumor_tissue = "tumor_tissue"
    ctDNA_plasma = "ctDNA_plasma"
    ctDNA_cfDNA = "ctDNA_cfDNA"
    normal_blood = "normal_blood"
    normal_tissue = "normal_tissue"
    unknown = "unknown"


class Assay(StrEnum):
    MSK_IMPACT_341 = "MSK_IMPACT_341"
    MSK_IMPACT_410 = "MSK_IMPACT_410"
    MSK_IMPACT_468 = "MSK_IMPACT_468"
    MSK_ACCESS = "MSK_ACCESS"
    FoundationOne_CDx = "FoundationOne_CDx"
    FoundationOne_Liquid_CDx = "FoundationOne_Liquid_CDx"
    Guardant360 = "Guardant360"
    Guardant360_CDx = "Guardant360_CDx"
    Tempus_xF = "Tempus_xF"
    Tempus_xT = "Tempus_xT"
    WES = "WES"
    WGS = "WGS"
    custom_panel = "custom_panel"
    unknown = "unknown"


class AlterationType(StrEnum):
    SNV = "SNV"
    indel_insertion = "indel_insertion"
    indel_deletion = "indel_deletion"
    indel_complex = "indel_complex"
    CNV_amplification = "CNV_amplification"
    CNV_deletion = "CNV_deletion"
    fusion = "fusion"
    rearrangement = "rearrangement"
    splice = "splice"
    promoter = "promoter"


class OncoKBOncogenic(StrEnum):
    oncogenic = "oncogenic"
    likely_oncogenic = "likely_oncogenic"
    predicted_oncogenic = "predicted_oncogenic"
    resistance = "resistance"
    likely_neutral = "likely_neutral"
    inconclusive = "inconclusive"
    unknown = "unknown"


class LineOfTherapy(StrEnum):
    first_line = "first_line"
    second_line = "second_line"
    third_line = "third_line"
    fourth_plus = "fourth_plus"
    unknown = "unknown"


class OutcomeEventType(StrEnum):
    osimertinib_start = "osimertinib_start"
    osimertinib_stop = "osimertinib_stop"
    progression_recist = "progression_recist"
    molecular_resistance = "molecular_resistance"
    death = "death"
    last_followup = "last_followup"


class RecistResponse(StrEnum):
    complete_response = "complete_response"
    partial_response = "partial_response"
    stable_disease = "stable_disease"
    progressive_disease = "progressive_disease"
    not_evaluable = "not_evaluable"
    unknown = "unknown"


class PatientRecord(BaseModel):
    """One row of patients.parquet. See DATASET_SPEC.md §2."""

    model_config = ConfigDict(use_enum_values=True)

    patient_id: str = Field(pattern=r"^[A-Z0-9_]+:[A-Za-z0-9_\-\.]+$")
    source_dataset: SourceDataset
    age_at_diagnosis_years: int = Field(ge=-1, le=110)
    sex: Sex
    smoking_status: SmokingStatus
    diagnosis_date: date
    stage_at_diagnosis: StageAtDiagnosis
    histology: Histology
    egfr_variant_class: EgfrVariantClass
    site_id: str = Field(min_length=1, max_length=32)
    vital_status_at_last_followup: VitalStatus
    last_followup_date: date
    included_in_v1_cohort: bool
    exclusion_reason: str | None = None


class VariantRecord(BaseModel):
    """One row of variants.parquet. See DATASET_SPEC.md §3."""

    model_config = ConfigDict(use_enum_values=True)

    variant_id: str
    patient_id: str
    sample_id: str = Field(min_length=1, max_length=64)
    sample_type: SampleType
    sample_date: date
    assay: Assay
    gene_symbol: str = Field(min_length=1)
    alteration_type: AlterationType
    protein_change_hgvs: str | None = None
    vaf: float = Field(ge=-1.0, le=1.0)
    read_depth: int = Field(ge=-1, le=100_000)
    oncokb_oncogenic: OncoKBOncogenic
    is_germline: bool | None = None
    is_baseline_driver: bool
    is_resistance_call: bool
    resistance_mechanism_class: ResistanceMechanism | None = None


class TreatmentRecord(BaseModel):
    """One row of treatments.parquet. See DATASET_SPEC.md §4."""

    model_config = ConfigDict(use_enum_values=True)

    treatment_id: str
    patient_id: str
    drug_name: str = Field(min_length=1)
    line_of_therapy: LineOfTherapy
    start_date: date
    end_date: date | None = None
    is_osimertinib: bool


class OutcomeRecord(BaseModel):
    """One row of outcomes.parquet. See DATASET_SPEC.md §5."""

    model_config = ConfigDict(use_enum_values=True)

    outcome_id: str
    patient_id: str
    event_type: OutcomeEventType
    event_date: date
    recist_response: RecistResponse | None = None
    resistance_mechanism_class: ResistanceMechanism | None = None
