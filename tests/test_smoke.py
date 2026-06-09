from datetime import date

from oncotraj import SCHEMA_VERSION, __version__
from oncotraj.schemas import (
    EgfrVariantClass,
    Histology,
    PatientRecord,
    Sex,
    SmokingStatus,
    SourceDataset,
    StageAtDiagnosis,
)


def test_version():
    assert __version__ == "0.1.0"
    assert SCHEMA_VERSION == "oncotraj-schema/1.0.0"


def test_patient_record_validates():
    rec = PatientRecord(
        patient_id="GENIE_BPC:GENIE-MSK-P-0001234",
        source_dataset=SourceDataset.GENIE_BPC,
        age_at_diagnosis_years=62,
        sex=Sex.female,
        smoking_status=SmokingStatus.never,
        diagnosis_date=date(2022, 4, 15),
        stage_at_diagnosis=StageAtDiagnosis.IVA,
        histology=Histology.adenocarcinoma,
        egfr_variant_class=EgfrVariantClass.exon19del,
        site_id="site_a1b2",
        vital_status_at_last_followup="alive",
        last_followup_date=date(2024, 9, 1),
        included_in_v1_cohort=True,
    )
    assert rec.patient_id.startswith("GENIE_BPC:")
