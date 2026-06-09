"""cBioPortal synthetic-fixture parser (kept for tests; superseded by chord.py).

**Status:** this parser was written against the synthetic-fixture layout
emitted by `parsers/synthetic.write_synthetic_cbioportal()` for the v0
tests. The real MSK-CHORD 2024 distribution (~25k patients, anonymised
integer-offset dates, separate `data_timeline_progression.txt`) is
parsed by `parsers/chord.py` instead. Tests still drive this module
through the synthetic fixture so the CI path stays self-contained.

The expected files for the synthetic fixture:
- data_clinical_patient.txt
- data_clinical_sample.txt
- data_mutations.txt
- data_timeline_treatment.txt
- data_timeline_status.txt
"""

from __future__ import annotations

import contextlib
import tarfile
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

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

from .common import (
    C797S_RE,
    ParsedTables,
    classify_egfr_variant,
    is_osimertinib,
    make_patient_id,
    new_uuid,
)

SOURCE_PREFIX = "MSK_CHORD"
CBIOPORTAL_DATAHUB = "https://cbioportal-datahub.s3.amazonaws.com"


def fetch_cbioportal(
    raw_dir: Path, study_id: str = "msk_chord_2024"
) -> Path:  # pragma: no cover - network
    """Download and extract a cBioPortal study tarball into `raw_dir`.

    Public datahub mirror; no auth required. Network access still has to be
    available — `--use-synthetic` is the right flag for offline runs.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{CBIOPORTAL_DATAHUB}/{study_id}.tar.gz"
    archive_path = raw_dir / f"{study_id}.tar.gz"
    with urlopen(url) as resp, open(archive_path, "wb") as out:
        out.write(resp.read())
    with tarfile.open(archive_path) as tf:
        tf.extractall(raw_dir)
    return raw_dir


def _read_clinical_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, na_values=["", "NA", "Unknown"])


def _map_sex(v: object) -> Sex:
    if not isinstance(v, str):
        return Sex.unknown
    s = v.strip().lower()
    return {"male": Sex.male, "female": Sex.female}.get(s, Sex.unknown)


def _map_smoke(v: object) -> SmokingStatus:
    if not isinstance(v, str):
        return SmokingStatus.unknown
    return {
        "never": SmokingStatus.never,
        "former": SmokingStatus.former,
        "current": SmokingStatus.current,
    }.get(v.strip().lower(), SmokingStatus.unknown)


def _map_vital_cbio(v: object) -> VitalStatus:
    """cBioPortal encodes OS_STATUS as '0:LIVING' / '1:DECEASED'."""
    if not isinstance(v, str):
        return VitalStatus.unknown
    s = v.strip().lower()
    if "deceased" in s or s.startswith("1"):
        return VitalStatus.deceased
    if "living" in s or s.startswith("0"):
        return VitalStatus.alive
    return VitalStatus.unknown


_VC_MAP = {
    "missense_mutation": AlterationType.SNV,
    "nonsense_mutation": AlterationType.SNV,
    "splice_site": AlterationType.splice,
    "in_frame_del": AlterationType.indel_deletion,
    "in_frame_ins": AlterationType.indel_insertion,
    "frame_shift_del": AlterationType.indel_deletion,
    "frame_shift_ins": AlterationType.indel_insertion,
    "amplification": AlterationType.CNV_amplification,
    "deletion": AlterationType.CNV_deletion,
    "fusion": AlterationType.fusion,
}


def _map_alteration(v: object) -> AlterationType:
    if not isinstance(v, str):
        return AlterationType.SNV
    return _VC_MAP.get(v.strip().lower(), AlterationType.SNV)


def _offset_to_date(anchor: date, offset_days: object) -> date | None:
    if offset_days is None or (isinstance(offset_days, float) and pd.isna(offset_days)):
        return None
    try:
        return anchor + timedelta(days=int(float(offset_days)))
    except (TypeError, ValueError):
        return None


def parse(raw_dir: Path) -> ParsedTables:
    raw_dir = Path(raw_dir)
    patient_df = _read_clinical_file(raw_dir / "data_clinical_patient.txt")
    sample_df = _read_clinical_file(raw_dir / "data_clinical_sample.txt")
    mutations = pd.read_csv(raw_dir / "data_mutations.txt", sep="\t", dtype=str, na_values=[""])
    treatments = pd.read_csv(
        raw_dir / "data_timeline_treatment.txt", sep="\t", dtype=str, na_values=[""]
    )
    status = pd.read_csv(raw_dir / "data_timeline_status.txt", sep="\t", dtype=str, na_values=[""])

    sample_to_patient = dict(zip(sample_df["SAMPLE_ID"], sample_df["PATIENT_ID"], strict=False))
    sample_to_dx_date: dict[str, date] = {}
    for sid, dx_iso in zip(sample_df["SAMPLE_ID"], sample_df["DX_DATE"], strict=False):
        try:
            sample_to_dx_date[sid] = (
                date.fromisoformat(dx_iso) if isinstance(dx_iso, str) else date(2022, 1, 1)
            )
        except ValueError:
            sample_to_dx_date[sid] = date(2022, 1, 1)

    mutations["_patient_id"] = mutations["Tumor_Sample_Barcode"].map(sample_to_patient)
    egfr_hgvsp = (
        mutations.loc[mutations["Hugo_Symbol"].str.upper() == "EGFR"]
        .groupby("_patient_id")["HGVSp_Short"]
        .apply(lambda s: [x for x in s.tolist() if isinstance(x, str)])
        .to_dict()
    )
    met_amplified = set(
        mutations.loc[
            (mutations["Hugo_Symbol"].str.upper() == "MET")
            & (mutations["Variant_Classification"].str.lower() == "amplification"),
            "_patient_id",
        ].dropna()
    )

    osi_patients: set[str] = set()
    for _, row in treatments.iterrows():
        if is_osimertinib(row.get("AGENT", "")):
            osi_patients.add(row["PATIENT_ID"])

    patient_records: list[dict] = []
    treatment_records: list[dict] = []
    outcome_records: list[dict] = []

    for _, row in patient_df.iterrows():
        raw_pid = row["PATIENT_ID"]
        pid = make_patient_id(SOURCE_PREFIX, raw_pid)
        hgvsp_list = egfr_hgvsp.get(raw_pid, [])
        variant_class = classify_egfr_variant(hgvsp_list)
        included = variant_class != EgfrVariantClass.unknown and raw_pid in osi_patients
        exclusion: str | None = None
        if not included:
            if variant_class == EgfrVariantClass.unknown:
                exclusion = "not_EGFR_sensitizing"
            elif raw_pid not in osi_patients:
                exclusion = "not_first_line_osimertinib"

        sample_rows_for_pt = sample_df.loc[sample_df["PATIENT_ID"] == raw_pid]
        dx_date = date(2022, 1, 1)
        if not sample_rows_for_pt.empty:
            dx_iso = sample_rows_for_pt.iloc[0].get("DX_DATE")
            if isinstance(dx_iso, str):
                with contextlib.suppress(ValueError):
                    dx_date = date.fromisoformat(dx_iso)

        try:
            os_months = float(row.get("OS_MONTHS"))
            last_fu = dx_date + timedelta(days=int(os_months * 30))
        except (TypeError, ValueError):
            last_fu = dx_date + timedelta(days=180)

        try:
            age = (
                int(float(sample_rows_for_pt.iloc[0].get("AGE_AT_SEQ_REPORT")))
                if not sample_rows_for_pt.empty
                else -1
            )
            age = age if 18 <= age <= 110 else -1
        except (TypeError, ValueError):
            age = -1

        patient_records.append(
            {
                "patient_id": pid,
                "source_dataset": SourceDataset.MSK_CHORD.value,
                "age_at_diagnosis_years": age,
                "sex": _map_sex(row.get("SEX")).value,
                "smoking_status": _map_smoke(row.get("SMOKING_HISTORY")).value,
                "diagnosis_date": dx_date,
                "stage_at_diagnosis": StageAtDiagnosis.IV_NOS.value,
                "histology": Histology.adenocarcinoma.value,
                "egfr_variant_class": variant_class.value,
                "site_id": (row.get("CENTER") or "msk")[:32],
                "vital_status_at_last_followup": _map_vital_cbio(row.get("OS_STATUS")).value,
                "last_followup_date": last_fu,
                "included_in_v1_cohort": included,
                "exclusion_reason": exclusion,
            }
        )

        for _, tx in treatments.loc[treatments["PATIENT_ID"] == raw_pid].iterrows():
            start = _offset_to_date(dx_date, tx.get("START_DATE")) or dx_date
            stop = _offset_to_date(dx_date, tx.get("STOP_DATE"))
            agent = (tx.get("AGENT") or "unknown").strip()
            treatment_records.append(
                {
                    "treatment_id": new_uuid(),
                    "patient_id": pid,
                    "drug_name": agent,
                    "line_of_therapy": LineOfTherapy.first_line.value
                    if str(tx.get("LINE", "")).strip() == "1"
                    else LineOfTherapy.unknown.value,
                    "start_date": start,
                    "end_date": stop,
                    "is_osimertinib": is_osimertinib(agent),
                }
            )

        outcome_records.append(
            {
                "outcome_id": new_uuid(),
                "patient_id": pid,
                "event_type": OutcomeEventType.last_followup.value,
                "event_date": last_fu,
                "recist_response": None,
                "resistance_mechanism_class": None,
            }
        )
        if raw_pid in met_amplified:
            outcome_records.append(
                {
                    "outcome_id": new_uuid(),
                    "patient_id": pid,
                    "event_type": OutcomeEventType.molecular_resistance.value,
                    "event_date": last_fu,
                    "recist_response": None,
                    "resistance_mechanism_class": ResistanceMechanism.MET_amplification.value,
                }
            )
        for _, st in status.loc[status["PATIENT_ID"] == raw_pid].iterrows():
            event_type = (st.get("EVENT_TYPE") or "").lower()
            event_date = _offset_to_date(dx_date, st.get("START_DATE")) or last_fu
            if "progression" in event_type:
                outcome_records.append(
                    {
                        "outcome_id": new_uuid(),
                        "patient_id": pid,
                        "event_type": OutcomeEventType.progression_recist.value,
                        "event_date": event_date,
                        "recist_response": RecistResponse.progressive_disease.value,
                        "resistance_mechanism_class": None,
                    }
                )

    variant_records: list[dict] = []
    for _, mrow in mutations.iterrows():
        sid = mrow.get("Tumor_Sample_Barcode")
        raw_pid = sample_to_patient.get(sid)
        if raw_pid is None:
            continue
        gene = (mrow.get("Hugo_Symbol") or "").upper()
        try:
            vaf_val = float(mrow.get("vaf") or 0.0)
            vaf_val = max(min(vaf_val, 1.0), -1.0)
        except (TypeError, ValueError):
            vaf_val = -1.0
        try:
            depth = int(float(mrow.get("t_depth") or -1))
        except (TypeError, ValueError):
            depth = -1
        protein = mrow.get("HGVSp_Short")
        if not isinstance(protein, str) or not protein.strip():
            protein = None
        alteration = _map_alteration(mrow.get("Variant_Classification"))
        is_resistance = (
            gene == "EGFR" and isinstance(protein, str) and bool(C797S_RE.search(protein))
        )
        resistance_class: str | None = None
        if alteration == AlterationType.CNV_amplification and gene == "MET":
            is_resistance = True
            resistance_class = ResistanceMechanism.MET_amplification.value
        elif is_resistance:
            resistance_class = ResistanceMechanism.EGFR_C797S.value
        variant_records.append(
            {
                "variant_id": new_uuid(),
                "patient_id": make_patient_id(SOURCE_PREFIX, raw_pid),
                "sample_id": sid,
                "sample_type": SampleType.tumor_tissue.value,
                "sample_date": sample_to_dx_date.get(sid, date(2022, 1, 1)),
                "assay": Assay.MSK_IMPACT_468.value,
                "gene_symbol": gene or "UNKNOWN",
                "alteration_type": alteration.value,
                "protein_change_hgvs": protein,
                "vaf": vaf_val,
                "read_depth": depth,
                "oncokb_oncogenic": OncoKBOncogenic.unknown.value,
                "is_germline": False,
                "is_baseline_driver": gene == "EGFR",
                "is_resistance_call": bool(is_resistance),
                "resistance_mechanism_class": resistance_class,
            }
        )

    return ParsedTables(
        patients=pd.DataFrame(patient_records),
        variants=pd.DataFrame(variant_records),
        treatments=pd.DataFrame(treatment_records),
        outcomes=pd.DataFrame(outcome_records),
    )
