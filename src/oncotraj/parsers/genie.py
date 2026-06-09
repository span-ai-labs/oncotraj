"""AACR Project GENIE BPC NSCLC parser.

Reads the standard GENIE distribution files from a local cache directory and
emits the four-table OncoTraj contract. Real-data acquisition is via Synapse
(`fetch_genie`); the parser itself operates on local files so tests can drive
it with synthetic fixtures.

Source files consumed (per the BPC NSCLC v3.0 distribution):
- data_clinical_patient.txt    — one row per patient, 4-line '#' header.
- data_clinical_sample.txt     — one row per sequenced sample.
- data_mutations_extended.txt  — MAF (one row per variant call).
- data_timeline_treatment.txt  — per-patient treatment intervals.
- data_timeline_status.txt     — per-patient status / progression events.

Field-level transformations are documented in parsers/README.md.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
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
    SampleType,
    Sex,
    SmokingStatus,
    SourceDataset,
    StageAtDiagnosis,
    VitalStatus,
)

from .common import (
    ParsedTables,
    classify_egfr_variant,
    is_osimertinib,
    make_patient_id,
    new_uuid,
)

SOURCE_PREFIX = "GENIE_BPC"
DEFAULT_DIAGNOSIS_DATE = date(2022, 1, 1)


def fetch_genie(raw_dir: Path, release: str = "syn7222066") -> Path:  # pragma: no cover - network
    """Download GENIE BPC NSCLC from Synapse to `raw_dir`.

    Requires `SYNAPSE_AUTH_TOKEN` in the environment. We deliberately do not
    bundle credentials; if the token is missing this raises rather than
    fabricating a fallback path.
    """
    token = os.environ.get("SYNAPSE_AUTH_TOKEN")
    if not token:
        raise RuntimeError(
            "SYNAPSE_AUTH_TOKEN is not set. Either export it (https://help.synapse.org/) "
            "or run the build with --use-synthetic for a no-network smoke run."
        )
    try:
        import synapseclient
    except ImportError as e:
        raise RuntimeError("synapseclient is not installed. Run: uv sync --extra synapse") from e

    raw_dir.mkdir(parents=True, exist_ok=True)
    syn = synapseclient.Synapse()
    syn.login(authToken=token)
    children = syn.getChildren(release)
    for child in children:
        syn.get(child["id"], downloadLocation=str(raw_dir))
    return raw_dir


def _read_clinical_file(path: Path) -> pd.DataFrame:
    """GENIE clinical files have 4 '#' header lines followed by the real header."""
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, na_values=["", "NA", "Unknown"])


_SEX_MAP = {"male": Sex.male, "female": Sex.female}
_SMOKE_MAP = {
    "never": SmokingStatus.never,
    "former": SmokingStatus.former,
    "current": SmokingStatus.current,
}
_STAGE_MAP = {
    "i": StageAtDiagnosis.I,
    "ia": StageAtDiagnosis.I,
    "ib": StageAtDiagnosis.I,
    "ii": StageAtDiagnosis.II,
    "iia": StageAtDiagnosis.II,
    "iib": StageAtDiagnosis.II,
    "iiia": StageAtDiagnosis.IIIA,
    "iiib": StageAtDiagnosis.IIIB,
    "iiic": StageAtDiagnosis.IIIC,
    "iva": StageAtDiagnosis.IVA,
    "ivb": StageAtDiagnosis.IVB,
    "iv": StageAtDiagnosis.IV_NOS,
}


def _map_sex(v: object) -> Sex:
    if not isinstance(v, str):
        return Sex.unknown
    return _SEX_MAP.get(v.strip().lower(), Sex.unknown)


def _map_smoke(v: object) -> SmokingStatus:
    if not isinstance(v, str):
        return SmokingStatus.unknown
    return _SMOKE_MAP.get(v.strip().lower(), SmokingStatus.unknown)


def _map_stage(v: object) -> StageAtDiagnosis:
    if not isinstance(v, str):
        return StageAtDiagnosis.unknown
    return _STAGE_MAP.get(v.strip().lower(), StageAtDiagnosis.unknown)


def _map_vital(v: object) -> VitalStatus:
    if not isinstance(v, str):
        return VitalStatus.unknown
    s = v.strip().lower()
    if "living" in s or s == "alive":
        return VitalStatus.alive
    if "deceased" in s or s == "dead":
        return VitalStatus.deceased
    return VitalStatus.unknown


def _map_assay(seq_assay_id: object) -> Assay:
    if not isinstance(seq_assay_id, str):
        return Assay.unknown
    s = seq_assay_id.upper().replace("-", "_")
    if "IMPACT468" in s:
        return Assay.MSK_IMPACT_468
    if "IMPACT410" in s:
        return Assay.MSK_IMPACT_410
    if "IMPACT341" in s:
        return Assay.MSK_IMPACT_341
    if "ACCESS" in s:
        return Assay.MSK_ACCESS
    if "FOUNDATIONONE_LIQUID" in s or "F1_LIQUID" in s:
        return Assay.FoundationOne_Liquid_CDx
    if "FOUNDATIONONE" in s or "F1" in s:
        return Assay.FoundationOne_CDx
    return Assay.custom_panel


_VARIANT_CLASSIFICATION_MAP = {
    "missense_mutation": AlterationType.SNV,
    "nonsense_mutation": AlterationType.SNV,
    "silent": AlterationType.SNV,
    "splice_site": AlterationType.splice,
    "in_frame_del": AlterationType.indel_deletion,
    "in_frame_ins": AlterationType.indel_insertion,
    "frame_shift_del": AlterationType.indel_deletion,
    "frame_shift_ins": AlterationType.indel_insertion,
    "amplification": AlterationType.CNV_amplification,
    "fusion": AlterationType.fusion,
}


def _map_alteration_type(v: object) -> AlterationType:
    if not isinstance(v, str):
        return AlterationType.SNV
    return _VARIANT_CLASSIFICATION_MAP.get(v.strip().lower(), AlterationType.SNV)


def _offset_to_date(anchor: date, days: object) -> date | None:
    if days is None or (isinstance(days, float) and pd.isna(days)):
        return None
    try:
        return anchor + timedelta(days=int(float(days)))
    except (TypeError, ValueError):
        return None


def parse(raw_dir: Path) -> ParsedTables:
    """Read a GENIE BPC NSCLC cache directory and emit the four-table contract.

    Cohort filter (DATASET_SPEC.md §10):
    - include patients with at least one EGFR sensitizing variant call,
    - and at least one osimertinib treatment record,
    - else emit with `included_in_v1_cohort=False` and an `exclusion_reason`.
    """
    raw_dir = Path(raw_dir)
    patient_df = _read_clinical_file(raw_dir / "data_clinical_patient.txt")
    sample_df = _read_clinical_file(raw_dir / "data_clinical_sample.txt")
    mutations = pd.read_csv(
        raw_dir / "data_mutations_extended.txt", sep="\t", dtype=str, na_values=[""]
    )
    treatments = pd.read_csv(
        raw_dir / "data_timeline_treatment.txt", sep="\t", dtype=str, na_values=[""]
    )
    status = pd.read_csv(raw_dir / "data_timeline_status.txt", sep="\t", dtype=str, na_values=[""])

    sample_to_patient = dict(zip(sample_df["SAMPLE_ID"], sample_df["PATIENT_ID"], strict=False))
    sample_to_assay = dict(zip(sample_df["SAMPLE_ID"], sample_df["SEQ_ASSAY_ID"], strict=False))
    sample_to_dx_date: dict[str, date] = {}
    for sid, dx_iso in zip(sample_df["SAMPLE_ID"], sample_df["DX_DATE"], strict=False):
        try:
            sample_to_dx_date[sid] = (
                date.fromisoformat(dx_iso) if isinstance(dx_iso, str) else DEFAULT_DIAGNOSIS_DATE
            )
        except ValueError:
            sample_to_dx_date[sid] = DEFAULT_DIAGNOSIS_DATE

    mutations["_patient_id"] = mutations["Tumor_Sample_Barcode"].map(sample_to_patient)
    egfr_hgvsp_by_patient: dict[str, list[str]] = (
        mutations.loc[mutations["Hugo_Symbol"].str.upper() == "EGFR"]
        .groupby("_patient_id")["HGVSp_Short"]
        .apply(lambda s: [x for x in s.tolist() if isinstance(x, str)])
        .to_dict()
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
        hgvsp_list = egfr_hgvsp_by_patient.get(raw_pid, [])
        variant_class = classify_egfr_variant(hgvsp_list)

        included = variant_class != EgfrVariantClass.unknown and raw_pid in osi_patients
        exclusion: str | None = None
        if not included:
            if variant_class == EgfrVariantClass.unknown:
                exclusion = "not_EGFR_sensitizing"
            elif raw_pid not in osi_patients:
                exclusion = "not_first_line_osimertinib"

        sample_rows_for_pt = sample_df.loc[sample_df["PATIENT_ID"] == raw_pid]
        if not sample_rows_for_pt.empty:
            dx_iso = sample_rows_for_pt.iloc[0].get("DX_DATE", "")
            try:
                dx_date = (
                    date.fromisoformat(dx_iso)
                    if isinstance(dx_iso, str)
                    else DEFAULT_DIAGNOSIS_DATE
                )
            except ValueError:
                dx_date = DEFAULT_DIAGNOSIS_DATE
        else:
            dx_date = DEFAULT_DIAGNOSIS_DATE

        os_months = row.get("OS_MONTHS")
        try:
            last_fu = dx_date + timedelta(days=int(float(os_months) * 30))
        except (TypeError, ValueError):
            last_fu = dx_date + timedelta(days=180)

        patient_records.append(
            {
                "patient_id": pid,
                "source_dataset": SourceDataset.GENIE_BPC.value,
                "age_at_diagnosis_years": _genie_age_at_diagnosis(sample_rows_for_pt),
                "sex": _map_sex(row.get("SEX")).value,
                "smoking_status": _map_smoke(row.get("SMOKING_HISTORY")).value,
                "diagnosis_date": dx_date,
                "stage_at_diagnosis": _map_stage(row.get("STAGE")).value,
                "histology": Histology.adenocarcinoma.value,
                "egfr_variant_class": variant_class.value,
                "site_id": (row.get("CENTER") or "unknown")[:32],
                "vital_status_at_last_followup": _map_vital(row.get("OS_STATUS")).value,
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
        for _, st in status.loc[status["PATIENT_ID"] == raw_pid].iterrows():
            event_type = (st.get("EVENT_TYPE") or "").upper()
            event_date = _offset_to_date(dx_date, st.get("START_DATE")) or last_fu
            if event_type == "PROGRESSION":
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
            elif event_type == "STATUS" and _map_vital(st.get("STATUS")) == VitalStatus.deceased:
                outcome_records.append(
                    {
                        "outcome_id": new_uuid(),
                        "patient_id": pid,
                        "event_type": OutcomeEventType.death.value,
                        "event_date": event_date,
                        "recist_response": None,
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
            vaf_val = max(min(vaf_val, 1.0), 0.0)
        except (TypeError, ValueError):
            vaf_val = -1.0
        try:
            depth = int(float(mrow.get("t_depth") or -1))
        except (TypeError, ValueError):
            depth = -1
        protein = mrow.get("HGVSp_Short")
        if not isinstance(protein, str) or not protein.strip():
            protein = None

        is_baseline_driver = gene == "EGFR"
        from .common import C797S_RE  # local to avoid top-level cycle noise

        is_resistance = (
            gene == "EGFR" and isinstance(protein, str) and bool(C797S_RE.search(protein))
        )
        variant_records.append(
            {
                "variant_id": new_uuid(),
                "patient_id": make_patient_id(SOURCE_PREFIX, raw_pid),
                "sample_id": sid,
                "sample_type": SampleType.tumor_tissue.value,
                "sample_date": sample_to_dx_date.get(sid, DEFAULT_DIAGNOSIS_DATE),
                "assay": _map_assay(sample_to_assay.get(sid)).value,
                "gene_symbol": gene or "UNKNOWN",
                "alteration_type": _map_alteration_type(mrow.get("Variant_Classification")).value,
                "protein_change_hgvs": protein,
                "vaf": vaf_val,
                "read_depth": depth,
                "oncokb_oncogenic": OncoKBOncogenic.unknown.value,
                "is_germline": False,
                "is_baseline_driver": is_baseline_driver,
                "is_resistance_call": bool(is_resistance),
                "resistance_mechanism_class": None,
            }
        )

    return ParsedTables(
        patients=pd.DataFrame(patient_records),
        variants=pd.DataFrame(variant_records),
        treatments=pd.DataFrame(treatment_records),
        outcomes=pd.DataFrame(outcome_records),
    )


def _genie_age_at_diagnosis(sample_rows: pd.DataFrame) -> int:
    """GENIE stores AGE_AT_SEQ_REPORT in days. Approximate diagnosis age in years."""
    if sample_rows.empty:
        return -1
    raw = sample_rows.iloc[0].get("AGE_AT_SEQ_REPORT")
    try:
        days = float(raw)
    except (TypeError, ValueError):
        return -1
    years = int(days / 365.25)
    return max(min(years, 110), 18) if years > 0 else -1
