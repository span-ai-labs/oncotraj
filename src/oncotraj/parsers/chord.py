"""MSK-CHORD (Jee et al. 2024) real-data parser.

Consumes the cBioPortal-distributed MSK-CHORD study tarball (`msk_chord_2024`)
and emits the four-table OncoTraj contract. Distinct from `cbioportal.py`,
which targets a simpler synthetic-fixture layout used in tests; the real
MSK-CHORD distribution has:

- ~25 000 patients pan-cancer; we filter to `CANCER_TYPE_DETAILED in
  {Lung Adenocarcinoma, Non-Small Cell Lung Cancer}` with osimertinib as
  the patient's first EGFR-TKI (1L-osimertinib cohort definition).
- Anonymised integer-offset dates (`START_DATE`, `STOP_DATE` in days
  relative to a per-patient reference). The parser anchors every patient
  at `OSI_EPOCH = 2020-01-01` on their osi_start and shifts all other
  events by the relative offset, so the emitted parquet has plausible
  ISO 8601 dates with correct per-patient relative timing.
- `data_timeline_progression.txt`: NLP-derived progression events from
  radiology-report impressions. We use `PROGRESSION='Y'` rows as the
  progression event; `Indeterminate` rows are dropped (would otherwise
  introduce label noise — Jee et al. discuss this threshold).
- `OS_MONTHS` / `OS_STATUS` on patient: real overall survival.

Cohort filter applied here:
1. CANCER_TYPE_DETAILED is Lung Adenocarcinoma or NSCLC.
2. Patient has ≥1 osimertinib treatment row.
3. The patient's FIRST EGFR-TKI by START_DATE offset is osimertinib
   (i.e., no prior erlotinib/afatinib/gefitinib → 1L osi).
4. Patient has ≥1 EGFR variant that classifies as a sensitizing class.

Patients meeting 1+2 but failing 3 are emitted with
`exclusion_reason="prior_EGFR_TKI"` for the held-out sensitivity slice.
"""

from __future__ import annotations

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
    ResistanceMechanism,
    SampleType,
    Sex,
    SmokingStatus,
    SourceDataset,
    StageAtDiagnosis,
    VitalStatus,
)

from .common import C797S_RE, ParsedTables, classify_egfr_variant, make_patient_id, new_uuid

SOURCE_PREFIX = "MSK_CHORD"
LUNG_DETAILED = {"Lung Adenocarcinoma", "Non-Small Cell Lung Cancer"}
EGFR_TKIS = {
    "OSIMERTINIB", "ERLOTINIB", "GEFITINIB", "AFATINIB",
    "DACOMITINIB", "ICOTINIB", "LAZERTINIB", "ROCILETINIB",
}
# All days-from-anchor offsets become real dates by shifting against this
# epoch. The 2020-01-01 choice is documented (not the patient's real osi
# start); per-patient *relative* timing is what matters and is preserved.
OSI_EPOCH = date(2020, 1, 1)

_PANEL_TO_ASSAY = {
    "IMPACT341": Assay.MSK_IMPACT_341,
    "IMPACT410": Assay.MSK_IMPACT_410,
    "IMPACT468": Assay.MSK_IMPACT_468,
    "IMPACT505": Assay.MSK_IMPACT_468,  # closest match; IMPACT505 isn't in spec enum yet.
    "MSK-ACCESS": Assay.MSK_ACCESS,
}


def _read_clinical(path: Path) -> pd.DataFrame:
    """MSK-CHORD clinical TSVs have 4 metadata rows (starting `#`) before
    the real header line."""
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)


def _read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, low_memory=False)


def _to_int_days(v: object) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _offset_to_date(anchor_offset_days: int | None, event_offset_days: int | None) -> date | None:
    if anchor_offset_days is None or event_offset_days is None:
        return None
    return OSI_EPOCH + timedelta(days=event_offset_days - anchor_offset_days)


def _map_sex(v: object) -> Sex:
    if not isinstance(v, str):
        return Sex.unknown
    s = v.strip().lower()
    if s.startswith("m"):
        return Sex.male
    if s.startswith("f"):
        return Sex.female
    return Sex.unknown


def _map_smoking(v: object) -> SmokingStatus:
    if not isinstance(v, str):
        return SmokingStatus.unknown
    s = v.strip().lower()
    if "never" in s:
        return SmokingStatus.never
    if "current" in s or "active" in s:
        return SmokingStatus.current
    if "former" in s or "ever" in s:
        return SmokingStatus.former
    return SmokingStatus.unknown


def _map_assay(panel: object) -> Assay:
    if not isinstance(panel, str):
        return Assay.unknown
    return _PANEL_TO_ASSAY.get(panel.strip().upper(), Assay.unknown)


def _map_alteration(vclass: object) -> AlterationType:
    if not isinstance(vclass, str):
        return AlterationType.SNV
    s = vclass.lower()
    if "missense" in s or "nonsense" in s or "silent" in s:
        return AlterationType.SNV
    if "splice" in s:
        return AlterationType.splice
    if "in_frame_ins" in s or "frame_shift_ins" in s:
        return AlterationType.indel_insertion
    if "in_frame_del" in s or "frame_shift_del" in s:
        return AlterationType.indel_deletion
    return AlterationType.SNV


def _map_vital(v: object) -> VitalStatus:
    if not isinstance(v, str):
        return VitalStatus.unknown
    s = v.lower()
    if "deceased" in s or "dead" in s or s.startswith("1"):
        return VitalStatus.deceased
    if "living" in s or "alive" in s or s.startswith("0"):
        return VitalStatus.alive
    return VitalStatus.unknown


def parse(study_dir: Path) -> ParsedTables:
    """Parse the real MSK-CHORD distribution into the OncoTraj four-table form."""
    base = Path(study_dir)
    # Tarball extracts to `msk_chord_2024/` — handle either layout.
    inner = base / "msk_chord_2024"
    if inner.exists():
        base = inner

    patients_raw = _read_clinical(base / "data_clinical_patient.txt")
    samples_raw = _read_clinical(base / "data_clinical_sample.txt")
    tx_raw = _read_tsv(base / "data_timeline_treatment.txt")
    prog_raw = _read_tsv(base / "data_timeline_progression.txt")
    muts_raw = _read_tsv(base / "data_mutations.txt")

    # Sample <-> patient lookup + filter to lung adeno / NSCLC.
    samples_raw = samples_raw.copy()
    lung_samples = samples_raw[samples_raw["CANCER_TYPE_DETAILED"].isin(LUNG_DETAILED)].copy()
    lung_patient_ids = set(lung_samples["PATIENT_ID"].unique())
    sample_to_pid = dict(zip(lung_samples["SAMPLE_ID"], lung_samples["PATIENT_ID"], strict=False))

    # Treatment timeline restricted to lung-adeno patients + EGFR TKIs.
    tx_raw["START_INT"] = tx_raw["START_DATE"].map(_to_int_days)
    tx_raw["STOP_INT"] = tx_raw["STOP_DATE"].map(_to_int_days)
    tx_lung = tx_raw[tx_raw["PATIENT_ID"].isin(lung_patient_ids)]
    tkis = tx_lung[tx_lung["AGENT"].isin(EGFR_TKIS)].copy()
    tkis = tkis.sort_values(["PATIENT_ID", "START_INT"])

    # First-TKI per patient determines 1L-osi status.
    first_tki = tkis.groupby("PATIENT_ID").head(1).set_index("PATIENT_ID")
    osi_only = tkis[tkis["AGENT"] == "OSIMERTINIB"]
    osi_first = osi_only.groupby("PATIENT_ID").head(1).set_index("PATIENT_ID")
    osi_pids = set(osi_first.index)

    # Variants per patient.
    muts = muts_raw[
        ["Hugo_Symbol", "Tumor_Sample_Barcode", "HGVSp_Short", "Variant_Classification", "Center"]
    ].copy()
    muts["PATIENT_ID"] = muts["Tumor_Sample_Barcode"].map(sample_to_pid)
    muts = muts[muts["PATIENT_ID"].notna()]
    egfr_hgvsp = (
        muts.loc[muts["Hugo_Symbol"] == "EGFR"]
        .groupby("PATIENT_ID")["HGVSp_Short"]
        .apply(lambda s: [x for x in s.tolist() if isinstance(x, str)])
        .to_dict()
    )

    # Progression timeline (only Y rows; drop NLP indeterminate).
    prog_raw["START_INT"] = prog_raw["START_DATE"].map(_to_int_days)
    prog_lung = prog_raw[
        prog_raw["PATIENT_ID"].isin(lung_patient_ids)
        & (prog_raw.get("PROGRESSION", "").astype(str) == "Y")
    ].copy()

    patient_records: list[dict] = []
    treatment_records: list[dict] = []
    outcome_records: list[dict] = []
    variant_records: list[dict] = []

    pat_lookup = patients_raw.set_index("PATIENT_ID")

    for pid in sorted(osi_pids):
        # Anchor at patient's first osimertinib start; everything else shifts.
        osi_start_offset = int(osi_first.loc[pid, "START_INT"])
        osi_stop_offset = osi_first.loc[pid, "STOP_INT"]
        osi_stop_offset = int(osi_stop_offset) if pd.notna(osi_stop_offset) else None
        first_tki_agent = first_tki.loc[pid, "AGENT"] if pid in first_tki.index else None

        is_first_line = first_tki_agent == "OSIMERTINIB"

        # EGFR class from this patient's variants (tissue-NGS; MSK-CHORD has
        # no serial ctDNA for these patients in the public release).
        hgvsp = [h for h in egfr_hgvsp.get(pid, []) if isinstance(h, str)]
        # Normalise to `p.` prefix for the regex classifier.
        hgvsp_norm = [h if h.startswith("p.") else f"p.{h}" for h in hgvsp]
        variant_class = classify_egfr_variant(hgvsp_norm)

        included = is_first_line and variant_class != EgfrVariantClass.unknown
        if not included:
            if variant_class == EgfrVariantClass.unknown:
                exclusion = "not_EGFR_sensitizing"
            elif not is_first_line:
                exclusion = "prior_EGFR_TKI"
            else:
                exclusion = "missing_outcome"
        else:
            exclusion = None

        # Dates: osi_start -> OSI_EPOCH; diagnosis_date = osi_start (we lack a
        # reliable diagnosis offset for every patient and don't want to fabricate
        # one); last_followup_date = osi_stop OR OS_MONTHS-derived OR osi_start.
        diag_date = OSI_EPOCH
        osi_start_date = OSI_EPOCH
        osi_stop_date = _offset_to_date(osi_start_offset, osi_stop_offset)

        os_months = pat_lookup.loc[pid].get("OS_MONTHS") if pid in pat_lookup.index else None
        try:
            os_days = int(float(os_months) * 30) if pd.notna(os_months) else None
        except (TypeError, ValueError):
            os_days = None
        os_date = OSI_EPOCH + timedelta(days=os_days) if os_days is not None else None

        last_fu = max(d for d in [osi_stop_date, os_date, OSI_EPOCH] if d is not None)

        # Patient row
        gender = pat_lookup.loc[pid].get("GENDER") if pid in pat_lookup.index else None
        smoke = (
            pat_lookup.loc[pid].get("SMOKING_PREDICTIONS_3_CLASSES")
            if pid in pat_lookup.index
            else None
        )
        vital = pat_lookup.loc[pid].get("OS_STATUS") if pid in pat_lookup.index else None
        stage = pat_lookup.loc[pid].get("STAGE_HIGHEST_RECORDED") if pid in pat_lookup.index else None
        age = pat_lookup.loc[pid].get("CURRENT_AGE_DEID") if pid in pat_lookup.index else None
        try:
            age_int = int(float(age)) if pd.notna(age) else -1
            age_int = age_int if 18 <= age_int <= 110 else -1
        except (TypeError, ValueError):
            age_int = -1

        patient_records.append(
            {
                "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                "source_dataset": SourceDataset.MSK_CHORD.value,
                "age_at_diagnosis_years": age_int,
                "sex": _map_sex(gender).value,
                "smoking_status": _map_smoking(smoke).value,
                "diagnosis_date": diag_date,
                "stage_at_diagnosis": _map_stage(stage).value,
                "histology": Histology.adenocarcinoma.value,
                "egfr_variant_class": variant_class.value,
                "site_id": "msk",
                "vital_status_at_last_followup": _map_vital(vital).value,
                "last_followup_date": last_fu,
                "included_in_v1_cohort": included,
                "exclusion_reason": exclusion,
            }
        )

        treatment_records.append(
            {
                "treatment_id": new_uuid(),
                "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                "drug_name": "osimertinib",
                "line_of_therapy": (
                    LineOfTherapy.first_line.value if is_first_line else LineOfTherapy.unknown.value
                ),
                "start_date": osi_start_date,
                "end_date": osi_stop_date,
                "is_osimertinib": True,
            }
        )

        # Outcomes: always emit last_followup; add progression rows from real
        # NLP-derived events that occurred AFTER osi start.
        outcome_records.append(
            {
                "outcome_id": new_uuid(),
                "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                "event_type": OutcomeEventType.last_followup.value,
                "event_date": last_fu,
                "recist_response": None,
                "resistance_mechanism_class": None,
            }
        )
        pt_progs = prog_lung[prog_lung["PATIENT_ID"] == pid]
        first_post_osi_prog_offset = pt_progs.loc[
            pt_progs["START_INT"].astype("Int64") >= osi_start_offset, "START_INT"
        ].min() if not pt_progs.empty else None
        if pd.notna(first_post_osi_prog_offset):
            event_date = _offset_to_date(osi_start_offset, int(first_post_osi_prog_offset))
            outcome_records.append(
                {
                    "outcome_id": new_uuid(),
                    "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                    "event_type": OutcomeEventType.progression_recist.value,
                    "event_date": event_date,
                    "recist_response": RecistResponse.progressive_disease.value,
                    "resistance_mechanism_class": _infer_mechanism_class(
                        muts.loc[muts["PATIENT_ID"] == pid]
                    ),
                }
            )

        # Variants — emit all EGFR + MET + ERBB2 + TP53 calls.
        pt_muts = muts.loc[muts["PATIENT_ID"] == pid]
        for _, vr in pt_muts.iterrows():
            sid = vr["Tumor_Sample_Barcode"]
            assay = _map_assay(
                lung_samples.loc[lung_samples["SAMPLE_ID"] == sid, "GENE_PANEL"].iloc[0]
                if (lung_samples["SAMPLE_ID"] == sid).any()
                else None
            )
            protein = vr.get("HGVSp_Short")
            if not isinstance(protein, str) or not protein.strip():
                protein = None
            elif not protein.startswith("p."):
                protein = f"p.{protein}"
            gene = (vr.get("Hugo_Symbol") or "").upper()
            is_c797s = (
                gene == "EGFR"
                and isinstance(protein, str)
                and bool(C797S_RE.search(protein))
            )
            variant_records.append(
                {
                    "variant_id": new_uuid(),
                    "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                    "sample_id": sid,
                    "sample_type": SampleType.tumor_tissue.value,
                    "sample_date": diag_date,  # MSK-CHORD has no per-sample dates in public release.
                    "assay": assay.value,
                    "gene_symbol": gene or "UNKNOWN",
                    "alteration_type": _map_alteration(vr.get("Variant_Classification")).value,
                    "protein_change_hgvs": protein,
                    "vaf": -1.0,
                    "read_depth": -1,
                    "oncokb_oncogenic": OncoKBOncogenic.unknown.value,
                    "is_germline": False,
                    "is_baseline_driver": gene == "EGFR" and not is_c797s,
                    "is_resistance_call": bool(is_c797s),
                    "resistance_mechanism_class": (
                        ResistanceMechanism.EGFR_C797S.value if is_c797s else None
                    ),
                }
            )

    return ParsedTables(
        patients=pd.DataFrame(patient_records),
        variants=pd.DataFrame(variant_records),
        treatments=pd.DataFrame(treatment_records),
        outcomes=pd.DataFrame(outcome_records),
    )


def _infer_mechanism_class(pt_muts: pd.DataFrame) -> str | None:
    """Coarse: C797S beats MET amp beats other. MSK-CHORD doesn't publish
    explicit resistance-call CNAs, so we only call C797S confidently; MET
    amp inference would require the CNA file (`data_cna.txt`) which is left
    as a follow-up."""
    egfr_snvs = pt_muts.loc[(pt_muts["Hugo_Symbol"] == "EGFR")]
    for protein in egfr_snvs["HGVSp_Short"].dropna():
        s = protein if isinstance(protein, str) and protein.startswith("p.") else f"p.{protein}"
        if C797S_RE.search(s):
            return ResistanceMechanism.EGFR_C797S.value
    return ResistanceMechanism.other_or_unknown.value


def _map_stage(v: object) -> StageAtDiagnosis:
    if not isinstance(v, str):
        return StageAtDiagnosis.unknown
    s = v.strip().lower()
    if "iv" in s or "stage 4" in s:
        return StageAtDiagnosis.IV_NOS
    if "iii" in s or "stage 3" in s:
        return StageAtDiagnosis.IIIA
    if "ii" in s or "stage 2" in s:
        return StageAtDiagnosis.II
    if "i" in s or "stage 1" in s:
        return StageAtDiagnosis.I
    return StageAtDiagnosis.unknown
