"""AACR Project GENIE BPC NSCLC v2.0-public real-data parser.

Distinct from `genie.py` (the synthetic-fixture / cBioPortal-style GENIE
parser used in tests). GENIE BPC is the *Biopharma Collaborative* curated
clinical-genomic dataset: a regimen-level analytic table with
abstractor-determined PFS/OS endpoints, plus the standard cBioPortal-style
clinical/sample/mutation files.

Cohort definition (matches chord.py for cross-source consistency):
- Patient's FIRST EGFR-TKI regimen (by `regimen_number_within_cancer`) is
  osimertinib → 1L-osimertinib. Patients whose first TKI is
  erlotinib/gefitinib/afatinib/dacomitinib are emitted with
  `exclusion_reason="prior_EGFR_TKI"` (held-out slice).
- Patient has ≥1 EGFR variant classifying as a sensitizing class.

GENIE BPC's enrollment window predates widespread 1L-osi adoption, so the
1L-osi yield is modest (~36 patients) versus its larger 2L-osi population.
We ingest it for source diversity and because its curated PFS endpoints are
higher-quality than NLP-derived progression.

Key BPC specifics:
- `record_id` == cBioPortal `PATIENT_ID` (e.g. GENIE-DFCI-000800).
- Anonymised integer day-offsets relative to diagnosis (`dx_reg_start_int`
  = days from diagnosis to regimen start). We anchor each patient's osi
  regimen start at OSI_EPOCH=2020-01-01.
- Curated PFS: `pfs_i_and_m_g_status` (1=event, 0=censored) and
  `tt_pfs_i_and_m_g_days` (days from regimen start), determined from
  imaging AND med-onc notes by trained abstractors.
- Curated OS: `os_g_status`, `tt_os_g_days`.
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

SOURCE_PREFIX = "GENIE_BPC"
OSI_EPOCH = date(2020, 1, 1)
EGFR_TKIS = [
    "Osimertinib",
    "Erlotinib",
    "Gefitinib",
    "Afatinib",
    "Dacomitinib",
    "Icotinib",
    "Lazertinib",
    "Rociletinib",
]


def _read_clinical(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", dtype=str, low_memory=False)


def _to_int(v: object) -> int | None:
    try:
        return int(float(v)) if v is not None and str(v).strip() not in ("", "nan") else None
    except (TypeError, ValueError):
        return None


def _first_egfr_tki(group: pd.DataFrame) -> str | None:
    g = group.sort_values("regnum")
    for _, r in g.iterrows():
        drugs = str(r["regimen_drugs"])
        for tki in EGFR_TKIS:
            if tki.lower() in drugs.lower():
                return tki
    return None


def _map_sex(v: object) -> Sex:
    if not isinstance(v, str):
        return Sex.unknown
    s = v.strip().lower()
    if s.startswith("m") or s == "1":
        return Sex.male
    if s.startswith("f") or s == "2":
        return Sex.female
    return Sex.unknown


def _map_stage(v: object) -> StageAtDiagnosis:
    if not isinstance(v, str):
        return StageAtDiagnosis.unknown
    s = v.strip().lower().replace("stage ", "")
    mapping = {
        "iv": StageAtDiagnosis.IV_NOS,
        "4": StageAtDiagnosis.IV_NOS,
        "iva": StageAtDiagnosis.IVA,
        "ivb": StageAtDiagnosis.IVB,
        "iiia": StageAtDiagnosis.IIIA,
        "iiib": StageAtDiagnosis.IIIB,
        "iiic": StageAtDiagnosis.IIIC,
        "iii": StageAtDiagnosis.IIIA,
        "3a": StageAtDiagnosis.IIIA,
        "3b": StageAtDiagnosis.IIIB,
        "ii": StageAtDiagnosis.II,
        "2": StageAtDiagnosis.II,
        "i": StageAtDiagnosis.I,
        "1": StageAtDiagnosis.I,
    }
    return mapping.get(s, StageAtDiagnosis.unknown)


def _map_alteration(vclass: object) -> AlterationType:
    if not isinstance(vclass, str):
        return AlterationType.SNV
    s = vclass.lower()
    if "splice" in s:
        return AlterationType.splice
    if "ins" in s:
        return AlterationType.indel_insertion
    if "del" in s:
        return AlterationType.indel_deletion
    return AlterationType.SNV


def parse(study_dir: Path) -> ParsedTables:
    base = Path(study_dir)
    patients_raw = _read_clinical(base / "data_clinical_patient.txt")
    samples_raw = _read_clinical(base / "data_clinical_sample.txt")
    muts_raw = pd.read_csv(
        base / "data_mutations_extended.txt",
        sep="\t",
        dtype=str,
        low_memory=False,
        usecols=["Hugo_Symbol", "Tumor_Sample_Barcode", "HGVSp_Short", "Variant_Classification"],
    )
    reg = pd.read_csv(base / "regimen_cancer_level_dataset.csv", dtype=str, low_memory=False)
    pat_level = pd.read_csv(base / "patient_level_dataset.csv", dtype=str, low_memory=False)

    reg["regnum"] = pd.to_numeric(reg["regimen_number_within_cancer"], errors="coerce")
    pat_lvl = pat_level.set_index("record_id")

    # First EGFR-TKI per patient → 1L-osi membership.
    first_tki = reg.groupby("record_id").apply(_first_egfr_tki, include_groups=False).dropna()
    osi_first_pids = set(first_tki[first_tki == "Osimertinib"].index)
    prior_tki_pids = set(first_tki[first_tki != "Osimertinib"].index)

    # First osimertinib regimen row per patient (for PFS/OS endpoints + start offset).
    osi_regs = reg[reg["regimen_drugs"].str.contains("Osimertinib", case=False, na=False)].copy()
    osi_first_reg = (
        osi_regs.sort_values("regnum").groupby("record_id").head(1).set_index("record_id")
    )

    # EGFR variants per patient.
    s2p = dict(zip(samples_raw["SAMPLE_ID"], samples_raw["PATIENT_ID"], strict=False))
    muts_raw["PATIENT_ID"] = muts_raw["Tumor_Sample_Barcode"].map(s2p)
    muts_raw = muts_raw[muts_raw["PATIENT_ID"].notna()]
    egfr_hgvsp = (
        muts_raw.loc[muts_raw["Hugo_Symbol"] == "EGFR"]
        .groupby("PATIENT_ID")["HGVSp_Short"]
        .apply(lambda s: [x for x in s.tolist() if isinstance(x, str)])
        .to_dict()
    )

    pat_clin = patients_raw.set_index("PATIENT_ID")
    sample_panel = dict(
        zip(samples_raw["SAMPLE_ID"], samples_raw.get("SEQ_ASSAY_ID", pd.Series()), strict=False)
    )

    # Candidate patients: anyone with an osimertinib regimen at all.
    candidate_pids = set(osi_first_reg.index)

    patient_records: list[dict] = []
    treatment_records: list[dict] = []
    outcome_records: list[dict] = []
    variant_records: list[dict] = []

    for pid in sorted(candidate_pids):
        is_first_line = pid in osi_first_pids
        hgvsp = [h for h in egfr_hgvsp.get(pid, []) if isinstance(h, str)]
        hgvsp_norm = [h if h.startswith("p.") else f"p.{h}" for h in hgvsp]
        variant_class = classify_egfr_variant(hgvsp_norm)

        included = is_first_line and variant_class != EgfrVariantClass.unknown
        if not included:
            exclusion = (
                "not_EGFR_sensitizing"
                if variant_class == EgfrVariantClass.unknown
                else "prior_EGFR_TKI"
                if pid in prior_tki_pids
                else "not_first_line_osimertinib"
            )
        else:
            exclusion = None

        reg_row = osi_first_reg.loc[pid]
        # PFS endpoint (imaging + medonc note, curated).
        pfs_days = _to_int(reg_row.get("tt_pfs_i_and_m_g_days"))
        pfs_event = reg_row.get("pfs_i_and_m_g_status") == "1"
        os_days = _to_int(reg_row.get("tt_os_g_days"))

        osi_start_date = OSI_EPOCH
        # last follow-up: osi_start + max(pfs, os) days observed.
        horizon = max([d for d in [pfs_days, os_days, 0] if d is not None])
        last_fu = OSI_EPOCH + timedelta(days=horizon)
        osi_stop_date = (OSI_EPOCH + timedelta(days=pfs_days)) if pfs_days is not None else None

        # Demographics from patient-level analytic file where available.
        sex_code = pat_lvl.loc[pid].get("naaccr_sex_code") if pid in pat_lvl.index else None
        age_fu = pat_lvl.loc[pid].get("age_last_fu_yrs") if pid in pat_lvl.index else None
        try:
            age_int = int(float(age_fu)) if age_fu and str(age_fu) != "nan" else -1
            age_int = age_int if 18 <= age_int <= 110 else -1
        except (TypeError, ValueError):
            age_int = -1
        death_int = pat_lvl.loc[pid].get("hybrid_death_int") if pid in pat_lvl.index else None
        vital = (
            VitalStatus.deceased
            if (death_int and str(death_int) not in ("", "nan"))
            else VitalStatus.alive
        )
        stage = pat_clin.loc[pid].get("STAGE_HIGHEST_RECORDED") if pid in pat_clin.index else None

        patient_records.append(
            {
                "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                "source_dataset": SourceDataset.GENIE_BPC.value,
                "age_at_diagnosis_years": age_int,
                "sex": _map_sex(sex_code).value,
                "smoking_status": SmokingStatus.unknown.value,
                "diagnosis_date": OSI_EPOCH,
                "stage_at_diagnosis": _map_stage(stage).value,
                "histology": Histology.adenocarcinoma.value,
                "egfr_variant_class": variant_class.value,
                "site_id": pid.split("-")[1] if "-" in pid else "unknown",
                "vital_status_at_last_followup": vital.value,
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
        if pfs_event and pfs_days is not None:
            outcome_records.append(
                {
                    "outcome_id": new_uuid(),
                    "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                    "event_type": OutcomeEventType.progression_recist.value,
                    "event_date": OSI_EPOCH + timedelta(days=pfs_days),
                    "recist_response": RecistResponse.progressive_disease.value,
                    "resistance_mechanism_class": _infer_mechanism(
                        muts_raw.loc[muts_raw["PATIENT_ID"] == pid]
                    ),
                }
            )

        for _, vr in muts_raw.loc[muts_raw["PATIENT_ID"] == pid].iterrows():
            protein = vr.get("HGVSp_Short")
            if not isinstance(protein, str) or not protein.strip():
                protein = None
            elif not protein.startswith("p."):
                protein = f"p.{protein}"
            gene = (vr.get("Hugo_Symbol") or "").upper()
            is_c797s = (
                gene == "EGFR" and isinstance(protein, str) and bool(C797S_RE.search(protein))
            )
            panel = sample_panel.get(vr["Tumor_Sample_Barcode"], "")
            assay = Assay.MSK_IMPACT_468 if "IMPACT" in str(panel).upper() else Assay.unknown
            variant_records.append(
                {
                    "variant_id": new_uuid(),
                    "patient_id": make_patient_id(SOURCE_PREFIX, pid),
                    "sample_id": vr["Tumor_Sample_Barcode"],
                    "sample_type": SampleType.tumor_tissue.value,
                    "sample_date": OSI_EPOCH,
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


def _infer_mechanism(pt_muts: pd.DataFrame) -> str | None:
    egfr = pt_muts.loc[pt_muts["Hugo_Symbol"] == "EGFR"]
    for protein in egfr["HGVSp_Short"].dropna():
        s = protein if isinstance(protein, str) and protein.startswith("p.") else f"p.{protein}"
        if C797S_RE.search(s):
            return ResistanceMechanism.EGFR_C797S.value
    return ResistanceMechanism.other_or_unknown.value
