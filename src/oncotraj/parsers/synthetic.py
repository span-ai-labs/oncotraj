"""Realistic-but-fake EGFR+ NSCLC fixtures in GENIE and cBioPortal on-disk formats.

Used by tests and by `scripts/build_dataset.py --use-synthetic` so the pipeline
is exercised end-to-end without requiring real Synapse / cBioPortal access.

Outputs match the exact file names and column conventions the real parsers
consume; the parsers do not branch on synthetic-vs-real.
"""

from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path


def _genie_clinical_header(
    attrs: list[str], descriptions: list[str], types: list[str]
) -> list[str]:
    """GENIE/cBioPortal clinical files start with 4 '#'-prefixed metadata header rows."""
    return [
        "#" + "\t".join(descriptions),
        "#" + "\t".join(descriptions),
        "#" + "\t".join(types),
        "#" + "\t".join(["1"] * len(attrs)),
        "\t".join(attrs),
    ]


def write_synthetic_genie(out_dir: Path, n_patients: int = 20, seed: int = 0) -> Path:
    """Emit GENIE-BPC-shaped clinical, sample, mutation, and timeline TSVs."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    egfr_variants = [
        ("p.Glu746_Ala750del", "exon19del"),
        ("p.Leu858Arg", "L858R"),
        ("p.Gly719Cys", "G719X"),
    ]
    sex_choices = ["Male", "Female"]
    smoking_choices = ["Never", "Former", "Current"]
    stages = ["IIIA", "IIIB", "IVA", "IVB"]
    sites = ["msk", "dfci", "vicc", "ucsf"]

    patient_rows: list[str] = []
    sample_rows: list[str] = []
    mutation_rows: list[str] = []
    treatment_rows: list[str] = []
    timeline_rows: list[str] = []

    for i in range(n_patients):
        pid = f"GENIE-MSK-P-{i:07d}"
        dx = date(2022, 1, 1) + timedelta(days=rng.randint(0, 365))
        age = rng.randint(45, 82)
        sex = rng.choice(sex_choices)
        smoke = rng.choice(smoking_choices)
        rng.choice(stages)
        site = rng.choice(sites)
        vital = rng.choice(["LIVING", "DECEASED"])
        os_months = rng.uniform(6, 36)
        last_followup_offset = int(os_months * 30)

        patient_rows.append(
            "\t".join(
                [
                    pid,
                    sex,
                    "White",
                    "Non-Spanish; non-Hispanic",
                    smoke,
                    vital,
                    f"{os_months:.1f}",
                    site,
                ]
            )
        )

        sid = f"{pid}-T01"
        sample_rows.append(
            "\t".join(
                [
                    pid,
                    sid,
                    "Non-Small Cell Lung Cancer",
                    "Lung Adenocarcinoma",
                    "LUAD",
                    "Primary",
                    str(age * 365),
                    "MSK-IMPACT468",
                    "Tumor",
                    dx.isoformat(),
                ]
            )
        )

        hgvsp, _ = rng.choice(egfr_variants)
        mutation_rows.append(
            "\t".join(
                [
                    "EGFR",
                    "Missense_Mutation" if "del" not in hgvsp else "In_Frame_Del",
                    sid,
                    hgvsp,
                    "c.2573T>G",
                    "7",
                    str(rng.randint(50, 200)),
                    str(rng.randint(10, 80)),
                    str(rng.uniform(0.1, 0.6)),
                ]
            )
        )
        if rng.random() < 0.4:
            mutation_rows.append(
                "\t".join(
                    [
                        "TP53",
                        "Missense_Mutation",
                        sid,
                        "p.Arg175His",
                        "c.524G>A",
                        "5",
                        str(rng.randint(50, 200)),
                        str(rng.randint(10, 80)),
                        str(rng.uniform(0.1, 0.6)),
                    ]
                )
            )

        treatment_rows.append("\t".join([pid, "0", "30", "Osimertinib", "1"]))
        timeline_rows.append("\t".join([pid, "0", str(last_followup_offset), "STATUS", vital]))
        if rng.random() < 0.7:
            prog_high = max(last_followup_offset - 30, 181)
            prog = rng.randint(180, prog_high)
            timeline_rows.append("\t".join([pid, str(prog), str(prog), "PROGRESSION", "PD"]))

    pat_attrs = [
        "PATIENT_ID",
        "SEX",
        "PRIMARY_RACE",
        "ETHNICITY",
        "SMOKING_HISTORY",
        "OS_STATUS",
        "OS_MONTHS",
        "CENTER",
    ]
    pat_types = ["STRING"] * 6 + ["NUMBER", "STRING"]
    pat_desc = pat_attrs
    pat_file = out_dir / "data_clinical_patient.txt"
    pat_file.write_text(
        "\n".join(_genie_clinical_header(pat_attrs, pat_desc, pat_types) + patient_rows) + "\n"
    )

    samp_attrs = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
        "ONCOTREE_CODE",
        "SAMPLE_TYPE",
        "AGE_AT_SEQ_REPORT",
        "SEQ_ASSAY_ID",
        "SAMPLE_CLASS",
        "DX_DATE",
    ]
    samp_types = ["STRING"] * 6 + ["NUMBER", "STRING", "STRING", "STRING"]
    samp_file = out_dir / "data_clinical_sample.txt"
    samp_file.write_text(
        "\n".join(_genie_clinical_header(samp_attrs, samp_attrs, samp_types) + sample_rows) + "\n"
    )

    maf_header = [
        "Hugo_Symbol",
        "Variant_Classification",
        "Tumor_Sample_Barcode",
        "HGVSp_Short",
        "HGVSc",
        "Chromosome",
        "t_depth",
        "t_alt_count",
        "vaf",
    ]
    maf_file = out_dir / "data_mutations_extended.txt"
    maf_file.write_text("\t".join(maf_header) + "\n" + "\n".join(mutation_rows) + "\n")

    tx_header = ["PATIENT_ID", "START_DATE", "STOP_DATE", "AGENT", "LINE"]
    tx_file = out_dir / "data_timeline_treatment.txt"
    tx_file.write_text("\t".join(tx_header) + "\n" + "\n".join(treatment_rows) + "\n")

    tl_header = ["PATIENT_ID", "START_DATE", "STOP_DATE", "EVENT_TYPE", "STATUS"]
    tl_file = out_dir / "data_timeline_status.txt"
    tl_file.write_text("\t".join(tl_header) + "\n" + "\n".join(timeline_rows) + "\n")

    return out_dir


def write_synthetic_cbioportal(out_dir: Path, n_patients: int = 30, seed: int = 1) -> Path:
    """Emit MSK-CHORD-shaped cBioPortal study files."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    egfr_variants = [
        ("p.Glu746_Ala750del", "In_Frame_Del"),
        ("p.Leu858Arg", "Missense_Mutation"),
        ("p.Ser768Ile", "Missense_Mutation"),
    ]

    patient_rows: list[str] = []
    sample_rows: list[str] = []
    mutation_rows: list[str] = []
    timeline_tx_rows: list[str] = []
    timeline_status_rows: list[str] = []

    for i in range(n_patients):
        pid = f"CHORD-{i:05d}"
        dx = date(2021, 6, 1) + timedelta(days=rng.randint(0, 600))
        age = rng.randint(40, 85)
        sex = rng.choice(["Male", "Female"])
        smoke = rng.choice(["Never", "Former", "Current"])
        vital = rng.choice(["0:LIVING", "1:DECEASED"])
        os_months = rng.uniform(8, 40)
        sid = f"{pid}-T01"

        patient_rows.append("\t".join([pid, sex, smoke, vital, f"{os_months:.2f}", "msk"]))

        sample_rows.append(
            "\t".join(
                [
                    pid,
                    sid,
                    "Non-Small Cell Lung Cancer",
                    "Lung Adenocarcinoma",
                    "LUAD",
                    "Primary",
                    str(age),
                    "IMPACT468",
                    dx.isoformat(),
                ]
            )
        )

        hgvsp, vclass = rng.choice(egfr_variants)
        mutation_rows.append(
            "\t".join(
                [
                    "EGFR",
                    vclass,
                    sid,
                    pid,
                    hgvsp,
                    "c.X",
                    "7",
                    str(rng.randint(80, 300)),
                    str(rng.randint(20, 120)),
                    f"{rng.uniform(0.05, 0.55):.3f}",
                ]
            )
        )
        if rng.random() < 0.3:
            mutation_rows.append(
                "\t".join(["MET", "Amplification", sid, pid, "", "", "7", "0", "0", "-1.0"])
            )

        tx_start = rng.randint(0, 60)
        tx_stop = tx_start + rng.randint(180, 720)
        timeline_tx_rows.append(
            "\t".join([pid, str(tx_start), str(tx_stop), "Treatment", "Osimertinib", "1"])
        )

        last_fu = tx_stop + rng.randint(0, 90)
        timeline_status_rows.append(
            "\t".join([pid, str(last_fu), str(last_fu), "Status", vital.split(":")[-1]])
        )
        if rng.random() < 0.6:
            prog = tx_start + rng.randint(150, 600)
            timeline_status_rows.append("\t".join([pid, str(prog), str(prog), "Progression", "PD"]))

    pat_attrs = ["PATIENT_ID", "SEX", "SMOKING_HISTORY", "OS_STATUS", "OS_MONTHS", "CENTER"]
    pat_types = ["STRING"] * 4 + ["NUMBER", "STRING"]
    pat_file = out_dir / "data_clinical_patient.txt"
    pat_file.write_text(
        "\n".join(_genie_clinical_header(pat_attrs, pat_attrs, pat_types) + patient_rows) + "\n"
    )

    samp_attrs = [
        "PATIENT_ID",
        "SAMPLE_ID",
        "CANCER_TYPE",
        "CANCER_TYPE_DETAILED",
        "ONCOTREE_CODE",
        "SAMPLE_TYPE",
        "AGE_AT_SEQ_REPORT",
        "SEQ_ASSAY_ID",
        "DX_DATE",
    ]
    samp_types = ["STRING"] * 6 + ["NUMBER", "STRING", "STRING"]
    samp_file = out_dir / "data_clinical_sample.txt"
    samp_file.write_text(
        "\n".join(_genie_clinical_header(samp_attrs, samp_attrs, samp_types) + sample_rows) + "\n"
    )

    maf_header = [
        "Hugo_Symbol",
        "Variant_Classification",
        "Tumor_Sample_Barcode",
        "Patient_ID",
        "HGVSp_Short",
        "HGVSc",
        "Chromosome",
        "t_depth",
        "t_alt_count",
        "vaf",
    ]
    (out_dir / "data_mutations.txt").write_text(
        "\t".join(maf_header) + "\n" + "\n".join(mutation_rows) + "\n"
    )

    tx_header = ["PATIENT_ID", "START_DATE", "STOP_DATE", "EVENT_TYPE", "AGENT", "LINE"]
    (out_dir / "data_timeline_treatment.txt").write_text(
        "\t".join(tx_header) + "\n" + "\n".join(timeline_tx_rows) + "\n"
    )

    status_header = ["PATIENT_ID", "START_DATE", "STOP_DATE", "EVENT_TYPE", "STATUS"]
    (out_dir / "data_timeline_status.txt").write_text(
        "\t".join(status_header) + "\n" + "\n".join(timeline_status_rows) + "\n"
    )

    return out_dir
