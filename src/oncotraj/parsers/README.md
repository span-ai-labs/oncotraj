# Parsers

One parser module per source dataset. Each emits the four-table contract
(`patients`, `variants`, `treatments`, `outcomes`) defined in
`DATASET_SPEC.md` (kept in `.planning/`, local-only).

## Modules

| Module | Source | Real-data fetch | Tested via |
| --- | --- | --- | --- |
| `genie.py` | AACR Project GENIE BPC NSCLC | Synapse (`SYNAPSE_AUTH_TOKEN`) | synthetic fixtures |
| `cbioportal.py` | cBioPortal studies (default `msk_chord_2024`) | Public datahub HTTPS | synthetic fixtures |
| `synthetic.py` | n/a (writes GENIE/cBioPortal-shaped fakes) | — | drives tests + `--use-synthetic` |
| `common.py` | shared utilities | — | exercised by both parsers |

## Cohort filter (applied identically across parsers)

A patient is set to `included_in_v1_cohort = true` iff:
1. they carry at least one EGFR sensitizing variant (per the `classify_egfr_variant`
   patterns in `common.py`), **and**
2. they have at least one timeline-treatment record whose agent normalises to
   osimertinib (`is_osimertinib`).

Otherwise the row is still emitted with `included_in_v1_cohort = false` and the
matching `exclusion_reason` (`not_EGFR_sensitizing` or `not_first_line_osimertinib`).
This preserves auditability of the cohort selection.

## Field-level transformations

### Patient table

| OncoTraj field | GENIE source | cBioPortal source | Transformation |
| --- | --- | --- | --- |
| `patient_id` | `PATIENT_ID` | `PATIENT_ID` | Prefixed: `GENIE_BPC:<id>`, `MSK_CHORD:<id>`. |
| `age_at_diagnosis_years` | `AGE_AT_SEQ_REPORT` (days) | `AGE_AT_SEQ_REPORT` (years) | GENIE: divide by 365.25 → int, clamp to [18, 110]. cBioPortal: cast → int. |
| `sex` | `SEX` | `SEX` | Lower-case lookup → `male` / `female` / `unknown`. |
| `smoking_status` | `SMOKING_HISTORY` | `SMOKING_HISTORY` | Lower-case lookup → enum; unmapped → `unknown`. |
| `diagnosis_date` | `DX_DATE` on first sample | `DX_DATE` on first sample | ISO 8601; falls back to 2022-01-01 if unparsable (spec §0 anchor convention). |
| `stage_at_diagnosis` | `STAGE` (if present) | not in MSK-CHORD slice | GENIE: AJCC string → enum. cBioPortal: defaults to `IV_NOS` (CHORD osimertinib cohort is advanced disease). |
| `histology` | derived | derived | Defaults to `adenocarcinoma`; transformation cases will be back-filled when biopsy data is added. |
| `egfr_variant_class` | derived from MAF HGVSp | derived from MAF HGVSp | `common.classify_egfr_variant`. Compound calls collapse per derivation rule R1. |
| `site_id` | `CENTER` | `CENTER` (defaults `msk`) | Truncated to 32 chars. |
| `vital_status_at_last_followup` | `OS_STATUS` | `OS_STATUS` | GENIE: `LIVING` / `DECEASED`. cBioPortal: `0:LIVING` / `1:DECEASED` prefix-stripped. |
| `last_followup_date` | `diagnosis_date + OS_MONTHS * 30` | same | OS_MONTHS treated as float; if missing, 180-day fallback. |
| `included_in_v1_cohort` | cohort filter above | same | — |
| `exclusion_reason` | — | — | Set iff `included_in_v1_cohort == false`. |

### Variant table

| OncoTraj field | MAF column | Notes |
| --- | --- | --- |
| `gene_symbol` | `Hugo_Symbol` | Uppercased; HGNC alias resolution is a TODO (spec §0). |
| `alteration_type` | `Variant_Classification` | Mapped via `_VARIANT_CLASSIFICATION_MAP`. |
| `protein_change_hgvs` | `HGVSp_Short` | Pass-through. Three-letter normalisation is a TODO. |
| `vaf` | `vaf` column when present | Clamped to [0, 1]; -1.0 sentinel for CNVs/fusions. |
| `read_depth` | `t_depth` | -1 sentinel if unparsable. |
| `assay` | GENIE `SEQ_ASSAY_ID` → enum; cBioPortal defaults `MSK_IMPACT_468` | `_map_assay`. |
| `is_baseline_driver` | derived | `true` for EGFR variants. |
| `is_resistance_call` | derived | EGFR C797S OR MET amplification (cBioPortal). |
| `resistance_mechanism_class` | derived | `EGFR_C797S` or `MET_amplification`; otherwise `null`. |

### Treatment table

Built from `data_timeline_treatment.txt`. `START_DATE`/`STOP_DATE` are
day-offsets from the patient's diagnosis_date anchor; we materialise absolute
ISO dates. `is_osimertinib` flags rows whose agent normalises to osimertinib
(synonyms: `tagrisso`, `azd9291`).

### Outcome table

Every patient gets at least one row: `event_type = last_followup` at
`last_followup_date`. Additional rows are emitted for:
- `progression_recist` from any timeline-status row whose `EVENT_TYPE`
  contains `progression`.
- `death` from timeline-status rows where the patient's `STATUS` resolves
  to `deceased`.
- `molecular_resistance` (cBioPortal only, for MET amplifications observed
  in the MAF — placed at `last_followup_date` since CHORD does not carry the
  resistance-call timestamp).

## Real-data fetch

```bash
# GENIE (requires Synapse credentials)
export SYNAPSE_AUTH_TOKEN=...
python -c "from oncotraj.parsers import genie; genie.fetch_genie('data/raw/genie')"

# cBioPortal MSK-CHORD (public)
python -c "from oncotraj.parsers import cbioportal; cbioportal.fetch_cbioportal('data/raw/cbioportal')"
```

## Synthetic mode

`scripts/build_dataset.py --use-synthetic` writes GENIE- and cBioPortal-shaped
fixtures into `data/raw/synthetic/{genie,cbioportal}/` and runs the real
parsers over them. This is what CI exercises and what materialises a
first-version harmonised dataset on disk with no network or credentials.
