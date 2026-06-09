# OncoTraj v1 — Dataset Harmonization Spec

**Status:** Implementation-ready — 2026-05-14
**Owner:** Abhijoy Sarkar
**Target consumer:** Claude Code (or any engineer) writing parsers per source dataset.
**Output of parsers:** Four Parquet files (`patients.parquet`, `variants.parquet`, `treatments.parquet`, `outcomes.parquet`) conforming to the schemas below, plus a `_meta.json` per source recording row counts, schema version, and harmonization decisions actually applied.
**Schema version:** `oncotraj-schema/1.0.0` (semver — bump minor for additive fields, major for breaking changes).

> **Hard rule:** Parsers MUST NOT silently coerce values. If a field cannot be filled per the rules below, write the explicit missingness sentinel for that type (§7) and log to `_meta.json.coercion_log`. Silent imputation is a benchmark integrity failure.

---

## 0. Conventions

- **Dates:** ISO 8601 `YYYY-MM-DD`. No times. Timezone-naive — GENIE and MSK-CHORD anonymize to month-precision; if only month is given, use day=15 and set `date_precision="month"` in the originating record's `_precision` sibling field.
- **IDs:** All record IDs are RFC-4122 v4 UUIDs unless explicitly stated as opaque strings (e.g., `sample_id`, `site_id`).
- **`patient_id` format:** `{source_prefix}:{stable_source_id}` — e.g., `GENIE_BPC:GENIE-MSK-P-0001234`, `MSK_CHORD:CHORD-12345`, `TRACERX:LTX0001`, `FLAURA_SUPP:flaura-pt-042`. The colon-prefixed scheme guarantees no cross-source collision.
- **Strings:** UTF-8. No leading/trailing whitespace. Empty string is NOT a valid value — use the missingness sentinel.
- **Floats:** IEEE 754 double. Percentages stored as fractions in [0, 1] unless explicitly named `*_pct` (e.g., `baseline_pdl1_tps_pct` in [0, 100]).
- **VAF:** Always stored as fraction in [0, 1]. If source reports percent, divide by 100 at parse time.
- **HGVS:** Protein change in HGVS p.-notation using one-letter or three-letter — parsers MUST normalize to three-letter (e.g., `p.Cys797Ser`). cDNA change normalized to `c.` notation.
- **Genes:** HGNC official symbol, uppercase (e.g., `EGFR`, `MET`, `ERBB2`). Aliases must be resolved against HGNC at parse time — log unresolved aliases.
- **Enum values:** snake_case ASCII, no spaces. Unknown values for enum fields use the literal string `"unknown"` — never `null`, never empty string.
- **Booleans:** `true` / `false`. Unknown booleans use the missingness sentinel `null` (booleans are the one type where `null` is allowed; see §7).
- **Schema enforcement:** Each Parquet file MUST validate against the Pydantic models in `oncotraj/schemas.py` (to be generated from this spec). Validation failure on any row = parser error.

---

## 1. Source-Dataset Roster (recap)

| `source_dataset` enum value | What it provides | Has serial ctDNA? | Has imaging? | Access |
|---|---|---|---|---|
| `GENIE_BPC` | NSCLC v3.0 — clinical timeline + panel-based genomics | Partial (treating-center dependent) | Yes (PRISSMM-derived) | Registered |
| `MSK_CHORD` | ~25K patients clinical-genomic; NSCLC slice | Limited | Yes | cBioPortal open |
| `TRACERX` | Multi-region WES + serial ctDNA + outcomes | Yes (dense) | Yes | EGA controlled |
| `FLAURA_SUPP` | De-identified ctDNA dynamics from FLAURA supplements | Yes | Aggregate only | Public supplements |
| `AURA3_SUPP` | De-identified ctDNA dynamics from AURA3 supplements | Yes | Aggregate only | Public supplements |
| `ELIOS_SUPP` | ELIOS sub-study supplements | Yes | Aggregate only | Public supplements |
| `FMI_REGISTRY` | Foundation Medicine de-identified registry | No (tissue + liquid mixed) | No | Vendor publication |
| `GUARDANT_REGISTRY` | Guardant de-identified registry | Yes (assay-only) | No | Vendor publication |

A parser per source. All parsers emit the same four-table schema.

---

## 2. Patient Record (`patients.parquet`)

**Grain:** one row per `patient_id`. Cross-source duplicates are forbidden; if a patient appears in two sources, the parser MUST emit them under their source-specific `patient_id` and the entity-resolution step (§9) decides whether to merge.

| Field | Type | Required | Allowed values / format | Source-mapping notes |
|---|---|---|---|---|
| `patient_id` | str | Y | `{source_prefix}:{stable_source_id}` | See §0. |
| `source_dataset` | enum | Y | See §1 enum values | Set by parser. |
| `age_at_diagnosis_years` | int | Y | 18–110, or `-1` (sentinel) | GENIE provides 5-yr buckets; use bucket midpoint and set `_precision="bucket_5yr"`. |
| `sex` | enum | Y | `male` / `female` / `unknown` | GENIE `Sex` → mapping table §11. |
| `race` | enum | N | `white` / `black` / `asian` / `american_indian_alaska_native` / `native_hawaiian_pacific_islander` / `other` / `unknown` | GENIE `PRIMARY_RACE` → §11. |
| `ethnicity` | enum | N | `hispanic` / `non_hispanic` / `unknown` | GENIE `ETHNICITY` → §11. |
| `smoking_status` | enum | Y | `never` / `former` / `current` / `unknown` | GENIE `SMOKING_HISTORY` → §11. |
| `pack_years` | float | N | 0–200, or `-1.0` (sentinel) | TRACERx provides; GENIE rarely. |
| `diagnosis_date` | date | Y | ISO 8601 | Anchor: many sources give days-from-diagnosis offsets — diagnosis_date is the offset zero. |
| `stage_at_diagnosis` | enum | Y | `I` / `II` / `IIIA` / `IIIB` / `IIIC` / `IVA` / `IVB` / `IV_NOS` / `unknown` | Map AJCC v7 to v8 via §11 lookup; preserve original in `_meta`. |
| `histology` | enum | Y | `adenocarcinoma` / `squamous` / `adenosquamous` / `large_cell` / `NSCLC_NOS` / `small_cell` / `other` | `small_cell` only appears for patients who undergo transformation (§5.3). |
| `primary_site` | enum | N | `lung_upper_lobe_R` / `lung_middle_lobe_R` / `lung_lower_lobe_R` / `lung_upper_lobe_L` / `lung_lower_lobe_L` / `lung_NOS` / `unknown` | |
| `egfr_variant_class` | enum | Y | `exon19del` / `L858R` / `G719X` / `L861Q` / `S768I` / `exon20ins` / `compound_sensitizing` / `compound_with_resistance` / `unknown` | Derived from baseline tissue variant — see §3 derivation rule R1. |
| `baseline_pdl1_tps_pct` | float | N | 0–100, or `-1.0` (sentinel) | Tumor proportion score. |
| `baseline_brain_mets` | bool | N | `true`/`false`/`null` | At diagnosis OR within ±30 days of osimertinib start, whichever earlier. |
| `baseline_liver_mets` | bool | N | `true`/`false`/`null` | Same window. |
| `baseline_bone_mets` | bool | N | `true`/`false`/`null` | Same window. |
| `baseline_ldh_uln_ratio` | float | N | 0.1–20.0, or `-1.0` | Lab value / upper limit of normal at last value pre-osimertinib. |
| `cooccurring_tp53` | bool | N | `null` allowed | Pathogenic per OncoKB at baseline tissue or first ctDNA. |
| `cooccurring_rb1` | bool | N | `null` allowed | Same. |
| `cooccurring_cdkn2a` | bool | N | `null` allowed | Same. |
| `site_id` | str | Y | Opaque, ≤32 chars | Anonymized institution code. Use `unknown` if not available. |
| `consent_tier` | enum | N | `tier1_research` / `tier2_clinical` / `tier3_publication_only` / `unknown` | Source-specific; pass through. |
| `vital_status_at_last_followup` | enum | Y | `alive` / `deceased` / `unknown` | |
| `last_followup_date` | date | Y | ISO 8601 | The latest date for which we have ANY information (visit, imaging, ctDNA draw, death). |
| `date_of_death` | date | N | ISO 8601, or `null` | Required if `vital_status_at_last_followup == "deceased"`. |
| `cause_of_death` | enum | N | `disease` / `treatment_related` / `other_known` / `unknown` / `null` | Required if `date_of_death` set. |
| `included_in_v1_cohort` | bool | Y | `true` / `false` | Set by parser per §10 inclusion/exclusion. Patients with `false` are still emitted (for audit) but not consumed downstream. |
| `exclusion_reason` | enum | N | `prior_EGFR_TKI` / `not_first_line_osimertinib` / `insufficient_followup` / `missing_outcome` / `not_EGFR_sensitizing` / `null` | Required iff `included_in_v1_cohort == false`. |

**Patient-level invariants (enforced at validation):**
- I1. `last_followup_date >= diagnosis_date`.
- I2. If `vital_status == "deceased"` then `date_of_death != null` and `date_of_death <= last_followup_date + 14` (allow 2-week reporting delay).
- I3. `egfr_variant_class != "unknown"` for any patient with `included_in_v1_cohort == true`.
- I4. `patient_id` is globally unique within `patients.parquet`.

---

## 3. Variant Record (`variants.parquet`)

**Grain:** one row per (variant call × sample × patient). The same biological variant detected at two ctDNA timepoints is two rows.

| Field | Type | Required | Allowed values / format | Notes |
|---|---|---|---|---|
| `variant_id` | str | Y | UUIDv4 | Parser-generated. |
| `patient_id` | str | Y | FK → patients | |
| `sample_id` | str | Y | Source-stable, ≤64 chars | E.g., GENIE `SAMPLE_ID`. |
| `sample_type` | enum | Y | `tumor_tissue` / `ctDNA_plasma` / `ctDNA_cfDNA` / `normal_blood` / `normal_tissue` / `unknown` | |
| `sample_date` | date | Y | ISO 8601 | If only days-from-diagnosis given, compute as `diagnosis_date + offset`. |
| `assay` | enum | Y | `MSK_IMPACT_341` / `MSK_IMPACT_410` / `MSK_IMPACT_468` / `MSK_ACCESS` / `FoundationOne_CDx` / `FoundationOne_Liquid_CDx` / `Guardant360` / `Guardant360_CDx` / `Tempus_xF` / `Tempus_xT` / `WES` / `WGS` / `custom_panel` / `unknown` | |
| `gene_symbol` | str | Y | HGNC uppercase | |
| `alteration_type` | enum | Y | `SNV` / `indel_insertion` / `indel_deletion` / `indel_complex` / `CNV_amplification` / `CNV_deletion` / `fusion` / `rearrangement` / `splice` / `promoter` | |
| `protein_change_hgvs` | str | N | HGVS p.-notation, three-letter (e.g., `p.Cys797Ser`) | Required if `alteration_type` ∈ {SNV, indel_*, splice}; else `null`. |
| `cdna_change_hgvs` | str | N | HGVS c.-notation | Same conditionality. |
| `exon` | int | N | 1–50, or `null` | |
| `vaf` | float | N | 0.0–1.0, or `-1.0` (sentinel) | Required for ctDNA SNVs/indels. `-1.0` for CNVs/fusions. |
| `copy_number` | float | N | 0.0–50.0, or `-1.0` | Required for CNVs. |
| `read_depth` | int | N | 1–100000, or `-1` | Required for ctDNA. |
| `alt_allele_count` | int | N | 0–100000, or `-1` | |
| `oncokb_oncogenic` | enum | Y | `oncogenic` / `likely_oncogenic` / `predicted_oncogenic` / `resistance` / `likely_neutral` / `inconclusive` / `unknown` | Re-annotate via OncoKB API at parse time using OncoKB build date pinned in `_meta`. |
| `oncokb_mutation_effect` | enum | N | `gain_of_function` / `likely_gain_of_function` / `loss_of_function` / `likely_loss_of_function` / `switch_of_function` / `inconclusive` / `unknown` | |
| `is_germline` | bool | Y | `true` / `false` / `null` | `null` only if assay cannot distinguish. |
| `is_baseline_driver` | bool | Y | `true` / `false` | Derived flag — see R1. |
| `resistance_mechanism_class` | enum | N | See §5.3 taxonomy, or `null` | Required if `is_resistance_call == true`. |
| `is_resistance_call` | bool | Y | `true` / `false` | Derived flag — see R2. |
| `evidence_level` | enum | N | `OncoKB_1` / `OncoKB_2` / `OncoKB_3A` / `OncoKB_3B` / `OncoKB_4` / `CIViC_A` / `CIViC_B` / `CIViC_C` / `CIViC_D` / `CIViC_E` / `manual` / `unknown` | |
| `raw_source_call` | str | N | ≤512 chars | The original line / row from the source file, JSON-encoded. For provenance only. |

**Derivation rules (parsers MUST implement these — they are not optional):**

- **R1 (`is_baseline_driver`):** `true` iff variant is in `EGFR` gene AND `oncokb_oncogenic` ∈ {`oncogenic`, `likely_oncogenic`} AND variant is observed in the patient's earliest-dated tumor_tissue OR ctDNA sample on/before `regimen_class == EGFR_TKI_3G_OSIMERTINIB` `start_date`. Drives the patient's `egfr_variant_class`.
- **R2 (`is_resistance_call`):** `true` iff variant matches one of the §5.3 mechanism patterns AND sample is post-osimertinib-start (`sample_date > osimertinib treatment.start_date`) AND VAF ≥ 0.001 (0.1%) for ctDNA SNVs/indels or `copy_number ≥ 6` for amplifications (FoundationOne and Guardant amp thresholds are vendor-specific; default to ≥6, see §11). For tissue samples, vendor amplification call is trusted. Histologic small-cell transformation is NOT a variant record — see §5.3.
- **R3 (variant deduplication within a sample):** Two variant rows with same `(patient_id, sample_id, gene_symbol, protein_change_hgvs)` are forbidden — collapse to one row, taking the maximum-VAF call and recording the conflict in `_meta`.

**Variant-level invariants:**
- I5. `sample_date >= patients.diagnosis_date` for the linked patient.
- I6. If `alteration_type == "SNV"`, `protein_change_hgvs` is non-null.
- I7. If `resistance_mechanism_class` is non-null, `is_resistance_call == true`.

---

## 4. Treatment Record (`treatments.parquet`)

**Grain:** one row per regimen episode per patient. A re-challenge with osimertinib after a chemo break is two episodes.

| Field | Type | Required | Allowed values / format | Notes |
|---|---|---|---|---|
| `treatment_id` | str | Y | UUIDv4 | |
| `patient_id` | str | Y | FK → patients | |
| `line_of_therapy` | int | Y | 1–20, or `-1` (sentinel) | Counted per standard oncology convention: a new line begins at progression OR at intolerable toxicity, NOT at planned maintenance. |
| `regimen_class` | enum | Y | `EGFR_TKI_1G` / `EGFR_TKI_2G` / `EGFR_TKI_3G_OSIMERTINIB` / `EGFR_TKI_3G_OTHER` / `EGFR_TKI_4G` / `chemo_platinum_doublet` / `chemo_single` / `IO_monotherapy` / `IO_chemo` / `IO_chemo_VEGF` / `bispecific_amivantamab` / `ADC_patritumab` / `ADC_other` / `targeted_MET_inhibitor` / `targeted_HER2` / `targeted_other` / `SBRT` / `surgery` / `best_supportive_care` / `other` / `unknown` | The v1 cohort selection requires `EGFR_TKI_3G_OSIMERTINIB` at `line_of_therapy == 1`. |
| `drug_names_rxnorm` | list[str] | Y | RxNorm RxCUI or generic name (lowercased) | E.g., `["osimertinib"]`. Empty list is invalid; use `["unknown"]` instead. |
| `start_date` | date | Y | ISO 8601 | |
| `end_date` | date | N | ISO 8601, or `null` | `null` means ongoing at last followup. |
| `dose_at_start_mg` | float | N | 0–10000, or `-1.0` | For osimertinib v1 cohort, default 80mg unless source says otherwise. |
| `dose_modification_occurred` | bool | N | `null` allowed | Any documented dose reduction during the episode. |
| `best_response` | enum | Y | `CR` / `PR` / `SD` / `PD` / `NE` / `unknown` | RECIST 1.1 best response during the episode. |
| `discontinuation_reason` | enum | N | `progression` / `toxicity` / `completed_planned` / `patient_choice` / `death` / `lost_to_followup` / `ongoing` / `unknown` / `null` | `null` only if `end_date == null`. |
| `concurrent_local_therapy` | bool | N | `null` allowed | Any SBRT/surgery within the episode dates. |
| `enrolled_in_trial` | bool | N | `null` allowed | |
| `trial_id` | str | N | NCT ID or `null` | |

**Treatment-level invariants:**
- I8. `start_date >= patients.diagnosis_date`.
- I9. For a given patient, no two treatment rows have overlapping `[start_date, end_date]` intervals UNLESS one is `regimen_class ∈ {SBRT, surgery, best_supportive_care}` — these can overlap systemic therapy.
- I10. `line_of_therapy` is monotonically non-decreasing in `start_date` per patient.
- I11. If `discontinuation_reason == "death"`, `end_date == patients.date_of_death`.

---

## 5. Outcome Record (`outcomes.parquet`)

**Grain:** one row per discrete clinical event. A single visit that produces both an imaging assessment and a ctDNA draw is two rows.

| Field | Type | Required | Allowed values / format | Notes |
|---|---|---|---|---|
| `outcome_id` | str | Y | UUIDv4 | |
| `patient_id` | str | Y | FK → patients | |
| `event_type` | enum | Y | `imaging_assessment` / `biopsy` / `ctdna_draw` / `lab_panel` / `progression_recist` / `progression_molecular` / `progression_clinical` / `treatment_change` / `hospitalization` / `death` / `last_followup` / `transformation_diagnosis` | |
| `event_date` | date | Y | ISO 8601 | |
| `linked_treatment_id` | str | N | FK → treatments, or `null` | The regimen the patient was on at `event_date`. |
| `linked_sample_id` | str | N | FK → variants.sample_id, or `null` | For `ctdna_draw`, `biopsy`, `progression_molecular`. |
| `recist_response` | enum | N | `CR` / `PR` / `SD` / `PD` / `NE` / `null` | Required iff `event_type == "imaging_assessment"`. |
| `sum_of_diameters_mm` | float | N | 0–10000, or `-1.0` | RECIST 1.1 sum of target lesions. |
| `new_lesion_detected` | bool | N | `null` allowed | |
| `ctdna_total_tumor_fraction` | float | N | 0.0–1.0, or `-1.0` | Vendor-reported (Guardant TF, FMI TF, etc.) when available. |
| `ctdna_egfr_sensitizing_vaf` | float | N | 0.0–1.0, or `-1.0` | Max VAF across detected sensitizing alleles at this draw. |
| `ctdna_assay` | enum | N | See §3 assay enum, or `null` | |
| `progression_confirmed` | bool | N | `null` allowed | RECIST 1.1: a `PD` call requires a confirmatory scan ≥4 weeks later UNLESS clinical deterioration is documented. `true` iff confirmation criteria met. |
| `mechanism_at_progression` | enum | N | See §5.3, or `null` | Required iff `event_type == "progression_molecular"`. |
| `is_resistance_event` | bool | Y | `true` / `false` | Derived flag — see §6. |
| `resistance_event_subtype` | enum | N | `recist_pd` / `molecular_emergence` / `vaf_resurgence` / `histologic_transformation` / `death_from_disease` / `null` | Required iff `is_resistance_event == true`. |
| `is_index_resistance_event` | bool | Y | `true` / `false` | The EARLIEST resistance event per patient (§6.3). At most one `true` per patient. |
| `notes` | str | N | ≤1024 chars, or `null` | Free-text not used by models. |

**Outcome-level invariants:**
- I12. Exactly one `event_type == "last_followup"` row per patient, on `patients.last_followup_date`.
- I13. Iff `patients.vital_status_at_last_followup == "deceased"`, exactly one `event_type == "death"` row with `event_date == patients.date_of_death`.
- I14. At most one `is_index_resistance_event == true` row per patient.
- I15. Every `is_resistance_event == true` row has `event_date <= patients.last_followup_date`.

---

## 5.3 Resistance Mechanism Taxonomy (locked for v1)

| `resistance_mechanism_class` enum | Variant-level pattern | Calling threshold |
|---|---|---|
| `EGFR_C797S` | EGFR `p.Cys797Ser` or `p.Cys797Gly` SNV | ctDNA VAF ≥ 0.001; tissue any |
| `MET_AMP` | MET amplification | copy_number ≥ 6 (ctDNA), or vendor-positive amp (tissue/FISH) |
| `HER2_AMP` | ERBB2 amplification | copy_number ≥ 6 (ctDNA), or vendor-positive (tissue/FISH) |
| `EGFR_AMP` | EGFR amplification (in absence of co-occurring C797S) | copy_number ≥ 6 (ctDNA), or vendor-positive (tissue/FISH) |
| `small_cell_transformation` | NOT a variant record — recorded as `outcomes.event_type == "transformation_diagnosis"` | Biopsy-confirmed small-cell or neuroendocrine histology on a sample dated AFTER osimertinib start |
| `other` | Any of: BRAF V600E SNV, KRAS G12X SNV, PIK3CA hotspot SNV (E545K/H1047R/E542K), FGFR1/2/3 amplification, NRG1 fusion, RET fusion, ALK fusion, ROS1 fusion, NTRK fusion, EGFR L718Q, EGFR G724S, exon 20 insertion emergent | Source-appropriate threshold per `alteration_type` |

> **Hard rule:** Histologic transformation is recorded ONLY in outcomes (event_type = `transformation_diagnosis`) and resistance_mechanism_class is set at the corresponding `progression_molecular`/`progression_clinical` row. It does NOT generate a variants.parquet row.

---

## 6. The Resistance Event Rule

This is the load-bearing definition for §5.1 (Task A) and §5.2 (Task B) of the paper. **Parsers MUST implement it exactly — do not interpret.**

### 6.1 What counts as a resistance event

A `resistance_event` is produced when ANY of the following criteria are met, evaluated PER PATIENT after `regimen_class == EGFR_TKI_3G_OSIMERTINIB` `start_date` and before `discontinuation_reason ∈ {toxicity, completed_planned, patient_choice}` (the latter excluded so that pure toxicity-driven discontinuations don't pollute the resistance label):

**Criterion A — RECIST 1.1 progression (`resistance_event_subtype = "recist_pd"`):**
A row with `event_type == "imaging_assessment"`, `recist_response == "PD"`, and `progression_confirmed == true`.
Confirmation requires either (a) a follow-up scan ≥28 days later also showing PD, or (b) documented clinical deterioration leading to treatment change within 14 days.

**Criterion B — Molecular emergence (`resistance_event_subtype = "molecular_emergence"`):**
A new variant row with `is_resistance_call == true` AND `resistance_mechanism_class` ∈ §5.3 AND the variant was NOT detected (above its calling threshold) in any sample dated before osimertinib start. Threshold per §5.3.

**Criterion C — VAF resurgence of sensitizing allele (`resistance_event_subtype = "vaf_resurgence"`):**
Two consecutive ctDNA draws ≥14 days apart, both AFTER a documented nadir (defined as the minimum sensitizing-allele VAF observed at least 21 days post-osimertinib start), each with `ctdna_egfr_sensitizing_vaf ≥ max(2 × nadir, 0.001)`. The event date is the date of the SECOND confirming draw.

**Criterion D — Histologic transformation (`resistance_event_subtype = "histologic_transformation"`):**
An `outcomes.event_type == "transformation_diagnosis"` row.

**Criterion E — Death from disease while on osimertinib (`resistance_event_subtype = "death_from_disease"`):**
`patients.cause_of_death == "disease"` AND `date_of_death` falls within the osimertinib episode AND no other resistance event has been recorded in the prior 60 days. The event date is `date_of_death`.

### 6.2 What does NOT count as a resistance event

- Toxicity-driven discontinuation (no progression evidence).
- Single PD scan without confirmation and without clinical deterioration.
- Single VAF rise that does not meet Criterion C (one rising draw is not enough).
- Detection of a §5.3 mechanism variant at baseline (pre-osimertinib) — that is a `is_baseline_driver` co-mutation or a pre-existing subclone, not an emergent resistance event.
- Death from non-disease causes (treatment-related toxicity, other_known cause).

### 6.3 The index resistance event

For each patient, the **index resistance event** is the EARLIEST event meeting any criterion A–E. Parsers set `is_index_resistance_event = true` on that row only.

If a patient has no event meeting any criterion by their `last_followup_date`, they are **right-censored** at `last_followup_date` and `is_index_resistance_event` is `false` for all their rows.

### 6.4 Mechanism assignment at the index event

If the index event is Criterion B or D, `mechanism_at_progression` is the mechanism that triggered it.
If the index event is Criterion A, C, or E, `mechanism_at_progression` is the mechanism observed on the NEAREST molecular sample within ±60 days of the event_date, preferring post-event samples. If no such sample exists within ±60 days, `mechanism_at_progression = "unknown"`.

> **This is the single most-contested definition in the spec.** Any deviation MUST be logged in `_meta.json.resistance_rule_deviations` with a per-patient justification — the reproducibility audit will check this.

---

## 7. Missingness Handling

### 7.1 Per-type sentinels

| Type | Missing sentinel | Rationale |
|---|---|---|
| `int` | `-1` | Reserves the value for explicit missingness; no clinical int field has a legitimate −1. |
| `float` | `-1.0` | Same. |
| `str` (enum) | `"unknown"` | Always present in the enum. |
| `str` (free) | `null` | Allowed only where field is non-required. |
| `bool` | `null` | The only type where `null` is the missing sentinel. |
| `date` | `null` | Allowed only where field is non-required. |
| `list[str]` | `["unknown"]` | Empty list is invalid. |

### 7.2 Required vs. optional

- `Required = Y` in the schema means the field MUST be present and non-null. If the source genuinely cannot provide it, parsers use the type-appropriate sentinel (NOT null, except for booleans and dates as noted above).
- `Required = N` means `null` is permitted.

### 7.3 Imputation policy

**Parsers MUST NOT impute.** Imputation, if any, happens in a downstream feature-engineering step that is explicitly versioned (`oncotraj-features/{semver}`). Parsers emit raw harmonized values plus sentinels. The benchmark harness is responsible for documenting any imputation it applies.

The one exception: **date precision rounding** as described in §0 (month → day=15). This is harmonization, not imputation, and must be logged in `_meta`.

### 7.4 Logging

Every parser writes `_meta.json` with:
- `source_dataset`, `schema_version`, `parser_version`, `oncokb_build_date`, `run_timestamp`.
- `row_counts`: per-table counts.
- `coercion_log`: every value-level coercion (allele-name normalization, date precision rounding, vendor-amp thresholding) with row IDs.
- `validation_failures`: rows dropped due to invariant violation, with the invariant ID (`I1`–`I15`) and reason.
- `resistance_rule_deviations`: per-patient deviations from §6, if any.

---

## 8. Cross-Table Foreign Keys (validated end-to-end)

- `variants.patient_id` → `patients.patient_id`
- `treatments.patient_id` → `patients.patient_id`
- `outcomes.patient_id` → `patients.patient_id`
- `outcomes.linked_treatment_id` → `treatments.treatment_id` (nullable)
- `outcomes.linked_sample_id` → `variants.sample_id` (nullable; note: `sample_id` is not unique across patients, so the (patient_id, sample_id) pair is the actual lookup key)

A post-parse validator runs these FK checks. Orphans are a parser bug, not a data issue.

---

## 9. Cross-Source Entity Resolution

Not performed at parse time. Each parser emits source-prefixed `patient_id`s and we accept the resulting duplication. The cross-source resolution step is a separate, explicitly versioned pipeline (`oncotraj-resolution/{semver}`) that runs AFTER all four parsers and produces a `patient_alias_map.parquet`. For v1, the default policy is **no merging** — TRACERx patients are treated as separate entities from GENIE-BPC even on the off-chance of overlap. This is a deliberate conservative choice; flag for review if any specific source pair is suspected to overlap.

---

## 10. v1 Cohort Inclusion / Exclusion (parser sets `included_in_v1_cohort`)

A patient is set to `included_in_v1_cohort = true` iff ALL of:

- C1. `egfr_variant_class` ∈ {`exon19del`, `L858R`, `G719X`, `L861Q`, `S768I`, `compound_sensitizing`}.
- C2. At least one `treatments` row with `regimen_class == EGFR_TKI_3G_OSIMERTINIB` AND `line_of_therapy == 1`.
- C3. No `treatments` row with `regimen_class ∈ {EGFR_TKI_1G, EGFR_TKI_2G}` at `line_of_therapy == 1` (no prior EGFR TKI).
- C4. At least one `outcomes` row dated ≥ osimertinib `start_date` + 30 days (some follow-up exists).
- C5. Either an index resistance event is recorded OR the patient has ≥6 months of follow-up from osimertinib start.

If any of C1–C5 fails, `included_in_v1_cohort = false` and `exclusion_reason` is set to the FIRST failing criterion's standard string (see schema).

**FLAURA2 chemo arm:** Included with `concurrent_chemo_at_start = true` recorded in a v1.1 follow-up field (placeholder — note this in `_meta` for v1).

---

## 11. Source-to-Enum Mapping Tables (parser must implement)

The mapping tables themselves live in `oncotraj/mappings/`. Each parser imports the relevant table. Examples (full tables to be generated):

- `mapping_sex.csv`: `GENIE.Sex == "Female"` → `female`, etc.
- `mapping_race.csv`: GENIE `PRIMARY_RACE` codes → enum.
- `mapping_smoking.csv`: GENIE `NAACCR_SMOKING_HISTORY` ↔ enum.
- `mapping_stage_ajcc.csv`: AJCC v7 → v8 → schema enum.
- `mapping_assay.csv`: source-specific assay names → enum.
- `mapping_amp_threshold.csv`: vendor → copy-number threshold (default 6 if not specified).
- `mapping_drug_to_regimen_class.csv`: RxNorm → `regimen_class`.

If a source value is encountered that is NOT in the mapping table, parsers MUST fail loudly, not silently bucket to `unknown`. Add a new row to the mapping table after manual review.

---

## 12. Acceptance Tests Per Parser (must pass before merge)

Each parser ships with:
- `tests/test_{source}_smoke.py`: parses a 10-patient fixture and validates schema.
- `tests/test_{source}_invariants.py`: runs all I1–I15 invariants against the full output.
- `tests/test_{source}_resistance_rule.py`: 5–10 hand-curated patients per resistance criterion A–E, with expected `is_index_resistance_event` and `mechanism_at_progression` values.
- `tests/test_{source}_no_silent_coercion.py`: artificially corrupt 3 known fields and assert that the parser raises or logs to `coercion_log` rather than silently fixing.

Coverage target: ≥80% on parser code.

---

## 13. Schema Versioning & Change Control

- Schema version is stamped into every Parquet file's metadata as `oncotraj_schema_version`.
- Minor bumps (1.0.x → 1.1.x): additive only — new optional fields, new enum values appended.
- Major bumps (1.x → 2.x): any breaking change. Parsers must explicitly opt in.
- Removal of fields or enums requires a major bump and a migration script (`migrations/1_to_2.py`).

---

## 14. What This Spec Deliberately Does Not Define (v1.1+)

These are out of scope for v1 — parsers should NOT silently include them:

- Imaging-derived features beyond RECIST scalars (e.g., radiomics).
- Pathology images / WSI features.
- Single-cell or spatial data.
- Microbiome / metabolome.
- Patient-reported outcomes (PRO).
- Pharmacy fill-rate data.
- Insurance / claims data.
- Free-text clinical notes (NLP-derived features deferred).
- Cross-source entity resolution (see §9).

Adding any of these requires a major schema bump.

---

## 15. Open Questions Flagged for Aarchi

| Question | Default if no decision | Lock by |
|---|---|---|
| Should histologic transformation under Criterion D require a *new* biopsy on osimertinib, or is paired re-review of an existing baseline biopsy acceptable? | Require new biopsy. | Before TRACERx parser. |
| For Criterion C (VAF resurgence), is the 2× / 0.1% threshold the right pair, or should we adopt the ELIOS analytical thresholds? | Use 2× / 0.1% (current spec). | Before ELIOS parser. |
| Should we record concomitant medications (steroids, PPIs)? | Defer to v1.1 — flag in `_meta` only. | Before MSK-CHORD parser. |
| Are FoundationOne Liquid and Guardant360 amp thresholds harmonizable, or do we keep vendor-specific? | Vendor-specific via `mapping_amp_threshold.csv`. | Before FMI/Guardant parsers. |

---

*End of spec. A parser conforming to this spec produces four Parquet files validatable end-to-end. If a parser implementer finds an ambiguity, that is a spec bug — file an issue against `oncotraj/DATASET_SPEC.md` rather than guessing.*
