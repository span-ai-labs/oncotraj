# OncoTraj v1 — Resistance-Event Labeling Guidelines

**Status:** DRAFT — awaiting Aarchi review before freeze
**Owner:** Abhijoy Sarkar
**Reviewer (required pre-freeze):** Aarchi Singh Thakur
**Audience:** Future labelers, parser implementers, peer reviewers, regulatory auditors
**Companions:** Reads alongside `DATASET_SPEC.md` (§5.3 taxonomy, §6 resistance rule). When this document and the spec disagree, **this document wins for labeling decisions; the spec wins for storage schema**.

> **Why this document exists.** The §5.3 taxonomy is defensible only if the *detection* rule for each class is defensible. A per-mechanism F1 number is a meaningless headline metric unless every reviewer can audit how a given patient got assigned to a given class. This document is the audit trail.

---

## 0. Operating Principles

1. **Every label is a falsifiable call.** For each patient, the label can be reconstructed from the underlying variant/outcome/treatment rows. If a label cannot be reconstructed, that is a labeling bug, not a "judgment call."
2. **Thresholds are explicit and applied uniformly.** No "case-by-case" softening of thresholds. If a borderline case fails the threshold, it gets the conservative call.
3. **The conservative call is "Unknown," not "Other."** When evidence is insufficient to assign a §5.3 mechanism with the rules below, the patient's index event still records that resistance occurred (§6 of the spec) but `mechanism_at_progression = "unknown"`. The Task C model is then evaluated on the patients with mechanism calls; the unknown rate is itself a headline statistic.
4. **Co-occurrence is the rule, not the exception.** Roughly 20–30% of post-osimertinib patients have >1 mechanism detectable at progression in the published literature. We assign a single *dominant* mechanism per the hierarchy in §3 and record secondaries — both for analysis and for ablations.
5. **Hierarchy is biological, not convenient.** When two mechanisms compete, the rule prefers the one with stronger biological evidence of clonal dominance, not the one easier to label.
6. **Every borderline call is logged.** Parsers write to `_meta.json.labeling_log` with the row IDs, the threshold-failure reason, and the conservative-default applied.

---

## 1. Per-Mechanism Detection Rules

The rules below define "detected" for the purposes of (a) populating `variants.resistance_mechanism_class` (where applicable), (b) firing Criterion B of §6 (molecular emergence), and (c) assigning `outcomes.mechanism_at_progression` at the index event.

For each mechanism we specify: **ctDNA rule**, **tissue rule**, **disqualifiers** (things that look like detection but aren't), and **edge-case handling**.

### 1.1 EGFR C797S (and related on-target resistance variants)

**Biology:** Acquired SNV at the osimertinib covalent-binding cysteine. Most commonly `p.Cys797Ser` (c.2389T>A); rarely `p.Cys797Gly` (c.2389T>G). Detected in ~6–22% of post-1L-osimertinib resistance cases [Chmielecki 2023; Oxnard 2018]. Treatment-relevant configurations (cis vs trans with T790M) matter clinically but **do not affect the resistance-mechanism label** — both call as `EGFR_C797S`.

**ctDNA detection rule:**
- Assay must cover EGFR exon 20 with reported coverage depth ≥1,000× at codon 797 (mandatory for negative calls; if depth is below threshold AND the call is negative, the result is `inconclusive`, not `negative`).
- VAF ≥ 0.1% (0.001 as fraction) at any post-osimertinib-start ctDNA draw.
- The variant must NOT have been called in any pre-osimertinib sample at the same calling threshold (pre-existing C797S → see disqualifiers).
- Confirmation on a second draw is **not required** for C797S — a single supra-threshold ctDNA call on a CLIA-grade assay is sufficient. Reason: C797S is uniquely specific to osimertinib selection pressure and has a very low background rate.

**Tissue detection rule:**
- Any tissue NGS call with VAF ≥ 0.05 (5%), sample dated after osimertinib start, with no prior C797S detection. The 5% tissue floor accounts for sampling and tumor cellularity; below 5% the call is `inconclusive`.

**Disqualifiers (do NOT label as C797S):**
- C797S detected at baseline or pre-osimertinib — this is a rare germline-context or prior-TKI artifact; do not call as acquired resistance. Set `is_baseline_driver_co_mutation = true`.
- C797S detected in a non-EGFR-context variant (i.e., another gene's residue 797). Parser bug — gene_symbol must be EGFR.
- C797S on a single draw at VAF < 0.1% with no confirmation — this is `inconclusive`, log to `_meta` and do not label.

**Edge cases:**
- **Both C797S and T790M present** (relevant for 2L cohort): label as `EGFR_C797S`. T790M is the upstream resistance allele that motivated 2L osi; C797S is the OncoTraj-relevant new event.
- **C797G (rare variant) present:** label as `EGFR_C797S` (same class — C797S is the class name, not a single variant). Record the exact protein change in `variants.protein_change_hgvs`.
- **L718Q, L792X, G796X, G724S** — these are on-target but distinct from the C797 class. Label as `other`, NOT `EGFR_C797S`. Record the specific variant. (Rationale: separating these preserves the per-mechanism F1 interpretation — C797S has a specific drug-discovery program around it; the rarer on-target variants do not.)

### 1.2 MET Amplification

**Biology:** Most common off-target resistance mechanism on first-line osimertinib (~15–30% across studies [Chmielecki 2023; Piotrowska 2017; Choudhury 2022]). Bypass-pathway activation via MET copy gain. Often co-occurs with other mechanisms.

**ctDNA detection rule:**
- Assay must report MET copy number or MET copy-ratio. Pure SNV panels without CNV calling cannot detect MET amp from ctDNA — patients with such assays only are marked `inconclusive` for MET amp, not `negative`.
- **Standardized threshold (vendor-normalized):** plasma copy number ≥ 6.0.
- **Vendor-specific accept rules** (when the vendor's positivity flag is the primary call):
  - Guardant360: vendor-positive amp call accepted.
  - FoundationOne Liquid CDx: vendor-positive amp call accepted.
  - Tempus xF: vendor-positive amp call accepted.
  - In all cases, record the vendor call and the raw copy number; if available, use vendor's positivity flag as the primary call and record the numeric CN for downstream auditing.
- The amp must NOT have been called in any pre-osimertinib sample (see disqualifiers).

**Tissue detection rule:**
- FISH: MET:CEP7 ratio ≥ 2.0 OR mean MET copies/cell ≥ 6.0.
- NGS-derived tissue copy number ≥ 6.0.
- IHC alone (e.g., MET 3+ staining) is NOT sufficient for the label — IHC is a screening tool, not the call. If only IHC is available, mark `inconclusive`.

**Disqualifiers (do NOT label as MET_AMP):**
- Pre-osimertinib MET amplification (de novo MET-amp, not resistance). Record as `is_baseline_driver_co_mutation = true`. Patient remains in cohort but MET amp is not the resistance label.
- MET exon 14 skipping mutation (`METΔex14`) without amplification — this is a distinct driver event, not amplification. Label any post-osimertinib emergent METΔex14 as `other` and record the specific alteration.
- Low-level MET gain (4 ≤ CN < 6): **does NOT count.** Mark as `inconclusive` for MET amp; the patient can still be assigned another mechanism via the hierarchy.
- Polysomy 7 (whole-chromosome gain) without focal MET amplification — disqualifies if FISH ratio < 2.0 even with absolute copies > 6 (the ratio criterion is the focal-amplification signal).

**Edge cases:**
- **MET amp detected only on tissue after a negative ctDNA panel that included MET:** label as `MET_AMP`. Tissue trumps ctDNA when both are run and disagree, because tissue captures non-shedding tumor.
- **MET amp + EGFR amp co-detected:** see §3 hierarchy.
- **MET amp at CN exactly 6.0:** counts (the threshold is ≥6, inclusive).
- **MET CN reported as fractional (e.g., 5.8 with reported uncertainty)** : does not count. Round down for conservatism; log the borderline call.

### 1.3 EGFR Amplification

**Biology:** Amplification of the EGFR locus carrying the sensitizing allele. Detected in ~5–15% of resistance cases. Frequently co-occurs with C797S; when C797S is also present, C797S wins as the dominant label (see §3).

**ctDNA detection rule:**
- Plasma copy number ≥ 6.0 at the EGFR locus, AND
- The sensitizing-allele VAF rises to ≥ 30% at the same draw (this distinguishes sensitizing-allele amplification — the relevant resistance pattern — from wildtype-allele amplification, which is biologically uninteresting).
- The amp must NOT have been called pre-osimertinib at the same threshold.

**Tissue detection rule:**
- FISH: EGFR:CEP7 ratio ≥ 2.0 OR mean EGFR copies/cell ≥ 6.0.
- NGS tissue copy number ≥ 6.0 with confirmed sensitizing-allele majority (mutant allele fraction > 50% of total EGFR reads).

**Disqualifiers (do NOT label as EGFR_AMP):**
- EGFR amplification at baseline (pre-osimertinib). De novo EGFR-amp is common in some EGFR-mutant tumors; only acquired EGFR-amp counts. Pre-osimertinib EGFR amp gets `is_baseline_driver_co_mutation = true`.
- Amplification of the wildtype EGFR allele only (mutant allele fraction < 50% despite high CN) — does not meet the resistance pattern. Mark `inconclusive`.
- C797S co-detected at the same draw — C797S wins per §3; EGFR amp is recorded as secondary mechanism but NOT as the dominant label.

**Edge cases:**
- **EGFR amp + MET amp co-detected, no C797S:** the higher CN/ratio wins. Tie-break: prefer the one with the higher *clonal* fraction (CN normalized by overall tumor fraction). See §3.5 for tie-break procedure.
- **EGFR amp with co-occurring L858R sensitizing mutation that has risen to ~50% VAF but no overt copy gain (CN ≈ 4):** does not count as EGFR_AMP — must meet the CN ≥ 6 floor. VAF rise alone is captured by the §6 Criterion C (VAF resurgence) trigger, but the mechanism label remains `unknown` unless another mechanism is detected.

### 1.4 Histologic Transformation (Small-Cell / Neuroendocrine)

**Biology:** Phenotypic switch — most commonly to small-cell lung carcinoma (SCLC), occasionally to large-cell neuroendocrine or squamous. Detected in ~3–10% of resistance cases. The most clinically distinct mechanism — patients require a fundamentally different therapeutic regimen.

**Tissue detection rule (mandatory — there is no pure-ctDNA call for transformation in v1):**
- Biopsy or rebiopsy after osimertinib start showing small-cell, large-cell neuroendocrine, or squamous histology by H&E morphology, confirmed by IHC: at least two of {synaptophysin, chromogranin, CD56, INSM1} positive for neuroendocrine transformation; p40/p63 positive for squamous transformation.
- The transformed biopsy must retain the original EGFR sensitizing mutation (by paired sequencing of the transformed sample). If sequencing of the transformed sample is unavailable, transformation is `inconclusive` — this is a hard rule because lineage proof is the only thing distinguishing transformation from a second primary tumor.

**Supporting ctDNA features (NOT sufficient alone but supporting):**
- Loss-of-function alterations in RB1 AND TP53 emerging post-osimertinib on ctDNA — these are the SCLC genotype. **In v1, this pattern WITHOUT paired tissue does NOT call transformation.** It can be recorded as `secondary_mechanism = "rb1_tp53_lof_no_paired_tissue"` for analysis. (Defer to v1.1 the question of whether ctDNA-only SCLC genotype calls should label transformation. Rationale: too many false-positives from incidental RB1 loss in non-transformed tumors.)

**Disqualifiers (do NOT label as `small_cell_transformation`):**
- ctDNA-only SCLC genotype without paired tissue.
- Pre-existing mixed histology at diagnosis — if the baseline biopsy already showed mixed adeno + small-cell components, the post-progression small-cell histology is NOT a transformation event (it's selection of a pre-existing clone). Record as `is_baseline_driver_co_mutation = true` in a derived flag.
- A new primary tumor (separate radiographic site with no shared sensitizing mutation) — this is not transformation. Adjudicate per §4.

**Edge cases:**
- **Mixed histology in the rebiopsy** (e.g., adenocarcinoma + small-cell components together): label as `small_cell_transformation`. A single transformed component is sufficient.
- **Neuroendocrine markers positive but H&E morphology ambiguous:** adjudicate (§4). Default to `inconclusive`.
- **Squamous transformation only (no neuroendocrine):** label as `small_cell_transformation` for v1 (the class name is a v1-scope abbreviation; the schema enum value covers all phenotypic transformations). Add `transformation_subtype = "squamous"` in a derived field. Re-evaluate splitting the class for v2.
- **Transformation diagnosed >18 months after osimertinib stop:** check whether the patient remained on osimertinib throughout. If the patient switched to a different regimen and transformed on that, this is NOT an osimertinib-resistance event — exclude from cohort or recode.

### 1.5 "Unknown" vs "Other" — the disposition of patients without a §5.3 hit

This is a labeling design decision, not a biological one, and it affects every Task C number we report.

The DATASET_SPEC §5.3 enum has a single `other` bucket that holds both:
- (a) Patients with an identified §5.3-eligible alteration that doesn't match the four named classes (e.g., BRAF V600E, KRAS G12X, PIK3CA hotspots, FGFR amp, NRG1/RET/ALK/ROS1/NTRK fusions, EGFR L718Q/L792X/G724S, emergent exon 20 insertions).
- (b) Patients with a confirmed Criterion-A/C/E resistance event (per §6) but NO mechanism-eligible alteration detected within ±60 days.

**Labeling decision for v1:** *split these two at labeling time, lump them only at schema time.*

- Patients in (a) → labeled `other_identified` internally, mapped to `other` at the storage layer.
- Patients in (b) → labeled `unknown` internally, mapped to `other` at the storage layer.

The split is preserved in a sidecar field `mechanism_label_subtype` (added to outcomes via a v1.0.1 minor schema bump — explicit pre-freeze decision: this is acceptable as an additive minor bump per DATASET_SPEC §13).

Reporting at paper time: we report per-mechanism F1 across the five named classes (`C797S`, `MET_AMP`, `HER2_AMP` [retained from DATASET_SPEC even though not in user's task brief], `EGFR_AMP`, `small_cell_transformation`) and one combined `other`. We also report the (a)/(b) split as a secondary analysis in §10.

**Rationale:** Lumping (a) and (b) inflates the `other` class and makes the model look better at classifying "the long tail" than it actually is. Splitting at labeling preserves the audit trail; lumping at storage keeps the headline F1 honest.

> **Stake-in-the-ground claim for §1.5:** *The DATASET_SPEC's single `other` bucket is correct for storage; this document's (a)/(b) split is correct for labeling. Both views are produced from the same labeling pass.*

---

## 2. Cross-Cutting Edge Cases

### 2.1 Low-VAF variants

For each ctDNA call, three regimes:

| VAF range | C797S call | Off-target SNVs (`other`) |
|---|---|---|
| ≥ 0.5% | Detected | Detected |
| 0.1% – 0.5% | Detected (single draw) IF depth ≥ 1000× AND assay LoD claim ≤ 0.1% | Detected (single draw) IF the variant was detected in ≥ 2 draws OR is an established hotspot (BRAF V600E, PIK3CA E545K/H1047R, KRAS G12X) |
| 0.05% – 0.1% | Inconclusive — log and do not label | Inconclusive — log and do not label |
| < 0.05% | Not detected | Not detected |

The 0.1% floor for single-draw C797S calls is stricter than for off-target SNVs because false-positive C797S would inflate the most clinically distinctive class. We err conservative on the on-target class specifically.

### 2.2 Co-occurring mechanisms

In ~20–30% of post-osimertinib resistance cases (Chmielecki 2023; Choudhury 2022), more than one §5.3 mechanism is detectable at progression. The labeling rule:

1. **Dominant mechanism** is assigned per §3 hierarchy and stored in `outcomes.mechanism_at_progression`.
2. **All secondary mechanisms** above their respective detection thresholds are stored in `outcomes.secondary_mechanisms` (list, schema v1.0.1 additive field).
3. **Patients with >1 mechanism are flagged** `is_polyclonal_resistance = true`. This is a feature the model can use AND a stratification variable for the §9 results tables.

### 2.3 Pre-existing subclones vs. emergent

A variant is "emergent" iff it is detected at threshold post-osimertinib AND not detected at threshold (using the SAME assay class) pre-osimertinib. If the pre-osimertinib assay was less sensitive than the post-osimertinib assay (e.g., lower depth, smaller panel), the variant gets `is_emergent_inferred = true` with a confidence flag, and the labeling proceeds — but the secondary analysis in §10 stratifies by `is_emergent_confirmed` (paired assay class) vs `is_emergent_inferred`.

### 2.4 Ambiguous transformations

Adjudicate per §4 ladder. Default for any ambiguous case is `inconclusive`. Specific recurring ambiguities:

- **Mixed-histology baseline:** as above (§1.4), this disqualifies transformation as resistance. Record in `_meta.json.labeling_log` with patient ID.
- **Neuroendocrine markers focal-positive only** (e.g., synaptophysin patchy, chromogranin negative): `inconclusive` for transformation. Two markers required.
- **Squamous transformation alongside ongoing adenocarcinoma response on imaging:** rare; treat as transformation (§1.4), but flag for §10 ablation.

### 2.5 Assay heterogeneity

A patient may have ctDNA over time from multiple vendors (e.g., Guardant at one center, Foundation Liquid at another). Rule:

- The detection threshold is applied per-draw using the assay-of-origin's reported LoD.
- The cross-draw analysis (e.g., VAF trajectory for §6 Criterion C) uses only draws from the same vendor when computing nadirs and resurgences. Cross-vendor comparisons of VAF magnitude are forbidden — vendors don't normalize identically.
- If a patient has ctDNA from multiple vendors over time, parsers emit per-vendor sub-trajectories and the feature pipeline (downstream) decides whether to concatenate or model separately.

### 2.6 Timing ambiguity

When the index event date is uncertain (e.g., a PD scan was at day 180 ± 30 days), use the midpoint and set `event_date_precision = "fortnight"`. Flag for §10 sensitivity analysis. Patients with `event_date_precision` worse than "month" are excluded from the Task B (time-to-resistance MAE) headline but retained for Tasks A and C.

---

## 3. Dominant-Mechanism Hierarchy (when ≥2 are co-detected)

When a patient meets detection thresholds for more than one §5.3 mechanism at the index event window (±60 days), the **dominant** mechanism is assigned by the following ordered rules. Apply rules in order and stop at the first one that resolves.

### 3.1 Rule H1 — Histologic transformation supersedes molecular mechanisms

If `small_cell_transformation` is detected per §1.4, it is the dominant mechanism regardless of any co-occurring molecular alterations. Rationale: transformation is the clinical event that drives treatment redirection (chemo + IO instead of next-line TKI); reviewers will reject a labeling scheme that hides it behind, e.g., a co-occurring C797S.

### 3.2 Rule H2 — On-target supersedes off-target

Among the molecular classes, `EGFR_C797S` supersedes `MET_AMP`, `HER2_AMP`, `EGFR_AMP`, and `other`. Rationale: C797S directly invalidates the drug-target binding, which is mechanistically primary even when MET amp clones are also present.

### 3.3 Rule H3 — Among off-target alterations, highest clonal fraction wins

Compare clonal fractions (CF), defined per alteration:
- For SNVs/indels: CF = VAF / (overall tumor fraction at the draw).
- For amplifications: CF = (CN − 2) / (overall tumor-fraction-corrected expected gain), bounded to [0, 1]. Use the vendor-reported tumor fraction where available; if not available, use 1.0 (treats CF as VAF / CN-deviation directly).

Highest CF wins. This favors clonal mechanisms over subclonal ones — the model's job is to predict the dominant clone, not a long-tail subclone.

### 3.4 Rule H4 — Tie-break by therapeutic actionability

If two mechanisms tie on CF (within 5% relative), prefer the one with higher therapeutic actionability:
- MET amp (multiple approved/late-stage MET inhibitors and bispecifics) > EGFR amp > HER2 amp > other.

Rationale: ties are rare; when they occur, the label that most affects clinical decision-making is the more useful one for benchmark purposes.

### 3.5 Rule H5 — If still tied, prefer the one with the earlier detection date

The mechanism whose detection date is earlier (within the ±60-day window) wins. Defensible because earlier detection is closer to the resistance-initiating event.

### 3.6 Worked example

Patient X has at index event:
- MET amp on ctDNA, CN = 8.5, vendor positive, draw day 540, plasma TF = 12%.
- EGFR amp on ctDNA, CN = 7.2, sensitizing-allele VAF 38%, draw day 540, plasma TF = 12%.
- No C797S, no transformation.

H1: no transformation. H2: no C797S. H3:
- MET CF ≈ (8.5 − 2) / [expected gain at TF 12%] ≈ 0.78.
- EGFR CF ≈ (7.2 − 2) / [same] ≈ 0.62.
- MET wins. Label: `MET_AMP`. Secondary: `EGFR_AMP`. `is_polyclonal_resistance = true`.

---

## 4. Adjudication Ladder (for ambiguous calls)

Every ambiguous call goes through this ladder. The ladder is the audit-defensible alternative to "we discussed it and decided."

### 4.1 Tier 1 — Rule-based auto-resolution

The parser applies §1, §2, §3. If the rules resolve, label is final.

### 4.2 Tier 2 — Human label (single annotator)

If parser returns `inconclusive` or flags an unresolved tie, a single annotator (Abhijoy in v1) reviews the relevant rows and makes a call. The annotator MUST record:
- The specific rule that triggered the ambiguity.
- The conservative-default option.
- The annotator's call and the one-line reason.

### 4.3 Tier 3 — Dual annotator with discrepancy resolution

For 10% of all human-labeled cases (sampled uniformly random per index event), a second annotator (Aarchi in v1) independently labels the case blinded to Abhijoy's call. Discrepancies are resolved by:
1. The two annotators discuss; if they reach consensus, the consensus call is recorded with both signatures.
2. If no consensus, the case is escalated to Tier 4.

We report Cohen's κ between the two annotators in the paper.

### 4.4 Tier 4 — External oncologist review

Unresolved Tier-3 cases go to an external oncologist for an arbitration call. For v1, the candidates are: Jayant Gadrey (Tufts oncologist, in our network) or a Tata Memorial collaborator. Each escalated case is reviewed in a 15–30 minute consult; the external reviewer's call is final.

### 4.5 Tier 5 — Last resort

If no external reviewer can adjudicate within 7 days, the case is recorded as `mechanism_at_progression = "unknown"` and the patient is excluded from the Task C training set (but retained for Tasks A and B). Document the exclusion in `_meta.json.labeling_log`.

---

## 5. Audit Log Requirements

Every parser MUST emit a `labeling_log.jsonl` (one JSON object per line) alongside `_meta.json` with the following fields:

| Field | Type | Notes |
|---|---|---|
| `patient_id` | str | FK |
| `outcome_id` | str | The index event being labeled (if applicable) |
| `tier_applied` | enum | `1_rule` / `2_single` / `3_dual` / `4_external` / `5_excluded` |
| `rules_fired` | list[str] | E.g., `["H1", "H3"]` |
| `mechanism_called` | str | Final label |
| `mechanism_label_subtype` | str | `"on_target_C797"`, `"other_identified_BRAFV600E"`, `"unknown_no_mechanism"`, etc. |
| `secondary_mechanisms` | list[str] | All thresholded mechanisms not chosen as dominant |
| `is_polyclonal_resistance` | bool | |
| `annotator_id` | str | `"parser"` for Tier 1; named annotator for ≥Tier 2 |
| `annotator_reason` | str | ≤200 chars; for Tier ≥2 |
| `external_reviewer_id` | str | For Tier 4 only |
| `timestamp` | ISO 8601 | When the label was recorded |

This log is checked into the repo (with all patient IDs replaced by their `{source_prefix}:{stable_source_id}` form — never raw identifiers).

---

## 6. Inter-Rater Reliability — Target & Reporting

- **Target Cohen's κ on the 10% dual-annotated subset: ≥ 0.80** for the §5.3 named classes; ≥ 0.70 if `other` is included. These are aspirational; lower κ values mean the spec is too loose somewhere and we revise.
- Report κ in the paper Methods section, broken down per class. A class with κ < 0.60 forfeits its headline F1 and is reported as "low-reliability — interpret with caution" instead.
- For C797S and MET_AMP specifically, target κ ≥ 0.90 — these are the easiest calls and below that suggests a spec bug.

---

## 7. Worked Examples

### 7.1 Example A — Clean C797S

- Day 0: osimertinib start.
- Day 450: ctDNA shows EGFR L858R VAF 4.2%, no C797S, MET CN 2.1, plasma TF 18%, depth 4800×.
- Day 620: ctDNA shows EGFR L858R VAF 11.3%, EGFR C797S VAF 2.8%, MET CN 2.3, plasma TF 25%, depth 4200×.
- Day 645: RECIST PD confirmed at day 670.

Labeling: C797S detected per §1.1 (single draw, VAF 2.8% > 0.1%, depth > 1000×, no pre-osi C797S detection). Index event = day 645 (RECIST PD per §6 Criterion A, confirmed day 670). `mechanism_at_progression = "EGFR_C797S"`. `is_polyclonal_resistance = false`. Tier 1 rule.

### 7.2 Example B — C797S + MET co-occurrence

- Day 0: osi start. Day 500: ctDNA shows EGFR L858R VAF 8%, C797S VAF 0.4%, MET CN 8.1, plasma TF 22%, depth 3500×. Day 510: RECIST PD confirmed.

Labeling: Both C797S (§1.1 met) and MET_AMP (§1.2 met) detected. Apply hierarchy §3. No transformation (H1 not triggered). H2: C797S wins. `mechanism_at_progression = "EGFR_C797S"`. `secondary_mechanisms = ["MET_AMP"]`. `is_polyclonal_resistance = true`. Tier 1.

### 7.3 Example C — Ambiguous transformation

- Day 0: osi start. Day 720: RECIST PD. Day 730: rebiopsy shows "adenocarcinoma with focal neuroendocrine differentiation"; synaptophysin patchy, chromogranin negative; H&E morphology equivocal. ctDNA shows no §5.3 SNV/CNV at threshold.

Labeling: Tier 1 — §1.4 transformation rule requires ≥2 neuroendocrine markers positive. Only synaptophysin patchy → does NOT meet threshold. Tier 1 returns `inconclusive` for transformation. ctDNA has no mechanism. Escalate to Tier 2: Abhijoy reviews; cannot resolve. Tier 3: Aarchi blinded label is `unknown`. Consensus reached. Final label: `unknown`. Logged. Excluded from Task C training, retained for Tasks A and B.

### 7.4 Example D — Low-level MET gain

- Day 600: ctDNA MET CN 5.2, vendor flag negative (below their threshold), plasma TF 8%.

Labeling: §1.2 fails (CN < 6 and vendor flag negative). MET_AMP NOT detected. The patient may still get another mechanism; if none, `unknown`. Tier 1.

### 7.5 Example E — Pre-existing EGFR amp

- Baseline tissue: EGFR L858R VAF 35%, EGFR CN 7.2, MET CN 2.0.
- Day 600: ctDNA EGFR L858R VAF 12%, EGFR CN 8.0, plasma TF 18%. Day 610: RECIST PD.

Labeling: §1.3 disqualifier — EGFR amp present at baseline. EGFR_AMP NOT labeled as resistance. `is_baseline_driver_co_mutation = true` for EGFR amp. Patient still has a Criterion-A resistance event but no §5.3 mechanism → `mechanism_at_progression = "unknown"` with `mechanism_label_subtype = "unknown_no_mechanism"`. Tier 1.

### 7.6 Example F — Squamous transformation

- Day 0: osi start. Day 800: RECIST PD. Day 810: rebiopsy shows squamous histology in 80% of the sample; p40 strongly positive; retains EGFR L858R by paired sequencing.

Labeling: §1.4 covers squamous transformation. `mechanism_at_progression = "small_cell_transformation"`. `transformation_subtype = "squamous"`. Tier 1.

---

## 8. Things This Spec Deliberately Does NOT Decide

These are flagged for Aarchi to call before freeze. Defaults below; flag if you want to override.

| # | Decision | Default | Why flagged |
|---|---|---|---|
| 1 | Should ctDNA-only RB1/TP53-LOF co-occurrence call transformation in v1? | **No** (require tissue) | Risk of false-positives from incidental LOF in non-transformed tumors. |
| 2 | Should C797G be merged into the C797S label class, or split? | **Merged** (one class) | Statistical power; biology is shared. |
| 3 | Should squamous transformation be merged with neuroendocrine transformation? | **Merged for v1** (record subtype) | Sample size; revisit in v2. |
| 4 | Should L718Q, L792X, G796X, G724S be merged into a single "rare on-target" class for v1? | **No** — keep in `other` | Sample size doesn't support a separate class yet. |
| 5 | What's the dual-annotation rate? 10% as proposed or higher? | **10%** | Practical effort vs. confidence; 10% gives reasonable κ CIs at our n. |
| 6 | Who is the Tier 4 external arbiter for v1? | Jayant Gadrey (first ask), Tata Memorial PI (second) | Need to confirm willingness. |
| 7 | Should we include HER2 amp as a named class (DATASET_SPEC has it; user's task brief omitted)? | **Yes, keep** | Mechanistic story is clean; ~3% of cases; we can always collapse it to `other` at reporting time if F1 is too noisy at small n. |
| 8 | What is the policy for patients with NO post-osimertinib ctDNA AND NO rebiopsy but with RECIST PD? | **`unknown`** | They satisfy Criterion A but provide no mechanism evidence. |
| 9 | Should we re-call vendor amplifications by re-thresholding raw copy numbers, or trust vendor flags? | **Trust vendor flag + record raw CN** | Re-calling raw data is out of scope per DATASET_SPEC §15. |
| 10 | Sidecar field `mechanism_label_subtype` — additive minor schema bump (v1.0.1)? | **Yes, ship the bump with the labeling pass** | Cleanest way to preserve the (a)/(b) split. |

---

## 9. What to Hand to Aarchi for Review

Aarchi should specifically pressure-test:

1. **§1.4 transformation rule** — the "paired sequencing of transformed sample" requirement. Is it too strict? Will it exclude many real cases?
2. **§3 hierarchy** — does the on-target-supersedes-off-target choice (H2) match clinical reality? Some MTBs would call this differently.
3. **§3.3 clonal-fraction tie-break** — the math is reasonable but unvalidated against published examples; she should sanity-check against the FLAURA/AURA3 supplements once we have them.
4. **§5 audit log fields** — anything missing she'd want to see during a paper rebuttal?
5. **§8 decisions #1, #3, #7** — these are the most-contested.
6. **§6 IRR targets** — are ≥0.80 / ≥0.90 realistic targets for these labels? She has more pathology experience here.

Estimated review time: 60–90 minutes. After her redlines land, this document gets a `v1.0-frozen` tag and any subsequent change requires a §13-equivalent minor/major bump.

---

## 10. Freeze Checklist

Before declaring this document frozen at `LABELING_GUIDELINES/1.0`:

- [ ] Aarchi has signed off on §1–§3, §8 decisions #1, #3, #7 explicitly.
- [ ] Jayant Gadrey (or replacement) has confirmed willingness to serve as Tier 4 arbiter.
- [ ] One end-to-end worked example per §1 mechanism is labeled by hand from a real published paper (FLAURA supplement) and matches what this spec predicts.
- [ ] §5 audit log schema is added to `oncotraj/schemas.py`.
- [ ] §1.5 `mechanism_label_subtype` sidecar field is added to DATASET_SPEC as a v1.0.1 minor bump.
- [ ] Dual-annotation κ on a 20-patient pilot is ≥0.75 across the named classes (this is the in-house gate before we trust the rest of the spec).

---

*End of labeling guidelines. Hand this to Aarchi with §9 highlighted; do not freeze until §10 is checked through.*
