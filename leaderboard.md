# OncoTraj v1 Leaderboard

**Cohort.** v1 comprises **813 EGFR-mutant NSCLC patients on first-line
osimertinib**, harmonized from three real-world clinical-genomic sources:
MSK-CHORD (672), the FLAURA molecular-resistance supplement (107), and the
AACR Project GENIE BPC NSCLC cohort (34). Patient-level test split: n=122
overall (Task A landmark-evaluable n=110, Task B/C event-observed n=85).
Headline results below are computed with bootstrap 95% CIs (oncotraj-eval
schema `1.1.0`).

**Task A** is the **12-month landmark progression** task. The most defensible
estimate — within a single source (MSK-CHORD only), so dataset membership
cannot leak into the label — is logistic AUC **0.596 [0.478, 0.714]**, whose
CI crosses 0.50: **within-source, Task A is not distinguishable from chance**
at this cohort size. The higher mixed-source figure (logistic AUC 0.680
[0.581, 0.781], no source flags) is **confounded by cross-source structure**:
the date-less FLAURA subset is landmark-negative by construction, so part of
that discrimination reflects which dataset a patient came from rather than
their biology. We therefore do not claim Task A is learnable above chance.
The associated signal — TP53 co-mutation (59% vs 29% 12-month progression
cohort-wide, 60% vs 42% within MSK-CHORD) — is consistent with known
prognosis and is suggestive, but at v1's sample size it does not yield
above-chance within-source discrimination. **Task B** (time-to-progression
regression) is the closest thing to a signal, but on clean within-source
evaluation it sits near the majority floor (~10 d MAE gap, ranking near
chance); the **252 d / C-index 0.66** mixed-source figure is
cross-source-confounded (FLAURA source one-hot) — see the v1.1 cross-source
section below. **Task C** (per-mechanism classification) is a structural
non-result — tissue NGS under-captures EGFR C797S (~1.6% of MSK-CHORD,
single-positive in the test fold), so a serial-ctDNA-enriched cohort is a
precondition for mechanism prediction; reported only for completeness.

Auto-generated from `src/oncotraj/eval`. Submissions are evaluated against
the locked test split via `oncotraj-eval --predictions <csv> --split test`
and dropped as JSON into `eval_reports_v1/`. Run `oncotraj-eval --refresh`
(or call `oncotraj.eval.update_leaderboard(...)` programmatically) to
regenerate the table below.

Default sort: **Task A ROC-AUC on test**, descending. Secondary sort:
**Task C macro F1**.

<!-- LEADERBOARD START -->
<!-- Auto-generated; do not edit between markers. -->

### Split: `test`

| Submission | n | Task A AUC | Task A Brier | Task A ECE | Task B MAE (d) | Task B C-index | Task C macro F1 | Task C accuracy | Generated |
|---|---|---|---|---|---|---|---|---|---|
| `logistic_taskA` | 110 | 0.680 | 0.220 | 0.071 | — | — | — | — | 2026-05-21T18:44:10 |
| `random_forest_taskA` | 110 | 0.678 | 0.214 | 0.041 | — | — | — | — | 2026-05-21T18:44:11 |
| `xgboost_taskA` | 110 | 0.630 | 0.255 | 0.159 | — | — | — | — | 2026-05-21T18:44:14 |
| `majority_taskA` | 110 | 0.500 | 0.246 | 0.017 | — | — | — | — | 2026-05-21T18:44:09 |
| `lstm_taskC` | 83 | — | — | — | — | — | 1.000 | 1.000 | 2026-05-15T07:36:00 |
| `logistic_taskC` | 85 | — | — | — | — | — | 0.517 | 0.835 | 2026-05-21T18:44:14 |
| `majority_taskC` | 85 | — | — | — | — | — | 0.497 | 0.988 | 2026-05-21T18:44:14 |
| `xgboost_taskC` | 85 | — | — | — | — | — | 0.491 | 0.965 | 2026-05-21T18:44:18 |
| `random_forest_taskC` | 85 | — | — | — | — | — | 0.488 | 0.953 | 2026-05-21T18:44:15 |
| `logistic_taskB` | 85 | — | — | — | 260.9 | 0.581 | — | — | 2026-05-21T18:44:14 |
| `lstm_taskB` | 83 | — | — | — | 280.9 | 0.568 | — | — | 2026-05-15T07:35:56 |
| `majority_taskB` | 85 | — | — | — | 344.3 | 0.500 | — | — | 2026-05-21T18:44:14 |
| `random_forest_taskB` | 85 | — | — | — | 251.8 | 0.656 | — | — | 2026-05-21T18:44:14 |
| `xgboost_taskB` | 85 | — | — | — | 265.7 | 0.643 | — | — | 2026-05-21T18:44:17 |

_No submissions on the `val` split yet._

<!-- LEADERBOARD END -->

> **Read this table with care — the mixed-source numbers are confounded.**
> The Task B headline above (random forest MAE **252 d**, C-index **0.656**;
> logistic 261 d) is computed on the **mixed-source** test split, where a
> model can exploit the `source_dataset` one-hot: FLAURA patients carry a
> constant pseudo-PFS (910 d) the model predicts exactly once it knows the
> source. The same confound inflates Task A (see the cohort note above).
> **The defensible estimates are the within-source v1.1 numbers below**
> (Cross-source generalization §): on clean within-source evaluation the
> best Task B MAE is 293 d vs a 303 d majority floor, and C-index is at
> chance. Treat the table above as an upper bound, not the result.

## Cross-source generalization (v1.1)

**Headline finding: on clean within-source evaluation with v1's snapshot
features, no task shows clear above-chance signal.** Binary 12-month
progression (Task A) is near-degenerate and within-source not distinguishable
from chance; time-to-progression (Task B) regression sits essentially at the
majority floor with at-chance ranking; mechanism classification (Task C) is
sampling-bound. The only thread is Cox's hazard ranking (in-distribution
C-index 0.541, cross-source 0.597), both with CIs crossing 0.50 — suggestive,
not significant. **This is the central evidence that the ceiling is the data
modality — single-timepoint tissue/ctDNA snapshots — not the model.** Serial
ctDNA trajectories, not a better algorithm, are the precondition for
above-chance performance. That is a real, reportable result.

The headline table above evaluates on a stratified test split that
contains patients from all three sources. **v1.1** adds a stricter
distribution-shift evaluation for Task B (time-to-progression): every
model is trained on **MSK-CHORD train-split patients only** (n=471) and
evaluated on the held-out sources. Reports live in
`eval_reports_v1/v11_<model>_<train>_<eval>.json`; the headline figure is
`paper/figures/figure_5_cross_source.pdf`. Cox PH (lifelines) is added as
a fifth Task B baseline alongside the regressors; it reuses the leak-safe
`build_features` pipeline (extended audit in
`tests/test_taska_feature_leakage.py::test_cox_uses_only_leak_safe_features`
and `::test_cox_fit_is_unchanged_by_pre_filter`).

> **Caveat (leakage audit scope).** The leakage tests currently run against
> the **synthetic fixture**, not the real MSK-CHORD v1 tables. The no-leakage
> *guarantee* is therefore proven on synthetic data only.
> **TODO:** run the leakage audit against the real `data/processed/oncotraj_v1/`
> tables before any external release.

**Train: MSK-CHORD (n=471, train split). Evaluate:**

| Model | In-distribution: MSK-CHORD test (n=101 · 72 events) | Cross-source: GENIE BPC (n=34 · 14 events) | Cross-source: FLAURA supp.† (n=107 · 73 "events") |
|---|---|---|---|
| Majority floor    | MAE **303 d** [227, 386] · C-idx 0.500 | MAE **97 d** [68, 127] · C-idx 0.500 | MAE 730 d · — |
| Logistic          | MAE 301 d [239, 373] · C-idx 0.492 [0.42, 0.57]  | MAE 119 d [78, 160]  · C-idx 0.424 [0.28, 0.57] | MAE 532 d · — |
| Random Forest     | MAE 293 d [232, 364] · C-idx 0.432 [0.36, 0.51]  | MAE 131 d [92, 174]  · C-idx 0.513 [0.37, 0.69] | MAE 523 d · — |
| XGBoost           | MAE 312 d [246, 387] · C-idx 0.441 [0.36, 0.52]  | MAE 190 d [135, 248] · C-idx 0.543 [0.38, 0.72] | MAE 528 d · — |
| **Cox PH (v1.1)** | MAE 513 d [444, 584] · **C-idx 0.541** [0.46, 0.62] | MAE 429 d [330, 535] · **C-idx 0.597** [0.45, 0.75] | MAE 201 d · — |

Bootstrap 95% CIs from 1000 test-set resamples. **Best non-trivial value per column in bold.**

**Three honest findings from v1.1:**

1. **On clean within-source evaluation, Task B does not show clear above-chance signal either.** Within MSK-CHORD only (no source-flag confound, no FLAURA constant-target inflation), the best MAE is **293 d** (random forest) — but the majority floor is **303 d** [227, 386], so the gap is only ~10 days and the CIs almost fully overlap (RF 293 [232, 364]). And the *ranking* is at chance: within-source C-index is **0.432 (RF)**, **0.492 (logistic)**, **0.441 (XGBoost)** — all at or below 0.50. So the mixed-source headline (RF MAE 252 d, C-index 0.656) was buoyed by FLAURA test patients predicted exactly via the source one-hot; strip that out and Task B is essentially at the floor. The only thread is **Cox's hazard ranking** (in-distribution C-index 0.541 [0.46, 0.62], cross-source 0.597 [0.45, 0.75]) — both CIs cross 0.50, so suggestive, not significant. The v1.1 in-distribution column is the most defensible Task B evaluation, and what it shows is a near-chance result.
2. **MSK → GENIE BPC: MAE generalization collapses; hazard ranking is suggestive but not yet significant.** Every learned model underperforms the majority floor on MAE (best learned 119 d vs majority 97 d), because the MSK training distribution (std 331 d) overshoots GENIE's narrower one (std 110 d). Cox PH reaches **C-index 0.597 [0.45, 0.75]** on the full GENIE cohort — better point estimate than the regressors, but its **CI crosses 0.50**, so cross-source hazard ranking is only **suggestive at n=34 and should tighten with more data**, not an established transfer result. We do not claim ranking "survives" the shift.
3. **MSK → FLAURA is a label artifact, not a generalization signal.** Every FLAURA patient has the same harmonization-time pseudo-PFS (910 d, the FLAURA-arm median), so MAE is dominated by the offset between the MSK-trained point prediction and that constant; C-index is undefined (no comparable pairs). FLAURA must move out of Task B in v2 unless individual progression dates become available. The figure marks FLAURA bars with a hatched pattern as a visual warning.

†FLAURA's `mae_ci` collapses to `[X, X]` because every resampled bootstrap draw on a constant target gives the same MAE — that is geometry, not noise.

## How to submit

1. Train a model on the OncoTraj v1 cohort (see `data/processed/oncotraj_v1/`).
2. Write predictions to a CSV conforming to the schema in
   `src/oncotraj/eval/report.py` (`SCHEMA_DOC`).
3. Run `oncotraj-eval --predictions yours.csv --split test \
       --submission-id <your-name> --output eval_reports_v1/<your-name>.json`.
4. Rerun with `--leaderboard leaderboard.md --reports-dir eval_reports_v1` to update this file.
