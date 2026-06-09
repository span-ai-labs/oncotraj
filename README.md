# OncoTraj

**A public benchmark for longitudinal resistance prediction in EGFR-mutant
non-small-cell lung cancer (NSCLC) on first-line osimertinib.**

OncoTraj harmonizes real-world clinical-genomic data into a single schema with
frozen, leakage-audited splits, and ships reproducible baselines and an
evaluation harness. Its purpose is to establish a public floor — and an honest
ceiling — for what snapshot (single-timepoint) features can predict about
acquired resistance.

> **Headline finding (read this first).** On clean *within-source* evaluation,
> snapshot-feature models do **not** beat chance on any of the three tasks at
> this cohort size. The binding constraint is the **data modality** —
> single-timepoint tissue/ctDNA snapshots — **not the model**. Serial ctDNA
> trajectories, not a better algorithm, are the precondition for above-chance
> performance. That negative result is the contribution: it tells the field
> where the signal must come from. See [`leaderboard.md`](leaderboard.md) for
> the full, CI-annotated numbers and the cross-source (v1.1) analysis.

## Cohort

v1 comprises **813 EGFR-mutant NSCLC patients on first-line osimertinib**,
harmonized from three real-world clinical-genomic sources:

| Source | n | Access |
|---|---|---|
| MSK-CHORD | 672 | cBioPortal (open) |
| FLAURA molecular-resistance supplement | 107 | Public journal supplements |
| AACR Project GENIE BPC (NSCLC) | 34 | Registered access |

Patient-level test split: n=122 (Task A landmark-evaluable n=110; Task B/C
event-observed n=85).

## Tasks

- **Task A — 12-month landmark progression** (binary classification).
- **Task B — time-to-progression** (regression; reported with MAE and C-index).
- **Task C — resistance mechanism** (6-class classification).

## Baselines

Majority/floor, logistic regression, random forest, XGBoost, LSTM, a small
Transformer, and a Cox proportional-hazards model (Task B). All baselines reuse
a single leak-safe feature pipeline; the leakage audit lives in
`tests/test_taska_feature_leakage.py`.

## Quickstart

```bash
uv sync
uv run pytest          # test suite, incl. leakage audits
uv run ruff check
```

Evaluate a submission against the locked test split:

```bash
uv run oncotraj-eval --predictions yours.csv --split test \
    --submission-id <your-name> --output eval_reports_v1/<your-name>.json
uv run oncotraj-eval --refresh   # regenerate leaderboard.md
```

The prediction CSV schema is documented in
`src/oncotraj/eval/report.py` (`SCHEMA_DOC`).

## Data

The harmonized dataset is **not redistributed in this repository**. The
underlying sources (AACR GENIE BPC, MSK-CHORD) carry Data Use Agreements that
restrict redistribution of patient-level data. Instead, OncoTraj ships the
**parsers and a build script** that regenerate the harmonized Parquet tables
from sources you obtain under their respective terms:

```bash
# after placing source files under data/raw/ (see docs/DATASET_SPEC.md and DATA.md)
uv run python scripts/build_dataset.py
uv run python scripts/build_splits.py
```

See [`DATA.md`](DATA.md) for how to obtain each source, and
[`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md) for the full harmonization
schema (one row per patient/variant/treatment/outcome, explicit missingness
sentinels, no silent imputation).

## Layout

```
src/oncotraj/      # library: schemas, parsers, splits, metrics, eval, baselines
scripts/           # build_dataset, build_splits, train_*, make_figures
tests/             # pytest suite, incl. leakage audits
eval_reports/      # baseline metric reports (JSON)
eval_reports_v1/   # v1 + v1.1 cross-source reports (JSON)
leaderboard.md     # auto-generated, CI-annotated results
docs/              # dataset spec and reproduction docs
data/, models/     # gitignored — built locally, not redistributed
```

## Citation

A preprint describing OncoTraj is available. Citation details will be added
here once the preprint is posted. In the meantime, please cite this repository.

## License

**Code: MIT** (see [`LICENSE`](LICENSE)). Data are not included; any data you
build with these tools remain governed by the upstream sources' terms (AACR
GENIE, MSK-CHORD, and the respective journal supplements).
