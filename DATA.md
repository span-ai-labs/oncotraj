# Obtaining the source data

OncoTraj does **not** redistribute patient-level data. AACR Project GENIE and
MSK-CHORD are released under Data Use Agreements that restrict redistribution,
so this repository ships the parsers and build script instead. Obtain each
source under its own terms, place the files under `data/raw/`, and run the
build.

## Sources

| Source | What to get | Where | Access terms |
|---|---|---|---|
| **MSK-CHORD** | NSCLC clinical + genomic tables | [cBioPortal](https://www.cbioportal.org/) (MSK-CHORD study) | Open download; cBioPortal terms of use |
| **AACR GENIE BPC (NSCLC)** | BPC NSCLC clinical-genomic release | [Synapse / AACR GENIE BPC](https://www.synapse.org/genie) | Free registration + click-through DUA |
| **FLAURA molecular-resistance supplement** | De-identified resistance tables from the published FLAURA supplements | Journal supplementary materials | Per publisher terms |

> The exact file names and column mappings each parser expects are documented
> in [`docs/DATASET_SPEC.md`](docs/DATASET_SPEC.md) and in the per-source
> parser modules under `src/oncotraj/parsers/`.

## Build

```bash
uv sync

# place obtained source files under data/raw/ per docs/DATASET_SPEC.md, then:
uv run python scripts/build_dataset.py     # -> data/processed/oncotraj_v1/*.parquet
uv run python scripts/build_splits.py      # -> frozen patient-level splits
uv run pytest                              # verify schema + leakage audits pass
```

The build emits a `_meta.json` per source recording row counts, schema version,
and every harmonization decision actually applied. Parsers never silently
impute: any field that cannot be filled gets an explicit missingness sentinel
(see `docs/DATASET_SPEC.md` §7).

## A note on data integrity

The leakage audit in `tests/test_taska_feature_leakage.py` currently runs
against a synthetic fixture; if you rebuild from real sources, re-run the audit
against your `data/processed/oncotraj_v1/` tables before reporting results.
