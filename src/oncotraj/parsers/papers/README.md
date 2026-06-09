# Per-study supplement adapters

One Python file per published study. Each adapter takes either:
1. **PDF supplements** the study published, extracted via `pdfplumber` through
   the `extract_pdf_tables` helper, or
2. **Hand-transcribed CSVs** when the PDF tables can't be auto-parsed
   (multi-page nested headers, scanned images, merged cells, etc.),

and emits the standard four-table `ParsedTables` contract.

## Convention: where files live

```
<papers_root>/<study_id>/
  manual/                # hand-transcribed CSVs (preferred for fidelity)
    cohort.csv
    variants.csv
    ...
  pdf/                   # original journal supplements
    nejm_2024_flaura_supp_tableS3.pdf
    ...
```

Add `papers_root` outside the repo (e.g. `data/raw/papers/`) so PDFs are never
committed. CSV transcriptions can live anywhere — if they're free of patient
identifiers they may be committed; if not, keep them out of git.

## Writing a new adapter

```python
# src/oncotraj/parsers/papers/black_2025_tracerx_evo.py
from oncotraj.schemas import SourceDataset
from ..common import ParsedTables
from . import register
from ._base import PaperAdapter, read_manual_csv


@register
class BlackTracerxEvoAdapter(PaperAdapter):
    study_id = "black_2025_tracerx_evo"
    source_dataset = SourceDataset.TRACERX
    citation = "Black JRM et al. Nature 2025."

    def load(self) -> ParsedTables:
        # 1. read manual_csvs() and/or pdfs()
        # 2. for each patient row, build patient/variant/treatment/outcome dicts
        # 3. return ParsedTables(patients=..., variants=..., ...)
        ...
```

Don't forget to add the module to the import list at the bottom of
`papers/__init__.py` so the registry sees it:

```python
from . import black_2025_tracerx_evo  # noqa: F401
```

## Hard rules (from DATASET_SPEC.md §0)

- **No silent imputation.** If a required field can't be filled for a row,
  exclude the patient with `included_in_v1_cohort=False` and an
  `exclusion_reason`. Never make up demographics, dates, or VAFs.
- **Source-prefixed patient IDs:** every patient_id is
  `<source_dataset>:<study_id>.<stable_id>`. This guarantees no collision with
  GENIE/CHORD patient IDs and lets the splits stage do entity resolution
  later.
- **Provenance:** if the row corresponds to a specific table/figure in the
  paper, note it in a comment in the adapter. Future-you and reviewer #3 will
  thank you.

## Manual CSV transcription tips

- One CSV per logical table; don't merge across tables.
- Column headers match the adapter's documented schema exactly
  (case-sensitive).
- Dates: ISO 8601 (`YYYY-MM-DD`) or day-offsets from a documented anchor.
  Document the anchor in the adapter's module docstring.
- Empty cell = unknown. Don't write "NA", "Unknown", "—".
- Save as UTF-8 with no BOM.

## When to give up on PDF extraction

You should switch to a manual CSV the moment you see any of:
- Tables that span multiple pages with mid-page headers.
- Cells containing line breaks (pdfplumber will split them into junk rows).
- Multi-level column headers (more than one header row).
- Image-based PDFs — `pdfplumber.extract_tables()` returns empty lists.
- More than ~10 manual corrections per page after auto-extraction.

A clean hand-transcription beats an auto-extraction with hidden errors. The
manual route is the default for ELIOS/AURA3-style supplements; expect to
auto-extract maybe 20–30% of PDFs.

## Cross-study deduplication

We do **not** deduplicate at parse time. Patients who appear in two studies
keep two rows under their respective source-prefixed IDs. The splits stage
will handle entity resolution before train/val/test assignment.
