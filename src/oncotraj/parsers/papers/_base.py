"""Base classes for per-study supplement adapters.

Every published-study adapter subclasses `PaperAdapter` and implements `load()`.
The adapter is responsible for taking either (a) a PDF supplement we can extract
tables from automatically, or (b) a hand-transcribed CSV that mirrors the
paper's per-patient table, and emitting the standard `ParsedTables` four-table
contract.

Cross-source entity resolution is deferred to the splits stage (DATASET_SPEC.md
§2). Every patient is emitted under their source-prefixed `patient_id`; we do
not de-duplicate across studies at parse time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oncotraj.schemas import SourceDataset

from ..common import ParsedTables


@dataclass
class PaperInputs:
    """Where the adapter looks for its input files.

    Convention (see papers/README.md):
        <papers_root>/<adapter.study_id>/manual/*.csv     -- hand-transcribed
        <papers_root>/<adapter.study_id>/pdf/*.pdf        -- PDF supplements
    """

    study_dir: Path

    @property
    def manual_dir(self) -> Path:
        return self.study_dir / "manual"

    @property
    def pdf_dir(self) -> Path:
        return self.study_dir / "pdf"

    def manual_csvs(self) -> list[Path]:
        return sorted(self.manual_dir.glob("*.csv")) if self.manual_dir.exists() else []

    def pdfs(self) -> list[Path]:
        return sorted(self.pdf_dir.glob("*.pdf")) if self.pdf_dir.exists() else []


class PaperAdapter(ABC):
    """One adapter per published study. Subclasses set the three class attributes."""

    study_id: str = ""
    source_dataset: SourceDataset = SourceDataset.FLAURA_SUPP
    citation: str = ""

    def __init__(self, inputs: PaperInputs):
        if not self.study_id:
            raise ValueError(f"{type(self).__name__} must set class attribute `study_id`.")
        self.inputs = inputs

    @abstractmethod
    def load(self) -> ParsedTables:
        """Read the study's supplement files and emit ParsedTables.

        Implementations MUST NOT silently impute missing values. If a column the
        spec requires cannot be filled for a given row, exclude the patient
        with an `exclusion_reason` rather than guessing.
        """

    def make_patient_id(self, stable_id: str) -> str:
        return f"{self.source_dataset.value}:{self.study_id}.{stable_id}"


def extract_pdf_tables(pdf_path: Path) -> list[pd.DataFrame]:
    """Pull every detectable table from a PDF as a list of DataFrames.

    Uses pdfplumber's default settings. Adapters that need different settings
    (e.g., explicit `table_settings={"vertical_strategy": "text", ...}`) should
    call pdfplumber directly rather than this helper.
    """
    import pdfplumber

    dataframes: list[pd.DataFrame] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for raw in page.extract_tables() or []:
                if not raw or len(raw) < 2:
                    continue
                header, *body = raw
                cleaned_header = [(h or f"col_{i}").strip() for i, h in enumerate(header)]
                dataframes.append(pd.DataFrame(body, columns=cleaned_header))
    return dataframes


def read_manual_csv(path: Path) -> pd.DataFrame:
    """Load a hand-transcribed CSV with the framework's expected conventions.

    Conventions (see papers/README.md):
    - Column names are case-sensitive and match the adapter's documented schema.
    - Missing cells are empty strings, not 'NA' or 'unknown'; the adapter's row
      builder is responsible for converting them to schema-correct sentinels.
    - Dates are ISO 8601 (`YYYY-MM-DD`) or day-offsets from a documented anchor.
    """
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
