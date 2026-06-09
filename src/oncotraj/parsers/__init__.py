"""Per-source parsers. One module per `SourceDataset` enum value.

Each parser exposes `parse(raw_dir) -> ParsedTables` and emits the four
in-memory DataFrames matching DATASET_SPEC.md §2-§5. The build script
(`scripts/build_dataset.py`) is responsible for writing them to Parquet.
"""

from . import cbioportal, chord, genie, genie_bpc, synthetic
from .common import ParsedTables, validate_tables

__all__ = [
    "ParsedTables",
    "cbioportal",
    "chord",
    "genie",
    "genie_bpc",
    "synthetic",
    "validate_tables",
]
