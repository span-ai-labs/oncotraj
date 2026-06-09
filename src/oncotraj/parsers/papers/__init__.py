"""Per-study supplement adapters.

One Python file per published study. Each module defines a single
`PaperAdapter` subclass and registers it with `register`. The build script
discovers all registered adapters when given `--papers-dir <root>`.
"""

from __future__ import annotations

from pathlib import Path

from ..common import ParsedTables
from ._base import PaperAdapter, PaperInputs, extract_pdf_tables, read_manual_csv

_REGISTRY: dict[str, type[PaperAdapter]] = {}


def register(adapter_cls: type[PaperAdapter]) -> type[PaperAdapter]:
    """Decorator: register an adapter so `load_all` can discover it."""
    if not adapter_cls.study_id:
        raise ValueError(f"{adapter_cls.__name__} has empty study_id; cannot register.")
    if adapter_cls.study_id in _REGISTRY:
        raise ValueError(f"Duplicate adapter registration for study_id={adapter_cls.study_id!r}.")
    _REGISTRY[adapter_cls.study_id] = adapter_cls
    return adapter_cls


def registered_study_ids() -> list[str]:
    return sorted(_REGISTRY)


def load_all(papers_root: Path) -> ParsedTables:
    """Run every registered adapter against its `<papers_root>/<study_id>/` dir.

    Adapters whose study directory is missing are skipped (with no row written),
    on the theory that not every user will have every supplement.
    """
    from ..common import ParsedTables as _PT

    empty = _PT(
        patients=__import__("pandas").DataFrame(),
        variants=__import__("pandas").DataFrame(),
        treatments=__import__("pandas").DataFrame(),
        outcomes=__import__("pandas").DataFrame(),
    )
    combined = empty
    for study_id, adapter_cls in _REGISTRY.items():
        study_dir = papers_root / study_id
        if not study_dir.exists():
            continue
        adapter = adapter_cls(PaperInputs(study_dir=study_dir))
        tables = adapter.load()
        combined = combined.concat(tables) if len(combined) else tables
    return combined


# Importing submodules registers their adapters via the decorator. Add new
# studies here as `from . import <study_id_module>  # noqa: F401`.
from . import (  # noqa: E402
    _example_synthetic,  # noqa: F401
    chmielecki_2023_aura3,  # noqa: F401
    chmielecki_2023_flaura,  # noqa: F401
    tan_2024_oscillate,  # noqa: F401
)

__all__ = [
    "PaperAdapter",
    "PaperInputs",
    "extract_pdf_tables",
    "load_all",
    "read_manual_csv",
    "register",
    "registered_study_ids",
]
