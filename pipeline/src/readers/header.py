"""Header-row detection and empty-sheet detection on raw ``header=None`` grids."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from .columns import normalise_column_name

_NO_RECORDS = re.compile(r"\bno\s+records?\b", re.IGNORECASE)
DEFAULT_MAX_DEPTH = 40
EMPTY_SHEET_MAX_CELLS = 5


class HeaderNotFound(ValueError):
    """No row within ``max_depth`` contains all the required column names."""

    def __init__(self, file: str, sheet: str, required: Iterable[str], preview: list[list[str]]):
        self.file, self.sheet, self.required, self.preview = file, sheet, sorted(required), preview
        lines = "\n".join(f"    row {i}: {cells}" for i, cells in enumerate(preview))
        super().__init__(
            f"header row not found in file {file!r}, sheet {sheet!r}: "
            f"no row contains all of {self.required}. First rows (normalised):\n{lines}"
        )


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:
        return True
    return isinstance(value, str) and not value.strip()


def row_names(raw: pd.DataFrame, index: int) -> list[str]:
    """Normalised column names for one raw row (blank cells -> ``""``)."""
    return [normalise_column_name(cell) for cell in raw.iloc[index].tolist()]


def find_header_row(
    raw: pd.DataFrame,
    expected_columns: Iterable[str],
    *,
    required: Iterable[str] | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> int | None:
    """Return the index of the first row whose normalised cells include every *required* column.

    ``expected_columns`` are canonical names (see :mod:`readers.columns`). ``required`` defaults to
    all of them; pass a subset when some columns are optional across years. Returns ``None`` when no
    row matches; the loader turns that into :class:`HeaderNotFound` or an empty frame.
    """
    expected = {normalise_column_name(c) for c in expected_columns}
    need = expected if required is None else {normalise_column_name(c) for c in required}
    if not need:
        raise ValueError("at least one required column name is needed to find a header row")
    for i in range(min(max_depth, len(raw))):
        present = {n for n in row_names(raw, i) if n}
        if need <= present:
            return i
    return None


def is_empty_sheet(raw: pd.DataFrame, *, max_cells: int = EMPTY_SHEET_MAX_CELLS) -> bool:
    """True for a placeholder sheet: at most ``max_cells`` non-blank cells, or a "No records" note."""
    non_blank = 0
    for col in raw.columns:
        for value in raw[col].tolist():
            if _is_blank(value):
                continue
            if isinstance(value, str) and _NO_RECORDS.search(value):
                return True
            non_blank += 1
    return non_blank <= max_cells


def preview_rows(raw: pd.DataFrame, n: int = 8) -> list[list[str]]:
    return [[c for c in row_names(raw, i) if c] for i in range(min(n, len(raw)))]
