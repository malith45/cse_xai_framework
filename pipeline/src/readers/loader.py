"""High-level readers: one sheet or a whole workbook -> DataFrame(s) with canonical columns."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from .columns import normalise_column_name, normalise_columns
from .formats import load_raw_sheets
from .header import DEFAULT_MAX_DEPTH, HeaderNotFound, find_header_row, is_empty_sheet, preview_rows
from .sheets import sort_sheet_names


def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    # pandas 3 may infer the dedicated "str" dtype instead of object; handle both.
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def _blank_mask(df: pd.DataFrame) -> pd.DataFrame:
    return df.isna() | df.map(lambda v: isinstance(v, str) and not v.strip())


def table_from_raw(
    raw: pd.DataFrame,
    header_index: int,
    *,
    strip_strings: bool = True,
    drop_blank_rows: bool = True,
    drop_unnamed_empty_columns: bool = True,
) -> pd.DataFrame:
    """Slice the data below ``header_index`` and give it canonical column names."""
    header_cells = raw.iloc[header_index].tolist()
    df = raw.iloc[header_index + 1 :].reset_index(drop=True).copy()
    df.columns = normalise_columns(header_cells)
    if strip_strings:
        df = _strip_strings(df)
    blank = _blank_mask(df)
    if drop_unnamed_empty_columns:
        keep = [c for c in df.columns if not (c.startswith("unnamed_") and blank[c].all())]
        df, blank = df[keep], blank[keep]
    if drop_blank_rows:
        df = df[~blank.all(axis=1)].reset_index(drop=True)
    return df


def spilled_rows(df: pd.DataFrame) -> pd.Index:
    """Index of rows that carry data in *unnamed* columns (cells beyond the header).

    Such rows are ragged: a field split on an unquoted comma pushed values to the right. The 2023
    daily price CSV does this for every price of Rs 1,000 or more (``1,225.00`` -> ``1`` | ``225.00``).
    The reader keeps the spilled cells rather than dropping them; repairing or excluding the rows is a
    decision for the file-specific parser (CLAUDE.md §0).
    """
    unnamed = [c for c in df.columns if str(c).startswith("unnamed_")]
    if not unnamed:
        return df.index[:0]
    blank = _blank_mask(df[unnamed])
    return df.index[~blank.all(axis=1)]


def empty_table(expected_columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame({normalise_column_name(c): pd.Series(dtype=object) for c in expected_columns})


def read_sheet(
    path: str | Path,
    sheet: str,
    expected_columns: Iterable[str],
    *,
    required: Iterable[str] | None = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
    raw: pd.DataFrame | None = None,
    **table_options,
) -> pd.DataFrame:
    """Read one sheet into a DataFrame whose columns are canonical names.

    - The header row is *found* by matching ``expected_columns`` (all of ``required``, default all).
    - A placeholder sheet ("No records ...", or almost no cells) returns an **empty** DataFrame with
      the expected canonical columns.
    - Otherwise a missing header raises :class:`HeaderNotFound` naming the file and sheet.

    ``raw`` lets a caller that has already loaded the grid skip re-reading the file.
    """
    path = Path(path)
    expected = list(expected_columns)
    if raw is None:
        raw = load_raw_sheets(path, [sheet])[sheet]
    header_index = find_header_row(raw, expected, required=required, max_depth=max_depth)
    if header_index is None:
        if is_empty_sheet(raw):
            return empty_table(expected)
        raise HeaderNotFound(path.name, sheet, required if required is not None else expected, preview_rows(raw))
    return table_from_raw(raw, header_index, **table_options)


def read_workbook(
    path: str | Path,
    expected_columns: Iterable[str],
    *,
    required: Iterable[str] | None = None,
    sheets: list[str] | None = None,
    chronological: bool = True,
    max_depth: int = DEFAULT_MAX_DEPTH,
    **table_options,
) -> dict[str, pd.DataFrame]:
    """Read every sheet (or ``sheets``) -> ``{sheet name: DataFrame}``.

    With ``chronological=True`` (default) the dict is ordered oldest-first using
    :func:`readers.sheets.parse_sheet_name`; workbooks listed newest-first (Dividends, Bonus Issues,
    Public Holding 2012-2017) come out in time order. Unparseable names keep workbook order, last.
    Empty placeholder sheets are included as empty frames so callers see every period.
    """
    path = Path(path)
    expected = list(expected_columns)
    raw_sheets = load_raw_sheets(path, sheets)
    names = sort_sheet_names(raw_sheets) if chronological else list(raw_sheets)
    return {
        name: read_sheet(path, name, expected, required=required, max_depth=max_depth, raw=raw_sheets[name], **table_options)
        for name in names
    }
