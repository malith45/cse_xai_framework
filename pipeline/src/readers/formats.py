"""Format-dispatching raw loader: .xls (xlrd), .xlsx (openpyxl), .csv (pandas).

Every loader returns ``{sheet name: raw grid}`` where the raw grid is a DataFrame read with
``header=None`` so that title rows, blank rows and the real header are all still present for
:mod:`readers.header` to inspect. A CSV has one "sheet", named after the file stem.

Cell types: Excel loaders keep native types (datetimes stay ``datetime``, numbers stay numeric).
The CSV loader reads everything as ``str`` (thousands separators such as ``"10,261"`` are kept as
text). Type coercion is the job of the file-specific parsers downstream, not of this package.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

SUPPORTED_EXTENSIONS = (".xls", ".xlsx", ".xlsm", ".csv")
_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


class UnsupportedFormat(ValueError):
    """The file extension is not one of :data:`SUPPORTED_EXTENSIONS`."""


def _read_excel_all(path: Path, engine: str, sheets: list[str] | None) -> dict[str, pd.DataFrame]:
    with warnings.catch_warnings():
        # openpyxl warns about unsupported extensions / data validation in CSE files; harmless.
        warnings.simplefilter("ignore")
        frames = pd.read_excel(path, sheet_name=sheets if sheets else None, header=None, engine=engine)
    return {str(name): frame for name, frame in frames.items()}


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(path, header=None, dtype=str, encoding=enc, skip_blank_lines=False)
        except UnicodeDecodeError as exc:  # try the next encoding
            last_error = exc
    raise UnicodeDecodeError(*last_error.args) if last_error else RuntimeError("csv read failed")


def load_raw_sheets(path: str | Path, sheets: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every sheet (or only ``sheets``) of a workbook as raw ``header=None`` grids.

    Raises :class:`FileNotFoundError` if the file is missing and :class:`UnsupportedFormat` for an
    unknown extension. Dispatch is by extension: ``.xls`` -> xlrd, ``.xlsx``/``.xlsm`` -> openpyxl,
    ``.csv`` -> pandas with encoding fallback.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".xls":
        return _read_excel_all(path, "xlrd", sheets)
    if ext in (".xlsx", ".xlsm"):
        return _read_excel_all(path, "openpyxl", sheets)
    if ext == ".csv":
        name = path.stem
        if sheets and name not in sheets:
            raise KeyError(f"{path.name} is a CSV with a single sheet named {name!r}; requested {sheets!r}")
        return {name: _read_csv(path)}
    raise UnsupportedFormat(f"{path.name}: unsupported extension {ext!r}; expected one of {SUPPORTED_EXTENSIONS}")


def list_sheet_names(path: str | Path) -> list[str]:
    """Sheet names in workbook order (a CSV reports its file stem) without loading cell data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return [path.stem]
    if ext == ".xls":
        engine = "xlrd"
    elif ext in (".xlsx", ".xlsm"):
        engine = "openpyxl"
    else:
        raise UnsupportedFormat(f"{path.name}: unsupported extension {ext!r}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelFile(path, engine=engine) as xl:
            return [str(s) for s in xl.sheet_names]
