"""Spreadsheet readers for the CSE Data Library (M1 Data Ingestion & Validation, Chapter 4 §4.7 step 1).

The library's workbooks are human-formatted: title text and blank rows above the header at varying
depths, three file formats (.xls, .xlsx, .csv), sheet names that change convention across years,
sheets in reverse chronological order, and "No records" placeholder sheets. This package turns any of
them into a pandas DataFrame with canonical column names, or raises a clear error.

Public API
----------
read_sheet(path, sheet, expected_columns)    one sheet -> DataFrame with canonical columns
read_workbook(path, expected_columns)        every sheet -> {sheet name: DataFrame}, chronological
load_raw_sheets(path)                        format-dispatching raw grid loader (no header handling)
find_header_row(raw, expected_columns)       locate the header row by matching column names
parse_sheet_name("2012 Q4")                  -> SheetPeriod(year=2012, quarter=4)
normalise_column_name("COMPANY_NAME")        -> "company_name"
"""

from .columns import (
    COLUMN_ALIASES,
    normalise_column_name,
    normalise_columns,
)
from .formats import SUPPORTED_EXTENSIONS, UnsupportedFormat, load_raw_sheets
from .header import HeaderNotFound, find_header_row, is_empty_sheet
from .loader import read_sheet, read_workbook, spilled_rows
from .sheets import SheetPeriod, parse_sheet_name, sort_sheet_names

__all__ = [
    "COLUMN_ALIASES",
    "HeaderNotFound",
    "SUPPORTED_EXTENSIONS",
    "SheetPeriod",
    "UnsupportedFormat",
    "find_header_row",
    "is_empty_sheet",
    "load_raw_sheets",
    "normalise_column_name",
    "normalise_columns",
    "parse_sheet_name",
    "read_sheet",
    "read_workbook",
    "sort_sheet_names",
    "spilled_rows",
]
