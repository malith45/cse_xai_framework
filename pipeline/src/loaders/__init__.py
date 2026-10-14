"""File-specific loaders that turn CSE Data Library workbooks into tidy tables (M1, Chapter 4 §4.7).

Each loader builds on :mod:`readers` (header detection, format dispatch, sheet naming) and
:mod:`identifiers` (canonical security IDs) and adds only the file's own semantics.
"""

from .public_float import (
    DUPLICATE_POLICIES,
    DuplicateRowsError,
    Convention,
    FloatOutOfRange,
    PublicFloatError,
    QuarterParseError,
    SheetReport,
    audit_duplicates,
    detect_numeric_convention,
    load_public_float,
    load_public_float_file,
    normalise_float_values,
    parse_quarter_code,
    quarter_end_date,
    resolve_duplicates,
)

__all__ = [
    "DUPLICATE_POLICIES",
    "DuplicateRowsError",
    "Convention",
    "FloatOutOfRange",
    "PublicFloatError",
    "QuarterParseError",
    "SheetReport",
    "audit_duplicates",
    "detect_numeric_convention",
    "load_public_float",
    "load_public_float_file",
    "normalise_float_values",
    "parse_quarter_code",
    "quarter_end_date",
    "resolve_duplicates",
]
