"""Public holding (free float) loader — CSE Data Library file 17 "Public Holding-Quarterly".

Fourteen yearly ``.xlsx`` files (2012–2025), four quarterly sheets each, give the percentage of each
listed security held by the public. The framework uses it to reconstruct S&P SL20 eligibility
(float-adjusted market capitalisation, Chapter 4 §4.2 / Chapter 3 §3.7).

Output (one tidy DataFrame):

==============  ==========  ======================================================
column          type        meaning
==============  ==========  ======================================================
security_id     str         canonical ``AAF-N-0000`` (see :mod:`identifiers`)
company_name    str         as printed in the source
year            int
quarter         int         1–4
quarter_end     Timestamp   last calendar day of the quarter
float_pct       float       public float in percentage points, 0–100 inclusive
==============  ==========  ======================================================

Value conventions (verified on 2026-09-19, see README): the float is stored either as a
**decimal fraction** (``0.1024`` = 10.24 %) or as a **percentage string** (``"2.84%"``). The
convention changes between sheets, not only between files, and one sheet (2020 Q2) mixes both.
Detection is therefore done per sheet and per value, from the values themselves, never from the year.

Duplicate rows (the same canonical security twice in one sheet) are resolved only after the whole
series has been loaded, because a stale duplicate is recognisable by *not moving* across quarters.
"""

from __future__ import annotations

import calendar
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import pandas as pd

from identifiers import MalformedIdentifier, UnresolvedSecurityId
from identifiers import parse as parse_identifier
from readers import parse_sheet_name, read_workbook

EXPECTED_COLUMNS = ["quarter", "company_name", "security_id", "public_float"]
FILE_GLOB = "Public Holding *.xlsx"
OUTPUT_COLUMNS = ["security_id", "company_name", "year", "quarter", "quarter_end", "float_pct"]
_AUDIT_COLUMNS = ["file", "sheet", "row", "security_id", "raw_id", "raw_float", "float_pct", "year", "quarter"]

# Plausibility window for a sheet's *median* float once normalised to percentage points. Observed
# medians across all 55 real data sheets lie between 21.7 and 26.6. A misdetected convention would put
# the median near 0.25 (fractions read as points) or 2,500 (points read as fractions).
MEDIAN_PLAUSIBLE_RANGE = (5.0, 60.0)

# What to do when one sheet lists the same canonical security twice.
#   "error"                halt (default).
#   "prefer_varying_value" resolved on the FULL series: among the duplicate rows of a security, the
#                          value that is identical in every duplicated quarter is the stale copy and
#                          is dropped; the row whose value varies across quarters is kept. Needs at
#                          least two duplicated quarters. If no candidate is frozen, if more than one
#                          is, or if a quarter does not reduce to exactly one row: halt, print all
#                          rows. Motivated by 2024 Q3 - 2025 Q4, where a "VLL.x0000" row frozen at
#                          70.29% sits beside the live "VLL.X0000" row. ID spelling is never used.
DUPLICATE_POLICIES = ("error", "prefer_varying_value")

_QUARTER_CODE = re.compile(r"^\s*(\d{4})\s*-\s*0?([1-4])\s*$")
_NO_RECORDS = re.compile(r"\bno\s+records?\b", re.IGNORECASE)
# ".01%" (no leading digit, seen in 2018 Q1) and "27.04%" are both valid percent strings
_PERCENT_STRING = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*%\s*$")


class PublicFloatError(ValueError):
    """Base class for public-float loading failures. Messages always name file and sheet."""


class FloatOutOfRange(PublicFloatError):
    """A normalised float is outside 0–100 inclusive, unreadable, or the sheet median is implausible."""


class QuarterParseError(PublicFloatError):
    """A QUARTER cell could not be parsed, or disagrees with the sheet name."""


class DuplicateRowsError(PublicFloatError):
    """Duplicate rows for one security in one sheet could not be resolved under the active policy."""


class Convention(str, Enum):
    FRACTION = "fraction"              # numeric 0-1, where 1 = 100 %
    PERCENT_STRING = "percent_string"  # "28.62%"
    PERCENT_POINTS = "percent_points"  # numeric 0-100 (not observed in the library; supported)
    MIXED = "mixed"                    # percent strings alongside numeric fractions (2020 Q2)
    EMPTY = "empty"                    # no values (2020 Q1 Covid note)


@dataclass(frozen=True)
class SheetReport:
    file: str
    sheet: str
    convention: Convention
    rows: int
    notes: str = ""


# ----------------------------------------------------------------------------- quarter parsing
def parse_quarter_code(value: object) -> tuple[int, int]:
    """``"2012-4"`` / ``"2018-1"`` / ``"2025-01"`` / ``"2016 - 1"`` -> ``(year, quarter)``.

    The second part is a **quarter number**, not a month, whatever its padding.
    """
    if value is None or (isinstance(value, float) and value != value):
        raise ValueError("blank quarter code")
    m = _QUARTER_CODE.fullmatch(str(value))
    if not m:
        raise ValueError(f"unrecognised quarter code {value!r}")
    return int(m.group(1)), int(m.group(2))


def quarter_end_date(year: int, quarter: int) -> pd.Timestamp:
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"quarter must be 1-4, got {quarter!r}")
    month = quarter * 3
    return pd.Timestamp(year=year, month=month, day=calendar.monthrange(year, month)[1])


# ----------------------------------------------------------------------------- unit detection
def _is_percent_string(v: object) -> bool:
    return isinstance(v, str) and _PERCENT_STRING.fullmatch(v) is not None


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v


def detect_numeric_convention(values: Iterable[object]) -> Convention:
    """Decide how the *numeric* cells of one sheet encode the float.

    Rule (from the distribution, never from the year): if every numeric value is <= 1.0 the sheet
    stores fractions (1.0 = 100 %); if any numeric value exceeds 1.0 the sheet stores percentage
    points. A points sheet in which every value happened to be <= 1 would be misread as fractions;
    the sheet-level median check in :func:`normalise_float_values` is the second line of defence.
    """
    nums = [float(v) for v in values if _is_number(v)]
    if not nums:
        return Convention.EMPTY
    return Convention.PERCENT_POINTS if max(nums) > 1.0 else Convention.FRACTION


def _classify_sheet(values: list[object]) -> Convention:
    has_pct = any(_is_percent_string(v) for v in values)
    numeric = detect_numeric_convention(values)
    if numeric is Convention.EMPTY:
        return Convention.PERCENT_STRING if has_pct else Convention.EMPTY
    return Convention.MIXED if has_pct else numeric


def normalise_float_values(
    values: pd.Series, *, file: str, sheet: str, row_labels: pd.Series | None = None
) -> tuple[pd.Series, Convention]:
    """Convert one sheet's raw float cells to percentage points and assert 0 <= value <= 100.

    Percent strings are converted directly. Numeric cells follow the sheet-level rule of
    :func:`detect_numeric_convention`. Any other content, any value outside 0-100, and any sheet whose
    median falls outside :data:`MEDIAN_PLAUSIBLE_RANGE` raises :class:`FloatOutOfRange` naming the
    file, sheet and row. Nothing is clamped or skipped.
    """
    raw_values = values.tolist()
    convention = _classify_sheet(raw_values)
    numeric_rule = detect_numeric_convention(raw_values)
    labels = row_labels if row_labels is not None else values.index.to_series().astype(str)
    out: list[float] = []
    for pos, (idx, raw) in enumerate(values.items()):
        label = f"row {idx} ({labels.iloc[pos]})"
        if _is_percent_string(raw):
            pct = float(_PERCENT_STRING.fullmatch(raw).group(1))
        elif _is_number(raw):
            pct = float(raw) * 100.0 if numeric_rule is Convention.FRACTION else float(raw)
        else:
            raise FloatOutOfRange(f"{file!r} sheet {sheet!r} {label}: unreadable public float {raw!r}")
        if not 0.0 <= pct <= 100.0:
            raise FloatOutOfRange(
                f"{file!r} sheet {sheet!r} {label}: public float {pct:.4f}% is outside 0-100 "
                f"(raw value {raw!r}, convention {convention.value})"
            )
        out.append(pct)
    result = pd.Series(out, index=values.index, dtype=float, name="float_pct")
    if len(result):
        median = float(result.median())
        lo, hi = MEDIAN_PLAUSIBLE_RANGE
        if not lo <= median <= hi:
            raise FloatOutOfRange(
                f"{file!r} sheet {sheet!r}: median public float {median:.3f}% is outside the plausible "
                f"range {lo}-{hi}%, so the unit convention ({convention.value}) was probably misdetected"
            )
    return result, convention


# ----------------------------------------------------------------------------- canonical IDs
def _canonical_id(raw: object, *, file: str, sheet: str, row: object) -> str:
    try:
        sid = parse_identifier(str(raw).strip())
    except MalformedIdentifier as exc:
        raise PublicFloatError(f"{file!r} sheet {sheet!r} row {row}: bad security id {raw!r}: {exc}") from exc
    if isinstance(sid, UnresolvedSecurityId):
        raise PublicFloatError(f"{file!r} sheet {sheet!r} row {row}: security id {raw!r} has no share class")
    return sid.canonical


def _empty_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": pd.Series(dtype=str),
            "company_name": pd.Series(dtype=str),
            "year": pd.Series(dtype=int),
            "quarter": pd.Series(dtype=int),
            "quarter_end": pd.Series(dtype="datetime64[ns]"),
            "float_pct": pd.Series(dtype=float),
            "file": pd.Series(dtype=str),
            "sheet": pd.Series(dtype=str),
            "row": pd.Series(dtype=int),
            "raw_id": pd.Series(dtype=str),
            "raw_float": pd.Series(dtype=object),
        }
    )


# ----------------------------------------------------------------------------- sheet loading
def _load_sheet(df: pd.DataFrame, *, file: str, sheet: str) -> tuple[pd.DataFrame, SheetReport]:
    """One sheet -> normalised rows (with provenance columns) + report. Duplicates are NOT resolved here."""
    expected = parse_sheet_name(sheet)
    # A "No records" note sometimes sits in the data area under a real header (2020 Q1, Covid).
    note_mask = df["quarter"].map(lambda v: isinstance(v, str) and bool(_NO_RECORDS.search(v)))
    notes = ""
    if note_mask.any():
        notes = " | ".join(" ".join(str(v).split()) for v in df.loc[note_mask, "quarter"])
        df = df.loc[~note_mask]
    if df.empty:
        return _empty_rows(), SheetReport(file, sheet, Convention.EMPTY, 0, notes or "no data rows")

    years: list[int] = []
    quarters: list[int] = []
    for idx, code in df["quarter"].items():
        try:
            y, q = parse_quarter_code(code)
        except ValueError as exc:
            raise QuarterParseError(f"{file!r} sheet {sheet!r} row {idx}: {exc}") from exc
        if expected is not None and expected.year is not None and (y, q) != (expected.year, expected.quarter):
            raise QuarterParseError(
                f"{file!r} sheet {sheet!r} row {idx}: quarter code {code!r} -> {y}Q{q} disagrees with the sheet name"
            )
        years.append(y)
        quarters.append(q)

    ids = [_canonical_id(v, file=file, sheet=sheet, row=idx) for idx, v in df["security_id"].items()]
    float_pct, convention = normalise_float_values(
        df["public_float"], file=file, sheet=sheet, row_labels=pd.Series(ids, index=df.index)
    )
    rows = pd.DataFrame(
        {
            "security_id": ids,
            "company_name": [" ".join(str(v).split()) if isinstance(v, str) else v for v in df["company_name"]],
            "year": years,
            "quarter": quarters,
            "quarter_end": [quarter_end_date(y, q) for y, q in zip(years, quarters)],
            "float_pct": float_pct.to_numpy(),
            "file": file,
            "sheet": sheet,
            "row": df.index.to_numpy(),
            "raw_id": [str(v).strip() for v in df["security_id"]],
            "raw_float": df["public_float"].to_numpy(dtype=object),
        }
    )
    return rows, SheetReport(file, sheet, convention, len(rows), notes)


def _load_file_rows(path: Path) -> tuple[pd.DataFrame, list[SheetReport]]:
    frames = read_workbook(path, EXPECTED_COLUMNS)  # chronological; fixes the 2012-2017 reverse order
    parts: list[pd.DataFrame] = []
    reports: list[SheetReport] = []
    for sheet, df in frames.items():
        rows, report = _load_sheet(df, file=path.name, sheet=sheet)
        parts.append(rows)
        reports.append(report)
    rows = pd.concat(parts, ignore_index=True) if parts else _empty_rows()
    return rows, reports


# ----------------------------------------------------------------------------- duplicate resolution
def _describe(rows: pd.DataFrame) -> str:
    return "; ".join(
        f"{r.file} sheet {r.sheet!r} row {r.row}: {r.raw_id!r} -> {r.raw_float!r} ({r.float_pct:.4g}%)"
        for r in rows.itertuples()
    )


def _resolve_prefer_varying_value(rows: pd.DataFrame, dup_mask: pd.Series) -> tuple[pd.Series, dict[tuple[str, str], list[str]]]:
    """Return (keep mask, notes per (file, sheet)) or raise :class:`DuplicateRowsError`."""
    keep = pd.Series(True, index=rows.index)
    notes: dict[tuple[str, str], list[str]] = {}
    for sid, group in rows[dup_mask].groupby("security_id", sort=True):
        by_quarter = {(y, q): g for (y, q), g in group.groupby(["year", "quarter"], sort=True)}
        if len(by_quarter) < 2:
            raise DuplicateRowsError(
                f"duplicate rows for {sid} in a single quarter only; cannot tell which is stale "
                f"without a second quarter to compare ({_describe(group)})"
            )
        value_sets = [set(g["float_pct"].round(6)) for g in by_quarter.values()]
        frozen = set.intersection(*value_sets)  # values present in EVERY duplicated quarter
        if len(frozen) != 1:
            why = "no candidate value is constant across the duplicated quarters (both vary)" if not frozen \
                else f"more than one candidate value is constant across the duplicated quarters (both frozen): {sorted(frozen)}"
            raise DuplicateRowsError(f"duplicate rows for {sid}: {why}. Rows: {_describe(group)}")
        stale = next(iter(frozen))
        for (y, q), g in by_quarter.items():
            is_stale = g["float_pct"].round(6) == stale
            live = g[~is_stale]
            if len(live) != 1:
                raise DuplicateRowsError(
                    f"duplicate rows for {sid} in {y}Q{q} do not reduce to one live row after removing the "
                    f"value frozen at {stale}%: {_describe(g)}"
                )
            live_row = live.iloc[0]
            for r in g[is_stale].itertuples():
                keep[r.Index] = False
                notes.setdefault((r.file, r.sheet), []).append(
                    f"dropped row {r.row} {r.raw_id!r} float {r.raw_float!r} (frozen at {stale:.4g}% across "
                    f"{len(by_quarter)} quarters) in favour of row {live_row['row']} {live_row['raw_id']!r} "
                    f"float {live_row['raw_float']!r}"
                )
    return keep, notes


def resolve_duplicates(
    rows: pd.DataFrame, reports: list[SheetReport], *, policy: str = "error"
) -> tuple[pd.DataFrame, list[SheetReport]]:
    """Apply :data:`DUPLICATE_POLICIES` to the fully loaded series and annotate the sheet reports."""
    if policy not in DUPLICATE_POLICIES:
        raise ValueError(f"duplicate_policy must be one of {DUPLICATE_POLICIES}, got {policy!r}")
    dup_mask = rows.duplicated(["security_id", "year", "quarter"], keep=False)
    if not dup_mask.any():
        return rows, reports
    if policy == "error":
        sid = rows.loc[dup_mask, "security_id"].iloc[0]
        first = rows[dup_mask & (rows["security_id"] == sid)]
        raise DuplicateRowsError(
            f"duplicate rows for {sid} (policy 'error'; {int(dup_mask.sum())} duplicate rows in total, "
            f"{rows.loc[dup_mask, 'security_id'].nunique()} securities). Rows: {_describe(first)}"
        )
    keep, notes = _resolve_prefer_varying_value(rows, dup_mask)
    annotated = [
        replace(rep, rows=rep.rows - len(notes.get((rep.file, rep.sheet), [])),
                notes=" | ".join(n for n in (rep.notes, "duplicates: " + "; ".join(notes[(rep.file, rep.sheet)])) if n))
        if (rep.file, rep.sheet) in notes else rep
        for rep in reports
    ]
    return rows[keep], annotated


# ----------------------------------------------------------------------------- public API
def load_public_float_file(
    path: str | Path, *, duplicate_policy: str = "error"
) -> tuple[pd.DataFrame, list[SheetReport]]:
    """Load one yearly workbook -> (tidy frame for all its quarters, per-sheet reports).

    Duplicates are resolved with only this file's quarters as context; prefer
    :func:`load_public_float` for the whole series.
    """
    rows, reports = _load_file_rows(Path(path))
    rows, reports = resolve_duplicates(rows, reports, policy=duplicate_policy)
    return rows[OUTPUT_COLUMNS].reset_index(drop=True), reports


def _file_year(path: Path) -> int:
    m = re.search(r"(?<!\d)(20\d{2})(?!\d)", path.stem)
    if not m:
        raise PublicFloatError(f"cannot find a year in file name {path.name!r}")
    return int(m.group(1))


def _load_folder_rows(folder: Path) -> tuple[pd.DataFrame, list[SheetReport]]:
    files = sorted(folder.glob(FILE_GLOB), key=_file_year)
    if not files:
        raise FileNotFoundError(f"no {FILE_GLOB!r} files under {folder}")
    parts: list[pd.DataFrame] = []
    reports: list[SheetReport] = []
    for f in files:
        rows, rep = _load_file_rows(f)
        parts.append(rows)
        reports.extend(rep)
    return pd.concat(parts, ignore_index=True), reports


def load_public_float(folder: str | Path, *, with_reports: bool = False, duplicate_policy: str = "error"):
    """Load every ``Public Holding *.xlsx`` in ``folder`` into one tidy DataFrame sorted by security
    and quarter. The full series is loaded first, then duplicates are resolved across all quarters
    (``duplicate_policy``, see :data:`DUPLICATE_POLICIES`). With ``with_reports=True`` also return
    the per-sheet :class:`SheetReport` list, which records every dropped row."""
    rows, reports = _load_folder_rows(Path(folder))
    rows, reports = resolve_duplicates(rows, reports, policy=duplicate_policy)
    out = (
        rows[OUTPUT_COLUMNS]
        .sort_values(["security_id", "year", "quarter"], kind="stable")
        .reset_index(drop=True)
    )
    return (out, reports) if with_reports else out


def audit_duplicates(folder: str | Path) -> pd.DataFrame:
    """Every row that shares its (security, year, quarter) with another row, with provenance and
    raw values, for manual inspection. No policy is applied and nothing is dropped."""
    rows, _ = _load_folder_rows(Path(folder))
    dup = rows.duplicated(["security_id", "year", "quarter"], keep=False)
    return (
        rows.loc[dup, _AUDIT_COLUMNS]
        .sort_values(["security_id", "year", "quarter", "raw_id"], kind="stable")
        .reset_index(drop=True)
    )
