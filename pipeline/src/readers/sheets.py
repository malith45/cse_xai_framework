"""Sheet-name parsing: extract (year, quarter) from every naming convention in the library.

Observed conventions (2026-09-18 survey):

=================  ===========================  ==========================================
Convention         Examples                     Files
=================  ===========================  ==========================================
``YYYY QN``        ``2012 Q4``, ``2016 Q2``     Public Holding 2012-2017, Beta 2016-2022
``YYYY_QN``        ``2018_Q1``, ``2009_Q1``     Public Holding 2018-2020, Beta 2009-2012
``YYYYQN``         ``2012Q2``, ``2025Q4``       Beta 2012-2015, 2022-2025
``N-YYYY``         ``1-2025``                   Public Holding 2021-2025 (N = quarter, not month)
``YYYY``           ``2025``, ``'2018 '``        Dividends, Splits, Rights, Bonus, Quoted Securities
``... QN``         ``Daily Share Price Q1``     Daily price lists 2021-2022 (year only in the file)
=================  ===========================  ==========================================

Anything else (``HighLow``, ``Index``, ``Sheet1``) parses to ``None``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class SheetPeriod:
    year: int | None
    quarter: int | None = None

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.year if self.year is not None else 9999, self.quarter or 0)

    def __str__(self) -> str:
        y = str(self.year) if self.year is not None else "????"
        return f"{y}Q{self.quarter}" if self.quarter else y


_YEAR_QUARTER = re.compile(r"^\s*(\d{4})\s*[_\- ]?\s*Q\s*[_\- ]?\s*([1-4])\s*$", re.IGNORECASE)
_QUARTER_YEAR = re.compile(r"^\s*Q?\s*([1-4])\s*[-_ ]\s*(\d{4})\s*$", re.IGNORECASE)
_YEAR_ONLY = re.compile(r"^\s*(\d{4})\s*$")
_QUARTER_ONLY = re.compile(r"\bQ\s*([1-4])\b", re.IGNORECASE)
# lookarounds rather than \b so a year glued to an underscore ("..._2020_to_...") still matches
_YEAR_INSIDE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")


def parse_sheet_name(name: object) -> SheetPeriod | None:
    """Return the :class:`SheetPeriod` encoded in a sheet name, or ``None`` if there is none."""
    if name is None:
        return None
    text = str(name)
    if m := _YEAR_QUARTER.fullmatch(text):
        return SheetPeriod(int(m.group(1)), int(m.group(2)))
    if m := _QUARTER_YEAR.fullmatch(text):
        return SheetPeriod(int(m.group(2)), int(m.group(1)))
    if m := _YEAR_ONLY.fullmatch(text):
        return SheetPeriod(int(m.group(1)), None)
    quarter = _QUARTER_ONLY.search(text)
    year = _YEAR_INSIDE.search(text)
    if quarter or year:
        return SheetPeriod(int(year.group(1)) if year else None, int(quarter.group(1)) if quarter else None)
    return None


def sort_sheet_names(names: Iterable[str]) -> list[str]:
    """Order sheet names chronologically (oldest first). Names without a parseable period keep
    their original relative order and go last."""
    names = list(names)
    parsed = [(parse_sheet_name(n), i, n) for i, n in enumerate(names)]
    with_period = sorted(((p.sort_key, i, n) for p, i, n in parsed if p is not None and p.year is not None))
    without = [(i, n) for p, i, n in parsed if p is None or p.year is None]
    return [n for _, _, n in with_period] + [n for _, n in without]
