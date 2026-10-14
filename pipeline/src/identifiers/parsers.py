"""Parsers: one per source format, plus an auto-detecting ``parse``.

Formats (all verified against the 2025 Q4 CSE Data Library on 2026-09-18):

===========  ==============================================  ==========================================
Name         Example                                         Source files
===========  ==============================================  ==========================================
components   ("AAF", "N", "0000") / ("AAF", "N", 0.0)        Daily Shares Price Lists, Foreign Holding
hyphen       "AAF-N-0000", "AAF -R-0000", "TILE-N-0"         List of Quoted Securities, Dividends
dot          "AAF.N0000", "VLL.x0000"                        Public Holding, Market Cap, Block Trades,
                                                             Sub Division 2025
bare         "JKH"                                           Sub Division 2011-2024
===========  ==============================================  ==========================================
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .core import (
    MalformedIdentifier,
    SecurityId,
    UnresolvedSecurityId,
    normalise_company_code,
)

_CODE = r"[A-Za-z][A-Za-z0-9]{1,4}"
_HYPHEN = re.compile(rf"^\s*({_CODE})\s*-\s*([A-Za-z])\s*-\s*(\d{{1,4}})\s*$")
_DOT = re.compile(rf"^\s*({_CODE})\s*\.\s*([A-Za-z])\s*(\d{{1,4}})\s*$")
_BARE = re.compile(rf"^\s*({_CODE})\s*$")


def _text(value: object, fmt: str) -> str:
    if value is None or isinstance(value, bool) or not isinstance(value, str):
        raise MalformedIdentifier(f"{fmt} identifier must be a string, got {value!r}")
    if not value.strip():
        raise MalformedIdentifier(f"{fmt} identifier is blank")
    return value


def _is_blank(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return isinstance(value, str) and not value.strip()


def parse_components(
    company_code: object, main_type: object, sub_type: object
) -> SecurityId | UnresolvedSecurityId:
    """Parse the three-column daily-price representation.

    ``main_type`` may be ``"N"``, ``" n "`` etc. ``sub_type`` may be ``"0000"``, ``"0"``, ``0``
    or ``0.0`` (pandas reads it as float in some years). A blank/NaN ``main_type`` yields an
    :class:`UnresolvedSecurityId` rather than a guess.
    """
    code = normalise_company_code(company_code)
    if _is_blank(main_type):
        return UnresolvedSecurityId(code, raw=str(company_code))
    if _is_blank(sub_type):
        raise MalformedIdentifier(f"sub type missing for {code}-{main_type}")
    return SecurityId(code, main_type, sub_type)


def parse_hyphen(value: object) -> SecurityId:
    """Parse ``"AAF-N-0000"`` (also tolerates ``"AAF -R-0000"``, ``"TILE-N-0"``, lower case)."""
    text = _text(value, "hyphen")
    match = _HYPHEN.fullmatch(text)
    if not match:
        raise MalformedIdentifier(f"not a hyphen-format identifier: {value!r}")
    return SecurityId(*match.groups())


def parse_dot(value: object) -> SecurityId:
    """Parse ``"AAF.N0000"`` (also tolerates ``"VLL.x0000"`` and surrounding whitespace)."""
    text = _text(value, "dot")
    match = _DOT.fullmatch(text)
    if not match:
        raise MalformedIdentifier(f"not a dot-format identifier: {value!r}")
    return SecurityId(*match.groups())


def parse_bare(value: object) -> UnresolvedSecurityId:
    """Parse a bare ticker such as ``"JKH"``. Never guesses the share class."""
    text = _text(value, "bare")
    match = _BARE.fullmatch(text)
    if not match:
        raise MalformedIdentifier(f"not a bare ticker: {value!r}")
    return UnresolvedSecurityId(match.group(1), raw=text.strip())


def parse(value: object) -> SecurityId | UnresolvedSecurityId:
    """Auto-detect the format and parse.

    Accepts a string in hyphen, dot or bare format, or a 3-item sequence
    ``(company_code, main_type, sub_type)`` for the components format.

    Returns a :class:`SecurityId` when the share class is stated, an
    :class:`UnresolvedSecurityId` when it is not (bare ticker, blank main type), and raises
    :class:`MalformedIdentifier` for anything else (blank, debenture codes like
    ``"AAF/BD/18/12/27"``, joint entries like ``"COMB-N-0000 & COMB-X-0000"``, company names,
    unknown class letters, non-numeric sub types).
    """
    if isinstance(value, str):
        if _DOT.fullmatch(value):
            return parse_dot(value)
        if _HYPHEN.fullmatch(value):
            return parse_hyphen(value)
        if _BARE.fullmatch(value):
            return parse_bare(value)
        raise MalformedIdentifier(f"unrecognised identifier format: {value!r}")
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        if len(value) != 3:
            raise MalformedIdentifier(
                f"components identifier needs exactly 3 items (code, main type, sub type), got {value!r}"
            )
        return parse_components(*value)
    raise MalformedIdentifier(f"cannot parse identifier from {type(value).__name__}: {value!r}")
