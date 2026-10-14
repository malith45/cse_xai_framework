"""CSE security identifier normaliser (M1 Data Ingestion & Validation, Chapter 4 §4.7 step 1).

The CSE Data Library spells the same security four different ways. This package parses every
observed spelling into one canonical form and formats the canonical form back out again.

Public API
----------
parse(value)                      auto-detect the format and return SecurityId | UnresolvedSecurityId
parse_components(code, cls, sub)  three-column daily-price format      ("AAF", "N", "0")
parse_hyphen("AAF-N-0000")        List of Quoted Securities / Dividends format
parse_dot("AAF.N0000")            Public Holding / Market Cap / Block Trades / Splits-2025 format
parse_bare("JKH")                 ticker only -> UnresolvedSecurityId (share class unknown)
to_components / to_hyphen / to_dot / to_bare   formatters from the canonical form

See README.md in this directory for the justification of the canonical form.
"""

from .core import (
    KNOWN_SHARE_CLASSES,
    SHARE_CLASS_MEANINGS,
    MalformedIdentifier,
    SecurityId,
    UnknownShareClass,
    UnresolvedSecurityId,
)
from .formatters import to_bare, to_components, to_dot, to_hyphen
from .parsers import parse, parse_bare, parse_components, parse_dot, parse_hyphen

__all__ = [
    "KNOWN_SHARE_CLASSES",
    "SHARE_CLASS_MEANINGS",
    "MalformedIdentifier",
    "SecurityId",
    "UnknownShareClass",
    "UnresolvedSecurityId",
    "parse",
    "parse_bare",
    "parse_components",
    "parse_dot",
    "parse_hyphen",
    "to_bare",
    "to_components",
    "to_dot",
    "to_hyphen",
]
