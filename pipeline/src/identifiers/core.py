"""Canonical security identifier types and validation rules.

Canonical form: ``COMPANY-CLASS-SUBTYPE`` (e.g. ``AAF-N-0000``), the CSE's own Security ID as printed
in "List of Quoted Securities And Issued Quantity". Justification is in README.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Share-class letters actually observed in the 2025 Q4 CSE Data Library (survey of 2026-09-18).
# These are DISTINCT securities that trade separately. They are never collapsed into one another.
SHARE_CLASS_MEANINGS: dict[str, str] = {
    "N": "ordinary voting shares",
    "X": "ordinary non-voting shares",
    "P": "preference shares",
    "R": "rights (provisional allotment letters)",
    "W": "warrants",
    "U": "units (closed-end fund / unit trust)",
    "B": "observed only in the quoted-securities master; meaning not documented in the library",
    "S": "observed only in the quoted-securities master; meaning not documented in the library",
}
KNOWN_SHARE_CLASSES: frozenset[str] = frozenset(SHARE_CLASS_MEANINGS)

# Company codes observed in the daily price files are 3-4 upper-case letters (300 distinct codes,
# no digits). The pattern allows 2-5 alphanumerics so that a future code with a digit still parses,
# while a company *name* that leaked into an ID column (e.g. "Unisyst", splits 2021) is rejected.
COMPANY_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9]{1,4}$")
SUB_TYPE_WIDTH = 4
SUB_TYPE_MAX = 10**SUB_TYPE_WIDTH - 1


class MalformedIdentifier(ValueError):
    """The input cannot be interpreted as any known CSE security identifier format."""


class UnknownShareClass(MalformedIdentifier):
    """The share-class letter is not one observed in the CSE Data Library."""


def normalise_company_code(raw: object) -> str:
    """Upper-case, strip and validate a company code such as ``"aaf "`` -> ``"AAF"``."""
    if raw is None or isinstance(raw, bool):
        raise MalformedIdentifier(f"company code must be text, got {raw!r}")
    code = str(raw).strip().upper()
    if not COMPANY_CODE_PATTERN.fullmatch(code):
        raise MalformedIdentifier(f"invalid company code {raw!r}")
    return code


def normalise_share_class(raw: object) -> str:
    """Upper-case, strip and validate a share-class letter (``"n"`` -> ``"N"``)."""
    if raw is None or isinstance(raw, bool):
        raise MalformedIdentifier(f"share class must be a letter, got {raw!r}")
    cls = str(raw).strip().upper()
    if len(cls) != 1 or not cls.isalpha():
        raise MalformedIdentifier(f"invalid share class {raw!r}")
    if cls not in KNOWN_SHARE_CLASSES:
        raise UnknownShareClass(
            f"share class {cls!r} is not one of the observed classes {sorted(KNOWN_SHARE_CLASSES)}"
        )
    return cls


_SUB_TYPE_TEXT = re.compile(r"^(\d{1,4})(?:\.0+)?$")


def normalise_sub_type(raw: object) -> str:
    """Zero-pad a sub type to four digits.

    Accepts ``"0000"``, ``"0"``, ``0``, ``0.0`` (pandas float), ``"0001"``, ``"1"``.
    Rejects negatives, non-integers, more than four significant digits, blanks and NaN.
    """
    if raw is None or isinstance(raw, bool):
        raise MalformedIdentifier(f"sub type missing: {raw!r}")
    if isinstance(raw, float):
        if raw != raw or not raw.is_integer():  # NaN or fractional
            raise MalformedIdentifier(f"invalid sub type {raw!r}")
        raw = int(raw)
    if isinstance(raw, int):
        if not 0 <= raw <= SUB_TYPE_MAX:
            raise MalformedIdentifier(f"sub type out of range: {raw!r}")
        return f"{raw:0{SUB_TYPE_WIDTH}d}"
    text = str(raw).strip()
    match = _SUB_TYPE_TEXT.fullmatch(text)
    if not match:
        raise MalformedIdentifier(f"invalid sub type {raw!r}")
    return f"{int(match.group(1)):0{SUB_TYPE_WIDTH}d}"


@dataclass(frozen=True, slots=True)
class SecurityId:
    """A fully resolved CSE security: company code + share class + sub type.

    Instances are immutable, hashable and compare by value, so they can be used as dictionary keys
    and pandas merge keys. ``COMB-N-0000`` and ``COMB-X-0000`` are different objects and never equal.
    """

    company_code: str
    share_class: str
    sub_type: str = "0000"

    def __post_init__(self) -> None:
        object.__setattr__(self, "company_code", normalise_company_code(self.company_code))
        object.__setattr__(self, "share_class", normalise_share_class(self.share_class))
        object.__setattr__(self, "sub_type", normalise_sub_type(self.sub_type))

    @property
    def is_resolved(self) -> bool:
        return True

    @property
    def canonical(self) -> str:
        """The canonical string, e.g. ``"AAF-N-0000"``."""
        return f"{self.company_code}-{self.share_class}-{self.sub_type}"

    def __str__(self) -> str:
        return self.canonical


@dataclass(frozen=True, slots=True)
class UnresolvedSecurityId:
    """A company code whose share class (and sub type) is not stated in the source.

    Produced for bare tickers such as ``"JKH"`` (splits 2024) or a daily-price row with a blank
    MAIN TYPE. The module never guesses the class; the caller must decide, typically by looking the
    code up in the quoted-securities master for the relevant date and calling :meth:`resolve`.
    """

    company_code: str
    raw: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "company_code", normalise_company_code(self.company_code))
        object.__setattr__(self, "raw", str(self.raw))

    @property
    def is_resolved(self) -> bool:
        return False

    @property
    def canonical(self) -> None:
        """Always ``None``: there is no canonical string without a share class."""
        return None

    def resolve(self, share_class: object, sub_type: object = "0000") -> SecurityId:
        """Return the :class:`SecurityId` once the caller has determined the share class."""
        return SecurityId(self.company_code, share_class, sub_type)

    def __str__(self) -> str:
        return f"{self.company_code}-?-????"
