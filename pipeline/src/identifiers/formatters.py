"""Formatters from the canonical :class:`SecurityId` back to each source format.

All formatters except :func:`to_bare` require a resolved :class:`SecurityId`; passing an
:class:`UnresolvedSecurityId` raises ``TypeError`` so that an unknown share class can never be
written out as if it were known.
"""

from __future__ import annotations

from .core import SecurityId, UnresolvedSecurityId


def _require_resolved(sid: object, fmt: str) -> SecurityId:
    if isinstance(sid, SecurityId):
        return sid
    if isinstance(sid, UnresolvedSecurityId):
        raise TypeError(
            f"cannot format {sid} as {fmt}: share class is unknown; call .resolve(...) first"
        )
    raise TypeError(f"{fmt} formatter expects a SecurityId, got {type(sid).__name__}")


def to_components(sid: SecurityId) -> tuple[str, str, str]:
    """``SecurityId("AAF","N","0")`` -> ``("AAF", "N", "0000")`` (daily price file columns)."""
    s = _require_resolved(sid, "components")
    return (s.company_code, s.share_class, s.sub_type)


def to_hyphen(sid: SecurityId) -> str:
    """-> ``"AAF-N-0000"`` (List of Quoted Securities, Dividends). Identical to ``sid.canonical``."""
    s = _require_resolved(sid, "hyphen")
    return f"{s.company_code}-{s.share_class}-{s.sub_type}"


def to_dot(sid: SecurityId) -> str:
    """-> ``"AAF.N0000"`` (Public Holding, Market Cap, Block Trades, Splits 2025)."""
    s = _require_resolved(sid, "dot")
    return f"{s.company_code}.{s.share_class}{s.sub_type}"


def to_bare(sid: SecurityId | UnresolvedSecurityId) -> str:
    """-> ``"AAF"``. Lossy: drops share class and sub type. Works for unresolved ids too."""
    if isinstance(sid, (SecurityId, UnresolvedSecurityId)):
        return sid.company_code
    raise TypeError(f"bare formatter expects a SecurityId or UnresolvedSecurityId, got {type(sid).__name__}")
