"""Column-name normalisation and canonical aliases.

Every header cell goes through :func:`normalise_column_name`:

1. strip whitespace, collapse internal runs of whitespace/newlines to one space;
2. drop parenthesised unit suffixes such as ``(Rs.)`` and ``(No.)``;
3. lower-case; replace every run of non-alphanumerics with ``_``; trim ``_``;
4. map known variants to one canonical name via :data:`COLUMN_ALIASES`.

So ``"COMPANY NAME"``, ``"COMPANY_NAME"``, ``" Company Name "`` and ``"NAME"`` all become
``"company_name"``, and ``"PRICE HIGH (Rs.)"`` becomes ``"price_high"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

_UNITS = re.compile(r"\((?:rs\.?|no\.?|%|lkr|mn|'000|000)\)", re.IGNORECASE)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# variant (already through steps 1-3) -> canonical. Only variants actually seen in the CSE Data
# Library are listed; the generic rules above handle everything else.
COLUMN_ALIASES: dict[str, str] = {
    # company / security identity
    "company_id": "company_code",
    "company_code": "company_code",
    "company": "company_name",
    "company_name": "company_name",
    "name": "company_name",
    "short_name": "short_name",
    "security": "security_id",
    "security_id": "security_id",
    "security_type": "security_id",  # Market Capitalisation file: the column holds "AAF.N0000"
    "main_type": "main_type",
    "sub_type": "sub_type",
    "isin_code": "isin",
    "isin": "isin",
    # public holding
    "public_float": "public_float",
    "quarter": "quarter",
    # prices
    "trading_date": "trading_date",
    "trade_date": "trade_date",
    "price_high": "price_high",
    "price_low": "price_low",
    "close_price": "close_price",
    "open_price": "open_price",
    "trade_volume": "trade_volume",
    "share_volume": "share_volume",
    "turnover": "turnover",
    # corporate actions
    "date_of_announcement": "date_of_announcement",
    "date_of_initial_announcement": "date_of_announcement",
    "rate_of_dividend": "rate_of_dividend",
    "rate_remarks": "rate_remarks",
    "remarks": "remarks",
    "date_of_ex": "date_of_ex",
    "date_of_payment": "date_of_payment",
    "cum_price": "cum_price",
    "ex_price": "ex_price",
    "old_proportion": "old_proportion",
    "new_proportion": "new_proportion",
    "effective_date": "effective_date",
    "allotment_date": "allotment_date",
    "date_of_p_allotment": "allotment_date",
    "no_of_shares_listed": "shares_listed",
    "date_listed": "date_listed",
    "no": "row_no",
    "issued_quantity": "issued_quantity",
    "market_cap": "market_cap",
    "indexed_price": "indexed_price",
    "indexed_quantity": "indexed_quantity",
}


def normalise_column_name(raw: object) -> str:
    """Return the canonical snake_case name for a header cell (``""`` for a blank cell)."""
    if raw is None:
        return ""
    if isinstance(raw, float) and raw != raw:  # NaN
        return ""
    text = " ".join(str(raw).split())
    text = _UNITS.sub("", text)
    key = _NON_ALNUM.sub("_", text.lower()).strip("_")
    return COLUMN_ALIASES.get(key, key)


def normalise_columns(names: Iterable[object]) -> list[str]:
    """Normalise a whole header row, making duplicates unique (``x``, ``x_2``, ``x_3`` ...)
    and naming blank headers ``unnamed_<position>``."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for pos, raw in enumerate(names):
        name = normalise_column_name(raw) or f"unnamed_{pos}"
        count = seen.get(name, 0) + 1
        seen[name] = count
        out.append(name if count == 1 else f"{name}_{count}")
    return out


def apply_canonical_columns(df: pd.DataFrame, header_cells: Iterable[object]) -> pd.DataFrame:
    """Set ``df.columns`` from raw header cells, normalised and de-duplicated."""
    df = df.copy()
    df.columns = normalise_columns(header_cells)
    return df
