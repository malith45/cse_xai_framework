"""M1 - index levels and trading calendar (Chapter 4 §4.7 step 2).

The trading calendar is the set of dates on which the ASPI was published in
``07Market Indices - Daily.xls`` (CLAUDE.md §4.1.9: never derive it from a single stock).
From 2020-01-02 the sheet carries a second header row; only the block below it is used, whose
columns are Date, ASPI, (blank MPI), S&P SL20 and the GICS industry-group indices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from readers import load_raw_sheets
from readers.columns import normalise_column_name

from .prices import STUDY_END, STUDY_START


def load_indices(root: Path) -> pd.DataFrame:
    raw = load_raw_sheets(root / "07Market Indices - Daily.xls", ["Index"])["Index"]
    hdr = next(i for i in range(len(raw)) if str(raw.iat[i, 0]).strip() == "Date")
    cols = [normalise_column_name(c) or f"unnamed_{j}" for j, c in enumerate(raw.iloc[hdr].tolist())]
    df = raw.iloc[hdr + 1:].copy()
    df.columns = cols
    df = df.rename(columns={"date": "date", "all_share_price_index": "aspi", "s_p_sri_lanka_20_index": "sl20"})
    df = df[[c for c in df.columns if not c.startswith("unnamed_")]]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()]
    for c in df.columns:
        if c != "date":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[(df["date"] >= STUDY_START) & (df["date"] <= STUDY_END) & df["aspi"].notna()]
    return df.sort_values("date").reset_index(drop=True)


def trading_calendar(indices: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(indices["date"].sort_values().unique(), name="date")
