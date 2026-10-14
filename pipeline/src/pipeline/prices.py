"""M1 Data Ingestion & Validation - daily share prices 2020-2025 (Chapter 4 §4.7 steps 1-3).

Reads the six flat-layout daily price files of the CSE Data Library into one long table keyed on
``security_id`` (canonical ``CODE-CLASS-SUBTYPE``) and ``date``.

The 2023 CSV repair (decided by the author, 2026-10-13: "rebuild from fragments only")
-----------------------------------------------------------------------------------------
Every number >= 1,000 that was written *unquoted* in ``2023 Data.csv`` split on its thousands
separator, and the file was later re-saved, so leading zeros of the fragment were lost:
``1,050.00`` became the two cells ``1.00`` and ``50.00``. A value is therefore recovered exactly as
``thousands * 1000 + remainder``. Which of the seven numeric fields were split is not marked, so
every assignment of the extra cells to fields is enumerated and checked against hard constraints:

* a split field's thousands part is an integer 1-999 and its remainder is in [0, 1000);
* volumes are non-negative integers and number of trades <= number of shares;
* high >= low, and the four prices lie within a factor of 1.5 of each other;
* the volume-weighted price ``turnover / shares`` lies inside [low, high] (1% tolerance).

A row is repaired only when exactly one assignment satisfies all constraints; if several do, the
one whose close is nearest the security's previous valid close is taken and the row is counted as
"resolved by continuity". Rows with no valid assignment are dropped and counted. All three counts
are returned in :class:`RepairReport` for Chapter 5.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from identifiers import parse_components
from readers import read_workbook, spilled_rows

PRICE_COLS = [
    "company_code", "main_type", "sub_type", "short_name", "trading_date",
    "price_high", "price_low", "close_price", "open_price", "trade_volume", "share_volume", "turnover",
]
# A repaired row's four prices must lie within this ratio of each other (CSE daily price bands are
# far narrower; the close can be a stale reference price, hence the generous bound).
MAX_INTRADAY_RATIO = 1.5
NUMERIC = ["price_high", "price_low", "close_price", "open_price", "trade_volume", "share_volume", "turnover"]

# Study period decided 2026-09-18 (CLAUDE.md §4.5)
STUDY_START = pd.Timestamp("2020-01-02")
STUDY_END = pd.Timestamp("2025-12-31")


def price_files(root: Path) -> dict[int, Path]:
    flat = root / "33Daily Shares Price List -2021-2025" / "40Daily Shares Price List -2021-2025"
    return {
        2020: root / "32Daily Shares Price List -2011-2020" / "2020 Data.xls",
        2021: flat / "2021 Data.xls",
        2022: flat / "2022 Data.xls",
        2023: flat / "2023 Data.csv",
        2024: flat / "2024 Data.xls",
        2025: flat / "2025 Data.xlsx",
    }


# --------------------------------------------------------------------------- parsing helpers

def _to_number(v: object) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s or s in {"-", "nan", "NaN"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _to_date(v: object) -> pd.Timestamp:
    if isinstance(v, (pd.Timestamp,)):
        return v.normalize()
    if hasattr(v, "year") and hasattr(v, "month"):
        return pd.Timestamp(v).normalize()
    s = str(v).strip()
    for fmt in ("%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


# --------------------------------------------------------------------------- 2023 repair

@dataclass
class RepairReport:
    spilled: int = 0
    repaired_unique: int = 0
    repaired_by_continuity: int = 0
    dropped: int = 0
    by_company: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "spilled_rows": self.spilled,
            "repaired_unique": self.repaired_unique,
            "repaired_by_continuity": self.repaired_by_continuity,
            "dropped_unrepairable": self.dropped,
            "by_company": self.by_company,
        }


def _is_int(x: float) -> bool:
    return not math.isnan(x) and abs(x - round(x)) < 1e-9


def candidate_assignments(tokens: list[float]) -> list[list[float]]:
    """All 7-field readings of ``tokens`` that satisfy the hard constraints (see module docstring)."""
    n_fields = len(NUMERIC)
    k = len(tokens) - n_fields
    out: list[list[float]] = []
    if k < 0:
        return out
    for split in itertools.combinations(range(n_fields), k):
        vals: list[float] = []
        i, ok = 0, True
        for f in range(n_fields):
            if f in split:
                a, b = tokens[i], tokens[i + 1]
                if not (_is_int(a) and 1 <= a <= 999 and 0 <= b < 1000):
                    ok = False
                    break
                vals.append(a * 1000 + b)
                i += 2
            else:
                vals.append(tokens[i])
                i += 1
        if not ok or any(math.isnan(v) for v in vals):
            continue
        high, low, close, open_, trades, shares, turnover = vals
        if not (_is_int(trades) and _is_int(shares) and trades >= 0 and shares >= 0 and trades <= shares):
            continue
        if high < low or low <= 0:
            continue
        prices = (high, low, close, open_)
        if min(prices) <= 0 or max(prices) / min(prices) > MAX_INTRADAY_RATIO:
            continue
        if shares > 0:
            vwap = turnover / shares
            if not (low * 0.99 <= vwap <= high * 1.01):
                continue
        out.append(vals)
    return out


def repair_2023(df: pd.DataFrame) -> tuple[pd.DataFrame, RepairReport]:
    """Repair the ragged rows of ``2023 Data.csv`` in place (returns a copy) and report counts."""
    df = df.copy()
    rep = RepairReport()
    sp = spilled_rows(df)
    for c in NUMERIC + [c for c in df.columns if str(c).startswith("unnamed_")]:
        df[c] = df[c].astype(object)
    rep.spilled = len(sp)
    extra = [c for c in df.columns if str(c).startswith("unnamed_")]
    token_cols = NUMERIC + extra
    clean_close: dict[str, float] = {}
    drop: list[int] = []
    sp_set = set(sp)
    for idx in df.index:
        code = str(df.at[idx, "company_code"]).strip()
        if idx not in sp_set:
            c = _to_number(df.at[idx, "close_price"])
            if not math.isnan(c):
                clean_close[code] = c
            continue
        tokens = [_to_number(df.at[idx, c]) for c in token_cols]
        while tokens and math.isnan(tokens[-1]):
            tokens.pop()
        cands = candidate_assignments(tokens)
        if not cands:
            drop.append(idx)
            rep.dropped += 1
            continue
        if len(cands) == 1:
            chosen = cands[0]
            rep.repaired_unique += 1
        else:
            prev = clean_close.get(code)
            if prev is None:
                drop.append(idx)
                rep.dropped += 1
                continue
            chosen = min(cands, key=lambda v: abs(v[2] - prev))
            rep.repaired_by_continuity += 1
        for c, v in zip(NUMERIC, chosen):
            df.at[idx, c] = v
        for c in extra:
            df.at[idx, c] = np.nan
        clean_close[code] = chosen[2]
        rep.by_company[code] = rep.by_company.get(code, 0) + 1
    df = df.drop(index=drop).drop(columns=extra)
    return df, rep


# --------------------------------------------------------------------------- main loader

@dataclass
class LoadReport:
    rows_raw: int = 0
    rows_blank_dropped: int = 0
    rows_bad_date: int = 0
    rows_duplicate: int = 0
    repair_2023: dict = field(default_factory=dict)
    close_outside_range: int = 0
    rows_final: int = 0


def load_prices(root: Path, main_types: tuple[str, ...] = ("N",)) -> tuple[pd.DataFrame, LoadReport]:
    """Load 2020-2025 daily prices -> long DataFrame (security_id, date, OHLCV...) and a report.

    Close prices outside [low, high] are FLAGGED in ``close_outside_range`` and kept unchanged
    (author decision 2026-09-18).
    """
    report = LoadReport()
    frames = []
    for year, path in price_files(root).items():
        sheets = read_workbook(path, PRICE_COLS, required=["company_code", "trading_date", "close_price"])
        for name, df in sheets.items():
            df = df.copy()
            if year == 2023:
                df, rep = repair_2023(df)
                report.repair_2023 = rep.as_dict()
            else:
                extra = [c for c in df.columns if str(c).startswith("unnamed_")]
                df = df.drop(columns=extra)
            df["source_year"] = year
            frames.append(df[PRICE_COLS + ["source_year"]])
    raw = pd.concat(frames, ignore_index=True)
    report.rows_raw = len(raw)

    blank = raw["company_code"].isna() | (raw["company_code"].astype(str).str.strip() == "")
    report.rows_blank_dropped = int(blank.sum())
    raw = raw[~blank].copy()

    raw["date"] = raw["trading_date"].map(_to_date)
    bad = raw["date"].isna()
    report.rows_bad_date = int(bad.sum())
    raw = raw[~bad]
    for c in NUMERIC:
        raw[c] = raw[c].map(_to_number).astype(float)

    ids = []
    for code, cls, sub in zip(raw["company_code"], raw["main_type"], raw["sub_type"]):
        try:
            sid = parse_components(code, cls, sub)
            ids.append((str(sid), getattr(sid, "company_code", str(code).strip()), getattr(sid, "share_class", None)))
        except Exception:
            ids.append((None, str(code).strip(), None))
    raw["security_id"] = [t[0] for t in ids]
    raw["code"] = [t[1] for t in ids]
    raw["share_class"] = [t[2] for t in ids]
    raw = raw[raw["security_id"].notna()]
    raw = raw[raw["share_class"].isin(main_types) & raw["security_id"].str.endswith("-0000")]
    raw = raw[(raw["date"] >= STUDY_START) & (raw["date"] <= STUDY_END)]

    dup = raw.duplicated(["security_id", "date"], keep="first")
    report.rows_duplicate = int(dup.sum())
    raw = raw[~dup]

    out = raw[["security_id", "code", "short_name", "date", "open_price", "price_high", "price_low",
               "close_price", "trade_volume", "share_volume", "turnover"]].rename(columns={
        "open_price": "open", "price_high": "high", "price_low": "low", "close_price": "close",
        "trade_volume": "trades", "share_volume": "shares",
    }).sort_values(["security_id", "date"]).reset_index(drop=True)
    out["close_outside_range"] = (out["close"] > out["high"] + 1e-9) | (out["close"] < out["low"] - 1e-9)
    report.close_outside_range = int(out["close_outside_range"].sum())
    report.rows_final = len(out)
    return out, report
