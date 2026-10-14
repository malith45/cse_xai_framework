"""Point-in-time modelling universe (S&P SL20 membership; author decision 2026-09-18).

``work/membership.csv`` (columns: security_id, start_date, end_date, source) is the transcription of
the CSE/S&P rebalancing announcements. A row (security, t) enters training and evaluation only when
the security was a constituent on day t. Until that file exists a PROVISIONAL universe is used: the
30 most liquid ordinary voting shares by median daily turnover with at least 900 traded days. The
provisional universe is survivorship-biased and is labelled as such in every output.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MEMBERSHIP = ROOT / "work" / "membership.csv"


def member_flag(panel: pd.DataFrame) -> pd.Series:
    """True where the security was an S&P SL20 constituent on that date."""
    m = pd.read_csv(MEMBERSHIP, parse_dates=["start_date", "end_date"])
    m["end_date"] = m["end_date"].fillna(pd.Timestamp("2099-12-31"))
    keep = pd.Series(False, index=panel.index)
    for r in m.itertuples():
        keep |= (panel["security_id"] == r.security_id) & (panel["date"] >= r.start_date) & (panel["date"] <= r.end_date)
    return keep


def load_universe_full(panel: pd.DataFrame) -> pd.DataFrame:
    """Every row of every security that was ever a constituent, with a ``member`` flag. Used where a
    model needs a security's history from before it joined the index (RNN look-back windows)."""
    m = pd.read_csv(MEMBERSHIP)
    out = panel[panel["security_id"].isin(set(m["security_id"]))].copy()
    out["member"] = member_flag(out).values
    return out


def load_universe(panel: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if MEMBERSHIP.exists():
        m = pd.read_csv(MEMBERSHIP)
        return panel[member_flag(panel)].copy(), f"point-in-time S&P SL20 ({m['security_id'].nunique()} securities, 2020-2025)"
    prices = pd.read_parquet(ROOT / "work" / "prices_adj.parquet")
    liq = prices.groupby("security_id").agg(t=("turnover", "median"), n=("date", "count"))
    ids = liq[liq["n"] >= 900].sort_values("t", ascending=False).index[:30].tolist()
    return panel[panel["security_id"].isin(ids)].copy(), "PROVISIONAL top-30 liquid (membership file missing)"
