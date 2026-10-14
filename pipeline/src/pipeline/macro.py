"""M1/M2 macro series and features (Chapter 4 §4.7 step 9: macro aligned as-of release date).

Sources (data_sources/macro/SOURCES.md): CBSL indicative USD/LKR spot rate (daily), CBSL policy rates
(SDFR/SLFR, OPR from 27 Nov 2024), CBSL weekly AWPR, CBSL weekly Treasury bill yields, and the DCS
Colombo Consumer Price Index (monthly).

Availability rule (author's decision, 2026-10-14, conservative):
- daily and weekly series and policy-rate changes dated d are used from the first CSE trading day
  strictly after d (the value is never visible on its own date, even if it was published before the
  14:30 close);
- CCPI for month m (released on the last working day of m) is used from the first trading day of
  month m+1.
Every macro feature on trading day t is therefore built only from values public before t.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MACRO_DIR = Path(__file__).resolve().parents[2] / "data_sources" / "macro"

MACRO = ["usd_lkr_chg5", "usd_lkr_chg20", "policy_rate", "awpr", "tbill_91d", "tbill_spread_364_91", "ccpi_yoy"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.strip().replace({"-": None, "": None, "nan": None}), errors="coerce")


def load_macro(folder: Path = MACRO_DIR) -> dict[str, pd.DataFrame]:
    """Raw series, each with an observation date ('date') and its value columns."""
    usd = pd.read_csv(folder / "cbsl_usd_lkr_daily.csv", parse_dates=["date"])
    pol = pd.read_csv(folder / "cbsl_policy_rates.csv", parse_dates=["effective_date"])
    pol = pol.rename(columns={"effective_date": "date"})
    pol["policy_rate"] = (pol["sdfr"] + pol["slfr"]) / 2  # corridor midpoint = OPR from 27 Nov 2024
    aw = pd.read_csv(folder / "cbsl_awpr_weekly.csv", parse_dates=["date"], dtype={"awpr_weekly": str})
    aw["awpr"] = _num(aw["awpr_weekly"])
    tb = pd.read_csv(folder / "cbsl_tbill_weekly.csv", dtype=str)
    tb["date"] = pd.to_datetime(tb["date"], format="%Y-%m-%d")
    for c in ("tb91", "tb182", "tb364"):
        tb[c] = _num(tb[c])
    cp = pd.read_csv(folder / "ccpi_monthly.csv", dtype={"month": str})
    cp["date"] = pd.PeriodIndex(cp["month"], freq="M").to_timestamp(how="end").normalize()  # last day of month m
    return {"usd": usd, "policy": pol, "awpr": aw, "tbill": tb, "ccpi": cp}


def _asof_next_day(cal: pd.DatetimeIndex, obs: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Value on trading day t = latest observation dated strictly before t (carried forward)."""
    o = obs.dropna(subset=cols, how="all").sort_values("date")[["date", *cols]]
    o = o.groupby("date", as_index=False).last()
    left = pd.DataFrame({"t": cal})
    # strictly before t: shift the lookup key by one nanosecond
    left["key"] = left["t"] - pd.Timedelta(1, "ns")
    out = pd.merge_asof(left.sort_values("key"), o.rename(columns={"date": "key"}), on="key", direction="backward")
    return out.set_index("t")[cols].reindex(cal)


def _asof_series_ffill(cal: pd.DatetimeIndex, obs: pd.DataFrame, col: str) -> pd.Series:
    """Like _asof_next_day for one column, skipping missing values (a tenor without an auction that week)."""
    return _asof_next_day(cal, obs[["date", col]].dropna(), [col])[col]


def macro_features(calendar: pd.DatetimeIndex, raw: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Macro feature table indexed by trading date (one row per CSE trading day)."""
    raw = raw or load_macro()
    cal = pd.DatetimeIndex(sorted(calendar))
    out = pd.DataFrame(index=cal)

    # USD/LKR: changes measured on the CBSL's own daily series, then made available from the next trading day
    usd = raw["usd"].sort_values("date").copy()
    lu = np.log(usd["usd_lkr"])
    usd["usd_lkr_chg5"] = lu.diff(5)
    usd["usd_lkr_chg20"] = lu.diff(20)
    u = _asof_next_day(cal, usd, ["usd_lkr_chg5", "usd_lkr_chg20"])
    out[["usd_lkr_chg5", "usd_lkr_chg20"]] = u.values

    out["policy_rate"] = _asof_series_ffill(cal, raw["policy"], "policy_rate").values
    out["awpr"] = _asof_series_ffill(cal, raw["awpr"], "awpr").values
    tb91 = _asof_series_ffill(cal, raw["tbill"], "tb91")
    tb364 = _asof_series_ffill(cal, raw["tbill"], "tb364")
    out["tbill_91d"] = tb91.values
    out["tbill_spread_364_91"] = (tb364 - tb91).values

    # CCPI for month m: usable from the first trading day of m+1 (date = last calendar day of m, strictly before)
    out["ccpi_yoy"] = _asof_series_ffill(cal, raw["ccpi"].rename(columns={"yoy_published": "ccpi_yoy"}), "ccpi_yoy").values
    out.index.name = "date"
    return out[MACRO]
