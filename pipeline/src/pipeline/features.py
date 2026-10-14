"""M2 Feature Engineering (Chapter 4 §4.7 steps 6-8, Table 4.6).

Every feature on row (security, t) uses only prices, volumes and index levels published at or before
the close of trading day t; the prediction is made after the close of t. Features are computed on
each security's own *traded* days (a day with no trades has no row: CLAUDE.md §4.1.9), and are
scale-free (ratios, returns, oscillators) so that the pooled stock-day panel (decision D9) is
comparable across securities. Per-stock standardisation is fitted later, inside each walk-forward
training window only (step 11).

Labels (step 7): y1 = 1 if ln(P[t+1]/P[t]) > 0, else 0, where t+1 is the next day of the *market*
trading calendar. If the security did not trade on t+1 the label is undefined (NaN) and the row is
excluded from training and evaluation for that horizon and counted (P5). y5 is defined the same way
with the close five calendar trading days ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Window lengths (Table 4.6 indicators, standard textbook parameters)
SMA_WINDOWS = (5, 10, 20, 50)
EMA_FAST, EMA_SLOW, MACD_SIGNAL = 12, 26, 9   # Appel (2005)
RSI_WINDOW = 14                               # Wilder (1978)
BB_WINDOW, BB_K = 20, 2.0                     # Bollinger (2002)
ATR_WINDOW = 14                               # Wilder (1978)
ROC_WINDOWS = (5, 10, 20)
LAGS = (1, 2, 3, 4, 5)
EPS = 1e-10  # |log return| below this is an unchanged price (float noise from adjustment factors)
VOL_WINDOWS = (5, 20, 60)

TECHNICAL = (
    [f"close_sma{w}" for w in SMA_WINDOWS]
    + ["macd", "macd_signal", "macd_hist", "rsi14", "bb_pctb", "bb_width", "atr14", "obv_chg20"]
    + [f"roc{w}" for w in ROC_WINDOWS]
)
RETURNS_VOL = [f"ret_lag{k}" for k in LAGS] + [f"rv{w}" for w in VOL_WINDOWS] + [
    "hl_range", "co_ret", "gap_days", "log_turnover_z20", "trades_z20",
]
MARKET = ["aspi_ret1", "aspi_ret5", "sl20_ret1", "sl20_ret5", "aspi_rv20", "aspi_close_sma50", "rel_ret5"]


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _security_features(g: pd.DataFrame, cal_pos: pd.Series) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    c, h, l, o = g["adj_close"], g["adj_high"], g["adj_low"], g["adj_open"]
    lr = np.log(c).diff()
    lr = lr.where(lr.abs() > EPS, 0.0).where(lr.notna())   # unchanged price = exactly 0
    out = pd.DataFrame(index=g.index)
    for w in SMA_WINDOWS:
        out[f"close_sma{w}"] = c / c.rolling(w, min_periods=w).mean() - 1
    ema_f = c.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean()
    ema_s = c.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean()
    macd = (ema_f - ema_s) / c
    sig = macd.ewm(span=MACD_SIGNAL, adjust=False, min_periods=MACD_SIGNAL).mean()
    out["macd"], out["macd_signal"], out["macd_hist"] = macd, sig, macd - sig
    out["rsi14"] = _rsi(c, RSI_WINDOW)
    m = c.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    s = c.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    out["bb_pctb"] = (c - (m - BB_K * s)) / (2 * BB_K * s).replace(0, np.nan)
    out["bb_width"] = (2 * BB_K * s) / m
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.ewm(alpha=1 / ATR_WINDOW, adjust=False, min_periods=ATR_WINDOW).mean() / c
    obv = (np.sign(lr.fillna(0)) * g["adj_shares"].fillna(0)).cumsum()
    vol20 = g["adj_shares"].rolling(20, min_periods=20).sum()
    out["obv_chg20"] = (obv - obv.shift(20)) / vol20.replace(0, np.nan)
    for w in ROC_WINDOWS:
        out[f"roc{w}"] = c / c.shift(w) - 1
    for k in LAGS:
        out[f"ret_lag{k}"] = lr.shift(k - 1)
    for w in VOL_WINDOWS:
        out[f"rv{w}"] = lr.rolling(w, min_periods=w).std()
    out["hl_range"] = (h - l) / c
    out["co_ret"] = np.log(c / o)
    pos = g["date"].map(cal_pos)
    out["gap_days"] = pos.diff()  # calendar trading days since the previous trade (1 = traded yesterday)
    lt = np.log1p(g["turnover"].astype(float))
    out["log_turnover_z20"] = (lt - lt.rolling(20, min_periods=20).mean()) / lt.rolling(20, min_periods=20).std()
    tv = g["trades"].astype(float)
    out["trades_z20"] = (tv - tv.rolling(20, min_periods=20).mean()) / tv.rolling(20, min_periods=20).std()
    out["lr"] = lr
    return out


def market_features(indices: pd.DataFrame) -> pd.DataFrame:
    ix = indices.sort_values("date").set_index("date")
    a, s = np.log(ix["aspi"]), np.log(ix["sl20"])
    out = pd.DataFrame(index=ix.index)
    out["aspi_ret1"] = a.diff()
    out["aspi_ret5"] = a.diff(5)
    out["sl20_ret1"] = s.diff()
    out["sl20_ret5"] = s.diff(5)
    out["aspi_rv20"] = a.diff().rolling(20, min_periods=20).std()
    out["aspi_close_sma50"] = ix["aspi"] / ix["aspi"].rolling(50, min_periods=50).mean() - 1
    return out


def build_panel(prices: pd.DataFrame, indices: pd.DataFrame, securities: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Return the point-in-time feature matrix with labels y1, y5 and a label-exclusion report."""
    cal = pd.DatetimeIndex(sorted(indices["date"].unique()))
    cal_pos = pd.Series(np.arange(len(cal)), index=cal)
    p = prices if securities is None else prices[prices["security_id"].isin(securities)]
    feats = []
    for sid, g in p.groupby("security_id", sort=True):
        f = _security_features(g, cal_pos)
        f["security_id"], f["date"] = sid, g.loc[f.index, "date"].values
        f["adj_close"] = g.loc[f.index, "adj_close"].values
        feats.append(f)
    panel = pd.concat(feats, ignore_index=True)
    mk = market_features(indices)
    panel = panel.merge(mk, left_on="date", right_index=True, how="left")
    panel["rel_ret5"] = panel["roc5"].pipe(np.log1p) - panel["sl20_ret5"]

    # labels on the market calendar
    panel["cal_pos"] = panel["date"].map(cal_pos).astype(int)
    key = panel.set_index(["security_id", "cal_pos"])["adj_close"]
    for h, name in ((1, "y1"), (5, "y5")):
        fut = key.reindex(pd.MultiIndex.from_arrays([panel["security_id"], panel["cal_pos"] + h])).values
        r = np.log(fut / panel["adj_close"].values)
        r = np.where(np.abs(r) <= EPS, 0.0, r)
        panel[f"fwd_ret{h}"] = r
        panel[name] = np.where(np.isnan(r), np.nan, (r > 0).astype(float))
    last = panel["cal_pos"].max()
    report = {
        "rows": int(len(panel)),
        "y1_undefined_no_trade_next_day": int((panel["y1"].isna() & (panel["cal_pos"] + 1 <= last)).sum()),
        "y5_undefined_no_trade_t_plus_5": int((panel["y5"].isna() & (panel["cal_pos"] + 5 <= last)).sum()),
        "y1_undefined_end_of_sample": int((panel["cal_pos"] + 1 > last).sum()),
        "y5_undefined_end_of_sample": int((panel["cal_pos"] + 5 > last).sum()),
    }
    return panel.sort_values(["date", "security_id"]).reset_index(drop=True), report


FEATURES = TECHNICAL + RETURNS_VOL + MARKET
