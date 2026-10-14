"""Benchmark forecasts (FR09): naive persistence, ARIMA and GARCH. Benchmarks only.

* Naive persistence: tomorrow's direction = today's direction (t+1); the next five days' direction =
  the last five days' direction (t+5).
* ARIMA: per security and block, the order (p, q) in {0,1,2}^2 with d = 0 on daily log returns is
  chosen by AIC on the training window; parameters are then held fixed while the model is updated
  with each new observation through the test block (``append(refit=False)``), so every forecast uses
  only data up to day t. Direction = sign of the 1-step (t+1) or summed 1-5-step (t+5) mean forecast.
* GARCH: AR(1)-GARCH(1,1) with Student-t errors (``arch``), fitted per security and block on the
  training window, parameters fixed through the test block; direction = sign of the conditional-mean
  forecast. (A GARCH model's variance forecast carries no sign, so the mean equation is what gives
  it a directional call.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA


def naive(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel[["security_id", "date", "cal_pos"]].copy()
    out["t1"] = (panel["lr"] > 0).astype(float)
    out["t5"] = (panel["roc5"] > 0).astype(float)
    out.loc[panel["lr"].isna(), "t1"] = np.nan
    out.loc[panel["roc5"].isna(), "t5"] = np.nan
    return out


def _arima_one(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in range(3):
            for q in range(3):
                try:
                    r = ARIMA(train, order=(p, 0, q), trend="c").fit()
                    if r.aic < best_aic:
                        best, best_aic = r, r.aic
                except Exception:
                    continue
        if best is None:
            return np.full(len(test), np.nan), np.full(len(test), np.nan)
        f1, f5 = np.empty(len(test)), np.empty(len(test))
        res = best
        for i, x in enumerate(test):
            res = res.append([x], refit=False)
            fc = res.forecast(5)
            f1[i], f5[i] = fc[0], fc.sum()
    return f1, f5


def _garch_one(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = np.concatenate([train, test]) * 100
        try:
            am = arch_model(full, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="t", rescale=False)
            res = am.fit(last_obs=len(train), disp="off")
            fc = res.forecast(horizon=5, start=len(train), reindex=False)
            m = fc.mean.values
            return m[:, 0], m.sum(axis=1)
        except Exception:
            return np.full(len(test), np.nan), np.full(len(test), np.nan)


def model_baseline(panel: pd.DataFrame, blk, kind: str) -> pd.DataFrame:
    """Forecasts for every (security, test day) of block ``blk``. Uses each security's own traded-day
    log-return series; the forecast on day t is made after observing r_t."""
    rows = []
    fn = _arima_one if kind == "arima" else _garch_one
    for sid, g in panel.groupby("security_id"):
        g = g.sort_values("cal_pos")
        tr = g[(g["cal_pos"] < blk.train[1]) & g["lr"].notna()]
        te = g[(g["cal_pos"] >= blk.test[0]) & (g["cal_pos"] < blk.test[1]) & g["lr"].notna()]
        if len(tr) < 250 or te.empty:
            continue
        # observations between the end of training and the test start (the purge gap) are fed to
        # the filter too, so the state at day t reflects all returns up to t
        gap = g[(g["cal_pos"] >= blk.train[1]) & (g["cal_pos"] < blk.test[0]) & g["lr"].notna()]
        series_test = np.concatenate([gap["lr"].values, te["lr"].values])
        f1, f5 = fn(tr["lr"].values[-750:], series_test)
        f1, f5 = f1[len(gap):], f5[len(gap):]
        rows.append(pd.DataFrame({"security_id": sid, "date": te["date"].values, "cal_pos": te["cal_pos"].values,
                                  "t1": (f1 > 0).astype(float), "t5": (f5 > 0).astype(float),
                                  "t1_mean": f1, "t5_mean": f5}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
