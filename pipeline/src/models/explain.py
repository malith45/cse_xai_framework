"""M5 Explainability Engine (§4.6.3): SHAP attributions and rule-based narratives (no LLM).

XGBoost attributions are exact TreeSHAP values computed by XGBoost's own implementation of the
Lundberg et al. (2020) algorithm (``pred_contribs=True``), which supports the categorical stock
identifier natively. Values are in log-odds units and satisfy f(x) = phi_0 + sum(phi_i) (eq. 4.3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from . import config as C

LABELS = {
    "close_sma5": "price vs its 5-day average", "close_sma10": "price vs its 10-day average",
    "close_sma20": "price vs its 20-day average", "close_sma50": "price vs its 50-day average",
    "macd": "MACD trend signal", "macd_signal": "MACD signal line", "macd_hist": "MACD momentum (histogram)",
    "rsi14": "RSI (14-day overbought/oversold)", "bb_pctb": "position inside the Bollinger Bands",
    "bb_width": "Bollinger Band width (volatility)", "atr14": "average true range (volatility)",
    "obv_chg20": "on-balance volume change (20 days)", "roc5": "5-day price change", "roc10": "10-day price change",
    "roc20": "20-day price change", "ret_lag1": "today's return", "ret_lag2": "yesterday's return",
    "ret_lag3": "return 2 days ago", "ret_lag4": "return 3 days ago", "ret_lag5": "return 4 days ago",
    "rv5": "5-day volatility", "rv20": "20-day volatility", "rv60": "60-day volatility",
    "hl_range": "today's high-low range", "co_ret": "today's open-to-close move",
    "gap_days": "days since the previous trade", "log_turnover_z20": "turnover vs its 20-day norm",
    "trades_z20": "number of trades vs its 20-day norm", "aspi_ret1": "ASPI change today",
    "aspi_ret5": "ASPI change over 5 days", "sl20_ret1": "S&P SL20 change today",
    "sl20_ret5": "S&P SL20 change over 5 days", "aspi_rv20": "market volatility (20 days)",
    "aspi_close_sma50": "ASPI vs its 50-day average", "rel_ret5": "5-day return relative to the S&P SL20",
    "stock": "company identity (its own history)",
    "usd_lkr_chg5": "rupee vs US dollar, 5-day change", "usd_lkr_chg20": "rupee vs US dollar, 20-day change",
    "policy_rate": "Central Bank policy rate", "awpr": "prime lending rate (AWPR)",
    "tbill_91d": "3-month Treasury bill yield", "tbill_spread_364_91": "1-year minus 3-month T-bill yield",
    "ccpi_yoy": "inflation (Colombo CPI, year on year)",
    "sent_co_mean": "today's news tone about the company", "sent_co_count": "number of news articles about the company today",
    "sent_co_decay3": "news tone about the company, last 3 days", "sent_mkt_mean": "today's market news tone",
    "sent_mkt_count": "number of market news articles today", "sent_mkt_decay3": "market news tone, last 3 days",
}


def tree_shap(model: xgb.XGBClassifier, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (phi [n, k], base [n]) for the rows of X."""
    dm = xgb.DMatrix(X, enable_categorical=True)
    contrib = model.get_booster().predict(dm, pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]


def top_k_long(phi: np.ndarray, base: np.ndarray, X_raw: pd.DataFrame, cols: list[str], keys: pd.DataFrame,
               k: int = C.SHAP_TOP_K) -> pd.DataFrame:
    """Long table: one row per (prediction, rank) with feature, raw value and SHAP value."""
    order = np.argsort(-np.abs(phi), axis=1)[:, :k]
    n = len(phi)
    idx = np.repeat(np.arange(n), k)
    feat_idx = order.ravel()
    raw_vals = X_raw.reindex(columns=cols).to_numpy(dtype=object)
    out = pd.DataFrame({
        "security_id": keys["security_id"].values[idx],
        "date": keys["date"].values[idx],
        "rank": np.tile(np.arange(1, k + 1), n),
        "feature": np.array(cols)[feat_idx],
        "feature_value": [float(raw_vals[i, j]) if not isinstance(raw_vals[i, j], str) else np.nan
                          for i, j in zip(idx, feat_idx)],
        "shap_value": phi[idx, feat_idx],
        "base_value": base[idx],
    })
    return out


def narrative(name: str, p_up: float, horizon: str, top: pd.DataFrame) -> str:
    """Template narrative from the top contributions of one prediction (no LLM, no free text)."""
    when = "on the next trading day" if horizon == "t1" else "over the next five trading days"
    direction = "higher" if p_up >= 0.5 else "lower"
    pct = round(100 * (p_up if p_up >= 0.5 else 1 - p_up))
    ups = top[top["shap_value"] > 0].head(2)
    downs = top[top["shap_value"] < 0].head(2)
    parts = [f"The model estimates a {pct}% chance that {name} closes {direction} {when}."]
    if len(ups):
        parts.append("Pushing the estimate up: " + "; ".join(LABELS.get(f, f) for f in ups["feature"]) + ".")
    if len(downs):
        parts.append("Pushing it down: " + "; ".join(LABELS.get(f, f) for f in downs["feature"]) + ".")
    if 0.45 <= p_up <= 0.55:
        parts.append("The estimate is close to 50%, so the model has no clear view.")
    parts.append("This is decision support, not financial advice.")
    return " ".join(parts)
