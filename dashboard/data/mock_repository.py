"""
MockRepository — realistic, reproducible synthetic data.

Determinism: every method derives its RNG seed from a fixed base seed plus a
hash of its arguments, so the same query always returns the same frame across
reruns and sessions. Nothing here is real market data.
"""
from __future__ import annotations

import time
import zlib
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import mock_scenario
from data.errors import NoData
from data.repository import Horizon, Repository

BASE_SEED = 20260918

# Illustrative S&P SL20-style universe. Names are real listed companies used
# purely as labels; every number attached to them is synthetic.
_UNIVERSE = [
    ("JKH", "John Keells Holdings", "Diversified"),
    ("COMB", "Commercial Bank of Ceylon", "Banks"),
    ("HNB", "Hatton National Bank", "Banks"),
    ("SAMP", "Sampath Bank", "Banks"),
    ("DIAL", "Dialog Axiata", "Telecom"),
    ("LOLC", "LOLC Holdings", "Diversified Financials"),
    ("EXPO", "Expolanka Holdings", "Transport"),
    ("CTC", "Ceylon Tobacco Company", "Consumer Staples"),
    ("MELS", "Melstacorp", "Beverages"),
    ("HAYL", "Hayleys", "Diversified"),
    ("RCL", "Royal Ceramics Lanka", "Materials"),
    ("SPEN", "Aitken Spence", "Diversified"),
    ("LIOC", "Lanka IOC", "Energy"),
    ("NDB", "National Development Bank", "Banks"),
    ("VONE", "Vallibel One", "Diversified"),
    ("HHL", "Hemas Holdings", "Health Care"),
    ("DIST", "Distilleries Company", "Beverages"),
    ("CARG", "Cargills (Ceylon)", "Retail"),
    ("TJL", "Teejay Lanka", "Textiles"),
    ("SLTL", "Sri Lanka Telecom", "Telecom"),
]

_FEATURES = [
    "RSI (14)", "MACD histogram", "5-day return", "Company sentiment (3-day decay)",
    "ASPI 1-day return", "USD/LKR change", "Bollinger %B", "Volume ratio (20-day)",
    "ATR (14)", "Market sentiment", "T-bill yield change", "OBV slope",
    "CCPI inflation (YoY)", "SL20 5-day return", "Sector momentum",
]

_HEADLINES = {
    "en": [
        "{name} reports stronger quarterly earnings on margin recovery",
        "{name} board approves interim dividend",
        "Analysts cautious on {name} as input costs rise",
        "{name} expands regional operations",
        "CBSL holds policy rate; banks including {name} steady",
    ],
    "si": [
        "{name} කාර්තුව සඳහා ලාභය ඉහළට",
        "{name} අතුරු ලාභාංශයක් ප්‍රකාශයට පත් කරයි",
        "{name} සමාගමේ පිරිවැය ඉහළ යාම ගැන විශ්ලේෂකයින් අවධානයෙන්",
        "{name} නව ව්‍යාපෘතියක් ආරම්භ කරයි",
    ],
    "ta": [
        "{name} காலாண்டு லாபம் உயர்வு",
        "{name} இடைக்கால ஈவுத்தொகை அறிவிப்பு",
        "{name} செலவு உயர்வால் ஆய்வாளர்கள் எச்சரிக்கை",
        "{name} புதிய முதலீட்டு திட்டம் அறிவிப்பு",
    ],
}
_SOURCES = {"en": ["Ada Derana", "Daily FT", "EconomyNext"],
            "si": ["Hiru News", "Lankadeepa"],
            "ta": ["Virakesari", "Thinakaran"]}


def _rng(*parts) -> np.random.Generator:
    key = zlib.crc32(repr(parts).encode()) & 0xFFFFFFFF
    return np.random.default_rng(BASE_SEED + key)


def _bdays(start: date, end: date) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end)


class MockRepository(Repository):
    MODEL_VERSION = "xgb-v0.0-mock"

    def __init__(self) -> None:
        self._stocks = pd.DataFrame(_UNIVERSE, columns=["ticker", "name", "sector"])
        self._as_of = date(2026, 9, 17)
        self._cold_t0 = 0.0   # cold_start scenario bookkeeping
        self._cold_n = 0

    # ----- mock-only failure scenarios (config.MOCK_SCENARIOS) -------------
    def _scenario(self, *no_data_for: str) -> str:
        """Apply this session's `?scenario=` hook, if any.

        db_down     → every call raises a connection error (unreachable panel).
        cold_start  → the first two calls in any 60-second window raise, then
                      succeed (exercises the waking message + retry backoff).
        no_*        → raise NoData when the scenario is in `no_data_for`.
        Returns the scenario name so callers can soft-handle it (empty frames).
        """
        s = mock_scenario()
        if not s:
            return ""
        if s == "db_down":
            raise ConnectionError("simulated outage: database unreachable")
        if s == "cold_start":
            now = time.time()
            if now - self._cold_t0 > 60:
                self._cold_t0, self._cold_n = now, 0
            self._cold_n += 1
            if self._cold_n <= 2:
                raise ConnectionError("simulated cold start: compute still waking")
            return ""
        if s in no_data_for:
            raise NoData(f"nothing stored for this selection ({s.replace('_', ' ')})")
        return s

    # ----- status ---------------------------------------------------------
    def get_data_status(self) -> dict:
        self._scenario()
        return {"as_of": self._as_of, "source": "mock", "model_version": self.MODEL_VERSION}

    # ----- universe -------------------------------------------------------
    def list_stocks(self) -> pd.DataFrame:
        self._scenario()
        return self._stocks.copy()

    def _name(self, ticker: str) -> str:
        row = self._stocks[self._stocks.ticker == ticker]
        return row.name.iloc[0] if len(row) else ticker

    # ----- market overview ------------------------------------------------
    def get_index_history(self, start: date, end: date) -> pd.DataFrame:
        self._scenario()
        # Generate from a fixed anchor so windows are consistent.
        anchor = date(2021, 1, 1)
        idx = _bdays(anchor, max(end, anchor + timedelta(days=30)))
        r = _rng("index")
        n = len(idx)
        # ASPI: crisis dip in 2022, recovery afterwards.
        drift = np.where((idx.year == 2022), -0.0012, 0.0006)
        vol = np.where((idx.year == 2022), 0.022, 0.011)
        aspi = 8500 * np.exp(np.cumsum(drift + vol * r.standard_normal(n)))
        sl20 = 2600 * np.exp(np.cumsum(drift * 1.1 + vol * 1.15 * r.standard_normal(n)))
        df = pd.DataFrame({"date": idx.date, "aspi": aspi.round(2), "sl20": sl20.round(2)})
        return df[(df.date >= start) & (df.date <= end)].reset_index(drop=True)

    def get_market_breadth(self, as_of: date) -> dict:
        self._scenario()
        r = _rng("breadth", as_of)
        adv = int(r.integers(70, 160))
        dec = int(r.integers(60, 150))
        return {"advances": adv, "declines": dec, "unchanged": int(286 - adv - dec)}

    def get_top_movers(self, as_of: date, n: int = 5) -> pd.DataFrame:
        self._scenario()
        r = _rng("movers", as_of)
        df = self._stocks.copy()
        df["close"] = (r.uniform(20, 900, len(df))).round(2)
        df["change_pct"] = (r.normal(0, 2.4, len(df))).round(2)
        return df.sort_values("change_pct", ascending=False).reset_index(drop=True)

    def get_market_sentiment(self, as_of: date) -> dict:
        self._scenario()
        r = _rng("mkt_sent", as_of)
        return {"score": float(np.clip(r.normal(0.12, 0.3), -1, 1)),
                "article_count": int(r.integers(40, 140))}

    def get_market_risk_summary(self, as_of: date) -> dict:
        self._scenario()
        r = _rng("mkt_risk", as_of)
        vol = float(r.uniform(0.9, 1.9))
        regime = "Elevated" if vol > 1.5 else "Moderate" if vol > 1.1 else "Calm"
        return {"var_1d_95": round(vol * 1.65, 2), "realised_vol_20d": round(vol, 2),
                "regime": regime}

    # ----- prices & predictions ------------------------------------------
    def get_price_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self._scenario()
        anchor = date(2021, 1, 1)
        idx = _bdays(anchor, max(end, anchor + timedelta(days=30)))
        r = _rng("price", ticker)
        n = len(idx)
        p0 = float(r.uniform(30, 600))
        vol = np.where(idx.year == 2022, 0.03, 0.016)
        close = p0 * np.exp(np.cumsum(0.0004 + vol * r.standard_normal(n)))
        spread = close * r.uniform(0.005, 0.02, n)
        df = pd.DataFrame({
            "date": idx.date,
            "open": (close + r.normal(0, 1, n) * spread).round(2),
            "high": (close + spread).round(2),
            "low": (close - spread).round(2),
            "close": close.round(2),
            "volume": r.integers(20_000, 1_200_000, n),
        })
        return df[(df.date >= start) & (df.date <= end)].reset_index(drop=True)

    def get_prediction(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        self._scenario("no_prediction")
        r = _rng("pred", ticker, as_of, horizon)
        p_up = float(np.clip(r.beta(2.2, 2.0), 0.05, 0.95))
        days = 1 if horizon == "t+1" else 5
        target = pd.bdate_range(as_of, periods=days + 1)[-1].date()
        return {"direction": "up" if p_up >= 0.5 else "down", "p_up": round(p_up, 3), "threshold": 0.5,
                "model": "XGBoost" if r.random() > 0.35 else "GRU",
                "model_version": self.MODEL_VERSION, "target_date": target}

    def get_signal_reliability(self, horizon: Horizon) -> pd.DataFrame:
        # synthetic, shaped like the published table (mock mode only)
        acc = (0.57, 0.59, 0.61, 0.65) if horizon == "t+1" else (0.53, 0.54, 0.54, 0.50)
        return pd.DataFrame({"min_distance": [0.0, 0.05, 0.10, 0.15], "coverage": [1.0, 0.6, 0.27, 0.1],
                             "accuracy": list(acc)})

    def get_block_accuracy(self, ticker: str, horizon: Horizon) -> pd.DataFrame:
        self._scenario("no_prediction")
        r = _rng("blocks", ticker, horizon)
        rows, start = [], date(2024, 1, 2)
        for b in range(1, 9):
            s = pd.bdate_range(start, periods=63 * b)[-63].date()
            e = pd.bdate_range(start, periods=63 * b)[-1].date()
            acc = float(np.clip(r.normal(0.56 if horizon == "t+1" else 0.60, 0.04), 0.45, 0.72))
            rows.append({"block": b, "test_start": s, "test_end": e, "accuracy": round(acc, 3),
                         "f1": round(acc - r.uniform(0, 0.03), 3),
                         "precision": round(acc + r.normal(0, 0.02), 3),
                         "recall": round(acc + r.normal(0, 0.02), 3), "n": 63})
        return pd.DataFrame(rows)

    # ----- explainability -------------------------------------------------
    def get_global_importance(self, horizon: Horizon) -> pd.DataFrame:
        self._scenario()
        r = _rng("global_shap", horizon)
        imp = np.sort(r.gamma(2.0, 0.04, len(_FEATURES)))[::-1]
        return pd.DataFrame({"feature": _FEATURES, "mean_abs_shap": imp.round(4)})

    def get_shap_explanation(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        self._scenario("no_prediction")
        r = _rng("local_shap", ticker, as_of, horizon)
        pred = self.get_prediction(ticker, as_of, horizon)
        logit = float(np.log(pred["p_up"] / (1 - pred["p_up"])))
        raw = r.normal(0, 0.3, len(_FEATURES))
        raw += (logit - 0.05 - raw.sum()) / len(_FEATURES)  # additive: base + Σshap = logit
        df = pd.DataFrame({"feature": _FEATURES, "value": r.normal(0, 1, len(_FEATURES)).round(2),
                           "shap": raw.round(4)})
        df = df.reindex(df.shap.abs().sort_values(ascending=False).index).head(10)
        return {"base_value": 0.05, "contributions": df.reset_index(drop=True)}

    def get_narrative(self, ticker: str, as_of: date, horizon: Horizon) -> str:
        self._scenario("no_prediction")
        pred = self.get_prediction(ticker, as_of, horizon)
        exp = self.get_shap_explanation(ticker, as_of, horizon)["contributions"]
        top_pos = exp[exp.shap > 0].head(2).feature.tolist()
        top_neg = exp[exp.shap < 0].head(2).feature.tolist()
        name = self._name(ticker)
        hz = "tomorrow" if horizon == "t+1" else "over the next five trading days"
        lean = "more likely to rise than fall" if pred["direction"] == "up" else "more likely to fall than rise"
        pct = int(round(pred["p_up"] * 100))
        return (
            f"For {name}, the {pred['model']} model estimates a {pct}% chance of an upward move {hz}, "
            f"so the share is judged {lean}. The strongest signals pushing towards a rise are "
            f"{' and '.join(top_pos) if top_pos else 'none of note'}; the main signals pulling the other way are "
            f"{' and '.join(top_neg) if top_neg else 'none of note'}. A probability near 50% means the model is "
            f"close to a coin toss — treat it as weak evidence, not a call to act."
        )

    # ----- risk -----------------------------------------------------------
    def get_risk_parameters(self, ticker: str, as_of: date) -> dict:
        self._scenario("no_risk")
        r = _rng("garch", ticker)
        return {"mu": 0.0003, "omega": 2.5e-6, "alpha": float(r.uniform(0.04, 0.08)),
                "gamma": float(r.uniform(0.03, 0.09)), "beta": float(r.uniform(0.82, 0.90)),
                "nu": float(r.uniform(4.0, 7.0)), "sigma_last": float(r.uniform(0.012, 0.024)),
                "ret_last": float(r.normal(0, 0.015)),
                "crisis_vol_multiplier": 2.4, "crisis_shock": -0.06}

    def get_var_estimate(self, ticker: str, as_of: date) -> pd.DataFrame:
        self._scenario("no_risk")
        p = self.get_risk_parameters(ticker, as_of)
        s = p["sigma_last"]
        rows = []
        for h in (1, 5):
            for c, z in ((0.95, 1.85), (0.99, 2.9)):  # t-ish quantiles
                rows.append({"horizon_days": h, "confidence": c,
                             "var_pct": round(100 * z * s * np.sqrt(h), 2)})
        return pd.DataFrame(rows)

    def get_kupiec_results(self, ticker: str) -> pd.DataFrame:
        self._scenario("no_risk")
        r = _rng("kupiec", ticker)
        rows = []
        T = 500
        for c in (0.95, 0.99):
            expected = T * (1 - c)
            x = int(r.poisson(expected))
            p = 1 - c
            with np.errstate(divide="ignore"):
                lr = -2 * ((T - x) * np.log(1 - p) + x * np.log(p)) + \
                     2 * ((T - x) * np.log(1 - x / T) + (x * np.log(x / T) if x else 0))
            lr = float(max(lr, 0))
            pval = float(np.exp(-lr / 2))  # chi2(1) survival approximation
            rows.append({"confidence": c, "observations": T, "breaches": x,
                         "expected_breaches": round(expected, 1), "lr_pof": round(lr, 3),
                         "p_value": round(pval, 3),
                         "verdict": "Accept" if pval > 0.05 else "Reject"})
        return pd.DataFrame(rows)

    # ----- news sentiment -------------------------------------------------
    def get_sentiment_series(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if self._scenario() == "no_news":
            return pd.DataFrame(columns=["date", "company_score", "market_score", "article_count"])
        idx = _bdays(start, end)
        r = _rng("sent", ticker)
        n = len(idx)
        mkt = np.clip(np.cumsum(r.normal(0, 0.06, n)) * 0.3, -1, 1)
        comp = np.clip(mkt * 0.6 + r.normal(0, 0.25, n), -1, 1)
        return pd.DataFrame({"date": idx.date, "company_score": comp.round(3),
                             "market_score": mkt.round(3),
                             "article_count": r.poisson(2.2, n)})

    def get_articles(self, ticker: str | None, start: date, end: date) -> pd.DataFrame:
        if self._scenario() == "no_news":
            return pd.DataFrame(columns=["published", "headline", "source", "language", "url", "ticker", "score"])
        r = _rng("articles", ticker, start, end)
        n = int(r.integers(18, 32))
        rows = []
        for _ in range(n):
            t = ticker or str(r.choice(self._stocks.ticker))
            lang = str(r.choice(["en", "si", "ta"], p=[0.5, 0.3, 0.2]))
            tmpl = str(r.choice(_HEADLINES[lang]))
            day = start + timedelta(days=int(r.integers(0, max(1, (end - start).days + 1))))
            rows.append({"published": day, "headline": tmpl.format(name=self._name(t)),
                         "source": str(r.choice(_SOURCES[lang])), "language": lang,
                         "url": f"https://example.org/news/{t.lower()}-{int(r.integers(1000, 9999))}",
                         "ticker": t, "score": round(float(np.clip(r.normal(0.1, 0.45), -1, 1)), 3)})
        return pd.DataFrame(rows).sort_values("published", ascending=False).reset_index(drop=True)
