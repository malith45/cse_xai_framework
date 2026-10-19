"""
PostgresRepository — read-only SQL implementation of the Repository contract.

Reads the tables written by the offline pipeline (cse/scripts/publish_dashboard.py, loaded to Neon by
cse/scripts/load_to_postgres.py). Works with any SQLAlchemy URL: Neon Postgres in production
(DATA_SOURCE="postgres", DATABASE_URL in Streamlit secrets, read-only role) and a local SQLite copy
(DATABASE_URL="sqlite:///.../cse_dashboard.sqlite") for development and screenshots.

Every method raises data.errors.NoData when the query is valid but nothing is stored, so pages
show their empty states; any other exception is treated as transient by ResilientRepository.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import sqlalchemy as sa

from data.errors import NoData
from data.repository import Horizon, Repository


class PostgresRepository(Repository):
    def __init__(self) -> None:
        from config import database_url

        url = database_url()
        if not url:
            raise RuntimeError("DATABASE_URL is not set. Add it to Streamlit secrets.")
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            url = "postgresql+psycopg://" + url[len("postgresql://"):]
        self._engine = sa.create_engine(url, pool_pre_ping=True, pool_recycle=240)
        self._sqlite = self._engine.dialect.name == "sqlite"
        self._tables: set[str] | None = None

    # ------------------------------------------------------------------ helpers
    def _p(self, v):
        """Bind-parameter adapter: ISO strings for SQLite, native dates for Postgres."""
        if isinstance(v, date) and self._sqlite:
            return v.isoformat()
        return v

    def _q(self, sql: str, **params) -> pd.DataFrame:
        with self._engine.connect() as con:
            return pd.read_sql(sa.text(sql), con, params={k: self._p(v) for k, v in params.items()})

    def _has(self, table: str) -> bool:
        if self._tables is None:
            self._tables = set(sa.inspect(self._engine).get_table_names())
        return table in self._tables

    @staticmethod
    def _dates(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
        for c in cols:
            if c in df:
                df[c] = pd.to_datetime(df[c]).dt.date
        return df

    def _sid(self, ticker: str) -> str:
        return ticker

    # ------------------------------------------------------------------ status
    def get_data_status(self) -> dict:
        d = self._q("SELECT as_of, source, model_version, universe FROM data_status")
        if d.empty:
            raise NoData("no data status")
        first = self._q("SELECT MIN(as_of) AS d FROM prediction")
        r = d.iloc[0]
        return {"as_of": pd.to_datetime(r["as_of"]).date(), "source": r["source"], "model_version": r["model_version"],
                "universe": r["universe"], "first_date": pd.to_datetime(first["d"].iloc[0]).date()}

    # ------------------------------------------------------------------ universe
    def list_stocks(self) -> pd.DataFrame:
        return self._q("SELECT ticker, name, sector FROM company ORDER BY sort_order")

    # ------------------------------------------------------------------ market overview
    def get_index_history(self, start: date, end: date) -> pd.DataFrame:
        d = self._q("SELECT date, aspi, sl20 FROM index_level WHERE date BETWEEN :s AND :e ORDER BY date", s=start, e=end)
        return self._dates(d, "date")

    def get_market_breadth(self, as_of: date) -> dict:
        d = self._q("SELECT advances, declines, unchanged FROM market_breadth WHERE date = "
                    "(SELECT MAX(date) FROM market_breadth WHERE date <= :d)", d=as_of)
        if d.empty:
            raise NoData("no breadth")
        return {k: int(d[k].iloc[0]) for k in ("advances", "declines", "unchanged")}

    def get_top_movers(self, as_of: date, n: int = 5) -> pd.DataFrame:
        d = self._q("""
            SELECT p.ticker, c.name, p.date, p.close FROM daily_price p JOIN company c ON c.ticker = p.ticker
            WHERE p.date IN (SELECT DISTINCT date FROM daily_price WHERE date <= :d ORDER BY date DESC LIMIT 2)""", d=as_of)
        if d.empty:
            raise NoData("no prices")
        d = self._dates(d, "date").sort_values("date")
        last = d["date"].max()
        piv = d.pivot_table(index=["ticker", "name"], columns="date", values="close")
        if piv.shape[1] < 2:
            raise NoData("not enough history")
        out = piv.reset_index()
        prev_col, last_col = sorted(piv.columns)[-2], last
        out["close"] = out[last_col]
        out["change_pct"] = (100 * (out[last_col] / out[prev_col] - 1)).round(2)
        return out.dropna(subset=["change_pct"])[["ticker", "name", "close", "change_pct"]] \
            .sort_values("change_pct", ascending=False).reset_index(drop=True)

    def get_market_sentiment(self, as_of: date) -> dict:
        if not self._has("sentiment_daily"):
            raise NoData("sentiment engine not run yet")
        d = self._q("SELECT market_score, market_articles FROM sentiment_daily WHERE ticker = '_MARKET' AND date = "
                    "(SELECT MAX(date) FROM sentiment_daily WHERE ticker = '_MARKET' AND date <= :d)", d=as_of)
        if d.empty:
            raise NoData("no market sentiment")
        return {"score": float(d["market_score"].iloc[0]), "article_count": int(d["market_articles"].iloc[0])}

    def get_market_risk_summary(self, as_of: date) -> dict:
        if not self._has("market_risk"):
            raise NoData("risk not run")
        d = self._q("SELECT var_1d_95, realised_vol_20d, regime FROM market_risk WHERE date = "
                    "(SELECT MAX(date) FROM market_risk WHERE date <= :d)", d=as_of)
        if d.empty:
            raise NoData("no market risk")
        r = d.iloc[0]
        return {"var_1d_95": None if pd.isna(r["var_1d_95"]) else float(r["var_1d_95"]),
                "realised_vol_20d": float(r["realised_vol_20d"]), "regime": r["regime"]}

    # ------------------------------------------------------------------ prices & predictions
    def get_price_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        d = self._q("SELECT date, open, high, low, close, volume FROM daily_price WHERE ticker = :t "
                    "AND date BETWEEN :s AND :e ORDER BY date", t=ticker, s=start, e=end)
        return self._dates(d, "date")

    def get_prediction(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        d = self._q("SELECT direction, p_up, threshold, model, model_version, target_date FROM prediction "
                    "WHERE ticker = :t AND as_of = :d AND horizon = :h", t=ticker, d=as_of, h=horizon)
        if d.empty:
            raise NoData("no prediction for this date")
        r = d.iloc[0]
        return {"direction": r["direction"], "p_up": float(r["p_up"]), "threshold": float(r["threshold"]), "model": r["model"],
                "model_version": r["model_version"], "target_date": pd.to_datetime(r["target_date"]).date()}

    def get_signal_reliability(self, horizon: Horizon) -> pd.DataFrame:
        if not self._has("signal_reliability"):
            return super().get_signal_reliability(horizon)
        return self._q("SELECT min_distance, coverage, accuracy FROM signal_reliability WHERE horizon = :h "
                       "ORDER BY min_distance", h=horizon)

    def get_block_accuracy(self, ticker: str, horizon: Horizon) -> pd.DataFrame:
        d = self._q("SELECT block, test_start, test_end, accuracy, f1, precision, recall, n FROM block_accuracy "
                    "WHERE ticker = :t AND horizon = :h ORDER BY block", t=ticker, h=horizon)
        if d.empty:
            raise NoData("no out-of-sample blocks")
        return self._dates(d, "test_start", "test_end")

    # ------------------------------------------------------------------ explainability
    def get_global_importance(self, horizon: Horizon) -> pd.DataFrame:
        d = self._q("SELECT feature, mean_abs_shap FROM global_importance WHERE horizon = :h "
                    "ORDER BY mean_abs_shap DESC", h=horizon)
        if d.empty:
            raise NoData("no global importance")
        return d

    def get_shap_explanation(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        d = self._q("SELECT feature, value, shap, base_value FROM shap_contribution WHERE ticker = :t AND as_of = :d "
                    "AND horizon = :h ORDER BY rank", t=ticker, d=as_of, h=horizon)
        if d.empty:
            raise NoData("no explanation for this date")
        return {"base_value": float(d["base_value"].iloc[0]), "contributions": d[["feature", "value", "shap"]]}

    def get_narrative(self, ticker: str, as_of: date, horizon: Horizon) -> str:
        d = self._q("SELECT narrative FROM prediction WHERE ticker = :t AND as_of = :d AND horizon = :h",
                    t=ticker, d=as_of, h=horizon)
        if d.empty or not d["narrative"].iloc[0]:
            raise NoData("no narrative")
        return str(d["narrative"].iloc[0])

    # ------------------------------------------------------------------ risk
    def get_var_estimate(self, ticker: str, as_of: date) -> pd.DataFrame:
        if not self._has("var_estimate"):
            raise NoData("risk not run")
        d = self._q("SELECT horizon_days, confidence, var_pct FROM var_estimate WHERE ticker = :t AND as_of = :d "
                    "ORDER BY horizon_days, confidence", t=ticker, d=as_of)
        if d.empty:
            raise NoData("no VaR for this date")
        return d

    def get_risk_parameters(self, ticker: str, as_of: date) -> dict:
        if not self._has("risk_parameters"):
            raise NoData("risk not run")
        d = self._q("SELECT * FROM risk_parameters WHERE ticker = :t AND as_of = "
                    "(SELECT MAX(as_of) FROM risk_parameters WHERE ticker = :t AND as_of <= :d)", t=ticker, d=as_of)
        if d.empty:
            raise NoData("no risk parameters")
        r = d.iloc[0]
        keys = ("mu", "omega", "alpha", "gamma", "beta", "nu", "sigma_last", "ret_last", "crisis_vol_multiplier", "crisis_shock")
        return {k: float(r[k]) for k in keys}

    def get_kupiec_results(self, ticker: str) -> pd.DataFrame:
        if not self._has("kupiec_result"):
            raise NoData("risk not run")
        d = self._q("SELECT confidence, observations, breaches, expected_breaches, lr_pof, p_value, verdict "
                    "FROM kupiec_result WHERE ticker = :t ORDER BY confidence", t=ticker)
        if d.empty:
            raise NoData("no backtest")
        return d

    # ------------------------------------------------------------------ news sentiment
    def get_sentiment_series(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        if not self._has("sentiment_daily"):
            return pd.DataFrame(columns=["date", "company_score", "market_score", "article_count"])
        # every trading day with market news, plus the company's score on days it was in the news
        d = self._q("""SELECT m.date, s.company_score, m.market_score, COALESCE(s.article_count, 0) AS article_count
                       FROM sentiment_daily m LEFT JOIN sentiment_daily s ON s.date = m.date AND s.ticker = :t
                       WHERE m.ticker = '_MARKET' AND m.date BETWEEN :s AND :e ORDER BY m.date""", t=ticker, s=start, e=end)
        return self._dates(d, "date")

    def get_articles(self, ticker: str | None, start: date, end: date) -> pd.DataFrame:
        cols = ["published", "headline", "source", "language", "url", "ticker", "score"]
        if not self._has("article"):
            return pd.DataFrame(columns=cols)
        if ticker:
            d = self._q("SELECT published, trading_day, headline, source, language, url, ticker, score FROM article "
                        "WHERE ticker = :t AND trading_day BETWEEN :s AND :e ORDER BY trading_day DESC", t=ticker, s=start, e=end)
        else:
            d = self._q("SELECT published, trading_day, headline, source, language, url, ticker, score FROM article "
                        "WHERE trading_day BETWEEN :s AND :e ORDER BY trading_day DESC LIMIT 500", s=start, e=end)
        d["published"] = pd.to_datetime(d["published"], errors="coerce").fillna(pd.to_datetime(d["trading_day"]))
        d["ticker"] = d["ticker"].fillna("Market")
        return self._dates(d, "published").drop(columns="trading_day")
