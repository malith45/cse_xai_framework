"""
Repository contract.

Every data method a page can call is declared here. Pages depend ONLY on this
abstract interface and obtain an instance through `get_repository()`.
Swapping mock data for Postgres means changing one config flag.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Literal

import pandas as pd
import streamlit as st

Horizon = Literal["t+1", "t+5"]


class Repository(ABC):
    """Read-only data access used by the dashboard."""

    # ----- status ---------------------------------------------------------
    @abstractmethod
    def get_data_status(self) -> dict:
        """{'as_of': date, 'source': str, 'model_version': str}"""

    # ----- universe -------------------------------------------------------
    @abstractmethod
    def list_stocks(self) -> pd.DataFrame:
        """Columns: ticker, name, sector"""

    # ----- market overview ------------------------------------------------
    @abstractmethod
    def get_index_history(self, start: date, end: date) -> pd.DataFrame:
        """Columns: date, aspi, sl20"""

    @abstractmethod
    def get_market_breadth(self, as_of: date) -> dict:
        """{'advances': int, 'declines': int, 'unchanged': int}"""

    @abstractmethod
    def get_top_movers(self, as_of: date, n: int = 5) -> pd.DataFrame:
        """Columns: ticker, name, close, change_pct  (sorted desc)"""

    @abstractmethod
    def get_market_sentiment(self, as_of: date) -> dict:
        """{'score': float in [-1,1], 'article_count': int}"""

    @abstractmethod
    def get_market_risk_summary(self, as_of: date) -> dict:
        """{'var_1d_95': float, 'realised_vol_20d': float, 'regime': str}"""

    # ----- prices & predictions ------------------------------------------
    @abstractmethod
    def get_price_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Columns: date, open, high, low, close, volume"""

    @abstractmethod
    def get_prediction(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        """{'direction': 'up'|'down', 'p_up': float, 'model': str,
            'model_version': str, 'target_date': date}"""

    @abstractmethod
    def get_block_accuracy(self, ticker: str, horizon: Horizon) -> pd.DataFrame:
        """Columns: block, test_start, test_end, accuracy, f1, precision, recall, n"""

    # ----- explainability -------------------------------------------------
    @abstractmethod
    def get_global_importance(self, horizon: Horizon) -> pd.DataFrame:
        """Columns: feature, mean_abs_shap  (sorted desc)"""

    @abstractmethod
    def get_shap_explanation(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        """{'base_value': float, 'contributions': DataFrame(feature, value, shap)}"""

    @abstractmethod
    def get_narrative(self, ticker: str, as_of: date, horizon: Horizon) -> str:
        """Plain-language explanation text."""

    # ----- risk -----------------------------------------------------------
    @abstractmethod
    def get_var_estimate(self, ticker: str, as_of: date) -> pd.DataFrame:
        """Columns: horizon_days (1|5), confidence (0.95|0.99), var_pct"""

    @abstractmethod
    def get_risk_parameters(self, ticker: str, as_of: date) -> dict:
        """Stored GJR-GARCH(1,1)-t parameters used by the lightweight stress
        engine: {'mu','omega','alpha','gamma','beta','nu','sigma_last','ret_last',
                 'crisis_vol_multiplier','crisis_shock'}"""

    @abstractmethod
    def get_kupiec_results(self, ticker: str) -> pd.DataFrame:
        """Columns: confidence, observations, breaches, expected_breaches,
                    lr_pof, p_value, verdict"""

    # ----- news sentiment -------------------------------------------------
    @abstractmethod
    def get_sentiment_series(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        """Columns: date, company_score, market_score, article_count"""

    @abstractmethod
    def get_articles(self, ticker: str | None, start: date, end: date) -> pd.DataFrame:
        """Columns: published, headline, source, language, url, ticker, score.
        Never contains article body text."""


@st.cache_resource(show_spinner=False)
def get_repository() -> Repository:
    """Factory. The ONLY place that knows which implementation exists.

    The concrete repository is wrapped in `ResilientRepository`, which adds
    retry-with-backoff, a short-TTL cache and typed errors (see data/resilient.py).
    Implementations should raise `data.errors.NoData` when a query is valid but
    has no rows; any other exception is treated as transient and retried.
    """
    from config import data_source  # local import keeps config free of cycles
    from data.resilient import ResilientRepository

    source = data_source()
    if source == "mock":
        from data.mock_repository import MockRepository

        return ResilientRepository(MockRepository())
    if source == "postgres":
        from data.postgres_repository import PostgresRepository

        return ResilientRepository(PostgresRepository())
    raise ValueError(f"Unknown DATA_SOURCE '{source}'. Use 'mock' or 'postgres'.")
