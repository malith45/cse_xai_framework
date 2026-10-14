"""
ResilientRepository — retry, backoff and caching around any Repository.

`get_repository()` returns one of these wrapped around the concrete
implementation (mock or Postgres). Pages keep calling the same methods; the
wrapper adds:

  * retry with exponential backoff on connection-type failures (Neon Postgres
    scales to zero after 5 minutes idle; the first query after that can fail
    or take several seconds while the compute wakes);
  * a per-process `st.cache_data` layer with a short TTL, so each precomputed
    read hits the database at most once every CACHE_TTL_SECONDS regardless of
    how many participants are on the app;
  * typed errors: `DataUnavailable` after the last retry, `NoData` passed
    through untouched so pages can render a plain-English empty state.

Cache key = (method name, mock scenario, positional arguments). Exceptions are
never cached. The wrapped repository instance is excluded from the key.
"""
from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Callable

import pandas as pd
import streamlit as st

from config import CACHE_TTL_SECONDS, RETRY_DELAYS_SECONDS, mock_scenario
from data.errors import DataUnavailable, NoData
from data.repository import Horizon, Repository

log = logging.getLogger("cse.data")

# Errors that mean "the request is wrong", not "the service is asleep". These are
# raised straight through; retrying them would only delay the same answer.
NON_RETRYABLE: tuple[type[BaseException], ...] = (
    NoData, NotImplementedError, LookupError, ValueError, TypeError, AttributeError,
)

# Signature: listener(attempt, total_attempts, delay_seconds, exception)
RetryListener = Callable[[int, int, float, BaseException], None]


def _notify(attempt: int, total: int, delay: float, exc: BaseException) -> None:
    """Tell the current session's UI (if it registered a listener) that we are retrying."""
    try:
        listener: RetryListener | None = st.session_state.get("_retry_listener")
    except Exception:  # no script context (unit tests)
        listener = None
    if listener is not None:
        try:
            listener(attempt, total, delay, exc)
        except Exception:  # a UI hiccup must never break the data path
            log.debug("retry listener failed", exc_info=True)


def call_with_retry(fn: Callable[..., Any], *args: Any,
                    delays: tuple[float, ...] = RETRY_DELAYS_SECONDS, name: str | None = None) -> Any:
    """Call `fn(*args)`, retrying transient failures with the given backoff schedule.

    With the default schedule (1, 2, 4, 8 s) that is five attempts over ~15 s,
    which comfortably covers a Neon cold start. Raises `DataUnavailable` after
    the final attempt; non-retryable errors propagate immediately.
    """
    total = len(delays) + 1
    name = name or getattr(fn, "__name__", repr(fn))
    for attempt in range(1, total + 1):
        try:
            return fn(*args)
        except NON_RETRYABLE:
            raise
        except Exception as exc:
            log.warning("data call %s failed (attempt %d/%d): %s", name, attempt, total, exc)
            if attempt == total:
                raise DataUnavailable(f"{name} failed after {total} attempts: {exc}") from exc
            delay = delays[attempt - 1]
            _notify(attempt, total, delay, exc)
            time.sleep(delay)
    raise DataUnavailable(name)  # unreachable, keeps type checkers happy


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def _cached(_inner: Repository, method: str, scenario: str, *args: Any) -> Any:
    """Shared cache in front of every repository read. `_inner` is excluded from
    the key (leading underscore); `scenario` is part of it so mock failure
    scenarios never collide with normal cached results.

    Deliberately does only the raw call: exceptions are not cached, so the retry
    loop wraps *this* function from outside. That keeps the retry listener's UI
    updates out of the cached body (Streamlit replays st.* calls made inside a
    cached function on cache hits, and would fail against a placeholder that no
    longer exists)."""
    return getattr(_inner, method)(*args)


class ResilientRepository(Repository):
    """Repository decorator: every read goes through `call_with_retry` → `_cached`."""

    def __init__(self, inner: Repository) -> None:
        self._inner = inner

    @property
    def inner(self) -> Repository:
        return self._inner

    def _get(self, method: str, *args: Any) -> Any:
        return call_with_retry(_cached, self._inner, method, mock_scenario(), *args, name=method)

    # ----- status ---------------------------------------------------------
    def get_data_status(self) -> dict:
        return self._get("get_data_status")

    # ----- universe -------------------------------------------------------
    def list_stocks(self) -> pd.DataFrame:
        return self._get("list_stocks")

    # ----- market overview ------------------------------------------------
    def get_index_history(self, start: date, end: date) -> pd.DataFrame:
        return self._get("get_index_history", start, end)

    def get_market_breadth(self, as_of: date) -> dict:
        return self._get("get_market_breadth", as_of)

    def get_top_movers(self, as_of: date, n: int = 5) -> pd.DataFrame:
        return self._get("get_top_movers", as_of, n)

    def get_market_sentiment(self, as_of: date) -> dict:
        return self._get("get_market_sentiment", as_of)

    def get_market_risk_summary(self, as_of: date) -> dict:
        return self._get("get_market_risk_summary", as_of)

    # ----- prices & predictions ------------------------------------------
    def get_price_history(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return self._get("get_price_history", ticker, start, end)

    def get_prediction(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        return self._get("get_prediction", ticker, as_of, horizon)

    def get_block_accuracy(self, ticker: str, horizon: Horizon) -> pd.DataFrame:
        return self._get("get_block_accuracy", ticker, horizon)

    def get_signal_reliability(self, horizon: Horizon) -> pd.DataFrame:
        return self._get("get_signal_reliability", horizon)

    # ----- explainability -------------------------------------------------
    def get_global_importance(self, horizon: Horizon) -> pd.DataFrame:
        return self._get("get_global_importance", horizon)

    def get_shap_explanation(self, ticker: str, as_of: date, horizon: Horizon) -> dict:
        return self._get("get_shap_explanation", ticker, as_of, horizon)

    def get_narrative(self, ticker: str, as_of: date, horizon: Horizon) -> str:
        return self._get("get_narrative", ticker, as_of, horizon)

    # ----- risk -----------------------------------------------------------
    def get_var_estimate(self, ticker: str, as_of: date) -> pd.DataFrame:
        return self._get("get_var_estimate", ticker, as_of)

    def get_risk_parameters(self, ticker: str, as_of: date) -> dict:
        return self._get("get_risk_parameters", ticker, as_of)

    def get_kupiec_results(self, ticker: str) -> pd.DataFrame:
        return self._get("get_kupiec_results", ticker)

    # ----- news sentiment -------------------------------------------------
    def get_sentiment_series(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        return self._get("get_sentiment_series", ticker, start, end)

    def get_articles(self, ticker: str | None, start: date, end: date) -> pd.DataFrame:
        return self._get("get_articles", ticker, start, end)
