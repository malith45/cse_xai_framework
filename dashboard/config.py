"""
Central configuration.

Everything that varies between a laptop, Streamlit Community Cloud and a
future Postgres deployment is resolved here and nowhere else.

Resolution order for each setting:
    1. st.secrets  (Streamlit Community Cloud / .streamlit/secrets.toml)
    2. environment variable
    3. hard-coded development default
"""
from __future__ import annotations

import os

import streamlit as st

APP_TITLE = "CSE Decision Support"
APP_ICON = "◆"

# Placeholder — replace with the real evaluation questionnaire link.
QUESTIONNAIRE_URL = "https://forms.gle/REPLACE-WITH-QUESTIONNAIRE"

DISCLAIMER = (
    "Research prototype for decision support only. Not financial advice. "
    "Predictions are probabilistic, may be wrong, and carry no guarantee of returns."
)

# Trading session close (Sri Lanka time) — after-close news rolls to next day.
CSE_CLOSE = "14:30"

# --------------------------------------------------------------------------- #
# Resilience & caching (see README → "Cold start, retries and caching")
# --------------------------------------------------------------------------- #
# Every repository read is memoised for this long (seconds). The data is
# precomputed offline and refreshed at most daily, so 10 minutes is safe.
CACHE_TTL_SECONDS = 600
# Backoff between attempts when the database is unreachable. Five attempts in
# total over ~15 s, enough for a Neon Postgres compute to wake from scale-to-zero.
RETRY_DELAYS_SECONDS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

# Mock-only failure scenarios, selected with ?scenario=<name> in the URL.
# They exist so the manual test checklist can exercise every empty/error
# state without a database. Ignored entirely when DATA_SOURCE != "mock".
MOCK_SCENARIOS = ("no_news", "no_prediction", "no_risk", "db_down", "cold_start")


def _setting(key: str, default: str) -> str:
    """Read a setting from st.secrets, then the environment, then a default."""
    try:
        if st.secrets.load_if_toml_exists() and key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # secrets unavailable (bare mode / malformed file)
        pass
    return os.getenv(key, default)


def data_source() -> str:
    """'mock' (default) or 'postgres'. Drives BOTH the repository factory
    and the demonstration banner, so the two can never disagree."""
    return _setting("DATA_SOURCE", "mock").strip().lower()


def is_mock_mode() -> bool:
    return data_source() == "mock"


def mock_scenario() -> str:
    """Active mock failure scenario for this session ('' when none / not mock mode)."""
    if not is_mock_mode():
        return ""
    try:
        value = str(st.query_params.get("scenario", "")).strip().lower()
    except Exception:  # no script context
        return ""
    return value if value in MOCK_SCENARIOS else ""


def passcode() -> str:
    """Shared passcode. Development fallback only — set PASSCODE in secrets."""
    return _setting("PASSCODE", "cse-demo")


def database_url() -> str:
    """Read-only Postgres connection string (used only when DATA_SOURCE=postgres)."""
    return _setting("DATABASE_URL", "")
