"""Shared fixtures. The CSE Data Library is licensed and lives outside the repo; tests that need it
are skipped when it is not present."""

import os
from pathlib import Path

import pytest

DEFAULT_ROOT = Path(r"C:\Users\theli\Downloads\CSE Docs\All Data Files\2025 Q4-20260918T083633Z-1-001\2025 Q4")
DATA_ROOT = Path(os.environ.get("CSE_DATA_ROOT", DEFAULT_ROOT))

PRICES_2021_2025 = DATA_ROOT / "33Daily Shares Price List -2021-2025" / "40Daily Shares Price List -2021-2025"
PRICES_2011_2020 = DATA_ROOT / "32Daily Shares Price List -2011-2020"
PUBLIC_HOLDING = DATA_ROOT / "17 Public Holding-Quarterly" / "Public Holding"

requires_data = pytest.mark.skipif(not DATA_ROOT.is_dir(), reason=f"CSE Data Library not found at {DATA_ROOT}")


@pytest.fixture(scope="session")
def data_root() -> Path:
    if not DATA_ROOT.is_dir():
        pytest.skip(f"CSE Data Library not found at {DATA_ROOT}")
    return DATA_ROOT
