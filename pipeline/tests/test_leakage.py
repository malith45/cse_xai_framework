"""NFR07 temporal-integrity tests: no feature may use information after the prediction date.

For several cut-off dates T, the panel is rebuilt from prices and index levels truncated at T.
Every feature value on day T must equal the value computed from the full sample. A feature that
looked ahead (centred windows, bfill, future index levels) would differ.

A second test checks scale invariance: multiplying all of a security's earlier prices by a constant
(what a backward corporate-action adjustment for a *later* event does) must not change any feature,
so adjusting with events after T cannot leak information about them into features at T.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.features import FEATURES, build_panel

WORK = Path(__file__).resolve().parents[1] / "work"
pytestmark = pytest.mark.skipif(not (WORK / "prices_adj.parquet").exists(), reason="run the M1 pipeline first")

SECURITIES = ["JKH-N-0000", "COMB-N-0000", "HNB-N-0000", "SAMP-N-0000", "DIAL-N-0000", "CTC-N-0000"]


@pytest.fixture(scope="module")
def data():
    p = pd.read_parquet(WORK / "prices_adj.parquet")
    p = p[p["security_id"].isin(SECURITIES)]
    ix = pd.read_parquet(WORK / "indices.parquet")
    full, _ = build_panel(p, ix)
    return p, ix, full


@pytest.mark.parametrize("cutoff", ["2021-06-30", "2022-04-27", "2023-08-17", "2024-11-06", "2025-12-31"])
def test_features_identical_when_future_removed(data, cutoff):
    p, ix, full = data
    t = pd.Timestamp(cutoff)
    t = ix.loc[ix["date"] <= t, "date"].max()
    trunc, _ = build_panel(p[p["date"] <= t], ix[ix["date"] <= t])
    a = full[full["date"] == t].set_index("security_id")[FEATURES].sort_index()
    b = trunc[trunc["date"] == t].set_index("security_id")[FEATURES].sort_index()
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


def test_features_invariant_to_backward_adjustment_scale(data):
    p, ix, full = data
    q = p.copy()
    for c in ("adj_open", "adj_high", "adj_low", "adj_close"):
        q[c] = q[c] * 0.37
    scaled, _ = build_panel(q, ix)
    a = full.set_index(["security_id", "date"])[FEATURES]
    b = scaled.set_index(["security_id", "date"])[FEATURES]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-7, atol=1e-9)


def test_labels_are_strictly_future(data):
    _, _, full = data
    j = full[full["security_id"] == "JKH-N-0000"].sort_values("date")
    # y1 on day t must equal the sign of the NEXT row's same-day log return when traded consecutively
    consecutive = j["cal_pos"].shift(-1) == j["cal_pos"] + 1
    nxt = (j["lr"].shift(-1) > 0).astype(float)
    assert (j.loc[consecutive, "y1"] == nxt[consecutive]).all()
    # no feature is perfectly correlated with the label (a crude leak detector)
    for f in FEATURES:
        ok = j[[f, "fwd_ret1"]].dropna()
        assert abs(ok[f].corr(ok["fwd_ret1"])) < 0.5, f


# ---------------------------------------------------------------- macro (Chapter 4 §4.7 step 9)
from pipeline.macro import MACRO, load_macro, macro_features  # noqa: E402


@pytest.fixture(scope="module")
def macro_raw():
    return load_macro()


@pytest.mark.parametrize("cutoff", ["2020-03-17", "2022-04-08", "2023-01-31", "2024-11-27", "2025-12-31"])
def test_macro_uses_only_values_published_before_the_day(data, macro_raw, cutoff):
    """Dropping every macro observation dated on or after day T must not change features on day T."""
    _, ix, _ = data
    cal = pd.DatetimeIndex(sorted(ix["date"].unique()))
    t = cal[cal <= pd.Timestamp(cutoff)].max()
    full = macro_features(cal, macro_raw).loc[t]
    trunc_raw = {k: v[v["date"] < t].copy() for k, v in macro_raw.items()}
    trunc = macro_features(cal, trunc_raw).loc[t]
    pd.testing.assert_series_equal(full, trunc, check_exact=False, rtol=1e-12)


def test_policy_change_visible_only_from_next_trading_day(data, macro_raw):
    _, ix, _ = data
    cal = pd.DatetimeIndex(sorted(ix["date"].unique()))
    m = macro_features(cal, macro_raw)
    assert m.loc["2024-11-27", "policy_rate"] == pytest.approx(8.75)   # OPR announced 27 Nov 2024
    assert m.loc["2024-11-28", "policy_rate"] == pytest.approx(8.00)
    # CCPI for a month is not visible on its own last trading day
    assert m.loc["2022-09-30", "ccpi_yoy"] == pytest.approx(64.3)      # August figure
    assert m.loc["2022-10-03", "ccpi_yoy"] == pytest.approx(69.8)      # September figure, next month
    assert m[MACRO].notna().all().all()
