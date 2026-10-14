"""Signal strength of a directional call, in plain words.

The model calls "up" when P(up) reaches its own cut-off (chosen on inner validation, so it is not
always 50%). How far P(up) sits from that cut-off separates firmer calls from coin tosses. Whether
that holds is checked, per horizon, against the out-of-sample "Accuracy by confidence" table that the
pipeline publishes (signal_reliability): a "strong signal" label is shown only where calls far from
the cut-off really were more accurate than calls overall.
"""

from __future__ import annotations

import pandas as pd

WEAK_BELOW = 0.05     # |P(up) - cut-off| < 5 points  -> no clear signal
STRONG_FROM = 0.10    # |P(up) - cut-off| >= 10 points -> strong signal (if the evidence supports it)
MIN_GAIN = 0.02       # strong calls must beat all calls by at least 2 points out of sample


def strength(p_up: float, threshold: float | None) -> str:
    """'weak' | 'moderate' | 'strong' by distance from the cut-off."""
    d = abs(p_up - (0.5 if threshold is None else threshold))
    if d < WEAK_BELOW:
        return "weak"
    return "strong" if d >= STRONG_FROM else "moderate"


def strength_supported(rel: pd.DataFrame | None) -> bool:
    """True when out-of-sample accuracy of calls >= STRONG_FROM from the cut-off beats all calls."""
    if rel is None or rel.empty:
        return False
    r = rel.set_index(rel["min_distance"].round(2))["accuracy"]
    return 0.0 in r.index and STRONG_FROM in r.index and r[STRONG_FROM] - r[0.0] >= MIN_GAIN


def headline(direction: str, p_up: float, threshold: float | None, supported: bool = True) -> tuple[str, str]:
    """(big text, small caption) for a prediction card."""
    word = "Up" if direction == "up" else "Down"
    s = strength(p_up, threshold)
    if s == "weak":
        return "Unclear", f"leans {word.lower()} · P(up) is close to the model's cut-off"
    arrow = "▲ " if direction == "up" else "▼ "
    if not supported:
        return arrow + word, "weak evidence at this horizon"
    return arrow + word, ("strong signal" if s == "strong" else "moderate signal")


def cutoff_note(threshold: float | None) -> str:
    return f"model calls 'up' from {(0.5 if threshold is None else threshold):.0%}"


def reliability_sentence(rel: pd.DataFrame | None, horizon: str) -> str:
    """One plain sentence from the published table, or '' when it is not available."""
    if rel is None or rel.empty:
        return ""
    r = rel.set_index(rel["min_distance"].round(2))
    if 0.0 not in r.index or STRONG_FROM not in r.index:
        return ""
    allacc, far, cov = r.loc[0.0, "accuracy"], r.loc[STRONG_FROM, "accuracy"], r.loc[STRONG_FROM, "coverage"]
    if strength_supported(rel):
        return (f"In the out-of-sample tests, {horizon} calls at least 10 points from the cut-off "
                f"({cov:.0%} of calls) were right {far:.0%} of the time, against {allacc:.0%} for all calls.")
    return (f"In the out-of-sample tests, {horizon} calls were right {allacc:.0%} of the time, and calls far from "
            f"the cut-off were no more reliable ({far:.0%}), so this horizon shows no 'strong signal' label.")
