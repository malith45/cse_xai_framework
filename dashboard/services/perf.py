"""
Performance instrumentation.

Two things are measured so the Chapter 4 targets can be reported as evidence:

  * page render time  — server-side execution of a page script, from the
    moment navigation dispatches it until `footer()` runs. Excludes the one-off
    wake-up wait and the browser's own paint time, so it is a lower bound on
    what a participant experiences. Target: under 5 s once the app is awake.
  * stress-test time  — wall-clock time of the Monte Carlo simulation
    (10,000 paths × two horizons). Target: under 10 s.

Every measurement is written to the `cse.perf` logger (visible in the Streamlit
Community Cloud "Manage app" log panel, or the terminal locally) and kept in a
small per-session ring buffer in `st.session_state["_perf_log"]`.

Log line format:
    2026-09-18 10:21:07 cse.perf kind=page name="Stock prediction" seconds=0.412 target=5.0 status=ok
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

import streamlit as st

TARGETS_SECONDS = {"page": 5.0, "stress": 10.0, "wake": None}
_HISTORY_MAX = 100

log = logging.getLogger("cse.perf")
if not log.handlers:  # configure once per process; Streamlit does not configure app loggers
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s", "%Y-%m-%d %H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)
    log.propagate = False


def record(kind: str, name: str, seconds: float) -> dict:
    """Log one measurement and append it to the session ring buffer."""
    target = TARGETS_SECONDS.get(kind)
    status = "ok" if target is None or seconds <= target else "SLOW"
    entry = {"kind": kind, "name": name, "seconds": round(seconds, 3), "target": target, "status": status,
             "at": time.strftime("%H:%M:%S")}
    log.info('kind=%s name="%s" seconds=%.3f target=%s status=%s', kind, name, seconds, target, status)
    try:
        hist = st.session_state.setdefault("_perf_log", [])
        hist.append(entry)
        del hist[:-_HISTORY_MAX]
    except Exception:  # no script context
        pass
    return entry


def start_page(name: str) -> None:
    st.session_state["_perf_page"] = (name, time.perf_counter())


def end_page() -> float | None:
    """Close the open page timer (idempotent). Returns elapsed seconds or None."""
    rec = st.session_state.pop("_perf_page", None)
    if not rec:
        return None
    name, t0 = rec
    seconds = time.perf_counter() - t0
    record("page", name, seconds)
    return seconds


@contextmanager
def timed(kind: str, name: str) -> Iterator[dict]:
    """`with timed("stress", "JKH") as t: ...` → t["seconds"] afterwards."""
    box: dict = {}
    t0 = time.perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = time.perf_counter() - t0
        box.update(record(kind, name, box["seconds"]))


def history() -> list[dict]:
    return list(st.session_state.get("_perf_log", []))


def fmt_seconds(seconds: float) -> str:
    """Human-readable duration: '4 ms' below one second, '1.23 s' above."""
    return f"{seconds * 1000:.0f} ms" if seconds < 1 else f"{seconds:.2f} s"
