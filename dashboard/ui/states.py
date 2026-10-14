"""
Loading, empty and error states written for a non-technical reader.

* `warm_up()`        — first data touch of a session, with an explicit "waking
                       up" message and retry progress (cold-start handling).
* `empty_state()`    — a valid selection with nothing stored for it.
* `unavailable_state()` — database unreachable after every retry.
* `error_state()`    — anything unexpected; never a stack trace.
* `try_fetch()`      — call a repository method, render an empty state on NoData.
* `run_guarded()`    — wraps the page dispatch so no exception reaches the browser.
"""
from __future__ import annotations

import html
import logging
import time
from typing import Any, Callable

import streamlit as st

from data.errors import DataUnavailable, NoData
from data.repository import get_repository
from services.perf import end_page, record
from ui.layout import chrome_drawn, chrome_reset, demo_banner, disclaimer

log = logging.getLogger("cse.ui")

WAKE_TITLE = "Waking things up…"
WAKE_BODY = ("Our database goes to sleep when nobody is using the app. Waking it usually takes "
             "5 to 15 seconds, and you only wait once per visit.")
RETRY_BODY = ("Still waking up. We will try again in {delay:.0f} seconds (attempt {next_attempt} of {total}). "
              "There is nothing you need to do.")


def _esc(s: Any) -> str:
    return html.escape(str(s))


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _panel(title: str, body: str, icon: str, tone: str = "") -> None:
    st.markdown(
        f"<div class='state{(' state-' + tone) if tone else ''}' role='status'>"
        f"<div class='state-icon' aria-hidden='true'>{icon}</div>"
        f"<div><div class='state-title'>{_esc(title)}</div><div class='state-body'>{_esc(body)}</div></div></div>",
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, action_label: str | None = None, key: str | None = None) -> bool:
    """Nothing stored for a valid selection. Returns True if the optional action button was pressed."""
    _panel(title, body, "◇")
    if action_label:
        return st.button(action_label, key=key or f"act-{abs(hash((title, action_label)))}")
    return False


def _full_page_panel(title: str, body: str, icon: str, button_key: str) -> bool:
    """A panel that may be the only thing on the page: keeps the demo banner above it and the
    disclaimer below it (both are non-negotiable on every page), without duplicating either
    if the page already drew them before failing. Returns True when Try again was pressed."""
    if not chrome_drawn("banner"):
        demo_banner()
    _panel(title, body, icon, tone="warn")
    pressed = st.button("Try again", type="primary", key=button_key)
    if not chrome_drawn("footer"):
        disclaimer()
    return pressed


def unavailable_state() -> None:
    """Database unreachable after every retry."""
    if _full_page_panel(
        "We can't reach the data right now",
        "The connection to our database did not come back after several tries. This is usually temporary: "
        "wait a moment and press Try again. If it keeps happening, tell the researcher — nothing you did caused it.",
        "⏻", "retry-unavailable",
    ):
        st.session_state.pop("_warm", None)
        st.rerun()


def error_state() -> None:
    """Unexpected exception. Details go to the server log, never to the screen."""
    if _full_page_panel(
        "Something went wrong on our side",
        "This page hit a problem we did not expect. Your selections are safe. Press Try again, or choose a "
        "different share or date. If it keeps happening, let the researcher know which page you were on.",
        "!", "retry-error",
    ):
        st.rerun()


def waking_html(title: str, body: str) -> str:
    return (f"<div class='waking' role='status' aria-live='polite'><div class='spinner' aria-hidden='true'></div>"
            f"<div><div class='state-title'>{_esc(title)}</div><div class='state-body'>{_esc(body)}</div></div></div>")


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #
def warm_up() -> None:
    """First data touch of the session, with a visible waking message.

    Runs once per session (flag `_warm`). Shows the message *before* the first
    call so a slow Neon wake-up is explained rather than silently endured, and
    updates it on every retry via the listener the resilient repository calls.
    """
    ss = st.session_state
    chrome_reset()
    if ss.get("_warm"):
        return
    box = st.empty()
    box.markdown(waking_html(WAKE_TITLE, WAKE_BODY), unsafe_allow_html=True)

    def _listener(attempt: int, total: int, delay: float, exc: BaseException) -> None:
        box.markdown(waking_html(WAKE_TITLE, RETRY_BODY.format(delay=delay, next_attempt=attempt + 1, total=total)),
                     unsafe_allow_html=True)

    ss["_retry_listener"] = _listener
    t0 = time.perf_counter()
    try:
        get_repository().get_data_status()
    except DataUnavailable:
        box.empty()
        unavailable_state()
        st.stop()
    except Exception:
        log.exception("warm-up failed for a non-connection reason (configuration?)")
        box.empty()
        error_state()
        st.stop()
    finally:
        ss.pop("_retry_listener", None)
    record("wake", "first data touch", time.perf_counter() - t0)
    box.empty()
    ss["_warm"] = True


# --------------------------------------------------------------------------- #
# Page helpers
# --------------------------------------------------------------------------- #
def try_fetch(fn: Callable[..., Any], *args: Any, missing: tuple[str, str] | None = None) -> Any:
    """Call a repository method. On `NoData`, render `missing` (title, body) if given and return None.
    Connection failures propagate to `run_guarded`, which shows the unreachable panel."""
    try:
        return fn(*args)
    except NoData as exc:
        log.info("no data: %s%s -> %s", getattr(fn, "__name__", fn), args, exc)
        if missing:
            empty_state(*missing)
        return None


def run_guarded(body: Callable[[], None]) -> None:
    """Run the sidebar + page dispatch; translate every failure into a friendly panel."""
    try:
        body()
    except DataUnavailable:
        end_page()
        unavailable_state()
    except NoData as exc:
        end_page()
        if not chrome_drawn("banner"):
            demo_banner()
        empty_state("Nothing to show for this selection",
                    "There is no stored result for the share and date you picked. Try another date or share "
                    "from the sidebar." + (f" ({exc})" if str(exc) else ""))
        if not chrome_drawn("footer"):
            disclaimer()
    except Exception:
        end_page()
        log.exception("unhandled error while rendering a page")
        error_state()
