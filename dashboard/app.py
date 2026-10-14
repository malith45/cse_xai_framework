"""
Entrypoint. Run with:  streamlit run app.py

Order matters:
  1. page config + CSS
  2. navigation is *registered* (hidden until authenticated) so a deep link such as
     /risk_stress_test survives the passcode rerun instead of falling back to the default page
  3. passcode gate  (stops the script for unauthenticated users — nothing below runs)
  4. warm-up        (first data touch, with a visible "waking up" message and retries)
  5. shared sidebar controls + page dispatch, inside a guard that turns any failure into a
     plain-English panel — a participant never sees a stack trace.
"""
import streamlit as st

from config import APP_ICON, APP_TITLE
from services.perf import start_page
from ui.auth import require_passcode
from ui.layout import inject_css, nav_active_css, sidebar_controls
from ui.states import run_guarded, warm_up

# "auto" = expanded on desktop, collapsed on phones (the sidebar would otherwise
# open as a full-screen overlay and hide the page on first load).
st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide",
                   initial_sidebar_state="auto")
inject_css()

pages = [
    st.Page("views/market_overview.py", title="Market overview", icon=":material/monitoring:", default=True),
    st.Page("views/stock_prediction.py", title="Stock prediction", icon=":material/query_stats:"),
    st.Page("views/explainability.py", title="Explainability", icon=":material/psychology:"),
    st.Page("views/risk_stress_test.py", title="Risk & stress test", icon=":material/shield:"),
    st.Page("views/news_sentiment.py", title="News sentiment", icon=":material/newspaper:"),
]
authenticated = bool(st.session_state.get("authenticated"))
nav = st.navigation(pages, position="sidebar" if authenticated else "hidden")

require_passcode()
warm_up()


def _dispatch() -> None:
    sidebar_controls()
    nav_active_css(nav.url_path)  # sidebar highlight for the current page (CSS only)
    start_page(nav.title)         # render timer; closed by footer() or the guard
    nav.run()


run_guarded(_dispatch)
