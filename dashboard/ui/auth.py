"""Shared-passcode gate. Called once in app.py before navigation runs."""
from __future__ import annotations

import hmac

import streamlit as st

from config import APP_TITLE, DISCLAIMER, passcode
from ui.layout import logo_svg


def require_passcode() -> None:
    """Render the gate and stop the script until the correct passcode is entered."""
    if st.session_state.get("authenticated"):
        return

    with st.container(key="login-shell"):
        st.markdown(
            f"""
            <div class="gate">
              <div class="gate-mark">{logo_svg(st.session_state.get("theme", "dark"))}</div>
              <div class="gate-kicker">Private workspace</div>
              <h1>{APP_TITLE}</h1>
              <p>Sign in to continue to your decision-support workspace.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("gate", border=False):
            entered = st.text_input("Access code", type="password", label_visibility="visible")
            ok = st.form_submit_button("Enter workspace", use_container_width=True, type="primary")
        if ok:
            if hmac.compare_digest(entered.strip(), passcode()):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("That passcode is not recognised. Check with the researcher and try again.")
        st.caption(DISCLAIMER)
    st.stop()
