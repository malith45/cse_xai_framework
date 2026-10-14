"""Shared-passcode gate. Called once in app.py before navigation runs."""
from __future__ import annotations

import hmac

import streamlit as st

from config import APP_TITLE, DISCLAIMER, passcode


def require_passcode() -> None:
    """Render the gate and stop the script until the correct passcode is entered."""
    if st.session_state.get("authenticated"):
        return

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown(
            f"""
            <div class="gate">
              <div class="gate-mark">◆</div>
              <div class="gate-kicker">Restricted access</div>
              <h1>{APP_TITLE}</h1>
              <p>Explainable machine-learning decision support for the Colombo Stock Exchange.
              Enter the access passcode to continue.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("gate", border=False):
            entered = st.text_input("Passcode", type="password", label_visibility="collapsed",
                                    placeholder="Passcode")
            ok = st.form_submit_button("Enter workspace", use_container_width=True, type="primary")
        if ok:
            if hmac.compare_digest(entered.strip(), passcode()):
                st.session_state["authenticated"] = True
                st.rerun()
            st.error("That passcode is not recognised. Check with the researcher and try again.")
        st.caption(DISCLAIMER)
    st.stop()
