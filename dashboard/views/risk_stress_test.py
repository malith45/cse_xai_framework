import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data.repository import get_repository
from services.perf import TARGETS_SECONDS, fmt_seconds, timed
from services.stress import simulate_returns, value_at_risk
from ui.charts import CONFIG
from ui.i18n import tr
from ui.layout import (DOWN, MUTED, UP, card, chart_caption, fmt_lkr, footer, page_header, responsive_table,
                       section, selection)
from ui.states import empty_state, try_fetch

repo = get_repository()
ticker, as_of, horizon = selection()
name = repo.list_stocks().set_index("ticker").loc[ticker, "name"]

page_header("Risk & stress test", f"How much {name} could lose · Monte Carlo VaR from a GJR-GARCH(1,1)-t fit.")

# The stored model parameters are the one thing every panel below depends on.
params = try_fetch(repo.get_risk_parameters, ticker, as_of)
if params is None:
    empty_state(f"Risk figures aren't ready for {name}",
                "The risk engine has not been run for this share yet, so there is no value-at-risk or stress test "
                "to show. Try another share from the sidebar, or check back after the next model run.")
    footer()
    st.stop()

# ---- inputs ---------------------------------------------------------------
ss = st.session_state
ss.setdefault("vol_mult", 1.0)
ss.setdefault("shock", 0.0)

section("Scenario", "Position, volatility scaling and a one-off shock")
i1, i2, i3, i4 = st.columns([1.3, 1, 1, 0.8])
with i1:
    notional = st.number_input(tr("Position size (LKR)"), min_value=10_000, value=1_000_000, step=50_000,
                               format="%d")
with i2:
    st.slider(tr("Volatility multiplier"), 0.5, 4.0, key="vol_mult", step=0.1,
              help="Scales the starting conditional volatility. 1.0 = as estimated.")
with i3:
    st.number_input(tr("Initial return shock (%)"), -20.0, 20.0, key="shock", step=0.5,
                    help="A one-off return applied on day one before the paths evolve.")
with i4:
    st.write("")
    st.write("")
    def _crisis():   # callbacks run before widgets are rebuilt, so the slider values can be set here
        ss.vol_mult = float(min(4.0, max(0.5, params["crisis_vol_multiplier"])))
        ss.shock = float(min(20.0, max(-20.0, params["crisis_shock"] * 100)))

    def _reset():
        ss.vol_mult, ss.shock = 1.0, 0.0

    st.button(tr("2022 crisis scenario"), use_container_width=True, type="primary", on_click=_crisis)
    st.button(tr("Reset"), use_container_width=True, on_click=_reset)

stressed = ss.vol_mult != 1.0 or ss.shock != 0.0
if stressed:
    st.markdown(
        f"<span class='pill pill-warn'>Stress applied · volatility ×{ss.vol_mult:.1f} · "
        f"initial shock {ss.shock:+.1f}%</span>"
        "<span class='section-cap' style='margin-left:0.6rem'>Reset the controls to return to the base estimate.</span>",
        unsafe_allow_html=True,
    )

# ---- VaR table (stored estimates or live stress) ----------------------------
base = try_fetch(repo.get_var_estimate, ticker, as_of)
with timed("stress", f"{ticker} 2×10,000 paths") as stress_time:
    sims = {h: simulate_returns(params, h, vol_multiplier=ss.vol_mult, initial_shock=ss.shock / 100)
            for h in (1, 5)}

section("Value at risk" + (" · stressed" if stressed else ""),
        "On 1 day in 20 (95%) or 1 in 100 (99%), losses are expected to exceed this amount")
if base is None and not stressed:
    empty_state("No stored risk estimate for this date",
                f"The overnight risk run has not produced value-at-risk figures for {name} on {as_of:%d %B %Y}. "
                "Move the volatility slider or apply the crisis scenario to run a live estimate instead.")
else:
    cols = st.columns(4)
    i = 0
    for h in (1, 5):
        for conf in (0.95, 0.99):
            if stressed or base is None:
                var_pct = value_at_risk(sims[h], conf) * 100
            else:
                var_pct = float(base[(base.horizon_days == h) & (base.confidence == conf)].var_pct.iloc[0])
            with cols[i]:
                card(f"{h}-day · {conf:.0%}", f"{var_pct:.2f}%", f"≈ {fmt_lkr(notional * var_pct / 100)} loss", "down")
            i += 1

st.write("")

# ---- distribution ----------------------------------------------------------
section(f"Simulated {1 if horizon == 't+1' else 5}-day return distribution",
        f"10,000 Monte Carlo paths · tail parameter ν = {params['nu']:.1f}")
h = 1 if horizon == "t+1" else 5
r = sims[h] * 100
fig = go.Figure(go.Histogram(x=r, nbinsx=90, marker=dict(color=UP, opacity=0.85), name="paths"))
var_labels = {}
for conf, dash, pos in ((0.95, "dash", "top right"), (0.99, "dot", "top left")):
    v = -value_at_risk(sims[h], conf) * 100
    var_labels[conf] = abs(v)
    fig.add_vline(x=v, line=dict(color=DOWN, dash=dash, width=1.5),
                  annotation_text=f"VaR {conf:.0%}: {abs(v):.2f}%", annotation_position=pos,
                  annotation_font=dict(color=DOWN, size=11),
                  annotation_bgcolor="rgba(16,20,29,0.85)")
fig.update_layout(height=340, xaxis_title="cumulative return (%)", yaxis_title="paths", showlegend=False,
                  bargap=0.06)
st.plotly_chart(fig, use_container_width=True, config=CONFIG)
chart_caption(
    f"Histogram of {len(r):,} simulated {h}-day returns for {name}: most paths cluster near "
    f"{np.mean(r):+.2f}%, with a long left tail. The two dashed lines mark the loss that only 5% "
    f"(−{var_labels[0.95]:.2f}%) and 1% (−{var_labels[0.99]:.2f}%) of paths exceed.")
st.caption(f"10,000 paths · mean {np.mean(r):+.2f}% · 1st percentile {np.percentile(r, 1):+.2f}% · "
           f"tail parameter ν = {params['nu']:.1f} · simulation ran in {fmt_seconds(stress_time['seconds'])} "
           f"(target under {TARGETS_SECONDS['stress']:.0f} s)")

with st.expander(tr("Details for experts")):
    # ---- Kupiec ----------------------------------------------------------------
    section("Kupiec backtest of the 1-day VaR",
            "Proportion-of-failures test on rolling out-of-sample estimates · p > 0.05 means the breach rate "
            "is consistent with the stated confidence level")
    k = try_fetch(repo.get_kupiec_results, ticker)
    if k is None or k.empty:
        empty_state("No backtest results yet",
                    f"The value-at-risk model for {name} has not been checked against history yet, so we cannot say "
                    "how reliable its loss estimates have been.")
    else:
        k = k.copy()
        k["confidence"] = k["confidence"].map(lambda c: f"{c:.0%}")
        responsive_table(
            "kupiec", k,
            column_config={
                "confidence": "Confidence",
                "observations": "Days", "breaches": "Breaches",
                "expected_breaches": st.column_config.NumberColumn("Expected", format="%.1f"),
                "lr_pof": st.column_config.NumberColumn("LR (POF)", format="%.3f"),
                "p_value": st.column_config.NumberColumn("p-value", format="%.3f"),
                "verdict": "Verdict",
            },
            title=lambda r: f"{r.confidence} confidence · {r.verdict}",
            fields=[("Days", lambda r: f"{int(r.observations)}"), ("Breaches", lambda r: f"{int(r.breaches)}"),
                    ("Expected", lambda r: f"{r.expected_breaches:.1f}"), ("LR (POF)", lambda r: f"{r.lr_pof:.3f}"),
                    ("p-value", lambda r: f"{r.p_value:.3f}")],
        )

footer()
