import plotly.graph_objects as go
import streamlit as st

from data.repository import get_repository
from ui.i18n import tr
from services.signal import cutoff_note, headline, reliability_sentence, strength, strength_supported
from ui.charts import CONFIG, show_chart
from ui.layout import (DOWN, MUTED, UP, card, chart_caption, footer, page_header, responsive_table, section,
                       selection, window)
from ui.states import empty_state, try_fetch

repo = get_repository()
ticker, as_of, horizon = selection()
name = repo.list_stocks().set_index("ticker").loc[ticker, "name"]
status = repo.get_data_status()

page_header("Stock prediction", f"{name} ({ticker}) · predicted direction from the {as_of:%d %b %Y} close.")

p1 = try_fetch(repo.get_prediction, ticker, as_of, "t+1")
p5 = try_fetch(repo.get_prediction, ticker, as_of, "t+5")
if p1 is None or p5 is None:
    if empty_state("No prediction for this date",
                   f"Predictions exist only for trading days, and only up to {status['as_of']:%d %B %Y}. Weekends, "
                   "market holidays and dates outside the model's history have none. Choose another date in the "
                   "sidebar, or jump straight to the latest one.",
                   action_label="Use the latest date", key="use-latest-date"):
        st.session_state["_reset_as_of"] = True
        st.rerun()
    footer()
    st.stop()

rel = {h: try_fetch(repo.get_signal_reliability, h) for h in ("t+1", "t+5")}
c1, c2, c3, c4 = st.columns(4)
for col, p, label, h in ((c1, p1, "Next day (t+1)", "t+1"), (c2, p5, "Five days (t+5)", "t+5")):
    with col:
        big, small = headline(p["direction"], p["p_up"], p.get("threshold"), strength_supported(rel[h]))
        tone = "" if strength(p["p_up"], p.get("threshold")) == "weak" else ("up" if p["direction"] == "up" else "down")
        card(label, big, f"{small} · target {p['target_date']:%d %b}", tone)
active = p1 if horizon == "t+1" else p5
with c3:
    card(f"P(up) · {horizon}", f"{active['p_up']*100:.0f}%", cutoff_note(active.get("threshold")))
with c4:
    card("Model", active["model"], active["model_version"])

st.write("")

# ---- price chart with prediction marker ----------------------------------
section("Recent price", f"{ticker} · 120-day window · OHLC")
px = try_fetch(repo.get_price_history, ticker, *window(as_of, 120))
if px is None or px.empty:
    empty_state("No recent prices", f"We have no closing prices for {name} in the 120 days before {as_of:%d %B %Y}.")
else:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=px.date, open=px.open, high=px.high, low=px.low, close=px.close,
                                 increasing_line_color=UP, decreasing_line_color=DOWN,
                                 increasing_fillcolor=UP, decreasing_fillcolor=DOWN,
                                 increasing_line_width=1.2, decreasing_line_width=1.2,
                                 whiskerwidth=0.3, name=ticker, showlegend=False))
    last_close = float(px.close.iloc[-1])
    colour = UP if active["direction"] == "up" else DOWN
    arrow = "▲" if active["direction"] == "up" else "▼"
    fig.add_trace(go.Scatter(
        x=[as_of, active["target_date"]], y=[last_close, last_close],
        mode="lines+markers", line=dict(color=colour, dash="dash", width=2),
        marker=dict(size=[7, 15], symbol=["circle", "triangle-up" if active["direction"] == "up" else "triangle-down"],
                    line=dict(color="#10141D", width=2)),
        name=f"Prediction {horizon}", hovertemplate=f"P(up) {active['p_up']:.0%}<extra></extra>",
    ))
    fig.add_annotation(x=active["target_date"], y=last_close, text=f"{arrow} {horizon} · {active['p_up']:.0%} up",
                       showarrow=False, yshift=24, font=dict(color=colour, size=12, family="Inter, system-ui, sans-serif"),
                       bgcolor="rgba(16,20,29,0.85)", bordercolor=colour, borderwidth=1, borderpad=4)
    fig.update_layout(height=400, xaxis_rangeslider_visible=False, hovermode="x unified",
                      legend=dict(font=dict(color=MUTED)))
    show_chart(fig, use_container_width=True, config=CONFIG, theme=None)
    chart_caption(
        f"Daily price candles for {name} over the last 120 trading days, closing at Rs {last_close:,.2f} on "
        f"{as_of:%d %B}. The dashed line and {arrow} marker show the model's call for {active['target_date']:%d %B}: "
        f"{'up' if active['direction'] == 'up' else 'down'}, with a {active['p_up']:.0%} chance of a rise "
        f"({cutoff_note(active.get('threshold'))}).")
with st.expander(tr("About this chart")):
  st.caption("How to read the signal: the model calls 'up' once P(up) reaches its own cut-off, which is tuned on past "
           "data and is not always 50%. Calls within 5 points of the cut-off are shown as 'Unclear' (with the side they lean to). "
           + reliability_sentence(rel[horizon], horizon))

st.page_link("views/explainability.py", label=tr("Why this prediction? See which signals drove it"),
             icon=":material/psychology:")

with st.expander(tr("Details for experts")):
    # ---- walk-forward accuracy ------------------------------------------------
    section(f"Historical directional accuracy · {horizon}",
            "Out-of-sample metrics per walk-forward test block (63 trading days each)")
    blocks = try_fetch(repo.get_block_accuracy, ticker, horizon)
    if blocks is None or blocks.empty:
        empty_state("No accuracy history yet",
                    f"The model has not been back-tested on {name} for the {horizon} horizon, so there is no track "
                    "record to show. The prediction above still stands, but treat it with extra caution.")
    else:
        responsive_table(
            "blocks", blocks,
            column_config={
                "block": "Block",
                "test_start": st.column_config.DateColumn("Test start", format="DD MMM YYYY"),
                "test_end": st.column_config.DateColumn("Test end", format="DD MMM YYYY"),
                "accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "n": "Days",
            },
            title=lambda r: f"Block {int(r.block)}",
            subtitle=lambda r: f"{r.test_start:%d %b %Y} – {r.test_end:%d %b %Y} · {int(r.n)} days",
            fields=[("Accuracy", lambda r: f"{r.accuracy:.3f}"), ("F1", lambda r: f"{r.f1:.3f}"),
                    ("Precision", lambda r: f"{r.precision:.3f}"), ("Recall", lambda r: f"{r.recall:.3f}")],
        )
        st.caption(f"Mean accuracy across blocks: {blocks.accuracy.mean():.3f} · a naive coin toss scores 0.500.")

footer()
