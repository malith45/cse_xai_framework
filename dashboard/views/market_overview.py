import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from data.repository import get_repository
from ui.charts import CONFIG, gauge, polarity_label, show_chart
from ui.layout import (DOWN, INDIGO, UP, card, chart_caption, footer, page_header, pill, responsive_table,
                       section, selection, window)
from ui.i18n import tr
from ui.states import empty_state, try_fetch

repo = get_repository()
_, as_of, _ = selection()

page_header("Market overview", f"Where the market stood at the close on {as_of:%d %B %Y}.")

# ---- headline cards -------------------------------------------------------
hist = try_fetch(repo.get_index_history, *window(as_of, 365))
if hist is None or len(hist) < 2:
    empty_state("No market data for this date",
                f"We have no index history ending on {as_of:%d %B %Y}. That usually means the market was closed "
                "that day, or the date is before our records begin (January 2022). Pick another date in the sidebar.")
    footer()
    st.stop()

breadth = try_fetch(repo.get_market_breadth, as_of)
sent = try_fetch(repo.get_market_sentiment, as_of)
risk = try_fetch(repo.get_market_risk_summary, as_of)
last, prev = hist.iloc[-1], hist.iloc[-2]

c1, c2, c3, c4 = st.columns(4)
with c1:
    chg = (last.aspi / prev.aspi - 1) * 100
    card("ASPI", f"{last.aspi:,.0f}", f"{'▲' if chg >= 0 else '▼'} {chg:+.2f}% on the day",
         "up" if chg >= 0 else "down")
with c2:
    chg = (last.sl20 / prev.sl20 - 1) * 100
    card("S&P SL20", f"{last.sl20:,.0f}", f"{'▲' if chg >= 0 else '▼'} {chg:+.2f}% on the day",
         "up" if chg >= 0 else "down")
with c3:
    if breadth:
        up_day = breadth["advances"] >= breadth["declines"]
        card("Breadth", f"{breadth['advances']} / {breadth['declines']}",
             f"advances / declines · {breadth['unchanged']} unchanged · {'more risers' if up_day else 'more fallers'}",
             "up" if up_day else "down")
    else:
        card("Breadth", "—", "not available for this date")
with c4:
    if risk:
        card("Market risk", tr(risk["regime"]),
             f"1-day VaR 95% ≈ {risk['var_1d_95']}% · 20-day vol {risk['realised_vol_20d']}%",
             "down" if risk["regime"] == "Elevated" else "")
    else:
        card("Market risk", "—", "not available for this date")

st.write("")

# ---- index history --------------------------------------------------------
left, right = st.columns([2.2, 1])
with left:
    section("Index history", "ASPI vs S&P SL20 · daily close")
    rng = st.segmented_control("Range", ["3M", "6M", "1Y"], default="6M", label_visibility="collapsed")
    days = {"3M": 92, "6M": 183, "1Y": 365}.get(rng or "6M", 183)
    h = try_fetch(repo.get_index_history, *window(as_of, days))
    if h is None or h.empty:
        empty_state("No index history for this range", "Try a longer range or a later date.")
    else:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=h.date, y=h.aspi, name="ASPI (left axis)", line=dict(width=2, color=UP)),
                      secondary_y=False)
        fig.add_trace(go.Scatter(x=h.date, y=h.sl20, name="S&P SL20 (right axis)",
                                 line=dict(width=2, dash="dot", color=INDIGO)), secondary_y=True)
        fig.update_yaxes(title_text="ASPI", secondary_y=False)
        fig.update_yaxes(title_text="S&P SL20", secondary_y=True, showgrid=False)
        fig.update_layout(height=360, hovermode="x unified", margin=dict(t=44),
                          legend=dict(entrywidth=0.45, entrywidthmode="fraction"))
        show_chart(fig, use_container_width=True, config=CONFIG, theme=None)
        first = h.iloc[0]
        chart_caption(
            f"Two lines over the last {rng or '6M'}: the solid ASPI line moved from {first.aspi:,.0f} to "
            f"{last.aspi:,.0f} and the dotted S&P SL20 line from {first.sl20:,.0f} to {last.sl20:,.0f}. "
            "Each index has its own scale, so compare the shapes of the lines rather than their heights.")

with right:
    section("Market sentiment")
    if sent:
        show_chart(gauge(sent["score"], f"{sent['article_count']} articles scored today"),
                        use_container_width=True, config=CONFIG, theme=None)
        chart_caption(f"Dial from −1 (very negative) to +1 (very positive). Today's news tone is "
                      f"{sent['score']:+.2f}, which reads as {polarity_label(sent['score']).split(' ', 1)[1].lower()}, "
                      f"based on {sent['article_count']} articles.")
    else:
        empty_state("No sentiment score for this date", "No articles were scored for this day.")
    if breadth:
        b = go.Figure(go.Bar(
            x=["▲ Advances", "▼ Declines", "● Unchanged"],
            y=[breadth["advances"], breadth["declines"], breadth["unchanged"]],
            marker_color=[UP, DOWN, "#3B4453"], marker_cornerradius=6,
            text=[breadth["advances"], breadth["declines"], breadth["unchanged"]],
            textposition="outside", textfont=dict(size=11, color="#B4BDCC"),
        ))
        b.update_layout(height=210, showlegend=False, yaxis=dict(visible=False), title="Breadth", bargap=0.45,
                        xaxis=dict(showgrid=False))
        show_chart(b, use_container_width=True, config=CONFIG, theme=None)
        chart_caption(f"Three bars counting shares that rose ({breadth['advances']}), fell "
                      f"({breadth['declines']}) or were unchanged ({breadth['unchanged']}) on the day.")

# ---- movers ---------------------------------------------------------------
section("Top movers", f"S&P SL20 constituents · close of {as_of:%d %b %Y}")
movers = try_fetch(repo.get_top_movers, as_of)
if movers is None or movers.empty:
    empty_state("No price moves recorded for this date",
                "There were no closing prices for this day, so we cannot rank gainers and losers.")
else:
    g, l = st.columns(2)

    def _table(df, caption: str, key: str) -> None:
        pill(caption, "up" if caption == "Gainers" else "down")
        responsive_table(
            key, df, columns=["ticker", "name", "close", "change_pct"],
            column_config={
                "ticker": "Ticker", "name": "Company",
                "close": st.column_config.NumberColumn("Close", format="Rs %.2f"),
                "change_pct": st.column_config.NumberColumn("Change", format="%+.2f%%"),
            },
            title=lambda r: f"{r.ticker} · {r['name']}",
            fields=[("Close", lambda r: f"Rs {r.close:,.2f}"),
                    ("Change", lambda r: f"{'▲' if r.change_pct >= 0 else '▼'} {r.change_pct:+.2f}%")],
        )

    with g:
        _table(movers.head(5), "Gainers", "gainers")
    with l:
        _table(movers.tail(5).iloc[::-1], "Losers", "losers")

footer()
