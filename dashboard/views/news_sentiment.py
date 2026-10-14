import plotly.graph_objects as go
import streamlit as st

from data.repository import get_repository
from ui.charts import CONFIG, polarity_label
from ui.layout import (DOWN, INDIGO, UP, chart_caption, footer, page_header, responsive_table, section,
                       selection, window)
from ui.states import empty_state, try_fetch

repo = get_repository()
ticker, as_of, _ = selection()
name = repo.list_stocks().set_index("ticker").loc[ticker, "name"]

page_header("News sentiment", f"Trilingual news signal for {name} against the wider market.")

section("Sentiment signal", "Company vs market score · daily article volume")
days = st.segmented_control("Window", ["30 days", "90 days", "180 days"], default="90 days",
                            label_visibility="collapsed")
n = int((days or "90 days").split()[0])
start, end = window(as_of, n)

LANG = {"en": "English", "si": "සිංහල", "ta": "தமிழ்"}

# ---- series ---------------------------------------------------------------
s = try_fetch(repo.get_sentiment_series, ticker, start, end)
if s is None or s.empty:
    empty_state(f"No news for {name} in this period",
                f"We did not find any scored articles about {name} in the {n} days before {as_of:%d %B %Y}. "
                "Try a longer window, or choose another share from the sidebar. Some smaller companies are rarely "
                "covered in the press.")
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=s.date, y=s.article_count, name="articles per day (bars)", yaxis="y2",
                         marker=dict(color="#2A3140", cornerradius=2), opacity=0.9))
    fig.add_trace(go.Scatter(x=s.date, y=s.market_score, name="Market tone (thin line)",
                             line=dict(width=1.6, color=INDIGO)))
    fig.add_trace(go.Scatter(x=s.date, y=s.company_score, name=f"{name} tone (thick, shaded)",
                             line=dict(width=2.2, color=UP), fill="tozeroy", fillcolor="rgba(15,168,144,0.12)"))
    fig.add_hline(y=0, line_color="#2A3140", line_width=1,
                  annotation_text="neutral", annotation_position="bottom right",
                  annotation_font=dict(size=10, color="#7C8698"))
    fig.update_layout(
        height=380, hovermode="x unified",
        yaxis=dict(title="sentiment score (−1 negative … +1 positive)", range=[-1, 1]),
        yaxis2=dict(title="articles / day", overlaying="y", side="right", showgrid=False, rangemode="tozero",
                    tickfont=dict(size=11, color="#7C8698"), title_font=dict(size=11, color="#7C8698")),
    )
    st.plotly_chart(fig, use_container_width=True, config=CONFIG)

    last = s.iloc[-1]
    chart_caption(
        f"Daily news tone over the last {n} days, from −1 (very negative) to +1 (very positive). The thick "
        f"shaded line is {name}, the thin line is the whole market, and the grey bars count articles per day. "
        f"On the latest day {name} scored {last.company_score:+.2f} ({polarity_label(last.company_score).split(' ', 1)[1].lower()}) "
        f"against a market tone of {last.market_score:+.2f}.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Company score (latest)", f"{last.company_score:+.2f}", polarity_label(last.company_score) + " tone",
              delta_color="off")
    c2.metric("Market score (latest)", f"{last.market_score:+.2f}", polarity_label(last.market_score) + " tone",
              delta_color="off")
    c3.metric("Articles in window", int(s.article_count.sum()))

# ---- articles -------------------------------------------------------------
section("Articles", "Headline, source, date, link, linked company and score only · article text is licensed and never shown")
f1, f2 = st.columns([1, 2])
with f1:
    scope = st.radio("Show", ["This company", "All companies"], horizontal=True, label_visibility="collapsed")
with f2:
    langs = st.multiselect("Language", ["en", "si", "ta"], default=["en", "si", "ta"],
                           format_func=LANG.get, label_visibility="collapsed")

articles = try_fetch(repo.get_articles, ticker if scope == "This company" else None, start, end)
if articles is not None:
    articles = articles[articles.language.isin(langs)].copy()
    articles["language"] = articles.language.map(LANG)

if articles is None or articles.empty:
    who = name if scope == "This company" else "any company"
    hint = ("Widen the window, tick more languages, or switch to “All companies” to see what the market was reading."
            if scope == "This company" else "Widen the window or tick more languages.")
    empty_state(f"No articles about {who} match these filters",
                f"Nothing was published in the selected languages in the {n} days before {as_of:%d %B %Y}. {hint}")
else:
    responsive_table(
        "articles", articles, columns=["published", "headline", "source", "language", "ticker", "score", "url"],
        height=460, limit=20,
        column_config={
            "published": st.column_config.DateColumn("Published", format="DD MMM YYYY"),
            "headline": st.column_config.TextColumn("Headline", width="large"),
            "source": "Source", "language": "Language", "ticker": "Company",
            "score": st.column_config.ProgressColumn("Score", min_value=-1, max_value=1, format="%+.2f"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
        },
        title=lambda r: r.headline,
        subtitle=lambda r: f"{r.source} · {r.published:%d %b %Y} · {r.language}",
        fields=[("Company", lambda r: r.ticker),
                ("Score", lambda r: f"{r.score:+.2f} {polarity_label(r.score)}")],
        link=lambda r: ("Open article", r.url),
    )

footer()
