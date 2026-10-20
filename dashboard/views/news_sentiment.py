import plotly.graph_objects as go
import streamlit as st

from data.repository import get_repository
from ui.charts import CONFIG, polarity_label, show_chart
from ui.i18n import tr
from ui.layout import (DOWN, INDIGO, UP, chart_caption, footer, page_header, responsive_table, section,
                       selection, window)
from ui.states import empty_state, try_fetch

repo = get_repository()
ticker, as_of, _ = selection()
name = repo.list_stocks().set_index("ticker").loc[ticker, "name"]

page_header("News sentiment", f"Trilingual news signal for {name} against the wider market.")

section("Sentiment signal", "Market mood line · company news dots")
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
    s = s.sort_values("date")
    s["market_7d"] = s["market_score"].rolling(7, min_periods=1).mean()
    co = s[s["article_count"] > 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.date, y=s.market_7d, name="Market mood (7-day)",
                             line=dict(width=2.4, color=INDIGO)))
    if len(co):
        fig.add_trace(go.Scatter(x=co.date, y=co.company_score, mode="markers", name=f"{name} news",
                                 marker=dict(size=7 + 3 * co.article_count.clip(upper=5),
                                             color=[UP if v >= 0 else DOWN for v in co.company_score],
                                             line=dict(color="#10141D", width=1)),
                                 customdata=co.article_count,
                                 hovertemplate="%{y:+.2f} · %{customdata} article(s)<extra></extra>"))
    fig.add_hline(y=0, line_color="#2A3140", line_width=1)
    fig.update_layout(height=340, hovermode="closest", yaxis=dict(title="news tone (−1 … +1)", range=[-1, 1]),
                      legend=dict(orientation="h", y=1.12, x=0, entrywidth=0.45, entrywidthmode="fraction"))
    show_chart(fig, use_container_width=True, config=CONFIG, theme=None)

    mk_last = s.dropna(subset=["market_score"]).iloc[-1]
    c1, c2, c3 = st.columns(3)
    if len(co):
        lc = co.iloc[-1]
        c1.metric(f"{name} · latest news", f"{lc.company_score:+.2f}", f"{polarity_label(lc.company_score)} · {lc.date:%d %b}",
                  delta_color="off")
    else:
        c1.metric(f"{name} · latest news", "—", f"no articles in {n} days", delta_color="off")
    c2.metric("Market mood (7-day)", f"{mk_last.market_7d:+.2f}", polarity_label(mk_last.market_7d), delta_color="off")
    c3.metric("Articles about the company", int(co.article_count.sum()) if len(co) else 0)
    chart_caption(
        f"The blue line is the 7-day average tone of all business news in English, Sinhala and Tamil over the last "
        f"{n} days (−1 very negative, +1 very positive). Each dot is a day with news about {name}: green above zero, "
        f"red below, bigger dots mean more articles. {len(co)} day(s) in this window had news about {name}.")

# ---- articles -------------------------------------------------------------
section("Articles", "Headline, source, date, link, linked company and score only · article text is licensed and never shown")
f1, f2 = st.columns([1, 2])
with f1:
    scope = st.radio("Show", ["This company", "All companies"], horizontal=True, label_visibility="collapsed",
                     format_func=tr)
with f2:
    langs = st.multiselect("Language", ["en", "si", "ta"], default=["en", "si", "ta"],
                           format_func=LANG.get, label_visibility="collapsed")

articles = try_fetch(repo.get_articles, ticker if scope == "This company" else None, start, end)
if articles is not None:
    articles = articles[articles.language.isin(langs)].copy()
    articles["language"] = articles.language.map(LANG)
    articles["tone"] = articles.score.map(polarity_label)
    articles = articles.sort_values("published", ascending=False)

if articles is None or articles.empty:
    who = name if scope == "This company" else "any company"
    hint = ("Widen the window, tick more languages, or switch to “All companies” to see what the market was reading."
            if scope == "This company" else "Widen the window or tick more languages.")
    empty_state(f"No articles about {who} match these filters",
                f"Nothing was published in the selected languages in the {n} days before {as_of:%d %B %Y}. {hint}")
else:
    responsive_table(
        "articles", articles, columns=["published", "headline", "language", "ticker", "tone", "score", "url"],
        height=460, limit=20,
        column_config={
            "published": st.column_config.DateColumn("Published", format="DD MMM YYYY"),
            "headline": st.column_config.TextColumn("Headline", width="large"),
            "language": "Language", "ticker": "Company", "tone": "Tone",
            "score": st.column_config.NumberColumn("Score (−1…+1)", format="%+.2f"),
            "url": st.column_config.LinkColumn("Link", display_text="Open"),
        },
        title=lambda r: r.headline,
        subtitle=lambda r: f"{r.source} · {r.published:%d %b %Y} · {r.language}",
        fields=[("Company", lambda r: r.ticker),
                ("Score", lambda r: f"{r.score:+.2f} {polarity_label(r.score)}")],
        link=lambda r: ("Open article", r.url),
    )

footer()
