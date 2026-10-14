import plotly.graph_objects as go
import streamlit as st

from data.repository import get_repository
from ui.charts import CONFIG
from ui.layout import DOWN, MUTED, UP, chart_caption, footer, page_header, section, selection
from ui.states import empty_state, try_fetch

repo = get_repository()
ticker, as_of, horizon = selection()
name = repo.list_stocks().set_index("ticker").loc[ticker, "name"]
status = repo.get_data_status()

page_header("Explainability", f"Why the model predicted what it did for {name} · {horizon} · {as_of:%d %b %Y}.")

with st.expander("How to read this page (no maths needed)"):
    st.markdown(
        """
- **What the model does.** For each share it looks at about fifteen signals — recent price moves, trading
  volume, news tone, the exchange rate, interest rates and so on — and turns them into one number: the chance
  that the price goes **up**.
- **Left chart: what matters in general.** Each bar is one signal. A longer bar means that signal changes the
  model's answer more, averaged over every share and every day we tested.
- **Right chart: what happened for *this* share on *this* date.** Each bar is one signal's nudge.
  A bar pointing **right (▲, teal)** pushed the answer towards *up*; a bar pointing **left (▼, orange)** pulled it
  towards *down*. Longer means a stronger nudge. Start from a neutral point, add up all the nudges, and you land on
  today's probability.
- **The numbers are called SHAP values.** Think of each one as "how much did this single signal change the
  model's mind, and in which direction?" That is all you need.
- **Reading the result.** A probability close to 50% means the signals roughly cancel out and the evidence is weak.
  The model summarises patterns in past data; it is not advice, and past patterns may not repeat.
        """
    )

left, right = st.columns(2)

# ---- global importance ----------------------------------------------------
with left:
    section("What drives predictions overall", f"Mean |SHAP| across all stocks and days · {horizon} model")
    g = try_fetch(repo.get_global_importance, horizon)
    if g is None or g.empty:
        empty_state("No overall importance figures", "The model summary for this horizon has not been produced yet.")
    else:
        g = g.head(12).iloc[::-1]
        fig = go.Figure(go.Bar(x=g.mean_abs_shap, y=g.feature, orientation="h",
                               marker=dict(color=UP, opacity=0.9, cornerradius=4),
                               text=[f"{v:.3f}" for v in g.mean_abs_shap], textposition="outside",
                               textfont=dict(size=10, color=MUTED), cliponaxis=False))
        fig.update_layout(height=440, xaxis_title="mean |SHAP| (average size of the nudge)", bargap=0.35,
                          xaxis=dict(range=[0, float(g.mean_abs_shap.max()) * 1.28]),  # room for the value labels
                          yaxis=dict(tickfont=dict(size=12, color="#B4BDCC"), showgrid=False))
        st.plotly_chart(fig, use_container_width=True, config=CONFIG)
        top = g.iloc[::-1]
        chart_caption(
            f"Horizontal bars ranking the twelve most influential signals for the {horizon} model. "
            f"The strongest is {top.feature.iloc[0]} ({top.mean_abs_shap.iloc[0]:.3f}), followed by "
            f"{top.feature.iloc[1]} and {top.feature.iloc[2]}. Longer bar = bigger average effect on the prediction.")

# ---- local explanation ----------------------------------------------------
with right:
    exp = try_fetch(repo.get_shap_explanation, ticker, as_of, horizon)
    pred = try_fetch(repo.get_prediction, ticker, as_of, horizon)
    if exp is None or pred is None:
        section("What drove this prediction")
        empty_state("No prediction to explain for this date",
                    f"An explanation needs a prediction, and there is none for {name} on {as_of:%d %B %Y}. "
                    f"Predictions exist only for trading days up to {status['as_of']:%d %B %Y}. Pick another date "
                    "in the sidebar.")
        c = None
    else:
        c = exp["contributions"]
        section("What drove this prediction",
                f"Top 10 contributions · base value {exp['base_value']:+.2f} → P(up) {pred['p_up']:.0%}")
        vals = c.shap.iloc[::-1]
        fig = go.Figure(go.Waterfall(
            orientation="h", y=c.feature.iloc[::-1], x=vals,
            measure=["relative"] * len(c),
            increasing=dict(marker=dict(color=UP)), decreasing=dict(marker=dict(color=DOWN)),
            connector=dict(line=dict(color="#2A3140", width=1)),
            text=[f"{'▲' if v > 0 else '▼'} {v:+.2f}" for v in vals], textposition="outside",
            textfont=dict(color="#B4BDCC", size=11), cliponaxis=False, showlegend=False,
        ))
        # Legend swatches so the colour meaning is spelled out in words as well.
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="▲ pushes towards UP",
                                 marker=dict(color=UP, symbol="square", size=11)))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="▼ pulls towards DOWN",
                                 marker=dict(color=DOWN, symbol="square", size=11)))
        fig.add_vline(x=0, line_color="#2A3140", line_width=1)
        # Pad the axis so the "▲ +0.37" labels sit clear of the bars instead of over them.
        running = [0.0]
        for v in vals:
            running.append(running[-1] + float(v))
        span = max(running) - min(running) or 1.0
        fig.update_layout(height=440, xaxis_title="contribution to log-odds of an up move", showlegend=True,
                          legend=dict(orientation="h", y=1.12, x=0),
                          xaxis=dict(range=[min(running) - 0.22 * span, max(running) + 0.22 * span]),
                          yaxis=dict(tickfont=dict(size=12, color="#B4BDCC"), showgrid=False))
        st.plotly_chart(fig, use_container_width=True, config=CONFIG)
        pos = c[c.shap > 0]
        neg = c[c.shap < 0]
        chart_caption(
            f"Waterfall of the ten signals that moved the {horizon} prediction for {name} on {as_of:%d %B}. "
            f"{len(pos)} signals pushed towards up (largest: "
            f"{pos.feature.iloc[0] + f' {pos.shap.iloc[0]:+.2f}' if len(pos) else 'none'}) and {len(neg)} pulled "
            f"towards down (largest: {neg.feature.iloc[0] + f' {neg.shap.iloc[0]:+.2f}' if len(neg) else 'none'}). "
            f"Together they take the model from neutral to a {pred['p_up']:.0%} chance of a rise.")

# ---- narrative ------------------------------------------------------------
narrative = try_fetch(repo.get_narrative, ticker, as_of, horizon) if c is not None else None
if narrative:
    section("In plain language", "Generated from the SHAP contributions above")
    st.markdown(f'<div class="narrative">{narrative}</div>', unsafe_allow_html=True)

if c is not None:
    st.write("")
    with st.expander("Feature values behind this explanation"):
        st.dataframe(c, hide_index=True, use_container_width=True,
                     column_config={"feature": "Feature",
                                    "value": st.column_config.NumberColumn("Standardised value", format="%.2f"),
                                    "shap": st.column_config.NumberColumn("SHAP", format="%+.4f")})

footer()
