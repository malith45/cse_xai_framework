"""Plotly template and small chart builders shared across pages."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from ui.layout import DOWN, INDIGO, INK, INK2, MUTED, UP, VIOLET

GRID = "rgba(148,163,184,0.08)"
AXIS = "rgba(148,163,184,0.16)"

_t = go.layout.Template()
_t.layout = go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=INK2, size=12),
    title=dict(font=dict(size=13, color=MUTED, family="Inter, system-ui, sans-serif"), x=0.02, xanchor="left"),
    margin=dict(l=12, r=12, t=40, b=12),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor=AXIS, showline=False, tickfont=dict(size=11, color=MUTED),
               title=dict(font=dict(size=11, color=MUTED)), ticks="", tickcolor="rgba(0,0,0,0)", automargin=True),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor=AXIS, showline=False, tickfont=dict(size=11, color=MUTED),
               title=dict(font=dict(size=11, color=MUTED)), ticks="", tickcolor="rgba(0,0,0,0)", automargin=True),
    legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED, size=11),
                itemsizing="constant"),
    # Validated categorical order (adjacent pairs ≥ 8 ΔE under CVD simulation, ≥ 15 normal vision).
    colorway=[UP, INDIGO, DOWN, VIOLET],
    hoverlabel=dict(bgcolor="#161B26", bordercolor="rgba(148,163,184,0.25)",
                    font=dict(color=INK, size=12, family="Inter, system-ui, sans-serif")),
    hovermode="closest",
    # Responsive: the figure fills whatever width the container gives it.
    autosize=True,
)
pio.templates["cse_dark"] = _t
pio.templates.default = "cse_dark"


def apply_chart_theme(theme: str) -> None:
    """Keep Plotly text and hover surfaces aligned with the selected UI theme."""
    if theme == "light":
        _t.layout.font.color = "#4B5568"
        _t.layout.title.font.color = "#6B7280"
        _t.layout.xaxis.tickfont.color = "#6B7280"
        _t.layout.xaxis.title.font.color = "#6B7280"
        _t.layout.yaxis.tickfont.color = "#6B7280"
        _t.layout.yaxis.title.font.color = "#6B7280"
        _t.layout.legend.font.color = "#6B7280"
        _t.layout.hoverlabel.bgcolor = "#FFFFFF"
        _t.layout.hoverlabel.font.color = "#17233D"
    else:
        _t.layout.font.color = INK2
        _t.layout.title.font.color = MUTED
        _t.layout.xaxis.tickfont.color = MUTED
        _t.layout.xaxis.title.font.color = MUTED
        _t.layout.yaxis.tickfont.color = MUTED
        _t.layout.yaxis.title.font.color = MUTED
        _t.layout.legend.font.color = MUTED
        _t.layout.hoverlabel.bgcolor = "#161B26"
        _t.layout.hoverlabel.font.color = INK
    pio.templates.default = "cse_dark"

CONFIG = {"displayModeBar": False, "responsive": True}


def polarity_label(value: float, pos: str = "Positive", neg: str = "Negative", neutral: str = "Neutral",
                   deadband: float = 0.05) -> str:
    """Text + arrow for a signed value so meaning never rests on colour alone."""
    if value > deadband:
        return f"▲ {pos}"
    if value < -deadband:
        return f"▼ {neg}"
    return f"● {neutral}"


def gauge(value: float, title: str) -> go.Figure:
    """Sentiment gauge on [-1, 1] with a written polarity label under the number."""
    tone = UP if value >= 0 else DOWN
    fig = go.Figure(go.Indicator(
        mode="gauge", value=value,
        title={"text": title, "font": {"size": 12, "color": MUTED}},
        gauge={
            "axis": {"range": [-1, 1], "tickvals": [-1, 0, 1], "ticktext": ["−1 negative", "0", "+1 positive"],
                     "tickcolor": "rgba(0,0,0,0)", "tickfont": {"color": MUTED, "size": 10}},
            "bar": {"color": tone, "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": [{"range": [-1, 0], "color": "rgba(212,106,18,0.10)"},
                      {"range": [0, 1], "color": "rgba(15,168,144,0.10)"}],
            "threshold": {"line": {"color": tone, "width": 2}, "thickness": 0.6, "value": value},
        },
    ))
    # The bundled plotly.js ignores number.valueformat on indicators, so the readout is drawn
    # as a pre-formatted annotation instead of the indicator's own number.
    fig.add_annotation(x=0.5, y=0.08, xref="paper", yref="paper", text=f"{value:+.2f}", showarrow=False,
                       font=dict(size=32, color=INK, family="Inter, system-ui, sans-serif"))
    fig.add_annotation(x=0.5, y=-0.14, xref="paper", yref="paper", text=polarity_label(value), showarrow=False,
                       font=dict(size=12, color=tone, family="Inter, system-ui, sans-serif"))
    fig.update_layout(height=240, margin=dict(l=24, r=24, t=44, b=30))
    return fig
