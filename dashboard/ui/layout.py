"""
Shared chrome: theme CSS, demonstration banner, disclaimer, sidebar controls
and small display helpers. Pages call `page_header()` first and `footer()` last.
"""
from __future__ import annotations

import html as _html
from datetime import date, timedelta
from pathlib import Path
from string import Template
from typing import Callable, Sequence

import pandas as pd
import streamlit as st

from ui.i18n import LANGS, guide, tr
from config import APP_TITLE, DISCLAIMER, QUESTIONNAIRE_URL, is_mock_mode
from data.repository import get_repository
from services.perf import TARGETS_SECONDS, end_page, fmt_seconds

# --------------------------------------------------------------------------- #
# Design tokens (kept in one place; Plotly template mirrors these in charts.py)
# --------------------------------------------------------------------------- #
INK = "#EEF2F7"          # primary text
INK2 = "#B4BDCC"         # secondary text
MUTED = "#7C8698"        # captions, axis labels
BG0 = "#060810"          # page plane
BG1 = "#0A0D15"          # sidebar plane
SURFACE = "#10141D"      # cards
SURFACE2 = "#161B26"     # inputs, hover
LINE = "#1E2431"         # hairline
LINE2 = "#2A3140"        # stronger hairline
# Polarity pair (up/down, positive/negative). Teal ↔ orange was chosen over
# teal ↔ rose because it survives red-green colour deficiency: OKLab ΔE 13.6
# under deutan simulation (≥ 8 target) vs 6.6 for rose. Colour is never the
# only cue — every polarity mark also carries ▲/▼ or a word.
UP = "#0FA890"           # rise / gain / pushes up   (validated chart slot 1)
INDIGO = "#5E6CEB"       # secondary series          (validated chart slot 2)
DOWN = "#D46A12"         # fall / loss / pulls down  (validated chart slot 3)
VIOLET = "#8F7AE5"       # tertiary series           (validated chart slot 4)
GOLD = "#C28306"         # spare; not in the colorway (too close to DOWN)
ACCENT = "#2DD4BF"       # bright accent for glows, focus rings, highlights only
AMBER = "#F5B731"        # reserved for the demonstration banner only
_LOGO_DIR = Path(__file__).resolve().parents[1] / "assets"


def logo_svg(theme: str = "dark") -> str:
    filename = "cse-decision-support-logo-light.svg" if theme == "light" else "cse-decision-support-logo-dark.svg"
    return (_LOGO_DIR / filename).read_text(encoding="utf-8")

_CSS = Template("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --ink: $INK; --ink2: $INK2; --muted: $MUTED;
  --bg0: $BG0; --bg1: $BG1; --surface: $SURFACE; --surface2: $SURFACE2;
  --line: $LINE; --line2: $LINE2;
  --up: $UP; --down: $DOWN; --accent: $ACCENT; --indigo: $INDIGO; --amber: $AMBER;
  --font: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --r-sm: 10px; --r-md: 14px; --r-lg: 20px;
  --shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 10px 30px -12px rgba(0,0,0,0.6);
  --glass: rgba(16,20,29,0.78);
}
body:has(.theme-light) {
  --ink: #17233D; --ink2: #4B5568; --muted: #6B7280;
  --bg0: #F5F7FA; --bg1: #FFFFFF; --surface: #FFFFFF; --surface2: #EEF2F5;
  --line: #E2E7ED; --line2: #CBD5DF; --glass: rgba(255,255,255,0.88);
  --up: #0F9F9A; --accent: #0F9F9A;
  --shadow: 0 1px 0 rgba(23,35,61,0.04) inset, 0 12px 30px -16px rgba(23,35,61,0.24);
}
body:has(.theme-light) .stApp {
  background: radial-gradient(1000px 500px at 8% -10%, rgba(15,159,154,0.09), transparent 60%), var(--bg0);
}
body:has(.theme-light) [data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(15,159,154,0.04), transparent 30%), var(--bg1) !important;
}
body:has(.theme-light) [data-testid="stBaseButton-primary"],
body:has(.theme-light) [data-testid="stBaseButton-primaryFormSubmit"],
body:has(.theme-light) [data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
  background: linear-gradient(135deg, #0F9F9A, #0B817D) !important;
}
body:has(.theme-light) .st-key-login-shell [data-testid="stTextInputRootElement"] {
  background: #FFFFFF !important; border-color: #CBD5DF !important;
  color: var(--ink) !important; box-shadow: 0 2px 8px rgba(23,35,61,0.06) !important;
}
body:has(.theme-light) .st-key-login-shell [data-testid="stTextInputRootElement"]:focus-within {
  background: #FFFFFF !important; border-color: #0F9F9A !important;
  box-shadow: 0 0 0 3px rgba(15,159,154,0.14) !important;
}
body:has(.theme-light) .st-key-login-shell [data-testid="stTextInput"] input {
  color: #17233D !important; caret-color: #0F9F9A !important;
}
body:has(.theme-light) [data-testid="stBaseButton-primary"] p,
body:has(.theme-light) [data-testid="stBaseButton-primaryFormSubmit"] p,
body:has(.theme-light) [data-testid="stFormSubmitButton"] button p {
  color: #FFFFFF !important;
}
body:has(.theme-light) .st-key-login-shell {
  background:
    linear-gradient(var(--glass), var(--glass)) padding-box,
    linear-gradient(160deg, rgba(15,159,154,0.32), rgba(23,35,61,0.08)) border-box;
}
.st-key-theme-bar {
  position: fixed; top: 3.5rem; right: 1.25rem; z-index: 100;
  margin: 0; width: 230px; max-width: calc(100vw - 2rem);
}
.st-key-theme-bar > div { display: flex; justify-content: flex-end; }
.st-key-theme-bar [data-testid="stHorizontalBlock"] { width: min(100%, 230px); margin-left: auto; }
.st-key-theme-bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child { display: none; }
.st-key-theme-bar [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child { width: 100%; }
.st-key-theme-bar [data-testid="stButtonGroup"] {
  background: var(--surface); border: 1px solid var(--line2); border-radius: 999px;
  padding: 0.18rem; box-shadow: var(--shadow);
}
.st-key-theme-bar button[data-variant="segmented_control"] {
  min-height: 2rem; border-radius: 999px !important; color: var(--muted) !important;
  padding: 0.25rem 0.7rem !important; font-size: 0.76rem !important;
}
.st-key-theme-bar button[data-variant="segmented_control"][aria-checked="true"],
.st-key-theme-bar [data-testid*="segmented_controlActive"],
.st-key-theme-bar button[aria-pressed="true"],
.st-key-theme-bar button[aria-selected="true"],
.st-key-theme-bar button[data-selected="true"] {
  background: var(--surface2) !important; color: var(--ink) !important;
  box-shadow: 0 1px 3px rgba(0,0,0,0.22);
}
[data-testid="stAppViewContainer"] .block-container:has(.gate) {
  min-height: 100vh; padding-top: 0; padding-bottom: 0;
  display: flex; flex-direction: column; justify-content: center;
}
[data-testid="stAppViewContainer"] .block-container:has(.gate) > div:first-child { width: 100%; }
[data-testid="stVerticalBlock"].st-key-login-shell {
  width: min(100%, 560px) !important; max-width: 560px !important;
  margin-left: auto !important; margin-right: auto !important;
  position: relative; top: -3.5rem;
}
[data-testid="stVerticalBlock"].st-key-login-shell > div,
[data-testid="stVerticalBlock"].st-key-login-shell > div > div,
[data-testid="stVerticalBlock"].st-key-login-shell [data-testid="stElementContainer"],
[data-testid="stVerticalBlock"].st-key-login-shell [data-testid="stMarkdown"] {
  width: 100% !important; max-width: 100% !important;
}
[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
[data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] { margin-bottom: 0.25rem; }

/* ---------- global type & plane ------------------------------------------ */
html, body, [class*="css"], .stApp, .stMarkdown, .stDataFrame, .stApp * {
  font-family: var(--font) !important;
}
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
code, pre, .mono { font-family: var(--mono) !important; }
.stApp {
  color: var(--ink);
  background:
    radial-gradient(1200px 600px at 8% -10%, rgba(45,212,191,0.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(94,108,235,0.10), transparent 60%),
    radial-gradient(700px 500px at 60% 120%, rgba(143,122,229,0.06), transparent 60%),
    var(--bg0);
  background-attachment: fixed;
}
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stHeader"] { background: transparent !important; backdrop-filter: none; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stAppDeployButton"] { display: none; }
body [data-testid="stToolbar"],
body [data-testid="stAppViewContainer"] [data-testid="stToolbar"] {
  display: none !important; visibility: hidden !important; pointer-events: none !important;
}
body [data-testid="stToolbar"] button[aria-label="Main menu"],
body [data-testid="stHeader"] button[aria-label="Main menu"],
body [data-testid="stHeader"] [data-testid="stMainMenu"] {
  display: none !important; visibility: hidden !important; pointer-events: none !important;
}
[data-testid="stAppViewContainer"] header [data-testid="stAppDeployButton"] { display: none !important; }
[data-testid="stAppViewContainer"] header a[href*="github.com"],
[data-testid="stAppViewContainer"] header a[aria-label*="GitHub"],
[data-testid="stAppViewContainer"] header button[aria-label*="Fork"],
[data-testid="stAppViewContainer"] header button[aria-label*="Deploy"] {
  display: none !important;
}
.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1360px; }
.block-container > div:first-child { max-width: 1320px; margin: 0 auto; }
/* markdown elements that only carry a <style> tag should not take up layout space */
[data-testid="stElementContainer"]:has(> .stMarkdown > [data-testid="stMarkdownContainer"] > style:only-child) { display: none; }

::selection { background: rgba(45,212,191,0.28); color: #fff; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: var(--line2); border-radius: 10px; border: 2px solid var(--bg0); }
::-webkit-scrollbar-track { background: transparent; }

h1, h2, h3, h4 { color: var(--ink) !important; }
h1 { font-weight: 800 !important; letter-spacing: -0.035em; font-size: 2.15rem !important;
     line-height: 1.1 !important; margin: 0 !important; padding: 0 !important; }
h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em; }
p, li { color: var(--ink2); }
a { color: var(--accent); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
  color: var(--muted) !important; font-size: 0.8rem !important; letter-spacing: 0.005em;
}
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label {
  color: var(--muted) !important; font-size: 0.72rem !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: 0.10em;
}
[data-testid="stTooltipIcon"] svg { color: var(--muted); }
hr, [data-testid="stDivider"] { border-color: var(--line) !important; opacity: 1; }
[data-testid="stSidebar"] hr { margin: 1.1rem 0; }

/* ---------- motion --------------------------------------------------------- */
@keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes shimmer { 0% { background-position: 0 0; } 100% { background-position: 200px 0; } }
.card, .kpi, .section, .narrative, [data-testid="stPlotlyChart"], [data-testid="stDataFrame"], [data-testid="stMetric"] {
  animation: rise 0.45s cubic-bezier(.2,.7,.2,1) both;
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ---------- sidebar -------------------------------------------------------- */
[data-testid="stSidebar"] {
  background:
    radial-gradient(420px 240px at 0% 0%, rgba(45,212,191,0.08), transparent 72%),
    linear-gradient(180deg, rgba(45,212,191,0.04), transparent 30%), var(--bg1) !important;
  border-right: 1px solid var(--line);
  width: 360px !important;
  min-width: 360px !important;
  max-width: 360px !important;
}
body:has(.theme-light) [data-testid="stSidebar"] { border-right-color: #E2E7ED; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0; }
[data-testid="stSidebarHeader"] { padding-bottom: 0; }
[data-testid="stSidebarNavSeparator"] { display: none; }
[data-testid="stSidebarNav"] { padding: 0.9rem 0.7rem 0.7rem 0.7rem; }
[data-testid="stSidebarNav"]::before {
  content: "Workspace"; display: block; color: var(--muted); font-size: 0.66rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; padding: 0.5rem 0.9rem 0.45rem 0.9rem;
}
[data-testid="stSidebarNavItems"] { padding: 0; gap: 2px; display: flex; flex-direction: column; }
[data-testid="stSidebarNavItems"] li > div { margin: 0; }
[data-testid="stSidebarNavLink"] {
  border-radius: 12px; padding: 0.7rem 0.85rem; margin: 0; position: relative;
  background: transparent !important; transition: background .18s ease, color .18s ease;
  border: 1px solid transparent;
}
[data-testid="stSidebarNavLink"] span { color: var(--ink2) !important; font-weight: 500; font-size: 0.9rem; }
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] {
  font-size: 1.15rem; color: var(--muted) !important; margin-right: 0.35rem;
}
[data-testid="stSidebarNavLink"]:hover { background: rgba(255,255,255,0.035) !important; }
[data-testid="stSidebarNavLink"]:hover span { color: var(--ink) !important; }
[data-testid="stSidebarNavLink"]:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible,
[data-testid="stLinkButton"] a:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px;
}
/* active page — `nav_active_css()` emits a per-page href rule that sets these variables */
[data-testid="stSidebarNavLink"].nav-active, [data-testid="stSidebarNavLink"][aria-current="page"] {
  background: linear-gradient(90deg, rgba(45,212,191,0.14), rgba(45,212,191,0.04)) !important;
  border-color: rgba(45,212,191,0.18);
  box-shadow: 0 0 0 1px rgba(45,212,191,0.06) inset;
}
[data-testid="stSidebarUserContent"] { padding: 0.4rem 1.25rem 1.5rem 1.25rem; }
[data-testid="stSidebar"] .stMarkdown p { color: var(--ink2); font-size: 0.85rem; }

/* brand block is rendered inside the user content but pinned to the top of the sidebar, above the nav */
[data-testid="stSidebarContent"] { position: relative; }
[data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .brand) { position: static; }
[data-testid="stSidebarNav"] { padding-top: 0.6rem !important; }
[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .brand) {
  position: absolute; top: 0.85rem; left: 1rem; right: 1rem; z-index: 2;
}
.brand {
  display: flex; align-items: center; gap: 0.75rem; padding: 0.65rem 0.7rem;
  border: 1px solid var(--line); border-radius: 16px; background: var(--surface);
  box-shadow: var(--shadow);
}
.brand-mark {
  width: 40px; height: 40px; border-radius: 12px; flex: none; display: grid; place-items: center;
  background: var(--surface2); border: 1px solid var(--line2); padding: 0.25rem;
  box-shadow: 0 8px 20px -8px rgba(0,0,0,0.45);
}
.brand-mark svg { width: 100%; height: 100%; display: block; }
.brand-name { font-weight: 800; letter-spacing: -0.02em; font-size: 1.02rem; color: var(--ink); line-height: 1.15; }
.brand > div { min-width: 0; }
.brand-tag { color: var(--muted); font-size: 0.72rem; letter-spacing: 0.02em; margin-top: 2px;
             white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-label { color: var(--muted); font-size: 0.66rem; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; margin: 1rem 0 0.35rem 0; padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--line); }
.status {
  display: grid; gap: 0.5rem; padding: 0.85rem 0.9rem; border-radius: 14px; margin: 0.75rem 0 0.9rem 0;
  background: var(--surface); border: 1px solid var(--line);
}
.status-row { display: flex; justify-content: space-between; gap: 0.5rem; font-size: 0.74rem; }
.status-row .k { color: var(--muted); }
.status-row .v { color: var(--ink2); font-family: var(--mono) !important; font-size: 0.72rem; }
.live { display: inline-flex; align-items: center; gap: 0.4rem; color: var(--ink2); }
.live::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
                box-shadow: 0 0 0 3px rgba(45,212,191,0.18); }

/* ---------- inputs --------------------------------------------------------- */
[data-baseweb="select"] > div:first-child,
[data-testid="stTextInputRootElement"],
[data-testid="stDateInputField"], [data-testid="stNumberInputContainer"] {
  background: rgba(22,27,38,0.88) !important; border: 1px solid var(--line2) !important;
  border-radius: var(--r-sm) !important; color: var(--ink) !important;
  transition: border-color .15s ease, box-shadow .15s ease;
}
body:has(.theme-light) [data-baseweb="select"] > div:first-child,
body:has(.theme-light) [data-testid="stTextInputRootElement"],
body:has(.theme-light) [data-testid="stDateInputField"],
body:has(.theme-light) [data-testid="stNumberInputContainer"] {
  background: #FFFFFF !important; border-color: #CBD5DF !important;
}
[data-baseweb="select"] > div:first-child:hover,
[data-testid="stTextInputRootElement"]:hover,
[data-testid="stDateInputField"]:hover, [data-testid="stNumberInputContainer"]:hover {
  border-color: rgba(45,212,191,0.38) !important;
}
[data-testid="stTextInputRootElement"] > div { background: transparent !important; border: none !important; }
[data-baseweb="select"] input { color: var(--ink) !important; }
.st-key-login-shell [data-testid="stTextInputRootElement"] {
  background: #1B2230 !important; border-color: #46536A !important;
  border: 1px solid #46536A !important; border-radius: 10px !important;
}
.st-key-login-shell [data-testid="stTextInputRootElement"] > div {
  background: transparent !important; border: none !important;
}
.st-key-login-shell [data-testid="stTextInput"] input {
  color: var(--ink) !important; caret-color: var(--accent) !important;
}
[data-baseweb="select"] > div:first-child:focus-within,
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stDateInputField"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within {
  border-color: rgba(45,212,191,0.55) !important; box-shadow: 0 0 0 3px rgba(45,212,191,0.14) !important;
}
[data-baseweb="select"] svg, [data-testid="stTextInputRootElement"] svg { color: var(--muted); }
[data-testid="stNumberInput"] button { background: transparent !important; color: var(--muted) !important; border-left: 1px solid var(--line2) !important; }
[data-testid="stNumberInput"] button:hover { color: var(--ink) !important; background: rgba(255,255,255,0.04) !important; }
[data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"], [data-baseweb="calendar"] {
  background: var(--surface2) !important; border: 1px solid var(--line2) !important;
  border-radius: 12px !important; box-shadow: 0 20px 40px -12px rgba(0,0,0,0.7) !important;
}
[data-baseweb="popover"] input {
  display: none !important;
}
[data-baseweb="popover"] li[role="option"] { color: var(--ink2) !important; }
[data-baseweb="popover"] li[role="option"]:hover, [data-baseweb="popover"] li[aria-selected="true"] {
  background: rgba(45,212,191,0.10) !important; color: var(--ink) !important;
}
[data-baseweb="tag"] { background: rgba(45,212,191,0.12) !important; border: 1px solid rgba(45,212,191,0.25) !important;
                       border-radius: 8px !important; }
[data-baseweb="tag"] span { color: var(--ink) !important; }

/* segmented control → pill group */
[data-testid="stButtonGroup"] {
  background: var(--surface2); border: 1px solid var(--line2); border-radius: 12px; padding: 3px; gap: 2px;
  display: inline-flex;
}
[data-testid="stButtonGroup"] button {
  border: none !important; border-radius: 9px !important; background: transparent !important;
  color: var(--muted) !important; padding: 0.3rem 0.85rem !important; min-height: 0 !important;
  font-weight: 600 !important; font-size: 0.8rem !important; transition: all .15s ease;
}
[data-testid="stButtonGroup"] button:hover { color: var(--ink) !important; background: rgba(255,255,255,0.04) !important; }
[data-testid="stButtonGroup"] button p { font-size: 0.8rem !important; }
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"],
[data-testid="stButtonGroup"] [data-testid*="segmented_controlActive"],
[data-testid="stButtonGroup"] button[aria-pressed="true"],
[data-testid="stButtonGroup"] button[aria-selected="true"],
[data-testid="stButtonGroup"] button[data-selected="true"] {
  background: linear-gradient(180deg, rgba(45,212,191,0.22), rgba(45,212,191,0.12)) !important;
  color: var(--ink) !important; box-shadow: 0 0 0 1px rgba(45,212,191,0.5) inset, 0 4px 12px -6px rgba(45,212,191,0.6);
}
[data-testid="stButtonGroup"] button[data-variant="segmented_control"][aria-checked="true"] p,
[data-testid="stButtonGroup"] [data-testid*="segmented_controlActive"] p,
[data-testid="stButtonGroup"] button[aria-pressed="true"] p,
[data-testid="stButtonGroup"] button[aria-selected="true"] p,
[data-testid="stButtonGroup"] button[data-selected="true"] p {
  color: var(--ink) !important; font-weight: 800 !important;
}
.st-key-horizon-control { width: 100% !important; margin-top: 0.35rem; }
.st-key-horizon-control [data-testid="stWidgetLabel"] {
  color: var(--muted) !important; font-size: 0.66rem !important; font-weight: 700 !important;
  letter-spacing: 0.12em; text-transform: uppercase;
}
.st-key-horizon-control [data-baseweb="select"] > div:first-child {
  min-height: 2.6rem; background: var(--surface2) !important;
  border: 1px solid var(--line2) !important; border-radius: 11px !important;
}
.st-key-horizon-control [data-baseweb="select"] input,
.st-key-horizon-control [data-baseweb="select"] [data-testid="stMarkdownContainer"] p {
  color: var(--ink) !important; font-weight: 600 !important;
}
body:has(.theme-light) .st-key-horizon-control [data-baseweb="select"] > div:first-child {
  background: #FFFFFF !important; border-color: #CBD5DF !important;
}

/* radio → pills */
[data-testid="stRadio"] [role="radiogroup"] { display: flex; flex-direction: row; flex-wrap: wrap; gap: 0.4rem; }
[data-testid="stRadio"] label {
  background: var(--surface2); border: 1px solid var(--line2); border-radius: 999px; padding: 0.3rem 0.8rem 0.3rem 0.6rem;
  white-space: nowrap; margin: 0;
}
[data-testid="stRadio"] label:has(input:checked) { border-color: rgba(45,212,191,0.5); background: rgba(45,212,191,0.10); }
[data-testid="stRadio"] label p { color: var(--ink2) !important; font-size: 0.82rem; }

/* slider */
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] {
  color: var(--muted); font-family: var(--mono) !important; font-size: 0.68rem;
}
[data-testid="stSlider"] [role="slider"] { box-shadow: 0 0 0 4px rgba(45,212,191,0.18), 0 2px 8px rgba(0,0,0,0.5) !important; }
[data-testid="stSlider"] [data-testid="stSliderThumbValue"] { color: var(--accent) !important; font-family: var(--mono) !important; }

/* buttons */
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stFormSubmitButton"] button[kind="primaryFormSubmit"] {
  background: linear-gradient(135deg, #2DD4BF 0%, #14B8A6 60%, #5E6CEB 140%) !important;
  color: #04110F !important; border: none !important; border-radius: 11px !important; font-weight: 700 !important;
  letter-spacing: 0.01em; box-shadow: 0 10px 24px -10px rgba(45,212,191,0.7), 0 0 0 1px rgba(255,255,255,0.10) inset;
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
}
[data-testid="stBaseButton-primary"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  transform: translateY(-1px); filter: brightness(1.06);
  box-shadow: 0 14px 30px -10px rgba(45,212,191,0.8), 0 0 0 1px rgba(255,255,255,0.14) inset; }
[data-testid="stBaseButton-primary"] p, [data-testid="stBaseButton-primaryFormSubmit"] p,
[data-testid="stFormSubmitButton"] button p { color: #04110F !important; font-weight: 700; }
[data-testid="stBaseButton-secondary"], [data-testid="stLinkButton"] a {
  background: rgba(255,255,255,0.03) !important; color: var(--ink2) !important;
  border: 1px solid var(--line2) !important; border-radius: 11px !important; font-weight: 600 !important;
  transition: all .15s ease;
}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stLinkButton"] a:hover {
  border-color: rgba(45,212,191,0.45) !important; color: #fff !important; background: rgba(45,212,191,0.07) !important;
}
[data-testid="stLinkButton"] a p { color: inherit !important; }
[data-testid="stFormSubmitButton"] button { min-height: 2.8rem; }

/* ---------- surfaces: charts, tables, metrics, expanders, alerts ---------- */
[data-testid="stPlotlyChart"] {
  background: linear-gradient(var(--glass), var(--glass)) padding-box,
              linear-gradient(160deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02)) border-box;
  border: 1px solid transparent; border-radius: var(--r-md); padding: 0.6rem 0.4rem 0.2rem 0.4rem;
  box-shadow: var(--shadow); overflow: hidden;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--line2); border-radius: var(--r-md); overflow: hidden; box-shadow: var(--shadow);
}
[data-testid="stDataFrame"] > div { border: none !important; }
[data-testid="stElementToolbar"] { background: var(--surface2) !important; border: 1px solid var(--line2); border-radius: 8px; }
[data-testid="stMetric"] {
  background: linear-gradient(var(--glass), var(--glass)) padding-box,
              linear-gradient(160deg, rgba(255,255,255,0.10), rgba(255,255,255,0.02)) border-box;
  border: 1px solid transparent; border-radius: var(--r-md); padding: 1rem 1.2rem; box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] { white-space: normal !important; overflow: visible !important; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: 0.68rem !important; font-weight: 600 !important;
  text-transform: uppercase; letter-spacing: 0.08em; white-space: normal !important; }
[data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -0.03em; font-size: 1.9rem; color: var(--ink); }
[data-testid="stExpander"] details {
  background: var(--surface); border: 1px solid var(--line) !important; border-radius: var(--r-md) !important;
}
[data-testid="stExpander"] summary { padding: 0.8rem 1rem; }
[data-testid="stExpander"] summary p { color: var(--ink2) !important; font-weight: 600; font-size: 0.88rem; }
[data-testid="stExpander"] summary:hover p { color: var(--ink) !important; }
[data-testid="stExpander"] summary svg { color: var(--muted); }
[data-testid="stExpander"] details[open] {
  border-color: rgba(45,212,191,0.28) !important;
  box-shadow: 0 12px 30px -20px rgba(45,212,191,0.45);
}
[data-testid="stAlert"], [data-testid="stAlertContainer"] {
  background: var(--surface) !important; border: 1px solid var(--line2) !important; border-radius: var(--r-md) !important;
  color: var(--ink2) !important;
}
[data-testid="stAlert"] p, [data-testid="stAlertContainer"] p { color: var(--ink2) !important; }
[data-testid="stDialog"] > div {
  background:
    radial-gradient(440px 220px at 0% 0%, rgba(45,212,191,0.10), transparent 70%),
    var(--surface) !important;
  border: 1px solid var(--line2); border-radius: var(--r-lg);
  box-shadow: 0 30px 90px -30px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.03) inset;
}
[data-testid="stDialog"] h2 { letter-spacing: -0.025em; }
[data-testid="stDialog"] [data-testid="stMarkdownContainer"] p { color: var(--ink2); }
[data-testid="stDialog"] [data-testid="stHorizontalBlock"] { gap: 0.7rem; }

/* ---------- demonstration banner — loud, striped, the only amber on screen -- */
.demo-banner {
  display: flex; align-items: center; gap: 0.9rem; border-radius: 12px; margin: 0 0 1.4rem 0; overflow: hidden;
  border: 1px solid rgba(245,183,49,0.45); background: rgba(245,183,49,0.07);
  box-shadow: 0 0 0 1px rgba(245,183,49,0.08) inset, 0 12px 30px -18px rgba(245,183,49,0.6);
}
.demo-stripe {
  align-self: stretch; width: 44px; flex: none;
  background: repeating-linear-gradient(135deg, $AMBER 0 9px, #17130A 9px 18px);
  background-size: 200px 100%; animation: shimmer 6s linear infinite;
}
.demo-body { display: flex; align-items: center; gap: 0.8rem; padding: 0.55rem 1rem 0.55rem 0.3rem; flex-wrap: wrap; }
.demo-tag { background: $AMBER; color: #17130A; font-weight: 800; font-size: 0.68rem; letter-spacing: 0.14em;
            padding: 0.25rem 0.6rem; border-radius: 6px; text-transform: uppercase; }
.demo-text { color: #F7D68A; font-weight: 600; font-size: 0.86rem; letter-spacing: 0.01em; }
.demo-text b { color: #FFE9B3; font-weight: 800; }

/* ---------- page header ---------------------------------------------------- */
.page-head { display: flex; flex-direction: column; gap: 0.35rem; margin-bottom: 1.6rem; }
.eyebrow { display: inline-flex; align-items: center; gap: 0.5rem; color: var(--accent); font-size: 0.68rem;
           font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; }
.eyebrow::before { content: ""; width: 18px; height: 2px; background: var(--accent); border-radius: 2px; }
.subtitle { color: var(--muted); font-size: 0.98rem; margin-top: 0.15rem; max-width: 70ch; }
.page-head::after {
  content: ""; display: block; width: 76px; height: 2px; margin-top: 0.8rem;
  background: linear-gradient(90deg, var(--accent), transparent); border-radius: 2px;
}

/* ---------- section headers ------------------------------------------------ */
.section { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
           margin: 1.6rem 0 0.7rem 0; padding-bottom: 0.55rem; border-bottom: 1px solid var(--line); }
.section-title { font-weight: 700; font-size: 1.05rem; letter-spacing: -0.015em; color: var(--ink); display: flex; align-items: center; gap: 0.55rem; }
.section-title::before { content: ""; width: 6px; height: 6px; border-radius: 2px; background: var(--accent);
                         box-shadow: 0 0 10px rgba(45,212,191,0.7); transform: rotate(45deg); }
.section-cap { color: var(--muted); font-size: 0.8rem; }

/* ---------- KPI cards ------------------------------------------------------ */
.card, .kpi {
  position: relative; overflow: hidden;
  background: linear-gradient(var(--glass), var(--glass)) padding-box,
              linear-gradient(160deg, rgba(255,255,255,0.12), rgba(255,255,255,0.02)) border-box;
  border: 1px solid transparent; border-radius: var(--r-md); padding: 1.15rem 1.25rem 1.05rem 1.25rem;
  box-shadow: var(--shadow); min-height: 118px; transition: transform .2s ease, box-shadow .2s ease;
}
.card:hover, .kpi:hover { transform: translateY(-2px); box-shadow: 0 1px 0 rgba(255,255,255,0.05) inset, 0 18px 40px -16px rgba(0,0,0,0.8); }
.card::after, .kpi::after {
  content: ""; position: absolute; right: -40px; top: -40px; width: 120px; height: 120px; border-radius: 50%;
  background: radial-gradient(closest-side, rgba(255,255,255,0.05), transparent); pointer-events: none;
}
.kpi-up::after { background: radial-gradient(closest-side, rgba(45,212,191,0.16), transparent); }
.kpi-down::after { background: radial-gradient(closest-side, rgba(212,106,18,0.16), transparent); }
.card .label, .kpi-label { color: var(--muted); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
                          text-transform: uppercase; margin-bottom: 0.55rem; display: flex; align-items: center; gap: 0.45rem; }
.kpi-up .kpi-label::before, .kpi-down .kpi-label::before { content: ""; width: 6px; height: 6px; border-radius: 50%; }
.kpi-up .kpi-label::before { background: var(--up); box-shadow: 0 0 8px rgba(15,168,144,0.9); }
.kpi-down .kpi-label::before { background: var(--down); box-shadow: 0 0 8px rgba(212,106,18,0.9); }
.card .value, .kpi-value { font-size: clamp(1.4rem, 1.15rem + 0.9vw, 1.85rem); font-weight: 800; letter-spacing: -0.03em;
                           line-height: 1.05; color: var(--ink); }
.card .sub, .kpi-sub { color: var(--muted); font-size: 0.78rem; margin-top: 0.45rem; line-height: 1.4; }
.up, .kpi-up .kpi-value { color: var(--up) !important; }
.down, .kpi-down .kpi-value { color: var(--down) !important; }

/* pill labels used above tables */
.pill { display: inline-flex; align-items: center; gap: 0.45rem; padding: 0.22rem 0.7rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
        background: var(--surface2); border: 1px solid var(--line2); color: var(--ink2); margin-bottom: 0.5rem; }
.pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--muted); }
.pill-up { color: var(--up); border-color: rgba(15,168,144,0.35); background: rgba(15,168,144,0.08); }
.pill-up::before { background: var(--up); }
.pill-down { color: var(--down); border-color: rgba(212,106,18,0.35); background: rgba(212,106,18,0.08); }
.pill-down::before { background: var(--down); }
.pill-warn { color: #F7D68A; border-color: rgba(245,183,49,0.35); background: rgba(245,183,49,0.08); }
.pill-warn::before { background: var(--amber); }

/* ---------- narrative & footer -------------------------------------------- */
.narrative {
  position: relative; background: linear-gradient(135deg, rgba(45,212,191,0.07), rgba(94,108,235,0.05)), var(--surface);
  border: 1px solid var(--line2); border-radius: var(--r-md); padding: 1.3rem 1.5rem 1.3rem 1.7rem;
  line-height: 1.75; font-size: 1.02rem; color: var(--ink);
}
.narrative::before { content: ""; position: absolute; left: 0; top: 14px; bottom: 14px; width: 3px; border-radius: 3px;
                     background: linear-gradient(180deg, var(--accent), var(--indigo)); }
.disclaimer { color: var(--muted); font-size: 0.76rem; border-top: 1px solid var(--line); padding-top: 1rem;
              margin-top: 2.5rem; line-height: 1.55; display: flex; gap: 0.6rem; align-items: flex-start; }
.disclaimer::before { content: "◆"; color: var(--line2); font-size: 0.6rem; margin-top: 0.25rem; }

/* ---------- gate ----------------------------------------------------------- */
.st-key-login-shell {
  background:
    radial-gradient(420px 220px at 50% -12%, rgba(45,212,191,0.14), transparent 72%),
    linear-gradient(var(--glass), var(--glass)) padding-box,
    linear-gradient(160deg, rgba(45,212,191,0.42), rgba(255,255,255,0.05) 42%, rgba(94,108,235,0.34)) border-box;
  width: min(100%, 560px) !important; max-width: 560px !important; flex: 0 1 560px !important;
  border: 1px solid transparent; border-radius: 24px; padding: 2.75rem 2.5rem 2.2rem 2.5rem;
  margin: 0 auto;
  box-shadow: 0 30px 80px -30px rgba(0,0,0,0.9), 0 0 0 1px rgba(255,255,255,0.03) inset,
              0 0 60px -36px rgba(45,212,191,0.55);
  backdrop-filter: blur(18px);
  margin-left: auto !important; margin-right: auto !important;
}
[data-testid="stVerticalBlock"].st-key-login-shell { top: 0; }
.gate { text-align: center; margin: 0 0 1.7rem 0; }
.gate-mark {
  width: 72px; height: 72px; margin: 0 auto 1.2rem auto; border-radius: 20px; display: grid; place-items: center;
  background: var(--surface2); border: 1px solid var(--line2); padding: 0.55rem;
  box-shadow: 0 18px 40px -14px rgba(0,0,0,0.45);
}
.gate-mark svg { width: 100%; height: 100%; display: block; }
.gate h1 { font-size: 1.85rem !important; margin: 0.2rem 0 0.65rem 0 !important; }
.gate p { color: var(--ink2); font-size: 0.94rem; line-height: 1.55; max-width: 32ch; margin: 0 auto; }
.gate-kicker { color: var(--accent); font-size: 0.64rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.7rem; }
.st-key-login-shell [data-testid="stForm"] { margin-top: 0.4rem; }
.st-key-login-shell [data-testid="InputInstructions"] { display: none !important; }
.st-key-login-shell .gate-field-label {
  color: var(--muted); font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; text-align: left; margin: 0 0 0.45rem 0.15rem;
}
.st-key-login-shell [data-testid="stTextInput"] input {
  min-height: 3rem; text-align: left; letter-spacing: 0.08em; font-size: 1rem;
}
.st-key-login-shell [data-testid="stTextInputRootElement"]:focus-within {
  background: #202A3A !important;
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(45,212,191,0.16) !important;
}
.st-key-login-shell [data-testid="stTextInput"] [data-testid="InputInstructions"],
.st-key-login-shell [data-testid="stTextInput"] button,
.st-key-login-shell [data-testid="stTextInputRootElement"] button {
  border-left: 0 !important; box-shadow: none !important;
}
.st-key-login-shell [data-testid="stFormSubmitButton"] button {
  min-height: 3rem; margin-top: 0.35rem; border-radius: 12px !important;
}
.st-key-login-shell [data-testid="stCaptionContainer"] {
  color: var(--muted) !important; font-size: 0.72rem !important; line-height: 1.5;
  text-align: center; border-top: 1px solid var(--line); margin-top: 1.35rem; padding-top: 1rem;
}

/* ---------- loading / empty / error states -------------------------------- */
.state { display: flex; gap: 1rem; align-items: flex-start; padding: 1.3rem 1.4rem; border-radius: var(--r-md);
         background: var(--surface); border: 1px dashed var(--line2); margin: 0.5rem 0 1rem 0; }
.state-icon { width: 40px; height: 40px; border-radius: 12px; display: grid; place-items: center; flex: none;
              background: rgba(45,212,191,0.10); color: var(--accent); font-size: 1.1rem; font-weight: 800; }
.state-warn { border-style: solid; border-color: rgba(245,183,49,0.35); }
.state-warn .state-icon { background: rgba(245,183,49,0.12); color: #F7D68A; }
.state-title { font-weight: 700; color: var(--ink); font-size: 1rem; margin-bottom: 0.3rem; }
.state-body { color: var(--ink2); font-size: 0.92rem; line-height: 1.55; max-width: 64ch; }
.waking { display: flex; gap: 1rem; align-items: center; padding: 1.2rem 1.4rem; border-radius: var(--r-md);
          background: var(--surface); border: 1px solid var(--line2); margin: 10vh auto 0 auto; max-width: 620px;
          box-shadow: var(--shadow); }
.spinner { width: 26px; height: 26px; border-radius: 50%; border: 3px solid rgba(45,212,191,0.18);
           border-top-color: var(--accent); animation: spin 0.9s linear infinite; flex: none; }
@keyframes spin { to { transform: rotate(360deg); } }

/* ---------- chart captions, performance line, mobile card lists ------------ */
.figcap { color: var(--muted); font-size: 0.8rem; line-height: 1.5; margin: 0.35rem 0.2rem 1rem 0.2rem; }
.figcap::before { content: "About this chart · "; color: var(--ink2); font-weight: 600; }
.perf { color: var(--muted); font-size: 0.72rem; margin-top: 0.45rem; font-family: var(--mono) !important; }
.mcard { border: 1px solid var(--line); border-radius: 12px; padding: 0.7rem 0.85rem; margin-bottom: 0.5rem;
         background: var(--surface); }
.mcard-t { font-weight: 600; color: var(--ink); font-size: 0.92rem; line-height: 1.35; }
.mcard-s { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }
.mcard-f { display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; margin-top: 0.45rem; font-size: 0.84rem; color: var(--ink2); }
.mcard-f b { color: var(--muted); font-weight: 600; font-size: 0.72rem; text-transform: uppercase;
             letter-spacing: 0.06em; margin-right: 0.3rem; }
.mcard-f a { color: var(--accent); font-weight: 600; }
.mcard-note { color: var(--muted); font-size: 0.78rem; margin: 0.2rem 0 0.8rem 0; }
.eyebrow-row { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; }
.q-link { display: none; }
[class*="st-key-mob-"] { display: none; }

/* ---------- phones (≤ 640px) ---------------------------------------------- */
@media (max-width: 640px) {
  .block-container { padding-top: 4rem; padding-left: 1rem; padding-right: 1rem; padding-bottom: 3rem; }
  h1 { font-size: 1.6rem !important; }
  .subtitle { font-size: 0.9rem; }
  .page-head { margin-bottom: 1rem; }
  /* the questionnaire link lives in the (hidden) sidebar on phones — surface it in the header */
  .q-link { display: inline-flex; align-items: center; gap: 0.3rem; padding: 0.3rem 0.7rem; border-radius: 999px;
            border: 1px solid rgba(45,212,191,0.45); background: rgba(45,212,191,0.10); color: var(--accent) !important;
            font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-decoration: none !important; white-space: nowrap; }
  .demo-stripe { width: 26px; } .demo-body { gap: 0.5rem; padding-right: 0.7rem; } .demo-text { font-size: 0.78rem; }
  /* KPI rows: two per line instead of four stacked cards */
  [data-testid="stHorizontalBlock"]:has(.kpi) { flex-direction: row !important; flex-wrap: wrap !important; gap: 0.6rem !important; }
  [data-testid="stHorizontalBlock"]:has(.kpi) > [data-testid="stColumn"] {
    flex: 1 1 calc(50% - 0.4rem) !important; min-width: calc(50% - 0.4rem) !important; width: auto !important; max-width: 100%;
  }
  .kpi { min-height: 96px; padding: 0.8rem 0.9rem 0.75rem 0.9rem; }
  .kpi-value { font-size: 1.3rem; }
  .kpi-sub { font-size: 0.74rem; }
  .section { margin-top: 1.1rem; gap: 0.2rem 0.6rem; }
  .section-title { font-size: 0.98rem; }
  .page-head { margin-bottom: 1.25rem; }
  .page-head::after { margin-top: 0.55rem; }
  [data-testid="stPlotlyChart"] { padding: 0.3rem 0 0 0; }
  .narrative { font-size: 0.95rem; padding: 1rem 1rem 1rem 1.2rem; }
  .state { flex-direction: column; gap: 0.6rem; padding: 1rem 1.1rem; }
  .waking { margin-top: 2vh; }
  .st-key-login-shell {
    width: calc(100% - 1rem) !important; max-width: none !important; flex-basis: calc(100% - 1rem) !important;
    padding: 1.6rem 1.2rem 1.2rem 1.2rem; margin-top: 4vh;
  }
  /* data tables become card lists; the grid needs horizontal scrolling on a phone */
  [class*="st-key-desk-"] { display: none; }
  [class*="st-key-mob-"] { display: block; }
}

/* ---------- focus visibility ----------------------------------------------- */
button:focus-visible, input:focus-visible, a:focus-visible { outline: 2px solid var(--accent) !important; outline-offset: 2px; }
</style>
""").substitute(INK=INK, INK2=INK2, MUTED=MUTED, BG0=BG0, BG1=BG1, SURFACE=SURFACE, SURFACE2=SURFACE2,
                LINE=LINE, LINE2=LINE2, UP=UP, DOWN=DOWN, ACCENT=ACCENT, INDIGO=INDIGO, AMBER=AMBER)


def inject_css() -> None:
    theme = st.session_state.get("theme", "dark")
    st.markdown(f"<div class='theme-marker theme-{theme}'></div>", unsafe_allow_html=True)
    st.markdown(_CSS, unsafe_allow_html=True)


def nav_active_css(url_path: str) -> None:
    """Highlight the current page in the sidebar nav (pure CSS, keyed on the link href)."""
    suffix = f"/{url_path}" if url_path else "/"
    link = f'[data-testid="stSidebarNavLink"][href$="{suffix}"]'
    st.markdown(
        f"<style>{link}{{background:linear-gradient(90deg,rgba(45,212,191,0.14),rgba(45,212,191,0.04)) !important;"
        f"border-color:rgba(45,212,191,0.18);box-shadow:0 0 0 1px rgba(45,212,191,0.06) inset;}}"
        f"{link}::before{{content:'';position:absolute;left:-0.6rem;top:22%;height:56%;width:3px;"
        f"background:var(--accent);border-radius:0 3px 3px 0;box-shadow:0 0 12px rgba(45,212,191,0.6);}}"
        f"{link} span{{color:#fff !important;font-weight:600;}}"
        f"{link} [data-testid=\"stIconMaterial\"]{{color:var(--accent) !important;}}</style>",
        unsafe_allow_html=True,
    )


def chrome_reset() -> None:
    """Called at the start of every run so error panels know whether the banner/disclaimer are already on screen."""
    st.session_state["_chrome"] = {"banner": False, "footer": False}


def chrome_drawn(part: str) -> bool:
    return bool(st.session_state.get("_chrome", {}).get(part))


def demo_banner() -> None:
    st.session_state.setdefault("_chrome", {})["banner"] = True
    if is_mock_mode():
        st.markdown(
            '<div class="demo-banner"><div class="demo-stripe"></div>'
            '<div class="demo-body"><span class="demo-tag">Demonstration data</span>'
            '<span class="demo-text"><b>Not system output</b> — every number on this page is synthetic</span>'
            '</div></div>',
            unsafe_allow_html=True,
        )


def disclaimer() -> None:
    st.markdown(f'<div class="disclaimer"><span>{tr(DISCLAIMER)}</span></div>', unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    demo_banner()
    # `.q-link` is hidden on desktop (the sidebar carries the questionnaire button) and shown on
    # phones, where the sidebar is collapsed, so the link stays visible without scrolling.
    st.markdown(
        f"<div class='page-head'><div class='eyebrow-row'><span class='eyebrow'>{APP_TITLE}</span>"
        f"<a class='q-link' href='{QUESTIONNAIRE_URL}' target='_blank' rel='noopener'>Questionnaire ↗</a></div>"
        f"<h1>{tr(title)}</h1><div class='subtitle'>{subtitle}</div></div>",
        unsafe_allow_html=True,
    )
    guide(title)


def chart_caption(text: str) -> None:
    """Plain-language description of a chart, for screen readers and for anyone who
    finds the chart itself hard to read. Rendered visibly beneath the figure."""
    with st.expander(tr("About this chart")):
        st.markdown(f"<p class='figcap'>{_html.escape(text)}</p>", unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    """Styled section heading with an optional right-aligned caption."""
    cap = f"<span class='section-cap'>{caption}</span>" if caption else ""
    st.markdown(f"<div class='section'><span class='section-title'>{tr(title)}</span>{cap}</div>",
                unsafe_allow_html=True)


def pill(text: str, tone: str = "") -> None:
    """Small uppercase label (tone: 'up' | 'down' | 'warn' | '')."""
    cls = f" pill-{tone}" if tone else ""
    st.markdown(f"<span class='pill{cls}'>{text}</span>", unsafe_allow_html=True)


def footer() -> None:
    st.session_state.setdefault("_chrome", {})["footer"] = True
    disclaimer()
    seconds = end_page()  # closes the page render timer started in app.py and logs it
    if seconds is not None:
        st.markdown(f"<div class='perf'>Page built in {fmt_seconds(seconds)} · target under "
                    f"{TARGETS_SECONDS['page']:.0f} s</div>", unsafe_allow_html=True)


def mobile_cards(df: pd.DataFrame, *, title: Callable[[pd.Series], str],
                 fields: Sequence[tuple[str, Callable[[pd.Series], str]]],
                 subtitle: Callable[[pd.Series], str] | None = None,
                 link: Callable[[pd.Series], tuple[str, str]] | None = None,
                 limit: int | None = None) -> None:
    """Render a DataFrame as a vertical list of cards (phone-friendly twin of st.dataframe)."""
    e = _html.escape
    rows = df if limit is None else df.head(limit)
    parts = []
    for _, row in rows.iterrows():
        sub = f"<div class='mcard-s'>{e(subtitle(row))}</div>" if subtitle else ""
        fld = "".join(f"<span><b>{e(label)}</b>{e(fmt(row))}</span>" for label, fmt in fields)
        if link:
            label, url = link(row)
            fld += f"<span><a href='{e(url)}' target='_blank' rel='noopener'>{e(label)} ↗</a></span>"
        parts.append(f"<div class='mcard'><div class='mcard-t'>{e(title(row))}</div>{sub}"
                     f"<div class='mcard-f'>{fld}</div></div>")
    if limit is not None and len(df) > limit:
        parts.append(f"<div class='mcard-note'>Showing the first {limit} of {len(df)}. "
                     f"Open the page on a larger screen for the full table.</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def responsive_table(key: str, df: pd.DataFrame, *, column_config: dict,
                     title: Callable[[pd.Series], str],
                     fields: Sequence[tuple[str, Callable[[pd.Series], str]]],
                     subtitle: Callable[[pd.Series], str] | None = None,
                     link: Callable[[pd.Series], tuple[str, str]] | None = None,
                     columns: list[str] | None = None, height: int | None = None,
                     limit: int | None = None) -> None:
    """A data grid on desktop and a card list on phones (switched purely in CSS via the
    `st-key-desk-*` / `st-key-mob-*` container classes)."""
    with st.container(key=f"desk-{key}"):
        st.dataframe(df[columns] if columns else df, hide_index=True, use_container_width=True,
                     column_config=column_config, height=height)
    with st.container(key=f"mob-{key}"):
        mobile_cards(df, title=title, subtitle=subtitle, fields=fields, link=link, limit=limit)


# --------------------------------------------------------------------------- #
# Sidebar — shared selectors persisted in session state
# --------------------------------------------------------------------------- #
@st.dialog("Sign out of the workspace")
def _confirm_sign_out() -> None:
    st.write("Are you sure you want to sign out? Your current stock, date, and horizon selections will be cleared.")
    cancel, confirm = st.columns(2)
    with cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with confirm:
        if st.button("Sign out", type="primary", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def _pick(label, options, key, **kw):
    """Segmented control that cannot be left empty: clicking the active option again (deselect) keeps the
    previous choice. The widget has its own key so the stored value is never written after it renders."""
    ss = st.session_state
    wkey = f"_{key}_w"
    ss[wkey] = ss[key]

    def _keep():
        if ss[wkey] is not None:
            ss[key] = ss[wkey]

    st.segmented_control(label, options=options, key=wkey, on_change=_keep, **kw)


def sidebar_controls() -> None:
    repo = get_repository()
    status = repo.get_data_status()
    stocks = repo.list_stocks()

    ss = st.session_state
    ss.setdefault("ticker", stocks.ticker.iloc[0])
    ss.setdefault("as_of", status["as_of"])
    ss.setdefault("horizon", "t+1")
    ss.setdefault("lang", "en")
    if ss.pop("_reset_as_of", False):  # requested by an empty-state action on the previous run
        ss["as_of"] = status["as_of"]

    with st.sidebar:
        st.markdown(
            f"<div class='brand'><div class='brand-mark'>{logo_svg(st.session_state.get('theme', 'dark'))}</div>"
            f"<div><div class='brand-name'>{APP_TITLE}</div>"
            f"<div class='brand-tag'>Explainable ML · S&amp;P SL20 constituents</div></div></div>",
            unsafe_allow_html=True,
        )
        _pick(tr("Language"), list(LANGS), "lang", format_func=LANGS.get)
        st.markdown(f"<div class='sb-label'>{tr('Selection')}</div>", unsafe_allow_html=True)

        labels = {r.ticker: f"{r.ticker} — {r.name}" for r in stocks.itertuples()}
        st.selectbox(tr("Stock"), options=list(labels), format_func=labels.get, key="ticker")
        st.date_input(tr("As-of date"), key="as_of",
                      min_value=status.get("first_date", date(2022, 1, 3)), max_value=status["as_of"])
        with st.container(key="horizon-control"):
            st.selectbox(
                tr("Prediction horizon"),
                options=["t+1", "t+5"],
                format_func=lambda value: (
                    tr("Next trading day (t+1)") if value == "t+1"
                    else tr("Five trading days (t+5)")
                ),
                key="horizon",
                help="Choose how far ahead the prediction is made.",
            )

        st.divider()
        st.link_button(tr("Open evaluation questionnaire"), QUESTIONNAIRE_URL, use_container_width=True)
        st.markdown(
            "<div class='status'>"
            f"<div class='status-row'><span class='k'>{tr('Data as of')}</span><span class='v'>{status['as_of']:%d %b %Y}</span></div>"
            f"<div class='status-row'><span class='k'>{tr('Source')}</span><span class='v live'>{status['source']}</span></div>"
            f"<div class='status-row'><span class='k'>{tr('Model')}</span><span class='v'>{status['model_version']}</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if st.button(tr("Sign out"), use_container_width=True):
            _confirm_sign_out()


def selection() -> tuple[str, date, str]:
    ss = st.session_state
    return ss["ticker"], ss["as_of"], ss["horizon"] or "t+1"


def window(as_of: date, days: int) -> tuple[date, date]:
    return as_of - timedelta(days=days), as_of


# --------------------------------------------------------------------------- #
# Small display helpers
# --------------------------------------------------------------------------- #
def card(label: str, value: str, sub: str = "", tone: str = "") -> None:
    cls = f" kpi-{tone}" if tone else ""
    st.markdown(
        f'<div class="kpi{cls}"><div class="kpi-label">{tr(label)}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def fmt_lkr(x: float) -> str:
    return f"Rs {x:,.0f}"
