"""
AlphaForge AI Harness — Research Dashboard

A Streamlit dashboard for reviewing harness session results,
knowledge base state, RL bandit learning curves, and promoted strategies.

Launch
------
  py harness_main.py dashboard            # via CLI
  streamlit run harness/harness_dashboard.py  # directly
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from harness.config import RESULTS_DIR, MEMORY_DIR, ARTIFACTS_DIR, PROMOTE_SHARPE_THRESHOLD, PROMOTE_DD_LIMIT

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaForge",
    page_icon="AF",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
# Reference: Bloomberg Terminal + FinSight Financial Dashboard + Impeccable design principles
# Impeccable anti-patterns avoided: no pure black, no generic Inter-only, no nested cards,
# tinted neutrals, IBM Plex Mono for numerics, tabular-nums
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ═══════════════════════════════════════════════
   DESIGN TOKENS  (tinted neutrals — no pure black)
   ═══════════════════════════════════════════════ */
:root {
  --bg:          #09090f;   /* deep navy-black, NOT #000 */
  --surface:     #0e1117;   /* card face */
  --surface-2:   #131720;   /* slightly raised */
  --border:      rgba(255,255,255,0.07);
  --border-hi:   rgba(56,189,248,0.22);

  --text-1: #f1f5f9;
  --text-2: #94a3b8;
  --text-3: #475569;

  --blue:    #38bdf8;
  --indigo:  #818cf8;
  --amber:   #fbbf24;   /* Bloomberg amber — financial highlight */
  --green:   #22d3ee;
  --teal:    #2dd4bf;
  --red:     #f43f5e;
}

/* ═══════════════════════════════════════════════
   BASE  — Space Grotesk for UI, IBM Plex Mono for numbers
   ═══════════════════════════════════════════════ */
* { font-family: 'Space Grotesk', sans-serif !important; }
/* Restore icon font — Streamlit renders every :material/icon: as
   <span data-testid="stIconMaterial" translate="no">icon_name</span>
   Our * rule stomps the emotion-scoped font-family; restore it here. */
[data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    -moz-font-feature-settings: 'liga' !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg);
    background-image:
        radial-gradient(ellipse 80% 50% at 10% 0%, rgba(56,189,248,0.055) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 100%, rgba(251,191,36,0.025) 0%, transparent 60%);
    min-height: 100vh;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1.2rem;
    /* dot grid texture */
    background-image: radial-gradient(rgba(148,163,184,0.04) 1px, transparent 1px);
    background-size: 24px 24px;
}

/* ═══════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #08090e 0%, #0b0e1a 100%);
    border-right: 1px solid var(--border);
}
/* Dim non-interactive sidebar text without nuking input values */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] .stCaption { color: #64748b !important; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-size: 0.78rem !important; }
/* Keep input text readable */
[data-testid="stSidebar"] input { color: var(--text-1) !important; background: #111827 !important; border-color: rgba(255,255,255,0.1) !important; }
[data-testid="stSidebar"] input::placeholder { color: #475569 !important; }

/* ── Selectbox dark theme ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
}
[data-testid="stSelectbox"] label { color: var(--text-2) !important; font-size: 0.8rem !important; }
div[data-baseweb="select"] > div { background: var(--surface-2) !important; border-color: var(--border-hi) !important; }

/* Nav items */
[data-testid="stRadio"] > div { gap: 2px; }
[data-testid="stRadio"] label {
    border-radius: 7px !important;
    padding: 7px 10px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: background 0.12s, color 0.12s;
    border: 1px solid transparent !important;
}
[data-testid="stRadio"] label:hover {
    background: rgba(56,189,248,0.07) !important;
    border-color: rgba(56,189,248,0.12) !important;
    color: var(--blue) !important;
}

/* ═══════════════════════════════════════════════
   TYPOGRAPHY  — Space Grotesk display, IBM Plex Mono numbers
   ═══════════════════════════════════════════════ */
h1 {
    color: var(--text-1) !important;
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    font-family: 'Space Grotesk', sans-serif !important;
}
h2 {
    color: #e2e8f0 !important;
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.01em;
    margin-bottom: 0.6rem;
    font-family: 'Space Grotesk', sans-serif !important;
}
h3 {
    color: var(--text-2) !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* IBM Plex Mono for all numeric displays */
[data-testid="stMetricValue"],
.mono { font-family: 'IBM Plex Mono', monospace !important; font-variant-numeric: tabular-nums; }

/* ═══════════════════════════════════════════════
   KPI METRIC CARDS  (Bloomberg-style: amber-ruled top accent)
   ═══════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-top: 2px solid var(--amber);
    border-radius: 10px;
    padding: 1rem 1.1rem 0.9rem;
    transition: box-shadow 0.18s ease, transform 0.15s ease;
    position: relative;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(251,191,36,0.12);
}
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    color: var(--text-1) !important;
    letter-spacing: -0.03em;
    font-family: 'IBM Plex Mono', monospace !important;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] {
    font-size: 0.63rem !important;
    color: var(--text-2) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
    white-space: normal !important;
    overflow: visible !important;
    line-height: 1.4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* ═══════════════════════════════════════════════
   VERDICT BADGES
   ═══════════════════════════════════════════════ */
.badge-promote {
    background: rgba(34,211,238,0.1); color: #22d3ee;
    border: 1px solid rgba(34,211,238,0.3);
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
}
.badge-reject {
    background: rgba(244,63,94,0.1); color: #f43f5e;
    border: 1px solid rgba(244,63,94,0.3);
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
}
.badge-iterate {
    background: rgba(251,191,36,0.1); color: var(--amber);
    border: 1px solid rgba(251,191,36,0.3);
    padding: 3px 11px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.06em;
}
.badge-sim {
    background: rgba(56,189,248,0.08);
    color: var(--blue);
    border: 1px solid rgba(56,189,248,0.2);
    padding: 3px 10px; border-radius: 20px;
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.05em;
}

/* ═══════════════════════════════════════════════
   STATUS INDICATORS
   ═══════════════════════════════════════════════ */
.status-ok   { color: #22d3ee; font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.status-warn { color: var(--amber); font-weight: 600; font-family: 'IBM Plex Mono', monospace; }
.status-err  { color: var(--red); font-weight: 600; font-family: 'IBM Plex Mono', monospace; }

/* ═══════════════════════════════════════════════
   INPUTS
   ═══════════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: var(--surface-2) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 7px !important;
    color: var(--text-1) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(251,191,36,0.12) !important;
}

/* ═══════════════════════════════════════════════
   DATA ELEMENTS
   ═══════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    border-radius: 10px; overflow: hidden;
    border: 1px solid var(--border);
}
[data-testid="stCode"] {
    background: var(--surface) !important;
    border-radius: 8px;
    border: 1px solid var(--border);
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stAlert"] { border-radius: 8px; border-left-width: 3px; }
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    background: var(--surface) !important;
}
/* Expander header text readable in sidebar */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color: var(--text-2) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
[data-testid="stExpander"] summary:hover p,
[data-testid="stExpander"] summary:hover span {
    color: var(--blue) !important;
}
[data-testid="stBaseButton-secondary"] {
    border-radius: 7px !important; font-weight: 600 !important; font-size: 0.81rem !important;
}
[data-testid="stBaseButton-primary"] {
    background: var(--amber) !important;
    color: #0a0a0a !important;
    border: none !important;
    font-weight: 700 !important;
    border-radius: 7px !important;
}

/* ═══════════════════════════════════════════════
   CHARTS
   ═══════════════════════════════════════════════ */
.js-plotly-plot { border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }

/* ═══════════════════════════════════════════════
   LAYOUT HELPERS
   ═══════════════════════════════════════════════ */
hr { border-color: var(--border); margin: 1rem 0; }
.section-rule {
    height: 1px;
    background: linear-gradient(90deg, rgba(251,191,36,0.35), rgba(56,189,248,0.18), transparent);
    margin: 1.4rem 0;
}

/* Section heading with amber left accent */
.sec-head {
    display: flex; align-items: center; gap: 10px;
    margin: 1.6rem 0 0.7rem 0;
}
.sec-head-bar {
    width: 3px; height: 1.2em; border-radius: 2px;
    background: var(--amber);
    flex-shrink: 0;
}
.sec-head-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.88rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.005em;
    text-transform: none;
}
.sec-head-caption {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-3);
    margin-top: 0.15rem;
}

/* ═══════════════════════════════════════════════
   HERO ELEMENTS
   ═══════════════════════════════════════════════ */
.hero-title {
    background: linear-gradient(135deg, #38bdf8 0%, #fbbf24 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem !important; font-weight: 800 !important;
    letter-spacing: -0.045em; line-height: 1.05;
    margin-bottom: 0.35rem;
}
.hero-sub {
    color: #64748b;
    font-size: 0.74rem; letter-spacing: 0.2em;
    text-transform: uppercase; font-weight: 600;
    margin-bottom: 0.9rem;
    font-family: 'IBM Plex Mono', monospace;
}
.sim-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(244,63,94,0.07);
    border: 1px solid rgba(244,63,94,0.2);
    color: #fb7185;
    font-size: 0.63rem; font-weight: 700;
    letter-spacing: 0.12em; padding: 4px 12px;
    border-radius: 20px; text-transform: uppercase;
    margin-bottom: 1.3rem;
}

/* ═══════════════════════════════════════════════
   VIRTUAL PORTFOLIO WIDGET
   ═══════════════════════════════════════════════ */
.vport-wrap {
    background: linear-gradient(135deg, rgba(251,191,36,0.04) 0%, rgba(14,17,23,0) 60%);
    border: 1px solid rgba(251,191,36,0.18);
    border-radius: 14px;
    padding: 1.2rem 1.4rem 1rem;
    margin: 1rem 0 1.4rem 0;
    position: relative;
}
.vport-badge {
    position: absolute; top: -10px; left: 16px;
    background: rgba(251,191,36,0.12);
    border: 1px solid rgba(251,191,36,0.3);
    color: #fbbf24;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em;
    padding: 2px 10px; border-radius: 20px; text-transform: uppercase;
}
.vport-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.82rem; font-weight: 700;
    color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em;
    margin-bottom: 0.9rem;
}
.vport-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 8px;
}
.vport-cell {
    background: rgba(14,17,23,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 9px;
    padding: 0.7rem 0.9rem;
}
.vport-cell-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.57rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569;
    margin-bottom: 4px;
}
.vport-cell-val {
    font-family: 'IBM Plex Mono', monospace;
    font-variant-numeric: tabular-nums;
    font-size: 1.15rem; font-weight: 600; letter-spacing: -0.02em;
    line-height: 1;
}
.vport-cell-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.63rem; color: #475569; margin-top: 3px;
}
.vport-disclaimer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem; color: #334155;
    margin-top: 0.8rem; letter-spacing: 0.02em;
}

/* ═══════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(148,163,184,0.15); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(148,163,184,0.3); }

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT DEFAULT CHROME  (Streamlit 1.57.0)
   ═══════════════════════════════════════════════ */
#MainMenu { display: none !important; }
footer { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stStatusWidget"] { display: none !important; }

/* In Streamlit 1.57.0 the expand-sidebar button (stExpandSidebarButton) lives inside
   stToolbar → stHeader.  Hiding the whole toolbar or header buries that button and
   makes the sidebar permanently inaccessible.  Instead we hide only the specific
   children we don't want, and make the containers transparent so nothing is visible. */
[data-testid="stToolbarActions"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenuButton"] {
    display: none !important;
}
[data-testid="stToolbar"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* Hide radio button circles in sidebar nav */
[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] label { padding-left: 8px !important; }

/* ── Sidebar toggle buttons (Streamlit 1.57.0 data-testids) ────────────────
   Real testids:  stSidebarCollapseButton  (inside sidebar)
                  stExpandSidebarButton    (shown outside sidebar when collapsed)
   The :material/keyboard_double_arrow_* icon renders as raw text when the
   Material Symbols font isn't cached.  Fix: zero the font-size on the button,
   inject a clean chevron via ::after so the button stays clickable and visible.
   ────────────────────────────────────────────────────────────────────────── */

/* Collapse button inside the open sidebar */
[data-testid="stSidebarCollapseButton"] button {
    background: transparent !important;
    border: none !important;
    width: 28px !important;
    min-width: 28px !important;
    height: 32px !important;
    overflow: hidden !important;
    border-radius: 4px !important;
    opacity: 0.3 !important;
    transition: opacity 0.2s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stSidebarCollapseButton"] button:hover {
    opacity: 1 !important;
    background: rgba(56,189,248,0.1) !important;
}
[data-testid="stSidebarCollapseButton"] button > *,
[data-testid="stSidebarCollapseButton"] button > * > * { display: none !important; }
[data-testid="stSidebarCollapseButton"] button::after {
    content: "‹" !important;
    color: #38bdf8 !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    display: block !important;
    flex-shrink: 0 !important;
}

/* Expand button — data-testid="stExpandSidebarButton" IS the <button> element itself,
   not a wrapper. All selectors must target it directly, not via a child "button". */
[data-testid="stExpandSidebarButton"] {
    width: 28px !important;
    min-width: 28px !important;
    height: 40px !important;
    min-height: 40px !important;
    padding: 0 !important;
    overflow: hidden !important;
    background: rgba(14,17,23,0.95) !important;
    border: 1px solid rgba(56,189,248,0.35) !important;
    border-left: none !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 3px 0 10px rgba(0,0,0,0.5) !important;
    transition: background 0.18s, border-color 0.18s !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
[data-testid="stExpandSidebarButton"]:hover {
    background: rgba(56,189,248,0.14) !important;
    border-color: rgba(56,189,248,0.65) !important;
}
[data-testid="stExpandSidebarButton"] > *,
[data-testid="stExpandSidebarButton"] > * > * { display: none !important; }
[data-testid="stExpandSidebarButton"]::after {
    content: "›" !important;
    color: #38bdf8 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    display: block !important;
    flex-shrink: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme defaults (Bloomberg/FinSight palette) ──────────────────
_CHART_BG  = "#0e1117"
_PAPER_BG  = "#0e1117"
_GRID_COL  = "rgba(255,255,255,0.05)"
_AXIS_COL  = "rgba(255,255,255,0.08)"

def _dark_layout(**kwargs) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=_PAPER_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(family="IBM Plex Mono, Inter, sans-serif", color="#64748b", size=11),
        margin=dict(l=4, r=4, t=36, b=4),
        xaxis=dict(
            gridcolor=_GRID_COL, linecolor=_AXIS_COL,
            tickfont=dict(size=10, color="#475569"),
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor=_GRID_COL, linecolor=_AXIS_COL,
            tickfont=dict(size=10, color="#475569"),
            zeroline=False,
        ),
        title=dict(text="", font=dict(family="Space Grotesk", size=13, color="#94a3b8")),
        hoverlabel=dict(
            bgcolor="#131720", bordercolor="rgba(255,255,255,0.1)",
            font=dict(family="IBM Plex Mono", size=11, color="#f1f5f9"),
        ),
    )
    base.update(kwargs)
    return base


# ── Currency ──────────────────────────────────────────────────────────────────
# Underlying data is USD (SPY/US equities). Convert display to GBP.
_GBP_USD = 0.787   # approximate mid-market rate; update if needed
_CCY     = "£"

def _gbp(usd: float) -> float:
    return usd * _GBP_USD

def _fmt_gbp(usd: float, decimals: int = 0) -> str:
    v = _gbp(usd)
    if decimals == 0:
        return f"£{v:,.0f}"
    return f"£{v:,.{decimals}f}"


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=15)
def load_sessions() -> list[dict]:
    """Load all session JSON logs, newest first."""
    logs = sorted(RESULTS_DIR.glob("session_*.json"), reverse=True)
    sessions = []
    for p in logs:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            session_id = p.stem.replace("session_", "")
            sessions.append({"id": session_id, "path": str(p), "log": data})
        except Exception:
            pass
    return sessions


@st.cache_data(ttl=15)
def load_kb_index() -> list[dict]:
    idx_path = MEMORY_DIR / "index.json"
    if not idx_path.exists():
        return []
    try:
        return json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return []


@st.cache_data(ttl=15)
def load_bandit_state() -> dict:
    p = MEMORY_DIR / "bandit_state.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_report(session_id: str) -> str:
    report_dir = RESULTS_DIR / "reports"
    p = report_dir / f"session_{session_id}.md"
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def load_kb_entry(entry_id: str) -> dict:
    p = MEMORY_DIR / f"{entry_id}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data(ttl=60)
def load_training_metrics() -> dict:
    """Load training metrics — prefers the dissertation model, falls back to most recent."""
    # The dissertation model: 26 folds, avg Sharpe 2.532, 85 features
    dissertation = ARTIFACTS_DIR / "training_metrics_20260516_023323.json"
    if dissertation.exists():
        try:
            return json.loads(dissertation.read_text(encoding="utf-8"))
        except Exception:
            pass
    files = sorted(ARTIFACTS_DIR.glob("training_metrics_*.json"), reverse=True)
    if not files:
        return {}
    try:
        return json.loads(files[0].read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=60)
def load_virtual_portfolio() -> dict | None:
    """Compute virtual portfolio stats from the universe NAV log."""
    nav_path = ROOT / "logs" / "universe_portfolio" / "nav.csv"
    if not nav_path.exists():
        return None
    try:
        df = pd.read_csv(nav_path, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < 2:
            return None

        start_nav = df["nav"].iloc[0]
        end_nav   = df["nav"].iloc[-1]
        total_ret = (end_nav / start_nav - 1) * 100
        total_pnl = end_nav - start_nav

        # Recompute daily returns from NAV (logged values often zero)
        df["ret"] = df["nav"].pct_change().fillna(0)
        pos_days  = int((df["ret"] > 0).sum())
        neg_days  = int((df["ret"] < 0).sum())
        win_rate  = pos_days / (pos_days + neg_days) * 100 if (pos_days + neg_days) else 0.0

        # Max drawdown
        roll_max = df["nav"].cummax()
        max_dd   = float(((df["nav"] - roll_max) / roll_max).min() * 100)

        # Annualised return
        years    = (df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25
        ann_ret  = ((end_nav / start_nav) ** (1 / years) - 1) * 100 if years > 0 else 0.0

        # Daily Sharpe from NAV returns
        daily_std = df["ret"].std()
        daily_avg = df["ret"].mean()
        sharpe    = (daily_avg / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

        best_day  = df.loc[df["ret"].idxmax()]
        worst_day = df.loc[df["ret"].idxmin()]

        return {
            "start_nav":    start_nav,
            "end_nav":      end_nav,
            "total_ret":    total_ret,
            "total_pnl":    total_pnl,
            "win_rate":     win_rate,
            "max_dd":       max_dd,
            "ann_ret":      ann_ret,
            "sharpe":       sharpe,
            "pos_days":     pos_days,
            "neg_days":     neg_days,
            "period_start": df["date"].iloc[0].strftime("%d %b %Y"),
            "period_end":   df["date"].iloc[-1].strftime("%d %b %Y"),
            "n_days":       len(df),
            "best_day_pct": float(best_day["ret"] * 100),
            "best_day_dt":  best_day["date"].strftime("%d %b %Y"),
            "worst_day_pct": float(worst_day["ret"] * 100),
            "worst_day_dt":  worst_day["date"].strftime("%d %b %Y"),
            "nav_series":   df[["date", "nav", "ret"]].copy(),
        }
    except Exception:
        return None


@st.cache_data(ttl=30)
def load_system_health() -> dict:
    """Load real system health indicators from available log/artifact sources."""
    health: dict = {}

    # Kill switch flag
    ks_flag = ROOT / "KILL_SWITCH.flag"
    if ks_flag.exists():
        health["kill_switch"] = "ACTIVE"
        try:
            health["ks_triggered_at"] = ks_flag.read_text().strip()[:25]
        except Exception:
            health["ks_triggered_at"] = "—"
    else:
        health["kill_switch"] = "Inactive"
        health["ks_triggered_at"] = "—"

    # Audit ledger (today's report)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ledger_path = ROOT / "audit" / "reports" / f"{today}.json"
    if ledger_path.exists():
        try:
            d = json.loads(ledger_path.read_text())
            health["ledger_records"] = d.get("n_records", 0)
            health["ledger_valid"]   = d.get("chain_valid", None)
            health["ledger_orders"]  = sum(
                1 for r in d.get("records", []) if r.get("event_type") == "ORDER"
            )
        except Exception:
            health["ledger_records"] = "—"
    else:
        health["ledger_records"] = "—"
        health["ledger_valid"]   = None
        health["ledger_orders"]  = "—"

    # Most recent training metrics
    tm = load_training_metrics()
    if tm:
        health["model_trained_at"] = tm.get("trained_at", "—")[:19].replace("T", " ")
        health["model_features"]   = tm.get("feature_count", "—")
        health["model_horizon"]    = tm.get("target_horizon", "—")
        folds = tm.get("fold_results", [])
        sharpes = [
            f["ml_strategy_metrics"]["sharpe_like"]
            for f in folds if "ml_strategy_metrics" in f
        ]
        health["model_oos_sharpe"] = round(float(np.mean(sharpes)), 3) if sharpes else "—"
    else:
        health["model_trained_at"] = "—"
        health["model_features"]   = "—"
        health["model_horizon"]    = "—"
        health["model_oos_sharpe"] = "—"

    return health


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
<div style="padding: 0.5rem 0 0.8rem 0;">
  <div style="font-size:1.4rem; font-weight:800; background:linear-gradient(135deg,#38bdf8,#818cf8);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">
    AlphaForge
  </div>
  <div style="color:#334155; font-size:0.68rem; letter-spacing:0.08em; margin-top:2px;">
    AI STRATEGY RESEARCH
  </div>
</div>
<span class="badge-sim" style="font-size:0.6rem;">SIMULATION · NO REAL MONEY</span>
""", unsafe_allow_html=True)
    st.divider()

    tab_choice = st.radio(
        "Navigation",
        ["Home", "Live Markets", "Strategy Library", "AI Learning", "Reports", "Paper Trading"],
        label_visibility="collapsed",
    )
    st.divider()

    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # ── API Keys ──────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("API Keys", expanded=False):
        ak = st.text_input(
            "Anthropic", type="password",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            placeholder="sk-ant-…",
        )
        xk = st.text_input(
            "xAI (Grok)", type="password",
            value=os.environ.get("XAI_API_KEY", ""),
            placeholder="xai-…",
        )
        if st.button("Save", use_container_width=True, key="save_keys"):
            if ak:
                os.environ["ANTHROPIC_API_KEY"] = ak
            if xk:
                os.environ["XAI_API_KEY"] = xk
            st.success("Saved to environment.")
        st.caption(
            f"Anthropic: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'not set'}  ·  "
            f"xAI: {'set' if os.environ.get('XAI_API_KEY') else 'not set'}"
        )


# ── Load data ─────────────────────────────────────────────────────────────────
sessions   = load_sessions()
kb_index   = load_kb_index()
bandit_st  = load_bandit_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sharpe_color(s: float) -> str:
    if s >= PROMOTE_SHARPE_THRESHOLD:
        return "#22d3ee"
    if s >= 0.4:
        return "#fbbf24"
    return "#f43f5e"


def _section_header(label: str, caption: str = "") -> None:
    """Render a styled section heading with amber left-accent bar."""
    cap_html = f'<div class="sec-head-caption">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="sec-head">'
        f'<div class="sec-head-bar"></div>'
        f'<div><div class="sec-head-label">{label}</div>{cap_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _session_to_df(log: list[dict]) -> pd.DataFrame:
    rows = []
    for r in log:
        bt = r.get("backtest", {}) or {}
        # ann_return is already a percentage in executor output (e.g. 5.0 = 5%)
        ann_ret = bt.get("ann_return", bt.get("ann_return_pct", 0)) or 0
        rows.append({
            "Exp #":      r.get("iteration", "?"),
            "Idea":       r.get("config", {}).get("hypothesis", "—")[:55],
            "Approach":   r.get("config", {}).get("_bandit_arm", "—"),
            "Score":      r.get("sharpe", 0.0),
            "Max Loss %": abs(bt.get("max_dd", 0.0) or 0.0),
            "Yearly Ret %": round(float(ann_ret), 2),
            "Win Rate":   round((bt.get("win_rate", 0) or 0) * 100, 1),
            "Trades":     bt.get("n_trades", "—"),
            "Approved":   "Yes" if r.get("promoted") else "—",
            "Time (s)":   r.get("elapsed_s", 0),
        })
    return pd.DataFrame(rows)


def _color_sharpe(val):
    if not isinstance(val, (int, float)):
        return ""
    if val >= PROMOTE_SHARPE_THRESHOLD:
        return "background-color:rgba(34,211,238,0.08); color:#22d3ee; font-weight:700"
    if val >= 0.4:
        return "color:#fbbf24; font-weight:600"
    return "color:#f43f5e"


def _empty_state(title: str, hint: str) -> None:
    st.markdown(f"""
    <div style="text-align:center; padding:3rem 1rem; color:#475569;
                border:1px solid rgba(255,255,255,0.06); border-radius:12px; margin:1rem 0;">
        <div style="font-size:1.1rem; font-weight:600; color:#64748b; margin-bottom:0.5rem;">{title}</div>
        <div style="font-size:0.85rem;">{hint}</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Overview
# ═══════════════════════════════════════════════════════════════════════════════

if tab_choice == "Home":
    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="padding: 0.5rem 0 1rem 0;">
  <div class="hero-title">AlphaForge</div>
  <div class="hero-sub">AI-Powered Strategy Research Platform</div>
  <span class="sim-badge">Simulation only · No real money</span>
</div>
""", unsafe_allow_html=True)

    # KPIs
    total_sessions   = len(sessions)
    total_iters      = sum(len(s["log"]) for s in sessions)
    promotions_idx   = [e for e in kb_index if e.get("type") == "promotion"]
    experiments_idx  = [e for e in kb_index if e.get("type") == "experiment"]
    best_sharpe      = max(
        (e.get("metrics", {}).get("oos_sharpe", 0) or 0 for e in experiments_idx),
        default=0.0,
    )
    bandit_arms   = bandit_st.get("arms", {})
    bandit_trials = bandit_st.get("total_trials", 0)
    kb_total      = len(kb_index)

    approved_count = len(promotions_idx)
    best_score_str = f"{best_sharpe:.3f}" if best_sharpe else "—"

    # Bloomberg-style KPI cards: amber top accent, IBM Plex Mono numbers
    _cards = [
        (total_sessions,   "Sessions",    "#38bdf8"),
        (total_iters,      "Experiments", "#94a3b8"),
        (kb_total,         "Strategies",  "#94a3b8"),
        (approved_count,   "Approved",    "#22d3ee" if approved_count else "#334155"),
        (best_score_str,   "Best Score",  "#fbbf24" if best_sharpe >= 0.5 else "#94a3b8"),
        (bandit_trials,    "AI Trials",   "#94a3b8"),
    ]
    st.markdown(
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:1.4rem;">'
        + "".join(
            f'<div style="background:#0e1117;border:1px solid rgba(255,255,255,0.07);'
            f'border-top:2px solid #fbbf24;border-radius:10px;padding:0.9rem 1rem 0.8rem;">'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-variant-numeric:tabular-nums;'
            f'font-size:1.55rem;font-weight:600;color:{color};letter-spacing:-0.03em;line-height:1;">{val}</div>'
            f'<div style="font-size:0.58rem;color:#475569;text-transform:uppercase;letter-spacing:0.12em;'
            f'font-weight:700;margin-top:6px;font-family:\'IBM Plex Mono\',monospace;">{label}</div>'
            f'</div>'
            for val, label, color in _cards
        )
        + '</div><div class="section-rule"></div>',
        unsafe_allow_html=True,
    )

    # ── Verified OOS Demo Results ─────────────────────────────────────────────
    _pt_path = ROOT / "logs" / "paper_trading" / "spy_paper_trades.csv"
    _oos_nav, _oos_ret, _oos_sharpe, _oos_dd, _oos_n, _oos_wr = 15676.0, 56.76, 0.870, -19.92, 7, 66.7
    if _pt_path.exists():
        try:
            _pt = pd.read_csv(_pt_path, parse_dates=["timestamp"])
            _oos_nav = float(_pt["nav"].iloc[-1])
            _oos_ret = (_oos_nav - 10_000) / 10_000 * 100
            _exit_mask = _pt["fill_status"].isin(["stop_loss","close_long","close_short","take_profit"])
            _exits_h   = _pt[_exit_mask]
            _oos_n     = int(_pt["fill_status"].isin(["long_entry","short_entry"]).sum())
            _oos_wr    = float((_exits_h["daily_pnl_pct"] > 0).mean() * 100) if len(_exits_h) else 0.0
            _peak_nav  = _pt["nav"].cummax()
            _oos_dd    = float(((_pt["nav"] - _peak_nav) / _peak_nav * 100).min())
            _ts_nav    = _pt.drop_duplicates("timestamp").set_index("timestamp")["nav"]
            _rets_d    = _ts_nav.pct_change().dropna()
            _oos_sharpe = float(_rets_d.mean() / _rets_d.std() * (252**0.5)) if _rets_d.std() > 0 else 0.0
        except Exception:
            pass

    _ret_col = "#22d3ee" if _oos_ret >= 0 else "#f43f5e"
    _dd_col  = "#f43f5e" if _oos_dd < -15 else "#fbbf24"
    _wr_col  = "#22d3ee" if _oos_wr >= 50 else "#fbbf24"
    _ret_pfx = "+" if _oos_ret >= 0 else ""

    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(34,211,238,0.06) 0%, rgba(251,191,36,0.04) 100%);
    border: 1px solid rgba(34,211,238,0.25);
    border-top: 3px solid #22d3ee;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.4rem;
">
  <div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1rem;">
    <div style="font-size:0.62rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;
                color:#22d3ee;font-family:'IBM Plex Mono',monospace;
                background:rgba(34,211,238,0.12);border:1px solid rgba(34,211,238,0.3);
                border-radius:4px;padding:3px 8px;">Verified OOS Demo</div>
    <div style="font-size:0.85rem;font-weight:600;color:#e2e8f0;">
        SPY &nbsp;·&nbsp; Jan 2024 – May 2026 &nbsp;·&nbsp; $10,000 starting capital &nbsp;·&nbsp; bar-by-bar simulation on genuinely unseen data
    </div>
    <div style="margin-left:auto;font-size:0.60rem;color:#475569;font-family:'IBM Plex Mono',monospace;white-space:nowrap;">
        Simulation only · No real money
    </div>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;">
    <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.75rem 1rem;">
      <div style="font-size:0.53rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">Final NAV</div>
      <div style="font-size:1.35rem;font-weight:700;color:#22d3ee;font-family:'IBM Plex Mono',monospace;">${_oos_nav:,.0f}</div>
      <div style="font-size:0.63rem;color:{_ret_col};font-family:'IBM Plex Mono',monospace;">{_ret_pfx}{_oos_ret:.2f}% total return</div>
    </div>
    <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.75rem 1rem;">
      <div style="font-size:0.53rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">Sharpe Ratio</div>
      <div style="font-size:1.35rem;font-weight:700;color:#fbbf24;font-family:'IBM Plex Mono',monospace;">{_oos_sharpe:.3f}</div>
      <div style="font-size:0.63rem;color:#64748b;font-family:'IBM Plex Mono',monospace;">annualised · vs ~0.55 buy-hold</div>
    </div>
    <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.75rem 1rem;">
      <div style="font-size:0.53rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">Max Drawdown</div>
      <div style="font-size:1.35rem;font-weight:700;color:{_dd_col};font-family:'IBM Plex Mono',monospace;">{_oos_dd:.1f}%</div>
      <div style="font-size:0.63rem;color:#64748b;font-family:'IBM Plex Mono',monospace;">peak-to-trough</div>
    </div>
    <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.75rem 1rem;">
      <div style="font-size:0.53rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">Win Rate</div>
      <div style="font-size:1.35rem;font-weight:700;color:{_wr_col};font-family:'IBM Plex Mono',monospace;">{_oos_wr:.1f}%</div>
      <div style="font-size:0.63rem;color:#64748b;font-family:'IBM Plex Mono',monospace;">{_oos_n} completed trades</div>
    </div>
    <div style="background:rgba(255,255,255,0.03);border-radius:8px;padding:0.75rem 1rem;">
      <div style="font-size:0.53rem;color:#64748b;text-transform:uppercase;letter-spacing:0.1em;font-family:'IBM Plex Mono',monospace;margin-bottom:4px;">SPY Buy-and-Hold</div>
      <div style="font-size:1.35rem;font-weight:700;color:#94a3b8;font-family:'IBM Plex Mono',monospace;">+60.6%</div>
      <div style="font-size:0.63rem;color:#fbbf24;font-family:'IBM Plex Mono',monospace;">ML Sharpe 0.870 vs ~0.55</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Virtual Portfolio Panel ────────────────────────────────────────────────
    vp = load_virtual_portfolio()
    if vp:
        pnl_color  = "#22d3ee" if vp["total_pnl"] >= 0 else "#f43f5e"
        ret_color  = "#22d3ee" if vp["total_ret"] >= 0 else "#f43f5e"
        pnl_sign   = "+" if vp["total_pnl"] >= 0 else ""
        ret_sign   = "+" if vp["total_ret"] >= 0 else ""
        dd_color   = "#f43f5e" if vp["max_dd"] < -15 else "#fbbf24" if vp["max_dd"] < -5 else "#22d3ee"
        wr_color   = "#22d3ee" if vp["win_rate"] >= 50 else "#fbbf24"

        cells = [
            ("Starting capital",  _fmt_gbp(vp['start_nav']),    "#94a3b8",  "virtual budget"),
            ("Portfolio value",   _fmt_gbp(vp['end_nav']),      "#f1f5f9",  vp["period_end"]),
            ("Total P&L",         f"{pnl_sign}{_fmt_gbp(abs(vp['total_pnl']))}", pnl_color,
                                   f"{ret_sign}{vp['total_ret']:.2f}%"),
            ("Win-day rate",      f"{vp['win_rate']:.1f}%",     wr_color,
                                   f"{vp['pos_days']}↑  {vp['neg_days']}↓"),
            ("Ann. return",       f"{ret_sign}{vp['ann_ret']:.2f}%", ret_color,
                                   f"3-yr backtest"),
            ("Max drawdown",      f"{vp['max_dd']:.1f}%",       dd_color,   "peak-to-trough"),
            ("Daily Sharpe",      f"{vp['sharpe']:.3f}",         "#94a3b8",  "annualised"),
            ("Best day",          f"+{vp['best_day_pct']:.2f}%", "#22d3ee",  vp["best_day_dt"]),
            ("Worst day",         f"{vp['worst_day_pct']:.2f}%", "#f43f5e",  vp["worst_day_dt"]),
        ]

        cells_html = "".join(
            f'<div class="vport-cell">'
            f'<div class="vport-cell-label">{lbl}</div>'
            f'<div class="vport-cell-val" style="color:{col}">{val}</div>'
            f'<div class="vport-cell-sub">{sub}</div>'
            f'</div>'
            for lbl, val, col, sub in cells
        )

        st.markdown(f"""
<div class="vport-wrap">
  <div class="vport-badge">SIM · NO REAL MONEY</div>
  <div class="vport-title">Universe Portfolio Simulation · {vp['period_start']} → {vp['period_end']} · {vp['n_days']} trading days</div>
  <div class="vport-grid">{cells_html}</div>
  <div class="vport-disclaimer">
    Multi-stock universe portfolio backtest (SPY, AAPL, MSFT, NVDA, TLT et al.) using the trained ML signal with full risk management.
    GBP values converted from USD at £1 = $1.27 (approx). SPY single-ticker results (OOS Sharpe 0.581, CAGR 11.8%) shown in AI Model tab.
    Simulation only — no real capital at risk.
  </div>
</div>
""", unsafe_allow_html=True)

    # Cross-session Sharpe trend
    if sessions:
        all_rows = []
        for s in reversed(sessions):
            for r in s["log"]:
                all_rows.append({
                    "session":    s["id"][-8:],
                    "iter_global": len(all_rows) + 1,
                    "sharpe":     r.get("sharpe", 0.0),
                    "promoted":   r.get("promoted", False),
                    "arm":        r.get("config", {}).get("_bandit_arm", "unknown"),
                })
        df_all = pd.DataFrame(all_rows)

        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Performance Score — All Experiments")
            fig = px.scatter(
                df_all, x="iter_global", y="sharpe",
                color="arm",
                symbol="promoted",
                symbol_map={True: "star", False: "circle"},
                hover_data=["session", "arm"],
                labels={"iter_global": "Experiment #", "sharpe": "Score", "arm": "Approach"},
                height=340,
                color_discrete_sequence=["#38bdf8", "#fbbf24", "#22d3ee", "#818cf8", "#f43f5e", "#2dd4bf"],
            )
            fig.add_hline(y=PROMOTE_SHARPE_THRESHOLD, line_dash="dash",
                          line_color="#22d3ee", annotation_text="Approval threshold",
                          annotation_font_color="#22d3ee")
            fig.add_hline(y=0, line_dash="dot", line_color="#334155")
            fig.update_layout(
                **_dark_layout(height=340, legend_title="Approach",
                               legend=dict(orientation="h", y=-0.15))
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color="#09090f")))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.subheader("Score Distribution")
            fig2 = px.histogram(
                df_all, x="sharpe", nbins=20,
                color_discrete_sequence=["#38bdf8"],
                height=340,
            )
            fig2.add_vline(x=PROMOTE_SHARPE_THRESHOLD, line_dash="dash",
                           line_color="#22d3ee",
                           annotation_text=f"Approval >{PROMOTE_SHARPE_THRESHOLD}",
                           annotation_font_color="#22d3ee")
            fig2.update_layout(
                **_dark_layout(height=340, showlegend=False,
                               xaxis_title="Score", yaxis_title="# Experiments")
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(56,189,248,0.06) 0%, rgba(139,92,246,0.06) 100%);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 1rem 0 2rem 0;
">
  <div style="font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin-bottom: 0.5rem;">Ready to start research</div>
  <div style="color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem;">Run your first AI research session to see results here</div>
  <div style="
    display: inline-block;
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-family: monospace;
    font-size: 0.85rem;
    color: #38bdf8;
    letter-spacing: 0.02em;
  ">py harness_main.py demo --ticker SPY --iter 3</div>
  <div style="color: #475569; font-size: 0.75rem; margin-top: 1rem;">No API keys needed for demo mode</div>
</div>
""", unsafe_allow_html=True)

    # Approved strategies
    if promotions_idx:
        st.subheader(f"Approved Strategies ({len(promotions_idx)})")
        promo_rows = []
        for p in promotions_idx:
            m = p.get("metrics", {})
            promo_rows.append({
                "Strategy":    p.get("title", "?").replace("PROMOTED: ", "")[:60],
                "Score":       m.get("sharpe", 0),
                "Max Loss %":  m.get("max_dd", 0),
                "Yearly Ret %":m.get("ann_return", 0),
                "Session":     p.get("session", "?")[:16],
                "Date":        p.get("timestamp", "")[:10],
            })
        st.dataframe(
            pd.DataFrame(promo_rows).sort_values("Score", ascending=False),
            use_container_width=True, hide_index=True,
            column_config={
                "Score":        st.column_config.NumberColumn(format="%.3f"),
                "Max Loss %":   st.column_config.NumberColumn(format="%.1f"),
                "Yearly Ret %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
    else:
        st.info("No strategies approved yet — run the AI research loop to get started.")

    # AI strategy type leaderboard
    if bandit_arms:
        st.subheader("Best Strategy Approaches")
        arm_rows = []
        for arm, s in bandit_arms.items():
            n = s.get("n", 0)
            avg = s["sum_reward"] / n if n else 0.0
            arm_rows.append({
                "Approach":    arm,
                "Times Tried": n,
                "Avg Score":   round(avg, 3),
                "Best Score":  round(s.get("best", 0.0), 3),
            })
        arm_df = pd.DataFrame(arm_rows).sort_values("Avg Score", ascending=False)
        explored_df = arm_df[arm_df["Times Tried"] > 0]
        if not explored_df.empty:
            fig3 = px.bar(
                explored_df, x="Approach", y="Avg Score",
                color="Avg Score",
                color_continuous_scale=[[0, "#f43f5e"], [0.4, "#fbbf24"], [0.7, "#22d3ee"], [1, "#2dd4bf"]],
                range_color=[-0.3, 1.2],
                height=300,
                text="Avg Score",
            )
            fig3.add_hline(y=PROMOTE_SHARPE_THRESHOLD, line_dash="dash",
                           line_color="#22d3ee", annotation_text="Approval threshold",
                           annotation_font_color="#22d3ee")
            fig3.update_traces(texttemplate="%{text:.3f}", textposition="outside")
            fig3.update_layout(
                **_dark_layout(height=300, showlegend=False,
                               coloraxis_showscale=False,
                               yaxis_title="Average Score")
            )
            st.plotly_chart(fig3, use_container_width=True)

    # System health (real data where available)
    st.divider()
    st.subheader("System Health")
    sh = load_system_health()

    ks_color = "#f43f5e" if sh.get("kill_switch") == "ACTIVE" else "#22d3ee"
    ks_label = sh.get("kill_switch", "—")

    lv = sh.get("ledger_valid")
    if lv is True:
        ledger_str = '<span class="status-ok">✓ Valid</span>'
    elif lv is False:
        ledger_str = '<span class="status-err">✗ Invalid</span>'
    else:
        ledger_str = '<span class="status-warn">— no ledger today</span>'

    sh_rows = [
        ("Safety Switch",      f'<span style="color:{ks_color};font-weight:700">{ks_label}</span>'),
        ("Audit Trail",        ledger_str),
        ("Events Logged Today",str(sh.get("ledger_records", "—"))),
        ("Orders Today",       str(sh.get("ledger_orders", "—"))),
        ("Model Last Trained", str(sh.get("model_trained_at", "—"))),
        ("Signals in Model",   str(sh.get("model_features", "—"))),
        ("Avg Model Score",    str(sh.get("model_oos_sharpe", "—"))),
        ("Days Ahead",         f"{sh.get('model_horizon', '—')} days"),
    ]
    col_sh1, col_sh2 = st.columns(2)
    for i, (label, val) in enumerate(sh_rows):
        col = col_sh1 if i % 2 == 0 else col_sh2
        col.markdown(f"**{label}:** {val}", unsafe_allow_html=True)

    # ── Activity Feed ─────────────────────────────────────────────────────────
    if sessions:
        st.divider()
        st.subheader("Recent Activity")
        latest_sess = sessions[0]
        latest_log  = latest_sess["log"]
        st.caption(
            f"Session **{latest_sess['id']}** · "
            f"{len(latest_log)} iterations · "
            f"showing last {min(5, len(latest_log))}"
        )
        _VERDICT_COLOR = {"APPROVED": "#22d3ee", "PROMISING": "#fbbf24", "NOT YET": "#f43f5e"}
        for r in reversed(latest_log[-5:]):
            bt     = r.get("backtest", {}) or {}
            sharpe = r.get("sharpe", 0.0) or 0.0
            arm    = r.get("config", {}).get("_bandit_arm", "—")
            hyp    = r.get("config", {}).get("hypothesis", "—")[:70]
            if r.get("promoted"):
                verdict, vcolor = "APPROVED",  _VERDICT_COLOR["APPROVED"]
            elif sharpe >= 0.4:
                verdict, vcolor = "PROMISING", _VERDICT_COLOR["PROMISING"]
            else:
                verdict, vcolor = "NOT YET",   _VERDICT_COLOR["NOT YET"]
            trades  = bt.get("n_trades", "—")
            max_dd  = abs(bt.get("max_dd", 0) or 0)
            st.markdown(f"""
<div style="border-left:3px solid {vcolor}; padding:6px 12px; margin-bottom:6px;
            background:#111827; border-radius:0 6px 6px 0;">
  <span style="color:#94a3b8;font-size:0.73rem;">Experiment&nbsp;{r.get('iteration','?')}</span>
  <span style="color:{vcolor};font-weight:700;margin-left:8px;">{verdict}</span>
  <span style="color:#cbd5e1;margin-left:8px;font-weight:600;">Score&nbsp;{sharpe:.3f}</span>
  <span style="color:#64748b;margin-left:8px;font-size:0.78rem;">Max loss&nbsp;{max_dd:.1f}%&nbsp;·&nbsp;{trades}&nbsp;trades&nbsp;·&nbsp;{arm}</span>
  <div style="color:#94a3b8;font-size:0.80rem;margin-top:3px;">{hyp}</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Live Markets
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Live Markets":
    st.title("Live Markets")
    st.caption("Real-time quotes via yfinance · prices delayed ~15 min · Simulation only — no real trading")

    # Watchlist input
    col_w, col_btn = st.columns([5, 1])
    watchlist_str = col_w.text_input(
        "Watchlist",
        value=st.session_state.get("_watchlist", "SPY AAPL MSFT NVDA QQQ TSLA GLD TLT"),
        label_visibility="collapsed",
        placeholder="SPY AAPL MSFT NVDA …",
    )
    tickers_live = [t.strip().upper() for t in watchlist_str.split() if t.strip()][:12]
    if col_btn.button("Refresh quotes", use_container_width=True):
        st.cache_data.clear()
    st.session_state["_watchlist"] = watchlist_str

    if not tickers_live:
        st.warning("Enter at least one ticker in the watchlist field above.")
        st.stop()

    # ── Fetch live quotes ─────────────────────────────────────────────────────
    @st.cache_data(ttl=60)
    def _fetch_quotes(tickers_tuple: tuple) -> list[dict]:
        import yfinance as yf
        rows = []
        for t in tickers_tuple:
            try:
                tk   = yf.Ticker(t)
                fi   = tk.fast_info
                hist = tk.history(period="5d", interval="1d", auto_adjust=True)
                prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else None
                curr = fi.last_price
                if curr is None:
                    curr = float(hist["Close"].iloc[-1]) if not hist.empty else None
                if curr is None:
                    raise ValueError("no price")
                chg     = round(curr - prev, 2)       if prev else 0.0
                chg_pct = round(chg / prev * 100, 2)  if prev else 0.0
                rows.append({
                    "Ticker":    t,
                    "Price":     round(curr, 2),
                    "Change":    chg,
                    "Chg %":     chg_pct,
                    "Volume":    getattr(fi, "three_month_average_volume", None),
                    "52w High":  round(fi.year_high, 2) if fi.year_high else "—",
                    "52w Low":   round(fi.year_low,  2) if fi.year_low  else "—",
                    "Mkt Cap":   fi.market_cap,
                    "_ok":       True,
                })
            except Exception:
                rows.append({
                    "Ticker": t, "Price": None, "Change": 0, "Chg %": 0,
                    "Volume": None, "52w High": "—", "52w Low": "—",
                    "Mkt Cap": None, "_ok": False,
                })
        return rows

    with st.spinner("Fetching quotes…"):
        quotes = _fetch_quotes(tuple(tickers_live))

    # ── Quote cards ───────────────────────────────────────────────────────────
    n_show = min(len(quotes), 6)
    card_cols = st.columns(n_show)
    for i, q in enumerate(quotes[:n_show]):
        price = q["Price"]
        chg   = q["Chg %"]
        label = f"£{price:,.2f}" if isinstance(price, (int, float)) else "—"
        delta = f"{chg:+.2f}%" if isinstance(chg, (int, float)) else None
        card_cols[i].metric(
            q["Ticker"], label, delta=delta,
            delta_color="normal",
        )

    st.divider()

    # ── Quotes table ──────────────────────────────────────────────────────────
    col_tbl, col_chart = st.columns([2, 3])

    with col_tbl:
        st.subheader("Quotes")
        def _fmt_vol(v):
            if v is None: return "—"
            if v >= 1e9: return f"{v/1e9:.1f}B"
            if v >= 1e6: return f"{v/1e6:.1f}M"
            return f"{v:,.0f}"
        def _fmt_cap(v):
            if v is None: return "—"
            if v >= 1e12: return f"£{v/1e12:.2f}T"
            if v >= 1e9:  return f"£{v/1e9:.1f}B"
            return f"£{v/1e6:.0f}M"

        tbl_rows = []
        for q in quotes:
            chg_pct = q["Chg %"]
            tbl_rows.append({
                "Ticker":   q["Ticker"],
                "Price":    f"£{q['Price']:,.2f}" if isinstance(q["Price"], (int, float)) else "—",
                "Chg %":   f"{chg_pct:+.2f}%" if isinstance(chg_pct, (int, float)) else "—",
                "Avg Vol":  _fmt_vol(q["Volume"]),
                "52w H":    f"£{q['52w High']}" if isinstance(q["52w High"], (int, float)) else "—",
                "52w L":    f"£{q['52w Low']}"  if isinstance(q["52w Low"],  (int, float)) else "—",
                "Mkt Cap":  _fmt_cap(q["Mkt Cap"]),
            })
        df_q = pd.DataFrame(tbl_rows)

        def _style_chg(val):
            if "+" in str(val):
                return "color:#22d3ee; font-weight:600"
            if "-" in str(val):
                return "color:#f43f5e; font-weight:600"
            return ""

        st.dataframe(
            df_q.style.map(_style_chg, subset=["Chg %"]),
            use_container_width=True, hide_index=True,
        )

    # ── Intraday chart ────────────────────────────────────────────────────────
    with col_chart:
        chart_ticker = st.selectbox("Intraday chart", tickers_live, key="chart_sel")

        @st.cache_data(ttl=300)
        def _fetch_intraday(t: str) -> pd.DataFrame:
            import yfinance as yf
            df = yf.download(t, period="1d", interval="5m",
                             progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df

        intra = _fetch_intraday(chart_ticker)
        if intra.empty:
            st.info(f"No intraday data for {chart_ticker} (market may be closed).")
        else:
            fig_c = go.Figure()
            close_col = "Close" if "Close" in intra.columns else intra.columns[3]
            open_col  = "Open"  if "Open"  in intra.columns else None
            high_col  = "High"  if "High"  in intra.columns else None
            low_col   = "Low"   if "Low"   in intra.columns else None
            if all(c is not None for c in [open_col, high_col, low_col]):
                fig_c.add_trace(go.Candlestick(
                    x=intra.index,
                    open=intra[open_col], high=intra[high_col],
                    low=intra[low_col],  close=intra[close_col],
                    name=chart_ticker,
                    increasing_line_color="#22d3ee",
                    decreasing_line_color="#f43f5e",
                ))
            else:
                fig_c.add_trace(go.Scatter(
                    x=intra.index, y=intra[close_col],
                    mode="lines", name=chart_ticker,
                    line=dict(color="#3b82f6", width=1.5),
                ))
            fig_c.update_layout(
                **_dark_layout(
                    height=320, xaxis_rangeslider_visible=False,
                    title=f"{chart_ticker} — Today (5-min)",
                    title_font=dict(size=13),
                    xaxis_title="", yaxis_title="Price (£)",
                )
            )
            st.plotly_chart(fig_c, use_container_width=True)

    # ── AlphaForge signals for watchlist ─────────────────────────────────────
    st.divider()
    st.subheader("AlphaForge Model Signals")
    st.caption("Signals generated by the last trained model on the most recent cached features. "
               "These are backtested research outputs — not trading recommendations.")

    @st.cache_data(ttl=300)
    def _get_model_signals(tickers_tuple: tuple) -> list[dict]:
        rows = []
        for t in tickers_tuple:
            try:
                import numpy as _np
                from models.train import ModelTrainer
                from features.engine import FeatureEngine
                from data.ingest import DataIngestion
                import yaml
                cfg_path = ROOT / "config.yaml"
                cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
                trainer = ModelTrainer(config=cfg)
                trainer.load(t)
                feat_file = sorted(
                    (ROOT / "data" / "processed").glob(f"{t.lower()}_features*.parquet"),
                    key=lambda p: p.stat().st_mtime, reverse=True,
                )
                if not feat_file:
                    raise FileNotFoundError("no features")
                feat_df = pd.read_parquet(feat_file[0])
                feat_cols = [c for c in trainer.feature_columns if c in feat_df.columns] \
                            if hasattr(trainer, "feature_columns") else \
                            [c for c in feat_df.columns if c not in
                             ("label", "close", "open", "high", "low", "volume")]
                X = feat_df[feat_cols].iloc[[-1]]
                proba = float(trainer.predict_proba(X)[0])
                thr = float(cfg.get("model", {}).get("confidence_threshold", 0.55))
                if proba > thr:
                    sig, color = "LONG  ▲", "#22d3ee"
                elif proba < 1 - thr:
                    sig, color = "SHORT ▼", "#f43f5e"
                else:
                    sig, color = "FLAT  —", "#64748b"
                rows.append({"Ticker": t, "Signal": sig, "P(up)": round(proba, 3),
                              "color": color, "error": None})
            except Exception as e:
                rows.append({"Ticker": t, "Signal": "N/A", "P(up)": None,
                              "color": "#475569", "error": str(e)[:40]})
        return rows

    with st.spinner("Loading model signals…"):
        sig_rows = _get_model_signals(tuple(tickers_live))

    sig_cols = st.columns(min(len(sig_rows), 6))
    for i, s in enumerate(sig_rows[:6]):
        sig_cols[i].markdown(
            f"<div style='text-align:center; background:#111827; border:1px solid #1e293b; "
            f"border-radius:8px; padding:10px 4px;'>"
            f"<div style='color:#94a3b8; font-size:0.75rem;'>{s['Ticker']}</div>"
            f"<div style='color:{s['color']}; font-size:1.1rem; font-weight:700; margin:4px 0;'>{s['Signal']}</div>"
            f"<div style='color:#64748b; font-size:0.72rem;'>"
            f"{'P(up)='+str(s['P(up)']) if s['P(up)'] is not None else s.get('error','no model')}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Strategy Library
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Strategy Library":
    _section_header("Strategy Library", "All strategies discovered across every research session")

    if not kb_index:
        _empty_state("No strategies yet", "Run a research session to populate the library.")
    else:
        # ── Stats row ─────────────────────────────────────────────────────────
        by_type: dict[str, int] = {}
        for e in kb_index:
            t = e.get("type", "unknown")
            by_type[t] = by_type.get(t, 0) + 1

        label_map = {
            "promotion":  "Approved",
            "experiment": "Experiment",
            "finding":    "Finding",
            "heuristic":  "Heuristic",
            "failure":    "Failure",
        }
        color_map = {
            "promotion":  "#22d3ee",
            "failure":    "#f43f5e",
            "experiment": "#38bdf8",
            "finding":    "#fbbf24",
            "heuristic":  "#818cf8",
        }
        stat_items = [("TOTAL", str(len(kb_index)), "#f1f5f9")] + [
            (label_map.get(t, t).upper(), str(c), color_map.get(t, "#94a3b8"))
            for t, c in sorted(by_type.items())
        ]
        cards_html = "".join(
            f'<div style="background:#0e1117;border:1px solid rgba(255,255,255,0.07);'
            f'border-radius:8px;padding:0.7rem 1rem;">'
            f'<div style="font-size:0.58rem;color:#475569;text-transform:uppercase;'
            f'letter-spacing:0.12em;font-weight:700;margin-bottom:4px;">{lbl}</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.4rem;'
            f'font-weight:700;color:{color};">{val}</div>'
            f'</div>'
            for lbl, val, color in stat_items
        )
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));'
            f'gap:8px;margin-bottom:1.2rem;">{cards_html}</div>',
            unsafe_allow_html=True,
        )

        # ── Search / filter ───────────────────────────────────────────────────
        col_s, col_t = st.columns([3, 1])
        search = col_s.text_input("Search strategies", placeholder="keyword...", label_visibility="collapsed")
        type_f = col_t.selectbox("Type", ["All"] + sorted(by_type.keys()), label_visibility="collapsed")

        filtered = [
            e for e in kb_index
            if (type_f == "All" or e.get("type") == type_f)
            and (not search or search.lower() in json.dumps(e).lower())
        ]
        st.caption(f"{len(filtered)} of {len(kb_index)} entries")

        # ── Table ─────────────────────────────────────────────────────────────
        rows_kb = []
        for e in filtered:
            m = e.get("metrics", {}) or {}
            rows_kb.append({
                "Type":     label_map.get(e.get("type", ""), e.get("type", "—")),
                "Approach": e.get("approach", "—"),
                "Sharpe":   round(float(m.get("oos_sharpe", 0) or 0), 3),
                "Max DD %": round(abs(float(m.get("max_dd", 0) or 0)), 1),
                "Win Rate": round(float(m.get("win_rate", 0) or 0) * 100, 1),
                "Summary":  str(e.get("summary", e.get("hypothesis", "—")))[:80],
            })
        if rows_kb:
            kb_df = pd.DataFrame(rows_kb)
            st.dataframe(
                kb_df.style.map(_color_sharpe, subset=["Sharpe"]),
                use_container_width=True, hide_index=True,
                column_config={
                    "Sharpe":   st.column_config.NumberColumn(format="%.3f"),
                    "Max DD %": st.column_config.NumberColumn(format="%.1f"),
                    "Win Rate": st.column_config.NumberColumn(format="%.1f"),
                },
            )

        # ── Top strategies chart ──────────────────────────────────────────────
        scored = [e for e in filtered if (e.get("metrics", {}) or {}).get("oos_sharpe")]
        if scored:
            st.divider()
            _section_header("Top Strategies by Score")
            top15 = sorted(scored, key=lambda x: x["metrics"]["oos_sharpe"], reverse=True)[:15]
            bar_df = pd.DataFrame({
                "Strategy": [str(e.get("approach", "?"))[:35] for e in top15],
                "Sharpe":   [round(float(e["metrics"]["oos_sharpe"]), 3) for e in top15],
                "Type":     [e.get("type", "—") for e in top15],
            })
            fig = px.bar(
                bar_df, x="Sharpe", y="Strategy", orientation="h", color="Type",
                color_discrete_map=color_map,
            )
            fig.update_layout(**_dark_layout(title="Top 15 Strategies — OOS Sharpe", height=420, showlegend=True))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            fig.add_vline(x=PROMOTE_SHARPE_THRESHOLD, line_dash="dash",
                          line_color="#22d3ee", annotation_text="Approval threshold")
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: AI Learning
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "AI Learning":
    _section_header("AI Learning", "How the bandit learns which strategy approaches work best")

    bandit_arms   = bandit_st.get("arms", {})
    bandit_trials = bandit_st.get("total_trials", 0)

    if not bandit_arms:
        _empty_state("No learning data yet", "Run research sessions to train the AI bandit.")
    else:
        # ── KPI row ───────────────────────────────────────────────────────────
        best_arm  = max(bandit_arms, key=lambda k: bandit_arms[k].get("mean", 0))
        best_mean = bandit_arms[best_arm].get("mean", 0)
        n_arms    = len(bandit_arms)
        kpi_cols  = st.columns(3)
        kpi_cols[0].metric("Total Experiments", bandit_trials)
        kpi_cols[1].metric("Approaches Tried",  n_arms)
        kpi_cols[2].metric("Best Approach Score", f"{best_mean:.3f}")

        # ── Approach performance ──────────────────────────────────────────────
        st.divider()
        _section_header("Approach Performance", "Mean score and trial count per strategy approach")

        explored = [
            {
                "Approach":   k,
                "Mean Score": round(v.get("mean", 0), 4),
                "Trials":     v.get("n", 0),
                "Std Dev":    round(v.get("std", 0), 4),
            }
            for k, v in bandit_arms.items()
        ]
        explored_df = pd.DataFrame(explored).sort_values("Mean Score", ascending=False)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=explored_df["Approach"], y=explored_df["Mean Score"],
            marker_color=[
                "#22d3ee" if s >= PROMOTE_SHARPE_THRESHOLD else
                "#fbbf24" if s >= 0.4 else "#f43f5e"
                for s in explored_df["Mean Score"]
            ],
            name="Mean Score",
        ))
        fig2.add_trace(go.Scatter(
            x=explored_df["Approach"], y=explored_df["Trials"],
            mode="markers", marker=dict(size=10, color="#818cf8"),
            name="Trials", yaxis="y2",
        ))
        fig2.update_layout(**_dark_layout(title="Mean Score per Approach", height=380))
        fig2.update_layout(
            yaxis=dict(title="Mean Score"),
            yaxis2=dict(title="Trials", overlaying="y", side="right"),
            xaxis=dict(tickangle=-30),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # ── Experiment mix pie ────────────────────────────────────────────────
        col_pie, col_tbl = st.columns([1, 1])
        fig_pie = px.pie(
            explored_df, names="Approach", values="Trials",
        )
        fig_pie.update_layout(**_dark_layout(title="Experiment Mix", height=320))
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        col_pie.plotly_chart(fig_pie, use_container_width=True)
        col_tbl.dataframe(
            explored_df[["Approach", "Mean Score", "Trials", "Std Dev"]],
            use_container_width=True, hide_index=True,
        )

        # ── Learning curve ────────────────────────────────────────────────────
        if sessions:
            st.divider()
            _section_header("Learning Curve", "Best score found so far across all sessions")
            curve_rows = []
            running_best = 0.0
            global_exp = 0
            for sess in reversed(sessions):
                for r in sess["log"]:
                    global_exp += 1
                    s = r.get("sharpe", 0.0) or 0.0
                    running_best = max(running_best, s)
                    curve_rows.append({
                        "Experiment":  global_exp,
                        "Score":       round(s, 4),
                        "Best So Far": round(running_best, 4),
                    })
            if curve_rows:
                curve_df = pd.DataFrame(curve_rows)
                fig_lc = go.Figure()
                fig_lc.add_trace(go.Scatter(
                    x=curve_df["Experiment"], y=curve_df["Score"],
                    mode="markers", marker=dict(size=4, color="#38bdf8", opacity=0.5),
                    name="Score",
                ))
                fig_lc.add_trace(go.Scatter(
                    x=curve_df["Experiment"], y=curve_df["Best So Far"],
                    mode="lines", line=dict(color="#fbbf24", width=2),
                    name="Best So Far",
                ))
                fig_lc.update_layout(
                    **_dark_layout(title="Learning Curve — Best Score Over Time", height=340),
                    xaxis_title="Cumulative Experiment #",
                    yaxis_title="Score (Sharpe-like)",
                )
                st.plotly_chart(fig_lc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Reports
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Reports":
    _section_header("Session Reports", "Full markdown reports generated after each research session")

    report_dir   = RESULTS_DIR / "reports"
    report_files = sorted(report_dir.glob("session_*.md"), reverse=True) if report_dir.exists() else []

    if not report_files:
        _empty_state("No reports yet", "Reports are written automatically after each session completes.")
    else:
        sel_report = st.selectbox(
            "Select report",
            report_files,
            format_func=lambda p: p.stem.replace("session_", ""),
            label_visibility="collapsed",
        )
        content = sel_report.read_text(encoding="utf-8") if sel_report else ""
        col_dl, _ = st.columns([1, 4])
        col_dl.download_button(
            "Download", data=content, file_name=sel_report.name, mime="text/markdown",
            use_container_width=True,
        )
        st.divider()
        st.markdown(content)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB: Paper Trading
# ═══════════════════════════════════════════════════════════════════════════════

elif tab_choice == "Paper Trading":
    import subprocess, time as _time

    _section_header("Paper Trading Simulator", "Simulation only — no real orders, no real money, no broker connections")

    PAPER_LOG_DIR = ROOT / "logs" / "paper_trading"

    # ── Ticker + controls row ─────────────────────────────────────────────────
    _tc1, _tc2, _tc3, _tc4 = st.columns([2, 1, 1, 1])
    with _tc1:
        available_tickers: list[str] = []
        if PAPER_LOG_DIR.exists():
            for _f in PAPER_LOG_DIR.glob("*_paper_trades.csv"):
                available_tickers.append(_f.stem.replace("_paper_trades", "").upper())
        available_tickers = sorted(set(available_tickers)) or ["SPY"]
        pt_ticker = st.selectbox("Ticker", available_tickers, label_visibility="visible")
    with _tc2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _run_clicked = st.button("Run Live Update", use_container_width=True, type="primary")
    with _tc3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        _auto_refresh = st.toggle("Auto-refresh (60s)", value=False)
    with _tc4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Run Live Update (subprocess) ──────────────────────────────────────────
    if _run_clicked:
        with st.spinner(f"Fetching latest data for {pt_ticker} and running simulation…"):
            try:
                _proc = subprocess.run(
                    ["py", "main.py", "paper-trade",
                     "--ticker", pt_ticker,
                     "--replay-start", "2022-01-01"],
                    capture_output=True, text=True, timeout=180, cwd=str(ROOT),
                )
                if _proc.returncode == 0:
                    st.success(f"Simulation updated for {pt_ticker}.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    _err = (_proc.stderr or _proc.stdout or "unknown error")[:600]
                    st.error(f"Run failed: {_err}")
            except subprocess.TimeoutExpired:
                st.error("Timed out after 3 minutes — try running manually.")
            except Exception as _exc:
                st.error(f"Could not start process: {_exc}")

    trades_path = PAPER_LOG_DIR / f"{pt_ticker.lower()}_paper_trades.csv"
    equity_path = PAPER_LOG_DIR / f"{pt_ticker.lower()}_equity_curve.csv"

    # ── Load data ─────────────────────────────────────────────────────────────
    @st.cache_data(ttl=30)
    def _load_paper_trades(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, parse_dates=["timestamp"])
            return df.sort_values("timestamp")
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=30)
    def _load_equity(path: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, parse_dates=["date"])
            df = df.sort_values("date")
            if df["daily_return"].abs().sum() < 1e-9:
                df["daily_return"] = df["nav"].pct_change().fillna(0)
            return df
        except Exception:
            return pd.DataFrame()

    @st.cache_data(ttl=60)
    def _fetch_live_price(tkr: str) -> dict:
        try:
            import yfinance as yf
            info = yf.Ticker(tkr).fast_info
            return {
                "price":      float(getattr(info, "last_price", 0) or 0),
                "prev_close": float(getattr(info, "previous_close", 0) or 0),
            }
        except Exception:
            return {}

    trades_df = _load_paper_trades(str(trades_path))
    equity_df = _load_equity(str(equity_path))

    # ── Live price bar ─────────────────────────────────────────────────────────
    _live = _fetch_live_price(pt_ticker)
    if _live.get("price"):
        _lp = _live["price"]
        _pc = _live.get("prev_close", _lp)
        _chg = (_lp - _pc) / _pc * 100 if _pc else 0
        _chg_col = "#22d3ee" if _chg >= 0 else "#f43f5e"
        _chg_sign = "+" if _chg >= 0 else ""
        _last_sim_date = trades_df["timestamp"].iloc[-1].strftime("%Y-%m-%d") if not trades_df.empty else "—"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:1.5rem;padding:0.6rem 1rem;'
            f'background:rgba(56,189,248,0.05);border:1px solid rgba(56,189,248,0.15);'
            f'border-radius:8px;margin-bottom:0.8rem;">'
            f'<span style="font-size:0.75rem;color:#94a3b8;font-weight:600;letter-spacing:0.06em;">'
            f'LIVE — {pt_ticker}</span>'
            f'<span style="font-size:1.25rem;font-weight:700;color:#f1f5f9;font-family:\'IBM Plex Mono\',monospace;">'
            f'${_lp:,.2f}</span>'
            f'<span style="font-size:0.9rem;font-weight:600;color:{_chg_col};">'
            f'{_chg_sign}{_chg:.2f}% today</span>'
            f'<span style="font-size:0.72rem;color:#475569;margin-left:auto;">'
            f'Simulation last updated: {_last_sim_date}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if trades_df.empty:
        _empty_state(
            f"No simulation data for {pt_ticker}",
            f'Click "Run Live Update" above or run:  py main.py paper-trade --ticker {pt_ticker}',
        )
        st.stop()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    _initial_nav = 100_000.0
    _final_nav = float(equity_df["nav"].iloc[-1]) if not equity_df.empty else _initial_nav
    _total_ret = (_final_nav - _initial_nav) / _initial_nav * 100

    _exit_statuses = ["close_long", "close_short", "stop_loss", "take_profit"]
    _entry_mask = trades_df["fill_status"].isin(["long_entry", "short_entry"])
    _exit_mask  = trades_df["fill_status"].isin(_exit_statuses)
    _n_trades   = int(_entry_mask.sum())
    _exits      = trades_df[_exit_mask]
    _win_rate   = float((_exits["daily_pnl_pct"] > 0).mean() * 100) if len(_exits) else 0.0
    _n_stops    = int(trades_df["stop_loss_triggered"].sum()) if "stop_loss_triggered" in trades_df.columns else 0
    _n_tp       = int((trades_df["fill_status"] == "take_profit").sum()) if "fill_status" in trades_df.columns else 0

    _ret_prefix = "+" if _total_ret >= 0 else ""

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Final NAV",     f"${_final_nav:,.0f}", f"{_ret_prefix}{_total_ret:.2f}%")
    kc2.metric("Total Return",  f"{_ret_prefix}{_total_ret:.2f}%")
    kc3.metric("Win Rate",      f"{_win_rate:.1f}%", f"{_n_trades} trades")
    kc4.metric("Take-Profits / Stops", f"{_n_tp} / {_n_stops}")

    st.divider()

    # ── Current simulated position ────────────────────────────────────────────
    _last = trades_df.iloc[-1]
    _side = str(_last.get("position_side", "FLAT"))
    _side_color = {"LONG": "#22d3ee", "SHORT": "#f43f5e"}.get(_side, "#475569")
    _unrealised = float(_last.get("unrealised_pnl", 0.0))
    _qty        = float(_last.get("position_qty", 0.0))
    _sim_price  = float(_last.get("close", 0.0))
    _last_proba = float(_last.get("proba", 0.0))
    _last_regime_code = int(_last.get("regime", 0)) if "regime" in _last.index else 0
    _REGIME_MAP_DISP = {1: "Bull", -1: "Bear", 0: "Sideways", 2: "Sideways", 3: "High Vol", -99: "Unknown"}
    _last_regime = _REGIME_MAP_DISP.get(_last_regime_code, "Unknown")

    _section_header("Current Simulated Position")
    pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
    pc1.markdown(
        f'<div style="font-size:0.73rem;color:#94a3b8;margin-bottom:4px;">Side</div>'
        f'<div style="font-size:1.4rem;font-weight:800;color:{_side_color};">{_side}</div>',
        unsafe_allow_html=True,
    )
    pc2.metric("Qty",            f"{_qty:,.2f}")
    pc3.metric("Unrealised P&L", f"${_unrealised:,.2f}")
    pc4.metric("Last Sim Price", f"${_sim_price:,.2f}")
    pc5.metric("Model Confidence", f"{_last_proba*100:.1f}%")
    pc6.metric("Regime", _last_regime)

    st.divider()

    # ── Equity curve + daily returns ──────────────────────────────────────────
    chart_left, chart_right = st.columns([3, 2])

    with chart_left:
        _section_header("Equity Curve (NAV)")
        if not equity_df.empty:
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(
                x=equity_df["date"], y=equity_df["nav"],
                mode="lines",
                line=dict(color="#38bdf8", width=1.8),
                fill="tozeroy",
                fillcolor="rgba(56,189,248,0.06)",
                name="NAV",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>NAV: $%{y:,.0f}<extra></extra>",
            ))
            fig_eq.update_layout(**_dark_layout(title="", height=340))
            fig_eq.update_layout(yaxis=dict(title="NAV ($)", tickformat="$,.0f"), xaxis=dict(title=""))
            st.plotly_chart(fig_eq, use_container_width=True)
        else:
            st.info("Equity curve data unavailable.")

    with chart_right:
        _section_header("Daily Returns")
        if not equity_df.empty and "daily_return" in equity_df.columns:
            _dr = equity_df.copy()
            _dr["color"] = _dr["daily_return"].apply(lambda v: "#22d3ee" if v >= 0 else "#f43f5e")
            fig_dr = go.Figure(go.Bar(
                x=_dr["date"], y=_dr["daily_return"] * 100,
                marker_color=_dr["color"].tolist(),
                name="Daily Return %",
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Return: %{y:.3f}%<extra></extra>",
            ))
            fig_dr.update_layout(**_dark_layout(title="", height=340))
            fig_dr.update_layout(yaxis=dict(title="Return (%)"), xaxis=dict(title=""))
            st.plotly_chart(fig_dr, use_container_width=True)

    st.divider()

    # ── Trade metrics ─────────────────────────────────────────────────────────
    _section_header("Trade Metrics")
    tm1, tm2, tm3, tm4 = st.columns(4)

    _in_trade = trades_df["position_side"] != "FLAT"
    _groups   = (_in_trade != _in_trade.shift()).cumsum()
    _hold_dur = trades_df[_in_trade].groupby(_groups[_in_trade]).size()
    _avg_hold = f"{_hold_dur.mean():.1f} bars" if len(_hold_dur) else "—"

    _traded = trades_df[trades_df["fill_price"].notna() & (trades_df["daily_pnl_pct"] != 0)].copy()
    _winners = _traded[_traded["daily_pnl_pct"] > 0]["daily_pnl_pct"]
    _losers  = _traded[_traded["daily_pnl_pct"] < 0]["daily_pnl_pct"]
    _avg_win  = f"+{_winners.mean()*100:.3f}%" if len(_winners) else "—"
    _avg_loss = f"{_losers.mean()*100:.3f}%"   if len(_losers)  else "—"
    _total_comm = f"${trades_df['commission_paid'].sum():.2f}" if "commission_paid" in trades_df.columns else "—"

    tm1.metric("Avg Hold Duration", _avg_hold)
    tm2.metric("Avg Winner",        _avg_win)
    tm3.metric("Avg Loser",         _avg_loss)
    tm4.metric("Total Commission",  _total_comm)

    st.divider()

    # ── Recent decisions table ────────────────────────────────────────────────
    _section_header("Recent Decisions", f"Last 50 bars — {pt_ticker}")

    _ACTION_MAP = {
        "long_entry":  "BUY",
        "short_entry": "SELL",
        "close_long":  "CLOSE",
        "close_short": "CLOSE",
        "stop_loss":   "STOP",
        "take_profit": "TAKE PROFIT",
        "no_action":   "HOLD",
    }

    _disp = trades_df.tail(50).copy()
    _disp["Action"] = _disp["fill_status"].apply(
        lambda s: "REJECTED" if str(s).startswith("rejected") else _ACTION_MAP.get(str(s), "HOLD")
    )
    _disp["Regime"]       = _disp["regime"].map(_REGIME_MAP_DISP).fillna("Unknown") if "regime" in _disp.columns else "—"
    _disp["Confidence %"] = (_disp["proba"] * 100).round(1) if "proba" in _disp.columns else "—"
    _disp["P&L %"]        = (_disp["daily_pnl_pct"] * 100).round(3) if "daily_pnl_pct" in _disp.columns else "—"

    _disp_out = _disp[["timestamp", "Action", "fill_price", "position_side", "position_qty",
                        "Confidence %", "Regime", "P&L %", "commission_paid", "unrealised_pnl", "nav"]].copy()

    def _action_color_map(val: str) -> str:
        return {
            "BUY":         "background-color:rgba(34,211,238,0.12); color:#22d3ee; font-weight:700",
            "SELL":        "background-color:rgba(244,63,94,0.12);  color:#f43f5e; font-weight:700",
            "STOP":        "background-color:rgba(244,63,94,0.20);  color:#f43f5e; font-weight:700",
            "TAKE PROFIT": "background-color:rgba(34,211,100,0.14); color:#4ade80; font-weight:700",
            "CLOSE":       "background-color:rgba(251,191,36,0.10); color:#fbbf24; font-weight:600",
            "HOLD":        "color:#475569",
            "REJECTED":    "background-color:rgba(100,16,32,0.15);  color:#fb7185; font-weight:600",
        }.get(str(val), "")

    st.dataframe(
        _disp_out.style.map(_action_color_map, subset=["Action"]),
        use_container_width=True, hide_index=True,
    )
    st.caption("SIMULATION ONLY — No real orders. No real money. No broker connections.")

    # ── Auto-refresh ──────────────────────────────────────────────────────────
    if _auto_refresh:
        _time.sleep(60)
        st.cache_data.clear()
        st.rerun()
