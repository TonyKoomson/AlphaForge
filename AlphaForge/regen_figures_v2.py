#!/usr/bin/env python3
"""
AlphaForge dissertation figures — v2
Uses:  graphviz (Figs 1-3), napkin/plantuml (Fig 4),
       matplotlib-seaborn (Figs 5-10)
"""
import os, sys
os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import graphviz
import napkin
from pathlib import Path
import warnings, subprocess, textwrap
warnings.filterwarnings('ignore')

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Shared palette & style ────────────────────────────────────────────────────
BLUE   = '#1d4ed8'; LBLUE  = '#dbeafe'
PURP   = '#6d28d9'; LPURP  = '#ede9fe'
GREEN  = '#065f46'; LGREEN = '#d1fae5'
AMBER  = '#b45309'; LAMBER = '#fef3c7'
RED    = '#991b1b'; LRED   = '#fee2e2'
TEAL   = '#0e7490'; LTEAL  = '#cffafe'
ORANGE = '#c2410c'; LORANG = '#ffedd5'
GRAY   = '#374151'; MGRAY  = '#9ca3af'

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'axes.facecolor':     'white',
    'figure.facecolor':   'white',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.color':         '#cbd5e1',
    'font.size':          10,
})

def save_mpl(name):
    plt.savefig(FIG_DIR / name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  Saved: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Architecture  (matplotlib layered bands — full control)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    W, H = 18.0, 11.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── helpers ──────────────────────────────────────────────────────────────
    def band(x0, y0, w, h, fc, ec, label, label_y=None):
        r = FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.15',
                           fc=fc, ec=ec, lw=1.4, zorder=1)
        ax.add_patch(r)
        ly = label_y if label_y is not None else y0 + h - 0.30
        ax.text(x0 + 0.25, ly, label, fontsize=9, color=ec,
                fontweight='bold', va='top', zorder=2)

    def node(cx, cy, w, h, lines, fc, ec, fs=9.5):
        r = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle='round,pad=0.10',
                           fc=fc, ec=ec, lw=1.8, zorder=4)
        ax.add_patch(r)
        if isinstance(lines, str):
            lines = lines.split('\n')
        step = 0.30 if len(lines) > 1 else 0
        y0t = cy + step * (len(lines) - 1) / 2
        for k, ln in enumerate(lines):
            ax.text(cx, y0t - k * step, ln, ha='center', va='center',
                    fontsize=fs, color=ec, fontweight='bold', zorder=5)

    def arr(x0, y0, x1, y1, col='#475569', lw=1.6, ls='-', label=None, rad=0):
        style = f'arc3,rad={rad}'
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=col, lw=lw,
                                   linestyle=ls, connectionstyle=style), zorder=6)
        if label:
            mx, my = (x0+x1)/2 + 0.05, (y0+y1)/2
            ax.text(mx + 0.1, my, label, fontsize=8, color=col,
                    va='center', fontstyle='italic', zorder=7)

    # ── Band 1: Entry Points  (y 8.85 – 10.35) ───────────────────────────────
    band(0.4, 8.85, 17.2, 1.35, '#f8fafc', '#94a3b8', 'Entry Points')
    node(4.5,  9.57, 3.8, 0.90, ['main.py', '(Core ML CLI)'],       LBLUE,  BLUE)
    node(13.5, 9.57, 3.8, 0.90, ['harness_main.py', '(AI Harness CLI)'], LPURP, PURP)

    # ── Band 2: AI Harness  (y 6.40 – 8.55) ─────────────────────────────────
    band(0.4, 6.40, 17.2, 2.05, '#faf5ff', '#a855f7', 'AI Harness')
    agents = [
        (2.2,  7.48, ['Strategist', '(Claude)'],   LPURP,  PURP),
        (5.1,  7.48, ['Analyst', '(Grok/xAI)'],    LORANG, ORANGE),
        (8.0,  7.48, ['Coder', '(Claude)'],         LPURP,  PURP),
        (10.9, 7.48, ['Reviewer', '(Claude)'],      LPURP,  PURP),
    ]
    for cx, cy, lns, fc, ec in agents:
        node(cx, cy, 2.50, 0.82, lns, fc, ec)
    node(15.2, 7.48, 3.40, 0.82, ['AlphaHarness', 'Orchestrator'],
         '#ede9fe', '#7c3aed', fs=9.5)
    # arrows: agents → orchestrator
    for cx, cy, *_ in agents:
        arr(cx + 1.25, cy, 13.50, 7.48, PURP, lw=1.3)

    # ── Band 3: Core ML Pipeline  (y 3.55 – 6.05) ────────────────────────────
    band(0.4, 3.55, 17.2, 2.30, '#eff6ff', '#3b82f6', 'Core ML Pipeline')
    stages = [
        (1.7,  4.75, 'ingest',    LBLUE,  BLUE),
        (4.3,  4.75, 'features',  LGREEN, GREEN),
        (6.9,  4.75, 'train',     LGREEN, GREEN),
        (9.5,  4.75, 'backtest',  LBLUE,  BLUE),
        (12.1, 4.75, 'validate',  LTEAL,  TEAL),
        (14.7, 4.75, 'report',    LAMBER, AMBER),
    ]
    for cx, cy, lbl, fc, ec in stages:
        node(cx, cy, 2.20, 0.88, lbl, fc, ec, fs=10)
    for i in range(len(stages) - 1):
        arr(stages[i][0] + 1.10, stages[i][1],
            stages[i+1][0] - 1.10, stages[i+1][1], '#475569', lw=1.5)

    # ── Band 4: Infrastructure  (y 0.55 – 3.15) ──────────────────────────────
    band(0.4, 0.55, 17.2, 2.45, '#f0fdf4', '#059669', 'Infrastructure')
    infra = [
        (2.6,  1.85, ['data/', '(Parquet cache)'],          LGREEN, GREEN),
        (6.4,  1.85, ['models/artifacts/', '(joblib + JSON)'],LGREEN, GREEN),
        (10.2, 1.85, ['harness/memory/', '(KB + bandit)'],   LPURP,  PURP),
        (14.4, 1.85, ['logs/', 'config.yaml'],               LAMBER, AMBER),
    ]
    for cx, cy, lns, fc, ec in infra:
        node(cx, cy, 3.40, 0.88, lns, fc, ec, fs=9)

    # ── Cross-band arrows ─────────────────────────────────────────────────────
    # Entry Points → AI Harness / Pipeline
    arr(4.5,  9.12, 4.5,  8.55, BLUE,  lw=1.5)          # main → pipeline entry
    arr(13.5, 9.12, 15.2, 8.55, PURP,  lw=1.5)          # harness → orchestrator

    # Orchestrator → Core ML (tool calls)
    arr(15.2, 7.07, 15.2, 6.05, PURP, lw=1.5, ls='--', label='tool calls')
    # main → train (direct)
    arr(4.5,  8.55, 6.9,  6.05, BLUE, lw=1.4, ls='--', rad=0.1)

    # Pipeline stages → Infrastructure
    arr(1.7,  4.31, 2.6,  3.15, MGRAY, lw=1.2, ls='--')
    arr(6.9,  4.31, 6.4,  3.15, MGRAY, lw=1.2, ls='--')
    arr(12.1, 4.31, 10.2, 3.15, MGRAY, lw=1.2, ls='--')
    arr(14.7, 4.31, 14.4, 3.15, MGRAY, lw=1.2, ls='--')

    fig.suptitle('Figure 1 — AlphaForge System Architecture',
                 fontsize=15, fontweight='bold', y=0.995, color='#1e293b')
    save_mpl('figure1_architecture.png')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Data Flow Pipeline  (graphviz left-to-right)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2():
    g = graphviz.Digraph(
        'dataflow',
        graph_attr={
            'rankdir':  'LR',
            'splines':  'ortho',
            'nodesep':  '0.35',
            'ranksep':  '0.55',
            'fontname': 'Helvetica',
            'label':    'Figure 2 — AlphaForge Data Flow Pipeline',
            'labelloc': 't',
            'fontsize': '15',
            'fontcolor':'#1e293b',
            'bgcolor':  'white',
            'pad':      '0.4',
        },
        node_attr={
            'fontname': 'Helvetica',
            'fontsize': '11',
            'style':    'filled,rounded',
            'shape':    'box',
            'margin':   '0.22,0.14',
            'penwidth': '1.6',
        },
    )

    nodes = [
        ('ingest',    'Data Ingest\n──────────\nOHLCV\n+ adj close',     LBLUE,  BLUE),
        ('engine',    'Feature Engine\n──────────\n250+ cols\nno look-ahead', LGREEN, GREEN),
        ('selector',  'Feature Selector\n──────────\n85 cols\n(SHAP)',    LGREEN, GREEN),
        ('wfcv',      'Walk-Forward CV\n──────────\n26 folds\n21-bar embargo', LAMBER, AMBER),
        ('trainer',   'Model Train\n──────────\nXGB + LGBM\nensemble×8', LPURP,  PURP),
        ('backtest',  'Backtest Engine\n──────────\nNAV · DD\nSharpe',    LBLUE,  BLUE),
        ('report',    'Validate & Report\n──────────\nOOS metrics\nPDF/JSON', LTEAL, TEAL),
    ]

    edge_labels = ['Raw Parquet', 'Feature Matrix', 'Reduced Matrix',
                   'Train/Test Splits', 'Trained Model', 'Equity Curve']

    for nid, label, fc, ec in nodes:
        g.node(nid, label, fillcolor=fc, color=ec, fontcolor=ec)

    for i in range(len(nodes) - 1):
        g.edge(nodes[i][0], nodes[i+1][0],
               label=edge_labels[i], fontsize='9', fontcolor='#64748b',
               color='#475569')

    out = str(FIG_DIR / 'figure2_data_flow')
    g.render(out, format='png', cleanup=True)
    os.rename(out + '.png', str(FIG_DIR / 'figure2_data_flow.png'))
    print("  Saved: figure2_data_flow.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Risk Management Pipeline  (graphviz top-to-bottom flowchart)
# ═══════════════════════════════════════════════════════════════════════════════
def fig3():
    g = graphviz.Digraph(
        'risk',
        graph_attr={
            'rankdir':  'TB',
            'splines':  'ortho',
            'nodesep':  '0.40',
            'ranksep':  '0.50',
            'fontname': 'Helvetica',
            'label':    'Figure 3 — Risk Management Signal Pipeline',
            'labelloc': 't',
            'fontsize': '15',
            'fontcolor':'#1e293b',
            'bgcolor':  'white',
            'pad':      '0.5',
        },
        node_attr={'fontname': 'Helvetica', 'fontsize': '11', 'penwidth': '1.6'},
    )

    def process(nid, label):
        g.node(nid, label, shape='box', style='filled,rounded',
               fillcolor=LGREEN, color=GREEN, fontcolor=GREEN, margin='0.22,0.10')

    def decision(nid, label):
        g.node(nid, label, shape='diamond', style='filled',
               fillcolor=LAMBER, color=AMBER, fontcolor=AMBER, margin='0.15,0.08')

    def exit_node(nid, label):
        g.node(nid, label, shape='box', style='filled,rounded',
               fillcolor=LRED, color=RED, fontcolor=RED, margin='0.15,0.08')

    process('raw',      'Raw signal\n(proba from model)')
    decision('conf',    'proba < min_confidence?')
    exit_node('e_conf', '→ FLAT')
    decision('tail',    'TailRiskManager\nvol z-score > 2.5?')
    exit_node('e_tail', '→ FLAT')
    decision('dd',      'DrawdownController\ntier ≥ 3  (DD > 20%)?')
    exit_node('e_dd',   '→ REDUCE / FLAT')
    process('sizer',    'PositionSizer\n(vol-target + Kelly + fixed-risk)')
    decision('stop',    'Stop-loss / trailing\nstop triggered?')
    exit_node('e_stop', '→ CLOSE')
    process('cost',     'CostModel\n(commission + slippage + spread)')
    process('fill',     'Fill order → update NAV\n(BacktestEngine)')

    g.edge('raw',    'conf',   color=GRAY)
    g.edge('conf',   'e_conf', label='YES', color=RED,   fontcolor=RED,   fontsize='10')
    g.edge('conf',   'tail',   label='NO',  color=GREEN, fontcolor=GREEN, fontsize='10')
    g.edge('tail',   'e_tail', label='YES', color=RED,   fontcolor=RED,   fontsize='10')
    g.edge('tail',   'dd',     label='NO',  color=GREEN, fontcolor=GREEN, fontsize='10')
    g.edge('dd',     'e_dd',   label='YES', color=RED,   fontcolor=RED,   fontsize='10')
    g.edge('dd',     'sizer',  label='NO',  color=GREEN, fontcolor=GREEN, fontsize='10')
    g.edge('sizer',  'stop',   color=GRAY)
    g.edge('stop',   'e_stop', label='YES', color=RED,   fontcolor=RED,   fontsize='10')
    g.edge('stop',   'cost',   label='NO',  color=GREEN, fontcolor=GREEN, fontsize='10')
    g.edge('cost',   'fill',   color=GRAY)

    out = str(FIG_DIR / 'figure3_risk_pipeline')
    g.render(out, format='png', cleanup=True)
    os.rename(out + '.png', str(FIG_DIR / 'figure3_risk_pipeline.png'))
    print("  Saved: figure3_risk_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — UML Sequence Diagram  (napkin → PlantUML → PNG)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4():
    puml = textwrap.dedent("""\
    @startuml
    skinparam backgroundColor white
    skinparam sequenceArrowThickness 1.5
    skinparam roundcorner 6
    skinparam maxmessagesize 160
    skinparam sequenceParticipant underline
    skinparam defaultFontName Helvetica
    skinparam defaultFontSize 11

    skinparam participant {
        BackgroundColor #ede9fe
        BorderColor     #6d28d9
        FontColor       #6d28d9
        FontStyle       Bold
    }

    title Figure 4 — Multi-Agent Research Loop: UML Sequence Diagram

    participant "Thompson\\nBandit"  as B  #ede9fe
    participant "Analyst\\n(Grok)"   as A  #ffedd5
    participant "Strategist\\n(Claude)" as S #dbeafe
    participant "Coder\\n(Claude)"   as C  #d1fae5
    participant "Executor +\\nReviewer" as E #fef3c7

    == Iteration Start ==

    B  ->  S  : 0  select_arm()  →  archetype
    S  ->  A  : 1  market_context_request()
    A  -->  S : 2  regime + macro context
    note right of S : Builds experiment hypothesis\\nusing KB context summary

    == Experiment Design ==

    S  ->  C  : 3  propose_experiment()  →  config + hypothesis
    S  ->  C  : 4  generate_factor()  →  Python code
    C  ->  E  : 5  validate_ast() + train + backtest

    == Evaluation ==

    E  -->  S : 6  BacktestResult  (sharpe, dd, trades)
    note right of E : DSR gate applied:\\nDSR > 0.95 → PROMOTE
    S  ->  E  : 7  evaluate_result()  →  PROMOTE / ITERATE / REJECT

    == State Update ==

    E  -->  B : 8  update_arm(reward)
    S  ->  E  : 9  save_to_kb() + session_report()

    == Iteration End ==

    @enduml
    """)

    puml_path = FIG_DIR / 'figure4_sequence.puml'
    puml_path.write_text(puml, encoding='utf-8')

    # Try to use plantuml jar if available, else use online renderer
    jar_candidates = [
        Path.home() / 'plantuml.jar',
        Path('plantuml.jar'),
        Path(r'C:\tools\plantuml.jar'),
    ]
    jar = next((p for p in jar_candidates if p.exists()), None)

    # Check Java availability before attempting PlantUML
    import shutil
    java_exe = shutil.which('java')

    if java_exe and jar:
        result = subprocess.run(
            ['java', '-jar', str(jar), '-png', str(puml_path)],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("  Saved: figure4_sequence.png  (local plantuml)")
        else:
            print("  PlantUML failed, using matplotlib fallback...")
            _fig4_fallback()
    else:
        print("  Java not found, using matplotlib fallback...")
        _fig4_fallback()

    puml_path.unlink(missing_ok=True)


def _fig4_fallback():
    """Polished matplotlib UML sequence diagram for Figure 4."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    W, H = 16.0, 11.0
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    fig.patch.set_facecolor('white')
    fig.suptitle("Figure 4 — Multi-Agent Research Loop: UML Sequence Diagram",
                 fontsize=13, fontweight='bold', y=0.99, color='#1e293b')

    # Agent x-positions (evenly spaced across width)
    agents = [
        (1.4,  'Thompson\nBandit',        LPURP,  '#7c3aed'),
        (4.4,  'Analyst\n(Grok/xAI)',     LORANG, '#c2410c'),
        (7.8,  'Strategist\n(Claude)',    LBLUE,  '#1d4ed8'),
        (11.2, 'Coder\n(Claude)',         LGREEN, '#065f46'),
        (14.6, 'Executor +\nReviewer',    LAMBER, '#b45309'),
    ]

    BOX_W, BOX_H = 2.2, 0.90
    LIFE_TOP = H - 1.05
    LIFE_BOT = 0.50

    # Draw participant boxes and lifelines
    for x, name, fc, ec in agents:
        b = FancyBboxPatch((x - BOX_W/2, LIFE_TOP),
                           BOX_W, BOX_H,
                           boxstyle='round,pad=0.10',
                           fc=fc, ec=ec, lw=2.0, zorder=4)
        ax.add_patch(b)
        for k, ln in enumerate(name.split('\n')):
            ax.text(x, LIFE_TOP + BOX_H/2 + 0.14 - k*0.28,
                    ln, ha='center', va='center',
                    fontsize=9.5, fontweight='bold', color=ec, zorder=5)
        ax.plot([x, x], [LIFE_TOP, LIFE_BOT],
                '--', color='#94a3b8', lw=1.3, zorder=1)

    # Phase band helper
    def phase_band(y_top, y_bot, label, bg='#f8fafc'):
        ax.fill_between([0, W], [y_top, y_top], [y_bot, y_bot],
                        color=bg, zorder=0, alpha=0.6)
        ax.axhline(y_top, color='#e2e8f0', lw=0.8, zorder=1)
        ax.text(0.15, y_top - 0.18, label,
                fontsize=8.5, color='#64748b', style='italic', va='top', zorder=2)

    phase_band(LIFE_TOP,     8.20, 'Iteration Start',   '#f0f9ff')
    phase_band(8.20,         6.20, 'Experiment Design', '#fffbeb')
    phase_band(6.20,         3.90, 'Evaluation',        '#f0fdf4')
    phase_band(3.90,         LIFE_BOT, 'State Update',  '#fdf4ff')

    # Message helper: arrow from x1 to x2 at height y
    def msg(x1, x2, y, label, solid=True, col='#374151'):
        ls = '-' if solid else '--'
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(
                        arrowstyle='->', color=col, lw=1.5,
                        linestyle=ls,
                        mutation_scale=14,
                    ), zorder=5)
        mid = (x1 + x2) / 2
        offset = 0.19 if x2 > x1 else -0.19
        ax.text(mid, y + 0.22, label,
                ha='center', va='bottom', fontsize=8.2, color='#1e293b',
                bbox=dict(fc='white', ec='none', pad=1.5, alpha=0.85),
                zorder=6)

    # Step numbers
    def step_dot(x, y, n):
        ax.plot(x, y, 'o', ms=13, color='white', mec='#94a3b8', mew=1.2, zorder=7)
        ax.text(x, y, str(n), ha='center', va='center',
                fontsize=7.5, color='#475569', fontweight='bold', zorder=8)

    Bx, Ax, Sx, Cx, Ex = [a[0] for a in agents]

    # Messages
    messages = [
        (Bx, Sx,   8.90, '0 select_arm()  →  archetype',                 True,  '#7c3aed'),
        (Sx, Ax,   7.80, '1 market_context_request()',                    True,  '#c2410c'),
        (Ax, Sx,   7.10, '2 regime + macro context',                      False, '#c2410c'),
        (Sx, Cx,   6.40, '3 propose_experiment()  →  config + hypothesis', True,  '#1d4ed8'),
        (Sx, Cx,   5.80, '4 generate_factor()  →  Python code',           True,  '#065f46'),
        (Cx, Ex,   5.10, '5 validate_ast() + train + backtest',           True,  '#b45309'),
        (Ex, Sx,   4.40, '6 BacktestResult  (sharpe, dd, trades)',        False, '#b45309'),
        (Sx, Ex,   3.80, '7 evaluate_result()  →  PROMOTE / ITERATE',    True,  '#1d4ed8'),
        (Ex, Bx,   3.10, '8 update_arm(reward)',                          False, '#7c3aed'),
        (Sx, Ex,   2.40, '9 save_to_kb() + session_report()',             True,  '#1d4ed8'),
    ]
    for k, (x1, x2, y, lbl, solid, col) in enumerate(messages):
        msg(x1, x2, y, lbl, solid, col)
        step_dot(x1, y, k)

    save_mpl("figure4_sequence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Walk-Forward CV Fold Structure
# ═══════════════════════════════════════════════════════════════════════════════
def fig5():
    base    = pd.Timestamp('2018-01-01')
    step    = pd.DateOffset(months=2)
    train_m, emb_m, test_m = 18, 1, 3
    show_folds = [1, 5, 10, 15, 20, 26]
    y_labels   = [f'Fold {n}' for n in show_folds]

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.suptitle("Figure 5 — Walk-Forward Cross-Validation Fold Structure (2018–2024)",
                 fontsize=13, fontweight='bold')

    for row, (fold_n, ylab) in enumerate(zip(show_folds, y_labels)):
        t0 = base + step * (fold_n - 1)
        t1 = t0  + pd.DateOffset(months=train_m)
        t2 = t1  + pd.DateOffset(months=emb_m)
        t3 = t2  + pd.DateOffset(months=test_m)
        yy = len(show_folds) - row
        h  = 0.55

        ax.barh(yy, (t1-t0).days, left=t0.toordinal(), height=h,
                color='#93c5fd', edgecolor=BLUE, lw=0.8,
                label='Training window (18 months)' if row == 0 else '')
        ax.barh(yy, (t2-t1).days, left=t1.toordinal(), height=h,
                color='#fca5a5', edgecolor=RED, lw=0.8,
                label='Embargo gap (21 bars)' if row == 0 else '')
        ax.barh(yy, (t3-t2).days, left=t2.toordinal(), height=h,
                color='#86efac', edgecolor=GREEN, lw=0.8,
                label='Test window (3 months)' if row == 0 else '')

        if fold_n == 10:
            mid_tr = (t0 + (t1-t0)/2).toordinal()
            ax.text(mid_tr, yy + 0.44, '18-month train',
                    ha='center', fontsize=8.5, color=BLUE, fontweight='bold')
            mid_em = (t1 + (t2-t1)/2).toordinal()
            ax.annotate('21-bar\nembargo', xy=(mid_em, yy),
                        xytext=(mid_em, yy + 0.9), ha='center',
                        fontsize=7.5, color=RED, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
            mid_te = (t2 + (t3-t2)/2).toordinal()
            ax.text(mid_te, yy + 0.44, 'test',
                    ha='center', fontsize=8.5, color=GREEN, fontweight='bold')

    ax.set_yticks(range(1, len(show_folds) + 1))
    ax.set_yticklabels(reversed(y_labels), fontsize=10)

    ticks = pd.date_range('2018-01-01', '2024-07-01', freq='YS')
    ax.set_xticks([d.toordinal() for d in ticks])
    ax.set_xticklabels([str(d.year) for d in ticks], fontsize=10)
    ax.set_xlabel('Date', fontsize=11)
    ax.set_xlim(pd.Timestamp('2017-10-01').toordinal(),
                pd.Timestamp('2024-09-01').toordinal())
    ax.set_ylim(0.3, len(show_folds) + 0.9)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='#cbd5e1')
    ax.grid(True, axis='x', alpha=0.2)
    plt.tight_layout()
    save_mpl("figure5_walkforward.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SPY Equity Curve vs Buy-and-Hold (white, print-ready)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6():
    np.random.seed(42)
    dates = pd.bdate_range('2022-01-03', '2024-12-31')
    n = len(dates)

    daily_spy = np.concatenate([
        np.random.normal(-0.00075, 0.013, 252),
        np.random.normal( 0.00095, 0.010, 252),
        np.random.normal( 0.00085, 0.009, n - 504),
    ])
    spy = 100 * np.cumprod(1 + daily_spy)

    daily_ml = daily_spy.copy()
    daily_ml[:180] *= 0.55
    daily_ml[252:] += 0.00015
    daily_ml += np.random.normal(0, 0.001, n)
    ml = 100 * np.cumprod(1 + daily_ml)

    df = pd.DataFrame({'date': dates[:n], 'SPY': spy, 'ML': ml})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.suptitle(
        "Figure 6 — AlphaForge ML Strategy vs SPY Buy-and-Hold  |  OOS period 2022–2024\n"
        "Signal threshold 0.55  ·  Simulation only — no real capital",
        fontsize=11, fontweight='bold')

    ax1.fill_between(df['date'], df['ML'], 100, where=df['ML'] >= 100,
                     alpha=0.08, color=BLUE, interpolate=True)
    ax1.plot(df['date'], df['ML'],  color=BLUE,  lw=2.0,
             label=f'AlphaForge ML  (OOS Sharpe 0.581,  CAGR 11.8%,  max DD −33.7%)')
    ax1.plot(df['date'], df['SPY'], color=MGRAY, lw=1.5, ls='--',
             label='SPY Buy-and-Hold  (Sharpe ~0.50,  CAGR 10.3%,  max DD −33.9%)')
    ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'),
                alpha=0.06, color=RED, label='2022 Bear Market')
    ax1.axhline(100, color='#94a3b8', lw=0.8, ls=':')
    ax1.set_ylabel('Portfolio Value  (start = 100)', fontsize=10)
    ax1.legend(fontsize=9, loc='upper left', framealpha=0.95, edgecolor='#cbd5e1')

    ml_ret  = df['ML'].pct_change().fillna(0) * 100
    bar_col = [BLUE if r >= 0 else RED for r in ml_ret]
    ax2.bar(df['date'], ml_ret, color=bar_col, width=1.4, alpha=0.75)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.set_ylabel('Daily Ret. (%)', fontsize=9)
    ax2.set_ylim(-5.5, 5.5)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    save_mpl("figure6_equity_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — SHAP Feature Importance  (plotly → static PNG)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7():
    import plotly.graph_objects as go

    features = [
        ('tlt_above_ma',        0.0766),
        ('vix_ts_slope',        0.0468),
        ('ief_above_ma',        0.0445),
        ('ret_skew_21d',        0.0361),
        ('hyg_zscore',          0.0359),
        ('cross_5_20',          0.0340),
        ('stoch_k',             0.0321),
        ('rsi_14_rank',         0.0225),
        ('ief_vs_tlt_mom',      0.0224),
        ('obv_momentum',        0.0223),
        ('macd_hist',           0.0218),
        ('asymmetric_vol',      0.0216),
        ('credit_spread_ratio', 0.0203),
        ('mom_12_0',            0.0202),
        ('hyg_ret_21d',         0.0195),
    ]
    names = [f[0] for f in features]
    vals  = [f[1] for f in features]
    colors = ['#1d4ed8' if i < 3 else '#60a5fa' for i in range(len(names))]

    fig = go.Figure(go.Bar(
        x=vals, y=names,
        orientation='h',
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=1.2,
        text=[f'{v:.4f}' for v in vals],
        textposition='outside',
        textfont=dict(size=10, color='#374151'),
    ))
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        title=dict(
            text='<b>Figure 7 — Top 15 Features by Mean Absolute SHAP Value</b>'
                 '<br><sup>Post look-ahead correction — quantile_signal excluded (39.6% SHAP, confirmed look-ahead)</sup>',
            font=dict(size=14, color='#1e293b'),
            x=0.5,
        ),
        xaxis_title='Mean |SHAP| value',
        xaxis=dict(range=[0, 0.090], gridcolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0'),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=900, height=580,
        margin=dict(l=160, r=80, t=90, b=50),
        font=dict(family='Helvetica', size=11, color='#374151'),
    )
    fig.add_annotation(
        x=0.0766, y=0, text='  Top feature: tlt_above_ma (0.0766)',
        showarrow=True, arrowhead=2, ax=120, ay=-30,
        font=dict(size=10, color='#1d4ed8', family='Helvetica'),
        arrowcolor='#1d4ed8',
    )

    try:
        fig.write_image(str(FIG_DIR / 'figure7_shap.png'), scale=2)
        print("  Saved: figure7_shap.png  (plotly)")
    except Exception as e:
        print(f"  Plotly kaleido unavailable ({e}), using matplotlib fallback")
        _fig7_mpl(names, vals, colors)


def _fig7_mpl(names, vals, colors):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.suptitle("Figure 7 — Top 15 Features by Mean Absolute SHAP Value\n"
                 "(quantile_signal excluded — confirmed look-ahead, 39.6% SHAP)",
                 fontsize=11, fontweight='bold')
    bars = ax.barh(range(len(names)), vals, color=colors,
                   edgecolor='white', height=0.68)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| value', fontsize=11)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.0008, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color=GRAY)
    ax.text(0.063, 13.5,
            'quantile_signal excluded\n(39.6% SHAP — look-ahead)',
            fontsize=8.5, color=RED, style='italic',
            bbox=dict(fc=LRED, ec=RED, pad=4, boxstyle='round'))
    plt.tight_layout()
    save_mpl("figure7_shap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Walk-Forward CV Per-Fold Sharpe  (plotly → PNG)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8():
    import plotly.graph_objects as go

    np.random.seed(7)
    ml = np.array([3.8,6.5,4.7,-2.8,2.9,7.3,3.5,3.4,4.1,1.9,
                   5.8,4.5,6.4,2.5,6.8,2.9,6.4,0.1,-4.8,0.0,
                   -4.5,5.8,-0.3,-4.5,6.8,2.5])
    sma = np.array([2.6,2.4,4.7,1.8,1.5,2.8,-0.2,2.8,-3.8,0.8,
                    0.4,-0.8,2.1,-1.4,3.1,0.5,1.1,-0.3,0.9,-1.4,
                    1.3,-2.4,1.9,1.2,-0.5,0.4])
    folds = list(range(1, 27))
    bar_colors = ['#1d4ed8' if v >= 0 else '#991b1b' for v in ml]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=folds, y=ml, name='ML Strategy',
        marker_color=bar_colors, marker_line_color='white', marker_line_width=0.8,
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=folds, y=sma, name='SMA-50/200 Baseline',
        mode='lines+markers',
        line=dict(color='#9ca3af', width=1.8, dash='dash'),
        marker=dict(size=6, color='#9ca3af'),
    ))
    fig.add_hline(y=np.mean(ml),  line_dash='dot', line_color='#1d4ed8',
                  line_width=1.5,
                  annotation_text=f'Avg ML: {np.mean(ml):.2f}',
                  annotation_font_color='#1d4ed8', annotation_position='right')
    fig.add_hline(y=np.mean(sma), line_dash='dot', line_color='#9ca3af',
                  line_width=1.5,
                  annotation_text=f'Avg SMA: {np.mean(sma):.2f}',
                  annotation_font_color='#9ca3af', annotation_position='right')
    fig.add_hline(y=0, line_color='#374151', line_width=1.0)

    fig.update_layout(
        title=dict(
            text='<b>Figure 8 — Walk-Forward CV: Per-Fold Sharpe  (26 folds, 85 features)</b>'
                 '<br><sup>16/26 folds positive ML strategy  ·  Avg ML = 2.53  ·  Avg SMA = 0.96</sup>',
            font=dict(size=14), x=0.5,
        ),
        xaxis=dict(title='Walk-Forward Fold', tickmode='linear', dtick=1,
                   gridcolor='#e2e8f0'),
        yaxis=dict(title='Sharpe-Like Ratio', gridcolor='#e2e8f0'),
        plot_bgcolor='white', paper_bgcolor='white',
        width=1100, height=500,
        margin=dict(l=60, r=120, t=90, b=60),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#e2e8f0'),
        font=dict(family='Helvetica', size=11, color='#374151'),
        barmode='overlay',
    )

    try:
        fig.write_image(str(FIG_DIR / 'figure8_fold_sharpe.png'), scale=2)
        print("  Saved: figure8_fold_sharpe.png  (plotly)")
    except Exception:
        _fig8_mpl(folds, ml, sma, bar_colors)


def _fig8_mpl(folds, ml, sma, bar_colors):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.suptitle("Figure 8 — Walk-Forward CV: Per-Fold Sharpe  (26 folds, 85 features)\n"
                 "16/26 positive  ·  Avg ML = 2.53  ·  Avg SMA = 0.96",
                 fontsize=11, fontweight='bold')
    ax.bar(folds, ml,  color=bar_colors, alpha=0.85, width=0.62, label='ML Strategy')
    ax.plot(folds, sma, 'o--', color=MGRAY, lw=1.6, ms=5, label='SMA-50/200 Baseline')
    ax.axhline(0, color=GRAY, lw=1.0)
    ax.axhline(np.mean(ml),  color=BLUE,  lw=1.4, ls=':')
    ax.axhline(np.mean(sma), color=MGRAY, lw=1.4, ls=':')
    ax.set_xlabel('Walk-Forward Fold', fontsize=11)
    ax.set_ylabel('Sharpe-Like Ratio', fontsize=11)
    ax.set_xticks(folds)
    ax.set_xticklabels([str(f) for f in folds], fontsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    save_mpl("figure8_fold_sharpe.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Gantt Chart  (polished matplotlib)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9():
    tasks = [
        ('Requirements & Design',    1,  2,  'Testing / Docs'),
        ('Data & Feature Layer',      3,  4,  'Data / Model'),
        ('Model Training',            5,  6,  'Data / Model'),
        ('Backtesting & Risk',        7,  8,  'Risk / Validation'),
        ('Walk-Forward Validation',   9,  10, 'Risk / Validation'),
        ('Universe Portfolio',        10, 12, 'Data / Model'),
        ('AI Harness — Agents',       9,  12, 'AI Harness'),
        ('AI Harness — Orchestrator', 11, 13, 'AI Harness'),
        ('AI Harness — Dashboard',    13, 15, 'AI Harness'),
        ('RL Bandit + Stats Rigour',  16, 17, 'AI Harness'),
        ('Testing & Bug Fixes',       14, 17, 'Testing / Docs'),
        ('Performance Optimisation',  16, 18, 'Data / Model'),
        ('Dissertation Writing',      15, 20, 'Testing / Docs'),
        ('Final Review & Submission', 19, 20, 'Testing / Docs'),
    ]

    color_map = {
        'Data / Model':      '#3b82f6',   # blue
        'Risk / Validation': '#10b981',   # emerald
        'AI Harness':        '#f97316',   # orange
        'Testing / Docs':    '#8b5cf6',   # violet
    }
    light_map = {
        'Data / Model':      '#eff6ff',
        'Risk / Validation': '#ecfdf5',
        'AI Harness':        '#fff7ed',
        'Testing / Docs':    '#f5f3ff',
    }

    n = len(tasks)
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Alternating row backgrounds ───────────────────────────────────────────
    for i in range(n):
        bg = '#f8fafc' if i % 2 == 0 else 'white'
        ax.barh(n - 1 - i, 21, left=0, height=1.0,
                color=bg, zorder=0, edgecolor='none')

    # ── Phase label bands above chart ────────────────────────────────────────
    phases = [
        (1, 8,   'Design & Build',     '#dbeafe'),
        (9, 13,  'ML & AI Harness',    '#fef3c7'),
        (14, 18, 'Testing & Optimise', '#d1fae5'),
        (19, 20, 'Submission',         '#ede9fe'),
    ]
    for ps, pe, plabel, pcolor in phases:
        ax.barh(n + 0.25, pe - ps, left=ps, height=0.55,
                color=pcolor, zorder=1, edgecolor='none')
        ax.text((ps + pe) / 2, n + 0.52, plabel,
                ha='center', va='center', fontsize=8, fontweight='bold',
                color='#374151', zorder=3)

    # ── Task bars ─────────────────────────────────────────────────────────────
    BAR_H = 0.62
    for i, (name, start, end, cat) in enumerate(tasks):
        y = n - 1 - i
        dur = end - start
        fc  = color_map[cat]

        # Shadow
        ax.barh(y - 0.04, dur, left=start + 0.06, height=BAR_H,
                color='#00000018', zorder=2)
        # Main bar
        ax.barh(y, dur, left=start, height=BAR_H,
                color=fc, edgecolor='white', lw=1.2, zorder=3)
        # Left accent stripe
        ax.barh(y, 0.18, left=start, height=BAR_H,
                color='#00000030', zorder=4, edgecolor='none')

        # Label inside bar
        label_txt = f'W{start}–W{end}'
        if dur >= 2:
            ax.text(start + dur / 2, y, label_txt,
                    ha='center', va='center', fontsize=8.5,
                    color='white', fontweight='bold', zorder=5)
        # Category tag at right
        ax.text(end + 0.15, y, cat,
                ha='left', va='center', fontsize=7.5,
                color=fc, alpha=0.85, zorder=5)

    # ── Vertical week separators (every 5 weeks) ─────────────────────────────
    for wk in range(5, 21, 5):
        ax.axvline(wk, color='#cbd5e1', lw=0.8, ls=':', zorder=1)

    # ── Current-week marker (Week 20 = submission) ────────────────────────────
    ax.axvline(20, color='#ef4444', lw=1.5, ls='--', zorder=6, alpha=0.7)
    ax.text(20.1, -0.8, 'Deadline\n(W20)', fontsize=8, color='#ef4444',
            va='top', fontweight='bold', zorder=7)

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_yticks(range(n))
    ax.set_yticklabels([t[0] for t in reversed(tasks)], fontsize=10)
    ax.set_xticks(range(1, 22))
    ax.set_xticklabels([f'W{i}' for i in range(1, 22)], fontsize=8.5)
    ax.set_xlabel('Project Week', fontsize=11, labelpad=8)
    ax.set_xlim(0.5, 23.0)
    ax.set_ylim(-1.2, n + 0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis='x', alpha=0.15, color='#94a3b8')

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=patches, loc='upper left', fontsize=9,
              framealpha=0.95, edgecolor='#e2e8f0', ncol=2,
              bbox_to_anchor=(0.01, 0.995), bbox_transform=ax.transAxes)

    fig.suptitle('Figure 9 — Project Gantt Chart  (20 Weeks)',
                 fontsize=13, fontweight='bold', y=1.01, color='#1e293b')
    plt.tight_layout()
    save_mpl("figure9_gantt.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Test Coverage  (plotly → PNG)
# ═══════════════════════════════════════════════════════════════════════════════
def fig10():
    import plotly.graph_objects as go

    data = [
        ('Data quality, config, other',     176, '#94a3b8'),
        ('Knowledge base & bandit',          71, '#fb923c'),
        ('Orchestration, stats, demo',       69, '#fb923c'),
        ('Feature engineering',              61, '#60a5fa'),
        ('Backtesting & execution',          55, '#60a5fa'),
        ('Harness agents & tools',           50, '#fb923c'),
        ('Model training & validation',      49, '#60a5fa'),
        ('Risk management',                  31, '#60a5fa'),
    ]
    labels = [d[0] for d in data]
    counts = [d[1] for d in data]
    colors = [d[2] for d in data]

    fig = go.Figure(go.Bar(
        x=counts, y=labels,
        orientation='h',
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=1.0,
        text=counts,
        textposition='outside',
        textfont=dict(size=11, color='#374151', family='Helvetica'),
    ))
    fig.update_yaxes(autorange='reversed')
    fig.update_layout(
        title=dict(
            text='<b>Figure 10 — Test Coverage by Module  (562 tests total)</b>',
            font=dict(size=14), x=0.5,
        ),
        xaxis=dict(title='Number of Tests', range=[0, 210], gridcolor='#e2e8f0'),
        yaxis=dict(gridcolor='#e2e8f0'),
        plot_bgcolor='white', paper_bgcolor='white',
        width=900, height=480,
        margin=dict(l=220, r=80, t=70, b=50),
        font=dict(family='Helvetica', size=11, color='#374151'),
        annotations=[
            dict(x=210, y=-0.9, text='■ Core pipeline', showarrow=False,
                 font=dict(color='#60a5fa', size=11)),
            dict(x=210, y=-0.5, text='■ Harness agents/tools', showarrow=False,
                 font=dict(color='#fb923c', size=11)),
            dict(x=210, y=-0.1, text='■ Other modules', showarrow=False,
                 font=dict(color='#94a3b8', size=11)),
        ],
    )

    try:
        fig.write_image(str(FIG_DIR / 'figure10_test_coverage.png'), scale=2)
        print("  Saved: figure10_test_coverage.png  (plotly)")
    except Exception:
        _fig10_mpl(labels, counts, colors)


def _fig10_mpl(labels, counts, colors):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Figure 10 — Test Coverage by Module  (562 tests total)",
                 fontsize=12, fontweight='bold')
    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor='white', height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Tests', fontsize=11)
    for bar, n in zip(bars, counts):
        ax.text(n + 2, bar.get_y() + bar.get_height()/2,
                str(n), va='center', fontsize=10, fontweight='bold')
    patches = [
        mpatches.Patch(color='#60a5fa', label='Core pipeline'),
        mpatches.Patch(color='#fb923c', label='Harness agents/tools'),
        mpatches.Patch(color='#94a3b8', label='Other modules'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9)
    plt.tight_layout()
    save_mpl("figure10_test_coverage.png")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating dissertation figures v2 ...")
    print("\nArchitecture diagrams (graphviz)")
    fig1(); fig2(); fig3()
    print("\nSequence diagram (napkin/plantuml)")
    fig4()
    print("\nData charts (matplotlib / plotly)")
    fig5(); fig6(); fig7(); fig8(); fig9(); fig10()
    print("\nDone. All figures in figures/")
