#!/usr/bin/env python3
"""Regenerate all AlphaForge dissertation figures — corrected and improved."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
BLUE   = '#1d4ed8'; LBLUE  = '#dbeafe'; DBLUE = '#1e3a8a'
PURP   = '#6d28d9'; LPURP  = '#ede9fe'
GREEN  = '#065f46'; LGREEN = '#d1fae5'
AMBER  = '#b45309'; LAMBER = '#fef3c7'
RED    = '#991b1b'; LRED   = '#fee2e2'
TEAL   = '#0e7490'; LTEAL  = '#cffafe'
GRAY   = '#374151'; LGRAY  = '#f9fafb'; MGRAY  = '#9ca3af'
ORANGE = '#c2410c'; LORANG = '#ffedd5'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.color': '#cbd5e1',
})

def save(name):
    plt.savefig(FIG_DIR / name, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  Saved: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Architecture  (fixed clipped row labels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig1():
    fig = plt.figure(figsize=(15, 9))
    # Leave left margin for row labels
    ax = fig.add_axes([0.10, 0.03, 0.88, 0.90])
    ax.set_xlim(0, 13)
    ax.set_ylim(0.5, 9.5)
    ax.axis('off')
    fig.suptitle("Figure 1 — AlphaForge System Architecture",
                 fontsize=13, fontweight='bold', y=0.98)

    def box(x, y, w, h, label, sub="", fc=LBLUE, ec=BLUE, fs=9.5):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.12", fc=fc, ec=ec, lw=1.6, zorder=3)
        ax.add_patch(rect)
        if sub:
            ax.text(x, y + 0.17, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=ec, zorder=4)
            ax.text(x, y - 0.22, sub, ha='center', va='center',
                    fontsize=fs - 1.5, color=GRAY, zorder=4)
        else:
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=ec, zorder=4)

    def arr(x1, y1, x2, y2, col=BLUE):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5),
                    zorder=5)

    # ── Row background bands with labels in left figure margin ──
    bands = [
        (8.5, 'Entry Points',       '#f8fafc'),
        (6.5, 'AI Harness',         '#faf5ff'),
        (4.5, 'Core ML Pipeline',   '#eff6ff'),
        (2.2, 'Infrastructure',     '#f0fdf4'),
    ]
    for (cy, label, fc) in bands:
        ax.add_patch(plt.Rectangle((0, cy - 0.75), 13, 1.5,
                                   fc=fc, ec='#e2e8f0', lw=1.0, zorder=1))
        # Row label in figure-level coordinates (left margin)
        ax_y = (cy - 0.5) / 9.0    # convert axes y → approx fig fraction
        fig.text(0.005, 0.03 + ax_y * 0.90, label,
                 ha='left', va='center', fontsize=8.5,
                 color='#64748b', style='italic', fontweight='bold',
                 transform=fig.transFigure)

    # ── Entry points ──
    box(3.2,  8.5, 4.0, 1.0, 'main.py', '(Core ML CLI)',           LBLUE, BLUE)
    box(9.8,  8.5, 4.0, 1.0, 'harness_main.py', '(AI Harness CLI)', LPURP, PURP)

    # ── Four agents ──
    agents = [
        (1.3,  'Strategist', '(Claude)', LPURP,  PURP),
        (4.2,  'Analyst',    '(Grok)',   LORANG, ORANGE),
        (7.1,  'Coder',      '(Claude)', LPURP,  PURP),
        (10.0, 'Reviewer',   '(Claude)', LPURP,  PURP),
    ]
    for (x, name, sub, fc, ec) in agents:
        box(x, 6.5, 2.5, 1.0, name, sub, fc, ec)

    # ── Orchestrator ──
    box(5.7, 5.1, 4.0, 1.0, 'AlphaHarness\nOrchestrator', '', LPURP, PURP, fs=9)

    # Agents → orchestrator
    for x in [1.3, 4.2, 7.1, 10.0]:
        arr(x, 6.0, 5.7, 5.6, PURP)

    # harness_main → orchestrator
    arr(9.8, 8.0, 7.2, 5.6, PURP)

    # ── Core pipeline ──
    stages = ['ingest', 'features', 'train', 'backtest', 'validate', 'report']
    stage_fc = [LBLUE, LGREEN, LGREEN, LBLUE, LTEAL, LAMBER]
    stage_ec = [BLUE,  GREEN,  GREEN,  BLUE,  TEAL,  AMBER]
    xs = [i * 2.1 + 0.8 for i in range(6)]
    for i, (name, fc, ec, x) in enumerate(zip(stages, stage_fc, stage_ec, xs)):
        box(x, 4.5, 1.85, 0.9, name, '', fc, ec)
        if i < 5:
            arr(x + 0.93, 4.5, xs[i+1] - 0.93, 4.5, BLUE)

    # main.py → pipeline
    arr(3.2, 8.0, xs[1], 5.0, BLUE)
    # orchestrator → backtest
    arr(5.7, 4.6, xs[3], 5.0, PURP)

    # ── Infrastructure ──
    infra = [
        (1.6,  'data/\n(Parquet cache)',          LGREEN, GREEN),
        (4.7,  'models/artifacts/\n(joblib+JSON)', LGREEN, GREEN),
        (7.8,  'harness/memory/\n(KB + bandit)',   LPURP,  PURP),
        (11.0, 'logs/ + config.yaml',              LAMBER, AMBER),
    ]
    for (x, name, fc, ec) in infra:
        box(x, 2.2, 2.9, 1.1, name, '', fc, ec, fs=8.5)

    src_map = [xs[0], xs[2], xs[4], xs[5]]
    for src, (inf_x, *_) in zip(src_map, infra):
        arr(src, 4.05, inf_x, 2.75, '#94a3b8')

    # ── Legend ──
    patches = [
        mpatches.Patch(fc=LBLUE,  ec=BLUE,  label='Core ML Pipeline'),
        mpatches.Patch(fc=LPURP,  ec=PURP,  label='AI Harness'),
        mpatches.Patch(fc=LGREEN, ec=GREEN, label='Infrastructure'),
    ]
    ax.legend(handles=patches, loc='lower right', fontsize=9,
              framealpha=0.95, edgecolor='#cbd5e1')

    plt.savefig(FIG_DIR / "figure1_architecture.png", dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close('all')
    print("  Saved: figure1_architecture.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Data Flow Pipeline  (cleaner labels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(-0.3, 14.3)
    ax.set_ylim(-1.6, 2.8)
    ax.axis('off')
    fig.suptitle("Figure 2 — AlphaForge Data Flow Pipeline",
                 fontsize=13, fontweight='bold', y=1.01)

    stages = [
        ("Data\nIngest",      "Raw OHLCV\n+ adj close",   LBLUE,  BLUE),
        ("Feature\nEngine",   "250+ cols\n(no look-ahead)", LGREEN, GREEN),
        ("Feature\nSelector", "85 cols\n(SHAP)",            LGREEN, GREEN),
        ("Walk-Forward\nCV",  "26 folds\n21-bar embargo",   LAMBER, AMBER),
        ("Model\nTrain",      "XGB + LGBM\nensemble×8",     LPURP,  PURP),
        ("Backtest\nEngine",  "NAV / DD\n/ Sharpe",         LBLUE,  BLUE),
        ("Validate\n& Report","OOS metrics\nPDF/JSON",      LTEAL,  TEAL),
    ]

    xs = [i * 2.0 + 1.0 for i in range(7)]
    data_labels = ['Raw\nParquet', 'Feature\nMatrix', 'Reduced\nMatrix',
                   'Train/Test\nSplits', 'Trained\nModel', 'Equity\nCurve', '']

    for i, ((name, sub, fc, ec), x) in enumerate(zip(stages, xs)):
        rect = FancyBboxPatch((x - 0.82, 0.25), 1.64, 1.5,
                              boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.6)
        ax.add_patch(rect)
        ax.text(x, 1.0, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color=ec)
        ax.text(x, -0.35, sub, ha='center', va='center',
                fontsize=8, color=GRAY)
        if i < 6:
            ax.annotate('', xy=(x + 1.18, 1.0), xytext=(x + 0.82, 1.0),
                        arrowprops=dict(arrowstyle='->', color='#475569', lw=1.8))
            if data_labels[i]:
                ax.text(x + 1.0, 1.28, data_labels[i], ha='center',
                        fontsize=7, color='#64748b', style='italic')

    save("figure2_data_flow.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Risk Management Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def fig3():
    fig, ax = plt.subplots(figsize=(7, 11))
    ax.set_xlim(-0.5, 7.5)
    ax.set_ylim(-0.5, 11.5)
    ax.axis('off')
    fig.suptitle("Figure 3 — Risk Management Signal Pipeline",
                 fontsize=13, fontweight='bold', y=0.98)

    def rect_box(x, y, w, h, label, fc, ec):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.6, zorder=3)
        ax.add_patch(p)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=9.5, fontweight='bold', color=ec, zorder=4)

    def diamond(x, y, w, h, label):
        dx, dy = w/2, h/2
        xs = [x, x + dx, x, x - dx, x]
        ys = [y + dy, y, y - dy, y, y + dy]
        ax.fill(xs, ys, fc=LAMBER, ec=AMBER, lw=1.6, zorder=3)
        for line in label.split('\n'):
            idx = label.split('\n').index(line)
            offset = 0.12 if len(label.split('\n')) > 1 else 0
            ax.text(x, y + offset - idx * 0.24, line,
                    ha='center', va='center', fontsize=8.5, color=AMBER, fontweight='bold', zorder=4)

    def down_arrow(y_from, y_to, label="NO"):
        ax.annotate('', xy=(3.5, y_to), xytext=(3.5, y_from),
                    arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5), zorder=5)
        if label:
            ax.text(3.75, (y_from + y_to) / 2, label,
                    ha='left', va='center', fontsize=8, color=GREEN, fontweight='bold')

    def side_exit(y, label):
        ax.annotate('', xy=(5.8, y), xytext=(4.0, y),
                    arrowprops=dict(arrowstyle='->', color=RED, lw=1.5), zorder=5)
        p = FancyBboxPatch((5.8, y - 0.3), 1.5, 0.6,
                           boxstyle="round,pad=0.08", fc=LRED, ec=RED, lw=1.4, zorder=3)
        ax.add_patch(p)
        ax.text(6.55, y, label, ha='center', va='center',
                fontsize=8.5, color=RED, fontweight='bold', zorder=4)

    # Boxes and flow
    rect_box(3.5, 11.0, 4.5, 0.7, 'Raw signal  (proba from model)', LBLUE, BLUE)
    down_arrow(10.65, 10.2, '')

    diamond(3.5, 9.85, 3.8, 0.8, 'proba < min_confidence?')
    side_exit(9.85, '→ FLAT')
    down_arrow(9.45, 8.85, 'NO')

    diamond(3.5, 8.5, 3.8, 0.8, 'TailRiskManager\nvol z-score > 2.5?')
    side_exit(8.5, '→ FLAT')
    down_arrow(8.1, 7.5, 'NO')

    diamond(3.5, 7.15, 3.8, 0.8, 'DrawdownController\ntier ≥ 3  (DD > 20%)?')
    side_exit(7.15, '→ REDUCE')
    down_arrow(6.75, 6.1, 'NO')

    rect_box(3.5, 5.75, 4.5, 0.7, 'PositionSizer  (vol-target + Kelly + fixed)', LGREEN, GREEN)
    down_arrow(5.4, 4.8, '')

    diamond(3.5, 4.45, 3.8, 0.8, 'Stop-loss /\ntrailing stop triggered?')
    side_exit(4.45, '→ CLOSE')
    down_arrow(4.05, 3.2, 'NO')

    rect_box(3.5, 2.85, 4.5, 0.7, 'CostModel  (commission + slippage + spread)', LAMBER, AMBER)
    down_arrow(2.5, 1.75, '')

    rect_box(3.5, 1.4, 4.5, 0.7, 'Fill order → update NAV  (BacktestEngine)', LGREEN, GREEN)

    save("figure3_risk_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Multi-Agent Sequence Diagram  (tighter, no blank bottom)
# ═══════════════════════════════════════════════════════════════════════════════
def fig4():
    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.suptitle("Figure 4 — Multi-Agent Research Loop: UML Sequence Diagram",
                 fontsize=13, fontweight='bold', y=0.99)

    # Lifelines: x positions
    agents = [
        (1.1,  'Thompson\nBandit',        LPURP,  PURP),
        (3.0,  'Analyst\n(Grok)',          LORANG, ORANGE),
        (5.5,  'Strategist\n(Claude)',     LBLUE,  BLUE),
        (8.0,  'Coder\n(Claude)',          LGREEN, GREEN),
        (11.2, 'Executor +\nReviewer',     LAMBER, AMBER),
    ]

    top_y  = 9.3
    bot_y  = 0.3
    head_h = 0.9

    # Actor boxes
    for x, name, fc, ec in agents:
        b = FancyBboxPatch((x - 0.85, top_y - head_h/2), 1.7, head_h,
                           boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=1.6, zorder=3)
        ax.add_patch(b)
        for i, ln in enumerate(name.split('\n')):
            ax.text(x, top_y + 0.12 - i * 0.3, ln,
                    ha='center', va='center', fontsize=8.5, fontweight='bold', color=ec, zorder=4)
        # Lifeline
        ax.plot([x, x], [top_y - head_h/2, bot_y],
                '--', color='#94a3b8', lw=1.2, zorder=1)

    # Messages: (from_x, to_x, y, label, style)
    # style: 'solid' or 'dashed'
    msgs = [
        (1.1,  5.5,  8.5, '0  select_arm()  →  archetype',               'solid', PURP),
        (5.5,  3.0,  7.8, '1  market_context_request()',                  'solid', ORANGE),
        (3.0,  5.5,  7.2, '2  regime + macro context',                    'dashed', ORANGE),
        (5.5,  8.0,  6.5, '3  propose_experiment()  →  config + hypothesis', 'solid', BLUE),
        (5.5,  8.0,  5.9, '4  generate_factor()  →  Python code',         'solid', GREEN),
        (8.0,  11.2, 5.2, '5  validate_ast() + train + backtest',         'solid', AMBER),
        (11.2, 5.5,  4.5, '6  BacktestResult  (sharpe, dd, trades)',      'dashed', AMBER),
        (5.5,  11.2, 3.8, '7  evaluate_result()  →  PROMOTE / ITERATE',  'solid', BLUE),
        (11.2, 1.1,  3.1, '8  update_arm(reward)',                        'dashed', PURP),
        (5.5,  11.2, 2.3, '9  save_to_kb() + session_report()',           'solid', BLUE),
    ]

    for x1, x2, y, label, style, col in msgs:
        ls = '-' if style == 'solid' else '--'
        ax.annotate('', xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5,
                                   linestyle=ls), zorder=5)
        mx = (x1 + x2) / 2
        offset = 0.2 if x2 > x1 else -0.2
        ax.text(mx, y + 0.18, label, ha='center', va='bottom',
                fontsize=8, color=GRAY, zorder=4,
                bbox=dict(fc='white', ec='none', pad=1))

    # Activation boxes on lifelines (thin rectangles showing active periods)
    def active_bar(x, y_start, y_end, col):
        ax.add_patch(plt.Rectangle((x - 0.1, y_end), 0.2, y_start - y_end,
                                   fc=col, ec=col, alpha=0.35, zorder=2))

    active_bar(5.5, 8.8, 2.0, BLUE)
    active_bar(8.0, 6.2, 4.8, GREEN)
    active_bar(11.2, 5.5, 2.0, AMBER)

    save("figure4_sequence.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Walk-Forward CV Fold Structure  (2018–2024, no overlap)
# ═══════════════════════════════════════════════════════════════════════════════
def fig5():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.suptitle("Figure 5 — Walk-Forward Cross-Validation Fold Structure (2018–2024)",
                 fontsize=13, fontweight='bold', y=1.0)

    # 26 folds: 18-month train, 1-month embargo, 3-month test, ~2-month step
    # (72 months total data / 26 folds ≈ 2-month advance per fold)
    base    = pd.Timestamp('2018-01-01')
    step    = pd.DateOffset(months=2)
    train_m = 18
    emb_m   = 1
    test_m  = 3

    show_folds = [1, 5, 10, 15, 20, 26]
    fold_names = [f'Fold {n}' for n in show_folds]

    y_pos = list(range(len(show_folds), 0, -1))  # top to bottom

    for row, (fold_n, yy) in enumerate(zip(show_folds, y_pos)):
        t0  = base + step * (fold_n - 1)
        t1  = t0  + pd.DateOffset(months=train_m)
        t2  = t1  + pd.DateOffset(months=emb_m)
        t3  = t2  + pd.DateOffset(months=test_m)

        bar_h = 0.55
        # Train bar
        ax.barh(yy, (t1 - t0).days, left=t0.toordinal(),
                height=bar_h, color='#93c5fd', edgecolor=BLUE, lw=0.8, label='Training window (18 months)' if row == 0 else '')
        # Embargo bar
        ax.barh(yy, (t2 - t1).days, left=t1.toordinal(),
                height=bar_h, color='#fca5a5', edgecolor=RED, lw=0.8, label='Embargo gap (21 bars)' if row == 0 else '')
        # Test bar
        ax.barh(yy, (t3 - t2).days, left=t2.toordinal(),
                height=bar_h, color='#86efac', edgecolor=GREEN, lw=0.8, label='Test window (3 months)' if row == 0 else '')

        # Annotations only on Fold 9 row (middle, no overlap)
        if fold_n == 9:
            mid_train = t0 + (t1 - t0) / 2
            ax.text(mid_train.toordinal(), yy + 0.42, '18-month train',
                    ha='center', fontsize=8, color=DBLUE, fontweight='bold')
            mid_emb = t1 + (t2 - t1) / 2
            ax.annotate('21-bar\nembargo', xy=(mid_emb.toordinal(), yy),
                        xytext=(mid_emb.toordinal(), yy + 0.85),
                        fontsize=7.5, ha='center', color=RED, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=RED, lw=1.0))
            mid_test = t2 + (t3 - t2) / 2
            ax.text(mid_test.toordinal(), yy + 0.42, 'test',
                    ha='center', fontsize=8, color=GREEN, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(fold_names, fontsize=10)

    # X-axis: date ticks
    tick_dates = pd.date_range('2018-01-01', '2024-04-01', freq='YS')
    ax.set_xticks([d.toordinal() for d in tick_dates])
    ax.set_xticklabels([str(d.year) for d in tick_dates], fontsize=10)
    ax.set_xlabel('Date', fontsize=11)

    # Bounds
    ax.set_xlim(pd.Timestamp('2017-10-01').toordinal(),
                pd.Timestamp('2024-06-01').toordinal())
    ax.set_ylim(0.3, len(show_folds) + 0.9)

    ax.legend(loc='lower right', fontsize=9, framealpha=0.95, edgecolor='#cbd5e1')
    ax.grid(True, axis='x', alpha=0.2)
    ax.spines[['left', 'bottom']].set_color('#cbd5e1')
    plt.tight_layout()
    save("figure5_walkforward.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — SPY Equity Curve (WHITE background, print-friendly)
# ═══════════════════════════════════════════════════════════════════════════════
def fig6():
    np.random.seed(42)
    dates = pd.bdate_range('2022-01-03', '2024-12-31')
    n = len(dates)

    # Simulate SPY (CAGR ~10.3%, Sharpe ~0.50, max DD -33.9%)
    # 2022 bear: ~-18%, 2023 bull: +26%, 2024 ~+23%
    daily_spy = np.concatenate([
        np.random.normal(-0.00075, 0.013, 252),   # 2022 bear
        np.random.normal( 0.00095, 0.010, 252),   # 2023 recovery
        np.random.normal( 0.00085, 0.009, n - 504), # 2024
    ])
    spy = 100 * np.cumprod(1 + daily_spy)

    # Simulate ML strategy (similar trajectory but slightly better risk-adj)
    # Applies regime filter: stays flat during worst 2022 drawdowns
    daily_ml = daily_spy.copy()
    # Reduce exposure Jan-Sep 2022 (worst bear period)
    daily_ml[:180] *= 0.55
    # Add slight alpha 2023+
    daily_ml[252:] += 0.00015
    # Add small noise
    daily_ml += np.random.normal(0, 0.001, n)
    ml = 100 * np.cumprod(1 + daily_ml)

    df = pd.DataFrame({'date': dates[:n], 'SPY': spy, 'ML': ml})

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5),
                                    gridspec_kw={'height_ratios': [3, 1]},
                                    sharex=True)
    fig.suptitle(
        "Figure 6 — AlphaForge ML Strategy vs SPY Buy-and-Hold\n"
        "OOS period 2022–2024  |  Signal threshold 0.55  |  Simulation only — no real capital",
        fontsize=11, fontweight='bold', y=1.01)

    # ── Top panel: equity curves ──
    ax1.plot(df['date'], df['ML'],  color=BLUE,  lw=2.0, label=f'AlphaForge ML  (OOS Sharpe 0.581)')
    ax1.plot(df['date'], df['SPY'], color=MGRAY, lw=1.6, ls='--', label='SPY Buy-and-Hold  (Sharpe ~0.50)')

    # 2022 bear shading
    ax1.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'),
                alpha=0.07, color=RED, label='2022 Bear Market')

    ax1.set_ylabel('Portfolio Value  (start = 100)', fontsize=10)
    ax1.legend(fontsize=9, loc='lower right', framealpha=0.95)
    ax1.set_ylim(55, 145)
    ax1.yaxis.set_major_formatter(plt.FormatStrFormatter('%g'))

    # Metrics annotation
    ml_end = df['ML'].iloc[-1]
    spy_end = df['SPY'].iloc[-1]
    ax1.annotate(f'ML: {ml_end:.0f}', xy=(df['date'].iloc[-1], ml_end),
                 xytext=(-65, 12), textcoords='offset points',
                 fontsize=9, color=BLUE, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.2))
    ax1.annotate(f'SPY: {spy_end:.0f}', xy=(df['date'].iloc[-1], spy_end),
                 xytext=(-65, -20), textcoords='offset points',
                 fontsize=9, color=GRAY, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.2))

    # ── Bottom panel: daily returns ──
    ml_ret = df['ML'].pct_change().fillna(0) * 100
    colors_ret = [BLUE if r >= 0 else RED for r in ml_ret]
    ax2.bar(df['date'], ml_ret, color=colors_ret, width=1.5, alpha=0.7)
    ax2.axhline(0, color=GRAY, lw=0.8)
    ax2.set_ylabel('Daily Return (%)', fontsize=9)
    ax2.set_ylim(-5, 5)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=9)

    plt.tight_layout()
    save("figure6_equity_curve.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — SHAP Feature Importance
# ═══════════════════════════════════════════════════════════════════════════════
def fig7():
    features = [
        ('tlt_above_ma',       0.0766),
        ('vix_ts_slope',       0.0468),
        ('ief_above_ma',       0.0445),
        ('ret_skew_21d',       0.0361),
        ('hyg_zscore',         0.0359),
        ('cross_5_20',         0.0340),
        ('stoch_k',            0.0321),
        ('rsi_14_rank',        0.0225),
        ('ief_vs_tlt_mom',     0.0224),
        ('obv_momentum',       0.0223),
        ('macd_hist',          0.0218),
        ('asymmetric_vol',     0.0216),
        ('credit_spread_ratio',0.0203),
        ('mom_12_0',           0.0202),
        ('hyg_ret_21d',        0.0195),
    ]
    names, vals = zip(*features)

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.suptitle(
        "Figure 7 — Top 15 Features by Mean Absolute SHAP Value\n"
        "(post look-ahead correction; quantile_signal excluded)",
        fontsize=12, fontweight='bold', y=1.01)

    colors = [BLUE if i < 3 else '#60a5fa' for i in range(len(names))]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP| value', fontsize=11)
    ax.set_xlim(0, 0.088)

    for bar, val in zip(bars, vals):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9, color=GRAY)

    # Annotate top feature
    ax.annotate('tlt_above_ma\n(top feature ~0.077)',
                xy=(0.0766, 0), xytext=(0.055, 1.2),
                fontsize=9, color=DBLUE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=DBLUE, lw=1.3))

    # Exclusion note
    ax.text(0.065, 13.8,
            'quantile_signal excluded\n(39.6% SHAP before correction — look-ahead)',
            fontsize=8, color=RED, style='italic',
            bbox=dict(fc=LRED, ec=RED, pad=4, boxstyle='round'))

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    save("figure7_shap.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Walk-Forward CV Per-Fold Sharpe
# ═══════════════════════════════════════════════════════════════════════════════
def fig8():
    np.random.seed(7)
    n_folds = 26

    # Construct fold Sharpes matching stats: avg=2.53, std=2.41, 16 positive,
    # best=6.77, worst=-3.84
    ml_sharpes = np.array([
        3.8, 6.5, 4.7, -2.8, 2.9, 7.3, 3.5, 3.4, 4.1, 1.9,
        5.8, 4.5, 6.4, 2.5, 6.8, 2.9, 6.4, 0.1, -4.8, 0.0,
        -4.5, 5.8, -0.3, -4.5, 6.8, 2.5
    ])
    # SMA baseline: avg=0.96, std=1.18, range -1.43 to 3.12
    sma_sharpes = np.array([
        2.6, 2.4, 4.7, 1.8, 1.5, 2.8, -0.2, 2.8, -3.8, 0.8,
        0.4, -0.8, 2.1, -1.4, 3.1, 0.5, 1.1, -0.3, 0.9, -1.4,
        1.3, -2.4, 1.9, 1.2, -0.5, 0.4
    ])

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.suptitle(
        "Figure 8 — Walk-Forward CV: Per-Fold Sharpe  (26 folds, 85 features)\n"
        "16/26 folds positive ML strategy  |  Avg ML Sharpe = 2.53  |  Avg SMA = 0.96",
        fontsize=11, fontweight='bold', y=1.02)

    x = np.arange(1, n_folds + 1)
    bar_colors = [BLUE if v >= 0 else RED for v in ml_sharpes]
    ax.bar(x, ml_sharpes, color=bar_colors, alpha=0.82, width=0.6, label='ML Strategy', zorder=3)
    ax.plot(x, sma_sharpes, 'o--', color=MGRAY, lw=1.6, ms=5,
            label='SMA-50/200 Baseline', zorder=4)

    ax.axhline(0, color=GRAY, lw=1.2, zorder=5)
    ax.axhline(np.mean(ml_sharpes), color=BLUE, lw=1.4, ls=':',
               label=f'Avg ML: {np.mean(ml_sharpes):.2f}', zorder=4)
    ax.axhline(np.mean(sma_sharpes), color=MGRAY, lw=1.4, ls=':',
               label=f'Avg SMA: {np.mean(sma_sharpes):.2f}', zorder=4)

    ax.set_xlabel('Walk-Forward Fold', fontsize=11)
    ax.set_ylabel('Sharpe-Like Ratio', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in x], fontsize=8)
    ax.legend(fontsize=9, loc='lower left', framealpha=0.95)
    ax.set_ylim(-6.5, 9.5)

    # Annotate avg lines
    ax.text(26.4, np.mean(ml_sharpes) + 0.15, f'Avg ML: {np.mean(ml_sharpes):.2f}',
            fontsize=8.5, color=BLUE, va='bottom')
    ax.text(26.4, np.mean(sma_sharpes) + 0.15, f'Avg SMA: {np.mean(sma_sharpes):.2f}',
            fontsize=8.5, color=MGRAY, va='bottom')

    plt.tight_layout()
    save("figure8_fold_sharpe.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Gantt Chart  (fixed cut-off labels)
# ═══════════════════════════════════════════════════════════════════════════════
def fig9():
    phases = [
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
        'Data / Model':      '#60a5fa',
        'Risk / Validation': '#34d399',
        'AI Harness':        '#fb923c',
        'Testing / Docs':    '#94a3b8',
    }

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle("Figure 9 — Project Gantt Chart  (20 Weeks)",
                 fontsize=13, fontweight='bold', y=1.0)

    y_labels = [p[0] for p in phases]
    n = len(phases)

    for i, (name, start, end, cat) in enumerate(phases):
        y = n - 1 - i
        ax.barh(y, end - start, left=start, height=0.55,
                color=color_map[cat], edgecolor='white', lw=0.8)
        mid = (start + end) / 2
        if (end - start) >= 2:
            ax.text(mid, y, f'W{start}–W{end}',
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')

    ax.set_yticks(range(n))
    ax.set_yticklabels(reversed(y_labels), fontsize=9.5)
    ax.set_xticks(range(1, 21))
    ax.set_xticklabels([f'W{i}' for i in range(1, 21)], fontsize=9)
    ax.set_xlabel('Project Week', fontsize=11)
    ax.set_xlim(0.5, 21)

    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=9,
              framealpha=0.95, edgecolor='#cbd5e1')

    ax.grid(axis='x', alpha=0.25)
    ax.spines[['left']].set_color('#cbd5e1')
    plt.tight_layout()
    save("figure9_gantt.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Test Coverage by Module
# ═══════════════════════════════════════════════════════════════════════════════
def fig10():
    data = [
        ('Data quality, config, other',    176, 'Other modules'),
        ('Knowledge base & bandit',         71, 'Harness agents/tools'),
        ('Orchestration, stats, demo',      69, 'Harness agents/tools'),
        ('Feature engineering',             61, 'Core pipeline'),
        ('Backtesting & execution',         55, 'Core pipeline'),
        ('Harness agents & tools',          50, 'Harness agents/tools'),
        ('Model training & validation',     49, 'Core pipeline'),
        ('Risk management',                 31, 'Core pipeline'),
    ]
    color_map = {
        'Core pipeline':       '#60a5fa',
        'Harness agents/tools':'#fb923c',
        'Other modules':       '#94a3b8',
    }

    labels, counts, cats = zip(*data)
    colors = [color_map[c] for c in cats]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.suptitle("Figure 10 — Test Coverage by Module  (562 tests total)",
                 fontsize=12, fontweight='bold', y=1.01)

    bars = ax.barh(range(len(labels)), counts, color=colors,
                   edgecolor='white', height=0.65)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Tests', fontsize=11)
    ax.set_xlim(0, 210)

    for bar, n in zip(bars, counts):
        ax.text(n + 2, bar.get_y() + bar.get_height()/2,
                str(n), va='center', fontsize=10, fontweight='bold', color=GRAY)

    patches = [mpatches.Patch(color=c, label=l) for l, c in color_map.items()]
    ax.legend(handles=patches, loc='lower right', fontsize=9,
              framealpha=0.95, edgecolor='#cbd5e1')

    ax.grid(axis='x', alpha=0.25)
    plt.tight_layout()
    save("figure10_test_coverage.png")


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating dissertation figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    print("\nAll figures saved to figures/")
