"""Generate architecture / timeline dissertation figures (1, 2, 3, 4, 5, 9)."""
import os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, text, fc="#dbeafe", ec="#3b82f6", fs=9, bold=False, wrap=False):
    r = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                       facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(r)
    weight = "bold" if bold else "normal"
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fs, fontweight=weight, zorder=4, wrap=wrap)

def arr(ax, x1, y1, x2, y2, color="#374151", lw=1.5, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — System Architecture
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 1 (System Architecture)...")
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Figure 1 — AlphaForge System Architecture", fontsize=12, fontweight="bold", y=0.99)

# Layer labels
for y, label in [(6.6, "Entry Points"), (5.2, "AI Harness (harness/)"),
                  (3.5, "Core ML Pipeline (main.py)"), (1.5, "Infrastructure")]:
    ax.text(0.15, y, label, fontsize=8, color="#6b7280", style="italic", va="center")

ax.axhline(6.3, xmin=0.08, xmax=0.92, color="#e5e7eb", lw=0.8)
ax.axhline(4.9, xmin=0.08, xmax=0.92, color="#e5e7eb", lw=0.8)
ax.axhline(3.1, xmin=0.08, xmax=0.92, color="#e5e7eb", lw=0.8)
ax.axhline(0.9, xmin=0.08, xmax=0.92, color="#e5e7eb", lw=0.8)

# Entry points row
box(ax, 1.0, 6.5, 4.0, 0.7, "main.py  (Classic ML CLI)", fc="#eff6ff", ec="#3b82f6", bold=True)
box(ax, 6.5, 6.5, 4.5, 0.7, "harness_main.py  (AI Harness CLI)", fc="#fdf4ff", ec="#a855f7", bold=True)

# AI Harness row
box(ax, 1.2, 5.1, 2.0, 0.7, "StrategistAgent\n(Claude)", fc="#fdf4ff", ec="#a855f7", fs=8)
box(ax, 3.5, 5.1, 2.0, 0.7, "AnalystAgent\n(Grok/xAI)", fc="#fff7ed", ec="#f97316", fs=8)
box(ax, 5.8, 5.1, 2.0, 0.7, "CoderAgent\n(Claude)", fc="#fdf4ff", ec="#a855f7", fs=8)
box(ax, 8.1, 5.1, 2.2, 0.7, "ReviewerAgent\n(Claude)", fc="#fdf4ff", ec="#a855f7", fs=8)
box(ax, 4.5, 3.9, 3.0, 0.6, "AlphaHarness Orchestrator", fc="#f5f3ff", ec="#7c3aed", fs=8, bold=True)

arr(ax, 5.2, 6.5, 5.2, 5.8, color="#7c3aed")
arr(ax, 2.2, 5.1, 3.2, 4.5, color="#7c3aed")
arr(ax, 4.5, 5.1, 5.2, 4.5, color="#7c3aed")
arr(ax, 6.8, 5.1, 6.5, 4.5, color="#7c3aed")
arr(ax, 9.2, 5.1, 8.0, 4.5, color="#7c3aed")

# Core ML Pipeline row
for i, (name, col) in enumerate([
    ("ingest", "#dbeafe"), ("features", "#dbeafe"), ("train", "#dcfce7"),
    ("backtest", "#dcfce7"), ("validate", "#dcfce7"), ("report", "#fef9c3")
]):
    x = 1.0 + i * 1.7
    box(ax, x, 3.2, 1.5, 0.6, name, fc=col, ec="#6b7280", fs=8)
    if i < 5:
        arr(ax, x + 1.5, 3.5, x + 1.7, 3.5, color="#374151")

arr(ax, 3.0, 6.5, 3.0, 3.8, color="#3b82f6", style="-|>")

# Infrastructure row
box(ax, 1.0, 1.0, 2.0, 0.7, "data/\n(Parquet cache)", fc="#f0fdf4", ec="#22c55e", fs=8)
box(ax, 3.3, 1.0, 2.0, 0.7, "models/artifacts/\n(joblib + JSON)", fc="#f0fdf4", ec="#22c55e", fs=8)
box(ax, 5.6, 1.0, 2.2, 0.7, "harness/memory/\n(KB + bandit)", fc="#faf5ff", ec="#a855f7", fs=8)
box(ax, 8.0, 1.0, 2.5, 0.7, "logs/ + config.yaml", fc="#fefce8", ec="#eab308", fs=8)

for x in [2.0, 4.3, 6.7, 9.25]:
    arr(ax, x, 3.2, x, 1.7, color="#9ca3af", style="-|>")

# Legend
p1 = mpatches.Patch(facecolor="#eff6ff", edgecolor="#3b82f6", label="Core ML Pipeline")
p2 = mpatches.Patch(facecolor="#fdf4ff", edgecolor="#a855f7", label="AI Harness")
p3 = mpatches.Patch(facecolor="#f0fdf4", edgecolor="#22c55e", label="Infrastructure")
ax.legend(handles=[p1, p2, p3], loc="lower right", fontsize=8, framealpha=0.8)
plt.tight_layout()
plt.savefig(f"{OUT}/figure1_architecture.png")
plt.close()
print("  Figure 1 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Data Flow Pipeline
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2 (Data Flow)...")
fig, ax = plt.subplots(figsize=(14, 4))
ax.set_xlim(0, 14); ax.set_ylim(0, 4)
ax.axis("off")
ax.set_title("Figure 2 — AlphaForge Data Flow Pipeline", fontsize=12, fontweight="bold")

stages = [
    ("Data\nIngest", "OHLCV\n+adj close", "#dbeafe", "#3b82f6"),
    ("Feature\nEngine", "250+ cols\n(no look-ahead)", "#dcfce7", "#16a34a"),
    ("Feature\nSelector", "85 cols\n(SHAP)", "#dcfce7", "#16a34a"),
    ("Walk-Forward\nCV", "26 folds\n21-bar embargo", "#fef9c3", "#ca8a04"),
    ("Model\nTrain", "XGB+LGBM\nensemble×8", "#fdf4ff", "#a855f7"),
    ("Backtest\nEngine", "NAV / DD\n/ Sharpe", "#fff7ed", "#ea580c"),
    ("Validate\n& Report", "OOS metrics\nPDF/JSON", "#f0fdf4", "#15803d"),
]

for i, (name, fmt, fc, ec) in enumerate(stages):
    x = 0.3 + i * 1.95
    box(ax, x, 1.4, 1.6, 1.2, name, fc=fc, ec=ec, fs=9, bold=True)
    ax.text(x + 0.8, 0.9, fmt, ha="center", va="top", fontsize=7.5, color="#374151")
    if i < len(stages) - 1:
        arr(ax, x + 1.6, 2.0, x + 1.95, 2.0, color="#374151")

# Data format annotations at transitions
transitions = ["Raw\nParquet", "Feature\nMatrix", "Reduced\nMatrix", "Train/Test\nSplits",
               "Trained\nModel", "Equity\nCurve"]
for i, t in enumerate(transitions):
    x = 1.9 + i * 1.95
    ax.text(x + 0.15, 2.5, t, ha="center", va="bottom", fontsize=7, color="#6b7280",
            style="italic")

plt.tight_layout()
plt.savefig(f"{OUT}/figure2_data_flow.png")
plt.close()
print("  Figure 2 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Risk Management Signal Pipeline
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3 (Risk Management)...")
fig, ax = plt.subplots(figsize=(6, 10))
ax.set_xlim(0, 6); ax.set_ylim(0, 10)
ax.axis("off")
ax.set_title("Figure 3 — Risk Management Signal Pipeline", fontsize=11, fontweight="bold")

def diamond(ax, cx, cy, w, h, text, fc="#fef9c3", ec="#ca8a04", fs=8):
    xs = [cx, cx+w/2, cx, cx-w/2, cx]
    ys = [cy+h/2, cy, cy-h/2, cy, cy+h/2]
    ax.fill(xs, ys, facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=4)

def rect(ax, cx, cy, w, h, text, fc="#dbeafe", ec="#3b82f6", fs=8.5, bold=False):
    r = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.05",
                       facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(r)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)

def flat(ax, cx, cy, text, color="#ef4444"):
    rect(ax, cx, cy, 1.1, 0.4, f"→ {text}", fc="#fee2e2", ec=color, fs=8)

# Start
rect(ax, 3, 9.4, 2.8, 0.55, "Raw signal (proba from model)", fc="#f8fafc", ec="#64748b", bold=True)
arr(ax, 3, 9.1, 3, 8.75)

# Gate 1
diamond(ax, 3, 8.4, 2.2, 0.65, "proba < min_confidence?")
flat(ax, 5.2, 8.4, "FLAT")
ax.text(5.1, 8.4, "YES", fontsize=7.5, color="#dc2626", ha="left", va="center")
ax.text(3.2, 8.0, "NO", fontsize=7.5, color="#15803d")
arr(ax, 3, 8.07, 3, 7.75)

# Gate 2
diamond(ax, 3, 7.4, 2.4, 0.65, "TailRiskManager\nvol z-score > 2.5?")
flat(ax, 5.25, 7.4, "FLAT")
ax.text(5.1, 7.4, "YES", fontsize=7.5, color="#dc2626", ha="left", va="center")
ax.text(3.2, 7.0, "NO", fontsize=7.5, color="#15803d")
arr(ax, 3, 7.07, 3, 6.75)

# Gate 3
diamond(ax, 3, 6.4, 2.6, 0.65, "DrawdownController\ntier ≥ 3 (DD>20%)?")
flat(ax, 5.3, 6.4, "REDUCE / FLAT")
ax.text(5.1, 6.4, "YES", fontsize=7.5, color="#dc2626", ha="left", va="center")
ax.text(3.2, 6.0, "NO", fontsize=7.5, color="#15803d")
arr(ax, 3, 6.07, 3, 5.75)

# Position sizing
rect(ax, 3, 5.4, 2.8, 0.6, "PositionSizer\n(vol-target + Kelly + fixed-risk)",
     fc="#dcfce7", ec="#16a34a")
arr(ax, 3, 5.1, 3, 4.75)

# Gate 4
diamond(ax, 3, 4.4, 2.4, 0.65, "Stop-loss\ntriggered?")
flat(ax, 5.25, 4.4, "CLOSE")
ax.text(5.1, 4.4, "YES", fontsize=7.5, color="#dc2626", ha="left", va="center")
ax.text(3.2, 4.0, "NO", fontsize=7.5, color="#15803d")
arr(ax, 3, 4.07, 3, 3.75)

# Cost model
rect(ax, 3, 3.4, 2.8, 0.6, "CostModel\n(commission + slippage)",
     fc="#fef9c3", ec="#ca8a04")
arr(ax, 3, 3.1, 3, 2.75)

# Output
rect(ax, 3, 2.35, 2.8, 0.65, "Fill order → update NAV\n(BacktestEngine)", fc="#f0fdf4", ec="#15803d", bold=True)

# Rejection path arrows (horizontal to reject box, then down)
for ry in [8.4, 7.4, 6.4, 4.4]:
    ax.annotate("", xy=(4.6, ry), xytext=(3 + 1.1, ry),
                arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))

plt.tight_layout()
plt.savefig(f"{OUT}/figure3_risk_pipeline.png")
plt.close()
print("  Figure 3 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Walk-Forward CV Fold Structure (timeline)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 5 (Walk-Forward CV timeline)...")
fig, ax = plt.subplots(figsize=(13, 5))
ax.set_title("Figure 5 — Walk-Forward Cross-Validation Fold Structure (2015–2024)",
             fontsize=11, fontweight="bold")

years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
year_x = {y: (y - 2015) / (2024 - 2015) * 12 for y in years}

ax.set_xlim(-0.3, 12.5)
ax.set_ylim(-0.5, 6.5)
ax.axis("off")

# Year ticks
for y in years:
    x = year_x[y]
    ax.axvline(x, color="#e5e7eb", lw=0.8, ymin=0.05, ymax=0.95)
    ax.text(x, -0.3, str(y), ha="center", fontsize=9, color="#374151")

# Show 6 representative folds
import json
folds_data = json.load(open("models/artifacts/training_metrics_20260516_023323.json"))["fold_results"]

def to_x(datestr):
    from datetime import datetime
    d = datetime.strptime(datestr, "%Y-%m-%d")
    return (d.year + (d.month-1)/12 + (d.day-1)/365 - 2015) / (2024 - 2015) * 12

show_folds = [0, 4, 8, 12, 17, 22]
labels_y   = [5.8, 4.8, 3.8, 2.8, 1.8, 0.8]

for i, (fi, ly) in enumerate(zip(show_folds, labels_y)):
    f = folds_data[fi]
    ts = to_x(f["train_start"])
    te = to_x(f["train_end"])
    es = te
    ee = to_x(f["test_start"])  # embargo start≈train_end; end≈test_start
    xs = to_x(f["test_start"])
    xe = to_x(f["test_end"])

    # Train bar
    ax.barh(ly, te - ts, left=ts, height=0.55, color="#bfdbfe", edgecolor="#3b82f6", lw=0.8)
    # Embargo bar
    ax.barh(ly, ee - es, left=es, height=0.55, color="#fca5a5", edgecolor="#ef4444", lw=0.8)
    # Test bar
    ax.barh(ly, xe - xs, left=xs, height=0.55, color="#bbf7d0", edgecolor="#16a34a", lw=0.8)
    ax.text(-0.25, ly, f"Fold {f['fold']}", ha="right", va="center", fontsize=8.5, color="#374151")

# Annotate fold 9 (3rd in display)
fi = 8
f = folds_data[fi]
ts = to_x(f["train_start"])
te = to_x(f["train_end"])
xs = to_x(f["test_start"])
xe = to_x(f["test_end"])
mid_train = (ts + te) / 2
mid_test  = (xs + xe) / 2
ly = labels_y[2]
ax.annotate("18-month train", xy=(mid_train, ly + 0.28), xytext=(mid_train, ly + 0.95),
            arrowprops=dict(arrowstyle="-", color="#1d4ed8", lw=1),
            fontsize=8, ha="center", color="#1d4ed8")
ax.annotate("21-bar embargo", xy=((te+xs)/2, ly + 0.28), xytext=((te+xs)/2, ly + 0.95),
            arrowprops=dict(arrowstyle="-", color="#dc2626", lw=1),
            fontsize=8, ha="center", color="#dc2626")
ax.annotate("3-month test", xy=(mid_test, ly + 0.28), xytext=(mid_test, ly + 0.95),
            arrowprops=dict(arrowstyle="-", color="#15803d", lw=1),
            fontsize=8, ha="center", color="#15803d")

# Legend
p1 = mpatches.Patch(facecolor="#bfdbfe", edgecolor="#3b82f6", label="Training window (18 months)")
p2 = mpatches.Patch(facecolor="#fca5a5", edgecolor="#ef4444", label="Embargo gap (21 bars)")
p3 = mpatches.Patch(facecolor="#bbf7d0", edgecolor="#16a34a", label="Test window (3 months)")
ax.legend(handles=[p1, p2, p3], loc="lower right", fontsize=9, framealpha=0.85)

plt.tight_layout()
plt.savefig(f"{OUT}/figure5_walkforward.png")
plt.close()
print("  Figure 5 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Multi-Agent Sequence Diagram
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 4 (Multi-Agent Sequence)...")
fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 13); ax.set_ylim(0, 8)
ax.axis("off")
ax.set_title("Figure 4 — Multi-Agent Research Loop: UML Sequence Diagram",
             fontsize=11, fontweight="bold")

lanes = [
    ("Thompson\nBandit", 1.1, "#fdf4ff", "#a855f7"),
    ("Analyst\n(Grok/xAI)", 3.1, "#fff7ed", "#f97316"),
    ("Strategist\n(Claude)", 5.1, "#dbeafe", "#3b82f6"),
    ("Coder\n(Claude)", 7.1, "#dcfce7", "#16a34a"),
    ("Executor +\nReviewer", 10.5, "#fef9c3", "#ca8a04"),
]

# Actor boxes
for name, x, fc, ec in lanes:
    box(ax, x - 0.7, 7.0, 1.4, 0.7, name, fc=fc, ec=ec, fs=8.5, bold=True)
    ax.axvline(x, ymin=0.04, ymax=0.87, color=ec, lw=1.0, linestyle=":", alpha=0.6)

# Messages (step, from_x, to_x, label, style)
msgs = [
    (0, 1.1, 5.1, "0  select_arm()\n→ archetype", "->"),
    (1, 5.1, 3.1, "1  market_context\nrequest", "->"),
    (2, 3.1, 5.1, "2  regime, macro\ncontext", "-->"),
    (3, 5.1, 5.1, "3  propose_experiment()\n→ config + hypothesis", "->"),
    (4, 5.1, 7.1, "4  generate_factor()\n→ Python code", "->"),
    (5, 7.1, 10.5, "5  validate_ast() +\ntrain + backtest", "->"),
    (6, 10.5, 5.1, "6  BacktestResult\n→ sharpe, dd, trades", "-->"),
    (7, 5.1, 5.1, "7  evaluate_result()\n→ APPROVED / PROMISING\n/ NOT YET", "->"),
    (8, 5.1, 1.1, "8  update_arm(reward)", "-->"),
    (9, 5.1, 10.5, "9  save_to_kb()\n+ session_report", "->"),
]

y_pos = [6.5, 6.0, 5.6, 5.1, 4.6, 4.1, 3.6, 3.0, 2.4, 1.9]
for (step, fx, tx, lbl, sty), y in zip(msgs, y_pos):
    if fx == tx:  # self-arrow
        ax.annotate("", xy=(fx + 0.6, y - 0.15), xytext=(fx + 0.6, y + 0.0),
                    arrowprops=dict(arrowstyle="->", color="#374151",
                                    connectionstyle="arc3,rad=-0.5"))
        ax.text(fx + 0.7, y - 0.07, lbl, fontsize=7.5, va="center", color="#374151")
    else:
        color = "#374151" if sty == "->" else "#6b7280"
        ls = "-" if sty == "->" else "--"
        ax.annotate("", xy=(tx, y), xytext=(fx, y),
                    arrowprops=dict(arrowstyle=f"-|>", color=color, lw=1.3,
                                    linestyle=ls))
        mid_x = (fx + tx) / 2
        ax.text(mid_x, y + 0.08, lbl, fontsize=7.5, ha="center", va="bottom",
                color=color)

plt.tight_layout()
plt.savefig(f"{OUT}/figure4_sequence.png")
plt.close()
print("  Figure 4 saved.")


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Project Gantt Chart
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 9 (Gantt Chart)...")
phases = [
    ("Requirements & Design",     1,  2,  "docs"),
    ("Data & Feature Layer",       3,  4,  "model"),
    ("Model Training",             5,  6,  "model"),
    ("Backtesting & Risk",         7,  8,  "risk"),
    ("Walk-Forward Validation",    9,  10, "risk"),
    ("AI Harness — Agents",        9,  12, "harness"),
    ("AI Harness — Orchestrator", 11, 13, "harness"),
    ("AI Harness — Dashboard",    13, 15, "harness"),
    ("Testing & Bug Fixes",       14, 17, "docs"),
    ("Performance Optimisation",  16, 18, "model"),
    ("Dissertation Writing",      15, 20, "docs"),
    ("Final Review & Submission", 19, 20, "docs"),
]
cat_colors = {"model": "#3b82f6", "risk": "#22c55e",
              "harness": "#f97316", "docs": "#9ca3af"}

fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0.5, 20.5)
ax.set_ylim(-0.5, len(phases))
ax.set_xlabel("Project Week", fontsize=11)
ax.set_title("Figure 9 — Project Gantt Chart  (20 Weeks)", fontsize=11, fontweight="bold")

for i, (name, start, end, cat) in enumerate(phases):
    y = len(phases) - 1 - i
    ax.barh(y, end - start, left=start, height=0.6,
            color=cat_colors[cat], alpha=0.82, edgecolor="white", linewidth=0.8)
    ax.text(start + (end - start) / 2, y, name, ha="center", va="center",
            fontsize=8.5, color="white", fontweight="bold")

ax.set_yticks([])
ax.set_xticks(range(1, 21))
ax.set_xticklabels([f"W{w}" for w in range(1, 21)], fontsize=8)
ax.grid(axis="x", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)

patches = [mpatches.Patch(color=cat_colors["model"],   label="Data / Model"),
           mpatches.Patch(color=cat_colors["risk"],    label="Risk / Validation"),
           mpatches.Patch(color=cat_colors["harness"], label="AI Harness"),
           mpatches.Patch(color=cat_colors["docs"],    label="Testing / Docs")]
ax.legend(handles=patches, loc="lower right", fontsize=9, framealpha=0.85)

plt.tight_layout()
plt.savefig(f"{OUT}/figure9_gantt.png")
plt.close()
print("  Figure 9 saved.")

print("\nDone. All architecture figures in:", OUT)
