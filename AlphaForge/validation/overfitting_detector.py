"""
validation/overfitting_detector.py
===================================
Automated overfitting detection for AI trading strategies.

Implements a multi-signal composite score drawing on:

  1. IS/OOS Performance Gap         — direct Sharpe / CAGR / win-rate / drawdown
                                       degradation from in-sample to out-of-sample
  2. Deflated Sharpe Ratio (DSR)    — Bailey & Lopez de Prado (2014)
                                       adjusts the observed Sharpe for the expected
                                       maximum obtainable by chance across N trials
  3. Probability of Backtest        — CSCV-inspired estimate (Bailey et al., 2016)
     Overfitting (PBO)                Spearman IS/OOS rank correlation across folds
  4. Feature Importance Stability   — coefficient of variation of importances across
                                       walk-forward folds (Lopez de Prado, 2018 Ch.8)
  5. Multiple Testing Penalty       — Bonferroni-adjusted significance threshold

Severity scale
--------------
  LOW      overfitting_score < 0.25   Research looks clean
  MODERATE 0.25 ≤ score < 0.50       Caution — review warnings before proceeding
  HIGH     0.50 ≤ score < 0.75       Do not proceed; apply recommended fixes first
  SEVERE   score ≥ 0.75              Strong evidence of data snooping / overfitting

Usage
-----
  # Minimal (aggregate IS/OOS metrics only)
  from validation.overfitting_detector import detect_overfitting, print_report
  report = detect_overfitting(is_metrics, oos_metrics, n_features=20, n_tests=50)
  print_report(report)

  # Full (per-fold and per-fold feature importances)
  report = detect_overfitting(
      in_sample_metrics=is_dict,
      out_of_sample_metrics=oos_dict,
      n_features=20,
      n_tests=50,
      fold_metrics=[{"is_sharpe": 1.2, "sharpe_ratio": 0.8, ...}, ...],
      feature_importances=[{"rsi_14": 0.12, ...}, ...],   # one dict per CV fold
      oos_bars=252,
  )

  # Via CLI (requires a prior 'alpha report' run)
  python main.py overfit-check --ticker SPY --n-tests 50
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm, spearmanr
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from utils.helpers import get_logger

# Suppress scipy.stats ConstantInputWarning when correlation is undefined
warnings.filterwarnings("ignore", category=Warning, module="scipy.stats")

logger = get_logger(__name__)
console = Console()

# ── Constants ─────────────────────────────────────────────────────────────────

_EULER_MASCHERONI = 0.5772156649015328

# Composite score weights — must sum to 1.0
_WEIGHTS = {
    "is_oos_gap":          0.40,
    "dsr_penalty":         0.25,
    "pbo":                 0.20,
    "feature_instability": 0.15,
}

# (upper_bound, label, rich_color)
_SEVERITY_LEVELS: list[tuple[float, str, str]] = [
    (0.25, "LOW",      "green"),
    (0.50, "MODERATE", "yellow"),
    (0.75, "HIGH",     "red"),
    (1.01, "SEVERE",   "bold red"),
]


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class OverfittingReport:
    """Complete diagnostic output from :func:`detect_overfitting`."""

    # Composite
    overfitting_score:         float
    severity:                  str        # LOW / MODERATE / HIGH / SEVERE

    # IS/OOS gap component
    is_oos_gap_score:          float      # 0-1 component score
    sharpe_is:                 float
    sharpe_oos:                float
    sharpe_gap_abs:            float      # IS − OOS
    sharpe_gap_pct:            float      # relative gap as fraction of IS (signed)
    cagr_is:                   float
    cagr_oos:                  float
    win_rate_is:               float
    win_rate_oos:              float
    max_drawdown_is:           float
    max_drawdown_oos:          float

    # Deflated Sharpe Ratio component
    deflated_sharpe_ratio:     float      # DSR ∈ [0,1] — P(edge is genuine)
    dsr_p_value:               float      # 1 − DSR
    dsr_penalty:               float      # component score (1 − DSR)
    min_backtest_length_yrs:   float      # years needed for statistical significance
    oos_years:                 float      # actual OOS years

    # PBO component
    pbo_estimate:              float      # 0 (none) → 1 (severe)
    is_oos_rank_correlation:   float      # Spearman r across folds
    n_folds_available:         int

    # Feature stability component
    feature_stability_score:   float      # 0 (unstable) → 1 (stable)
    feature_instability_score: float      # complement
    unstable_features:         list[str]

    # Multiple testing (informational)
    n_tests:                   int
    n_features:                int
    bonferroni_threshold:      float
    multiple_testing_penalty:  float

    # Diagnostics
    warnings:                  list[str] = field(default_factory=list)
    recommendations:           list[str] = field(default_factory=list)
    details:                   dict      = field(default_factory=dict)

    # ── Convenience ──────────────────────────────────────────────────────────

    @property
    def color(self) -> str:
        return {
            "LOW": "green", "MODERATE": "yellow",
            "HIGH": "red",  "SEVERE":   "bold red",
        }.get(self.severity, "white")

    def to_dict(self) -> dict:
        """Serialisable summary for inclusion in JSON / Markdown reports."""
        return {
            "overfitting_score":        round(self.overfitting_score, 4),
            "severity":                 self.severity,
            "sharpe_is":                round(self.sharpe_is, 4),
            "sharpe_oos":               round(self.sharpe_oos, 4),
            "sharpe_gap_abs":           round(self.sharpe_gap_abs, 4),
            "sharpe_gap_pct":           round(self.sharpe_gap_pct * 100, 2),
            "cagr_is":                  round(self.cagr_is, 4),
            "cagr_oos":                 round(self.cagr_oos, 4),
            "win_rate_is":              round(self.win_rate_is, 4),
            "win_rate_oos":             round(self.win_rate_oos, 4),
            "max_drawdown_is":          round(self.max_drawdown_is, 4),
            "max_drawdown_oos":         round(self.max_drawdown_oos, 4),
            "deflated_sharpe_ratio":    round(self.deflated_sharpe_ratio, 4),
            "dsr_p_value":              round(self.dsr_p_value, 4),
            "min_backtest_length_yrs":  self.min_backtest_length_yrs,
            "oos_years":                round(self.oos_years, 2),
            "pbo_estimate":             round(self.pbo_estimate, 4),
            "is_oos_rank_correlation":  round(self.is_oos_rank_correlation, 4),
            "n_folds_available":        self.n_folds_available,
            "feature_stability_score":  round(self.feature_stability_score, 4),
            "unstable_features":        self.unstable_features,
            "n_tests":                  self.n_tests,
            "n_features":               self.n_features,
            "bonferroni_threshold":     round(self.bonferroni_threshold, 6),
            "multiple_testing_penalty": round(self.multiple_testing_penalty, 4),
            "warnings":                 self.warnings,
            "recommendations":          self.recommendations,
        }


# ── Public API ────────────────────────────────────────────────────────────────

def detect_overfitting(
    in_sample_metrics: dict,
    out_of_sample_metrics: dict,
    feature_count: int,
    number_of_tests: int,
    fold_metrics: Optional[list[dict]] = None,
    feature_importances: Optional[list[dict]] = None,
    oos_bars: int = 252,
    annual_trading_days: int = 252,
) -> OverfittingReport:
    """
    Run the full overfitting detection pipeline.

    Parameters
    ----------
    in_sample_metrics : dict
        Performance metrics from the in-sample / training period.
        Required key: ``sharpe_ratio``.
        Optional: ``cagr``, ``max_drawdown``, ``win_rate``,
        ``return_skewness``, ``return_excess_kurtosis``.
    out_of_sample_metrics : dict
        Same structure for the out-of-sample / test period.
    feature_count : int
        Number of features fed to the model.
    number_of_tests : int
        Total independent strategy configurations trialled
        (hyperparameter search iterations + strategy variants).
        Higher counts require stronger evidence of genuine edge.
    fold_metrics : list[dict], optional
        Per-fold metrics from walk-forward validation. Each dict should
        carry ``is_sharpe`` and one of ``oos_sharpe`` / ``sharpe_ratio``.
        Used for PBO estimation; falls back to neutral if < 3 folds.
    feature_importances : list[dict], optional
        Feature importance dicts, one per CV fold. A single dict is
        wrapped automatically (stability cannot be estimated in that case).
    oos_bars : int
        Length of the OOS sample in bars (used for DSR calculation).
    annual_trading_days : int
        Trading days per year (252 for daily equity data).

    Returns
    -------
    OverfittingReport
        Complete diagnostic report with composite score, component
        breakdown, per-signal analysis, warnings, and recommendations.
    """
    # ── IS / OOS gap ──────────────────────────────────────────────────────
    gap_score, gap_details = _compute_gap_score(in_sample_metrics, out_of_sample_metrics)

    # ── Deflated Sharpe Ratio ─────────────────────────────────────────────
    sr_hat = float(in_sample_metrics.get("sharpe_ratio", 0.0))
    skew   = float(in_sample_metrics.get("return_skewness", 0.0))
    kurt   = float(in_sample_metrics.get("return_excess_kurtosis", 0.0))

    dsr, dsr_p_value = _compute_deflated_sharpe(
        sr_hat=sr_hat,
        n_trials=max(number_of_tests, 1),
        n_obs=max(oos_bars, 10),
        skewness=skew,
        excess_kurtosis=kurt,
        annual_factor=annual_trading_days,
    )
    dsr_penalty = round(max(0.0, min(1.0, 1.0 - dsr)), 4)

    min_btl = _compute_min_backtest_length(
        sr_hat=sr_hat,
        n_trials=number_of_tests,
        annual_factor=annual_trading_days,
    )
    oos_years = round(oos_bars / annual_trading_days, 2)

    # ── PBO estimate ──────────────────────────────────────────────────────
    pbo, rank_corr = _compute_pbo(fold_metrics or [])
    n_folds_used = sum(
        1 for f in (fold_metrics or [])
        if _has_is_oos(f)
    )

    # ── Feature stability ─────────────────────────────────────────────────
    imp_list: list[dict] = []
    if feature_importances is not None:
        imp_list = [feature_importances] if isinstance(feature_importances, dict) else list(feature_importances)
    feat_stability, unstable = _compute_feature_stability(imp_list)
    feat_instability = round(1.0 - feat_stability, 4)

    # ── Multiple testing (informational only, not in composite) ──────────
    mt_penalty, bonferroni_threshold = _compute_multiple_testing_penalty(
        n_tests=number_of_tests,
        n_features=feature_count,
    )

    # ── Composite score ───────────────────────────────────────────────────
    composite = (
        _WEIGHTS["is_oos_gap"]          * gap_score
        + _WEIGHTS["dsr_penalty"]       * dsr_penalty
        + _WEIGHTS["pbo"]               * pbo
        + _WEIGHTS["feature_instability"] * feat_instability
    )
    composite = round(min(1.0, max(0.0, composite)), 4)
    severity  = _severity_from_score(composite)

    # ── Diagnostics ───────────────────────────────────────────────────────
    diag_ctx = dict(
        gap_score=gap_score, gap_details=gap_details,
        dsr=dsr, dsr_p_value=dsr_p_value,
        pbo=pbo, rank_corr=rank_corr,
        feat_stability=feat_stability, unstable=unstable,
        n_tests=number_of_tests, n_features=feature_count,
        oos_years=oos_years, min_btl=min_btl,
        mt_penalty=mt_penalty,
    )
    warnings        = _build_warnings(**diag_ctx)
    recommendations = _build_recommendations(
        **diag_ctx,
        severity=severity,
    )

    return OverfittingReport(
        overfitting_score          = composite,
        severity                   = severity,
        is_oos_gap_score           = round(gap_score, 4),
        sharpe_is                  = gap_details["sharpe_is"],
        sharpe_oos                 = gap_details["sharpe_oos"],
        sharpe_gap_abs             = gap_details["sharpe_gap_abs"],
        sharpe_gap_pct             = gap_details["sharpe_gap_pct"] / 100.0,
        cagr_is                    = gap_details["cagr_is"],
        cagr_oos                   = gap_details["cagr_oos"],
        win_rate_is                = gap_details["win_rate_is"],
        win_rate_oos               = gap_details["win_rate_oos"],
        max_drawdown_is            = gap_details["max_drawdown_is"],
        max_drawdown_oos           = gap_details["max_drawdown_oos"],
        deflated_sharpe_ratio      = dsr,
        dsr_p_value                = dsr_p_value,
        dsr_penalty                = dsr_penalty,
        min_backtest_length_yrs    = min_btl,
        oos_years                  = oos_years,
        pbo_estimate               = pbo,
        is_oos_rank_correlation    = rank_corr,
        n_folds_available          = n_folds_used,
        feature_stability_score    = feat_stability,
        feature_instability_score  = feat_instability,
        unstable_features          = unstable,
        n_tests                    = number_of_tests,
        n_features                 = feature_count,
        bonferroni_threshold       = bonferroni_threshold,
        multiple_testing_penalty   = mt_penalty,
        warnings                   = warnings,
        recommendations            = recommendations,
        details                    = gap_details,
    )


# ---------------------------------------------------------------------------
# Equity curve quality gates
# ---------------------------------------------------------------------------

def equity_curve_r2(equity: pd.Series) -> float:
    """
    Coefficient of determination (R²) of the equity curve against a linear trend.

    A genuine edge produces a smooth, upward-sloping equity curve (high R²).
    Luck / overfitting produces a jagged curve that fits a line poorly (low R²).

    Returns R² in [-1, 1].  Threshold: reject if R² < 0.30.
    """
    if len(equity) < 5:
        return 0.0
    y   = np.asarray(equity, dtype=float)
    y   = y / y[0]          # normalise to 1
    x   = np.arange(len(y), dtype=float)
    ss_res = np.sum((y - (np.polyval(np.polyfit(x, y, 1), x))) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0


def positive_folds_fraction(fold_metrics: list[dict]) -> float:
    """
    Fraction of OOS folds with Sharpe > 0.

    A strategy with genuine edge should be profitable in the majority of folds.
    Threshold: reject if fraction < 0.60 (fewer than 60% of folds profitable).

    Returns fraction in [0, 1].
    """
    if not fold_metrics:
        return 0.0
    oos_vals = [float(f.get("sharpe_ratio", f.get("oos_sharpe", 0.0))) for f in fold_metrics]
    positive = sum(1 for v in oos_vals if v > 0.0)
    return positive / len(oos_vals)


def print_report(report: OverfittingReport, ticker: str = "") -> None:
    """Render the full overfitting report to the console using Rich."""
    header = f"Overfitting Detection{' — ' + ticker if ticker else ''}"

    # Score progress bar
    filled = int(round(report.overfitting_score * 24))
    bar    = "█" * filled + "░" * (24 - filled)
    console.print(Panel(
        f"[bold]Composite Overfitting Score:[/] [{report.color}]{report.overfitting_score:.3f}[/]  "
        f"[{report.color}]● {report.severity}[/]\n"
        f"[dim]{bar}[/]  [dim]0.00 ──────────────── 1.00[/]",
        title=f"[bold cyan]{header}[/]",
        expand=False,
    ))

    # Component breakdown
    comp_t = Table(
        title="[bold]Component Breakdown[/]",
        show_header=True, header_style="bold",
    )
    comp_t.add_column("Signal",          min_width=26)
    comp_t.add_column("Weight", justify="right")
    comp_t.add_column("Score",  justify="right")
    comp_t.add_column("Contribution", justify="right")
    comp_t.add_column("Level", justify="center")
    _comp_row(comp_t, "IS/OOS Gap",          0.40, report.is_oos_gap_score)
    _comp_row(comp_t, "DSR Penalty",          0.25, report.dsr_penalty)
    _comp_row(comp_t, "PBO Estimate",         0.20, report.pbo_estimate)
    _comp_row(comp_t, "Feature Instability",  0.15, report.feature_instability_score)
    console.print(comp_t)

    # IS/OOS gap table
    gap_t = Table(
        title="[bold]IS vs OOS Performance Gap[/]",
        show_header=True, header_style="bold",
    )
    gap_t.add_column("Metric",         min_width=14)
    gap_t.add_column("In-Sample",      justify="right")
    gap_t.add_column("Out-of-Sample",  justify="right")
    gap_t.add_column("Change",         justify="right")
    gap_t.add_row(
        "Sharpe Ratio",
        f"{report.sharpe_is:.3f}",
        f"{report.sharpe_oos:.3f}",
        _delta_cell(report.sharpe_oos - report.sharpe_is, pct=False),
    )
    gap_t.add_row(
        "CAGR",
        f"{report.cagr_is*100:.1f}%",
        f"{report.cagr_oos*100:.1f}%",
        _delta_cell(report.cagr_oos - report.cagr_is, pct=True),
    )
    gap_t.add_row(
        "Win Rate",
        f"{report.win_rate_is*100:.1f}%",
        f"{report.win_rate_oos*100:.1f}%",
        _delta_cell(report.win_rate_oos - report.win_rate_is, pct=True),
    )
    gap_t.add_row(
        "Max Drawdown",
        f"{abs(report.max_drawdown_is)*100:.1f}%",
        f"{abs(report.max_drawdown_oos)*100:.1f}%",
        _delta_cell(
            abs(report.max_drawdown_is) - abs(report.max_drawdown_oos),
            pct=True, invert=True,
        ),
    )
    console.print(gap_t)

    # DSR / PBO diagnostics panel
    dsr_col = "green" if report.deflated_sharpe_ratio >= 0.80 else \
              "yellow" if report.deflated_sharpe_ratio >= 0.50 else "red"
    pbo_col = "green" if report.pbo_estimate < 0.30 else \
              "yellow" if report.pbo_estimate < 0.45 else "red"
    stab_col = "green" if report.feature_stability_score >= 0.70 else \
               "yellow" if report.feature_stability_score >= 0.50 else "red"

    btl_note = (
        f"[red]⚠ need {report.min_backtest_length_yrs:.1f}y, have {report.oos_years:.1f}y[/]"
        if math.isfinite(report.min_backtest_length_yrs) and report.oos_years < report.min_backtest_length_yrs
        else f"[green]✓ have {report.oos_years:.1f}y, need {report.min_backtest_length_yrs:.1f}y[/]"
    )
    diag_lines = [
        f"[bold]Deflated Sharpe Ratio:[/]       [{dsr_col}]{report.deflated_sharpe_ratio:.3f}[/]  "
        f"(p-value {report.dsr_p_value:.3f})",
        f"[bold]Min Backtest Length:[/]          {btl_note}",
        f"[bold]PBO Estimate:[/]                 [{pbo_col}]{report.pbo_estimate:.3f}[/]  "
        f"(IS/OOS rank r = {report.is_oos_rank_correlation:.3f},"
        f"  {report.n_folds_available} folds)",
        f"[bold]Feature Stability:[/]            [{stab_col}]{report.feature_stability_score:.3f}[/] / 1.00",
        f"[bold]Multiple Testing (Bonferroni):[/] {report.n_tests} trials  ·  "
        f"α = {report.bonferroni_threshold:.5f}  "
        f"(vs naïve 0.05000)",
    ]
    if report.unstable_features:
        diag_lines.append(
            f"[dim]Unstable features:[/] "
            + ", ".join(f"[italic]{f}[/]" for f in report.unstable_features[:6])
        )
    console.print(Panel("\n".join(diag_lines), title="[bold]Diagnostics[/]", expand=False))

    # Warnings
    if report.warnings:
        w_text = "\n".join(f"  ⚠  {w}" for w in report.warnings)
        console.print(Panel(w_text, title="[bold yellow]Warnings[/]", expand=False))

    # Recommendations
    if report.recommendations:
        r_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(report.recommendations))
        console.print(Panel(r_text, title="[bold green]Recommended Actions[/]", expand=False))


def format_report_section(report: OverfittingReport) -> str:
    """Return a Markdown string for inclusion in full validation reports."""
    badge = {"LOW": "✅", "MODERATE": "⚠️", "HIGH": "🔴", "SEVERE": "🚨"}.get(report.severity, "")
    btl_ok = (
        not math.isfinite(report.min_backtest_length_yrs)
        or report.oos_years >= report.min_backtest_length_yrs
    )

    lines: list[str] = [
        "---",
        "",
        "## Overfitting Detection Analysis",
        "",
        f"> **Composite Overfitting Score: `{report.overfitting_score:.3f}` — {badge} {report.severity}**",
        "",
        "> *Based on: IS/OOS performance gap (40%), Deflated Sharpe Ratio (25%), "
        "Probability of Backtest Overfitting (20%), Feature Importance Stability (15%).*",
        "",
        "### Component Breakdown",
        "",
        "| Signal | Weight | Component Score | Contribution |",
        "|--------|--------|-----------------|--------------|",
        f"| IS/OOS Performance Gap | 40% | `{report.is_oos_gap_score:.3f}` "
        f"| `{report.is_oos_gap_score * 0.40:.3f}` |",
        f"| Deflated Sharpe Penalty | 25% | `{report.dsr_penalty:.3f}` "
        f"| `{report.dsr_penalty * 0.25:.3f}` |",
        f"| PBO Estimate | 20% | `{report.pbo_estimate:.3f}` "
        f"| `{report.pbo_estimate * 0.20:.3f}` |",
        f"| Feature Instability | 15% | `{report.feature_instability_score:.3f}` "
        f"| `{report.feature_instability_score * 0.15:.3f}` |",
        "",
        "### IS vs OOS Performance Gap",
        "",
        "| Metric | In-Sample | Out-of-Sample | Change |",
        "|--------|-----------|---------------|--------|",
        f"| Sharpe Ratio | `{report.sharpe_is:.3f}` | `{report.sharpe_oos:.3f}` "
        f"| `{report.sharpe_gap_abs:+.3f}` ({report.sharpe_gap_pct*100:+.1f}%) |",
        f"| CAGR | `{report.cagr_is*100:.1f}%` | `{report.cagr_oos*100:.1f}%` "
        f"| `{(report.cagr_oos-report.cagr_is)*100:+.1f}pp` |",
        f"| Win Rate | `{report.win_rate_is*100:.1f}%` | `{report.win_rate_oos*100:.1f}%` "
        f"| `{(report.win_rate_oos-report.win_rate_is)*100:+.1f}pp` |",
        f"| Max Drawdown | `{abs(report.max_drawdown_is)*100:.1f}%` "
        f"| `{abs(report.max_drawdown_oos)*100:.1f}%` "
        f"| `{(abs(report.max_drawdown_oos)-abs(report.max_drawdown_is))*100:+.1f}pp` |",
        "",
        "### Deflated Sharpe Ratio *(Bailey & Lopez de Prado, 2014)*",
        "",
        f"- **Observed IS Sharpe (annualised):** `{report.sharpe_is:.3f}`",
        f"- **Trials tested (N):** `{report.n_tests}`",
        f"- **Deflated Sharpe Ratio (DSR):** `{report.deflated_sharpe_ratio:.3f}` "
        "— probability the edge is genuine after multiple-testing correction",
        f"- **DSR p-value:** `{report.dsr_p_value:.3f}` "
        f"({report.dsr_p_value*100:.0f}% probability the Sharpe is a false positive)",
        f"- **Minimum backtest length:** `{report.min_backtest_length_yrs:.1f} years` "
        f"required for {report.n_tests} trials at 5% significance; "
        f"actual OOS: `{report.oos_years:.1f} years` "
        f"({'✅ sufficient' if btl_ok else '⚠️ insufficient'})",
        "",
        "> *The DSR adjusts the observed Sharpe ratio for the fact that testing many "
        "strategy configurations inflates apparent performance. "
        "DSR < 0.5 means the edge is more likely noise than signal given the number of trials.*",
        "",
        "### Probability of Backtest Overfitting *(CSCV-inspired)*",
        "",
        f"- **PBO estimate:** `{report.pbo_estimate:.3f}` "
        "(0.0 = no overfitting, 0.5 = random, 1.0 = severe overfitting)",
        f"- **IS/OOS Spearman rank correlation:** `{report.is_oos_rank_correlation:.3f}` "
        f"across `{report.n_folds_available}` walk-forward folds",
        "",
        "> *PBO measures whether in-sample performance rank predicts out-of-sample "
        "performance rank across folds. PBO > 0.5 means the best IS fold tends to "
        "perform worst OOS — a clear sign of overfitting.*",
        "",
        "### Feature Importance Stability",
        "",
        f"- **Stability score:** `{report.feature_stability_score:.3f}` / 1.00 "
        "(1.0 = perfectly consistent across folds)",
    ]

    if report.unstable_features:
        lines.append(
            "- **Most unstable features:** "
            + ", ".join(f"`{f}`" for f in report.unstable_features[:8])
        )
    else:
        lines.append("- No unstable features detected (or per-fold importances not available)")

    lines += [
        "",
        "> *Stability measures how consistently features are ranked across CV folds. "
        "Coefficient of variation > 0.5 on an important feature suggests the model "
        "is fitting noise in that fold rather than a robust signal.*",
        "",
        "### Multiple Testing",
        "",
        f"- **Configurations tested (N):** `{report.n_tests}`",
        f"- **Features used:** `{report.n_features}`",
        f"- **Bonferroni significance threshold:** `{report.bonferroni_threshold:.5f}` "
        "(vs naïve `0.05000`)",
        f"- **Multiple-testing penalty score:** `{report.multiple_testing_penalty:.3f}`",
        "",
    ]

    if report.warnings:
        lines += ["### Warnings", ""]
        for w in report.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    if report.recommendations:
        lines += ["### Recommended Actions", ""]
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)


# ── Private helpers ───────────────────────────────────────────────────────────

def _compute_gap_score(is_m: dict, oos_m: dict) -> tuple[float, dict]:
    """Compute a 0-1 IS/OOS gap score and a detail dict."""
    eps = 1e-9

    is_sr   = float(is_m.get("sharpe_ratio",  0.0))
    oos_sr  = float(oos_m.get("sharpe_ratio", 0.0))
    is_cagr = float(is_m.get("cagr",  0.0))
    oos_cagr = float(oos_m.get("cagr", 0.0))
    is_wr   = float(is_m.get("win_rate",  0.50))
    oos_wr  = float(oos_m.get("win_rate", 0.50))
    is_dd   = abs(float(is_m.get("max_drawdown",  0.0)))
    oos_dd  = abs(float(oos_m.get("max_drawdown", 0.0)))

    # Sharpe degradation (normalised by IS magnitude)
    # Clamp denominator to 0.30 so near-zero IS Sharpe doesn't inflate the gap score.
    sr_gap_abs = is_sr - oos_sr
    sr_gap_rel = sr_gap_abs / max(abs(is_sr), 0.30, eps) if is_sr != 0 else 0.0
    sr_score   = max(0.0, min(1.0, sr_gap_rel))

    # CAGR degradation
    cagr_gap_rel = (is_cagr - oos_cagr) / max(abs(is_cagr), eps) if is_cagr != 0 else 0.0
    cagr_score   = max(0.0, min(1.0, cagr_gap_rel))

    # Win rate drop (absolute)
    wr_drop_score = max(0.0, min(1.0, (is_wr - oos_wr) * 5))   # 20pp → 1.0

    # Drawdown worsening OOS
    dd_worsening_score = max(0.0, min(1.0, (oos_dd - is_dd) * 5))  # 20pp → 1.0

    gap_score = (
        0.55 * sr_score
        + 0.25 * cagr_score
        + 0.10 * wr_drop_score
        + 0.10 * dd_worsening_score
    )

    return round(gap_score, 4), {
        "sharpe_is":        round(is_sr, 4),
        "sharpe_oos":       round(oos_sr, 4),
        "sharpe_gap_abs":   round(sr_gap_abs, 4),
        "sharpe_gap_pct":   round(sr_gap_rel * 100, 2),
        "cagr_is":          round(is_cagr, 4),
        "cagr_oos":         round(oos_cagr, 4),
        "cagr_gap_pct":     round(cagr_gap_rel * 100, 2),
        "win_rate_is":      round(is_wr, 4),
        "win_rate_oos":     round(oos_wr, 4),
        "max_drawdown_is":  round(is_dd, 4),
        "max_drawdown_oos": round(oos_dd, 4),
    }


def _compute_deflated_sharpe(
    sr_hat: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
    annual_factor: int = 252,
) -> tuple[float, float]:
    """
    Compute the Deflated Sharpe Ratio (DSR).

    Ref: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."

    Returns (dsr, p_value).  DSR ≈ P(edge is genuine | N trials tested).
    """
    if n_obs < 5:
        return 0.5, 0.5

    # Convert annualised SR to per-bar SR
    sr_bar = sr_hat / math.sqrt(annual_factor)

    # Expected maximum Sharpe from N independent trials (per-bar units).
    # Formula: E[max SR_N] ≈ [(1-γ)·Φ^{-1}(1-1/N) + γ·Φ^{-1}(1-1/(N·e))] / √n_obs
    # where γ = Euler-Mascheroni constant ≈ 0.5772.
    n = max(n_trials, 2)
    try:
        sr_star_bar = (
            (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n)
            + _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n * math.e))
        ) / math.sqrt(n_obs)
    except Exception:
        sr_star_bar = 0.0

    # Variance of the SR estimator (Mertens 2002, adjusted for skew/kurtosis).
    # Var[SR_bar] ≈ (1 + SR_bar²*(1/2 + κ/4) − γ₃·SR_bar) / T
    var_sr = (
        1.0
        + sr_bar ** 2 * (0.5 + excess_kurtosis / 4.0)
        - skewness * sr_bar
    ) / n_obs
    var_sr = max(var_sr, 1e-12)

    z   = (sr_bar - sr_star_bar) / math.sqrt(var_sr)
    dsr = float(norm.cdf(z))
    return round(dsr, 4), round(1.0 - dsr, 4)


def _compute_pbo(fold_metrics: list[dict]) -> tuple[float, float]:
    """
    Estimate the Probability of Backtest Overfitting from fold performance.

    Uses Spearman rank correlation between IS and OOS Sharpe as a proxy:
    - Perfect positive correlation (r=1) → PBO ≈ 0 (ranks preserved IS→OOS)
    - Zero correlation (r=0)             → PBO ≈ 0.5 (random)
    - Negative correlation (r<0)         → PBO → 1 (best IS is worst OOS)

    Returns (pbo_estimate, is_oos_rank_correlation).
    """
    valid = [f for f in fold_metrics if _has_is_oos(f)]
    if len(valid) < 3:
        return 0.5, 0.0

    is_vals  = [float(f["is_sharpe"]) for f in valid]
    oos_vals = [float(f.get("oos_sharpe", f.get("sharpe_ratio", 0.0))) for f in valid]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            corr_result = spearmanr(is_vals, oos_vals)
        corr = float(corr_result.statistic) if hasattr(corr_result, "statistic") else float(corr_result[0])
    except Exception:
        corr = 0.0

    if not math.isfinite(corr):
        corr = 0.0

    pbo = max(0.0, min(1.0, 0.5 - corr * 0.5))
    return round(pbo, 4), round(corr, 4)


def _compute_feature_stability(importances_list: list[dict]) -> tuple[float, list[str]]:
    """
    Compute feature importance stability across CV folds.

    Uses coefficient of variation (CoV = std / mean) per feature, measured
    on features with above-median mean importance.

    Returns (stability_score ∈ [0,1], list_of_unstable_feature_names).
    """
    if len(importances_list) < 2:
        return 0.5, []

    all_features = sorted(set().union(*(set(d.keys()) for d in importances_list)))
    if not all_features:
        return 0.5, []

    matrix = np.zeros((len(importances_list), len(all_features)))
    for i, fold_imp in enumerate(importances_list):
        for j, feat in enumerate(all_features):
            matrix[i, j] = float(fold_imp.get(feat, 0.0))

    eps       = 1e-9
    mean_imp  = matrix.mean(axis=0)
    std_imp   = matrix.std(axis=0)
    cov       = std_imp / (mean_imp + eps)

    # Focus stability measure on above-median importance features
    threshold = max(np.median(mean_imp), eps)
    important = mean_imp > threshold

    stability_per = np.clip(1.0 - cov, 0.0, 1.0)
    stability = float(stability_per[important].mean()) if important.any() else float(stability_per.mean())

    unstable_mask = (cov > 0.5) & important
    unstable = [all_features[j] for j in range(len(all_features)) if unstable_mask[j]]

    return round(stability, 4), unstable[:10]


def _compute_multiple_testing_penalty(
    n_tests: int,
    n_features: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """
    Compute a 0-1 penalty for multiple testing bias.

    Returns (penalty_score, bonferroni_threshold).
    """
    effective_n = max(n_tests, 1)
    bonferroni  = alpha / effective_n

    # Feature count penalty: every feature is a potential data snoop
    feature_penalty = max(0.0, min(0.5, (n_features - 10) / 40.0))

    # Test count penalty on log scale (1 → 0, 10 → ~0.24, 100 → ~0.5, 1000 → ~0.75)
    test_penalty = min(0.5, math.log(effective_n + 1) / math.log(1000))

    penalty = round(min(1.0, feature_penalty + test_penalty), 4)
    return penalty, round(bonferroni, 6)


def _compute_min_backtest_length(
    sr_hat: float,
    n_trials: int,
    alpha: float = 0.05,
    annual_factor: int = 252,
) -> float:
    """
    Minimum OOS length (years) for the observed SR to be statistically
    significant at level alpha after Bonferroni correction for N trials.

    Formula: T_years = [Φ^{-1}(1 − α/N) / SR_annualised]²
    """
    if sr_hat <= 0.0 or n_trials < 1:
        return float("inf")
    try:
        z = norm.ppf(1.0 - alpha / max(n_trials, 1))
        years = (z / sr_hat) ** 2
        return round(years, 2)
    except Exception:
        return float("inf")


def _severity_from_score(score: float) -> str:
    for upper, label, _ in _SEVERITY_LEVELS:
        if score < upper:
            return label
    return "SEVERE"


def _has_is_oos(fold: dict) -> bool:
    return "is_sharpe" in fold and ("oos_sharpe" in fold or "sharpe_ratio" in fold)


# ── Recommendation / warning builders ────────────────────────────────────────

def _build_warnings(
    gap_score: float,
    gap_details: dict,
    dsr: float,
    dsr_p_value: float,
    pbo: float,
    rank_corr: float,
    feat_stability: float,
    unstable: list[str],
    n_tests: int,
    n_features: int,
    oos_years: float,
    min_btl: float,
    mt_penalty: float,
) -> list[str]:
    w: list[str] = []
    sr_gap_pct = gap_details.get("sharpe_gap_pct", 0.0)

    if sr_gap_pct > 50:
        w.append(
            f"Sharpe degrades {sr_gap_pct:.0f}% IS→OOS "
            f"({gap_details['sharpe_is']:.2f} → {gap_details['sharpe_oos']:.2f}) — "
            "strong in-sample overfitting"
        )
    elif sr_gap_pct > 25:
        w.append(
            f"Sharpe degrades {sr_gap_pct:.0f}% IS→OOS "
            f"({gap_details['sharpe_is']:.2f} → {gap_details['sharpe_oos']:.2f}) — "
            "moderate overfitting detected"
        )

    wr_drop = gap_details.get("win_rate_is", 0.5) - gap_details.get("win_rate_oos", 0.5)
    if wr_drop > 0.08:
        w.append(
            f"Win rate drops {wr_drop*100:.1f}pp OOS "
            f"({gap_details['win_rate_is']*100:.1f}% → {gap_details['win_rate_oos']*100:.1f}%) — "
            "signals are less reliable out-of-sample"
        )

    if dsr < 0.5:
        w.append(
            f"Deflated Sharpe Ratio = {dsr:.3f} — after correcting for {n_tests} trials, "
            "the Sharpe ratio is not statistically significant (DSR < 0.5)"
        )
    elif dsr_p_value > 0.30:
        w.append(
            f"DSR p-value = {dsr_p_value:.3f} — {dsr_p_value*100:.0f}% probability "
            "the Sharpe is a false positive from multiple testing"
        )

    if pbo > 0.45:
        w.append(
            f"PBO estimate = {pbo:.3f} — {pbo*100:.0f}% probability of backtest overfitting; "
            f"IS/OOS rank correlation = {rank_corr:.3f}"
        )
    elif pbo > 0.35:
        w.append(
            f"PBO estimate = {pbo:.3f} — moderate risk; "
            f"IS/OOS rank correlation = {rank_corr:.3f}"
        )

    if feat_stability < 0.40:
        w.append(
            f"Feature importance stability = {feat_stability:.3f} — "
            "importances vary greatly across folds; model may be learning noise"
        )

    if n_tests > 50:
        w.append(
            f"{n_tests} strategy configurations tested — severe multiple testing risk; "
            f"Bonferroni significance threshold = {0.05/n_tests:.5f} (not 0.05)"
        )
    elif n_tests > 20:
        w.append(
            f"{n_tests} configurations tested — elevated multiple testing risk; "
            f"apparent edge may not survive Bonferroni correction"
        )

    if math.isfinite(min_btl) and oos_years < min_btl:
        w.append(
            f"Insufficient OOS data: have {oos_years:.1f} years, need ~{min_btl:.1f} years "
            f"for {n_tests} trials at 5% significance"
        )

    if n_features > 25:
        w.append(
            f"High feature count ({n_features}) expands the search space and "
            "increases the risk of spurious correlations"
        )

    return w


def _build_recommendations(
    gap_score: float,
    gap_details: dict,
    dsr: float,
    dsr_p_value: float,
    pbo: float,
    rank_corr: float,
    feat_stability: float,
    unstable: list[str],
    n_tests: int,
    n_features: int,
    oos_years: float,
    min_btl: float,
    mt_penalty: float,
    severity: str,
) -> list[str]:
    r: list[str] = []
    sr_gap_pct = gap_details.get("sharpe_gap_pct", 0.0)

    # 1. Feature reduction
    if n_features > 20 or sr_gap_pct > 40:
        r.append(
            f"Reduce features to 8–12 using MI + SHAP selection "
            f"(currently {n_features}): "
            "`python main.py features --ticker <TICKER> --select`"
        )

    # 2. Regularisation
    if sr_gap_pct > 35 or dsr_p_value > 0.5:
        r.append(
            "Increase XGBoost regularisation to reduce in-sample overfitting: "
            "set `lambda: 5.0`, `alpha: 1.0`, `min_child_weight: 5`, `max_depth: 3` "
            "in the `model` section of config.yaml"
        )

    # 3. Confidence threshold
    if dsr_p_value > 0.35 or gap_score > 0.4:
        r.append(
            "Raise the signal confidence threshold to filter low-conviction trades: "
            "set `backtest.signal_threshold: 0.65` in config.yaml "
            "(default is 0.55)"
        )

    # 4. Embargo extension
    if pbo > 0.40:
        r.append(
            "Extend the walk-forward embargo period to prevent label leakage: "
            "set `validation.purge_gap_bars: 42` in config.yaml "
            "(default 21 bars ≈ 1 month)"
        )

    # 5. Feature stability — specific unstable features
    if feat_stability < 0.50 and unstable:
        feat_str = ", ".join(f"'{f}'" for f in unstable[:5])
        r.append(
            f"Remove or constrain unstable features: {feat_str}. "
            "These features have high importance variance across folds, suggesting "
            "they encode fold-specific noise rather than a structural signal"
        )

    # 6. Conservative position sizing (HIGH/SEVERE)
    if severity in ("HIGH", "SEVERE"):
        r.append(
            "Apply conservative position sizing while the edge is uncertain: "
            "set `risk.kelly_fraction: 0.10` and "
            "`risk.target_annual_volatility: 0.10` in config.yaml"
        )

    # 7. Independent holdout (HIGH/SEVERE)
    if severity in ("HIGH", "SEVERE"):
        r.append(
            "Reserve a final OOS holdout (last 20% of your data, never touched during "
            "model development) and test on it exactly once before any further validation"
        )

    # 8. More data
    if math.isfinite(min_btl) and oos_years < min_btl:
        shortfall = round(min_btl - oos_years, 1)
        r.append(
            f"Collect ~{shortfall:.1f} more years of OOS data, or reduce the number "
            f"of strategy trials (currently {n_tests}) to lower the MinBTL requirement"
        )

    return r


# ── Rich display helpers ──────────────────────────────────────────────────────

def _comp_row(table: Table, label: str, weight: float, score: float) -> None:
    if score < 0.25:
        level = "[green]Low[/]"
    elif score < 0.50:
        level = "[yellow]Moderate[/]"
    elif score < 0.75:
        level = "[red]High[/]"
    else:
        level = "[bold red]Severe[/]"
    table.add_row(
        label,
        f"{weight*100:.0f}%",
        f"{score:.3f}",
        f"{score * weight:.3f}",
        level,
    )


def _delta_cell(delta: float, pct: bool = False, invert: bool = False) -> str:
    """Colour-coded delta cell: green if positive (or negative when invert=True)."""
    is_good = (delta >= 0) if not invert else (delta <= 0)
    color = "green" if is_good else "red"
    sign  = "+" if delta >= 0 else ""
    if pct:
        return f"[{color}]{sign}{delta*100:.1f}pp[/]"
    return f"[{color}]{sign}{delta:.3f}[/]"
