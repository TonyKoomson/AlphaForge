"""
Alpha Forge — Final Validation Report
======================================
Produces a comprehensive, publication-quality report comparing:

  1. Momentum baseline  (theory)
  2. Momentum baseline  (realistic costs)
  3. ML model           (theory — zero costs)
  4. ML model           (realistic costs)
  5. ML + risk management (volatility-targeted sizing, realistic costs)
  6. ML + execution simulator (full fill simulation)

For each variant the report records: Sharpe, Sortino, CAGR, Max Drawdown,
Max DD Duration, Win Rate, Profit Factor, Total Trades, Cost Drag.

The report also includes:
  - Regime breakdown (Bull / Bear / High-Vol / Neutral)
  - Purged walk-forward fold results
  - Top-20 feature importances (from trained model)
  - Feature drift summary (PSI reference vs recent)
  - Final verdict & recommendation

Outputs
-------
  reports/<ticker>_report_<ts>.json      machine-readable results
  reports/<ticker>_report_<ts>.md        human-readable Markdown
  reports/<ticker>_report_<ts>.pdf       PDF (requires weasyprint; skipped otherwise)
  reports/<ticker>_report_<ts>.png       multi-panel figure (always written)

Usage
-----
  python main.py report --ticker SPY
  python main.py report --ticker SPY --format markdown
  python main.py report --ticker SPY --output reports/spy/
"""

from __future__ import annotations

import json
import textwrap
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from backtest.engine import BacktestEngine, CostModel, BacktestResult, run_backtest
from execution.simulator import simulate_execution
from features.engine import FeatureEngine, _get_feature_cols
from features.momentum import MomentumStrategy
from models.train import ModelTrainer
from risk.capacity import CapacityEstimator
from risk.position_sizing import PositionSizer
from utils.helpers import compute_all_metrics, ensure_dir, get_logger, load_config

logger = get_logger(__name__)
console = Console()

# ---------------------------------------------------------------------------
# Colour palette — consistent across all charts
# ---------------------------------------------------------------------------
_COLORS = {
    "baseline_theoretical": "#9E9E9E",
    "baseline_realistic":   "#607D8B",
    "ml_theoretical":       "#64B5F6",
    "ml_realistic":         "#2196F3",
    "ml_risk_managed":      "#4CAF50",
    "ml_executed":          "#FF9800",
    "benchmark":            "#E0E0E0",
}

_STRATEGY_LABELS = {
    "baseline_theoretical": "Baseline (theory)",
    "baseline_realistic":   "Baseline (w/ costs)",
    "ml_theoretical":       "ML (theory)",
    "ml_realistic":         "ML (realistic)",
    "ml_risk_managed":      "ML + Risk Mgmt",
    "ml_executed":          "ML + Execution Sim",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _metrics_from_result(r: BacktestResult) -> dict:
    m = r.metrics.copy()
    m["total_costs_pct"] = m.get("total_costs_pct", 0.0)
    m["theory_vs_real_gap"] = m.get("theory_vs_real_gap", 0.0)
    return m


def _metrics_from_sim(sim: dict, initial_capital: float = 100_000.0) -> dict:
    eq = sim.get("equity_curve", pd.Series(dtype=float))
    if eq.empty:
        return {}
    ret = eq.pct_change().dropna()
    m = compute_all_metrics(ret, eq)
    m["total_trades"] = int(len(sim.get("trade_log", pd.DataFrame())))
    m["total_costs_pct"] = 0.0
    return m


def _regime_label(code: int) -> str:
    return {0: "bear", 1: "neutral", 2: "bull", 3: "high_vol"}.get(code, "unknown")


def _psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna(), cur.dropna()
    if len(ref) < 10 or len(cur) < 5:
        return 0.0
    breaks = np.unique(np.percentile(ref, np.linspace(0, 100, bins + 1)))
    if len(breaks) < 3:
        return 0.0
    r, _ = np.histogram(ref, bins=breaks)
    c, _ = np.histogram(cur, bins=breaks)
    r_pct = np.clip(r / max(r.sum(), 1), 1e-6, 1.0)
    c_pct = np.clip(c / max(c.sum(), 1), 1e-6, 1.0)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))


def _progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


# ---------------------------------------------------------------------------
# ValidationReport
# ---------------------------------------------------------------------------

class ValidationReport:
    """
    Full validation report comparing all strategy variants.

    Backward-compatible contract: ``run(features, ticker)`` works as before.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg         = config or load_config()
        val_cfg          = self.cfg.get("validation", {})
        self.n_splits    = val_cfg.get("n_splits", 5)
        self.purge_gap   = val_cfg.get("purge_gap_bars", 10)
        self.output_dir  = ensure_dir(val_cfg.get("metrics_output_dir", "reports"))
        self.initial_capital = float(
            self.cfg.get("backtest", {}).get("initial_capital", 100_000.0)
        )
        self.threshold   = float(self.cfg.get("backtest", {}).get("signal_threshold", 0.55))

        self.engine         = BacktestEngine(config=self.cfg)
        self.feature_engine = FeatureEngine(config=self.cfg)
        self.trainer        = ModelTrainer(config=self.cfg)
        self.sizer          = PositionSizer(config=self.cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, features: pd.DataFrame, ticker: str) -> dict:
        """
        Execute the full validation pipeline and write all outputs.

        Returns the complete results dict (also saved as JSON).
        """
        ts_label = datetime.now().strftime("%Y%m%d_%H%M")
        logger.info("=== VALIDATION REPORT: %s ===", ticker)

        console.print(Panel(
            f"[bold cyan]Alpha Forge — Full Validation Report[/]\n"
            f"Ticker: [bold]{ticker}[/]   "
            f"Bars: [bold]{len(features):,}[/]   "
            f"Period: [dim]{features.index.min().date()} → {features.index.max().date()}[/]",
            expand=False,
        ))

        results: dict = {
            "ticker":       ticker,
            "generated_at": datetime.now().isoformat(),
            "data_period":  f"{features.index.min().date()} → {features.index.max().date()}",
            "n_bars":       len(features),
        }

        with _progress() as prog:
            total = 10
            t = prog.add_task("Loading model…", total=total)

            try:
                self.trainer.load(ticker)
                ml_loaded = True
            except Exception:
                ml_loaded = False
                logger.warning("No trained model found for %s — ML variants skipped", ticker)

            # 1. Signals
            close = features["close"]
            feat_cols = [c for c in self.feature_engine.feature_columns if c in features.columns]
            X = features[feat_cols]

            ml_signals = pd.Series(0, index=features.index)
            ml_probas  = np.full(len(features), 0.5)
            if ml_loaded:
                ml_probas  = self.trainer.predict_proba(X)
                ml_signals = pd.Series(
                    np.where(ml_probas > self.threshold, 1,
                             np.where(ml_probas < 1 - self.threshold, -1, 0)),
                    index=X.index,
                )

            momentum = MomentumStrategy()
            base_signals = momentum.generate(features)

            prog.advance(t); prog.update(t, description="Running baseline (theory)…")

            # 2. Baseline theory
            base_theory  = run_backtest(base_signals, close, costs=CostModel.zero(),
                                        initial_capital=self.initial_capital, label="baseline_theory")

            prog.advance(t); prog.update(t, description="Running baseline (realistic)…")

            # 3. Baseline realistic
            base_real    = run_backtest(base_signals, close,
                                        costs=CostModel.for_stock(),
                                        initial_capital=self.initial_capital, label="baseline_real")

            prog.advance(t); prog.update(t, description="Running ML (theory)…")

            # 4. ML theory
            ml_theory    = run_backtest(ml_signals, close, costs=CostModel.zero(),
                                        initial_capital=self.initial_capital, label="ml_theory") \
                           if ml_loaded else None

            prog.advance(t); prog.update(t, description="Running ML (realistic)…")

            # 5. ML realistic
            ml_real      = run_backtest(ml_signals, close, costs=CostModel.for_stock(),
                                        initial_capital=self.initial_capital, label="ml_real") \
                           if ml_loaded else None

            prog.advance(t); prog.update(t, description="Running ML + risk management…")

            # 6. ML + risk management (volatility-targeted sizing)
            ml_risk_signals = None
            ml_risk_result  = None
            if ml_loaded:
                rolling_vol = close.pct_change().rolling(21).std().mul(np.sqrt(252))
                ml_risk_signals = self.sizer.compute_position_series(
                    ml_signals, close,
                    rolling_vol=rolling_vol,
                    initial_equity=self.initial_capital,
                )
                ml_risk_result = run_backtest(
                    ml_risk_signals, close, costs=CostModel.for_stock(),
                    initial_capital=self.initial_capital, label="ml_risk"
                )

            prog.advance(t); prog.update(t, description="Running ML + execution simulator…")

            # 7. ML + full execution simulation
            ml_exec_result = None
            if ml_loaded:
                try:
                    ml_exec_result = simulate_execution(
                        target_positions=ml_signals.astype(float),
                        prices=features[["close", "high", "low", "volume"]]
                        if all(c in features.columns for c in ["high", "low", "volume"])
                        else features[["close"]],
                        volume=features["volume"] if "volume" in features.columns else None,
                        mode="realistic",
                        initial_capital=self.initial_capital,
                        config=self.cfg,
                    )
                except Exception as exc:
                    logger.warning("Execution simulation failed: %s", exc)

            prog.advance(t); prog.update(t, description="Regime breakdown…")

            # 8. Regime breakdown
            regime_data = self._regime_breakdown(features, ml_signals if ml_loaded else base_signals)

            prog.advance(t); prog.update(t, description="Walk-forward validation…")

            # 9. Purged walk-forward
            wf_results = self._walk_forward_validation(features, ticker, feat_cols)

            prog.advance(t); prog.update(t, description="Feature importance & drift…")

            # 10. Feature importance
            feat_importance = self._feature_importance(ticker, feat_cols) if ml_loaded else {}

            # 11. Drift analysis
            drift_data = self._drift_analysis(X)

            prog.advance(t); prog.update(t, description="Overfitting analysis…")

            # 12. Overfitting detection (needs strategies + walk-forward; built after loop)
            # Placeholder — computed below once results dict is populated.
            prog.advance(t)

        # Package results
        def _safe_metrics(r):
            return _metrics_from_result(r) if r is not None else {}

        results["strategies"] = {
            "baseline_theoretical": _safe_metrics(base_theory),
            "baseline_realistic":   _safe_metrics(base_real),
            "ml_theoretical":       _safe_metrics(ml_theory),
            "ml_realistic":         _safe_metrics(ml_real),
            "ml_risk_managed":      _safe_metrics(ml_risk_result),
            "ml_executed":          _metrics_from_sim(ml_exec_result, self.initial_capital)
                                    if ml_exec_result else {},
        }
        results["regime_breakdown"] = regime_data
        results["walk_forward"]     = wf_results
        results["feature_importance"] = feat_importance
        results["drift"]            = drift_data
        results["summary"]          = self._build_summary(results)

        # 12. Overfitting detection
        overfit_report = self._run_overfitting_detection(results, feat_cols)
        results["overfitting_analysis"] = overfit_report.to_dict()

        # 13. Capacity estimation
        results["capacity"] = self._capacity_analysis(results)

        # Render & save
        self._print_console_report(results, overfit_report=overfit_report)
        fig_path = self._generate_figure(
            results,
            {
                "baseline_theoretical": base_theory,
                "baseline_realistic":   base_real,
                "ml_theoretical":       ml_theory,
                "ml_realistic":         ml_real,
                "ml_risk_managed":      ml_risk_result,
                "ml_executed":          ml_exec_result,
            },
            ticker, ts_label,
        )
        results["figure_path"] = str(fig_path) if fig_path else ""

        json_path = self._write_json(results, ticker, ts_label)
        md_path   = self._write_markdown(results, ticker, ts_label, fig_path,
                                         overfit_report=overfit_report)
        pdf_path  = self._write_pdf(md_path, ticker, ts_label)

        # Final file list
        files_table = Table(show_header=False, box=None)
        files_table.add_column(style="dim", min_width=10)
        files_table.add_column()
        files_table.add_row("JSON",     str(json_path))
        files_table.add_row("Markdown", str(md_path))
        if pdf_path:
            files_table.add_row("PDF",  str(pdf_path))
        if fig_path:
            files_table.add_row("Figure", str(fig_path))
        console.print(Panel(files_table, title="[green]Report files written[/]", expand=False))

        return results

    # ------------------------------------------------------------------
    # Strategy runners
    # ------------------------------------------------------------------

    def _regime_breakdown(self, features: pd.DataFrame, signals: pd.Series) -> dict:
        regime_col = "regime" if "regime" in features.columns else None
        if regime_col is None:
            from features.engine import add_regime_labels
            features = add_regime_labels(features)
            regime_col = "regime"

        out: dict[str, dict] = {}
        for code in sorted(features[regime_col].dropna().unique()):
            name = _regime_label(int(code))
            mask = features[regime_col] == code
            if mask.sum() < 30:
                continue
            try:
                r = run_backtest(
                    signals[mask], features.loc[mask, "close"],
                    costs=CostModel.for_stock(),
                    initial_capital=self.initial_capital,
                )
                out[name] = {
                    "n_bars":       int(mask.sum()),
                    "sharpe_ratio": round(r.metrics.get("sharpe_ratio", 0), 3),
                    "cagr":         round(r.metrics.get("cagr", 0), 4),
                    "max_drawdown": round(r.metrics.get("max_drawdown", 0), 4),
                    "win_rate":     round(r.metrics.get("win_rate", 0), 4),
                    "total_trades": int(r.metrics.get("total_trades", 0)),
                }
            except Exception as exc:
                out[name] = {"error": str(exc)}
        return out

    def _walk_forward_validation(
        self, features: pd.DataFrame, ticker: str, feat_cols: list[str]
    ) -> list[dict]:
        X = features[feat_cols]
        y = features["label"]
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            purge_end = train_idx[-1]
            purged_train_idx = train_idx[train_idx < purge_end - self.purge_gap]
            if len(purged_train_idx) < 50:
                continue

            X_train = X.iloc[purged_train_idx]
            y_train = y.iloc[purged_train_idx]
            X_test  = X.iloc[test_idx]

            try:
                import xgboost as xgb
                scaler   = StandardScaler()
                X_tr_sc  = scaler.fit_transform(X_train)
                X_te_sc  = scaler.transform(X_test)
                model    = xgb.XGBClassifier(
                    n_estimators=self.cfg["model"].get("n_estimators", 200),
                    max_depth=self.cfg["model"].get("max_depth", 4),
                    learning_rate=self.cfg["model"].get("learning_rate", 0.05),
                    random_state=42, eval_metric="logloss", verbosity=0,
                )
                model.fit(X_tr_sc, y_train, verbose=False)
                probas  = model.predict_proba(X_te_sc)[:, 1]
                signals = pd.Series(
                    np.where(probas > self.threshold, 1,
                             np.where(probas < 1 - self.threshold, -1, 0)),
                    index=X_test.index,
                )
                bt = self.engine.run(features.iloc[test_idx], signals, ticker)
                m  = bt.metrics.copy()

                # IS Sharpe: quick estimate from model predictions on training slice
                probas_is = model.predict_proba(X_tr_sc)[:, 1]
                sig_is = pd.Series(
                    np.where(probas_is > self.threshold, 1,
                             np.where(probas_is < 1 - self.threshold, -1, 0)),
                    index=X_train.index,
                )
                close_is = features.iloc[purged_train_idx]["close"]
                ret_is   = close_is.pct_change().dropna()
                sig_shifted = sig_is.shift(1).reindex(ret_is.index).fillna(0)
                strat_ret   = sig_shifted * ret_is
                std_is = strat_ret.std()
                m["is_sharpe"] = round(
                    float(strat_ret.mean() / std_is * np.sqrt(252)) if std_is > 1e-9 else 0.0,
                    4,
                )
            except Exception as exc:
                logger.warning("WF fold %d failed: %s", fold_idx + 1, exc)
                m = {}

            m.update({
                "fold":        fold_idx + 1,
                "train_start": str(X_train.index[0].date()),
                "train_end":   str(X_train.index[-1].date()),
                "test_start":  str(X_test.index[0].date()),
                "test_end":    str(X_test.index[-1].date()),
                "n_train":     len(X_train),
                "n_test":      len(X_test),
            })
            fold_results.append(m)
            logger.info(
                "WF fold %d | Sharpe=%.2f | CAGR=%.1f%% | MaxDD=%.1f%%",
                fold_idx + 1,
                m.get("sharpe_ratio", 0),
                m.get("cagr", 0) * 100,
                m.get("max_drawdown", 0) * 100,
            )
        return fold_results

    def _feature_importance(self, ticker: str, feat_cols: list[str]) -> dict:
        try:
            imp = self.trainer.feature_importance(feat_cols)
            return imp.head(20).round(6).to_dict()
        except Exception:
            return {}

    def _run_overfitting_detection(self, results: dict, feat_cols: list[str]):
        """Run overfitting detection against the already-computed strategy metrics."""
        from validation.overfitting_detector import detect_overfitting

        strats    = results.get("strategies", {})
        is_m      = strats.get("ml_theoretical",     strats.get("baseline_theoretical", {}))
        oos_m     = strats.get("ml_realistic",        strats.get("baseline_realistic",   {}))
        wf        = results.get("walk_forward", [])
        feat_imp  = results.get("feature_importance", {})

        n_feat    = max(len(feat_cols), 1)
        # Estimate n_tests: walk-forward folds × feature groups + 6 strategy variants
        n_tests   = int(self.cfg.get("overfitting", {}).get(
            "n_tests_estimate",
            max(self.n_splits * max(1, n_feat // 5) + 6, 10),
        ))
        oos_bars  = results.get("n_bars", 252) // max(len(wf) + 1, 2)

        return detect_overfitting(
            in_sample_metrics      = is_m,
            out_of_sample_metrics  = oos_m,
            feature_count          = n_feat,
            number_of_tests        = n_tests,
            fold_metrics           = wf,
            feature_importances    = [feat_imp] if feat_imp else None,
            oos_bars               = oos_bars,
        )

    def _capacity_analysis(self, results: dict) -> dict:
        """Estimate capital capacity from the ML realistic backtest Sharpe and trade count."""
        strats = results.get("strategies", {})
        ml_m   = strats.get("ml_realistic", strats.get("baseline_realistic", {}))
        sharpe_0 = float(ml_m.get("sharpe_ratio", 0.0))

        n_bars  = max(results.get("n_bars", 252), 1)
        n_trades = int(ml_m.get("total_trades", 4))
        annual_turnover = max(n_trades / n_bars * 252, 0.5)

        cap_cfg = self.cfg.get("capacity", {})
        adv_usd         = float(cap_cfg.get("adv_usd", 50_000_000))
        impact_exponent = float(cap_cfg.get("impact_exponent", 0.60))
        base_slippage   = float(cap_cfg.get("base_slippage", 0.0005))
        max_adv_frac    = float(cap_cfg.get("max_adv_fraction", 0.05))

        try:
            estimator = CapacityEstimator(
                adv_usd=adv_usd,
                impact_exponent=impact_exponent,
                base_slippage=base_slippage,
                max_adv_fraction=max_adv_frac,
            )
            cap_result = estimator.estimate(
                sharpe_0=sharpe_0,
                annual_turnover=annual_turnover,
                backtest_capital=self.initial_capital,
            )
            d = cap_result.to_dict()
            d["annual_turnover"] = round(annual_turnover, 2)
            d["sharpe_0"] = round(sharpe_0, 3)
            return d
        except Exception as exc:
            logger.warning("Capacity analysis failed: %s", exc)
            return {"error": str(exc)}

    def _drift_analysis(self, X: pd.DataFrame) -> dict:
        split  = int(len(X) * 0.70)
        X_ref  = X.iloc[:split]
        X_cur  = X.iloc[split:]
        scores = {}
        for col in X_ref.columns[:30]:
            scores[col] = _psi(X_ref[col], X_cur[col])
        threshold = self.cfg.get("monitoring", {}).get("feature_drift_threshold", 0.05)
        top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
        return {
            "mean_psi":             round(float(np.mean(list(scores.values()))), 5),
            "max_psi":              round(float(max(scores.values(), default=0)), 5),
            "n_drifted":            int(sum(v > threshold for v in scores.values())),
            "threshold":            threshold,
            "top_drifted":          [{"feature": k, "psi": round(v, 5)} for k, v in top],
            "drift_alert":          any(v > threshold for v in scores.values()),
        }

    # ------------------------------------------------------------------
    # Summary & verdict
    # ------------------------------------------------------------------

    def _build_summary(self, results: dict) -> dict:
        strats  = results.get("strategies", {})
        wf      = results.get("walk_forward", [])
        drift   = results.get("drift", {})

        ml_real = strats.get("ml_realistic", {})
        ml_risk = strats.get("ml_risk_managed", {})
        base_r  = strats.get("baseline_realistic", {})
        ml_th   = strats.get("ml_theoretical", {})

        sharpe_ml   = ml_real.get("sharpe_ratio", 0.0)
        sharpe_base = base_r.get("sharpe_ratio", 0.0)
        sharpe_th   = ml_th.get("sharpe_ratio", 0.0)
        sharpe_risk = ml_risk.get("sharpe_ratio", 0.0)

        wf_sharpes = [r.get("sharpe_ratio", 0) for r in wf if "sharpe_ratio" in r]
        mean_oos   = float(np.mean(wf_sharpes)) if wf_sharpes else 0.0
        std_oos    = float(np.std(wf_sharpes))  if wf_sharpes else 0.0

        theory_gap = 0.0
        if abs(sharpe_th) > 1e-9:
            theory_gap = (sharpe_th - sharpe_ml) / abs(sharpe_th) * 100

        edge_survives = sharpe_ml > 0.5
        risk_helps    = sharpe_risk > sharpe_ml + 0.05

        overfitting_score = max(0.0, (sharpe_th - mean_oos) / max(abs(sharpe_th), 1e-9))

        verdict, recommendation = _verdict_and_recommendation(
            mean_oos_sharpe=mean_oos,
            edge_survives=edge_survives,
            overfitting_score=overfitting_score,
            drift_alert=drift.get("drift_alert", False),
            ml_beats_baseline=sharpe_ml > sharpe_base + 0.1,
        )

        return {
            "mean_oos_sharpe":              round(mean_oos, 3),
            "std_oos_sharpe":               round(std_oos, 3),
            "n_folds_positive_sharpe":      int(sum(s > 0 for s in wf_sharpes)),
            "n_folds":                      len(wf_sharpes),
            "sharpe_ml_realistic":          round(sharpe_ml, 3),
            "sharpe_ml_risk_managed":       round(sharpe_risk, 3),
            "sharpe_baseline_realistic":    round(sharpe_base, 3),
            "ml_beats_baseline":            bool(sharpe_ml > sharpe_base + 0.1),
            "risk_mgmt_improves_sharpe":    bool(risk_helps),
            "theory_vs_reality_gap_pct":    round(theory_gap, 2),
            "edge_survives_costs":          edge_survives,
            "overfitting_score":            round(overfitting_score, 3),
            "drift_alert":                  drift.get("drift_alert", False),
            "verdict":                      verdict,
            "recommendation":               recommendation,
        }

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------

    def _print_console_report(self, results: dict, overfit_report=None) -> None:
        summary = results["summary"]
        strats  = results["strategies"]

        # Strategy comparison table
        comp = Table(
            title="[bold]Strategy Comparison[/]",
            show_header=True, header_style="bold cyan",
        )
        comp.add_column("Strategy",      min_width=22)
        comp.add_column("Sharpe",        justify="right")
        comp.add_column("CAGR",          justify="right")
        comp.add_column("Max DD",        justify="right")
        comp.add_column("Win Rate",      justify="right")
        comp.add_column("Profit Factor", justify="right")
        comp.add_column("Trades",        justify="right")
        comp.add_column("Cost Drag",     justify="right")

        for key, label in _STRATEGY_LABELS.items():
            m = strats.get(key, {})
            if not m:
                continue
            sh  = m.get("sharpe_ratio", 0)
            col = "green" if sh > 0.5 else "yellow" if sh > 0 else "red"
            comp.add_row(
                label,
                f"[{col}]{sh:.2f}[/]",
                f"{m.get('cagr', 0)*100:.1f}%",
                f"{m.get('max_drawdown', 0)*100:.1f}%",
                f"{m.get('win_rate', 0)*100:.1f}%",
                f"{m.get('profit_factor', 0):.2f}",
                str(int(m.get("total_trades", 0))),
                f"{m.get('total_costs_pct', 0)*100:.3f}%",
            )
        console.print(comp)

        # Regime table
        if results.get("regime_breakdown"):
            reg = Table(title="[bold]Regime Performance (ML Realistic)[/]",
                        show_header=True, header_style="bold magenta")
            reg.add_column("Regime",    min_width=10)
            reg.add_column("Bars",      justify="right")
            reg.add_column("Sharpe",    justify="right")
            reg.add_column("CAGR",      justify="right")
            reg.add_column("Max DD",    justify="right")
            reg.add_column("Win Rate",  justify="right")
            for name, m in results["regime_breakdown"].items():
                if "error" in m:
                    continue
                sh  = m.get("sharpe_ratio", 0)
                col = "green" if sh > 0 else "red"
                reg.add_row(
                    name.upper(),
                    str(m.get("n_bars", 0)),
                    f"[{col}]{sh:.2f}[/]",
                    f"{m.get('cagr', 0)*100:.1f}%",
                    f"{m.get('max_drawdown', 0)*100:.1f}%",
                    f"{m.get('win_rate', 0)*100:.1f}%",
                )
            console.print(reg)

        # Walk-forward table
        if results.get("walk_forward"):
            wf_t = Table(title="[bold]Walk-Forward Fold Results[/]",
                         show_header=True, header_style="bold yellow")
            for col in ["Fold", "Train Start", "Test Start", "Test End",
                        "Sharpe", "CAGR", "Max DD", "Win Rate"]:
                wf_t.add_column(col)
            for r in results["walk_forward"]:
                sh  = r.get("sharpe_ratio", 0)
                col = "green" if sh > 0 else "red"
                wf_t.add_row(
                    str(r.get("fold", "")),
                    str(r.get("train_start", "")),
                    str(r.get("test_start", "")),
                    str(r.get("test_end", "")),
                    f"[{col}]{sh:.3f}[/]",
                    f"{r.get('cagr', 0)*100:.1f}%",
                    f"{r.get('max_drawdown', 0)*100:.1f}%",
                    f"{r.get('win_rate', 0)*100:.1f}%",
                )
            console.print(wf_t)

        # Summary verdict
        s = summary
        ov_col  = "green" if s["overfitting_score"] < 0.3 else \
                  "yellow" if s["overfitting_score"] < 0.7 else "red"
        sh_col  = "green" if s["mean_oos_sharpe"] > 0.5 else \
                  "yellow" if s["mean_oos_sharpe"] > 0 else "red"
        verdict_col = "green" if "STRONG" in s["verdict"] else \
                      "yellow" if "MODERATE" in s["verdict"] else "red"

        console.print(Panel(
            f"[bold]Mean OOS Sharpe:[/]         [{sh_col}]{s['mean_oos_sharpe']:.3f}[/]  "
            f"(±{s['std_oos_sharpe']:.3f}, {s['n_folds_positive_sharpe']}/{s['n_folds']} folds positive)\n"
            f"[bold]ML vs Baseline:[/]          "
            f"{'[green]+' if s['ml_beats_baseline'] else '[red]–'}{'better' if s['ml_beats_baseline'] else 'not better'}[/]\n"
            f"[bold]Risk Mgmt Benefit:[/]       "
            f"{'[green]yes[/]' if s['risk_mgmt_improves_sharpe'] else '[dim]no[/]'}\n"
            f"[bold]Theory vs Reality Gap:[/]   {s['theory_vs_reality_gap_pct']:.1f}%\n"
            f"[bold]Overfitting Score:[/]        [{ov_col}]{s['overfitting_score']:.2f}[/] "
            f"(0=none, 1=full degradation)\n"
            f"[bold]Feature Drift Alert:[/]     "
            f"{'[bold red]YES[/]' if s['drift_alert'] else '[green]no[/]'}\n\n"
            f"[bold {verdict_col}]Verdict:[/] [{verdict_col}]{s['verdict']}[/]\n\n"
            f"[dim]{s['recommendation']}[/]",
            title="[bold cyan]Alpha Forge — Validation Summary[/]",
            expand=False,
        ))

        # Overfitting detection block
        if overfit_report is not None:
            from validation.overfitting_detector import print_report as _print_overfit
            _print_overfit(overfit_report)

        # Capacity block
        cap = results.get("capacity", {})
        if cap and "error" not in cap:
            def _fmt_usd(v):
                return f"${v:,.0f}" if isinstance(v, (int, float)) else str(v)
            slip_col = "green" if cap.get("passes_slippage_gate") else "red"
            slip_txt = "[green]PASS[/]" if cap.get("passes_slippage_gate") else "[red]FAIL[/]"
            cap_panel_text = (
                f"[bold]Frictionless Sharpe:[/]    {cap.get('sharpe_0', 0):.3f}  "
                f"(annual turnover {cap.get('annual_turnover', 0):.1f}×)\n"
                f"[bold]Optimal capacity:[/]       {_fmt_usd(cap.get('optimal_capacity_usd', 0))}  "
                f"(Sharpe degrades to half here)\n"
                f"[bold]Max capacity:[/]           {_fmt_usd(cap.get('max_capacity_usd', 0))}  "
                f"(Sharpe ≥ 0.30)\n"
                f"[bold]Sharpe at $100K / $1M:[/]  "
                f"{cap.get('sharpe_at_100k', 0):.3f} / {cap.get('sharpe_at_1m', 0):.3f}\n"
                f"[bold]Sharpe at $10M / $100M:[/] "
                f"{cap.get('sharpe_at_10m', 0):.3f} / {cap.get('sharpe_at_100m', 0):.3f}\n"
                f"[bold]3× Slippage stress:[/]     {slip_txt}  "
                f"(SR={cap.get('sr_at_3x_slippage', 0):.3f})\n"
                f"[bold]ADV fraction at C*:[/]     {cap.get('adv_fraction_at_optimal', 0):.2%}"
                + (f"\n[yellow]{cap['warning']}[/]" if cap.get("warning") else "")
            )
            console.print(Panel(
                cap_panel_text,
                title="[bold cyan]Capacity Estimation[/]",
                expand=False,
            ))

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def _generate_figure(
        self,
        results: dict,
        bt_results: dict,
        ticker: str,
        ts_label: str,
    ) -> Optional[Path]:
        try:
            return self._draw_figure(results, bt_results, ticker, ts_label)
        except Exception as exc:
            logger.warning("Figure generation failed: %s", exc)
            logger.debug(traceback.format_exc())
            return None

    def _draw_figure(
        self,
        results: dict,
        bt_results: dict,
        ticker: str,
        ts_label: str,
    ) -> Path:
        fig = plt.figure(figsize=(18, 22))
        fig.patch.set_facecolor("#1a1a2e")
        gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

        ax_eq   = fig.add_subplot(gs[0, :])
        ax_dd   = fig.add_subplot(gs[1, :])
        ax_wf   = fig.add_subplot(gs[2, 0])
        ax_reg  = fig.add_subplot(gs[2, 1])
        ax_imp  = fig.add_subplot(gs[3, 0])
        ax_mnth = fig.add_subplot(gs[3, 1])

        _dark_ax(ax_eq)
        _dark_ax(ax_dd)
        _dark_ax(ax_wf)
        _dark_ax(ax_reg)
        _dark_ax(ax_imp)
        _dark_ax(ax_mnth)

        fig.suptitle(
            f"Alpha Forge — Validation Report  [{ticker}]  {ts_label}",
            fontsize=14, fontweight="bold", color="white", y=0.99,
        )

        # — Panel 1: Equity curves —
        ax_eq.set_title("Strategy Equity Curves", color="white")
        ax_eq.set_ylabel("Portfolio Value ($)", color="white")
        for key, r in bt_results.items():
            eq = None
            if isinstance(r, BacktestResult) and r is not None:
                eq = r.equity_curve
            elif isinstance(r, dict) and r and "equity_curve" in r:
                eq = r["equity_curve"]
            if eq is None or eq.empty:
                continue
            ax_eq.plot(eq.index, eq.values,
                       color=_COLORS.get(key, "white"),
                       linewidth=1.4 if "ml" in key else 0.9,
                       linestyle="--" if "theoretical" in key else "-",
                       label=_STRATEGY_LABELS.get(key, key), alpha=0.9)
        ax_eq.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_eq.legend(fontsize=8, loc="upper left", facecolor="#1a1a2e", labelcolor="white")
        ax_eq.grid(alpha=0.15)

        # — Panel 2: Drawdown curves —
        ax_dd.set_title("Drawdown Curves", color="white")
        ax_dd.set_ylabel("Drawdown (%)", color="white")
        for key, r in bt_results.items():
            eq = None
            if isinstance(r, BacktestResult) and r is not None:
                eq = r.equity_curve
            elif isinstance(r, dict) and r and "equity_curve" in r:
                eq = r["equity_curve"]
            if eq is None or eq.empty:
                continue
            dd = (eq - eq.cummax()) / eq.cummax() * 100
            ax_dd.fill_between(dd.index, dd.values, 0,
                               color=_COLORS.get(key, "white"), alpha=0.18)
            ax_dd.plot(dd.index, dd.values,
                       color=_COLORS.get(key, "white"), linewidth=0.8,
                       label=_STRATEGY_LABELS.get(key, key))
        ax_dd.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        ax_dd.legend(fontsize=7, loc="lower left", facecolor="#1a1a2e", labelcolor="white")
        ax_dd.grid(alpha=0.15)

        # — Panel 3: Walk-forward Sharpe bars —
        wf = results.get("walk_forward", [])
        if wf:
            folds   = [r["fold"] for r in wf]
            sharpes = [r.get("sharpe_ratio", 0) for r in wf]
            colors  = ["#4CAF50" if s > 0 else "#F44336" for s in sharpes]
            ax_wf.bar(folds, sharpes, color=colors, alpha=0.8, edgecolor="none")
            ax_wf.axhline(0, color="white", linewidth=0.8)
            ax_wf.axhline(0.5, color="#FFD700", linewidth=0.8, linestyle="--", alpha=0.6)
            ax_wf.set_xlabel("Fold", color="white")
            ax_wf.set_ylabel("Sharpe Ratio", color="white")
            ax_wf.set_title("Walk-Forward OOS Sharpe", color="white")
            ax_wf.grid(alpha=0.15, axis="y")

        # — Panel 4: Regime performance —
        regime_data = results.get("regime_breakdown", {})
        if regime_data:
            names   = [n.upper() for n, m in regime_data.items() if "sharpe_ratio" in m]
            sharpes = [regime_data[n.lower()].get("sharpe_ratio", 0) for n in names]
            rc      = ["#4CAF50" if s > 0 else "#F44336" for s in sharpes]
            ax_reg.barh(names, sharpes, color=rc, alpha=0.85, edgecolor="none")
            ax_reg.axvline(0, color="white", linewidth=0.8)
            ax_reg.set_xlabel("Sharpe Ratio", color="white")
            ax_reg.set_title("Sharpe by Market Regime", color="white")
            ax_reg.grid(alpha=0.15, axis="x")

        # — Panel 5: Feature importance —
        feat_imp = results.get("feature_importance", {})
        if feat_imp:
            imp_s  = pd.Series(feat_imp).sort_values(ascending=True).tail(15)
            colors = ["#64B5F6"] * len(imp_s)
            ax_imp.barh(imp_s.index, imp_s.values, color=colors, alpha=0.85, edgecolor="none")
            ax_imp.set_xlabel("Importance", color="white")
            ax_imp.set_title("Top Feature Importances (ML Model)", color="white")
            ax_imp.grid(alpha=0.15, axis="x")

        # — Panel 6: Monthly returns heatmap (ML realistic) —
        ml_real_r = bt_results.get("ml_realistic")
        if ml_real_r is not None and isinstance(ml_real_r, BacktestResult):
            _plot_monthly_heatmap(ml_real_r.equity_curve, ax_mnth, title="Monthly Returns — ML Realistic")

        path = self.output_dir / f"{ticker.lower()}_report_{ts_label}.png"
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Figure saved: %s", path)
        return path

    # ------------------------------------------------------------------
    # Markdown output
    # ------------------------------------------------------------------

    def _write_markdown(
        self,
        results: dict,
        ticker: str,
        ts_label: str,
        fig_path: Optional[Path],
        overfit_report=None,
    ) -> Path:
        summary = results["summary"]
        strats  = results["strategies"]
        s = summary

        lines = [
            f"# Alpha Forge — Validation Report: {ticker}",
            "",
            f"> Generated: `{results['generated_at']}`  |  "
            f"Period: `{results['data_period']}`  |  Bars: `{results['n_bars']:,}`",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]

        # Key numbers at a glance
        lines += [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Mean OOS Sharpe (walk-forward) | **{s['mean_oos_sharpe']:.3f}** ±{s['std_oos_sharpe']:.3f} |",
            f"| Folds with positive Sharpe | {s['n_folds_positive_sharpe']} / {s['n_folds']} |",
            f"| ML vs Baseline | {'✅ ML better' if s['ml_beats_baseline'] else '❌ No improvement'} |",
            f"| Risk management helps | {'✅ Yes' if s['risk_mgmt_improves_sharpe'] else '—'} |",
            f"| Theory vs Reality gap | {s['theory_vs_reality_gap_pct']:.1f}% |",
            f"| Overfitting score | {s['overfitting_score']:.2f} (0=none, 1=full) |",
            f"| Feature drift alert | {'⚠️ YES' if s['drift_alert'] else '✅ No'} |",
            "",
            f"### Verdict",
            "",
            f"**{s['verdict']}**",
            "",
            f"{s['recommendation']}",
            "",
            "---",
            "",
            "## Strategy Comparison",
            "",
            "| Strategy | Sharpe | Sortino | CAGR | Max DD | Win Rate | Profit Factor | Trades | Cost Drag |",
            "|----------|--------|---------|------|--------|----------|---------------|--------|-----------|",
        ]
        for key, label in _STRATEGY_LABELS.items():
            m = strats.get(key, {})
            if not m:
                continue
            lines.append(
                f"| {label} "
                f"| {m.get('sharpe_ratio', 0):.3f} "
                f"| {m.get('sortino_ratio', 0):.3f} "
                f"| {m.get('cagr', 0)*100:.1f}% "
                f"| {m.get('max_drawdown', 0)*100:.1f}% "
                f"| {m.get('win_rate', 0)*100:.1f}% "
                f"| {m.get('profit_factor', 0):.2f} "
                f"| {int(m.get('total_trades', 0))} "
                f"| {m.get('total_costs_pct', 0)*100:.3f}% |"
            )

        lines += ["", "---", "", "## Walk-Forward Fold Results", "",
                  "| Fold | Train Start | Train End | Test Start | Test End | "
                  "Sharpe | CAGR | Max DD | Win Rate |",
                  "|------|-------------|-----------|------------|----------|"
                  "--------|------|--------|----------|"]
        for r in results.get("walk_forward", []):
            lines.append(
                f"| {r.get('fold','')} "
                f"| {r.get('train_start','')} "
                f"| {r.get('train_end','')} "
                f"| {r.get('test_start','')} "
                f"| {r.get('test_end','')} "
                f"| {r.get('sharpe_ratio',0):.3f} "
                f"| {r.get('cagr',0)*100:.1f}% "
                f"| {r.get('max_drawdown',0)*100:.1f}% "
                f"| {r.get('win_rate',0)*100:.1f}% |"
            )

        lines += ["", "---", "", "## Regime Performance Breakdown", "",
                  "| Regime | Bars | Sharpe | CAGR | Max DD | Win Rate | Trades |",
                  "|--------|------|--------|------|--------|----------|--------|"]
        for name, m in results.get("regime_breakdown", {}).items():
            if "error" in m:
                continue
            lines.append(
                f"| {name.upper()} "
                f"| {m.get('n_bars', 0)} "
                f"| {m.get('sharpe_ratio', 0):.3f} "
                f"| {m.get('cagr', 0)*100:.1f}% "
                f"| {m.get('max_drawdown', 0)*100:.1f}% "
                f"| {m.get('win_rate', 0)*100:.1f}% "
                f"| {m.get('total_trades', 0)} |"
            )

        # Feature importance
        feat_imp = results.get("feature_importance", {})
        if feat_imp:
            lines += ["", "---", "", "## Top Feature Importances", "",
                      "| Feature | Importance |",
                      "|---------|-----------|"]
            for feat, imp in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]:
                lines.append(f"| `{feat}` | {imp:.5f} |")

        # Drift
        drift = results.get("drift", {})
        if drift:
            lines += ["", "---", "", "## Feature Drift Analysis", "",
                      f"- Mean PSI across features: `{drift.get('mean_psi', 0):.5f}`",
                      f"- Max PSI: `{drift.get('max_psi', 0):.5f}` (threshold: `{drift.get('threshold', 0.05)}`)",
                      f"- Features above threshold: `{drift.get('n_drifted', 0)}`",
                      f"- Drift alert: `{'YES ⚠️' if drift.get('drift_alert') else 'No ✅'}`",
                      ""]
            if drift.get("top_drifted"):
                lines += ["**Top drifted features:**", ""]
                for item in drift["top_drifted"]:
                    lines.append(f"- `{item['feature']}` — PSI = `{item['psi']:.5f}`")

        # Capacity analysis section
        cap = results.get("capacity", {})
        if cap and "error" not in cap:
            def _fmt_usd_md(v):
                return f"${v:,.0f}" if isinstance(v, (int, float)) else str(v)
            lines += [
                "", "---", "", "## Capacity Estimation", "",
                f"> Based on frictionless Sharpe of **{cap.get('sharpe_0', 0):.3f}** "
                f"and annual turnover **{cap.get('annual_turnover', 0):.1f}×**.",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| Optimal capacity (half-Sharpe) | **{_fmt_usd_md(cap.get('optimal_capacity_usd', 0))}** |",
                f"| Max capacity (Sharpe ≥ 0.30) | {_fmt_usd_md(cap.get('max_capacity_usd', 0))} |",
                f"| Sharpe at $100K | {cap.get('sharpe_at_100k', 0):.3f} |",
                f"| Sharpe at $1M | {cap.get('sharpe_at_1m', 0):.3f} |",
                f"| Sharpe at $10M | {cap.get('sharpe_at_10m', 0):.3f} |",
                f"| Sharpe at $100M | {cap.get('sharpe_at_100m', 0):.3f} |",
                f"| 3× slippage stress test | {'✅ PASS' if cap.get('passes_slippage_gate') else '❌ FAIL'} "
                f"(SR={cap.get('sr_at_3x_slippage', 0):.3f}) |",
                f"| ADV fraction at C* | {cap.get('adv_fraction_at_optimal', 0):.2%} |",
            ]
            if cap.get("warning"):
                lines.append(f"\n> ⚠️ {cap['warning']}")

        # Overfitting detection section
        if overfit_report is not None:
            from validation.overfitting_detector import format_report_section as _fmt_overfit
            lines.append(_fmt_overfit(overfit_report))

        # Figure
        if fig_path:
            lines += [
                "", "---", "", "## Charts", "",
                f"![Validation Charts]({fig_path.name})", "",
            ]

        lines += [
            "---",
            "",
            "> **Disclaimer:** This report is for research and educational purposes only. "
            "Alpha Forge does not connect to any broker, does not manage real capital, "
            "and makes no investment recommendations. "
            "Past simulated performance does not guarantee future results.",
            "",
        ]

        path = self.output_dir / f"{ticker.lower()}_report_{ts_label}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Markdown report: %s", path)
        return path

    # ------------------------------------------------------------------
    # PDF output
    # ------------------------------------------------------------------

    def _write_pdf(self, md_path: Path, ticker: str, ts_label: str) -> Optional[Path]:
        pdf_path = self.output_dir / f"{ticker.lower()}_report_{ts_label}.pdf"
        if self._try_weasyprint(md_path, pdf_path):
            return pdf_path
        if self._try_pdfkit(md_path, pdf_path):
            return pdf_path
        logger.info(
            "PDF generation skipped (install weasyprint or pdfkit+wkhtmltopdf). "
            "Markdown report is at %s", md_path
        )
        return None

    def _try_weasyprint(self, md_path: Path, pdf_path: Path) -> bool:
        try:
            import markdown as md_lib
            from weasyprint import HTML, CSS
        except ImportError:
            return False
        try:
            html_body = md_lib.markdown(
                md_path.read_text(encoding="utf-8"),
                extensions=["tables", "fenced_code"],
            )
            html = _html_wrap(html_body)
            HTML(string=html, base_url=str(self.output_dir)).write_pdf(
                str(pdf_path),
                stylesheets=[CSS(string=_PDF_CSS)],
            )
            logger.info("PDF (weasyprint): %s", pdf_path)
            return True
        except Exception as exc:
            logger.debug("weasyprint failed: %s", exc)
            return False

    def _try_pdfkit(self, md_path: Path, pdf_path: Path) -> bool:
        try:
            import markdown as md_lib
            import pdfkit
        except ImportError:
            return False
        try:
            html_body = md_lib.markdown(
                md_path.read_text(encoding="utf-8"),
                extensions=["tables", "fenced_code"],
            )
            pdfkit.from_string(_html_wrap(html_body), str(pdf_path),
                               options={"quiet": ""})
            logger.info("PDF (pdfkit): %s", pdf_path)
            return True
        except Exception as exc:
            logger.debug("pdfkit failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------

    def _write_json(self, results: dict, ticker: str, ts_label: str) -> Path:
        path = self.output_dir / f"{ticker.lower()}_report_{ts_label}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("JSON report: %s", path)
        return path


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _verdict_and_recommendation(
    mean_oos_sharpe: float,
    edge_survives: bool,
    overfitting_score: float,
    drift_alert: bool,
    ml_beats_baseline: bool,
) -> tuple[str, str]:
    if mean_oos_sharpe >= 0.8 and edge_survives and overfitting_score < 0.3:
        verdict = "STRONG EDGE"
        rec = (
            "The strategy shows consistent positive risk-adjusted returns out-of-sample "
            "with minimal overfitting. Edge survives realistic transaction costs. "
            "Recommend continuing to paper-trade while monitoring for regime changes and drift."
        )
    elif mean_oos_sharpe >= 0.5 and edge_survives and overfitting_score < 0.6:
        verdict = "MODERATE EDGE"
        rec = (
            "The strategy has a measurable edge that survives costs, but IS/OOS degradation "
            "suggests some overfitting. "
            "Recommend extending the paper-trading window and stress-testing across more regimes "
            "before considering further validation."
        )
    elif mean_oos_sharpe > 0.0 and overfitting_score < 0.8:
        verdict = "WEAK EDGE"
        rec = (
            "Marginal edge after costs — high risk of overfitting or data snooping. "
            "Do NOT proceed without: (1) expanding the dataset, "
            "(2) adding more aggressive walk-forward purging, "
            "(3) a second independent OOS holdout test."
        )
    else:
        verdict = "NO EDGE DETECTED"
        rec = (
            "The strategy does not demonstrate positive risk-adjusted returns out-of-sample. "
            "Re-examine feature engineering, signal generation, and the hypothesis itself "
            "before investing further research time."
        )

    extras = []
    if drift_alert:
        extras.append("⚠ Feature drift detected — consider retraining the model.")
    if not ml_beats_baseline:
        extras.append("⚠ ML model does not outperform the simple momentum baseline — "
                      "model complexity may not be justified.")
    if extras:
        rec += " " + " ".join(extras)
    return verdict, rec


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _dark_ax(ax: plt.Axes) -> None:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#444466")


def _plot_monthly_heatmap(equity: pd.Series, ax: plt.Axes, title: str = "") -> None:
    try:
        ret = equity.pct_change().dropna()
        monthly = ret.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        df = monthly.to_frame("r")
        df["y"] = df.index.year
        df["m"] = df.index.month
        pivot = df.pivot_table("r", "y", "m")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
        import matplotlib.colors as mcolors
        norm  = mcolors.TwoSlopeNorm(vmin=-0.08, vcenter=0, vmax=0.08)
        im    = ax.imshow(pivot.values, cmap="RdYlGn", norm=norm, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=6, color="white")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index.astype(str), fontsize=6, color="white")
        ax.set_title(title or "Monthly Returns", color="white")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val*100:.1f}", ha="center", va="center",
                            fontsize=5, color="black")
    except Exception:
        ax.text(0.5, 0.5, "Monthly heatmap\nnot available",
                ha="center", va="center", transform=ax.transAxes, color="white")


# ---------------------------------------------------------------------------
# HTML / CSS for PDF
# ---------------------------------------------------------------------------

def _html_wrap(body: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>{_PDF_CSS}</style>
</head><body>{body}</body></html>"""


_PDF_CSS = """
body  { font-family: 'Segoe UI', Arial, sans-serif; font-size: 10pt;
         color: #1a1a2e; max-width: 900px; margin: 40px auto; }
h1    { color: #1565C0; border-bottom: 2px solid #1565C0; padding-bottom: 4px; }
h2    { color: #1976D2; margin-top: 28px; }
h3    { color: #1E88E5; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 9pt; }
th    { background: #1565C0; color: white; padding: 5px 8px; text-align: left; }
td    { padding: 4px 8px; border-bottom: 1px solid #ddd; }
tr:nth-child(even) td { background: #f5f8ff; }
code  { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 8.5pt; }
pre   { background: #eef; padding: 10px; border-radius: 4px; font-size: 8pt;
         overflow-x: auto; }
blockquote { border-left: 3px solid #90CAF9; margin: 8px 0; padding-left: 12px;
              color: #555; font-style: italic; }
img   { max-width: 100%; margin: 12px 0; }
hr    { border: none; border-top: 1px solid #ccc; margin: 20px 0; }
"""
