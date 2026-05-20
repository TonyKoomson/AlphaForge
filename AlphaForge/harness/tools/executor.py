"""
AlphaForge AI Harness — Tool Executor

Dispatches tool calls (from Claude or Grok) to the actual implementation
functions that wrap the existing AlphaForge infrastructure.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from utils.helpers import get_logger

logger = get_logger(__name__)


def _offline_fallback(query: str) -> dict:
    """Return curated domain knowledge when web search is unavailable."""
    q = query.lower()
    if any(w in q for w in ["momentum", "trend"]):
        text = (
            "Momentum factor (12-1 month) is one of the most robust equity factors. "
            "Works best in trending markets (VIX < 20, SMA50 above SMA200). "
            "Crashes after market reversals — combine with volatility filter to reduce whipsaw. "
            "Academic refs: Jegadeesh & Titman (1993), Asness et al. (2013)."
        )
    elif any(w in q for w in ["value", "book"]):
        text = (
            "Value factor (P/B, P/E) underperformed 2015-2020 but recovered in 2022. "
            "Works best in rising rate environments and post-recession periods. "
            "Combine with quality screen (profitability) to avoid value traps. "
            "Academic refs: Fama & French (1993), Asness & Frazzini (2013)."
        )
    elif any(w in q for w in ["mean reversion", "rsi", "reversal"]):
        text = (
            "Short-term mean reversion (5-day) is strongest in high-vol regimes. "
            "RSI below 30 signals overextension; Bollinger Band squeeze precedes breakouts. "
            "Hold periods of 1-5 days; higher transaction costs vs momentum strategies. "
            "Academic refs: Lehmann (1990), Jegadeesh (1990)."
        )
    elif any(w in q for w in ["regime", "bull", "bear", "macro"]):
        text = (
            "Market regimes: Bull (SPY > SMA200, low VIX) favours momentum/growth. "
            "Bear (SPY < SMA200, VIX > 25) favours low-vol, defensive, and value. "
            "Transition regimes (high VIX, mixed signals) — reduce exposure and increase threshold. "
            "SMA 50/200 crossover (Death Cross) is a reliable bear signal with 6-month lag."
        )
    elif any(w in q for w in ["volatility", "vix", "vol"]):
        text = (
            "Volatility regime matters more than direction. Low-vol (VIX < 15) → momentum works. "
            "High-vol (VIX > 25) → mean reversion and low-vol factor outperform. "
            "ATR expansion after contraction signals breakout; BB squeeze is a leading indicator. "
            "Volatility-adjusted position sizing reduces drawdowns significantly."
        )
    else:
        text = (
            "General quantitative finance context: factor strategies (momentum, value, quality, "
            "low-vol) each have distinct regime dependencies. Purged walk-forward CV with a 21-bar "
            "embargo is best practice for avoiding look-ahead bias. IS/OOS Sharpe gap > 50% "
            "is a strong signal of overfitting. OOS Sharpe > 0.8 with max DD < 25% is promotion threshold."
        )
    return {"source": "embedded_domain_knowledge", "text": text}


class ToolExecutor:
    """Dispatch tool-call requests from agents to real implementation functions."""

    def __init__(self, knowledge_base=None) -> None:
        from harness.memory.knowledge_base import KnowledgeBase
        self.kb: KnowledgeBase = knowledge_base or KnowledgeBase()
        self._n_trials: int = 1   # tracks number of strategy trials for DSR computation

    def execute(self, tool_name: str, tool_input: dict) -> str:
        """
        Execute a tool call. Returns a JSON string suitable for feeding
        back into the agent as a tool result.
        """
        try:
            handler = getattr(self, f"_tool_{tool_name}", None)
            if handler is None:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
            result = handler(**tool_input)
            return json.dumps(result, default=str)
        except Exception as exc:
            logger.debug("Tool %s failed: %s", tool_name, exc)
            return json.dumps({"error": str(exc), "traceback": traceback.format_exc()[-500:]})

    # ── Tool implementations ───────────────────────────────────────────────────

    def _tool_run_backtest(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end: str   = "2023-12-31",
        features:   list[str] | None = None,
        model_params: dict | None = None,
        signal_threshold: float = 0.55,
        use_spy_timing: bool = True,
    ) -> dict:
        import joblib
        import numpy as np
        import pandas as pd
        from data.ingest import DataIngestion
        from features.feature_cache import get_or_compute_features
        from backtest.engine import run_backtest, CostModel
        from utils.helpers import load_config

        cfg = load_config()
        raw_dir = ROOT / cfg.get("data", {}).get("cache_dir", "data/raw")
        safe = ticker.lower().replace(".", "_").replace("-", "_")
        raw_path = raw_dir / f"{safe}_daily.parquet"

        if not raw_path.exists():
            DataIngestion(config=cfg).download_data(ticker, start="2015-01-01", end=end)

        feats = get_or_compute_features(ticker, raw_path, config=cfg, as_of_date=end)
        if feats is None:
            return {"error": f"Could not compute features for {ticker}"}

        feats = feats[(feats.index >= start) & (feats.index <= end)]
        if len(feats) < 50:
            return {"error": f"Insufficient data for {ticker} in {start}–{end} ({len(feats)} bars)"}

        # Load trained model
        model_path = ROOT / "models" / "artifacts" / f"{safe}_model.joblib"
        if not model_path.exists():
            return {"error": f"No trained model for {ticker} at {model_path}. Run 'train' first."}

        try:
            model = joblib.load(model_path)
            avail_cols = [c for c in model.feature_columns if c in feats.columns]
            if len(avail_cols) < 5:
                return {"error": f"Too few matching features ({len(avail_cols)}) between model and computed features"}

            X = feats[avail_cols].ffill().fillna(0)
            # predict() returns (mean_pred, std_pred, lower, upper)
            predict_out = model.predict(X)
            probas = predict_out[0] if isinstance(predict_out, tuple) else predict_out
            if hasattr(probas, "values"):
                probas = probas.values

            # Long-only signals: buy when P(up) >= threshold
            sigs = pd.Series(0, index=feats.index, dtype=int)
            sigs.iloc[probas >= signal_threshold] = 1

            costs = CostModel.for_stock()
            result = run_backtest(sigs, feats, costs=costs, label=ticker)
            metrics = result.metrics if hasattr(result, "metrics") else {}
            if not metrics and isinstance(result, dict):
                metrics = result

            sharpe_val = round(float(metrics.get("sharpe_ratio", 0.0) or 0.0), 3)
            n_bars_val = len(feats)

            # Statistical significance (Bailey & Lopez de Prado, 2012)
            try:
                from harness.stats import sharpe_tstat, deflated_sharpe_ratio, probabilistic_sharpe_ratio
                n_trials = getattr(self, "_n_trials", 1)
                tstat  = sharpe_tstat(sharpe_val, n_bars_val)
                psr    = probabilistic_sharpe_ratio(sharpe_val, n_bars_val)
                dsr    = deflated_sharpe_ratio(sharpe_val, n_bars_val, n_trials=n_trials)
                significance = {
                    "sharpe_t_stat":      tstat["t_stat"],
                    "sharpe_p_value":     tstat["p_value"],
                    "n_years":            tstat["n_years"],
                    "significant_at_05":  tstat["significant_at_05"],
                    "psr":                psr["psr"],
                    "dsr":                dsr["dsr"],
                    "dsr_sr_star":        dsr["sr_star_ann"],
                    "dsr_n_trials":       n_trials,
                    "dsr_verdict":        dsr["interpretation"],
                }
                # Increment trial counter for next call
                self._n_trials = n_trials + 1
            except Exception:
                significance = {}

            return {
                "ticker":      ticker,
                "period":      f"{start} to {end}",
                "sharpe":      sharpe_val,
                "ann_return":  round(float(metrics.get("ann_return_pct", metrics.get("total_return_net", 0.0)) or 0.0), 3),
                "max_dd":      round(float((metrics.get("max_drawdown", 0.0) or 0.0) * 100), 3),
                "n_trades":    int(metrics.get("total_trades", 0) or 0),
                "win_rate":    round(float(metrics.get("win_rate", 0.0) or 0.0), 3),
                "cost_drag":   round(float(metrics.get("total_costs_pct", 0.0) or 0.0), 3),
                "signal_threshold": signal_threshold,
                "n_bars":      n_bars_val,
                "n_features":  len(avail_cols),
                **significance,
            }
        except Exception as exc:
            import traceback
            return {"error": str(exc), "traceback": traceback.format_exc()}

    def _tool_train_model(
        self,
        ticker: str,
        start: str = "2015-01-01",
        end: str   = "2023-12-31",
        features:  list[str] | None = None,
        model_params: dict | None = None,
        target_horizon: int = 5,
    ) -> dict:
        import subprocess, sys
        cmd = [
            sys.executable, str(ROOT / "main.py"), "train",
            "--ticker", ticker,
            "--start", start,
            "--end", end,
        ]
        if target_horizon != 5:
            cmd += ["--target-horizon", str(target_horizon)]

        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)

        # Find most-recently written training_metrics_*.json (timestamped filenames)
        artifacts_dir = ROOT / "models" / "artifacts"
        candidates = sorted(
            artifacts_dir.glob("training_metrics_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        metrics_path = candidates[0] if candidates else None
        if metrics_path and metrics_path.exists():
            import json as _json
            m = _json.loads(metrics_path.read_text())
            fold_results = m.get("fold_results", [])
            oos_sharpes = [
                f.get("ml_strategy_metrics", {}).get("sharpe_like", 0)
                for f in fold_results
                if "ml_strategy_metrics" in f
            ]
            avg_oos_sharpe = round(float(sum(oos_sharpes) / max(len(oos_sharpes), 1)), 3)
            return {
                "ticker":        ticker,
                "trained_at":    m.get("trained_at", ""),
                "feature_count": m.get("feature_count", 0),
                "oos_sharpe":    avg_oos_sharpe,
                "n_folds":       len(fold_results),
                "top_features":  list(m.get("feature_importance_shap", {}).keys())[:10],
                "stdout_tail":   proc.stdout[-500:] if proc.stdout else "",
            }
        return {
            "status": "trained" if proc.returncode == 0 else "error",
            "stdout": proc.stdout[-1000:],
            "stderr": proc.stderr[-500:],
        }

    def _tool_fetch_market_data(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end:   str = "2023-12-31",
    ) -> dict:
        import numpy as np
        from data.ingest import DataIngestion
        from utils.helpers import load_config

        cfg = load_config()
        raw_dir = ROOT / cfg.get("data", {}).get("cache_dir", "data/raw")
        safe = ticker.lower().replace(".", "_").replace("-", "_")
        raw_path = raw_dir / f"{safe}_daily.parquet"

        if not raw_path.exists():
            DataIngestion(config=cfg).download_data(ticker, start="2015-01-01", end=end)

        import pandas as pd
        df = pd.read_parquet(raw_path)
        df = df[(df.index >= start) & (df.index <= end)]
        if df.empty:
            return {"error": f"No data for {ticker} in range {start}–{end}"}

        ret = df["close"].pct_change().dropna()
        ann_ret = float((1 + ret.mean()) ** 252 - 1)
        ann_vol = float(ret.std() * np.sqrt(252))
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
        total_ret = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)

        return {
            "ticker":       ticker,
            "start":        str(df.index[0].date()),
            "end":          str(df.index[-1].date()),
            "n_bars":       len(df),
            "total_return": round(total_ret * 100, 2),
            "ann_return":   round(ann_ret * 100, 2),
            "ann_vol":      round(ann_vol * 100, 2),
            "sharpe_bnh":   round(sharpe, 3),
            "max_dd":       round(float((df["close"] / df["close"].cummax() - 1).min() * 100), 2),
            "avg_volume_m": round(float(df["volume"].mean() / 1e6), 1),
        }

    def _tool_compute_features(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end:   str = "2023-12-31",
        top_n: int = 20,
    ) -> dict:
        import numpy as np
        from data.ingest import DataIngestion
        from features.feature_cache import get_or_compute_features
        from utils.helpers import load_config

        cfg = load_config()
        raw_dir = ROOT / cfg.get("data", {}).get("cache_dir", "data/raw")
        safe = ticker.lower().replace(".", "_").replace("-", "_")
        raw_path = raw_dir / f"{safe}_daily.parquet"

        if not raw_path.exists():
            DataIngestion(config=cfg).download_data(ticker, start="2015-01-01", end=end)

        feats = get_or_compute_features(ticker, raw_path, config=cfg, as_of_date=end)
        if feats is None:
            return {"error": f"Feature computation failed for {ticker}"}

        feats = feats[(feats.index >= start) & (feats.index <= end)]
        skip = {"label", "fwd_return", "tb_label", "tb_ret", "meta_label",
                "regime", "open", "high", "low", "close", "volume"}
        feat_cols = [c for c in feats.columns if c not in skip]

        # Compute IC (information coefficient) with 5-bar forward return
        ic_map: dict[str, float] = {}
        if "fwd_return" in feats.columns:
            fwd = feats["fwd_return"].shift(-5)
            for col in feat_cols:
                corr = feats[col].corr(fwd)
                if not np.isnan(corr):
                    ic_map[col] = round(float(corr), 4)

        top = sorted(ic_map.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]

        return {
            "ticker":          ticker,
            "n_features":      len(feat_cols),
            "all_features":    feat_cols,
            "top_by_ic":       {k: v for k, v in top},
            "sample_row":      {c: round(float(feats[c].iloc[-1]), 4) for c in feat_cols[:15]
                                if feats[c].dtype in ("float64", "float32", "int64")},
        }

    def _tool_add_alpha_factor(
        self,
        factor_name: str,
        python_code: str,
        hypothesis: str,
    ) -> dict:
        """Validate and register a new alpha factor."""
        import ast, importlib, textwrap

        # 1. Syntax check
        try:
            ast.parse(python_code)
        except SyntaxError as e:
            return {"success": False, "error": f"Syntax error: {e}"}

        # 2. Look-ahead bias check (heuristic — flag dangerous patterns)
        danger_patterns = [".shift(-", "pct_change(-", "rolling(", "expanding(", "iloc[-"]
        flags = [p for p in danger_patterns if p in python_code and p == ".shift(-"]
        if flags:
            return {
                "success": False,
                "error":   "Potential look-ahead bias: use of .shift(-N) suggests forward-looking data",
            }

        # 3. Validate function signature
        func_name = f"compute_{factor_name}"
        if func_name not in python_code:
            return {
                "success": False,
                "error":   f"Code must define a function named '{func_name}'",
            }

        # 4. Save to custom factors directory
        factors_dir = ROOT / "features" / "custom_factors"
        factors_dir.mkdir(exist_ok=True)
        factor_file = factors_dir / f"{factor_name}.py"
        header = (
            f'"""Auto-generated alpha factor: {factor_name}\n'
            f'Hypothesis: {hypothesis}\n"""\n'
            "import numpy as np\nimport pandas as pd\n\n"
        )
        factor_file.write_text(header + python_code)

        # 5. Test-compile the factor
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(factor_name, factor_file)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            has_fn = hasattr(mod, func_name)
        except Exception as exc:
            factor_file.unlink(missing_ok=True)
            return {"success": False, "error": f"Import error: {exc}"}

        if not has_fn:
            factor_file.unlink(missing_ok=True)
            return {"success": False, "error": f"Function {func_name} not found after import"}

        # 6. Save to KB
        self.kb.save_finding(
            title=f"Alpha factor added: {factor_name}",
            insight=f"Hypothesis: {hypothesis}. Saved to features/custom_factors/{factor_name}.py",
            tags=[factor_name, "custom_factor"],
        )

        return {
            "success":     True,
            "factor_name": factor_name,
            "func_name":   func_name,
            "file":        str(factor_file),
            "message":     f"Factor '{factor_name}' added. Include it in the features list when training.",
        }

    def _tool_run_validation(
        self,
        ticker: str,
        end: str = "2023-12-31",
    ) -> dict:
        import subprocess, sys
        cmd = [sys.executable, str(ROOT / "main.py"), "validate", "--ticker", ticker, "--end", end]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=600)

        metrics_path = ROOT / "models" / "artifacts" / f"training_metrics_{ticker.lower()}.json"
        if metrics_path.exists():
            import json as _json
            m = _json.loads(metrics_path.read_text())
            fold_results = m.get("fold_results", [])
            oos_sharpes = [
                f.get("ml_strategy_metrics", {}).get("sharpe_like", 0)
                for f in fold_results if "ml_strategy_metrics" in f
            ]
            is_sharpes = [
                f.get("baseline_strategy_metrics", {}).get("sharpe_like", 0)
                for f in fold_results if "baseline_strategy_metrics" in f
            ]
            avg_oos = float(sum(oos_sharpes) / max(len(oos_sharpes), 1))
            avg_is  = float(sum(is_sharpes)  / max(len(is_sharpes),  1))
            gap     = (avg_is - avg_oos) / max(abs(avg_is), 1e-9)
            verdict = ("STRONG" if avg_oos > 0.8 else
                       "MODERATE" if avg_oos > 0.4 else
                       "WEAK" if avg_oos > 0 else "NO_EDGE")
            return {
                "ticker":        ticker,
                "oos_sharpe":    round(avg_oos, 3),
                "is_sharpe":     round(avg_is, 3),
                "is_oos_gap":    round(gap, 3),
                "n_folds":       len(fold_results),
                "verdict":       verdict,
                "overfitting":   "HIGH" if gap > 0.5 else "MODERATE" if gap > 0.25 else "LOW",
            }
        return {
            "status": "done" if proc.returncode == 0 else "error",
            "output": proc.stdout[-1000:],
        }

    def _tool_run_universe_trade(
        self,
        universe: str = "sp500",
        top_n: int = 10,
        start: str = "2020-01-01",
        end: str   = "2023-12-31",
        ranking_factor: str = "model",
        spy_timing: str = "sma50_200",
        stop_loss_pct: float = 0.0,
    ) -> dict:
        import subprocess, sys
        cmd = [
            sys.executable, str(ROOT / "main.py"), "universe-trade",
            "--universe", universe,
            "--top-n", str(top_n),
            "--start", start,
            "--end", end,
            "--ranking-factor", ranking_factor,
            "--spy-timing", spy_timing,
            "--stop-loss", str(stop_loss_pct),
            "--min-dollar-volume", "0",
            "--calibrated-scores",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=900)
        output = proc.stdout

        # Parse result table from output
        result: dict[str, Any] = {"universe": universe, "top_n": top_n, "period": f"{start} to {end}"}
        for line in output.split("\n"):
            for key, label in [
                ("total_return",  "Total Return"),
                ("ann_return",    "Ann. Return"),
                ("sharpe",        "Sharpe Ratio"),
                ("max_dd",        "Max Drawdown"),
                ("n_trades",      "Total Trades"),
                ("avg_positions", "Avg Positions"),
                ("final_nav",     "Final NAV"),
            ]:
                if label in line:
                    try:
                        val = line.split("│")[-2].strip().replace("%", "").replace("$", "").replace(",", "").replace("+", "")
                        result[key] = float(val)
                    except Exception:
                        pass
        result["status"] = "ok" if proc.returncode == 0 else "error"
        return result

    def _tool_search_knowledge_base(
        self,
        query: str,
        entry_type: str | None = None,
        min_sharpe: float | None = None,
        limit: int = 5,
    ) -> dict:
        results = self.kb.search(query=query, entry_type=entry_type,
                                 min_sharpe=min_sharpe, limit=limit)
        return {"results": results, "count": len(results)}

    def _tool_save_finding(
        self,
        entry_type: str,
        title: str,
        body: str,
        tags: list[str] | None = None,
    ) -> dict:
        entry_id = self.kb.save(entry_type=entry_type, title=title, body=body, tags=tags)
        return {"saved": True, "id": entry_id, "type": entry_type}

    def _tool_web_search(self, query: str, context: str = "") -> dict:
        """
        Perform a web search using DuckDuckGo Instant Answer API (no API key required).
        Falls back to a curated offline summary for common quantitative finance queries.
        """
        import urllib.request
        import urllib.parse

        # Enrich query with financial context
        enriched = query
        if context == "factor_research":
            enriched = f"{query} quantitative finance factor alpha academic research"
        elif context == "macro_context":
            enriched = f"{query} macro market regime equity"
        elif context == "sector_rotation":
            enriched = f"{query} sector rotation US equity market"
        elif context == "regime_analysis":
            enriched = f"{query} market regime VIX volatility trend"

        try:
            encoded = urllib.parse.urlencode({"q": enriched, "format": "json", "no_html": "1", "skip_disambig": "1"})
            url = f"https://api.duckduckgo.com/?{encoded}"
            req = urllib.request.Request(url, headers={"User-Agent": "AlphaForge-Research/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            abstract = data.get("AbstractText", "")
            if abstract:
                results.append({"source": data.get("AbstractSource", ""), "text": abstract[:500]})

            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append({"source": topic.get("FirstURL", ""), "text": topic["Text"][:300]})

            if not results:
                return {
                    "query": query,
                    "status": "no_results",
                    "note": "No instant answer found. Consider refining the query.",
                }

            return {
                "query":   query,
                "context": context,
                "results": results[:6],
                "n":       len(results),
                "status":  "ok",
            }

        except Exception as exc:
            # Offline fallback: return curated domain knowledge for common queries
            return {
                "query":  query,
                "status": "offline_fallback",
                "note":   f"Web search unavailable ({exc}). Using embedded domain knowledge.",
                "results": [_offline_fallback(query)],
            }

    def _tool_get_model_metrics(self, ticker: str) -> dict:
        import json as _json

        # Find most-recently written training_metrics_*.json
        artifacts_dir = ROOT / "models" / "artifacts"
        candidates = sorted(
            artifacts_dir.glob("training_metrics_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        metrics_path = candidates[0] if candidates else None
        if not metrics_path or not metrics_path.exists():
            return {"error": f"No training metrics found for {ticker}. Run train_model first."}

        m = _json.loads(metrics_path.read_text())
        fold_results = m.get("fold_results", [])
        oos_sharpes = [
            f.get("ml_strategy_metrics", {}).get("sharpe_like", 0)
            for f in fold_results if "ml_strategy_metrics" in f
        ]
        shap = m.get("feature_importance_shap", {})
        top_features = sorted(shap.items(), key=lambda x: x[1], reverse=True)[:15]

        return {
            "ticker":        ticker,
            "trained_at":    m.get("trained_at", ""),
            "feature_count": m.get("feature_count", 0),
            "target_horizon":m.get("target_horizon", 5),
            "oos_sharpe_avg":round(sum(oos_sharpes) / max(len(oos_sharpes), 1), 3),
            "n_folds":       len(fold_results),
            "top_features":  {k: round(v, 4) for k, v in top_features},
            "fold_sharpes":  [round(s, 3) for s in oos_sharpes],
        }
