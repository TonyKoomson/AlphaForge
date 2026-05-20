"""
Alpha Forge — Universe Portfolio Simulation (Ranking-Based)
Simulation only. No real broker. No real money.

Loads trained models for every ticker in a universe (e.g., S&P 500), then runs a
bar-by-bar ranking strategy:

  Each day →  Score all trained tickers with their respective models
              Rank by predicted confidence (long probability)
              Long the top-N highest-confidence tickers
              Equal-weight each position (1/N of portfolio)
              Rebalance daily

This mirrors how quantitative systematic funds operate at scale.

Usage:
    from paper_trading.universe_portfolio import UniversePortfolio
    port = UniversePortfolio(universe="sp500", top_n=10, capital=100_000)
    results = port.run(start="2020-01-01", end="2023-12-31")

Or via CLI:
    python main.py universe-trade --universe sp500 --top-n 10 \\
           --start 2020-01-01 --end 2023-12-31
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data.ingest import DataIngestion
from features.engine import FeatureEngine
from utils.helpers import load_config, ensure_dir, get_logger

logger = get_logger(__name__)

_utf8_out = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
    if hasattr(sys.stdout, "buffer") else sys.stdout
console = Console(file=_utf8_out, highlight=False)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# UniversePortfolio
# ---------------------------------------------------------------------------

class UniversePortfolio:
    """
    Ranking-based portfolio: each day, score all stocks and long the top-N.

    Parameters
    ----------
    universe_tickers : list[str]
        All tickers in the universe (e.g., S&P 500 component list).
    top_n : int
        Number of stocks to hold at once (e.g., 20).
    capital : float
        Total starting capital.
    config : dict | None
        Loaded config.yaml dict. If None, loaded from default path.
    confidence_weighted : bool
        If True, weight positions by model confidence instead of equal weight.
    min_confidence : float
        Minimum predicted probability to consider a ticker (default 0.55).
    commission_pct : float
        Commission per trade as fraction of trade value.
    slippage_pct : float
        Slippage per trade as fraction of trade value.
    """

    def __init__(
        self,
        universe_tickers: list[str],
        top_n: int = 10,                           # concentrated top-10 (was 20)
        capital: float = 100_000.0,
        config: Optional[dict] = None,
        confidence_weighted: bool = True,           # weight by conviction (was False)
        min_confidence: float = 0.52,              # calibrated-score threshold (all stocks pass at 0.5469)
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        rebalance_threshold: float = 0.05,         # only rebalance if >5% off target
        bear_min_confidence: float = 0.52,         # same as min_confidence (bear guard off)
        trailing_stop_pct: float = 0.08,           # 8% regime-adaptive trailing stop
        trailing_stop_activate_pct: float = 0.04,  # activate trailing stop at 4% gain
        min_holding_bars: int = 3,                 # hold minimum 3 bars before exit
        use_momentum_filter: bool = False,          # opt-in: buy only SMA50>SMA200 uptrend stocks
        market_breadth_threshold: float = 0.0,     # 0 = disabled; set >0 to halve positions in bear
        vol_adjusted_ranking: bool = False,         # opt-in: rank by confidence/vol_21d
        use_raw_scores: bool = False,              # False = calibrated scores for threshold filtering
        ranking_factor: str = "liquidity",         # "model" | "momentum" | "liquidity" | "composite"
        min_dollar_volume: float = 500_000_000.0, # min $500M daily dollar vol — selects mega-caps
        spy_timing_method: str = "sma50_200",      # "sma50_200" | "price_200" | "price_50" | "none" (disable)
        stop_loss_pct: float = 0.0,                # hard stop: exit if loss >= this fraction (0 = off)
        stop_loss_cooldown: int = 10,              # bars to exclude stopped stock from re-entry
    ) -> None:
        self.universe_tickers = [t.upper() for t in universe_tickers]
        self.top_n = top_n
        self.initial_capital = capital
        self.cfg = config or load_config()
        self.confidence_weighted = confidence_weighted
        self.min_confidence = min_confidence
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.rebalance_threshold = rebalance_threshold
        self.bear_min_confidence = bear_min_confidence
        self.trailing_stop_pct = trailing_stop_pct
        self.trailing_stop_activate_pct = trailing_stop_activate_pct
        self.min_holding_bars = min_holding_bars
        self.use_momentum_filter = use_momentum_filter
        self.market_breadth_threshold = market_breadth_threshold
        self.vol_adjusted_ranking = vol_adjusted_ranking
        self.use_raw_scores = use_raw_scores
        self.ranking_factor = ranking_factor
        self.min_dollar_volume = min_dollar_volume
        self.spy_timing_method = spy_timing_method
        self.stop_loss_pct = stop_loss_pct
        self.stop_loss_cooldown = stop_loss_cooldown

        artifacts_dir = Path(
            self.cfg.get("model", {}).get("artifacts_dir", "models/artifacts")
        )

        # Load all available models (skip tickers without a model)
        self.models: dict[str, object] = {}
        self.scalers: dict[str, object] = {}
        missing: list[str] = []

        for ticker in self.universe_tickers:
            model_path  = artifacts_dir / f"{ticker.lower()}_model.joblib"
            scaler_path = artifacts_dir / f"{ticker.lower()}_scaler.joblib"
            if model_path.exists():
                self.models[ticker] = joblib.load(model_path)
                if scaler_path.exists():
                    self.scalers[ticker] = joblib.load(scaler_path)
            else:
                missing.append(ticker)

        trained = len(self.models)
        console.print(
            f"[cyan]Universe:[/] {len(self.universe_tickers)} tickers | "
            f"[green]{trained} models loaded[/] | "
            + (f"[yellow]{len(missing)} missing[/]" if missing else "[green]all trained[/]")
        )
        if trained < top_n:
            console.print(
                f"[red]WARNING:[/] Only {trained} models available but top_n={top_n}. "
                f"Consider running train-universe first."
            )

        self.trained_tickers = list(self.models.keys())

        # Universal cross-sectional model (fallback for tickers without a per-ticker model)
        self.universal_model  = None
        self.universal_scaler = None
        universal_path = artifacts_dir / "universal_model.joblib"
        if universal_path.exists():
            try:
                self.universal_model = joblib.load(universal_path)
                u_scaler = artifacts_dir / "universal_scaler.joblib"
                if u_scaler.exists():
                    self.universal_scaler = joblib.load(u_scaler)
                console.print(f"[green]Universal model loaded[/] (fallback for untrained tickers)")
            except Exception as exc:
                logger.debug("Failed to load universal model: %s", exc)

        # Scoreable tickers = per-ticker models + universal model covers any ticker with data
        if self.universal_model is not None:
            self.scoreable_tickers = self.universe_tickers
        else:
            self.scoreable_tickers = self.trained_tickers

    # ------------------------------------------------------------------
    # Feature generation (reads from bulk-ingested Parquet cache)
    # ------------------------------------------------------------------

    def _load_features(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Return feature-engineered DataFrame for ticker, or None on failure."""
        try:
            from features.feature_cache import get_or_compute_features

            raw_dir = ROOT / self.cfg.get("data", {}).get("cache_dir", "data/raw")
            safe = ticker.lower().replace(".", "_").replace("-", "_")
            raw_path = raw_dir / f"{safe}_daily.parquet"

            if not raw_path.exists():
                # Fall back to DataIngestion download
                ingester = DataIngestion(config=self.cfg)
                ingester.download_data(ticker, start="2016-01-01", end=end)
                if not raw_path.exists():
                    return None

            feats = get_or_compute_features(
                ticker=ticker,
                raw_path=raw_path,
                config=self.cfg,
                as_of_date=end,
            )
            if feats is None or len(feats) < 30:
                return None

            # Trim to simulation window
            start_ts = pd.Timestamp(start)
            end_ts   = pd.Timestamp(end)
            feats = feats[(feats.index >= start_ts) & (feats.index <= end_ts)]
            return feats if len(feats) >= 20 else None
        except Exception as exc:
            logger.debug("Feature gen failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Scoring — DataFrame-native (fast) and dict fallback
    # ------------------------------------------------------------------

    _LABEL_COLS = frozenset({"label", "fwd_return", "tb_label", "tb_ret", "meta_label",
                              "regime", "open", "high", "low", "close", "volume"})

    def _score_day(self, day_df: pd.DataFrame) -> dict[str, float]:
        """Score all tickers in a date-slice DataFrame in one or few predict calls.

        day_df: index=ticker, columns=features  (already filtered for valid close price)
        Returns {ticker: long_probability}.
        """
        from models.train import TrainedEnsembleModel

        results: dict[str, float] = {}
        if day_df.empty:
            return results

        per_ticker_idx = [t for t in day_df.index if t in self.models]
        universal_idx  = [t for t in day_df.index if t not in self.models]

        # Batch score all universal-model tickers with a single predict call
        if universal_idx and self.universal_model is not None:
            model  = self.universal_model
            scaler = self.universal_scaler
            if isinstance(model, TrainedEnsembleModel):
                X = day_df.loc[universal_idx].reindex(
                    columns=model.feature_columns, fill_value=0.0
                ).fillna(0.0)
                if scaler is not None:
                    try:
                        X = pd.DataFrame(
                            scaler.transform(X.values),
                            index=X.index, columns=X.columns,
                        )
                    except Exception:
                        pass
                try:
                    # Use raw (uncalibrated) scores when enabled — isotonic calibration
                    # has a wide flat step [0.34, 0.61] → 0.5469 that destroys cross-
                    # sectional rank information; raw ensemble scores preserve ordinal signal
                    predict_fn = model.predict_raw if self.use_raw_scores else model.predict
                    probas, _, _, _ = predict_fn(X)
                    for t, p in zip(universal_idx, probas):
                        results[t] = float(np.clip(p, 0.0, 1.0))
                except Exception:
                    pass

        # Per-ticker models (individual calls — rare once universal model covers most tickers)
        for ticker in per_ticker_idx:
            model  = self.models[ticker]
            scaler = self.scalers.get(ticker)
            try:
                if isinstance(model, TrainedEnsembleModel):
                    X = day_df.loc[[ticker]].reindex(
                        columns=model.feature_columns, fill_value=0.0
                    ).fillna(0.0)
                    if scaler is not None:
                        try:
                            X = pd.DataFrame(
                                scaler.transform(X.values),
                                index=X.index, columns=X.columns,
                            )
                        except Exception:
                            pass
                    predict_fn = model.predict_raw if self.use_raw_scores else model.predict
                    proba, _, _, _ = predict_fn(X)
                    results[ticker] = float(np.clip(proba[0], 0.0, 1.0))
                else:
                    row = day_df.loc[ticker]
                    feat_cols = [c for c in row.index if c not in self._LABEL_COLS]
                    X = row[feat_cols].values.reshape(1, -1)
                    X = np.nan_to_num(X, nan=0.0)
                    if scaler is not None:
                        X = scaler.transform(X)
                    proba  = model.predict_proba(X)[0]
                    classes = list(model.classes_)
                    results[ticker] = float(
                        proba[classes.index(1)] if 1 in classes else proba[-1]
                    )
            except Exception:
                pass

        return results

    def _score_batch(self, rows: dict[str, pd.Series]) -> dict[str, float]:
        """Dict-of-series fallback — delegates to _score_day."""
        if not rows:
            return {}
        day_df = pd.DataFrame(rows).T
        return self._score_day(day_df)

    def _score(self, ticker: str, row: pd.Series) -> float:
        """Single-ticker fallback."""
        return self._score_day(pd.DataFrame([row], index=[ticker])).get(ticker, 0.0)

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def run(self, start: str, end: str) -> dict:
        """
        Run the ranking portfolio simulation.

        Returns a dict with summary metrics and DataFrames.
        """
        u_mode = "universal model" if self.universal_model else "per-ticker models"
        filters = []
        if self.use_momentum_filter:
            filters.append("momentum(SMA50>200)")
        if self.vol_adjusted_ranking:
            filters.append("vol-adj ranking")
        filter_str = " | ".join(filters) if filters else "none"
        console.print(Panel(
            f"[bold]Universe Ranking Portfolio[/]\n"
            f"Universe: [cyan]{len(self.scoreable_tickers)} scoreable tickers[/] "
            f"([dim]{u_mode}[/])\n"
            f"Top-N: [green]{self.top_n}[/] | Min confidence: [yellow]{self.min_confidence}[/] | "
            f"Capital: [yellow]${self.initial_capital:,.0f}[/]\n"
            f"Filters: [cyan]{filter_str}[/] | Breadth threshold: [yellow]{self.market_breadth_threshold:.0%}[/]\n"
            f"Period: {start} to {end}",
            expand=False,
        ))

        # ---- Step 1: Load features for all tickers ----
        import concurrent.futures

        console.print("\n[bold]Loading data for all tickers...[/]")
        ticker_features: dict[str, pd.DataFrame] = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as prog:
            task = prog.add_task("Loading features...", total=len(self.scoreable_tickers))
            # Use threads for loading (features are cached — disk I/O bound, not CPU bound)
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                futures = {pool.submit(self._load_features, t, start, end): t
                           for t in self.scoreable_tickers}
                for fut in concurrent.futures.as_completed(futures):
                    ticker = futures[fut]
                    try:
                        feat = fut.result()
                        if feat is not None and len(feat) >= 20:
                            ticker_features[ticker] = feat
                    except Exception as exc:
                        logger.debug("Feature load failed for %s: %s", ticker, exc)
                    prog.advance(task)

        loaded = len(ticker_features)
        console.print(f"[green]{loaded}/{len(self.scoreable_tickers)} tickers loaded with data[/]")
        if loaded < self.top_n:
            console.print(f"[red]Not enough tickers ({loaded}) to fill top_n={self.top_n}[/]")
            return {}

        # ---- Step 1b: Load SPY timing data (market timing signal) ----
        spy_market_signal: dict[pd.Timestamp, int] = {}  # 1 = bull, 0 = bear/cash
        try:
            if self.spy_timing_method == "none":
                console.print("[dim]SPY timing: disabled — always invested[/]")
                spy_feat = None
            else:
                spy_feat = self._load_features("SPY", start, end)
            if spy_feat is not None:
                method = self.spy_timing_method
                if method == "price_200" and "sma_200_dist" in spy_feat.columns:
                    # Price vs 200-day SMA — faster exit/entry than Death Cross
                    for dt, row in spy_feat.iterrows():
                        spy_market_signal[dt] = 1 if float(row["sma_200_dist"]) > 0 else 0
                    timing_label = "price>SMA200"
                elif method == "price_50" and "sma_50_dist" in spy_feat.columns:
                    # Price vs 50-day SMA — fastest, most responsive, more whipsaws
                    for dt, row in spy_feat.iterrows():
                        spy_market_signal[dt] = 1 if float(row["sma_50_dist"]) > 0 else 0
                    timing_label = "price>SMA50"
                elif "sma_50_above_200" in spy_feat.columns:
                    # Default: Golden/Death Cross (SMA50 vs SMA200) — smoothest
                    for dt, row in spy_feat.iterrows():
                        spy_market_signal[dt] = int(row["sma_50_above_200"])
                    timing_label = "SMA50>SMA200 (Golden/Death Cross)"
                elif "sma_200_dist" in spy_feat.columns:
                    # Fallback if sma_50_above_200 not in cache
                    for dt, row in spy_feat.iterrows():
                        spy_market_signal[dt] = 1 if float(row["sma_200_dist"]) > 0 else 0
                    timing_label = "price>SMA200 (fallback)"
                if spy_market_signal:
                    bull_pct = sum(spy_market_signal.values()) / len(spy_market_signal)
                    console.print(f"[dim]SPY timing loaded ({len(spy_market_signal)} bars, {bull_pct:.0%} bull) [{timing_label}][/]")
        except Exception as exc:
            logger.debug("SPY timing load failed: %s", exc)

        # ---- Step 2: Pre-index features by date for O(1) simulation access ----
        # Concatenate all ticker DataFrames into a single (ticker, date) MultiIndex,
        # then group by date once — eliminates 500 df.loc[date] calls per bar.
        console.print("[dim]Pre-indexing features by date...[/]")
        combined = pd.concat(ticker_features, names=["ticker", "date"])
        # Group into {date: DataFrame(index=ticker, cols=features)} — O(1) lookup in loop
        date_groups: dict[pd.Timestamp, pd.DataFrame] = {
            date: grp.droplevel("date")
            for date, grp in combined.groupby(level="date")
        }
        all_dates = sorted(date_groups.keys())
        console.print(f"[dim]Date range: {all_dates[0].date()} to {all_dates[-1].date()} "
                      f"({len(all_dates)} bars)[/]")

        # ---- Step 3: Bar-by-bar simulation ----
        nav = self.initial_capital
        cash = self.initial_capital
        positions: dict[str, dict] = {}  # ticker -> {qty, entry_price, entry_bar, trail_high}
        last_prices: dict[str, float] = {}  # most recent real market price per ticker
        sl_cooldown: dict[str, int] = {}   # ticker -> bars remaining before re-entry allowed

        nav_history:   list[dict] = []
        trade_log:     list[dict] = []
        daily_top_log: list[dict] = []

        peak_nav = nav
        max_dd   = 0.0
        bar_idx  = 0  # running bar counter for min_holding_bars
        breadth         = 1.0            # market breadth (fraction of stocks in uptrend)
        effective_top_n = self.top_n     # may be halved during broad bear markets

        console.print("\n[bold]Running simulation...[/]")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as prog:
            task = prog.add_task("Simulating...", total=len(all_dates))

            for date in all_dates:
                bar_idx += 1

                # ---- Score all tickers on this date (one dict lookup + one predict call) ----
                scores: dict[str, float] = {}
                prices: dict[str, float] = {}

                day_df = date_groups.get(date)
                if day_df is not None and not day_df.empty:
                    # Filter to tickers with a valid positive close price
                    if "close" in day_df.columns:
                        close_s = pd.to_numeric(day_df["close"], errors="coerce")
                        valid_mask = close_s.notna() & (close_s > 0)
                        # Dollar volume filter: exclude micro-caps and illiquid stocks
                        if self.min_dollar_volume > 0 and "volume" in day_df.columns:
                            vol_s = pd.to_numeric(day_df["volume"], errors="coerce").fillna(0)
                            dv_mask = (close_s * vol_s) >= self.min_dollar_volume
                            valid_mask = valid_mask & dv_mask
                        day_valid = day_df.loc[valid_mask]
                        for t in day_valid.index:
                            last_prices[t] = float(close_s[t])
                    else:
                        day_valid = day_df

                    if not day_valid.empty:
                        batch_scores = self._score_day(day_valid)
                        regime_col_present = "regime" in day_valid.columns
                        has_momentum_col   = "sma_50_above_200" in day_valid.columns
                        has_vol_col        = "vol_21d" in day_valid.columns

                        # Market breadth: fraction of stocks in SMA50>SMA200 uptrend
                        if has_momentum_col:
                            breadth = float((day_valid["sma_50_above_200"] == 1).mean())
                        else:
                            breadth = 1.0

                        has_mom_rank_col  = "mom_12_1_rank" in day_valid.columns
                        has_channel_col   = "channel_pos_52w" in day_valid.columns
                        has_amihud_col    = "amihud_illiq" in day_valid.columns

                        # Pre-compute cross-sectional momentum rank for the whole day at once
                        # (avoids O(n) per-ticker rank recomputation inside the loop)
                        cross_mom_ranks: dict[str, float] = {}
                        if self.ranking_factor == "cross_momentum" and "mom_12_1" in day_valid.columns:
                            mom_series = pd.to_numeric(day_valid["mom_12_1"], errors="coerce")
                            ranked = mom_series.rank(pct=True, na_option="bottom")
                            cross_mom_ranks = ranked.to_dict()

                        for ticker in day_valid.index:
                            conf  = batch_scores.get(ticker, 0.0)
                            price = float(day_valid.at[ticker, "close"]) if "close" in day_valid.columns else 0.0
                            if regime_col_present:
                                row_regime = int(day_valid.at[ticker, "regime"])
                            else:
                                row_regime = 1
                            eff_min_conf = (self.bear_min_confidence if row_regime == 0
                                            else self.min_confidence)
                            if conf < eff_min_conf:
                                continue
                            # Momentum filter: skip stocks in downtrend (SMA50 < SMA200)
                            if self.use_momentum_filter and has_momentum_col:
                                if day_valid.at[ticker, "sma_50_above_200"] != 1:
                                    continue
                            # Ranking score selection
                            if self.ranking_factor == "cross_momentum":
                                # True cross-sectional 12-1 month momentum rank computed at simulation time.
                                # Ranks all tickers by absolute mom_12_1 on this bar — avoids the
                                # self-relative bias of the pre-computed mom_12_1_rank feature.
                                rank_score = cross_mom_ranks.get(ticker, 0.5)
                            elif self.ranking_factor == "liquidity":
                                # Rank by dollar volume (close × volume) → largest-cap stocks first
                                if "volume" in day_valid.columns:
                                    vol_shares = float(day_valid.at[ticker, "volume"])
                                    rank_score = price * vol_shares  # daily dollar volume
                                elif has_amihud_col:
                                    illiq = float(day_valid.at[ticker, "amihud_illiq"])
                                    rank_score = price * price / max(illiq, 1e-12)  # dollar-adjusted
                                else:
                                    rank_score = conf
                            elif self.ranking_factor == "momentum" and has_mom_rank_col:
                                # Academic 12-1 month momentum rank (already cross-sectionally ranked)
                                mr = float(day_valid.at[ticker, "mom_12_1_rank"])
                                rank_score = mr if not np.isnan(mr) else 0.5
                            elif self.ranking_factor == "momentum" and has_channel_col:
                                # Fallback: 52-week channel position
                                cp = float(day_valid.at[ticker, "channel_pos_52w"])
                                rank_score = cp if not np.isnan(cp) else 0.5
                            elif self.ranking_factor == "composite" and has_mom_rank_col:
                                # Blend model confidence + 12-1 momentum rank
                                mr = float(day_valid.at[ticker, "mom_12_1_rank"])
                                mom = mr if not np.isnan(mr) else 0.5
                                rank_score = 0.5 * conf + 0.5 * mom
                            elif self.vol_adjusted_ranking and has_vol_col:
                                vol = float(day_valid.at[ticker, "vol_21d"])
                                rank_score = conf / max(vol, 0.08)
                            else:
                                rank_score = conf
                            scores[ticker] = rank_score
                            prices[ticker] = price

                # ---- Phase 1: Hard stop loss + trailing stop exits (bypass min_holding_bars) ----
                ts_exits: list[str] = []
                sl_exits: list[str] = []
                for ticker, pos in positions.items():
                    cur_price = prices.get(ticker, pos["entry_price"])
                    # Update trailing high
                    pos["trail_high"] = max(pos.get("trail_high", cur_price), cur_price)
                    unrealised_pct = (cur_price - pos["entry_price"]) / (pos["entry_price"] + 1e-9)
                    # Hard stop loss: exit immediately if loss exceeds threshold
                    if self.stop_loss_pct > 0 and unrealised_pct <= -self.stop_loss_pct:
                        sl_exits.append(ticker)
                    elif unrealised_pct >= self.trailing_stop_activate_pct:
                        trail_stop = pos["trail_high"] * (1.0 - self.trailing_stop_pct)
                        if cur_price <= trail_stop:
                            ts_exits.append(ticker)

                def _close_position(ticker: str, action: str) -> None:
                    nonlocal cash
                    pos = positions.pop(ticker)
                    # Use current bar price if available, else last known real price
                    price = prices.get(ticker) or last_prices.get(ticker, pos["entry_price"])
                    slip_price = price * (1.0 - self.slippage_pct)
                    proceeds = pos["qty"] * slip_price
                    commission = abs(proceeds) * self.commission_pct
                    cash += proceeds - commission
                    pnl_pct = (slip_price - pos["entry_price"]) / (pos["entry_price"] + 1e-9) * 100.0
                    trade_log.append({
                        "date": date, "ticker": ticker, "action": action,
                        "price": round(slip_price, 4), "qty": pos["qty"],
                        "pnl_pct": round(pnl_pct, 3), "commission": round(commission, 2),
                    })

                for ticker in sl_exits:
                    _close_position(ticker, "STOP_LOSS")
                    if self.stop_loss_cooldown > 0:
                        sl_cooldown[ticker] = self.stop_loss_cooldown

                for ticker in ts_exits:
                    _close_position(ticker, "TRAIL_STOP")

                # ---- Rank and select top-N (SPY SMA200 timing + optional breadth overlay) ----
                # SPY below SMA200 = bear market → go to cash (effective_top_n = 0)
                spy_bull = spy_market_signal.get(date, None)
                if spy_bull is not None and spy_bull == 0:
                    effective_top_n = 0  # SPY below SMA200 → no new positions
                elif self.market_breadth_threshold > 0 and breadth < self.market_breadth_threshold:
                    effective_top_n = max(1, self.top_n // 2)
                else:
                    effective_top_n = self.top_n

                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                target_set = {t for t, _ in ranked[:effective_top_n]
                              if sl_cooldown.get(t, 0) == 0}

                # ---- Phase 2: Ranking exits (respect min_holding_bars) ----
                to_exit = [
                    t for t in list(positions)
                    if t not in target_set
                    and bar_idx - positions[t].get("entry_bar", 0) >= self.min_holding_bars
                ]
                for ticker in to_exit:
                    _close_position(ticker, "EXIT")

                # ---- Enter new top-N positions ----
                if target_set and len(target_set) > 0:
                    # Compute weights.
                    # When stop loss is active, use top_n as the denominator so that
                    # stopped-out slots stay as cash (no doubling-down into survivors).
                    slot_denom = self.top_n if self.stop_loss_pct > 0 else len(target_set)
                    if self.confidence_weighted and scores:
                        total_conf = sum(scores[t] for t in target_set if t in scores)
                        # Scale by filled-slot fraction so total weight <= 1
                        slot_frac = len(target_set) / max(slot_denom, 1)
                        weights = {
                            t: (scores[t] / (total_conf + 1e-9)) * slot_frac
                            for t in target_set if t in scores
                        }
                    else:
                        weights = {t: 1.0 / slot_denom for t in target_set}

                    # Determine total investable capital
                    # Recompute current position value
                    position_value = sum(
                        positions[t]["qty"] * prices.get(t, positions[t]["entry_price"])
                        for t in positions
                    )
                    nav = cash + position_value

                    for ticker in target_set:
                        price = prices.get(ticker)
                        if price is None or price <= 0:
                            continue
                        target_value = nav * weights.get(ticker, 0.0)

                        # Already holding — check if we need to rebalance
                        current_value = (
                            positions[ticker]["qty"] * price
                            if ticker in positions else 0.0
                        )
                        delta_value = target_value - current_value
                        # Skip rebalance if within threshold of target (avoids churn)
                        if abs(delta_value) < max(price, self.rebalance_threshold * target_value):
                            continue

                        delta_qty = delta_value / price
                        slip_sign  = 1.0 if delta_qty > 0 else -1.0
                        fill_price = price * (1.0 + self.slippage_pct * slip_sign)
                        cost       = abs(delta_qty) * fill_price
                        commission = cost * self.commission_pct

                        if delta_qty > 0 and cash < cost + commission:
                            # Not enough cash — scale down
                            affordable = cash / (fill_price * (1.0 + self.commission_pct))
                            if affordable < 0.5:
                                continue
                            delta_qty = affordable
                            cost = delta_qty * fill_price
                            commission = cost * self.commission_pct

                        # Commission always reduces cash (buy: cost+comm; sell: proceeds-comm)
                        cash -= delta_qty * fill_price + commission

                        if ticker in positions:
                            old_qty = positions[ticker]["qty"]
                            new_qty = old_qty + delta_qty
                            if new_qty <= 0:
                                positions.pop(ticker, None)
                            else:
                                positions[ticker]["qty"] = new_qty
                                if delta_qty > 0:
                                    # Adding shares: update weighted average cost basis
                                    old_cost = positions[ticker]["entry_price"] * old_qty
                                    positions[ticker]["entry_price"] = (
                                        old_cost + fill_price * delta_qty
                                    ) / new_qty
                                # Reducing shares: entry_price stays unchanged
                        else:
                            if delta_qty > 0:
                                positions[ticker] = {
                                    "qty": delta_qty,
                                    "entry_price": fill_price,
                                    "entry_bar":   bar_idx,
                                    "trail_high":  fill_price,
                                }

                        action = "BUY" if delta_qty > 0 else "REDUCE"
                        trade_log.append({
                            "date":       date, "ticker": ticker,
                            "action":     action, "price": round(fill_price, 4),
                            "qty":        round(abs(delta_qty), 4),
                            "pnl_pct":    0.0,
                            "commission": round(commission, 2),
                        })

                # ---- Log daily top tickers ----
                daily_top_log.append({
                    "date":          date,
                    "top_tickers":   ",".join(t for t, _ in ranked[:min(5, len(ranked))]),
                    "n_scored":      len(scores),
                    "n_held":        len(positions),
                    "breadth":       round(breadth, 3),
                    "effective_topn": effective_top_n,
                })

                # ---- Update NAV ----
                position_value = sum(
                    positions[t]["qty"] * prices.get(t, positions[t]["entry_price"])
                    for t in positions
                )
                nav = cash + position_value
                daily_return = (nav / (nav_history[-1]["nav"] if nav_history else self.initial_capital)) - 1.0
                nav_history.append({
                    "date":         date,
                    "nav":          round(nav, 2),
                    "cash":         round(cash, 2),
                    "n_positions":  len(positions),
                    "daily_return": round(daily_return, 6),
                })

                # Drawdown
                if nav > peak_nav:
                    peak_nav = nav
                dd = (peak_nav - nav) / (peak_nav + 1e-9)
                if dd > max_dd:
                    max_dd = dd

                # Decrement stop loss cooldown counters
                for t in list(sl_cooldown):
                    sl_cooldown[t] -= 1
                    if sl_cooldown[t] <= 0:
                        del sl_cooldown[t]

                prog.advance(task)

        # ---- Step 4: Compute summary metrics ----
        nav_df   = pd.DataFrame(nav_history).set_index("date")
        trade_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()

        total_return = (nav - self.initial_capital) / self.initial_capital * 100.0
        if len(nav_df) > 1:
            rets = nav_df["daily_return"].dropna()
            sharpe = float(rets.mean() / (rets.std() + 1e-9) * np.sqrt(252)) if rets.std() > 0 else 0.0
            ann_return = float((1 + rets.mean()) ** 252 - 1) * 100.0
        else:
            sharpe = 0.0
            ann_return = 0.0

        n_trades   = len(trade_df[trade_df["action"] == "EXIT"]) if not trade_df.empty else 0
        avg_n_held = float(nav_df["n_positions"].mean()) if len(nav_df) else 0.0

        # ---- Save output ----
        log_dir = ensure_dir(ROOT / "logs" / "universe_portfolio")
        nav_df.to_csv(log_dir / "nav.csv")
        if not trade_df.empty:
            trade_df.to_csv(log_dir / "trades.csv", index=False)
        pd.DataFrame(daily_top_log).to_csv(log_dir / "daily_top.csv", index=False)

        # ---- Print results ----
        console.print()
        result_table = Table(title="Universe Portfolio Results", show_header=True,
                             header_style="bold cyan")
        result_table.add_column("Metric", style="dim", width=28)
        result_table.add_column("Value", justify="right", width=16)

        color = "green" if total_return > 0 else "red"
        result_table.add_row("Total Return",     f"[{color}]{total_return:+.2f}%[/]")
        result_table.add_row("Ann. Return",      f"{ann_return:+.2f}%")
        result_table.add_row("Sharpe Ratio",     f"{sharpe:.3f}")
        result_table.add_row("Max Drawdown",     f"[red]-{max_dd*100:.2f}%[/]")
        result_table.add_row("Total Trades",     str(n_trades))
        result_table.add_row("Avg Positions",    f"{avg_n_held:.1f}")
        result_table.add_row("Final NAV",        f"${nav:,.2f}")
        result_table.add_row("Universe Size",    str(len(self.scoreable_tickers)))
        result_table.add_row("Top-N Held",       str(self.top_n))
        console.print(result_table)

        console.print(f"\n[dim]Logs saved to: {log_dir}[/]")

        return {
            "total_return_pct": total_return,
            "ann_return_pct":   ann_return,
            "sharpe":           sharpe,
            "max_dd_pct":       max_dd * 100,
            "n_trades":         n_trades,
            "avg_n_held":       avg_n_held,
            "final_nav":        nav,
            "nav_df":           nav_df,
            "trade_df":         trade_df,
        }
