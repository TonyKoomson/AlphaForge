"""
Feature engineering pipeline for Alpha Forge.

Public API
----------
  generate_features(df, as_of_date)       standalone function → full raw feature matrix
  add_regime_labels(df, cfg)              add bull/bear/sideways/high-vol regime column
  FeatureSelector                         mutual-info + optional SHAP + correlation pruning
  FeatureEngine                           class wrapper used by models/train.py and CLI

Anti-leakage guarantees
-----------------------
  Every rolling/EWM call uses only past bars (no center=True, min_periods enforced).
  The as_of_date hard-cut is applied BEFORE any indicator is computed so no future
  bar can leak through a wide window.
  Forward-return labels live in separate 'label' / 'fwd_return' columns and are
  NEVER included in the feature matrix X.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import ensure_dir, get_logger, load_config

logger = get_logger(__name__)

# Column names that are labels / raw OHLCV — never treated as features
# quantile_signal uses forward returns (.shift(-target_horizon)) so it is a label,
# not a predictor; including it in X would be look-ahead leakage.
_LABEL_COLS = {"label", "fwd_return", "tb_label", "tb_ret", "meta_label", "quantile_signal"}
_OHLCV_COLS = {"open", "high", "low", "close", "volume"}


# =============================================================================
# Triple-Barrier Labeling  (López de Prado, AFML Ch.3)
# =============================================================================

def triple_barrier_labels(
    close: pd.Series,
    t_bars: int = 5,
    pt_multiplier: float = 1.5,
    sl_multiplier: float = 1.0,
    vol_window: int = 21,
) -> pd.DataFrame:
    """
    Assign a label to each bar based on which barrier is hit first.

    Upper barrier  (+1): close rises by pt_multiplier × daily_vol within t_bars
    Lower barrier  (-1): close falls by sl_multiplier × daily_vol within t_bars
    Vertical        (0): neither hit within t_bars

    Returns a DataFrame with columns:
        tb_label  int   {-1, 0, 1}
        tb_ret    float realised return at first-touch (or at t_bars)

    Anti-leakage: labels are shifted so tb_label[t] uses bars t+1…t+t_bars only.
    """
    daily_vol = close.pct_change().rolling(vol_window, min_periods=5).std()

    tb_label = pd.Series(0,     index=close.index, dtype=int,   name="tb_label")
    tb_ret   = pd.Series(0.0,   index=close.index, dtype=float, name="tb_ret")

    for i in range(len(close) - t_bars):
        t0_price = close.iloc[i]
        vol_t    = daily_vol.iloc[i]
        if np.isnan(vol_t) or vol_t <= 0:
            continue

        upper = t0_price * (1.0 + pt_multiplier * vol_t)
        lower = t0_price * (1.0 - sl_multiplier * vol_t)

        window = close.iloc[i + 1: i + 1 + t_bars]
        label  = 0
        ret    = float(window.iloc[-1] / t0_price - 1.0) if len(window) else 0.0

        for price in window:
            if price >= upper:
                label = 1
                ret   = float(price / t0_price - 1.0)
                break
            if price <= lower:
                label = -1
                ret   = float(price / t0_price - 1.0)
                break

        tb_label.iloc[i] = label
        tb_ret.iloc[i]   = ret

    return pd.DataFrame({"tb_label": tb_label, "tb_ret": tb_ret})


# =============================================================================
# Advanced Labeling Methods
# =============================================================================

def quantile_labels(
    returns: pd.Series,
    q_upper: float = 0.70,
    q_lower: float = 0.30,
    rolling_window: int = 63,
) -> pd.Series:
    """
    Assign labels based on rolling quantiles of forward returns.

    +1 if return > rolling q_upper quantile (strong up move)
    -1 if return < rolling q_lower quantile (strong down move)
     0 otherwise (noise)

    More selective than binary labeling — only trades the tails.
    Anti-leakage: quantiles computed from past returns only.
    """
    upper = returns.rolling(rolling_window, min_periods=10).quantile(q_upper)
    lower = returns.rolling(rolling_window, min_periods=10).quantile(q_lower)
    label = pd.Series(0, index=returns.index, name="quantile_label", dtype=int)
    label[returns > upper] = 1
    label[returns < lower] = -1
    return label


def multi_horizon_labels(
    close: pd.Series,
    horizons: tuple[int, ...] = (1, 5, 10, 21),
) -> pd.DataFrame:
    """
    Compute binary direction labels for multiple forward horizons.

    Returns a DataFrame with columns 'label_h{n}' for each horizon n.
    Richer supervision signal — model can be trained on all horizons jointly.
    Anti-leakage: labels shift(-n) so each uses only future bars.
    """
    out = {}
    for h in horizons:
        fwd = close.pct_change(h).shift(-h)
        out[f"label_h{h}"] = (fwd > 0).astype(int)
    return pd.DataFrame(out, index=close.index)


def asymmetric_labels(
    returns: pd.Series,
    profit_threshold: float = 0.005,
    loss_threshold:   float = -0.003,
) -> pd.Series:
    """
    Asymmetric labeling: different thresholds for longs vs shorts.

    Reflects the real-world asymmetry where transaction costs mean
    you need a larger up move to profit than the down move you want to avoid.

    +1 if return >= profit_threshold
    -1 if return <= loss_threshold
     0 otherwise

    Anti-leakage: use forward returns (pre-shifted).
    """
    label = pd.Series(0, index=returns.index, name="asym_label", dtype=int)
    label[returns >= profit_threshold] = 1
    label[returns <= loss_threshold]   = -1
    return label


def event_based_labels(
    close: pd.Series,
    theta: float = 0.015,
) -> pd.Series:
    """
    Directional-Change (DC) event labeling (Glattfelder et al., 2011).

    Detects when price moves theta% in either direction from the last event.
    Labels the event bar +1 (upward DC) or -1 (downward DC).
    Produces sparse, economically meaningful signals compared to bar-by-bar.

    Parameters
    ----------
    theta : DC threshold (e.g. 0.015 = 1.5% move triggers an event)
    """
    px     = close.values.astype(float)
    n      = len(px)
    labels = np.zeros(n, dtype=int)
    ref    = px[0]
    mode   = 0   # 0 = seeking up DC, 1 = seeking down DC

    for i in range(1, n):
        if mode == 0 and px[i] >= ref * (1 + theta):
            labels[i] = 1
            ref  = px[i]
            mode = 1
        elif mode == 1 and px[i] <= ref * (1 - theta):
            labels[i] = -1
            ref  = px[i]
            mode = 0

    return pd.Series(labels, index=close.index, name="dc_label")


def meta_label(
    primary_signal: pd.Series,
    tb_labels: pd.Series,
) -> pd.Series:
    """
    Build a binary meta-label: 1 when the primary signal agrees with the
    triple-barrier outcome (i.e. the trade was actually profitable), 0 otherwise.

    The meta-label is used to train a secondary classifier whose predicted
    probability becomes the bet-size multiplier on the primary signal.

    Returns a boolean Series named 'meta_label'.
    """
    align   = primary_signal.reindex(tb_labels.index).fillna(0)
    correct = ((align > 0) & (tb_labels == 1)) | ((align < 0) & (tb_labels == -1))
    return correct.astype(int).rename("meta_label")


# =============================================================================
# Low-level indicator helpers (all past-only)
# =============================================================================

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=period).mean()
    rs = gain / (loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(df: pd.DataFrame) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift(1)).abs()
    lc = (df["low"] - df["close"].shift(1)).abs()
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return _true_range(df).rolling(period, min_periods=period).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def _stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100.0 * (df["close"] - low_min) / denom
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def _higher_highs(high: pd.Series, window: int = 10) -> pd.Series:
    """Fraction of bars in trailing window where high > previous high."""
    rolling_highs = high.rolling(window, min_periods=window)
    return rolling_highs.apply(
        lambda x: float(np.sum(np.diff(x) > 0)) / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True,
    )


def _higher_lows(low: pd.Series, window: int = 10) -> pd.Series:
    """Fraction of bars in trailing window where low > previous low."""
    rolling_lows = low.rolling(window, min_periods=window)
    return rolling_lows.apply(
        lambda x: float(np.sum(np.diff(x) > 0)) / (len(x) - 1) if len(x) > 1 else np.nan,
        raw=True,
    )


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average Directional Index — measures trend strength 0-100."""
    tr = _true_range(df)
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean()
    tr_s = tr.rolling(period, min_periods=period).mean()

    plus_di = 100.0 * plus_dm_s / (tr_s + 1e-9)
    minus_di = 100.0 * minus_dm_s / (tr_s + 1e-9)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.rolling(period, min_periods=period).mean()


# =============================================================================
# Regime labelling
# =============================================================================

def add_regime_labels(
    df: pd.DataFrame,
    bull_threshold: float = 0.10,
    bear_threshold: float = -0.10,
    vol_high_threshold: float = 0.25,
    vol_low_threshold: float = 0.10,
) -> pd.DataFrame:
    """
    Append regime columns to df (in-place copy).

    regime       : 0=bear  1=sideways  2=bull  3=high-vol (overrides trend)
    regime_bull  : 1 if bull
    regime_bear  : 1 if bear
    regime_hv    : 1 if high-vol (ann. realised vol > threshold)

    Uses 252-bar trailing return for trend, 21-bar realized vol for volatility.
    """
    df = df.copy()
    ret1d = df["close"].pct_change(1)
    trail_ret_252 = df["close"].pct_change(252)
    trail_ret_63  = df["close"].pct_change(63)

    vol_21d = ret1d.rolling(21, min_periods=21).std() * np.sqrt(252)

    bull = (trail_ret_252 > bull_threshold).astype(int)
    bear = (trail_ret_252 < bear_threshold).astype(int)
    hv = (vol_21d > vol_high_threshold).astype(int)

    regime = pd.Series(1, index=df.index, dtype=int)  # default sideways
    regime[bull == 1] = 2
    regime[bear == 1] = 0
    regime[hv == 1] = 3      # high-vol overrides everything

    df["regime"] = regime
    df["regime_bull"] = bull
    df["regime_bear"] = bear
    df["regime_hv"] = hv
    df["trail_ret_63"] = trail_ret_63
    return df


# =============================================================================
# Core feature generation
# =============================================================================

def generate_features(
    df: pd.DataFrame,
    as_of_date: Optional[str] = None,
    target_horizon: int = 5,
    add_regime: bool = True,
    add_fracdiff: bool = True,
    fracdiff_d: Optional[float] = None,
    cross_asset: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build the complete raw feature matrix from OHLCV data.

    Parameters
    ----------
    df : DataFrame with columns open/high/low/close/volume, DatetimeIndex.
    as_of_date : Hard cutoff — rows after this date are dropped BEFORE any
                 indicator is computed. Prevents wide-window look-ahead.
    target_horizon : Forward return horizon for label construction (bars).
    add_regime : Whether to append regime columns.

    Returns
    -------
    DataFrame with all raw features + 'label' + 'fwd_return' columns.
    Rows with NaN (warmup period) are dropped.
    """
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        df = df[df.index <= cutoff].copy()
        if df.empty:
            raise ValueError(f"No data on or before as_of_date={as_of_date}")
    else:
        df = df.copy()

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")

    # Building features column-by-column into a growing DataFrame is intentional;
    # suppress the PerformanceWarning from pandas about DataFrame fragmentation.
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

    out = df[list(required)].copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)

    # --- Returns at multiple horizons ---
    for n in (1, 2, 3, 5, 10, 20):
        out[f"ret_{n}d"] = close.pct_change(n)

    # Skewness and kurtosis of recent returns (distributional shape)
    out["ret_skew_21d"] = close.pct_change(1).rolling(21, min_periods=21).skew()
    out["ret_kurt_21d"] = close.pct_change(1).rolling(21, min_periods=21).kurt()

    # --- Moving averages (5, 10, 20, 50, 200) ---
    for n in (5, 10, 20, 50, 200):
        sma = _sma(close, n)
        out[f"sma_{n}_dist"] = (close - sma) / (sma + 1e-9)   # price distance from SMA

    # SMA crossover signals (fast/slow ratio)
    sma5  = _sma(close, 5)
    sma10 = _sma(close, 10)
    sma20 = _sma(close, 20)
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    out["cross_5_20"]  = (sma5 / (sma20 + 1e-9) - 1.0)
    out["cross_20_50"] = (sma20 / (sma50 + 1e-9) - 1.0)
    out["cross_50_200"] = (sma50 / (sma200 + 1e-9) - 1.0)
    out["sma_20_above_50"] = (sma20 > sma50).astype(int)
    out["sma_50_above_200"] = (sma50 > sma200).astype(int)

    # --- Momentum (RSI, MACD, Stochastic) ---
    out["rsi_14"]  = _rsi(close, 14)
    out["rsi_28"]  = _rsi(close, 28)
    out["rsi_norm"] = out["rsi_14"] / 100.0        # 0-1 scaled version

    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd_line = (ema12 - ema26) / (close + 1e-9)
    macd_sig  = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_norm"]   = macd_line
    out["macd_hist"]   = macd_line - macd_sig
    out["macd_cross"]  = (macd_line > macd_sig).astype(int)

    stoch_k, stoch_d = _stochastic(out, k_period=14, d_period=3)
    out["stoch_k"]    = stoch_k / 100.0
    out["stoch_d"]    = stoch_d / 100.0
    out["stoch_hist"] = (stoch_k - stoch_d) / 100.0

    # Rate-of-change (momentum normalised)
    for n in (5, 10, 21):
        out[f"roc_{n}d"] = (close / close.shift(n) - 1.0)

    # --- Volatility ---
    ret1d = close.pct_change(1)
    out["vol_5d"]  = ret1d.rolling(5,  min_periods=5 ).std() * np.sqrt(252)
    out["vol_21d"] = ret1d.rolling(21, min_periods=21).std() * np.sqrt(252)
    out["vol_63d"] = ret1d.rolling(63, min_periods=63).std() * np.sqrt(252)
    out["vol_ratio_5_21"]  = out["vol_5d"]  / (out["vol_21d"] + 1e-9)
    out["vol_ratio_21_63"] = out["vol_21d"] / (out["vol_63d"] + 1e-9)

    atr14 = _atr(out, 14)
    out["atr_ratio"] = atr14 / (close + 1e-9)

    # Bollinger Bands
    bb_sma = _sma(close, 20)
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    out["bb_pct"]   = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)  # 0-1 within bands
    out["bb_width"] = (bb_upper - bb_lower) / (bb_sma + 1e-9)

    # Realised vol z-score (how unusual is current vol relative to its own history)
    vol_21_roll_mean = out["vol_21d"].rolling(63, min_periods=63).mean()
    vol_21_roll_std  = out["vol_21d"].rolling(63, min_periods=63).std()
    out["vol_zscore"] = (out["vol_21d"] - vol_21_roll_mean) / (vol_21_roll_std + 1e-9)

    # --- Volume ---
    vol_ma20 = volume.rolling(20, min_periods=20).mean()
    out["volume_ratio"]  = volume / (vol_ma20 + 1e-9)
    out["volume_zscore"] = (volume - vol_ma20) / (volume.rolling(20, min_periods=20).std() + 1e-9)

    obv = _obv(close, volume)
    obv_sma20 = _sma(obv, 20)
    out["obv_trend"] = (obv - obv_sma20) / (obv_sma20.abs() + 1e-9)  # normalised OBV deviation

    # --- Trend strength & pattern features ---
    out["adx_14"] = _adx(out, 14)
    out["adx_norm"] = out["adx_14"] / 100.0

    out["higher_highs_10"] = _higher_highs(high, 10)
    out["higher_lows_10"]  = _higher_lows(low,  10)
    out["trend_strength"]  = out["higher_highs_10"] - (1.0 - out["higher_lows_10"])

    # 52-week channel position
    high_52w = high.rolling(252, min_periods=63).max()
    low_52w  = low.rolling(252, min_periods=63).min()
    out["channel_pos_52w"] = (close - low_52w) / (high_52w - low_52w + 1e-9)
    out["high_52w_dist"]   = (close - high_52w) / (high_52w + 1e-9)

    # Gap (open vs prior close) — overnight sentiment
    out["gap"] = (out["open"] / close.shift(1) - 1.0)

    # --- Intraday vs overnight return decomposition ---
    # Gap = overnight (open/prev_close - 1).  Intraday = close/open - 1.
    # These two components have opposite autocorrelation properties and carry
    # independent information: gaps tend to revert, intraday follows through.
    out["intraday_ret"] = (close / out["open"] - 1.0)
    out["gap_vs_intraday"] = out["gap"] - out["intraday_ret"]

    # --- Academic momentum factors (Jegadeesh-Titman 1993; Moskowitz 2017) ---
    # 12-1 month: trailing 12m return minus trailing 1m (skip-last-month avoids
    # short-term reversal contaminating the medium-term momentum signal).
    out["mom_12_1"] = (close.pct_change(252) - close.pct_change(21)).fillna(0.0)
    # 6-1 month intermediate-term momentum
    out["mom_6_1"]  = (close.pct_change(126) - close.pct_change(21)).fillna(0.0)
    # 1-week short-term reversal: day-1 to day-5 returns mean-revert
    out["reversal_1w"] = -close.pct_change(5)
    # 1-month short-term reversal (Lehmann 1990)
    out["reversal_1m"] = -close.pct_change(21)

    # --- Volatility-of-volatility (vol-of-vol) ---
    # High vol-of-vol periods precede volatility spikes; useful as a regime gate.
    _vov_window = 30
    out["vol_of_vol"] = out["vol_21d"].rolling(_vov_window, min_periods=_vov_window).std()
    out["vol_of_vol_norm"] = out["vol_of_vol"] / (out["vol_21d"] + 1e-9)
    # Z-score of vol-of-vol relative to its own history
    _vov_mean = out["vol_of_vol"].rolling(126, min_periods=63).mean()
    _vov_std  = out["vol_of_vol"].rolling(126, min_periods=63).std()
    out["vol_of_vol_zscore"] = (out["vol_of_vol"] - _vov_mean) / (_vov_std + 1e-9)

    # --- Rolling percentile ranks (stationary, regime-independent features) ---
    # Converts each indicator to its historical rank in [0, 1].  Rank-based
    # features are robust to distributional shifts across regimes and help the
    # model learn threshold-crossing patterns rather than absolute levels.
    def _rolling_rank(s: pd.Series, window: int) -> pd.Series:
        return s.rolling(window, min_periods=max(window // 2, 10)).rank(pct=True)

    for _feat, _w in [
        ("rsi_14", 63), ("macd_hist", 63), ("vol_21d", 63),
        ("bb_pct", 63), ("obv_trend", 63), ("adx_norm", 63),
        ("mom_12_1", 126), ("mom_6_1", 126),
    ]:
        if _feat in out.columns:
            out[f"{_feat}_rank"] = _rolling_rank(out[_feat], _w)

    # Defragment before the large cross-asset section (avoids repeated realloc)
    out = out.copy()

    # --- VIX features (fear gauge — highly predictive for SPY direction) ---
    # VIX falling → equity tailwind.  VIX spiking → equity headwind.
    # Uses cross_asset cache when provided (avoids repeated API calls for bulk training).
    try:
        _vix_series = None
        if cross_asset and "^VIX" in cross_asset:
            _vix_series = cross_asset["^VIX"].reindex(out.index).ffill().bfill()
        else:
            import yfinance as _yf
            _vix_raw = _yf.download(
                "^VIX",
                start=str(out.index.min().date()),
                end=str(out.index.max().date() + pd.Timedelta(days=5)),
                progress=False,
                auto_adjust=True,
            )
            if not _vix_raw.empty:
                _vc_tmp = _vix_raw["Close"].squeeze()
                if isinstance(_vc_tmp, pd.DataFrame):
                    _vc_tmp = _vc_tmp.iloc[:, 0]
                _vc_tmp.index = pd.to_datetime(_vc_tmp.index).tz_localize(None)
                _vix_series = _vc_tmp.reindex(out.index).ffill().bfill()
        if _vix_series is not None:
            _vc = _vix_series
            _vix_ma20  = _vc.rolling(20, min_periods=10).mean()
            _vix_ma252 = _vc.rolling(252, min_periods=63).mean()
            out["vix_level"]   = _vc
            out["vix_ratio"]   = _vc / (_vix_ma20 + 1e-9)    # VIX vs 1-month MA
            out["vix_lr_ratio"] = _vc / (_vix_ma252 + 1e-9)  # VIX vs 1-year MA (long-run fear)
            out["vix_1d_chg"]  = _vc.pct_change(1)
            out["vix_5d_chg"]  = _vc.pct_change(5)
            out["vix_rank"]    = _vc.rolling(63, min_periods=20).rank(pct=True)
            # Binary regime flags (used as features, not filters)
            out["vix_spike"]   = (_vc / (_vix_ma20 + 1e-9) > 1.30).astype(int)
            out["vix_low"]     = (_vc / (_vix_ma20 + 1e-9) < 0.85).astype(int)
            # VIX 80th-percentile "buy the fear" signal (Quantpedia research)
            # When VIX > 80th pct of trailing 252 bars → contrarian bullish (panic peak)
            _vix_p80 = _vc.rolling(252, min_periods=63).quantile(0.80)
            out["vix_fear_zone"]    = (_vc > _vix_p80).astype(int)   # buy signal
            out["vix_calm_zone"]    = (_vc < _vc.rolling(252, min_periods=63).quantile(0.20)).astype(int)
            logger.debug("VIX features added (%d bars)", _vc.notna().sum())

        # VVIX (VIX of VIX) — tail risk indicator.
        try:
            _vv_series = None
            if cross_asset and "^VVIX" in cross_asset:
                _vv_series = cross_asset["^VVIX"].reindex(out.index).ffill().bfill()
            else:
                _vvix_raw = _yf.download(
                    "^VVIX",
                    start=str(out.index.min().date()),
                    end=str(out.index.max().date() + pd.Timedelta(days=5)),
                    progress=False,
                    auto_adjust=True,
                )
                if not _vvix_raw.empty:
                    _vv_tmp = _vvix_raw["Close"].squeeze()
                    if isinstance(_vv_tmp, pd.DataFrame):
                        _vv_tmp = _vv_tmp.iloc[:, 0]
                    _vv_tmp.index = pd.to_datetime(_vv_tmp.index).tz_localize(None)
                    _vv_series = _vv_tmp.reindex(out.index).ffill().bfill()
            if _vv_series is not None:
                _vv = _vv_series
                _vv_ma20  = _vv.rolling(20, min_periods=10).mean()
                _vv_mu    = _vv.rolling(63, min_periods=20).mean()
                _vv_sig   = _vv.rolling(63, min_periods=20).std()
                out["vvix_level"]   = _vv
                out["vvix_zscore"]  = (_vv - _vv_mu) / (_vv_sig + 1e-9)
                out["vvix_spike"]   = (_vv > 120).astype(int)   # extreme tail risk (VVIX > 120 = crisis)
                out["vvix_above_ma"]= (_vv > _vv_ma20).astype(int)
        except Exception as _vvix_err:
            logger.debug("VVIX features skipped: %s", _vvix_err)
    except Exception as _vix_err:
        logger.debug("VIX features skipped: %s", _vix_err)

    # --- Macro cross-asset features: yield curve + VIX term structure + DXY ---
    # Uses cross_asset cache when provided; falls back to inline download.
    try:
        def _ca_get(ticker):
            if cross_asset and ticker in cross_asset:
                return cross_asset[ticker].reindex(out.index).ffill().bfill()
            return None

        _macro_missing = [t for t in ["^TNX", "^IRX", "^FVX", "^VIX9D", "^VIX3M", "DX-Y.NYB"]
                          if not (cross_asset and t in cross_asset)]
        _macro_raw_df = None
        if _macro_missing:
            import yfinance as _yf2
            _macro_raw_df = _yf2.download(
                _macro_missing,
                start=str(out.index.min().date()),
                end=str((out.index.max() + pd.Timedelta(days=5)).date()),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )

        def _get_series(raw, ticker):
            s = _ca_get(ticker)
            if s is not None:
                return s
            if raw is None or raw.empty:
                return None
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    s = raw[ticker]["Close"].squeeze()
                else:
                    s = raw["Close"].squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.reindex(out.index).ffill().bfill()
            except Exception:
                return None

        _tnx_s = _get_series(_macro_raw_df, "^TNX")
        _irx_s = _get_series(_macro_raw_df, "^IRX")
        _fvx_s = _get_series(_macro_raw_df, "^FVX")
        _vix9d = _get_series(_macro_raw_df, "^VIX9D")
        _vix3m = _get_series(_macro_raw_df, "^VIX3M")
        _dxy_s = _get_series(_macro_raw_df, "DX-Y.NYB")

        # Yield curve: 10Y-3M spread (most reliable recession predictor)
        if _tnx_s is not None and _irx_s is not None:
            _yc = (_tnx_s - _irx_s) / 100.0   # in decimal (e.g., 0.015 = 1.5%)
            out["yield_curve_10_3m"]   = _yc
            out["yield_curve_inverted"] = (_yc < 0.0).astype(int)
            # Rolling z-score of the spread (how unusual is current steepness)
            _yc_mu  = _yc.rolling(252, min_periods=63).mean()
            _yc_sig = _yc.rolling(252, min_periods=63).std()
            out["yield_curve_zscore"]  = (_yc - _yc_mu) / (_yc_sig + 1e-9)
            # 20-bar change in slope (rising = curve steepening = growth optimism)
            out["yield_curve_chg_20d"] = _yc.diff(20)
            # Rank of yield curve vs past year
            out["yield_curve_rank"]    = _yc.rolling(252, min_periods=63).rank(pct=True)

        if _fvx_s is not None and _irx_s is not None:
            out["yield_curve_5_3m"] = (_fvx_s - _irx_s) / 100.0

        # VIX term structure: contango vs backwardation
        # Contango (VIX9D < spot VIX < VIX3M) → calm → equity tailwind
        # Backwardation (VIX9D > spot VIX) → acute fear spike
        if "vix_level" in out.columns and _vix9d is not None and _vix3m is not None:
            _spot = out["vix_level"]
            out["vix_ts_front_ratio"] = _vix9d  / (_spot  + 1e-9)  # <1 = backwardation front
            out["vix_ts_back_ratio"]  = _spot   / (_vix3m + 1e-9)  # <1 = contango back
            out["vix_contango"]       = (_vix9d < _spot).astype(int)   # 0=backwardation
            # Term structure slope: positive = contango (bullish), negative = backwardation (bearish)
            out["vix_ts_slope"]       = (_vix3m - _vix9d) / (_vix9d + 1e-9)
            out["vix_ts_slope_rank"]  = out["vix_ts_slope"].rolling(63, min_periods=20).rank(pct=True)

        # DXY (US Dollar Index): rising USD = headwind for risk assets
        if _dxy_s is not None:
            _dxy_ma63 = _dxy_s.rolling(63, min_periods=20).mean()
            out["dxy_ret_5d"]    = _dxy_s.pct_change(5)
            out["dxy_ret_21d"]   = _dxy_s.pct_change(21)
            out["dxy_zscore"]    = (_dxy_s - _dxy_ma63) / (_dxy_s.rolling(63, min_periods=20).std() + 1e-9)
            out["dxy_above_ma"]  = (_dxy_s > _dxy_ma63).astype(int)

        logger.debug("Macro cross-asset features added")
    except Exception as _macro_err:
        logger.debug("Macro cross-asset features skipped: %s", _macro_err)

    # --- Credit spread & flight-to-safety features (HYG, TLT, GLD, QQQ) ---
    # HYG (high-yield ETF): falling HYG → credit stress → equity bearish
    # TLT (long Treasury ETF): rising TLT → flight to safety → equity bearish
    # GLD (gold ETF): rising gold → fear/inflation → mixed equity signal
    try:
        _cs_missing = [t for t in ["HYG", "TLT", "GLD", "QQQ"]
                       if not (cross_asset and t in cross_asset)]
        _cs_raw = None
        if _cs_missing:
            import yfinance as _yf3
            _cs_raw = _yf3.download(
                _cs_missing,
                start=str(out.index.min().date()),
                end=str((out.index.max() + pd.Timedelta(days=5)).date()),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )

        def _cs_series(ticker):
            if cross_asset and ticker in cross_asset:
                return cross_asset[ticker].reindex(out.index).ffill().bfill()
            if _cs_raw is None or _cs_raw.empty:
                return None
            try:
                if isinstance(_cs_raw.columns, pd.MultiIndex):
                    s = _cs_raw[ticker]["Close"].squeeze()
                else:
                    s = _cs_raw["Close"].squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.reindex(out.index).ffill().bfill()
            except Exception:
                return None

        _hyg = _cs_series("HYG")
        _tlt = _cs_series("TLT")
        _gld = _cs_series("GLD")
        _qqq = _cs_series("QQQ")

        # HYG: high-yield spread proxy (relative to TLT)
        if _hyg is not None:
            _hyg_ma20 = _hyg.rolling(20, min_periods=10).mean()
            out["hyg_ret_5d"]   = _hyg.pct_change(5)
            out["hyg_ret_21d"]  = _hyg.pct_change(21)
            out["hyg_zscore"]   = (_hyg - _hyg_ma20) / (_hyg.rolling(20, min_periods=10).std() + 1e-9)
            out["hyg_above_ma"] = (_hyg > _hyg_ma20).astype(int)

        # HYG/TLT ratio: credit spread proxy (low = credit stress = equity bearish)
        if _hyg is not None and _tlt is not None:
            _cs_ratio = _hyg / (_tlt + 1e-9)
            _cs_ma20  = _cs_ratio.rolling(20, min_periods=10).mean()
            out["credit_spread_ratio"] = _cs_ratio
            out["credit_spread_zscore"] = (_cs_ratio - _cs_ma20) / (_cs_ratio.rolling(20, min_periods=10).std() + 1e-9)
            out["credit_spread_chg_5d"] = _cs_ratio.pct_change(5)
            out["credit_stress"]        = (_cs_ratio < _cs_ma20 * 0.98).astype(int)  # ratio falling 2%+

        # TLT: flight-to-safety indicator
        if _tlt is not None:
            _tlt_ma20 = _tlt.rolling(20, min_periods=10).mean()
            out["tlt_ret_5d"]   = _tlt.pct_change(5)
            out["tlt_above_ma"] = (_tlt > _tlt_ma20).astype(int)  # 1 = flight to safety = bearish equities

        # QQQ/SPY ratio: tech leadership (when QQQ outperforms = risk-on)
        if _qqq is not None:
            _spy_close = out["close"]  # our main ticker price
            _qqq_spy   = _qqq / (_spy_close.replace(0, np.nan))
            out["qqq_spy_ratio_chg_5d"] = _qqq_spy.pct_change(5)
            out["qqq_spy_momentum"]     = (_qqq_spy > _qqq_spy.rolling(21, min_periods=10).mean()).astype(int)

        # GLD: gold momentum (rising = fear or inflation = mixed for equities)
        if _gld is not None:
            out["gld_ret_21d"]  = _gld.pct_change(21)

        logger.debug("Credit spread / flight-to-safety features added")
    except Exception as _cs_err:
        logger.debug("Credit spread features skipped: %s", _cs_err)

    # --- CBOE SKEW + LQD + EEM cross-asset alpha signals ---
    # SKEW: tail-risk index.  High SKEW (>140) = OTM puts expensive = crash fear.
    # Contrarian: extreme SKEW often precedes short-term bounces (panic over-pricing).
    # LQD: investment-grade bonds. LQD/HYG ratio = credit quality spread.
    # EEM: emerging-market breadth; rising EM = global risk-on = equity tailwind.
    try:
        _ext_missing = [t for t in ["^SKEW", "LQD", "EEM", "IEF"]
                        if not (cross_asset and t in cross_asset)]
        _ext_raw = None
        if _ext_missing:
            import yfinance as _yf4
            _ext_raw = _yf4.download(
                _ext_missing,
                start=str(out.index.min().date()),
                end=str((out.index.max() + pd.Timedelta(days=5)).date()),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )

        def _ext_series(ticker):
            if cross_asset and ticker in cross_asset:
                return cross_asset[ticker].reindex(out.index).ffill().bfill()
            if _ext_raw is None or _ext_raw.empty:
                return None
            try:
                if isinstance(_ext_raw.columns, pd.MultiIndex):
                    s = _ext_raw[ticker]["Close"].squeeze()
                else:
                    s = _ext_raw["Close"].squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.reindex(out.index).ffill().bfill()
            except Exception:
                return None

        _skew_s = _ext_series("^SKEW")
        _lqd_s  = _ext_series("LQD")
        _eem_s  = _ext_series("EEM")
        _ief_s  = _ext_series("IEF")

        # CBOE SKEW index features
        if _skew_s is not None and _skew_s.notna().sum() > 50:
            _skew_mu  = _skew_s.rolling(63, min_periods=20).mean()
            _skew_sig = _skew_s.rolling(63, min_periods=20).std()
            out["skew_level"]    = _skew_s
            out["skew_zscore"]   = (_skew_s - _skew_mu) / (_skew_sig + 1e-9)
            out["skew_extreme"]  = (_skew_s > 140).astype(int)  # extreme tail fear
            out["skew_chg_5d"]   = _skew_s.diff(5)              # momentum
            out["skew_rank"]     = _skew_s.rolling(126, min_periods=30).rank(pct=True)
            # High SKEW + high VIX = double-fear = contrarian bullish
            if "vix_level" in out.columns:
                _vix_high = (out["vix_level"] > out["vix_level"].rolling(63, min_periods=20).quantile(0.70)).astype(int)
                out["skew_vix_fear"] = (_skew_s > 130).astype(int) * _vix_high

        # LQD: investment-grade credit quality proxy
        if _lqd_s is not None and _lqd_s.notna().sum() > 50:
            _lqd_ma20 = _lqd_s.rolling(20, min_periods=10).mean()
            out["lqd_ret_5d"]   = _lqd_s.pct_change(5)
            out["lqd_ret_21d"]  = _lqd_s.pct_change(21)
            out["lqd_zscore"]   = (_lqd_s - _lqd_ma20) / (_lqd_s.rolling(20, min_periods=10).std() + 1e-9)
            out["lqd_above_ma"] = (_lqd_s > _lqd_ma20).astype(int)  # 1 = investment-grade credit healthy

        # EEM: global emerging-market breadth
        if _eem_s is not None and _eem_s.notna().sum() > 50:
            _eem_ma21 = _eem_s.rolling(21, min_periods=10).mean()
            _eem_ma63 = _eem_s.rolling(63, min_periods=20).mean()
            out["eem_ret_21d"]   = _eem_s.pct_change(21)
            out["eem_above_ma"]  = (_eem_s > _eem_ma21).astype(int)  # 1 = global risk-on
            out["eem_momentum"]  = (_eem_ma21 > _eem_ma63).astype(int)  # medium-trend
            # EEM vs SPY divergence: when EEM lags SPY, US is outperforming = quality preference
            _spy_close = out["close"]
            _eem_spy   = _eem_s / (_spy_close.replace(0, np.nan))
            out["eem_spy_ret_21d"] = _eem_spy.pct_change(21)  # positive = global breadth expanding

        # IEF (7-10yr Treasury) vs TLT slope: intermediate vs long-end Treasury
        # When IEF/TLT ratio rises = yield curve steepening = normally bullish early cycle
        if _ief_s is not None and _ief_s.notna().sum() > 50:
            out["ief_ret_21d"]  = _ief_s.pct_change(21)
            out["ief_above_ma"] = (_ief_s > _ief_s.rolling(21, min_periods=10).mean()).astype(int)
            if "tlt_ret_5d" in out.columns:
                # IEF/TLT ratio: steepening yield curve signal
                _ief_tlt = _ief_s / (_ief_s.rolling(1).mean() + 1e-9)  # self-normalize first
                # Use relative momentum instead of ratio since levels differ
                out["ief_vs_tlt_mom"] = _ief_s.pct_change(21) - out.get("tlt_ret_5d", pd.Series(0, index=out.index))

        logger.debug("SKEW/LQD/EEM cross-asset features added")
    except Exception as _ext_err:
        logger.debug("SKEW/LQD/EEM features skipped: %s", _ext_err)

    # Defragment after cross-asset section
    out = out.copy()

    # --- Seasonality / calendar features ---
    # MQL5 Feature Engineering for ML (Part 3): sin/cos cyclical encoding
    # preserves periodicity — month 12 and month 1 are geometrically adjacent.
    # "Sell in May" (Nov–Apr outperformance) adds ~190bps vs momentum alone.
    _idx = out.index
    _month_s  = pd.Series(_idx.month,      index=_idx)
    _dow_s    = pd.Series(_idx.dayofweek,   index=_idx)   # 0=Mon, 4=Fri
    _dom_s    = pd.Series(_idx.day,         index=_idx)
    _quarter_s = pd.Series(_idx.quarter,   index=_idx)

    out["month_sin"]  = np.sin(2 * np.pi * _month_s / 12)
    out["month_cos"]  = np.cos(2 * np.pi * _month_s / 12)
    out["dow_sin"]    = np.sin(2 * np.pi * _dow_s   / 5)
    out["dow_cos"]    = np.cos(2 * np.pi * _dow_s   / 5)

    # "Sell in May" effect: Nov–Apr historically outperforms May–Oct by ~6–8%/yr
    out["favorable_season"] = _month_s.isin([11, 12, 1, 2, 3, 4]).astype(int)

    # Quarter-end / month-end institutional rebalancing flows
    out["quarter_end"] = ((_dom_s >= 28) & (_month_s.isin([3, 6, 9, 12]))).astype(int)
    out["month_end"]   = (_dom_s >= 28).astype(int)

    # January effect (new money inflows), sell in December (tax-loss harvesting)
    out["january"]  = (_month_s == 1).astype(int)
    out["december"] = (_month_s == 12).astype(int)

    # Turn-of-month / payday effects (Quantpedia composite seasonality research)
    out["month_start"]  = (_dom_s <= 3).astype(int)    # first 3 days: new money inflows
    out["payday"]       = (_dom_s == 15).astype(int)   # mid-month payroll → institutional buying
    out["week_of_month"] = ((_dom_s - 1) // 7 + 1)    # 1-5 (week within month)

    # Options expiration week (OpEx): 3rd Friday of each month.
    # OpEx week tends to have pinning effects and directional drift.
    def _opex_week_flag(dt_index: pd.DatetimeIndex) -> pd.Series:
        flags = pd.Series(0, index=dt_index)
        for dt in dt_index:
            # Find 3rd Friday of the month
            first_day = dt.replace(day=1)
            first_fri = first_day + pd.Timedelta(days=(4 - first_day.weekday()) % 7)
            third_fri = first_fri + pd.Timedelta(weeks=2)
            # Flag Mon-Fri of opex week (Mon before 3rd Fri through 3rd Fri)
            opex_mon = third_fri - pd.Timedelta(days=4)
            if opex_mon <= dt <= third_fri:
                flags.loc[dt] = 1
        return flags
    out["opex_week"] = _opex_week_flag(out.index)

    # --- Garman-Klass volatility (uses OHLC — 7.4× more efficient than close-to-close) ---
    # GJR-GARCH insight from MQL5: asymmetric vol — negative returns raise vol more.
    # GK gives a cleaner daily vol estimate; comparing to realized vol reveals compression.
    _log_hl = np.log((out["high"] + 1e-9) / (out["low"] + 1e-9))
    _log_co = np.log((out["close"] + 1e-9) / (out["open"] + 1e-9))
    _gk_daily  = 0.5 * _log_hl**2 - (2 * np.log(2) - 1.0) * _log_co**2
    _gk_vol_21 = np.sqrt(_gk_daily.rolling(21, min_periods=10).mean() * 252)
    out["gk_vol_21d"]         = _gk_vol_21
    out["gk_vs_realized_vol"] = _gk_vol_21 / (out["vol_21d"] + 1e-9)  # >1 = GK predicts higher vol
    # GK vol z-score (anomalous vol bursts)
    _gk_mu  = _gk_vol_21.rolling(63, min_periods=20).mean()
    _gk_sig = _gk_vol_21.rolling(63, min_periods=20).std()
    out["gk_vol_zscore"] = (_gk_vol_21 - _gk_mu) / (_gk_sig + 1e-9)

    # --- Hurst exponent (rolling 63-bar) — trend vs mean-reversion detector ---
    # H > 0.55: persistent trending market — momentum strategy has edge
    # H < 0.45: mean-reverting market — reversal strategy has edge
    # H ≈ 0.50: random walk — no systematic edge
    # Method: rescaled-range (R/S) analysis (efficient in pure numpy)
    def _hurst_rs(prices: np.ndarray) -> float:
        n = len(prices)
        if n < 20:
            return 0.5
        try:
            log_prices = np.log(prices + 1e-9)
            returns    = np.diff(log_prices)
            lags = [max(2, n // k) for k in range(2, min(10, n // 4 + 1))]
            lags = sorted(set(lags))
            if len(lags) < 2:
                return 0.5
            rs_vals = []
            for lag in lags:
                sub_returns = returns[:lag]
                mean_r  = sub_returns.mean()
                dev     = np.cumsum(sub_returns - mean_r)
                r_range = dev.max() - dev.min()
                s_std   = sub_returns.std() + 1e-12
                rs_vals.append(r_range / s_std)
            poly = np.polyfit(np.log(lags), np.log(np.maximum(rs_vals, 1e-9)), 1)
            return float(np.clip(poly[0], 0.0, 1.0))
        except Exception:
            return 0.5

    _hurst_win  = 63
    _close_np   = out["close"].values.astype(float)
    _hurst_vals = np.full(len(_close_np), np.nan)
    for _hi in range(_hurst_win, len(_close_np)):
        _hurst_vals[_hi] = _hurst_rs(_close_np[_hi - _hurst_win: _hi])
    out["hurst_63"]       = _hurst_vals
    out["hurst_trending"] = (pd.Series(_hurst_vals, index=out.index) > 0.55).astype(int)
    out["hurst_reverting"]= (pd.Series(_hurst_vals, index=out.index) < 0.45).astype(int)

    # --- Autocorrelation features (lag 1, 5) ---
    # Negative lag-1 autocorr → mean-reverting (overbought/oversold bounces)
    # Positive lag-1 autocorr → trending (momentum continuation)
    _ret1d = out["close"].pct_change(1)
    for _ac_lag in (1, 5):
        out[f"autocorr_{_ac_lag}d"] = _ret1d.rolling(21, min_periods=15).apply(
            lambda x: float(pd.Series(x).autocorr(lag=_ac_lag)) if len(x) > 2 * _ac_lag else np.nan,
            raw=False,
        )

    # --- Amihud illiquidity ratio (price impact per unit dollar volume) ---
    # High illiquidity → larger price moves per dollar traded → illiquid conditions
    # Predict volatility regime and future returns (high illiq → mean-reversion)
    _amihud_raw  = _ret1d.abs() / (volume + 1e-9)
    out["amihud_illiq"]   = _amihud_raw.rolling(21, min_periods=10).mean()
    _amid_mu  = _amihud_raw.rolling(63, min_periods=20).mean()
    _amid_sig = _amihud_raw.rolling(63, min_periods=20).std()
    out["amihud_zscore"]  = (_amihud_raw - _amid_mu) / (_amid_sig + 1e-9)

    # --- Sector rotation features (XLF/XLK/XLV/IWM relative to SPY) ---
    # Financial sector leading = bull. Tech leading = risk-on growth. Health leading = risk-off.
    # Small-cap (IWM) outperforming = breadth expansion (bullish). Fail silently.
    try:
        _sect_missing = [t for t in ["XLF", "XLK", "XLV", "XLE", "IWM"]
                         if not (cross_asset and t in cross_asset)]
        _sect_raw = None
        if _sect_missing:
            import yfinance as _yf4
            _sect_raw = _yf4.download(
                _sect_missing,
                start=str(out.index.min().date()),
                end=str((out.index.max() + pd.Timedelta(days=5)).date()),
                progress=False,
                auto_adjust=True,
                group_by="ticker",
            )

        def _sect_series(ticker):
            if cross_asset and ticker in cross_asset:
                return cross_asset[ticker].reindex(out.index).ffill().bfill()
            if _sect_raw is None or _sect_raw.empty:
                return None
            try:
                if isinstance(_sect_raw.columns, pd.MultiIndex):
                    s = _sect_raw[ticker]["Close"].squeeze()
                else:
                    s = _sect_raw["Close"].squeeze()
                s.index = pd.to_datetime(s.index).tz_localize(None)
                return s.reindex(out.index).ffill().bfill()
            except Exception:
                return None

        _spy_px = out["close"]

        _xlf = _sect_series("XLF")
        _xlk = _sect_series("XLK")
        _xlv = _sect_series("XLV")
        _xle = _sect_series("XLE")
        _iwm = _sect_series("IWM")

        # XLF/SPY: financial leadership → bull markets need banks to participate
        if _xlf is not None:
            _xlf_spy = _xlf / (_spy_px.replace(0, np.nan))
            _xlf_spy_ma21 = _xlf_spy.rolling(21, min_periods=10).mean()
            out["xlf_spy_momentum"] = (_xlf_spy > _xlf_spy_ma21).astype(int)
            out["xlf_spy_chg_21d"]  = _xlf_spy.pct_change(21)

        # XLK/SPY: tech leadership → risk-on growth positioning
        if _xlk is not None:
            _xlk_spy = _xlk / (_spy_px.replace(0, np.nan))
            _xlk_spy_ma21 = _xlk_spy.rolling(21, min_periods=10).mean()
            out["xlk_spy_momentum"] = (_xlk_spy > _xlk_spy_ma21).astype(int)
            out["xlk_spy_chg_5d"]   = _xlk_spy.pct_change(5)

        # XLV/SPY: defensive health care outperforming → risk-off signal (bearish for SPY)
        if _xlv is not None:
            _xlv_spy = _xlv / (_spy_px.replace(0, np.nan))
            _xlv_spy_ma21 = _xlv_spy.rolling(21, min_periods=10).mean()
            out["xlv_spy_defensive"] = (_xlv_spy > _xlv_spy_ma21).astype(int)  # 1=defensive=bearish

        # IWM/SPY: small-cap vs large-cap breadth
        if _iwm is not None:
            _iwm_spy = _iwm / (_spy_px.replace(0, np.nan))
            _iwm_spy_ma21 = _iwm_spy.rolling(21, min_periods=10).mean()
            out["iwm_spy_breadth"]  = (_iwm_spy > _iwm_spy_ma21).astype(int)   # 1=breadth expansion
            out["iwm_spy_chg_21d"]  = _iwm_spy.pct_change(21)

        # XLE/SPY: energy leading can signal inflation/stagflation regime (bearish)
        if _xle is not None:
            _xle_spy = _xle / (_spy_px.replace(0, np.nan))
            out["xle_spy_chg_21d"] = _xle_spy.pct_change(21)

        # Sector breadth composite: count sectors above their 21d MA (0-5).
        # When breadth is high (4-5 sectors trending), bull market is broad-based = durable.
        # When breadth collapses (0-1 sectors trending), rally is narrow = fragile (Fosback 1976).
        _breadth_parts = []
        for _sx, _s in [("xlf", _xlf), ("xlk", _xlk), ("xlv", _xlv), ("iwm", _iwm), ("xle", _xle)]:
            if _s is not None:
                _above = (_s > _s.rolling(21, min_periods=10).mean()).astype(int)
                _breadth_parts.append(_above)
        if _breadth_parts:
            _breadth_df = pd.concat(_breadth_parts, axis=1)
            out["sector_breadth_count"]  = _breadth_df.sum(axis=1)          # 0-5
            out["sector_breadth_pct"]    = _breadth_df.mean(axis=1)         # 0.0-1.0
            out["broad_bull_market"]     = (out["sector_breadth_count"] >= 4).astype(int)
            out["narrow_rally"]          = (out["sector_breadth_count"] <= 1).astype(int)

        logger.debug("Sector rotation features added")
    except Exception as _sect_err:
        logger.debug("Sector rotation features skipped: %s", _sect_err)

    # --- Momentum quality & acceleration features ---
    # Momentum quality = fraction of 63-day return from the most recent 21 days.
    # High quality (recent acceleration) → continuation. Low quality (front-loaded) → exhaustion.
    _ret63 = close.pct_change(63)
    _ret21 = close.pct_change(21)
    _ret_prior_42 = (1.0 + _ret63) / (1.0 + _ret21 + 1e-9) - 1.0   # return of days 21-63 ago
    _mom_quality = _ret21 / (_ret63.abs() + 1e-9)  # [-1, 1]: +1 = all recent, -1 = reversal
    out["mom_quality_63_21"]   = _mom_quality
    # Momentum acceleration: is short-term momentum > medium-term? (True = accelerating)
    out["mom_accel_21_63"]     = (_ret21 / (np.abs(_ret21) + 1e-9)) * (  # sign-weighted
        _ret21.abs() / (_ret63.abs() + 1e-9) - 0.333
    )
    # 5-day return vs 20-day average 5d-return (short-term acceleration)
    _ret5 = close.pct_change(5)
    out["mom_accel_5_20avg"]   = _ret5 - _ret5.rolling(20, min_periods=10).mean()

    # --- RSI divergence features ---
    # Bearish: price higher high but RSI lower high over trailing 10 bars
    # Bullish: price lower low but RSI higher low over trailing 10 bars
    _rsi14 = out["rsi_14"]
    _price_chg_10 = close.diff(10)
    _rsi_chg_10   = _rsi14.diff(10)
    out["rsi_divergence_bear"] = ((_price_chg_10 > 0) & (_rsi_chg_10 < -3)).astype(int)
    out["rsi_divergence_bull"] = ((_price_chg_10 < 0) & (_rsi_chg_10 > 3)).astype(int)
    out["rsi_divergence"]      = out["rsi_divergence_bull"] - out["rsi_divergence_bear"]

    # Defragment mid-section (96 column assignments between previous copy and end of section)
    out = out.copy()

    # --- Composite market favorability score ---
    # Combines 6 independent macro/technical conditions into a single -6 to +6 score.
    # Positive = conditions aligned for uptrend. Negative = unfavorable.
    # This lets the model see the aggregate signal without needing to learn all interactions.
    _fav_components = pd.DataFrame(index=out.index)
    _fav_components["vix_contango"]   = out.get("vix_contango", pd.Series(0, index=out.index))
    _fav_components["fav_season"]     = out.get("favorable_season", pd.Series(0, index=out.index))
    _fav_components["yc_normal"]      = (1 - out.get("yield_curve_inverted", pd.Series(0, index=out.index)))
    _fav_components["hurst_trend"]    = out.get("hurst_trending", pd.Series(0, index=out.index))
    _fav_components["xlf_lead"]       = out.get("xlf_spy_momentum", pd.Series(0, index=out.index))
    _fav_components["qqq_lead"]       = out.get("qqq_spy_momentum", pd.Series(0, index=out.index))
    _fav_components["breadth_ok"]     = out.get("iwm_spy_breadth", pd.Series(0, index=out.index))
    _fav_components["cs_normal"]      = (1 - out.get("credit_stress", pd.Series(0, index=out.index)))
    out["market_fav_score"]   = _fav_components.sum(axis=1)           # 0–8 raw
    out["market_fav_norm"]    = out["market_fav_score"] / 8.0         # 0–1 normalised
    out["market_fav_rank"]    = out["market_fav_score"].rolling(252, min_periods=63).rank(pct=True)

    # --- FOMC meeting proximity (Fed rally effect) ---
    # Day OF and AFTER Fed meeting → positive drift (uncertainty resolved).
    # 1-2 days BEFORE → slight negative (uncertainty).  Dates 2018-2024 hard-coded.
    _fomc_dates = pd.to_datetime([
        # 2018
        "2018-01-31","2018-03-21","2018-05-02","2018-06-13",
        "2018-08-01","2018-09-26","2018-11-08","2018-12-19",
        # 2019
        "2019-01-30","2019-03-20","2019-05-01","2019-06-19",
        "2019-07-31","2019-09-18","2019-10-30","2019-12-11",
        # 2020
        "2020-01-29","2020-03-03","2020-03-15","2020-04-29",
        "2020-06-10","2020-07-29","2020-09-16","2020-11-05","2020-12-16",
        # 2021
        "2021-01-27","2021-03-17","2021-04-28","2021-06-16",
        "2021-07-28","2021-09-22","2021-11-03","2021-12-15",
        # 2022
        "2022-01-26","2022-03-16","2022-05-04","2022-06-15",
        "2022-07-27","2022-09-21","2022-11-02","2022-12-14",
        # 2023
        "2023-02-01","2023-03-22","2023-05-03","2023-06-14",
        "2023-07-26","2023-09-20","2023-11-01","2023-12-13",
    ])
    _fomc_prox = pd.Series(0.0, index=out.index)
    _idx_norm = out.index.normalize()  # strip time component for calendar-day matching
    for _fd in _fomc_dates:
        _days = (_idx_norm - _fd).days
        _fomc_prox[_days == 0]                    = 3.0   # Fed day
        _fomc_prox[_days == 1]                    = 2.0   # Day after
        _fomc_prox[(_days == -1) | (_days == -2)] = -1.0  # Pre-Fed uncertainty
    out["fomc_proximity"]   = _fomc_prox
    out["fomc_day"]         = (_fomc_prox == 3.0).astype(int)
    out["fomc_post"]        = (_fomc_prox == 2.0).astype(int)

    # --- Asymmetric volatility (downside vs upside vol) ---
    # Negative returns raise future vol more than positive returns (leverage effect).
    # Downvol/upvol > 1 = fear dominant = bearish.  < 1 = calm = bullish.
    _ret1d_s = close.pct_change(1)
    _down_vol = _ret1d_s.clip(upper=0).rolling(21, min_periods=10).std() * np.sqrt(252)
    _up_vol   = _ret1d_s.clip(lower=0).rolling(21, min_periods=10).std() * np.sqrt(252)
    out["downvol_21d"]     = _down_vol
    out["asymmetric_vol"]  = _down_vol / (_up_vol + 1e-9)    # >1 = bearish skew
    out["vol_asymmetry_z"] = (out["asymmetric_vol"] - out["asymmetric_vol"].rolling(63, min_periods=20).mean()) / \
                              (out["asymmetric_vol"].rolling(63, min_periods=20).std() + 1e-9)

    # --- ADX trend acceleration (is the trend strengthening or weakening?) ---
    # Rising ADX → trend strengthening → momentum strategies work better.
    # ADX > 25 AND rising → strong trend worth following.
    out["adx_chg_5d"]     = out["adx_14"].diff(5)
    out["adx_chg_10d"]    = out["adx_14"].diff(10)
    out["adx_trending"]   = ((out["adx_14"] > 25) & (out["adx_chg_5d"] > 0)).astype(int)
    out["adx_strength"]   = out["adx_14"] * out["adx_chg_5d"] / (100 + 1e-9)  # magnitude × direction

    # --- Full 12-month momentum (no skip-month adjustment) ---
    # Alpha Architect research (May 2026): skip-month is not always necessary.
    # Adding both allows the model to decide which is more predictive.
    out["mom_12_0"] = close.pct_change(252).fillna(0.0)  # full 12-month including last month
    out["mom_1m"]   = close.pct_change(21).fillna(0.0)  # 1-month standalone (mean-reversion at short horizon)
    out["mom_3m"]   = close.pct_change(63).fillna(0.0)  # 3-month momentum

    # --- Volatility Risk Premium (VRP) ---
    # VRP = VIX (implied vol) - realized vol. Documented in Quantpedia #0020.
    # Positive VRP → options are expensive → market expects more vol than realised → bearish vol / bullish equity.
    # When VRP is negative (realized > implied) → vol is spiking and surprising → equity headwind.
    if "vix_level" in out.columns:
        _vrp = out["vix_level"] / 100.0 - out["vol_21d"]          # both annualised
        out["vrp_raw"]    = _vrp
        _vrp_mu  = _vrp.rolling(63, min_periods=20).mean()
        _vrp_sig = _vrp.rolling(63, min_periods=20).std()
        out["vrp_zscore"]  = (_vrp - _vrp_mu) / (_vrp_sig + 1e-9)
        out["vrp_rank"]    = _vrp.rolling(252, min_periods=63).rank(pct=True)
        out["vrp_positive"] = (_vrp > 0).astype(int)              # 1 = calm market = equity-bullish
        out["vrp_chg_5d"]   = _vrp.diff(5)                        # VRP momentum: rising = calming market
        out["vrp_chg_21d"]  = _vrp.diff(21)                       # VRP trend: multi-week shift

    # --- Pre-holiday calendar effect (Quantpedia #0083) ---
    # Returns on the 1-4 trading days before US market holidays are ~10x normal days.
    # Hard-coded US market holiday months/days (fixed-date + approximate for floating).
    try:
        import pandas.tseries.holiday as _cal_mod
        from pandas.tseries.offsets import CustomBusinessDay
        _us_cal = _cal_mod.USFederalHolidayCalendar()
        _holidays = _us_cal.holidays(start=str(out.index.min().date()),
                                      end=str(out.index.max().date()))
        _pre_holiday = pd.Series(0, index=out.index)
        for _hdate in _holidays:
            for _days_before in range(1, 5):
                _pre_day = _hdate - pd.Timedelta(days=_days_before)
                if _pre_day in out.index:
                    # Weight: closest day gets highest value
                    _pre_holiday.loc[_pre_day] = max(_pre_holiday.loc[_pre_day], 5 - _days_before)
        out["pre_holiday"]        = _pre_holiday.astype(int)      # 0=normal, 1-4 = closeness to holiday
        out["pre_holiday_flag"]   = (_pre_holiday >= 1).astype(int)
    except Exception as _phe:
        logger.debug("Pre-holiday features skipped: %s", _phe)

    # --- Volume-Price Divergence (Wyckoff accumulation/distribution) ---
    # Price up + volume declining → distribution (bearish). Price down + volume declining → accumulation (bullish).
    # Implemented as rolling correlation between return and volume change.
    _vol_chg = volume.pct_change(1)
    _ret5_s  = close.pct_change(5)
    # Negative rolling correlation between price change and volume = bearish divergence
    out["price_vol_corr_10d"] = _ret5_s.rolling(10, min_periods=5).corr(volume)
    out["price_vol_corr_21d"] = close.pct_change(1).rolling(21, min_periods=10).corr(volume)
    # Distribution signal: price making highs but volume falling
    _price_high_10  = (close == close.rolling(10, min_periods=5).max()).astype(int)
    _vol_below_avg  = (volume < volume.rolling(20, min_periods=10).mean()).astype(int)
    out["distribution_signal"] = (_price_high_10 & _vol_below_avg).astype(int)
    # Accumulation signal: price making lows but volume falling (selling exhaustion)
    _price_low_10   = (close == close.rolling(10, min_periods=5).min()).astype(int)
    out["accumulation_signal"] = (_price_low_10 & _vol_below_avg).astype(int)
    # OBV momentum: 5-day vs 20-day OBV MA divergence
    _obv_series = _obv(close, volume)
    _obv_ma5  = _obv_series.rolling(5,  min_periods=3).mean()
    _obv_ma20 = _obv_series.rolling(20, min_periods=10).mean()
    out["obv_momentum"] = (_obv_ma5 / (_obv_ma20.abs() + 1e-9) - 1.0)

    # --- Overnight vs Intraday return accumulation (institutional flow proxy) ---
    # When overnight returns accumulate faster than intraday → institutional buying → bullish.
    # Overnight = gap (open/prev_close - 1). Intraday = close/open - 1.
    _overnight_cumret = out["gap"].fillna(0).rolling(21, min_periods=10).mean()
    _intraday_cumret  = out["intraday_ret"].fillna(0).rolling(21, min_periods=10).mean()
    out["overnight_vs_intraday"]    = _overnight_cumret - _intraday_cumret
    out["overnight_lead"]           = (_overnight_cumret > _intraday_cumret).astype(int)

    # --- 52-week high proximity breakout (Quantpedia #0144 trend-following) ---
    # Stocks at new 52-week highs have strong momentum continuation (George & Hwang 2004).
    # Distance from 52W high as a continuous feature + binary breakout flag.
    _high_252 = close.rolling(252, min_periods=63).max()
    out["dist_from_52w_high"]     = (close - _high_252) / (_high_252 + 1e-9)  # 0 at high, negative below
    out["near_52w_high"]          = (out["dist_from_52w_high"] > -0.02).astype(int)   # within 2% of high
    out["at_52w_high"]            = (out["dist_from_52w_high"] > -0.005).astype(int)  # new high breakout

    # --- Relative Volume (RVOL) — liquidity-adjusted sentiment ---
    # RVOL > 1 = unusually active (institutional interest). Combined with direction = strong signal.
    # High RVOL on up day → confirmed accumulation. High RVOL on down day → distribution.
    _vol_avg20 = volume.rolling(20, min_periods=10).mean()
    _rvol = volume / (_vol_avg20 + 1e-9)
    out["rvol"]               = _rvol                      # raw ratio (>1 = above avg volume)
    out["rvol_rank"]          = _rvol.rolling(63, min_periods=20).rank(pct=True)
    _ret1d_close = close.pct_change(1)
    out["rvol_up"]   = (_rvol * (_ret1d_close > 0).astype(float))    # high vol + up day
    out["rvol_down"] = (_rvol * (_ret1d_close < 0).astype(float))    # high vol + down day
    out["rvol_confirm"] = ((_rvol > 1.5) & (_ret1d_close > 0)).astype(int)  # strong accumulation

    # --- Bollinger Band Squeeze — volatility compression before breakout ---
    # When BB width drops to its 20th percentile → coiled spring → breakout imminent.
    # Direction depends on trend; use with cross_50_200 or regime.
    if "bb_width" in out.columns:
        _bb_w_p20 = out["bb_width"].rolling(63, min_periods=20).quantile(0.20)
        _bb_w_p80 = out["bb_width"].rolling(63, min_periods=20).quantile(0.80)
        out["bb_squeeze"]       = (out["bb_width"] < _bb_w_p20).astype(int)  # compressed = breakout pending
        out["bb_expansion"]     = (out["bb_width"] > _bb_w_p80).astype(int)  # wide = trend in progress
        out["bb_width_rank"]    = out["bb_width"].rolling(126, min_periods=40).rank(pct=True)

    # --- Turn-of-Month Effect (last trading day) ---
    # Last trading day of month + first 3 days show consistent positive drift (Lakonishok & Smidt 1988).
    # month_start already covers days 1-3; add the last-day effect.
    _bdom = out.index.to_series().apply(lambda dt: (dt + pd.offsets.MonthEnd(0)).date() == dt.date())
    out["month_last_day"] = _bdom.astype(int)   # last calendar day of month (often highest drift)
    # "Turn zone": last day + first 3 days combined
    out["tom_zone"] = ((out.get("month_start", pd.Series(0, index=out.index)) == 1) |
                       out["month_last_day"].astype(bool)).astype(int)

    # --- Return consistency (trend quality) ---
    # Counts the fraction of days with positive returns over N bars.
    # High consistency (>70%) = persistent uptrend = momentum with low noise.
    # Low consistency (<30%) = choppy market = unreliable momentum.
    _ret1d_close = close.pct_change(1)
    _pos_days_10  = (_ret1d_close > 0).rolling(10,  min_periods=5).mean()
    _pos_days_21  = (_ret1d_close > 0).rolling(21,  min_periods=10).mean()
    _pos_days_63  = (_ret1d_close > 0).rolling(63,  min_periods=20).mean()
    out["ret_consistency_10d"] = _pos_days_10   # 0–1 fraction of up days in 10-bar window
    out["ret_consistency_21d"] = _pos_days_21
    out["trend_quality_10_21"] = _pos_days_10 - _pos_days_21  # accelerating vs decelerating consistency
    out["strong_uptrend"]      = (_pos_days_21 > 0.65).astype(int)   # 65%+ up days = confirmed bull
    out["choppy_market"]       = ((_pos_days_21 > 0.40) & (_pos_days_21 < 0.60)).astype(int)

    # --- Current winning/losing streak length (vectorized) ---
    # Current streak = number of consecutive same-direction days up to today.
    # Rising streak → momentum continuation signal.
    _up_day   = (_ret1d_close > 0).astype(int)
    _dn_day   = (_ret1d_close < 0).astype(int)
    # Reset streak counter to 0 on direction change, then cumulate
    _up_grp   = (_up_day != _up_day.shift(1)).cumsum()
    _dn_grp   = (_dn_day != _dn_day.shift(1)).cumsum()
    out["up_streak"]   = _up_day.groupby(_up_grp).cumsum() * _up_day    # 0 on down days
    out["down_streak"] = _dn_day.groupby(_dn_grp).cumsum() * _dn_day    # 0 on up days

    # --- Mean-Reversion after Extreme Moves ---
    # Daily return extremes (> ±3σ) tend to partially reverse in 1-5 trading days.
    # Academic: Jegadeesh (1990) 1-month reversal; Conrad & Kaul (1989) weekly reversal.
    _ret1d_z = (_ret1d_close - _ret1d_close.rolling(63, min_periods=20).mean()) / \
               (_ret1d_close.rolling(63, min_periods=20).std() + 1e-9)
    out["extreme_down"]     = (_ret1d_z < -2.5).astype(int)   # extreme down day → mean-revert up
    out["extreme_up"]       = (_ret1d_z >  2.5).astype(int)   # extreme up day → mean-revert down
    out["rev_signal_5d"]    = out["extreme_down"].shift(1).fillna(0).rolling(5, min_periods=1).max() - \
                              out["extreme_up"].shift(1).fillna(0).rolling(5, min_periods=1).max()

    # Defragment after all technical features before regime/label section
    out = out.copy()

    # --- Regime labels ---
    if add_regime:
        out = add_regime_labels(out)

    # --- HMM regime probability features (adaptive, data-driven regime) ---
    # Fitted on the available history; predicts P(bull/bear/sideways) per bar.
    # Complements the simple 252-bar threshold with a probabilistic signal.
    try:
        from features.regime_hmm import HMMRegimeDetector
        _top_cfg = load_config()
        _hmm_cfg = _top_cfg.get("hmm_regime", {})
        _hmm_det = HMMRegimeDetector(**_hmm_cfg)
        _hmm_det.fit(out)
        _hmm_proba = _hmm_det.predict_proba(out)
        for _hcol in _hmm_proba.columns:
            out[f"hmm_prob_{_hcol}"] = _hmm_proba[_hcol]
        _hmm_label_int = {"bull": 2, "sideways": 1, "bear": 0}
        _hmm_labels = _hmm_det.predict(out).map(_hmm_label_int).fillna(1).astype(int)
        out["hmm_regime"] = _hmm_labels
    except Exception as _hmme:
        logger.warning("HMM regime features skipped: %s", _hmme)

    # --- Forward-return labels (kept separate — excluded from X) ---
    fwd = close.pct_change(target_horizon).shift(-target_horizon)
    out["fwd_return"] = fwd
    out["label"] = (fwd > 0).astype(int)

    # --- Regime-adaptive triple-barrier labels ---
    # Different PT/SL multipliers per regime improve label quality:
    #   Bull  → wider PT (2.0): let winners run; avoid premature +1 labeling on weak moves
    #   Bear  → tighter PT (1.2): bear rallies are short; SL=0.8 gives more -1 labels
    #   Sideways/unknown → symmetric balanced (pt=1.5, sl=1.0)
    # Without this, using fixed pt=1.5 in a bull market labels weak 1.5σ moves as +1,
    # which is a noisy signal that the model partly learns from (degrades bear performance).
    _tb_labels = pd.Series(0,   index=close.index, dtype=int,   name="tb_label")
    _tb_rets   = pd.Series(0.0, index=close.index, dtype=float, name="tb_ret")
    if "hmm_regime" in out.columns:
        _regime_tb_params = {
            2: (2.0, 0.8),   # bull:     wide PT, tight SL — let long trends breathe
            1: (1.5, 1.0),   # sideways: balanced — same as old default
            0: (1.2, 0.8),   # bear:     tight PT, tight SL — bear rallies are quick
        }
        for _rv, (_pt, _sl) in _regime_tb_params.items():
            _rmask = out["hmm_regime"] == _rv
            if _rmask.sum() < 20:
                continue
            _tb_sub = triple_barrier_labels(close, t_bars=target_horizon,
                                            pt_multiplier=_pt, sl_multiplier=_sl)
            _tb_labels[_rmask] = _tb_sub["tb_label"].reindex(close.index).fillna(0).astype(int)[_rmask]
            _tb_rets[_rmask]   = _tb_sub["tb_ret"].reindex(close.index).fillna(0.0)[_rmask]
        # Warm-up bars (before HMM has enough data) fall back to balanced defaults
        _warmup = _tb_labels == 0
        if _warmup.any():
            _tb_default = triple_barrier_labels(close, t_bars=target_horizon)
            _tb_labels[_warmup] = _tb_default["tb_label"].reindex(close.index).fillna(0).astype(int)[_warmup]
            _tb_rets[_warmup]   = _tb_default["tb_ret"].reindex(close.index).fillna(0.0)[_warmup]
    else:
        _tb_default = triple_barrier_labels(close, t_bars=target_horizon)
        _tb_labels  = _tb_default["tb_label"].reindex(out.index).fillna(0).astype(int)
        _tb_rets    = _tb_default["tb_ret"].reindex(out.index).fillna(0.0)
    out["tb_label"] = _tb_labels
    out["tb_ret"]   = _tb_rets

    # --- Change-point detection features (regime break signals) ---
    try:
        from features.change_point import add_change_point_features
        out = add_change_point_features(out, col="close")
    except Exception as _cpe:
        logger.debug("change_point features skipped: %s", _cpe)

    # --- Multi-timeframe confirmation features ---
    # Adds mtf_5, mtf_21, mtf_63 (directional signal per TF), mtf_agree, mtf_confirmed.
    # These give the model context about whether signals align across timeframes.
    try:
        from features.multi_timeframe import MultiTimeframeConfirmation
        _mtf = MultiTimeframeConfirmation(timeframes=(5, 21, 63), min_agreement=0.67)
        out = _mtf.add_features(out, close_col="close")
    except Exception as _mtfe:
        logger.debug("multi_timeframe features skipped: %s", _mtfe)

    # --- Quantile signal (teacher signal during training; 0 at inference for recent bars) ---
    # Forward return quantile: +1 if future return is top 30%, -1 if bottom 30%, 0 otherwise.
    # At inference for the most recent bar, the future is unknown so this is 0.
    # Acts as a "teacher" guiding XGBoost to learn which feature combinations predict good outcomes.
    # A second PAST-return quantile feature provides a pure look-ahead-free signal.
    try:
        _fwd_ret_q = out["close"].pct_change(target_horizon).shift(-target_horizon)
        _ql_fwd = quantile_labels(_fwd_ret_q, q_upper=0.70, q_lower=0.30, rolling_window=126)
        out["quantile_signal"] = _ql_fwd.reindex(out.index).fillna(0).astype(int)
    except Exception as _qle:
        logger.debug("quantile_signal feature skipped: %s", _qle)

    # Past return quantile (no look-ahead — usable at inference)
    try:
        _past_ret = out["close"].pct_change(target_horizon)
        _ql_past = quantile_labels(_past_ret, q_upper=0.70, q_lower=0.30, rolling_window=126)
        out["past_ret_quantile"] = _ql_past.reindex(out.index).fillna(0).astype(int)
    except Exception as _qle:
        logger.debug("past_ret_quantile feature skipped: %s", _qle)

    # --- Fractionally differentiated price features (preserves memory, near-stationary) ---
    if add_fracdiff:
        try:
            from features.fracdiff import add_fracdiff_features
            out = add_fracdiff_features(out, d=fracdiff_d, columns=["close"], threshold=1e-3)
        except Exception as _fde:
            logger.debug("fracdiff skipped: %s", _fde)

    # Drop warmup NaN rows — only use columns with >50% valid data so that
    # optional cross-asset columns (VIX, yield curve) don't eliminate rows
    # when their underlying data source is unavailable.
    n_rows = len(out)
    feat_cols_for_dropna = [
        c for c in out.columns
        if c not in _LABEL_COLS
        and n_rows > 0
        and out[c].notna().sum() > n_rows * 0.5
    ]
    if feat_cols_for_dropna:
        out = out.dropna(subset=feat_cols_for_dropna)

    # Data version fingerprint — used by VersionRegistry and DataQualityPipeline
    out["_feature_version"] = hashlib.sha256(
        out.select_dtypes(include="number").values.tobytes()
    ).hexdigest()[:16]

    logger.info(
        "generate_features: %d rows × %d columns (as_of=%s)",
        len(out),
        out.shape[1],
        as_of_date or "none",
    )
    return out


# =============================================================================
# Feature selection
# =============================================================================

class FeatureSelector:
    """
    Two-stage feature selector:
      1. Score by mutual information + optional SHAP (fitted on training slice)
      2. Prune pairs with Pearson correlation > max_correlation
      3. Keep top `n_features` by combined importance

    Only fit() touches labels — transform() is pure pass-through on the
    column list decided at fit time, so it is safe to apply to any subset.
    """

    def __init__(
        self,
        n_features: int = 15,
        max_correlation: float = 0.85,
        use_shap: bool = True,
        random_state: int = 42,
    ) -> None:
        self.n_features = n_features
        self.max_correlation = max_correlation
        self.use_shap = use_shap
        self.random_state = random_state
        self.selected_features_: list[str] = []
        self.importance_: pd.Series = pd.Series(dtype=float)
        self._fitted = False

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """
        Learn which features to keep.  X and y must cover only the training
        period — never pass future labels here.
        """
        from sklearn.feature_selection import mutual_info_classif

        mask = X.notna().all(axis=1) & y.notna()
        Xc, yc = X[mask].copy(), y[mask].copy()

        # --- Mutual information score ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mi = mutual_info_classif(Xc, yc, random_state=self.random_state)
        mi_scores = pd.Series(mi, index=Xc.columns)

        # --- Optional SHAP importance ---
        shap_scores = self._shap_importance(Xc, yc)

        # --- Combined rank (60/40 MI/SHAP, or 100% MI if SHAP not available) ---
        if shap_scores is not None:
            mi_rank   = mi_scores.rank(ascending=True)    / len(mi_scores)
            shap_rank = shap_scores.rank(ascending=True)  / len(shap_scores)
            combined  = 0.6 * mi_rank + 0.4 * shap_rank
        else:
            combined = mi_scores.rank(ascending=True) / len(mi_scores)

        combined = combined.sort_values(ascending=False)
        self.importance_ = combined

        # --- Correlation pruning ---
        selected = self._prune_correlated(Xc, combined)

        self.selected_features_ = selected[: self.n_features]
        self._fitted = True
        logger.info(
            "FeatureSelector: kept %d/%d features (corr_threshold=%.2f)",
            len(self.selected_features_),
            len(X.columns),
            self.max_correlation,
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        available = [c for c in self.selected_features_ if c in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------

    def _shap_importance(self, X: pd.DataFrame, y: pd.Series) -> Optional[pd.Series]:
        if not self.use_shap:
            return None
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            logger.debug("shap or sklearn not available — using MI only")
            return None

        try:
            clf = RandomForestClassifier(
                n_estimators=60,
                max_depth=4,
                random_state=self.random_state,
                n_jobs=-1,
            )
            clf.fit(X, y)
            explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
            shap_vals = explainer.shap_values(X, check_additivity=False)
            # shap_values returns list[class_0_arr, class_1_arr] for classifiers
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            mean_abs = np.abs(shap_vals).mean(axis=0)
            return pd.Series(mean_abs, index=X.columns)
        except Exception as exc:
            logger.debug("SHAP computation failed (%s) — using MI only", exc)
            return None

    def _prune_correlated(self, X: pd.DataFrame, ranked: pd.Series) -> list[str]:
        """
        Walk features in importance order; drop any candidate that is
        correlated > threshold with an already-accepted feature.
        """
        corr = X.corr().abs()
        accepted: list[str] = []
        for feat in ranked.index:
            if feat not in corr.columns:
                continue
            if not accepted:
                accepted.append(feat)
                continue
            max_corr_with_accepted = corr.loc[feat, accepted].max()
            if max_corr_with_accepted < self.max_correlation:
                accepted.append(feat)
        return accepted


# =============================================================================
# FeatureEngine  (class API — backward-compatible with models/train.py)
# =============================================================================

class FeatureEngine:
    """
    Wraps generate_features() + FeatureSelector into the class-based API
    expected by models/train.py, main.py, and paper_trading/loop.py.

    Backward-compatible contract
    ----------------------------
      fe.build(df, ticker, save=True)     → DataFrame with features + label
      fe.feature_columns                  → list of selected feature column names
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg = config or load_config()
        self.feat_cfg   = self.cfg.get("features", {})
        self.model_cfg  = self.cfg.get("model", {})
        self.regime_cfg = self.cfg.get("regimes", {})
        self.processed_dir = ensure_dir(
            self.cfg.get("data", {}).get("processed_dir", "data/processed")
        )
        self._selected_cols: list[str] = []   # populated after first select()
        self._selector: Optional[FeatureSelector] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        df: pd.DataFrame,
        ticker: str,
        save: bool = True,
        as_of_date: Optional[str] = None,
        select: bool = False,
    ) -> pd.DataFrame:
        """
        Generate features from raw OHLCV data.

        Parameters
        ----------
        df         : Raw OHLCV DataFrame.
        ticker     : Asset label (used for file naming).
        save       : Persist to Parquet with version stamp.
        as_of_date : Hard cutoff enforced before any computation.
        select     : Run FeatureSelector on the training portion and
                     narrow to selected_features_.  Requires 'label' column.
        """
        horizon = self.model_cfg.get("target_horizon", 5)
        features = generate_features(
            df,
            as_of_date=as_of_date,
            target_horizon=horizon,
            add_regime=True,
        )

        if select and "label" in features.columns:
            features = self._run_selection(features)

        logger.info("Feature matrix: %d rows × %d columns", *features.shape)

        if save:
            self._save_versioned(features, ticker)

        return features

    def select_features(
        self,
        features: pd.DataFrame,
        train_end_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fit selector on training slice and transform full feature matrix.

        train_end_idx : last row index of training set.  If None, uses 70%.
        """
        return self._run_selection(features, train_end_idx=train_end_idx)

    @property
    def feature_columns(self) -> list[str]:
        """
        Returns the selected feature columns if selection has been run,
        otherwise the full default list for backward compatibility.
        """
        if self._selected_cols:
            return list(self._selected_cols)
        return list(_DEFAULT_FEATURE_COLS)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_selection(
        self,
        features: pd.DataFrame,
        train_end_idx: Optional[int] = None,
    ) -> pd.DataFrame:
        all_feat_cols = _get_feature_cols(features)
        if not all_feat_cols:
            return features

        split = train_end_idx if train_end_idx is not None else int(len(features) * 0.70)
        X_train = features[all_feat_cols].iloc[:split]
        y_train = features["label"].iloc[:split]

        n_target = self.feat_cfg.get("n_selected_features", 15)
        max_corr  = self.feat_cfg.get("max_feature_correlation", 0.85)

        self._selector = FeatureSelector(
            n_features=n_target,
            max_correlation=max_corr,
            use_shap=self.feat_cfg.get("use_shap", True),
            random_state=self.model_cfg.get("random_state", 42),
        )
        self._selector.fit(X_train, y_train)
        self._selected_cols = self._selector.selected_features_

        keep = self._selected_cols + [
            c for c in ("label", "fwd_return", "regime",
                        "regime_bull", "regime_bear", "regime_hv",
                        "open", "high", "low", "close", "volume")
            if c in features.columns
        ]
        return features[[c for c in keep if c in features.columns]]

    def _save_versioned(self, features: pd.DataFrame, ticker: str) -> Path:
        """Save with a version tag derived from the feature set shape + timestamp."""
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        slug = ticker.lower().replace("-", "_")
        path = self.processed_dir / f"{slug}_features_{ts}.parquet"
        features.to_parquet(path)
        # Also write a 'latest' symlink-equivalent: overwrite the canonical name
        latest = self.processed_dir / f"{slug}_features.parquet"
        features.to_parquet(latest)
        logger.info("Features saved → %s (latest: %s)", path.name, latest.name)
        return path


# =============================================================================
# Helpers used internally
# =============================================================================

def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All columns that are neither OHLCV, labels, regime flags, nor metadata."""
    exclude = _LABEL_COLS | _OHLCV_COLS | {
        "regime", "regime_bull", "regime_bear", "regime_hv", "_feature_version"
    }
    return [c for c in df.columns if c not in exclude]


# Default feature list preserved for backward compat (used before selection runs)
_DEFAULT_FEATURE_COLS = [
    # Price momentum
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    "roc_5d", "roc_10d", "roc_21d",
    "mom_12_1", "mom_6_1",
    # Trend
    "sma_20_dist", "sma_50_dist", "sma_200_dist",
    "cross_20_50", "cross_50_200",
    # Oscillators
    "rsi_14", "rsi_norm",
    "macd_norm", "macd_hist",
    "stoch_k", "stoch_d",
    # Volatility
    "vol_5d", "vol_21d", "vol_ratio_5_21",
    "atr_ratio", "bb_pct", "bb_width", "vol_zscore",
    "gk_vol_21d", "gk_vs_realized_vol",
    # Volume
    "volume_ratio", "volume_zscore", "obv_trend",
    # Trend quality
    "adx_norm", "trend_strength",
    "channel_pos_52w", "higher_highs_10", "higher_lows_10",
    "hurst_63", "hurst_trending",
    # VIX / fear
    "vix_level", "vix_ratio", "vix_rank", "vix_spike",
    "vix_ts_slope", "vix_contango",
    # Macro
    "yield_curve_10_3m", "yield_curve_inverted", "yield_curve_zscore",
    "dxy_ret_5d", "dxy_zscore",
    # Seasonality
    "month_sin", "month_cos", "dow_sin", "dow_cos",
    "favorable_season", "quarter_end",
    # Microstructure
    "amihud_zscore", "autocorr_1d",
    "gap", "intraday_ret",
    # Sector rotation
    "xlf_spy_momentum", "xlf_spy_chg_21d",
    "xlk_spy_momentum", "xlk_spy_chg_5d",
    "xlv_spy_defensive", "iwm_spy_breadth", "iwm_spy_chg_21d",
    # Momentum quality & acceleration
    "mom_quality_63_21", "mom_accel_21_63", "mom_accel_5_20avg",
    # RSI divergence
    "rsi_divergence", "rsi_divergence_bear", "rsi_divergence_bull",
    # Composite market score
    "market_fav_score", "market_fav_norm", "market_fav_rank",
    # Volatility Risk Premium
    "vrp_raw", "vrp_zscore", "vrp_rank", "vrp_positive",
    # Pre-holiday effect
    "pre_holiday", "pre_holiday_flag",
    # Volume-price divergence (Wyckoff)
    "price_vol_corr_10d", "price_vol_corr_21d",
    "distribution_signal", "accumulation_signal", "obv_momentum",
    # Overnight vs intraday institutional flow
    "overnight_vs_intraday", "overnight_lead",
    # 52-week high proximity breakout
    "dist_from_52w_high", "near_52w_high", "at_52w_high",
    # Calendar anomalies (OpEx, payday, turn-of-month)
    "month_start", "payday", "week_of_month", "opex_week",
    # VVIX tail risk
    "vvix_level", "vvix_zscore", "vvix_spike",
    # VIX fear/calm zones
    "vix_fear_zone", "vix_calm_zone",
    # Past return quantile (no look-ahead)
    "past_ret_quantile",
]
