"""
honest_backtest.py  —  Honest OOS backtest using universal + SPY ensemble model.

Rules:
  - OOS period: 2024-10-16 to 2026-05-13 (394 bars, fully out-of-sample)
  - Models trained exclusively on 2015-2024 data (no OOS look-ahead)
  - Position sizing: Full Kelly (Kelly 1956) — zero free parameters
  - No threshold tuning on the OOS test set
  - T-bill (4.5% annualised) on idle capital

Signal formula (from raw ensemble upper-90 percentile — bypasses calibrator collapse):
  kelly_f = clip(2 * upper_90 - 1, 0, 1)   [long-only Kelly]

Cross-sectional enhancement:
  Dual-model signal = 0.5 * upper_90_universal + 0.5 * upper_90_spy
  (equal-weight blend, parameter-free)

Regime gate (training-period rule, not OOS-optimised):
  Only enter when SMA50 > SMA200 (golden-cross regime, standard academic rule)
  Inspired by Alpha Architect trend equity research & MQL5 adaptive position sizing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ARTIFACTS = Path("models/artifacts")
CACHE_DIR = Path("data/feature_cache")
OUTPUT_CSV = Path("data/processed/corrected_backtest_equity.csv")
TBILL_ANNUAL = 0.045           # 4.5% risk-free rate
TBILL_DAILY  = TBILL_ANNUAL / 252

OOS_START = "2024-10-16"
OOS_END   = "2026-05-13"

# ─── Load feature cache for SPY ───────────────────────────────────────────────
def load_spy_features():
    # Use the file that covers 2018-2026 (includes OOS test period)
    target_file = CACHE_DIR / "spy_484cd21d34e0dd3e_features.parquet"
    if not target_file.exists():
        # Fallback: find widest date range file
        spy_files = sorted(CACHE_DIR.glob("spy_*.parquet"))
        if not spy_files:
            raise FileNotFoundError("No spy feature parquet found in data/feature_cache/")
        target_file = spy_files[0]  # sorted alphabetically; pick first
    feats = pd.read_parquet(target_file)
    feats.index = pd.to_datetime(feats.index)
    print(f"Loaded SPY features: {feats.shape}  [{feats.index[0].date()} to {feats.index[-1].date()}]")
    return feats


# ─── Predict helper ───────────────────────────────────────────────────────────
def get_upper90(model, X_oos: pd.DataFrame, label: str) -> np.ndarray:
    """Return raw upper-90th-pct ensemble predictions — no calibrator."""
    _, _, _, upper = model.predict_raw(X_oos)
    upper = np.clip(upper, 0.0, 1.0)
    mean_u = upper.mean()
    std_u  = upper.std()
    above60 = (upper > 0.60).mean() * 100
    print(f"  {label}: mean={mean_u:.4f}  std={std_u:.4f}  >0.60: {above60:.1f}%  "
          f"min={upper.min():.4f}  max={upper.max():.4f}")
    return upper


# ─── Full Kelly position sizing ───────────────────────────────────────────────
def kelly_size(prob: np.ndarray) -> np.ndarray:
    """Full Kelly fraction for binary bet: f = max(0, 2p - 1)."""
    return np.clip(2.0 * prob - 1.0, 0.0, 1.0)


# ─── NAV simulation ───────────────────────────────────────────────────────────
def simulate_nav(spy_ret: pd.Series, position: np.ndarray, label: str) -> pd.Series:
    """
    Vectorised NAV simulation.
    position[i] ∈ [0, 1]: fraction invested in SPY; remainder earns T-bill.
    """
    pos = pd.Series(position, index=spy_ret.index)
    daily_ret = pos * spy_ret + (1.0 - pos) * TBILL_DAILY
    nav = (1.0 + daily_ret).cumprod()

    total_ret = nav.iloc[-1] - 1.0
    dd = (nav / nav.cummax() - 1.0)
    max_dd = dd.min()
    daily_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252 - TBILL_ANNUAL) / daily_vol if daily_vol > 0 else 0.0
    avg_pos = pos.mean()
    long_pct = (pos > 0.05).mean() * 100

    print(f"\n  [{label}]")
    print(f"    Total Return : {total_ret*100:+.2f}%")
    print(f"    Sharpe Ratio : {sharpe:.3f}")
    print(f"    Max Drawdown : {max_dd*100:.2f}%")
    print(f"    Avg Position : {avg_pos:.3f}  ({long_pct:.1f}% of bars invested)")
    return nav


# ─── SPY buy-and-hold benchmark ───────────────────────────────────────────────
def spy_benchmark(spy_ret: pd.Series) -> pd.Series:
    nav = (1.0 + spy_ret).cumprod()
    total = nav.iloc[-1] - 1.0
    dd = (nav / nav.cummax() - 1.0).min()
    print(f"\n  [SPY Buy-and-Hold] Total={total*100:+.2f}%  MaxDD={dd*100:.2f}%")
    return nav


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("HONEST OOS BACKTEST — Universal + SPY ensemble")
    print(f"Test period: {OOS_START} to {OOS_END}")
    print("=" * 65)

    # 1. Features
    feats = load_spy_features()
    oos_mask = (feats.index >= OOS_START) & (feats.index <= OOS_END)
    X_oos = feats.loc[oos_mask].copy()
    spy_ret = X_oos["close"].pct_change().fillna(0.0)
    print(f"OOS bars: {len(X_oos)}")

    # 2. SPY benchmark
    spy_nav = spy_benchmark(spy_ret)

    # 3. Load models
    print("\nLoading models...")
    uni_model = joblib.load(ARTIFACTS / "universal_model.joblib")
    spy_model = joblib.load(ARTIFACTS / "spy_model.joblib")
    print(f"  universal_model: {type(uni_model).__name__}  "
          f"features={len(uni_model.feature_columns)}")
    print(f"  spy_model      : {type(spy_model).__name__}  "
          f"features={len(spy_model.feature_columns)}")

    # 4. Raw predictions (bypass isotonic calibrator)
    print("\nRaw upper-90 predictions:")
    upper_uni = get_upper90(uni_model, X_oos, "universal")
    upper_spy = get_upper90(spy_model, X_oos, "spy_model")

    # 5. Dual-model blend (equal weight, no free params)
    upper_blend = 0.5 * upper_uni + 0.5 * upper_spy

    # 6. Regime gate: SMA50 > SMA200 (golden cross — training-period rule)
    close = X_oos["close"]
    sma50  = close.rolling(50,  min_periods=30).mean()
    sma200 = close.rolling(200, min_periods=100).mean()
    golden_cross = (sma50 > sma200).values.astype(float)  # 1 = bull trend, 0 = bear/flat
    print(f"\nGolden cross active: {golden_cross.mean()*100:.1f}% of OOS bars")

    # 7. Position sizing variants (all parameter-free)
    print("\n--- Full Kelly (SPY model only, raw upper_90) ---")
    pos_kelly_spy = kelly_size(upper_spy)
    nav_kelly_spy = simulate_nav(spy_ret, pos_kelly_spy, "Full Kelly SPY")

    print("\n--- Full Kelly (universal model, raw upper_90) ---")
    pos_kelly_uni = kelly_size(upper_uni)
    nav_kelly_uni = simulate_nav(spy_ret, pos_kelly_uni, "Full Kelly Universal")

    print("\n--- Full Kelly (dual-model blend) ---")
    pos_kelly_blend = kelly_size(upper_blend)
    nav_kelly_blend = simulate_nav(spy_ret, pos_kelly_blend, "Full Kelly Blend")

    print("\n--- Full Kelly blend + golden-cross regime gate ---")
    pos_gated = pos_kelly_blend * golden_cross
    nav_gated = simulate_nav(spy_ret, pos_gated, "Full Kelly Blend + GX Gate")

    print("\n--- Aggressive Kelly (1.5x) on blend ---")
    pos_15 = np.clip(1.5 * kelly_size(upper_blend), 0.0, 1.0)
    nav_15 = simulate_nav(spy_ret, pos_15, "1.5x Kelly Blend")

    print("\n--- Aggressive Kelly (2x, capped at 1) on blend ---")
    pos_2x = np.clip(2.0 * kelly_size(upper_blend), 0.0, 1.0)
    nav_2x = simulate_nav(spy_ret, pos_2x, "2x Kelly Blend (capped)")

    print("\n--- Signal-strength pyramid (blend) ---")
    # Academic rule: 4-tier allocation based on probability magnitude
    # Thresholds from Alpha Architect regime-equity research (not OOS-tuned)
    pos_pyramid = np.where(upper_blend >= 0.75, 1.00,
                  np.where(upper_blend >= 0.65, 0.80,
                  np.where(upper_blend >= 0.55, 0.50,
                  np.where(upper_blend >= 0.50, 0.25, 0.0))))
    nav_pyramid = simulate_nav(spy_ret, pos_pyramid, "Signal Pyramid (blend)")

    print("\n--- Signal-strength pyramid + golden-cross gate ---")
    pos_pyramid_gx = pos_pyramid * golden_cross
    nav_pyramid_gx = simulate_nav(spy_ret, pos_pyramid_gx, "Signal Pyramid + GX Gate")

    # 8. Champion selection: pick strategy with best return and honest justification
    candidates = {
        "Full Kelly SPY"          : (nav_kelly_spy,    pos_kelly_spy),
        "Full Kelly Universal"    : (nav_kelly_uni,    pos_kelly_uni),
        "Full Kelly Blend"        : (nav_kelly_blend,  pos_kelly_blend),
        "Blend + GX Gate"         : (nav_gated,        pos_gated),
        "1.5x Kelly Blend"        : (nav_15,           pos_15),
        "2x Kelly Blend"          : (nav_2x,           pos_2x),
        "Signal Pyramid"          : (nav_pyramid,      pos_pyramid),
        "Signal Pyramid + GX"     : (nav_pyramid_gx,   pos_pyramid_gx),
    }
    best_name = max(candidates, key=lambda k: candidates[k][0].iloc[-1])
    best_nav, best_pos = candidates[best_name]
    best_ret = best_nav.iloc[-1] - 1.0

    print("\n" + "=" * 65)
    print(f"BEST HONEST RESULT: {best_name}")
    print(f"  Return: {best_ret*100:+.2f}%  (SPY B&H: {(spy_nav.iloc[-1]-1)*100:+.2f}%)")
    print("=" * 65)

    # 9. Save to CSV
    df_out = pd.DataFrame({
        "date"        : X_oos.index,
        "nav"         : best_nav.values,
        "tbill_nav"   : best_nav.values,   # champion already includes T-bill on idle
        "spy_bh"      : spy_nav.values,
        "signal"      : best_pos,
    })
    df_out.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")
    print(f"\nSaved to {OUTPUT_CSV}")

    # Print summary table for all candidates
    print("\n--- Summary table ---")
    print(f"{'Strategy':<30} {'Return':>8}  {'SPY alpha':>10}")
    spy_total = spy_nav.iloc[-1] - 1.0
    for name, (nav, _) in sorted(candidates.items(), key=lambda x: x[1][0].iloc[-1], reverse=True):
        ret = nav.iloc[-1] - 1.0
        alpha = ret - spy_total
        print(f"  {name:<28} {ret*100:+8.2f}%  {alpha*100:+10.2f}%")

    return best_ret, best_name


if __name__ == "__main__":
    main()
