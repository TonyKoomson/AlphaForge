"""
Alpha Forge — Feature Matrix Cache
====================================
Caches the computed feature DataFrame to disk so re-training the same ticker
does not recompute all rolling indicators from scratch.

How it works
------------
1. Hash the raw OHLCV Parquet file (last-modified time + row count + date range).
2. If a Parquet cache file exists with the same hash, load it in ~0.1 seconds.
3. Otherwise call generate_features(), save the result, return it.

Impact
------
Without cache: every `main.py train` call spends 60–90 seconds on feature
engineering before a single model is fitted.

With cache: first run computes and saves. Every subsequent run for the same
ticker (same raw data) loads from cache in < 1 second.  Hyperparameter sweeps,
multi-seed campaigns, and parallel training all skip feature recomputation.

Usage
-----
    from features.feature_cache import get_or_compute_features

    features_df = get_or_compute_features(
        ticker="AAPL",
        raw_path=Path("data/raw/aapl_daily.parquet"),
        cross_asset=ca_data,   # optional pre-loaded cross-asset dict
    )
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.helpers import ensure_dir, get_logger

logger = get_logger(__name__)

ROOT        = Path(__file__).parent.parent
_CACHE_DIR  = ROOT / "data" / "feature_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Bump this version string whenever engine.py logic changes to invalidate stale caches.
_ENGINE_VERSION = "v7"  # restored: horizon=5, max_depth=3 sweep (2026-05-16)


def _data_fingerprint(raw_path: Path) -> str:
    """
    Fast fingerprint of a raw Parquet file + engine version.
    Uses: file size + mtime + first/last date + engine version tag.
    Avoids reading the whole file — just stat() + minimal Parquet metadata.
    Incrementing _ENGINE_VERSION invalidates all caches on the next run.
    """
    try:
        st = os.stat(raw_path)
        # Read just the index (date column) — very fast via Parquet
        idx = pd.read_parquet(raw_path, columns=[]).index
        first = str(idx.min().date()) if len(idx) else "none"
        last  = str(idx.max().date()) if len(idx) else "none"
        raw = f"{raw_path.name}_{st.st_size}_{st.st_mtime:.0f}_{first}_{last}_{_ENGINE_VERSION}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]
    except Exception:
        # Fallback: just use mtime + version
        try:
            st = os.stat(raw_path)
            raw = f"{raw_path.name}_{st.st_size}_{st.st_mtime:.0f}_{_ENGINE_VERSION}"
            return hashlib.md5(raw.encode()).hexdigest()[:16]
        except Exception:
            return "no_fingerprint"


def _cache_path(ticker: str, fingerprint: str) -> Path:
    return _CACHE_DIR / f"{ticker.lower()}_{fingerprint}_features.parquet"


def get_or_compute_features(
    ticker: str,
    raw_path: Path,
    cross_asset: Optional[dict] = None,
    config: Optional[dict] = None,
    as_of_date: Optional[str] = None,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """
    Return the feature DataFrame for a ticker, using disk cache when available.

    Parameters
    ----------
    ticker         : e.g. "AAPL"
    raw_path       : Path to the raw OHLCV Parquet file
    cross_asset    : Pre-loaded cross-asset data dict (from CrossAssetCache)
    config         : Loaded config dict (used to pass to generate_features)
    as_of_date     : Hard date cutoff for anti-leakage
    force_recompute: If True, ignore cache and recompute

    Returns
    -------
    pd.DataFrame with all feature columns + label/fwd_return columns
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")

    fingerprint = _data_fingerprint(raw_path)
    cached = _cache_path(ticker, fingerprint)

    if cached.exists() and not force_recompute:
        logger.debug("Feature cache HIT for %s (%s)", ticker, fingerprint)
        try:
            df = pd.read_parquet(cached)
            logger.info("Loaded features from cache: %s (%d rows)", ticker, len(df))
            return df
        except Exception as exc:
            logger.warning("Cache read failed for %s: %s — recomputing", ticker, exc)

    # Cache miss — compute features
    logger.info("Feature cache MISS for %s — computing...", ticker)
    import time
    t0 = time.time()

    from data.ingest import DataIngestion
    from features.engine import generate_features

    cfg = config or {}
    ingester = DataIngestion(config=cfg)
    raw_df = pd.read_parquet(raw_path)

    # Normalise columns
    raw_df.columns = [c.lower() for c in raw_df.columns]
    if hasattr(raw_df.index, "tz") and raw_df.index.tz is not None:
        raw_df.index = raw_df.index.tz_localize(None)

    feat_df = generate_features(
        raw_df,
        as_of_date=as_of_date,
        cross_asset=cross_asset,
    )

    elapsed = time.time() - t0
    logger.info("Feature computation done for %s: %d rows in %.1fs", ticker, len(feat_df), elapsed)

    # Save to cache
    try:
        feat_df.to_parquet(cached)
        logger.debug("Feature cache saved: %s", cached)
    except Exception as exc:
        logger.warning("Cache save failed for %s: %s", ticker, exc)

    return feat_df


def invalidate_cache(ticker: str) -> int:
    """Remove all cached feature files for a ticker. Returns count deleted."""
    pattern = f"{ticker.lower()}_*_features.parquet"
    deleted = 0
    for f in _CACHE_DIR.glob(pattern):
        f.unlink()
        deleted += 1
    return deleted


def cache_stats() -> dict:
    """Return summary stats about the feature cache."""
    files = list(_CACHE_DIR.glob("*_features.parquet"))
    total_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)
    return {
        "n_files":  len(files),
        "total_mb": round(total_mb, 1),
        "cache_dir": str(_CACHE_DIR),
    }
