"""
Data Ingestion — Alpha Forge
============================
Downloads, caches, and serves daily OHLCV data for stocks and crypto.

Design principles
-----------------
- **Look-ahead bias is impossible by construction.**  The only public
  retrieval function, `get_data()`, accepts an `as_of_date` parameter and
  hard-blocks every row whose timestamp exceeds that date.  A post-filter
  assertion raises `DataValidationError` if any future row survives, making
  silent leakage impossible.
- **One Parquet file per symbol.**  `BTC-USD` → `data/raw/btc_usd_daily.parquet`,
  `SPY` → `data/raw/spy_daily.parquet`.  Files are stable and cacheable.
- **Source agnostic.**  yfinance handles equities and major crypto pairs
  (BTC-USD, ETH-USD, …).  CCXT is available for exchange-specific pairs when
  `source="ccxt"` is passed.

Quick start
-----------
    from data.ingest import download_data, get_data

    # 1. Download once and cache
    download_data(["SPY", "BTC-USD", "AAPL"], start_date="2018-01-01")

    # 2. Access with strict look-ahead bias protection
    df = get_data("SPY", as_of_date="2022-06-30")
    assert df.index.max() <= pd.Timestamp("2022-06-30")
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

from utils.helpers import ensure_dir, get_logger, load_config

logger = get_logger(__name__)

# Module-level import so it can be patched in tests with @patch("data.ingest.yf")
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # type: ignore[assignment]

OHLCV_COLS: List[str] = ["open", "high", "low", "close", "volume"]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DataValidationError(Exception):
    """Raised when data fails an integrity or look-ahead-bias check."""


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, easily unit-tested)
# ---------------------------------------------------------------------------

def symbol_to_filename(symbol: str) -> str:
    """
    Convert a ticker symbol to a safe Parquet filename.

    Examples
    --------
    >>> symbol_to_filename("BTC-USD")
    'btc_usd_daily.parquet'
    >>> symbol_to_filename("SPY")
    'spy_daily.parquet'
    >>> symbol_to_filename("BRK.B")
    'brk_b_daily.parquet'
    """
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", symbol).strip("_").lower()
    return f"{safe}_daily.parquet"


def _parse_cutoff(as_of_date: Optional[Union[str, date, datetime]]) -> pd.Timestamp:
    """
    Normalise ``as_of_date`` to a **timezone-naive** pd.Timestamp set to
    23:59:59 of that day, so that all bars whose date equals the cutoff
    are included in the result.

    When ``as_of_date`` is ``None`` the cutoff defaults to *yesterday*,
    preventing accidental ingestion of today's incomplete bar.
    """
    if as_of_date is None:
        cutoff = pd.Timestamp(date.today() - timedelta(days=1))
    elif isinstance(as_of_date, str):
        cutoff = pd.Timestamp(as_of_date)
    elif isinstance(as_of_date, (date, datetime)):
        cutoff = pd.Timestamp(as_of_date)
    else:
        raise TypeError(
            f"as_of_date must be str, date, or datetime — got {type(as_of_date).__name__!r}"
        )

    # Strip timezone so comparison with timezone-naive index works cleanly
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    # End-of-day so the cutoff date itself is fully included
    return cutoff.normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)


def _validate_raw(df: pd.DataFrame, symbol: str) -> None:
    """
    Validate freshly downloaded data before caching.

    Checks
    ------
    - DataFrame is not empty
    - No duplicate timestamps
    - No NaN in any OHLCV column
    - All OHLC prices are strictly positive
    - High >= Low on every row
    """
    if df.empty:
        raise DataValidationError(f"[{symbol}] DataFrame is empty after download.")

    dupes = df.index.duplicated()
    if dupes.any():
        raise DataValidationError(
            f"[{symbol}] {int(dupes.sum())} duplicate timestamps found in downloaded data."
        )

    nan_cols = [c for c in OHLCV_COLS if df[c].isna().any()]
    if nan_cols:
        raise DataValidationError(
            f"[{symbol}] NaN values found in columns: {nan_cols}"
        )

    for col in ["open", "high", "low", "close"]:
        if (df[col] <= 0).any():
            raise DataValidationError(
                f"[{symbol}] Non-positive price values found in column {col!r}."
            )

    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        raise DataValidationError(
            f"[{symbol}] Found {int(bad_hl.sum())} row(s) where high < low."
        )

    logger.debug("[%s] Raw validation passed (%d rows)", symbol, len(df))


def _validate_with_cutoff(
    df: pd.DataFrame, symbol: str, cutoff: pd.Timestamp
) -> None:
    """
    Validate data returned by ``get_data()``.

    The critical invariant: **zero rows may have a timestamp after ``cutoff``**.
    Any violation raises ``DataValidationError`` with the tag
    ``LOOK-AHEAD BIAS DETECTED`` so callers can never silently consume
    future data.
    """
    if df.empty:
        logger.warning("[%s] No data available as of %s", symbol, cutoff.date())
        return

    # *** The look-ahead bias assertion ***
    future_rows = df[df.index > cutoff]
    if not future_rows.empty:
        raise DataValidationError(
            f"[{symbol}] LOOK-AHEAD BIAS DETECTED — "
            f"{len(future_rows)} row(s) have timestamps after as_of_date={cutoff.date()}. "
            f"Latest offending timestamp: {future_rows.index.max().date()}"
        )

    dupes = df.index.duplicated()
    if dupes.any():
        raise DataValidationError(
            f"[{symbol}] {int(dupes.sum())} duplicate timestamps in filtered data."
        )

    nan_cols = [c for c in OHLCV_COLS if df[c].isna().any()]
    if nan_cols:
        raise DataValidationError(
            f"[{symbol}] NaN values found in columns after cutoff filter: {nan_cols}"
        )

    logger.debug(
        "[%s] Cutoff validation passed: %d rows, latest=%s, cutoff=%s",
        symbol,
        len(df),
        df.index.max().date(),
        cutoff.date(),
    )


# ---------------------------------------------------------------------------
# DataIngestion class
# ---------------------------------------------------------------------------

class DataIngestion:
    """
    Downloads, caches, and retrieves daily OHLCV data for stocks and crypto.

    Parameters
    ----------
    config : optional config dict (loaded from config.yaml when omitted)
    raw_dir : directory where per-symbol Parquet files are stored
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        raw_dir: Union[str, Path] = "data/raw",
    ) -> None:
        self.cfg = config if config is not None else load_config()
        self.raw_dir = ensure_dir(raw_dir)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(
        self,
        symbols: List[str] = ["BTC-USD", "SPY"],
        start_date: str = "2018-01-01",
        end_date: Optional[str] = None,
        force_download: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download daily OHLCV for one or more symbols and cache each as a
        Parquet file in ``raw_dir/``.

        Parameters
        ----------
        symbols       : list of ticker strings — equities, ETFs, or crypto pairs
                        (e.g. ``["SPY", "BTC-USD", "ETH-USD", "AAPL"]``)
        start_date    : history start, ISO format ``"YYYY-MM-DD"``
        end_date      : history end (defaults to today)
        force_download: re-download even when a cached file already exists

        Returns
        -------
        dict mapping symbol → cleaned DataFrame (DatetimeIndex named "timestamp",
        columns: open high low close volume, sorted ascending)

        Examples
        --------
        >>> ing = DataIngestion()
        >>> data = ing.download_data(["SPY", "BTC-USD"], start_date="2020-01-01")
        >>> data["SPY"].head()
        """
        if end_date is None:
            end_date = date.today().isoformat()

        results: Dict[str, pd.DataFrame] = {}

        for symbol in symbols:
            path = self._symbol_path(symbol)

            if path.exists() and not force_download:
                logger.info("[%s] Cache hit — loading %s", symbol, path.name)
                results[symbol] = pd.read_parquet(path)
                continue

            logger.info("[%s] Downloading %s → %s via %s", symbol, start_date, end_date, self._source())
            try:
                if self._source() == "ccxt":
                    raw = self._fetch_ccxt(symbol, start_date, end_date)
                else:
                    raw = self._fetch_yfinance(symbol, start_date, end_date)

                df = self._normalize(raw, symbol)
                _validate_raw(df, symbol)
                df.to_parquet(path)
                logger.info("[%s] Saved %d rows → %s", symbol, len(df), path)
                results[symbol] = df

            except Exception:
                logger.exception("[%s] Download failed", symbol)
                raise

        return results

    # ------------------------------------------------------------------
    # Retrieval with look-ahead bias protection
    # ------------------------------------------------------------------

    def get_data(
        self,
        symbol: str,
        as_of_date: Optional[Union[str, date, datetime]] = None,
    ) -> pd.DataFrame:
        """
        Return OHLCV data for ``symbol`` **up to and including** ``as_of_date``.

        This is the **only** correct way to access data during model training,
        feature engineering, and backtesting.  The cutoff is enforced by a
        two-step mechanism:

        1. A ``df[df.index <= cutoff]`` filter drops all future rows.
        2. A post-filter assertion raises ``DataValidationError`` if any future
           row still exists — making silent look-ahead bias impossible.

        Parameters
        ----------
        symbol      : ticker string, e.g. ``"SPY"`` or ``"BTC-USD"``
        as_of_date  : rows with timestamp > as_of_date are unconditionally
                      excluded.  Defaults to *yesterday* when ``None`` to
                      prevent accidental use of today's incomplete bar.

        Returns
        -------
        pd.DataFrame with DatetimeIndex named ``"timestamp"`` (sorted ascending),
        columns: open  high  low  close  volume

        Raises
        ------
        FileNotFoundError   : symbol has not been downloaded yet
        DataValidationError : a future row survived the filter (should never
                              happen under normal operation)

        Examples
        --------
        >>> df = ing.get_data("SPY", as_of_date="2022-06-30")
        >>> assert df.index.max() <= pd.Timestamp("2022-06-30")
        >>> assert "2022-07-01" not in df.index.astype(str)
        """
        path = self._symbol_path(symbol)
        if not path.exists():
            raise FileNotFoundError(
                f"No cached data for {symbol!r}.  "
                f"Run download_data([{symbol!r}]) first."
            )

        df = pd.read_parquet(path)

        cutoff = _parse_cutoff(as_of_date)

        # Step 1 — filter
        df = df[df.index <= cutoff].copy()

        # Step 2 — assert (raises on any surviving future row)
        _validate_with_cutoff(df, symbol, cutoff)

        logger.debug("[%s] Loaded %d rows as of %s", symbol, len(df), cutoff.date())
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _symbol_path(self, symbol: str) -> Path:
        return self.raw_dir / symbol_to_filename(symbol)

    def _source(self) -> str:
        return self.cfg.get("data", {}).get("source", "yfinance")

    def _fetch_yfinance(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        if yf is None:  # pragma: no cover
            raise ImportError("yfinance is not installed.  Run: pip install yfinance")

        raw = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )

        if raw is None or (hasattr(raw, "empty") and raw.empty):
            raise ValueError(
                f"yfinance returned no data for {symbol!r} [{start} → {end}]"
            )

        # Newer yfinance versions (≥0.2.x) may return a MultiIndex when
        # group_by='ticker' is implied — flatten to simple column names.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [str(col[0]).lower() for col in raw.columns]
        else:
            raw.columns = [str(c).lower() for c in raw.columns]

        raw.index = pd.to_datetime(raw.index)
        raw.index.name = "timestamp"
        return raw

    def _fetch_ccxt(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ccxt is not installed.  Run: pip install ccxt") from exc

        exchange_id = self.cfg.get("data", {}).get("ccxt_exchange", "binance")
        exchange = getattr(ccxt, exchange_id)()

        since_ms = int(pd.Timestamp(start).timestamp() * 1000)
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)

        all_bars: list = []
        while True:
            bars = exchange.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=1000)
            if not bars:
                break
            all_bars.extend(bars)
            last_ms = bars[-1][0]
            if last_ms >= end_ms:
                break
            since_ms = last_ms + 86_400_000  # advance one day

        if not all_bars:
            raise ValueError(f"ccxt returned no data for {symbol!r} [{start} → {end}]")

        df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        df = df[df.index <= pd.Timestamp(end)]
        return df

    def _normalize(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardise column names, strip timezone, sort ascending, cast to float64."""
        # Rename common aliases produced by different yfinance versions
        rename_map = {"adj close": "close", "adj_close": "close"}
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        missing = [c for c in OHLCV_COLS if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"[{symbol}] Missing columns after normalisation: {missing}. "
                f"Available: {list(df.columns)}"
            )

        df = df[OHLCV_COLS].copy()

        # Strip timezone so all comparisons are timezone-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        df.index = pd.to_datetime(df.index)
        df.index.name = "timestamp"
        df = df.sort_index()

        for col in OHLCV_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def download_data(
    symbols: List[str] = ["BTC-USD", "SPY"],
    start_date: str = "2018-01-01",
    end_date: Optional[str] = None,
    force_download: bool = False,
    raw_dir: Union[str, Path] = "data/raw",
    config: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Download and cache daily OHLCV data for multiple symbols.

    Thin wrapper around :class:`DataIngestion`.

    Examples
    --------
    >>> from data.ingest import download_data
    >>> data = download_data(["SPY", "BTC-USD", "ETH-USD"], start_date="2018-01-01")
    >>> print(data["SPY"].shape)
    """
    return DataIngestion(config=config, raw_dir=raw_dir).download_data(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        force_download=force_download,
    )


def get_data(
    symbol: str,
    as_of_date: Optional[Union[str, date, datetime]] = None,
    raw_dir: Union[str, Path] = "data/raw",
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Return OHLCV data for ``symbol`` strictly limited to rows on or before
    ``as_of_date``.

    **This is the only safe entry-point for data access during model
    training, feature engineering, and backtesting.**

    Examples
    --------
    >>> from data.ingest import get_data
    >>> df = get_data("SPY", as_of_date="2022-06-30")
    >>> assert df.index.max() <= pd.Timestamp("2022-06-30")
    """
    return DataIngestion(config=config, raw_dir=raw_dir).get_data(
        symbol=symbol,
        as_of_date=as_of_date,
    )
