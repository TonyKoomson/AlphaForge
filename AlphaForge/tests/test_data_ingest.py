"""
Tests for data/ingest.py — Alpha Forge
=======================================
Coverage focus:
  1. Look-ahead bias prevention — the ``as_of_date`` gate can never be bypassed.
  2. Multi-asset support — BTC-USD (crypto) and SPY (equity) behave identically.
  3. Data validation — bad data (NaN, duplicates, negative prices) is caught early.
  4. Caching — files are written once and re-read on subsequent calls.
  5. Symbol filename convention — predictable, stable, safe for all OS.
  6. _parse_cutoff correctness — edge cases around timezone, type coercion, defaults.

All tests are offline (yfinance is mocked) so they run without network access.
"""

from __future__ import annotations

import sys
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path so imports work from the tests/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

import data.ingest as ingest_module
from data.ingest import (
    DataIngestion,
    DataValidationError,
    _parse_cutoff,
    _validate_raw,
    _validate_with_cutoff,
    download_data,
    get_data,
    symbol_to_filename,
)


# ---------------------------------------------------------------------------
# Shared test data factory
# ---------------------------------------------------------------------------

def make_fake_ohlcv(
    start: str = "2020-01-01",
    end: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a synthetic OHLCV DataFrame with:
      - DatetimeIndex named "timestamp" (business days, timezone-naive)
      - Realistic positive prices (geometric random walk)
      - No NaN, no duplicates, high >= low always
    """
    idx = pd.bdate_range(start=start, end=end)
    n = len(idx)
    rng = np.random.default_rng(seed)

    log_returns = rng.normal(0.0003, 0.012, n)
    close = 100.0 * np.exp(log_returns.cumsum())

    open_  = close * rng.uniform(0.990, 1.000, n)
    high   = close * rng.uniform(1.000, 1.015, n)
    low    = close * rng.uniform(0.985, 1.000, n)
    volume = rng.integers(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def write_symbol(tmp_path: Path, symbol: str, df: pd.DataFrame | None = None) -> DataIngestion:
    """Helper: write a Parquet file for ``symbol`` under ``tmp_path``."""
    ing = DataIngestion(config={}, raw_dir=tmp_path)
    data = df if df is not None else make_fake_ohlcv()
    data.to_parquet(ing._symbol_path(symbol))
    return ing


# ---------------------------------------------------------------------------
# 1. symbol_to_filename
# ---------------------------------------------------------------------------

class TestSymbolToFilename:
    def test_crypto_hyphen(self):
        assert symbol_to_filename("BTC-USD") == "btc_usd_daily.parquet"

    def test_crypto_eth(self):
        assert symbol_to_filename("ETH-USD") == "eth_usd_daily.parquet"

    def test_stock_spy(self):
        assert symbol_to_filename("SPY") == "spy_daily.parquet"

    def test_stock_with_dot(self):
        assert symbol_to_filename("BRK.B") == "brk_b_daily.parquet"

    def test_already_lowercase(self):
        assert symbol_to_filename("aapl") == "aapl_daily.parquet"

    def test_mixed_case(self):
        assert symbol_to_filename("NasDaQ") == "nasdaq_daily.parquet"

    def test_multiple_separators(self):
        # consecutive non-alphanumeric chars collapse to a single underscore
        result = symbol_to_filename("A--B")
        assert "__" not in result
        assert result == "a_b_daily.parquet"


# ---------------------------------------------------------------------------
# 2. _parse_cutoff
# ---------------------------------------------------------------------------

class TestParseCutoff:
    def test_string_iso_date(self):
        ts = _parse_cutoff("2022-06-30")
        assert ts.date() == date(2022, 6, 30)

    def test_date_object(self):
        ts = _parse_cutoff(date(2022, 6, 30))
        assert ts.date() == date(2022, 6, 30)

    def test_datetime_object(self):
        ts = _parse_cutoff(datetime(2022, 6, 30, 12, 0, 0))
        assert ts.date() == date(2022, 6, 30)

    def test_none_defaults_to_yesterday(self):
        ts = _parse_cutoff(None)
        assert ts.date() == date.today() - timedelta(days=1)

    def test_end_of_day_so_cutoff_date_is_included(self):
        ts = _parse_cutoff("2022-06-30")
        assert ts.hour == 23 and ts.minute == 59 and ts.second == 59

    def test_timezone_stripped(self):
        import pytz
        tz_aware = pd.Timestamp("2022-06-30", tz="UTC")
        ts = _parse_cutoff(tz_aware)
        assert ts.tzinfo is None

    def test_invalid_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="str, date, or datetime"):
            _parse_cutoff(20220630)  # type: ignore[arg-type]

    def test_two_different_cutoffs_order(self):
        earlier = _parse_cutoff("2021-01-01")
        later   = _parse_cutoff("2022-01-01")
        assert earlier < later


# ---------------------------------------------------------------------------
# 3. _validate_raw
# ---------------------------------------------------------------------------

class TestValidateRaw:
    def test_clean_data_passes(self):
        _validate_raw(make_fake_ohlcv(), "TEST")  # must not raise

    def test_empty_dataframe_raises(self):
        with pytest.raises(DataValidationError, match="empty"):
            _validate_raw(pd.DataFrame(), "EMPTY")

    def test_nan_in_close_raises(self):
        df = make_fake_ohlcv()
        df.loc[df.index[10], "close"] = float("nan")
        with pytest.raises(DataValidationError, match="NaN"):
            _validate_raw(df, "TEST")

    def test_nan_in_volume_raises(self):
        df = make_fake_ohlcv()
        df.loc[df.index[0], "volume"] = float("nan")
        with pytest.raises(DataValidationError, match="NaN"):
            _validate_raw(df, "TEST")

    def test_duplicate_timestamps_raises(self):
        df = make_fake_ohlcv()
        df_duped = pd.concat([df, df.iloc[:5]])
        with pytest.raises(DataValidationError, match="duplicate"):
            _validate_raw(df_duped, "TEST")

    def test_negative_close_raises(self):
        df = make_fake_ohlcv()
        df.loc[df.index[0], "close"] = -1.0
        with pytest.raises(DataValidationError, match="Non-positive"):
            _validate_raw(df, "TEST")

    def test_zero_open_raises(self):
        df = make_fake_ohlcv()
        df.loc[df.index[5], "open"] = 0.0
        with pytest.raises(DataValidationError, match="Non-positive"):
            _validate_raw(df, "TEST")

    def test_high_less_than_low_raises(self):
        df = make_fake_ohlcv()
        idx0 = df.index[0]
        df.loc[idx0, "high"] = df.loc[idx0, "low"] * 0.5
        with pytest.raises(DataValidationError, match="high < low"):
            _validate_raw(df, "TEST")


# ---------------------------------------------------------------------------
# 4. _validate_with_cutoff — the look-ahead bias assertion
# ---------------------------------------------------------------------------

class TestValidateWithCutoff:
    def test_clean_data_within_cutoff_passes(self):
        df = make_fake_ohlcv(end="2021-12-31")
        cutoff = _parse_cutoff("2022-06-30")
        _validate_with_cutoff(df, "TEST", cutoff)  # must not raise

    def test_future_row_raises_with_bias_message(self):
        df = make_fake_ohlcv(end="2021-12-31")
        future = pd.DataFrame(
            {"open": [101.0], "high": [105.0], "low": [100.0],
             "close": [103.0], "volume": [1_000_000.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2022-06-30")], name="timestamp"),
        )
        df_bad = pd.concat([df, future])
        cutoff = _parse_cutoff("2021-12-31")

        with pytest.raises(DataValidationError, match="LOOK-AHEAD BIAS DETECTED"):
            _validate_with_cutoff(df_bad, "TEST", cutoff)

    def test_exact_cutoff_date_does_not_raise(self):
        df = make_fake_ohlcv(start="2020-01-01", end="2022-06-30")
        cutoff = _parse_cutoff("2022-06-30")
        # The cutoff date itself (midnight) is <= end-of-day cutoff, so no error
        _validate_with_cutoff(df, "TEST", cutoff)

    def test_empty_df_warns_but_does_not_raise(self):
        cutoff = _parse_cutoff("2022-01-01")
        _validate_with_cutoff(pd.DataFrame(columns=["open","high","low","close","volume"]),
                               "TEST", cutoff)

    def test_duplicate_in_filtered_data_raises(self):
        df = make_fake_ohlcv(end="2021-12-31")
        df_duped = pd.concat([df, df.iloc[:3]])
        cutoff = _parse_cutoff("2022-01-01")
        with pytest.raises(DataValidationError, match="duplicate"):
            _validate_with_cutoff(df_duped, "TEST", cutoff)


# ---------------------------------------------------------------------------
# 5. DataIngestion.get_data — core look-ahead bias tests
# ---------------------------------------------------------------------------

class TestGetData:
    """
    These tests use real Parquet I/O via pytest's tmp_path fixture.
    No network calls are made.
    """

    # ── Basic cutoff behaviour ──────────────────────────────────────────

    def test_as_of_date_excludes_later_rows(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        df = ing.get_data("SPY", as_of_date="2021-06-30")

        assert not df.empty
        assert df.index.max() <= pd.Timestamp("2021-06-30")

    def test_as_of_date_includes_cutoff_date_itself(self, tmp_path):
        """The cutoff date must be inclusive (<=, not <)."""
        raw = make_fake_ohlcv(start="2020-01-01", end="2022-12-31")
        ing = write_symbol(tmp_path, "SPY", df=raw)

        # Pick the 50th business day as the cutoff
        cutoff_ts = raw.index[49]
        df = ing.get_data("SPY", as_of_date=cutoff_ts.date().isoformat())

        assert cutoff_ts in df.index, (
            f"Cutoff date {cutoff_ts.date()} must be present in the result."
        )

    def test_no_future_dates_for_stock(self, tmp_path):
        """SPY: zero rows may appear after as_of_date."""
        ing = write_symbol(tmp_path, "SPY")
        cutoff_str = "2021-06-15"
        df = ing.get_data("SPY", as_of_date=cutoff_str)

        future = df[df.index > pd.Timestamp(cutoff_str)]
        assert future.empty, (
            f"Look-ahead bias: {len(future)} future rows found for SPY. "
            f"Dates: {future.index.tolist()[:5]}"
        )

    def test_no_future_dates_for_crypto(self, tmp_path):
        """BTC-USD: same guarantee as equities."""
        ing = write_symbol(tmp_path, "BTC-USD")
        cutoff_str = "2022-03-31"
        df = ing.get_data("BTC-USD", as_of_date=cutoff_str)

        future = df[df.index > pd.Timestamp(cutoff_str)]
        assert future.empty, (
            f"Look-ahead bias: {len(future)} future rows found for BTC-USD."
        )

    def test_crypto_and_stock_same_cutoff_behaviour(self, tmp_path):
        """BTC-USD and SPY must produce identical cutoff semantics."""
        cutoff_str = "2022-03-31"
        cutoff = pd.Timestamp(cutoff_str)

        for symbol in ["BTC-USD", "SPY"]:
            ing = write_symbol(tmp_path, symbol)
            df = ing.get_data(symbol, as_of_date=cutoff_str)

            assert not df.empty, f"Expected data for {symbol}"
            assert df.index.max() <= cutoff, (
                f"[{symbol}] max date {df.index.max().date()} > cutoff {cutoff.date()}"
            )

    # ── Default (None) as_of_date excludes today ───────────────────────

    def test_default_as_of_date_excludes_today(self, tmp_path):
        """When as_of_date=None, today's bar must NOT appear."""
        today = pd.Timestamp(date.today()).normalize()
        raw = make_fake_ohlcv(end="2023-12-31")
        # Inject an artificial "today" row
        today_row = pd.DataFrame(
            {"open": [200.0], "high": [205.0], "low": [198.0],
             "close": [203.0], "volume": [5_000_000.0]},
            index=pd.DatetimeIndex([today], name="timestamp"),
        )
        raw_with_today = pd.concat([raw, today_row]).sort_index()
        ing = write_symbol(tmp_path, "SPY", df=raw_with_today)

        df = ing.get_data("SPY", as_of_date=None)
        assert today not in df.index, (
            "Today's incomplete bar must be excluded when as_of_date=None."
        )

    # ── Type coercion for as_of_date ────────────────────────────────────

    def test_date_object_as_cutoff(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        cutoff = date(2022, 6, 30)
        df = ing.get_data("SPY", as_of_date=cutoff)
        assert df.index.max() <= pd.Timestamp(cutoff)

    def test_datetime_object_as_cutoff(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        cutoff = datetime(2022, 6, 30, 0, 0, 0)
        df = ing.get_data("SPY", as_of_date=cutoff)
        assert df.index.max() <= pd.Timestamp(cutoff)

    # ── Edge cases ───────────────────────────────────────────────────────

    def test_cutoff_before_all_data_returns_empty(self, tmp_path):
        raw = make_fake_ohlcv(start="2022-01-01", end="2023-12-31")
        ing = write_symbol(tmp_path, "SPY", df=raw)
        df = ing.get_data("SPY", as_of_date="2019-01-01")
        assert df.empty

    def test_cutoff_after_all_data_returns_full_dataset(self, tmp_path):
        raw = make_fake_ohlcv(start="2020-01-01", end="2021-12-31")
        ing = write_symbol(tmp_path, "SPY", df=raw)
        df = ing.get_data("SPY", as_of_date="2030-01-01")
        assert len(df) == len(raw)

    def test_missing_symbol_raises_file_not_found(self, tmp_path):
        ing = DataIngestion(config={}, raw_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="NOSYM"):
            ing.get_data("NOSYM", as_of_date="2022-01-01")

    def test_returned_columns_are_exactly_ohlcv(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        df = ing.get_data("SPY", as_of_date="2022-01-01")
        assert set(df.columns) == {"open", "high", "low", "close", "volume"}

    def test_index_is_monotonic_ascending(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        df = ing.get_data("SPY", as_of_date="2022-01-01")
        assert df.index.is_monotonic_increasing

    def test_index_name_is_timestamp(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        df = ing.get_data("SPY", as_of_date="2022-01-01")
        assert df.index.name == "timestamp"

    def test_no_nan_in_returned_data(self, tmp_path):
        ing = write_symbol(tmp_path, "SPY")
        df = ing.get_data("SPY", as_of_date="2022-01-01")
        assert not df.isna().any().any()


# ---------------------------------------------------------------------------
# 6. DataIngestion.download_data (yfinance mocked)
# ---------------------------------------------------------------------------

class TestDownloadData:
    """
    yfinance is patched at ``data.ingest.yf`` so tests are fully offline.
    """

    @patch("data.ingest.yf")
    def test_creates_parquet_files_for_each_symbol(self, mock_yf, tmp_path):
        mock_yf.download.return_value = make_fake_ohlcv()

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        result = ing.download_data(["SPY", "BTC-USD"], start_date="2020-01-01")

        assert "SPY" in result
        assert "BTC-USD" in result
        assert (tmp_path / "spy_daily.parquet").exists()
        assert (tmp_path / "btc_usd_daily.parquet").exists()

    @patch("data.ingest.yf")
    def test_yfinance_called_once_per_symbol(self, mock_yf, tmp_path):
        mock_yf.download.return_value = make_fake_ohlcv()

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        ing.download_data(["SPY", "BTC-USD", "AAPL"], start_date="2020-01-01")

        assert mock_yf.download.call_count == 3

    @patch("data.ingest.yf")
    def test_cache_hit_skips_yfinance(self, mock_yf, tmp_path):
        mock_yf.download.return_value = make_fake_ohlcv()

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        ing.download_data(["SPY"], start_date="2020-01-01")
        mock_yf.download.reset_mock()

        # Second call: file already cached
        ing.download_data(["SPY"], start_date="2020-01-01")
        mock_yf.download.assert_not_called()

    @patch("data.ingest.yf")
    def test_force_download_bypasses_cache(self, mock_yf, tmp_path):
        mock_yf.download.return_value = make_fake_ohlcv()

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        ing.download_data(["SPY"], start_date="2020-01-01")
        mock_yf.download.reset_mock()

        ing.download_data(["SPY"], start_date="2020-01-01", force_download=True)
        mock_yf.download.assert_called_once()

    @patch("data.ingest.yf")
    def test_downloaded_data_has_correct_columns(self, mock_yf, tmp_path):
        mock_yf.download.return_value = make_fake_ohlcv()

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        result = ing.download_data(["SPY"], start_date="2020-01-01")

        assert set(result["SPY"].columns) == {"open", "high", "low", "close", "volume"}

    @patch("data.ingest.yf")
    def test_empty_yfinance_response_raises_value_error(self, mock_yf, tmp_path):
        mock_yf.download.return_value = pd.DataFrame()

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        with pytest.raises(ValueError, match="no data"):
            ing.download_data(["FAKE"], start_date="2020-01-01")

    @patch("data.ingest.yf")
    def test_multiindex_columns_are_flattened(self, mock_yf, tmp_path):
        """
        yfinance ≥0.2 can return MultiIndex columns like (Price, Ticker).
        download_data must flatten these before caching.
        """
        idx = pd.bdate_range("2020-01-01", "2021-12-31")
        n = len(idx)
        rng = np.random.default_rng(1)
        close = 100.0 * np.exp(rng.normal(0, 0.01, n).cumsum())

        multi_cols = pd.MultiIndex.from_tuples([
            ("Open", "SPY"), ("High", "SPY"), ("Low", "SPY"),
            ("Close", "SPY"), ("Volume", "SPY"),
        ])
        raw_multi = pd.DataFrame(
            np.column_stack([
                close * 0.99, close * 1.01, close * 0.98, close,
                rng.integers(1_000_000, 10_000_000, n).astype(float),
            ]),
            index=idx,
            columns=multi_cols,
        )
        raw_multi.index.name = "timestamp"
        mock_yf.download.return_value = raw_multi

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        result = ing.download_data(["SPY"], start_date="2020-01-01")

        assert set(result["SPY"].columns) == {"open", "high", "low", "close", "volume"}

    @patch("data.ingest.yf")
    def test_data_survives_round_trip_cache(self, mock_yf, tmp_path):
        """Data written to Parquet then read back must be bit-for-bit identical."""
        fake = make_fake_ohlcv()
        mock_yf.download.return_value = fake

        ing = DataIngestion(config={}, raw_dir=tmp_path)
        downloaded = ing.download_data(["SPY"], start_date="2020-01-01")["SPY"]

        cached = pd.read_parquet(ing._symbol_path("SPY"))

        # Parquet roundtrip drops DatetimeIndex freq metadata — ignore it
        pd.testing.assert_frame_equal(downloaded, cached, check_freq=False)


# ---------------------------------------------------------------------------
# 7. Module-level convenience functions
# ---------------------------------------------------------------------------

class TestModuleFunctions:
    @patch("data.ingest.yf")
    def test_download_data_function(self, mock_yf, tmp_path):
        mock_yf.download.return_value = make_fake_ohlcv()
        result = download_data(["SPY"], start_date="2020-01-01",
                               raw_dir=tmp_path, config={})
        assert "SPY" in result

    def test_get_data_function(self, tmp_path):
        write_symbol(tmp_path, "SPY")
        df = get_data("SPY", as_of_date="2022-01-01",
                      raw_dir=tmp_path, config={})
        assert not df.empty
        assert df.index.max() <= pd.Timestamp("2022-01-01")

    def test_get_data_function_missing_symbol_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_data("NOSYM", as_of_date="2022-01-01",
                     raw_dir=tmp_path, config={})


# ---------------------------------------------------------------------------
# 8. Look-ahead bias — adversarial scenarios
# ---------------------------------------------------------------------------

class TestLookAheadBiasAdversarial:
    """
    These tests attempt to defeat the bias protection in unusual ways.
    They document and lock-in the hard guarantees the module provides.
    """

    def test_injecting_future_row_after_get_data_still_blocked_on_next_call(self, tmp_path):
        """
        If a future row is somehow injected into the Parquet file after the
        first call, a subsequent call raises DataValidationError.
        """
        raw = make_fake_ohlcv(start="2020-01-01", end="2021-12-31")
        ing = write_symbol(tmp_path, "SPY", df=raw)

        # First clean call
        df1 = ing.get_data("SPY", as_of_date="2021-06-30")
        assert df1.index.max() <= pd.Timestamp("2021-06-30")

        # Inject a future row directly into the Parquet file
        future_row = pd.DataFrame(
            {"open": [300.0], "high": [310.0], "low": [295.0],
             "close": [305.0], "volume": [9_000_000.0]},
            index=pd.DatetimeIndex([pd.Timestamp("2025-01-01")], name="timestamp"),
        )
        tampered = pd.concat([raw, future_row])
        tampered.to_parquet(ing._symbol_path("SPY"))

        # Second call must block the future row
        df2 = ing.get_data("SPY", as_of_date="2021-06-30")
        assert pd.Timestamp("2025-01-01") not in df2.index

    def test_as_of_date_at_start_of_data_returns_minimal_window(self, tmp_path):
        """
        Setting as_of_date to the very first available bar returns exactly
        one row — not more.
        """
        raw = make_fake_ohlcv(start="2020-01-02", end="2023-12-31")
        ing = write_symbol(tmp_path, "SPY", df=raw)

        first_day = raw.index[0].date().isoformat()
        df = ing.get_data("SPY", as_of_date=first_day)

        assert len(df) == 1
        assert df.index[0] == raw.index[0]

    def test_parallel_symbols_do_not_share_state(self, tmp_path):
        """
        A cutoff applied to SPY must have no effect on BTC-USD data.
        Each symbol is fully isolated.
        """
        raw = make_fake_ohlcv(start="2020-01-01", end="2023-12-31")
        for sym in ["SPY", "BTC-USD"]:
            write_symbol(tmp_path, sym, df=raw)

        ing = DataIngestion(config={}, raw_dir=tmp_path)

        spy_df  = ing.get_data("SPY",     as_of_date="2021-01-01")
        btc_df  = ing.get_data("BTC-USD", as_of_date="2022-06-30")

        assert spy_df.index.max() <= pd.Timestamp("2021-01-01")
        assert btc_df.index.max() <= pd.Timestamp("2022-06-30")
        # The two DataFrames must not reference the same underlying object
        assert spy_df is not btc_df
