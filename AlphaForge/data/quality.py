"""
data/quality.py
================
Data validation and cleaning pipeline for AlphaForge v2.0.

Pipeline steps (in order)
--------------------------
1. Detect outliers — IQR + z-score on OHLCV columns
2. Cap outliers    — winsorise at detection thresholds (never drop rows)
3. Impute missing  — ffill → bfill; linear interpolation for gaps ≤ 3 bars
4. Validate        — assert: no negative prices, no zero volume, no future dates
5. Version hash    — SHA-256[:16] of cleaned numeric values for data provenance

Usage
-----
    pipeline = DataQualityPipeline()
    clean_df, report = pipeline.run(df, ticker="SPY")
    print(report.version_hash)   # fingerprint for replay/rollback
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from utils.helpers import get_logger

logger = get_logger(__name__)

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]
_PRICE_COLS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    ticker:            str
    rows_total:        int
    rows_dropped:      int
    outliers_detected: int
    outliers_capped:   int          # winsorised, not dropped
    missing_filled:    int
    version_hash:      str          # SHA-256[:16] of clean numeric bytes
    issues:            list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ticker":            self.ticker,
            "rows_total":        self.rows_total,
            "rows_dropped":      self.rows_dropped,
            "outliers_detected": self.outliers_detected,
            "outliers_capped":   self.outliers_capped,
            "missing_filled":    self.missing_filled,
            "version_hash":      self.version_hash,
            "issues":            self.issues,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class DataQualityPipeline:
    """
    Validate, clean, and fingerprint OHLCV DataFrames.

    Parameters
    ----------
    zscore_threshold : float
        Rows where any OHLCV z-score exceeds this are flagged.
    iqr_multiplier : float
        Rows where any value is >{multiplier}×IQR from Q1/Q3 are flagged.
    max_missing_pct : float
        Warn (but don't reject) if missing fraction exceeds this.
    imputation : str
        "ffill" (default), "linear", or "median".
    """

    def __init__(
        self,
        zscore_threshold: float = 4.0,
        iqr_multiplier:   float = 3.0,
        max_missing_pct:  float = 0.05,
        imputation:       str   = "ffill",
    ) -> None:
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier   = iqr_multiplier
        self.max_missing_pct  = max_missing_pct
        self.imputation       = imputation

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        df: pd.DataFrame,
        ticker: str = "",
    ) -> tuple[pd.DataFrame, QualityReport]:
        """
        Clean *df* and return (clean_df, QualityReport).
        The input DataFrame is never modified in-place.
        """
        df = df.copy()
        rows_original = len(df)
        issues: list[str] = []

        # 1. Outlier detection + capping
        outlier_mask, n_detected = self._detect_outliers(df)
        if n_detected > 0:
            df, n_capped = self._cap_outliers(df, outlier_mask)
            issues.append(f"Capped {n_capped} outlier values in OHLCV columns")
        else:
            n_capped = 0

        # 2. Missing data imputation
        n_missing_before = int(df.isnull().sum().sum())
        df = self._impute_missing(df)
        n_missing_after  = int(df.isnull().sum().sum())
        n_filled = n_missing_before - n_missing_after
        if n_filled > 0:
            issues.append(f"Imputed {n_filled} missing values using '{self.imputation}'")

        # 3. Validation
        validation_issues = self._validate(df, ticker)
        issues.extend(validation_issues)

        # 4. Check missing fraction (warn only)
        for col in _OHLCV_COLS:
            if col in df.columns:
                miss_frac = df[col].isnull().mean()
                if miss_frac > self.max_missing_pct:
                    issues.append(
                        f"Column '{col}' still has {miss_frac:.1%} missing after imputation"
                    )

        rows_final  = len(df)
        version_hash = self._version_hash(df)

        report = QualityReport(
            ticker=ticker,
            rows_total=rows_original,
            rows_dropped=rows_original - rows_final,
            outliers_detected=n_detected,
            outliers_capped=n_capped,
            missing_filled=n_filled,
            version_hash=version_hash,
            issues=issues,
        )

        if issues:
            logger.info(
                "DataQualityPipeline [%s]: %d issues found; hash=%s",
                ticker or "?", len(issues), version_hash,
            )
        else:
            logger.debug(
                "DataQualityPipeline [%s]: clean — hash=%s", ticker or "?", version_hash
            )

        return df, report

    # ── Step implementations ──────────────────────────────────────────────────

    def _detect_outliers(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Return boolean mask (True = outlier cell) and total count."""
        cols = [c for c in _OHLCV_COLS if c in df.columns]
        if not cols:
            return pd.DataFrame(False, index=df.index, columns=[]), 0

        mask = pd.DataFrame(False, index=df.index, columns=cols)

        for col in cols:
            series = df[col].astype(float)
            # IQR method
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
                iqr_flag = (series < lower) | (series > upper)
            else:
                iqr_flag = pd.Series(False, index=df.index)

            # Z-score method
            mean, std = series.mean(), series.std()
            if std > 0:
                z_flag = ((series - mean) / std).abs() > self.zscore_threshold
            else:
                z_flag = pd.Series(False, index=df.index)

            mask[col] = iqr_flag | z_flag

        n_detected = int(mask.any(axis=1).sum())
        return mask, n_detected

    def _cap_outliers(
        self,
        df: pd.DataFrame,
        mask: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        """Winsorise flagged values at IQR bounds; return modified df and cap count."""
        n_capped = 0
        for col in mask.columns:
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.iqr_multiplier * iqr
            upper = q3 + self.iqr_multiplier * iqr
            flagged = mask[col]
            n_capped += int(flagged.sum())
            df[col] = series.clip(lower=lower, upper=upper)
        return df, n_capped

    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.imputation == "median":
            for col in _OHLCV_COLS:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
        elif self.imputation == "linear":
            for col in _OHLCV_COLS:
                if col in df.columns:
                    df[col] = df[col].interpolate(method="linear", limit=3)
            df = df.ffill().bfill()
        else:  # "ffill" (default)
            df = df.ffill().bfill()
        return df

    def _validate(self, df: pd.DataFrame, ticker: str, as_of_date=None) -> list[str]:
        issues: list[str] = []

        # Negative prices
        for col in _PRICE_COLS:
            if col in df.columns:
                neg = (df[col] < 0).sum()
                if neg > 0:
                    issues.append(f"{ticker}: {neg} negative values in '{col}' column")

        # Zero volume
        if "volume" in df.columns:
            zero_vol = (df["volume"] == 0).sum()
            if zero_vol > 0:
                issues.append(f"{ticker}: {zero_vol} zero-volume bars")

        # Future dates (compare against as_of_date when provided, else today)
        if hasattr(df.index, "max") and len(df) > 0:
            today = pd.Timestamp(as_of_date.date() if as_of_date is not None else datetime.now(timezone.utc).date())
            future = (df.index > today).sum()
            if future > 0:
                issues.append(f"{ticker}: {future} rows with future dates detected")

        # High < Low
        if "high" in df.columns and "low" in df.columns:
            inverted = (df["high"] < df["low"]).sum()
            if inverted > 0:
                issues.append(f"{ticker}: {inverted} bars where high < low")

        return issues

    def _version_hash(self, df: pd.DataFrame) -> str:
        numeric_df = df.select_dtypes(include="number")
        raw = numeric_df.values.astype(np.float64).tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]
