"""Shared utilities: logging, config loading, date helpers, and common metrics."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def to_datetime(value: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value)


def date_range(start: str, end: str, freq: str = "B") -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq=freq)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Financial metrics
# ---------------------------------------------------------------------------

def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty:
        return 0.0
    total = (1 + returns).prod()
    n = len(returns)
    return float(total ** (periods_per_year / n) - 1)


def annualised_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    vol = annualised_volatility(excess, periods_per_year)
    if vol == 0:
        return 0.0
    return float(annualised_return(excess, periods_per_year) / vol)


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - risk_free / periods_per_year
    downside = excess[excess < 0]
    downside_vol = float(downside.std() * np.sqrt(periods_per_year)) if not downside.empty else 0.0
    if downside_vol == 0:
        return 0.0
    return float(annualised_return(excess, periods_per_year) / downside_vol)


def max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def max_drawdown_duration(equity_curve: pd.Series) -> int:
    rolling_max = equity_curve.cummax()
    underwater = equity_curve < rolling_max
    durations = []
    streak = 0
    for flag in underwater:
        if flag:
            streak += 1
        else:
            if streak:
                durations.append(streak)
            streak = 0
    if streak:
        durations.append(streak)
    return max(durations, default=0)


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return float(annualised_return(returns, periods_per_year) / mdd)


def win_rate(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float("inf")
    return float(gross_profit / gross_loss)


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity_curve) < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_years = len(equity_curve) / periods_per_year
    if n_years <= 0:
        return 0.0
    return float((1 + total_return) ** (1 / n_years) - 1)


def compute_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    return {
        "cagr": cagr(equity_curve, periods_per_year),
        "annualised_return": annualised_return(returns, periods_per_year),
        "annualised_volatility": annualised_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=periods_per_year),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=periods_per_year),
        "max_drawdown": max_drawdown(equity_curve),
        "max_drawdown_duration_bars": max_drawdown_duration(equity_curve),
        "calmar_ratio": calmar_ratio(returns, equity_curve, periods_per_year),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
        "total_trades": int((returns != 0).sum()),
    }
