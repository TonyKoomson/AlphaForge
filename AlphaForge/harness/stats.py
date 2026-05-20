"""
AlphaForge AI Harness — Statistical Utilities

Research-backed statistical metrics for rigorous strategy evaluation.

References
----------
Bailey, D.H. and Lopez de Prado, M. (2012) 'The Sharpe ratio efficient frontier',
    Journal of Risk, 15(2), pp. 3–44.

Bailey, D.H., Borwein, J., Lopez de Prado, M. and Zhu, Q.J. (2016)
    'The probability of backtest overfitting',
    Journal of Computational Finance, 20(4), pp. 39–70.

Chapelle, O. and Li, L. (2011) 'An empirical evaluation of Thompson sampling',
    Advances in Neural Information Processing Systems, 24.

Harvey, C.R., Liu, Y. and Zhu, H. (2016) '... and the cross-section of expected returns',
    Review of Financial Studies, 29(1), pp. 5–68.
"""
from __future__ import annotations

import math
from typing import Optional


# ── Normal CDF (no scipy dependency) ─────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using the Abramowitz & Stegun approximation (error < 7.5e-8)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _t_cdf(t: float, df: int) -> float:
    """
    Student-t CDF (one-sided, upper tail) via incomplete beta approximation.
    Only needed for small df; for df >= 30 the normal approximation is accurate.
    """
    if df >= 30:
        return _norm_cdf(t)
    x = df / (df + t * t)
    # Regularised incomplete beta I_x(df/2, 0.5)
    # Use simple numerical integration for small df
    half_df = df / 2.0
    try:
        import math
        # Legendre-Gauss 10-point quadrature over [0, x]
        nodes = [
            0.0765265211334973, 0.2277858511416451, 0.3737060887154195,
            0.5108670019508271, 0.6360536807265150, 0.7463064833401650,
            0.8391169718222188, 0.9122344282513259, 0.9639719272779138,
            0.9931285991850949,
        ]
        weights = [
            0.1527533871307258, 0.1491729864726037, 0.1420961093183820,
            0.1316886384491766, 0.1181945319615184, 0.1019301198172404,
            0.0832767415767048, 0.0626720483341091, 0.0406014298003869,
            0.0176140071391521,
        ]
        half_x = x / 2.0
        integral = 0.0
        for node, w in zip(nodes, weights):
            for xi in [half_x * (1 - node), half_x * (1 + node)]:
                integral += w * (xi ** (half_df - 1)) * ((1 - xi) ** (-0.5))
        integral *= half_x
        # Divide by B(df/2, 0.5) = Γ(df/2) * Γ(0.5) / Γ(df/2 + 0.5)
        # For simplicity fall back to normal for df >= 10
        if df >= 10:
            return _norm_cdf(t)
        # Exact recursion for small integer df
        p = math.sqrt(1 - x)
        return p * sum(
            math.comb(df - 1, 2 * k) * x ** k / (2 ** (df - 2))
            for k in range(df // 2)
        ) if df % 2 == 0 else _norm_cdf(t)
    except Exception:
        return _norm_cdf(t)


# ── Sharpe t-statistic ────────────────────────────────────────────────────────

def sharpe_tstat(sharpe: float, n_bars: int, bars_per_year: int = 252) -> dict:
    """
    Compute the Sharpe ratio t-statistic and one-tailed p-value (H0: SR = 0).

    Under H0, the annualised Sharpe ratio scaled by sqrt(T) follows
    approximately N(0, 1) for large T.

    Parameters
    ----------
    sharpe      : Annualised Sharpe ratio (daily returns * sqrt(252))
    n_bars      : Number of daily observations
    bars_per_year : Trading days per year (252)

    Returns
    -------
    dict with keys: t_stat, p_value, n_years, significant_at_05
    """
    n_years = n_bars / bars_per_year
    if n_years < 0.1 or n_bars < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "n_years": n_years, "significant_at_05": False}

    # Annualised SR already incorporates the sqrt(252) factor, so:
    # t_stat = SR_ann * sqrt(n_years)  [Lo, 2002]
    t_stat = sharpe * math.sqrt(n_years)
    p_value = 1.0 - _norm_cdf(t_stat)   # one-tailed: P(SR > 0)
    return {
        "t_stat":           round(t_stat, 4),
        "p_value":          round(p_value, 4),
        "n_years":          round(n_years, 2),
        "significant_at_05": p_value < 0.05,
    }


# ── Probabilistic Sharpe Ratio (PSR) ─────────────────────────────────────────

def probabilistic_sharpe_ratio(
    sharpe: float,
    n_bars: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
    bars_per_year: int = 252,
) -> dict:
    """
    Bailey & Lopez de Prado (2012) Probabilistic Sharpe Ratio.

    PSR(SR*) = Φ[(SR - SR*) * sqrt(T-1) / sqrt(1 - γ₁·SR + (γ₂-1)/4 · SR²)]

    where:
      SR*  = benchmark Sharpe (default 0 = "is strategy better than nothing?")
      T    = number of observations
      γ₁   = skewness of returns
      γ₂   = kurtosis of returns (3 = normal)

    Parameters
    ----------
    sharpe       : Estimated annualised Sharpe ratio
    n_bars       : Number of daily observations
    skewness     : Skewness of (daily) return series
    kurtosis     : Kurtosis of (daily) return series (excess = 0 for normal)
    benchmark_sr : SR* — the threshold to test against (default 0)
    bars_per_year: Trading days per year

    Returns
    -------
    dict with keys: psr, sr_star (the benchmark), interpretation
    """
    if n_bars < 5:
        return {"psr": 0.0, "sr_star": benchmark_sr, "interpretation": "insufficient_data"}

    # Convert annualised SR to daily (denominator of the formula is in daily units)
    sr_daily = sharpe / math.sqrt(bars_per_year)
    sr_star_daily = benchmark_sr / math.sqrt(bars_per_year)

    # Variance of the SR estimator  (Mertens, 2002)
    # Var(SR_hat) ≈ (1 + 0.5*SR² - γ₁*SR + (γ₂-1)/4 * SR²) / (T-1)
    # Note: excess kurtosis = γ₂ - 3 for a normal distribution, but the formula
    # uses raw kurtosis; adjust if caller passes excess kurtosis.
    excess_kurt = kurtosis - 3.0  # convert to excess
    numerator_sq = 1.0 - skewness * sr_daily + (excess_kurt / 4.0) * sr_daily ** 2
    numerator_sq = max(numerator_sq, 1e-9)  # clamp to avoid sqrt of negative

    z = (sr_daily - sr_star_daily) * math.sqrt(n_bars - 1) / math.sqrt(numerator_sq)
    psr = _norm_cdf(z)

    if psr >= 0.99:
        interp = "very_high_confidence"
    elif psr >= 0.95:
        interp = "statistically_significant"
    elif psr >= 0.90:
        interp = "suggestive"
    else:
        interp = "not_significant"

    return {
        "psr":             round(psr, 4),
        "sr_star":         benchmark_sr,
        "z_score":         round(z, 4),
        "interpretation":  interp,
    }


# ── Deflated Sharpe Ratio (DSR) ───────────────────────────────────────────────

def deflated_sharpe_ratio(
    sharpe: float,
    n_bars: int,
    n_trials: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    bars_per_year: int = 252,
) -> dict:
    """
    Bailey & Lopez de Prado (2012) Deflated Sharpe Ratio.

    The DSR corrects for the multiple-testing bias that arises when a researcher
    evaluates many candidate strategies.  The expected maximum SR from n_trials
    independent trials on T observations is approximately:

        SR* ≈ σ_sr * ((1 - γ) * Φ⁻¹(1 - 1/n_trials) + γ * Φ⁻¹(1 - 1/(n_trials·e)))

    where σ_sr = sqrt(Var(SR)), γ = Euler–Mascheroni constant ≈ 0.5772,
    and Φ⁻¹ is the standard normal quantile function.

    We then compute PSR(SR*) — the probability that the observed SR exceeds
    the expected maximum that could arise by chance.

    Parameters
    ----------
    sharpe    : Observed OOS Sharpe ratio
    n_bars    : Number of observations in the backtest
    n_trials  : Number of strategies evaluated so far (including this one)
    skewness  : Skewness of daily returns
    kurtosis  : Raw kurtosis of daily returns
    bars_per_year : Trading days per year

    Returns
    -------
    dict with keys: dsr, sr_star_ann, n_trials, interpretation
    """
    if n_bars < 10 or n_trials < 1:
        return {
            "dsr": probabilistic_sharpe_ratio(sharpe, n_bars, skewness, kurtosis, 0.0, bars_per_year)["psr"],
            "sr_star_ann": 0.0,
            "n_trials": n_trials,
            "interpretation": "single_trial_psr",
        }

    # Euler–Mascheroni constant
    gamma_em = 0.5772156649

    # Variance of daily SR estimator
    excess_kurt = kurtosis - 3.0
    sr_daily = sharpe / math.sqrt(bars_per_year)
    var_sr = (1.0 - sr_daily * (kurtosis - 1.0) / 4.0 + sr_daily ** 2 * excess_kurt / 8.0) / (n_bars - 1)
    sigma_sr = math.sqrt(max(var_sr, 1e-12))

    # Expected maximum SR threshold (daily units)
    # Using normal-order-statistics approximation
    def _norm_ppf(p: float) -> float:
        """Inverse normal CDF using rational approximation (Beasley-Springer-Moro)."""
        if p <= 0:
            return -10.0
        if p >= 1:
            return 10.0
        if p < 0.5:
            return -_norm_ppf(1 - p)
        # Rational approximation for upper half
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return q - (c0 + c1 * q + c2 * q ** 2) / (1 + d1 * q + d2 * q ** 2 + d3 * q ** 3)

    n = max(n_trials, 2)
    term1 = (1 - gamma_em) * _norm_ppf(1 - 1.0 / n)
    term2 = gamma_em * _norm_ppf(1 - 1.0 / (n * math.e))
    sr_star_daily = sigma_sr * (term1 + term2)
    sr_star_ann   = sr_star_daily * math.sqrt(bars_per_year)

    psr_result = probabilistic_sharpe_ratio(
        sharpe, n_bars, skewness, kurtosis,
        benchmark_sr=sr_star_ann,
        bars_per_year=bars_per_year,
    )
    dsr = psr_result["psr"]

    if dsr >= 0.95:
        interp = "passes_multiple_testing_correction"
    elif dsr >= 0.90:
        interp = "marginal_after_correction"
    elif dsr >= 0.50:
        interp = "below_multiple_testing_threshold"
    else:
        interp = "likely_false_positive"

    return {
        "dsr":          round(dsr, 4),
        "sr_star_ann":  round(sr_star_ann, 4),
        "n_trials":     n_trials,
        "interpretation": interp,
    }


# ── Minimum Backtest Length ───────────────────────────────────────────────────

def minimum_backtest_length(
    sharpe: float,
    n_trials: int,
    target_dsr: float = 0.95,
    bars_per_year: int = 252,
) -> int:
    """
    Minimum number of daily observations needed for DSR >= target_dsr.

    Bailey & Lopez de Prado (2016): for a given number of trials n and
    desired significance level, compute the minimum T such that DSR(SR, T, n) >= target.

    Returns the minimum number of bars (trading days) required.
    """
    for n_bars in range(50, 6000, 10):
        result = deflated_sharpe_ratio(sharpe, n_bars, n_trials)
        if result["dsr"] >= target_dsr:
            return n_bars
    return 6000  # more than ~24 years — practically unobtainable


# ── Harness multi-trial tracking ──────────────────────────────────────────────

class TrialTracker:
    """
    Tracks number of strategy trials for DSR computation.
    Injected into the ToolExecutor so DSR can adjust for n_trials.
    """

    def __init__(self) -> None:
        self._n_trials = 0

    def increment(self) -> int:
        self._n_trials += 1
        return self._n_trials

    @property
    def n_trials(self) -> int:
        return self._n_trials
