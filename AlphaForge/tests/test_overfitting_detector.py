"""Tests for validation/overfitting_detector.py."""

from __future__ import annotations

import math

import pytest

from validation.overfitting_detector import (
    OverfittingReport,
    _compute_deflated_sharpe,
    _compute_feature_stability,
    _compute_gap_score,
    _compute_min_backtest_length,
    _compute_multiple_testing_penalty,
    _compute_pbo,
    detect_overfitting,
    format_report_section,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _clean_is():
    return {"sharpe_ratio": 1.5, "cagr": 0.20, "win_rate": 0.55, "max_drawdown": -0.08}

def _clean_oos():
    return {"sharpe_ratio": 1.3, "cagr": 0.17, "win_rate": 0.53, "max_drawdown": -0.10}

def _overfit_is():
    return {"sharpe_ratio": 2.5, "cagr": 0.40, "win_rate": 0.65, "max_drawdown": -0.04}

def _overfit_oos():
    return {"sharpe_ratio": 0.2, "cagr": 0.02, "win_rate": 0.50, "max_drawdown": -0.25}


# ── detect_overfitting — severity classification ──────────────────────────────

class TestSeverityClassification:
    def test_clean_strategy_is_low_or_moderate(self):
        report = detect_overfitting(
            in_sample_metrics=_clean_is(),
            out_of_sample_metrics=_clean_oos(),
            feature_count=10,
            number_of_tests=5,
            oos_bars=504,
        )
        assert report.severity in ("LOW", "MODERATE")
        assert report.overfitting_score < 0.50

    def test_severe_overfit_gets_high_or_severe(self):
        report = detect_overfitting(
            in_sample_metrics=_overfit_is(),
            out_of_sample_metrics=_overfit_oos(),
            feature_count=50,
            number_of_tests=200,
            oos_bars=126,
        )
        assert report.severity in ("HIGH", "SEVERE")
        assert report.overfitting_score > 0.50

    def test_score_is_bounded(self):
        for is_m, oos_m in [(_clean_is(), _clean_oos()), (_overfit_is(), _overfit_oos())]:
            report = detect_overfitting(is_m, oos_m, feature_count=15, number_of_tests=30)
            assert 0.0 <= report.overfitting_score <= 1.0


# ── IS/OOS gap ────────────────────────────────────────────────────────────────

class TestGapScore:
    def test_no_gap_gives_zero_score(self):
        same = {"sharpe_ratio": 1.0, "cagr": 0.12, "win_rate": 0.52, "max_drawdown": -0.10}
        score, details = _compute_gap_score(same, same)
        assert score == pytest.approx(0.0, abs=1e-6)
        assert details["sharpe_gap_abs"] == pytest.approx(0.0, abs=1e-6)

    def test_full_degradation_approaches_one(self):
        is_m  = {"sharpe_ratio": 3.0, "cagr": 0.50, "win_rate": 0.70, "max_drawdown": -0.02}
        oos_m = {"sharpe_ratio": -1.0, "cagr": -0.10, "win_rate": 0.35, "max_drawdown": -0.45}
        score, _ = _compute_gap_score(is_m, oos_m)
        assert score > 0.70

    def test_slight_degradation_is_low(self):
        is_m  = {"sharpe_ratio": 1.2, "cagr": 0.15}
        oos_m = {"sharpe_ratio": 1.1, "cagr": 0.14}
        score, _ = _compute_gap_score(is_m, oos_m)
        assert score < 0.15

    def test_gap_details_keys_present(self):
        _, details = _compute_gap_score(_clean_is(), _clean_oos())
        for key in ("sharpe_is", "sharpe_oos", "sharpe_gap_abs", "sharpe_gap_pct",
                    "cagr_is", "cagr_oos", "win_rate_is", "win_rate_oos",
                    "max_drawdown_is", "max_drawdown_oos"):
            assert key in details


# ── Deflated Sharpe Ratio ─────────────────────────────────────────────────────

class TestDeflatedSharpe:
    def test_high_sr_few_trials_long_sample_is_significant(self):
        dsr, p = _compute_deflated_sharpe(
            sr_hat=2.0, n_trials=3, n_obs=1260, annual_factor=252
        )
        assert dsr > 0.80, f"Expected DSR > 0.80, got {dsr}"
        assert p < 0.20

    def test_modest_sr_many_trials_short_sample_not_significant(self):
        dsr, p = _compute_deflated_sharpe(
            sr_hat=1.0, n_trials=100, n_obs=252, annual_factor=252
        )
        assert dsr < 0.50, f"Expected DSR < 0.50 (many trials, short OOS), got {dsr}"

    def test_more_trials_lower_dsr_same_sr(self):
        dsr5, _  = _compute_deflated_sharpe(sr_hat=1.5, n_trials=5,   n_obs=504)
        dsr50, _ = _compute_deflated_sharpe(sr_hat=1.5, n_trials=50,  n_obs=504)
        dsr200, _ = _compute_deflated_sharpe(sr_hat=1.5, n_trials=200, n_obs=504)
        assert dsr5 > dsr50 > dsr200, "More trials should lower DSR"

    def test_dsr_and_pvalue_sum_to_one(self):
        dsr, p = _compute_deflated_sharpe(sr_hat=1.0, n_trials=10, n_obs=252)
        assert dsr + p == pytest.approx(1.0, abs=1e-4)

    def test_zero_sr_gives_low_dsr(self):
        dsr, p = _compute_deflated_sharpe(sr_hat=0.0, n_trials=10, n_obs=252)
        assert dsr < 0.50


# ── PBO estimate ──────────────────────────────────────────────────────────────

class TestPBO:
    def _folds(self, is_vals, oos_vals):
        return [{"is_sharpe": i, "sharpe_ratio": o} for i, o in zip(is_vals, oos_vals)]

    def test_perfectly_correlated_folds_gives_low_pbo(self):
        folds = self._folds([1.5, 1.2, 0.9, 0.6, 0.3], [1.4, 1.1, 0.8, 0.5, 0.2])
        pbo, corr = _compute_pbo(folds)
        assert pbo < 0.10, f"Perfect correlation should give low PBO, got {pbo}"
        assert corr > 0.90

    def test_anti_correlated_folds_gives_high_pbo(self):
        folds = self._folds([1.5, 1.2, 0.9, 0.6, 0.3], [0.2, 0.5, 0.8, 1.1, 1.4])
        pbo, corr = _compute_pbo(folds)
        assert pbo > 0.80, f"Anti-correlation should give high PBO, got {pbo}"
        assert corr < -0.90

    def test_too_few_folds_returns_neutral(self):
        folds = [{"is_sharpe": 1.0, "sharpe_ratio": 0.8}]  # only 1 fold
        pbo, corr = _compute_pbo(folds)
        assert pbo == pytest.approx(0.5)
        assert corr == pytest.approx(0.0)

    def test_pbo_bounded(self):
        import random
        random.seed(42)
        folds = self._folds(
            [random.gauss(1, 0.3) for _ in range(10)],
            [random.gauss(0.8, 0.4) for _ in range(10)],
        )
        pbo, _ = _compute_pbo(folds)
        assert 0.0 <= pbo <= 1.0


# ── Feature stability ─────────────────────────────────────────────────────────

class TestFeatureStability:
    def test_identical_importances_across_folds_gives_high_stability(self):
        imp = {"rsi_14": 0.20, "vol_21d": 0.15, "ret_5d": 0.10}
        folds = [imp, imp, imp, imp]
        score, unstable = _compute_feature_stability(folds)
        assert score > 0.90, f"Identical importances should give high stability, got {score}"
        assert unstable == []

    def test_wildly_varying_importances_gives_low_stability(self):
        folds = [
            {"rsi_14": 0.50, "vol_21d": 0.01},
            {"rsi_14": 0.01, "vol_21d": 0.50},
            {"rsi_14": 0.40, "vol_21d": 0.05},
            {"rsi_14": 0.05, "vol_21d": 0.45},
        ]
        score, unstable = _compute_feature_stability(folds)
        assert score < 0.30, f"High variance importances should give low stability, got {score}"
        assert len(unstable) > 0

    def test_too_few_folds_returns_neutral(self):
        score, unstable = _compute_feature_stability([{"rsi_14": 0.20}])
        assert score == pytest.approx(0.5)
        assert unstable == []

    def test_empty_returns_neutral(self):
        score, unstable = _compute_feature_stability([])
        assert score == pytest.approx(0.5)
        assert unstable == []


# ── Multiple testing penalty ──────────────────────────────────────────────────

class TestMultipleTestingPenalty:
    def test_single_trial_gives_low_penalty(self):
        penalty, bonf = _compute_multiple_testing_penalty(n_tests=1, n_features=10)
        assert penalty < 0.15
        assert bonf == pytest.approx(0.05, abs=1e-6)

    def test_many_trials_gives_high_penalty(self):
        penalty, bonf = _compute_multiple_testing_penalty(n_tests=500, n_features=50)
        assert penalty > 0.70
        assert bonf < 0.001

    def test_high_feature_count_adds_penalty(self):
        p_few, _ = _compute_multiple_testing_penalty(n_tests=20, n_features=10)
        p_many, _ = _compute_multiple_testing_penalty(n_tests=20, n_features=40)
        assert p_many > p_few

    def test_bonferroni_threshold_is_alpha_over_n(self):
        _, bonf = _compute_multiple_testing_penalty(n_tests=25, n_features=10)
        assert bonf == pytest.approx(0.05 / 25, abs=1e-8)


# ── MinBTL ────────────────────────────────────────────────────────────────────

class TestMinBTL:
    def test_zero_sr_gives_inf(self):
        assert _compute_min_backtest_length(0.0, 10) == float("inf")

    def test_negative_sr_gives_inf(self):
        assert _compute_min_backtest_length(-0.5, 10) == float("inf")

    def test_higher_sr_needs_less_data(self):
        t1 = _compute_min_backtest_length(2.0, 10)
        t2 = _compute_min_backtest_length(1.0, 10)
        assert t1 < t2

    def test_more_trials_needs_more_data(self):
        t1 = _compute_min_backtest_length(1.0, 5)
        t2 = _compute_min_backtest_length(1.0, 100)
        assert t2 > t1


# ── format_report_section ─────────────────────────────────────────────────────

class TestFormatReportSection:
    def _make_report(self) -> OverfittingReport:
        return detect_overfitting(
            in_sample_metrics=_clean_is(),
            out_of_sample_metrics=_clean_oos(),
            feature_count=15,
            number_of_tests=30,
            oos_bars=504,
        )

    def test_contains_key_headings(self):
        md = format_report_section(self._make_report())
        for heading in [
            "Overfitting Detection Analysis",
            "Component Breakdown",
            "IS vs OOS Performance Gap",
            "Deflated Sharpe Ratio",
            "Probability of Backtest Overfitting",
            "Feature Importance Stability",
            "Multiple Testing",
        ]:
            assert heading in md, f"Missing heading: {heading}"

    def test_contains_numeric_score(self):
        report = self._make_report()
        md = format_report_section(report)
        assert str(round(report.overfitting_score, 4)) in md or \
               f"{report.overfitting_score:.3f}" in md

    def test_severity_badge_in_output(self):
        report = self._make_report()
        md = format_report_section(report)
        assert report.severity in md

    def test_recommendations_included_when_present(self):
        report = detect_overfitting(
            in_sample_metrics=_overfit_is(),
            out_of_sample_metrics=_overfit_oos(),
            feature_count=40,
            number_of_tests=100,
        )
        md = format_report_section(report)
        if report.recommendations:
            assert "Recommended Actions" in md

    def test_returns_string(self):
        md = format_report_section(self._make_report())
        assert isinstance(md, str)
        assert len(md) > 500


# ── to_dict / serialisation ───────────────────────────────────────────────────

class TestSerialization:
    def test_to_dict_is_json_serialisable(self):
        import json
        report = detect_overfitting(
            in_sample_metrics=_clean_is(),
            out_of_sample_metrics=_clean_oos(),
            feature_count=15,
            number_of_tests=20,
        )
        d = report.to_dict()
        dumped = json.dumps(d)   # must not raise
        loaded = json.loads(dumped)
        assert loaded["overfitting_score"] == pytest.approx(report.overfitting_score, abs=1e-4)

    def test_to_dict_has_required_keys(self):
        report = detect_overfitting(_clean_is(), _clean_oos(), 10, 5)
        d = report.to_dict()
        for key in ("overfitting_score", "severity", "sharpe_is", "sharpe_oos",
                    "deflated_sharpe_ratio", "pbo_estimate", "n_tests", "n_features",
                    "warnings", "recommendations"):
            assert key in d, f"Missing key in to_dict(): {key}"


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_metrics_does_not_crash(self):
        report = detect_overfitting({}, {}, feature_count=5, number_of_tests=1)
        assert 0.0 <= report.overfitting_score <= 1.0

    def test_missing_optional_keys_uses_defaults(self):
        report = detect_overfitting(
            {"sharpe_ratio": 1.0},
            {"sharpe_ratio": 0.7},
            feature_count=10,
            number_of_tests=10,
        )
        assert report.cagr_is == 0.0
        assert report.win_rate_is == pytest.approx(0.5)

    def test_single_dict_importances_wrapped_correctly(self):
        report = detect_overfitting(
            _clean_is(), _clean_oos(),
            feature_count=5, number_of_tests=5,
            feature_importances={"rsi_14": 0.2, "vol_21d": 0.15},
        )
        # With a single fold, stability should return neutral 0.5
        assert report.feature_stability_score == pytest.approx(0.5)

    def test_fold_metrics_without_is_sharpe_gives_neutral_pbo(self):
        folds = [{"sharpe_ratio": 0.8, "fold": 1}, {"sharpe_ratio": 0.6, "fold": 2}]
        report = detect_overfitting(
            _clean_is(), _clean_oos(),
            feature_count=10, number_of_tests=5,
            fold_metrics=folds,
        )
        assert report.pbo_estimate == pytest.approx(0.5)
        assert report.n_folds_available == 0
