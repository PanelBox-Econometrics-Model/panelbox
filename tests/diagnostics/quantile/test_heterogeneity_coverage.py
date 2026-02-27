"""Tests for panelbox.diagnostics.quantile.heterogeneity module.

Covers HeterogeneityTests methods, result summary methods,
MonotonicityTestResult.plot(), and plot_coefficient_paths to raise coverage
from ~52.90% to 80%+.
"""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.diagnostics.quantile.heterogeneity import (
    HeterogeneityTests,
    JointEqualityTestResult,
    MonotonicityTestResult,
    SlopeEqualityTestResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _close_figures():
    """Prevent matplotlib memory leaks."""
    yield
    plt.close("all")


def _make_tau_result(params, cov_matrix=None, bse=None):
    """Build a minimal per-tau result namespace."""
    p = len(params)
    if cov_matrix is None:
        cov_matrix = np.eye(p) * 0.01
    ns = SimpleNamespace(params=params, cov_matrix=cov_matrix)
    if bse is not None:
        ns.bse = bse
    return ns


def _make_result(tau_params_map, cov_matrices=None):
    """Build a mock QuantilePanelResult with .results dict."""
    results = {}
    for tau, params in tau_params_map.items():
        cov = cov_matrices.get(tau) if cov_matrices else None
        results[tau] = _make_tau_result(params, cov_matrix=cov)
    return SimpleNamespace(results=results)


@pytest.fixture
def multi_tau_result():
    """Result with 3 quantiles and 2 params (intercept + slope)."""
    return _make_result(
        {
            0.25: np.array([1.0, 2.0]),
            0.50: np.array([1.5, 2.5]),
            0.75: np.array([2.0, 3.0]),
        }
    )


@pytest.fixture
def heterogeneity_tests(multi_tau_result):
    return HeterogeneityTests(multi_tau_result)


# ---------------------------------------------------------------------------
# Tests: HeterogeneityTests.__init__
# ---------------------------------------------------------------------------


class TestHeterogeneityTestsInit:
    def test_init_valid(self, multi_tau_result):
        """Test initialization with valid multi-tau result."""
        ht = HeterogeneityTests(multi_tau_result)
        assert ht.tau_list == [0.25, 0.50, 0.75]

    def test_init_too_few_quantiles(self):
        """Test initialization raises with < 2 quantiles."""
        result = _make_result({0.5: np.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="Need at least 2 quantiles"):
            HeterogeneityTests(result)


# ---------------------------------------------------------------------------
# Tests: HeterogeneityTests.test_slope_equality
# ---------------------------------------------------------------------------


class TestSlopeEquality:
    def test_slope_equality_default(self, heterogeneity_tests):
        """Test slope equality with default args (all vars, adjacent pairs)."""
        result = heterogeneity_tests.test_slope_equality()
        assert isinstance(result, SlopeEqualityTestResult)
        assert result.df > 0
        assert 0 <= result.p_value <= 1

    def test_slope_equality_single_var(self, heterogeneity_tests):
        """Test slope equality for a single variable index (scalar)."""
        result = heterogeneity_tests.test_slope_equality(var_idx=0)
        assert isinstance(result, SlopeEqualityTestResult)
        assert result.var_idx == [0]

    def test_slope_equality_specific_pairs(self, heterogeneity_tests):
        """Test slope equality with specific tau pairs."""
        result = heterogeneity_tests.test_slope_equality(tau_pairs=[(0.25, 0.75)])
        assert len(result.tau_pairs) == 1
        assert len(result.individual_stats) == 1

    def test_slope_equality_singular_cov(self):
        """Test slope equality with singular covariance (pinv fallback)."""
        # Make cov_matrix singular
        cov = np.array([[1.0, 1.0], [1.0, 1.0]])
        result = _make_result(
            {0.25: np.array([1.0, 2.0]), 0.75: np.array([2.0, 3.0])},
            cov_matrices={0.25: cov, 0.75: cov},
        )
        ht = HeterogeneityTests(result)
        res = ht.test_slope_equality()
        assert isinstance(res, SlopeEqualityTestResult)


# ---------------------------------------------------------------------------
# Tests: HeterogeneityTests.test_joint_equality
# ---------------------------------------------------------------------------


class TestJointEquality:
    def test_joint_equality_default(self, heterogeneity_tests):
        """Test joint equality with default (all quantiles)."""
        result = heterogeneity_tests.test_joint_equality()
        assert isinstance(result, JointEqualityTestResult)
        assert result.df > 0
        assert result.coef_matrix.shape == (3, 2)

    def test_joint_equality_subset(self, heterogeneity_tests):
        """Test joint equality with a subset of quantiles."""
        result = heterogeneity_tests.test_joint_equality(tau_subset=[0.25, 0.75])
        assert isinstance(result, JointEqualityTestResult)
        assert result.coef_matrix.shape == (2, 2)


# ---------------------------------------------------------------------------
# Tests: HeterogeneityTests.test_monotonicity
# ---------------------------------------------------------------------------


class TestMonotoncityMethod:
    def test_monotonicity_increasing(self, heterogeneity_tests):
        """Test monotonicity for an increasing coefficient path."""
        result = heterogeneity_tests.test_monotonicity(var_idx=1)
        assert isinstance(result, MonotonicityTestResult)
        assert result.is_increasing is True
        assert result.var_idx == 1

    def test_monotonicity_non_monotonic(self):
        """Test monotonicity for a non-monotonic coefficient path."""
        result = _make_result(
            {
                0.25: np.array([1.0, 3.0]),
                0.50: np.array([1.5, 1.0]),
                0.75: np.array([2.0, 2.0]),
            }
        )
        ht = HeterogeneityTests(result)
        res = ht.test_monotonicity(var_idx=1)
        assert res.is_increasing is False
        assert res.is_decreasing is False


# ---------------------------------------------------------------------------
# Tests: HeterogeneityTests.interquantile_range_test
# ---------------------------------------------------------------------------


class TestInterquantileRangeTest:
    def test_iqr_test(self):
        """Test interquantile range test with 0.25 and 0.75."""
        result = _make_result(
            {
                0.25: np.array([1.0, 2.0]),
                0.50: np.array([1.5, 2.5]),
                0.75: np.array([2.0, 4.0]),
            }
        )
        ht = HeterogeneityTests(result)
        stat, pval = ht.interquantile_range_test()
        assert stat >= 0
        assert 0 <= pval <= 1

    def test_iqr_test_missing_quantiles(self):
        """Test IQR test raises when 0.25 or 0.75 not available."""
        result = _make_result(
            {
                0.10: np.array([1.0, 2.0]),
                0.90: np.array([2.0, 3.0]),
            }
        )
        ht = HeterogeneityTests(result)
        with pytest.raises(ValueError, match=r"Need quantiles 0\.25 and 0\.75"):
            ht.interquantile_range_test()


# ---------------------------------------------------------------------------
# Tests: SlopeEqualityTestResult.summary()
# ---------------------------------------------------------------------------


class TestSlopeEqualityResultSummary:
    def test_summary_reject(self, capsys, heterogeneity_tests):
        """Test SlopeEqualityTestResult.summary() when H0 is rejected."""
        result = SlopeEqualityTestResult(
            statistic=15.0,
            p_value=0.001,
            df=2,
            tau_pairs=[(0.25, 0.50), (0.50, 0.75)],
            var_idx=[0, 1],
            individual_stats=[8.0, 7.0],
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Slope Equality Test" in captured.out
        assert "REJECT equality" in captured.out
        assert "Individual pair tests" in captured.out

    def test_summary_fail_to_reject(self, capsys):
        """Test SlopeEqualityTestResult.summary() when H0 not rejected."""
        result = SlopeEqualityTestResult(
            statistic=1.5,
            p_value=0.50,
            df=1,
            tau_pairs=[(0.25, 0.75)],
            var_idx=[0],
            individual_stats=[1.5],
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Cannot reject equality" in captured.out

    def test_summary_single_pair(self, capsys):
        """Test summary with a single tau pair (no individual pair section)."""
        result = SlopeEqualityTestResult(
            statistic=2.0,
            p_value=0.30,
            df=1,
            tau_pairs=[(0.25, 0.75)],
            var_idx=[0],
            individual_stats=[2.0],
        )
        result.summary()
        captured = capsys.readouterr()
        # With len(individual_stats)==1, "Individual pair tests" should not print
        assert "Individual pair tests" not in captured.out


# ---------------------------------------------------------------------------
# Tests: JointEqualityTestResult.summary()
# ---------------------------------------------------------------------------


class TestJointEqualityResultSummary:
    def test_summary_reject(self, capsys):
        """Test JointEqualityTestResult.summary() when H0 is rejected."""
        result = JointEqualityTestResult(
            statistic=20.0,
            p_value=0.001,
            df=4,
            tau_subset=[0.25, 0.50, 0.75],
            coef_matrix=np.array([[1, 2], [1.5, 2.5], [2, 3]]),
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Joint Equality Test" in captured.out
        assert "REJECT H0" in captured.out
        assert "Quantile regression is justified" in captured.out

    def test_summary_fail_to_reject(self, capsys):
        """Test JointEqualityTestResult.summary() when H0 not rejected."""
        result = JointEqualityTestResult(
            statistic=2.0,
            p_value=0.50,
            df=4,
            tau_subset=[0.25, 0.50, 0.75],
            coef_matrix=np.array([[1, 2], [1, 2], [1, 2]]),
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Cannot reject H0" in captured.out
        assert "Mean regression may be sufficient" in captured.out


# ---------------------------------------------------------------------------
# Tests: MonotonicityTestResult.summary() and .plot()
# ---------------------------------------------------------------------------


class TestMonotonicityResultSummary:
    def test_summary_increasing_significant(self, capsys):
        """Test summary for significant increasing monotonicity."""
        result = MonotonicityTestResult(
            correlation=0.95,
            p_value=0.01,
            is_increasing=True,
            is_decreasing=False,
            coef_path=np.array([1.0, 2.0, 3.0]),
            tau_list=[0.25, 0.50, 0.75],
            var_idx=0,
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Monotonicity Test" in captured.out
        assert "Strictly Increasing" in captured.out
        assert "Strong increasing trend" in captured.out

    def test_summary_decreasing_significant(self, capsys):
        """Test summary for significant decreasing monotonicity."""
        result = MonotonicityTestResult(
            correlation=-0.95,
            p_value=0.01,
            is_increasing=False,
            is_decreasing=True,
            coef_path=np.array([3.0, 2.0, 1.0]),
            tau_list=[0.25, 0.50, 0.75],
            var_idx=1,
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Strictly Decreasing" in captured.out
        assert "Strong decreasing trend" in captured.out

    def test_summary_non_monotonic(self, capsys):
        """Test summary for non-monotonic pattern."""
        result = MonotonicityTestResult(
            correlation=0.2,
            p_value=0.50,
            is_increasing=False,
            is_decreasing=False,
            coef_path=np.array([1.0, 3.0, 2.0]),
            tau_list=[0.25, 0.50, 0.75],
            var_idx=0,
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Non-monotonic" in captured.out
        assert "No clear monotonic pattern" in captured.out

    def test_plot_increasing(self):
        """Test MonotonicityTestResult.plot() visualization for increasing."""
        result = MonotonicityTestResult(
            correlation=0.95,
            p_value=0.01,
            is_increasing=True,
            is_decreasing=False,
            coef_path=np.array([1.0, 2.0, 3.0]),
            tau_list=[0.25, 0.50, 0.75],
            var_idx=0,
        )
        fig = result.plot()
        assert fig is not None

    def test_plot_decreasing(self):
        """Test MonotonicityTestResult.plot() for decreasing pattern."""
        result = MonotonicityTestResult(
            correlation=-0.95,
            p_value=0.01,
            is_increasing=False,
            is_decreasing=True,
            coef_path=np.array([3.0, 2.0, 1.0]),
            tau_list=[0.25, 0.50, 0.75],
            var_idx=1,
        )
        fig = result.plot()
        assert fig is not None

    def test_plot_non_monotonic(self):
        """Test MonotonicityTestResult.plot() for non-monotonic pattern."""
        result = MonotonicityTestResult(
            correlation=0.1,
            p_value=0.80,
            is_increasing=False,
            is_decreasing=False,
            coef_path=np.array([1.0, 3.0, 2.0]),
            tau_list=[0.25, 0.50, 0.75],
            var_idx=0,
        )
        fig = result.plot()
        assert fig is not None


# ---------------------------------------------------------------------------
# Tests: plot_coefficient_paths
# ---------------------------------------------------------------------------


class TestPlotCoefficientPaths:
    def test_plot_coefficient_paths_default(self, heterogeneity_tests):
        """Test plot_coefficient_paths with default var_names."""
        fig = heterogeneity_tests.plot_coefficient_paths()
        assert fig is not None

    def test_plot_coefficient_paths_custom_names(self, heterogeneity_tests):
        """Test plot_coefficient_paths with custom variable names."""
        fig = heterogeneity_tests.plot_coefficient_paths(var_names=["Intercept", "X1"])
        assert fig is not None

    def test_plot_coefficient_paths_no_confidence_bands(self, heterogeneity_tests):
        """Test plot_coefficient_paths without confidence bands."""
        fig = heterogeneity_tests.plot_coefficient_paths(confidence_bands=False)
        assert fig is not None

    def test_plot_coefficient_paths_with_bse(self):
        """Test plot_coefficient_paths when result has bse attribute."""
        results = {}
        for tau, params in [
            (0.25, np.array([1.0, 2.0])),
            (0.75, np.array([2.0, 3.0])),
        ]:
            results[tau] = _make_tau_result(params, bse=np.array([0.1, 0.2]))
        result = SimpleNamespace(results=results)
        ht = HeterogeneityTests(result)
        fig = ht.plot_coefficient_paths()
        assert fig is not None

    def test_plot_odd_number_params(self):
        """Test plot_coefficient_paths with odd number of params (hides unused axes)."""
        result = _make_result(
            {
                0.25: np.array([1.0, 2.0, 3.0]),
                0.75: np.array([1.5, 2.5, 3.5]),
            }
        )
        ht = HeterogeneityTests(result)
        fig = ht.plot_coefficient_paths()
        assert fig is not None
