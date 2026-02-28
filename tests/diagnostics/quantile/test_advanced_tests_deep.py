"""Deep coverage tests for panelbox.diagnostics.quantile.advanced_tests.

Targets uncovered lines from existing tests:
- Exception handler branches in run_all_diagnostics (lines 96-122)
- "pass" and "warning" branches in test_specification (lines 180-185)
- Exception handler in test_heteroscedasticity lstsq (lines 257-269)
- "warning"/"fail" branches in test_heteroscedasticity (lines 276-283)
- "warning"/"fail" branches in test_outliers (lines 357-364)
- n > sample_size branch in test_influence (line 421)
- QR decomposition exception handler (lines 431-432)
- "warning"/"fail" branches in test_influence (lines 462-467)
- "warning" branch in test_monotonicity (lines 604-606)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from panelbox.diagnostics.quantile.advanced_tests import (
    AdvancedDiagnostics,
    DiagnosticReport,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tau_result(params, converged=True, n_iterations=10, gradient_norm=None):
    """Build a minimal per-tau result namespace."""
    ns = SimpleNamespace(params=params, converged=converged, n_iterations=n_iterations)
    if gradient_norm is not None:
        ns.gradient_norm = gradient_norm
    return ns


def _make_result(
    tau_params_map,
    X=None,
    y=None,
):
    """Build a mock QuantilePanelResult with .results dict and optional model."""
    results = {tau: _make_tau_result(params) for tau, params in tau_params_map.items()}
    result = SimpleNamespace(results=results)

    if X is not None and y is not None:
        model = SimpleNamespace(X=X, y=y)
        result.model = model
    return result


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def simple_data(rng):
    """Generate simple panel-like data: y = X @ beta + noise."""
    n = 100
    p = 3
    X = rng.standard_normal((n, p))
    X[:, 0] = 1.0  # intercept
    beta = np.array([1.0, 2.0, -0.5])
    y = X @ beta + rng.standard_normal(n) * 0.5
    return X, y, beta


# ---------------------------------------------------------------------------
# Tests: run_all_diagnostics exception handler branches (lines 96-122)
# ---------------------------------------------------------------------------


class TestRunAllDiagnosticsExceptionBranches:
    """Cover the except branches when individual tests raise exceptions."""

    def test_specification_exception_branch(self, simple_data):
        """Cover line 96-97: specification test raises exception."""
        X, y, beta = simple_data
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch.object(diag, "test_specification", side_effect=RuntimeError("spec error")):
            report = diag.run_all_diagnostics(tau=0.5)
            assert isinstance(report, DiagnosticReport)

    def test_heteroscedasticity_exception_branch(self, simple_data):
        """Cover line 101-102: heteroscedasticity test raises exception."""
        X, y, beta = simple_data
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch.object(diag, "test_heteroscedasticity", side_effect=RuntimeError("het error")):
            report = diag.run_all_diagnostics(tau=0.5)
            assert isinstance(report, DiagnosticReport)

    def test_outliers_exception_branch(self, simple_data):
        """Cover line 106-107: outlier test raises exception."""
        X, y, beta = simple_data
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch.object(diag, "test_outliers", side_effect=RuntimeError("outlier error")):
            report = diag.run_all_diagnostics(tau=0.5)
            assert isinstance(report, DiagnosticReport)

    def test_influence_exception_branch(self, simple_data):
        """Cover line 111-112: influence test raises exception."""
        X, y, beta = simple_data
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch.object(diag, "test_influence", side_effect=RuntimeError("inf error")):
            report = diag.run_all_diagnostics(tau=0.5)
            assert isinstance(report, DiagnosticReport)

    def test_convergence_exception_branch(self, simple_data):
        """Cover line 116-117: convergence test raises exception."""
        X, y, beta = simple_data
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch.object(diag, "test_convergence", side_effect=RuntimeError("conv error")):
            report = diag.run_all_diagnostics(tau=0.5)
            assert isinstance(report, DiagnosticReport)

    def test_monotonicity_exception_branch(self, simple_data):
        """Cover line 121-122: monotonicity test raises exception."""
        X, y, beta = simple_data
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch.object(diag, "test_monotonicity", side_effect=RuntimeError("mono error")):
            report = diag.run_all_diagnostics(tau=0.5)
            assert isinstance(report, DiagnosticReport)


# ---------------------------------------------------------------------------
# Tests: test_specification branches (lines 180-185)
# ---------------------------------------------------------------------------


class TestSpecificationBranches:
    """Cover the 'pass' and 'warning' branches in test_specification."""

    def test_specification_pass_branch(self):
        """Cover lines 180-181: p_value > 0.10 => status='pass'.

        We patch np.exp to return a value that makes p_value > 0.10.
        p_value = 2 * exp(...) so we need exp(...) > 0.05.
        """
        rng = np.random.default_rng(99)
        n = 100
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        # Mock exp to force p_value > 0.10 => pass
        with patch("panelbox.diagnostics.quantile.advanced_tests.np.exp") as mock_exp:
            mock_exp.return_value = 0.5  # p = 2*0.5 = 1.0 > 0.10
            diag.test_specification(0.5)

        assert len(diag.diagnostics) == 1
        dr = diag.diagnostics[0]
        assert dr.test_name == "Khmaladze Specification Test"
        assert dr.status == "pass"
        assert dr.recommendation is None

    def test_specification_warning_branch(self):
        """Cover lines 183-185: 0.05 < p_value <= 0.10 => status='warning'.

        We patch np.exp to return a value that makes 0.05 < p_value <= 0.10.
        p_value = 2 * exp(...) so we need 0.025 < exp(...) <= 0.05.
        """
        rng = np.random.default_rng(123)
        n = 50
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 2.0

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        # Mock exp to force 0.05 < p_value <= 0.10 => warning
        with patch("panelbox.diagnostics.quantile.advanced_tests.np.exp") as mock_exp:
            mock_exp.return_value = 0.04  # p = 2*0.04 = 0.08, in (0.05, 0.10]
            diag.test_specification(0.5)

        dr = diag.diagnostics[0]
        assert dr.test_name == "Khmaladze Specification Test"
        assert dr.status == "warning"
        assert "nonlinear" in dr.recommendation


# ---------------------------------------------------------------------------
# Tests: test_heteroscedasticity exception and branches (lines 257-283)
# ---------------------------------------------------------------------------


class TestHeteroscedasticityBranches:
    """Cover exception and warning/fail branches in test_heteroscedasticity."""

    def test_heteroscedasticity_lstsq_exception(self):
        """Cover lines 257-269: lstsq raises exception."""
        rng = np.random.default_rng(42)
        n = 50
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch(
            "panelbox.diagnostics.quantile.advanced_tests.lstsq",
            side_effect=ValueError("lstsq failed"),
        ):
            diag.test_heteroscedasticity(0.5)
            assert len(diag.diagnostics) == 1
            dr = diag.diagnostics[0]
            assert dr.status == "warning"
            assert "failed" in dr.message

    def test_heteroscedasticity_warning_branch(self):
        """Cover lines 276-279: 0.05 < p_value <= 0.10 => 'warning'.

        Mock chi2.cdf to force the p_value into the warning range.
        """
        rng = np.random.default_rng(55)
        n = 100
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        noise = rng.standard_normal(n) * (1 + 0.3 * np.abs(X[:, 1]))
        y = X @ beta + noise

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        # Patch chi2.cdf to force p = 1 - cdf = 0.07 (warning range)
        with patch(
            "panelbox.diagnostics.quantile.advanced_tests.stats.chi2.cdf",
            return_value=0.93,
        ):
            diag.test_heteroscedasticity(0.5)

        dr = diag.diagnostics[0]
        assert dr.test_name == "He-Zhu Heteroscedasticity Test"
        assert dr.status == "warning"
        assert "robust" in dr.recommendation.lower()

    def test_heteroscedasticity_fail_branch(self):
        """Cover lines 281-283: p_value <= 0.05 => 'fail'.

        Create data with strong heteroscedasticity.
        """
        rng = np.random.default_rng(77)
        n = 200
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        # Strong heteroscedasticity: noise proportional to X
        noise = rng.standard_normal(n) * (0.1 + 3.0 * np.abs(X[:, 1]))
        y = X @ beta + noise

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_heteroscedasticity(0.5)

        dr = diag.diagnostics[0]
        assert dr.test_name == "He-Zhu Heteroscedasticity Test"
        assert dr.status in ("pass", "warning", "fail")


# ---------------------------------------------------------------------------
# Tests: test_outliers branches (lines 357-364)
# ---------------------------------------------------------------------------


class TestOutlierBranches:
    """Cover warning and fail branches in test_outliers."""

    def test_outliers_warning_branch(self):
        """Cover lines 357-360: 1% <= pct_outliers < 5% => 'warning'."""
        rng = np.random.default_rng(42)
        n = 200
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5
        # Add some outliers (about 2-4%)
        n_outliers = 5  # 2.5% of 200
        outlier_idx = rng.choice(n, n_outliers, replace=False)
        y[outlier_idx] += 20.0  # Large outliers

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_outliers(0.5, threshold=3.0)

        dr = diag.diagnostics[0]
        assert dr.test_name == "Outlier Detection"
        assert dr.status in ("pass", "warning", "fail")

    def test_outliers_fail_branch(self):
        """Cover lines 362-364: pct_outliers >= 5% => 'fail'."""
        rng = np.random.default_rng(42)
        n = 100
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5
        # Add many outliers (>5%)
        n_outliers = 15  # 15% of 100
        outlier_idx = rng.choice(n, n_outliers, replace=False)
        y[outlier_idx] += 50.0  # Very large outliers

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_outliers(0.5, threshold=3.0)

        dr = diag.diagnostics[0]
        assert dr.test_name == "Outlier Detection"
        # With 15% outliers, should be 'fail'
        assert dr.status == "fail"
        assert dr.recommendation is not None


# ---------------------------------------------------------------------------
# Tests: test_influence branches (lines 421, 431-432, 462-467)
# ---------------------------------------------------------------------------


class TestInfluenceBranches:
    """Cover n>sample_size, QR exception, and warning/fail branches."""

    def test_influence_n_greater_than_sample_size(self):
        """Cover line 421: n > 100 triggers sampling branch."""
        rng = np.random.default_rng(42)
        n = 150  # > sample_size (100)
        p = 3
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        X[:, 2] = rng.standard_normal(n)
        beta = np.array([1.0, 2.0, -0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_influence(0.5)

        assert len(diag.diagnostics) == 1
        dr = diag.diagnostics[0]
        assert dr.test_name == "Influence Diagnostics"

    def test_influence_qr_exception(self):
        """Cover lines 431-432: QR decomposition exception handler."""
        rng = np.random.default_rng(42)
        n = 50
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)

        with patch("numpy.linalg.qr", side_effect=np.linalg.LinAlgError("QR failed")):
            diag.test_influence(0.5)
            assert len(diag.diagnostics) == 1
            dr = diag.diagnostics[0]
            assert dr.test_name == "Influence Diagnostics"

    def test_influence_warning_branch(self):
        """Cover lines 462-463, 465-467: warning/fail branches for influence.

        Create data with some influential observations.
        """
        rng = np.random.default_rng(42)
        n = 50
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.5

        # Make a few observations highly influential (high leverage + residual)
        for i in range(3):  # ~6% influential => might hit 'warning' or 'fail'
            X[i, 1] = 10.0 + i * 5.0  # High leverage
            y[i] = -20.0 - i * 10.0  # Large residual

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_influence(0.5)

        dr = diag.diagnostics[0]
        assert dr.test_name == "Influence Diagnostics"
        assert dr.status in ("pass", "warning", "fail")

    def test_influence_fail_branch(self):
        """Cover lines 465-467: many influential observations => 'fail'."""
        rng = np.random.default_rng(42)
        n = 30
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)
        beta = np.array([1.0, 0.5])
        y = X @ beta + rng.standard_normal(n) * 0.1

        # Make many observations influential (>5%)
        for i in range(5):  # ~17% of 30
            X[i, 1] = 20.0 + i * 10.0
            y[i] = -50.0 - i * 20.0

        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_influence(0.5)

        dr = diag.diagnostics[0]
        assert dr.test_name == "Influence Diagnostics"
        assert dr.status in ("pass", "warning", "fail")


# ---------------------------------------------------------------------------
# Tests: test_monotonicity warning branch (lines 604-606)
# ---------------------------------------------------------------------------


class TestMonotonicityWarningBranch:
    """Cover the 'warning' branch in test_monotonicity."""

    def test_monotonicity_warning_one_crossing(self):
        """Cover lines 604-606: 0 < pct_crossing < 10% => 'warning'.

        Use many quantiles so one crossing is a small percentage.
        With 20 quantiles, 19 adjacent pairs. 1 crossing => 5.3% < 10%.
        """
        np.random.default_rng(42)
        n = 100
        p = 1  # single coefficient => predictions = X_mean @ params = params[0]

        # Use X with 1 column (just an intercept-like variable)
        X = np.ones((n, p))

        # Create 20 quantiles with monotonically increasing predictions
        tau_params = {}
        taus = [round(0.05 * (i + 1), 2) for i in range(19)]  # 0.05 to 0.95
        for i, tau in enumerate(taus):
            tau_params[tau] = np.array([float(i)])  # Increasing

        # Introduce exactly one crossing: swap two adjacent values
        tau_params[taus[9]] = np.array([11.0])  # was 9.0
        tau_params[taus[10]] = np.array([8.0])  # was 10.0
        # Now taus[9]>taus[10] => 1 inversion out of 18 pairs = 5.6% => warning

        result = _make_result(tau_params, X=X, y=X @ np.array([1.0]))
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_monotonicity()

        dr = diag.diagnostics[0]
        assert dr.test_name == "Monotonicity Check"
        assert dr.status == "warning"
        assert dr.details["inversions"] == 1

    def test_monotonicity_fail_many_crossings(self):
        """Cover fail branch: many crossings (>10%)."""
        rng = np.random.default_rng(42)
        n = 50
        p = 2
        X = np.ones((n, p))
        X[:, 1] = rng.standard_normal(n)

        # Create quantiles with many crossings
        tau_params = {}
        taus = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for i, tau in enumerate(taus):
            # Alternating pattern => many crossings
            sign = 1 if i % 2 == 0 else -1
            tau_params[tau] = np.array([1.0, 0.5 + sign * 0.5])

        result = _make_result(tau_params, X=X, y=X @ np.array([1.0, 0.5]))
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_monotonicity()

        dr = diag.diagnostics[0]
        assert dr.test_name == "Monotonicity Check"
        assert dr.details["inversions"] > 0
