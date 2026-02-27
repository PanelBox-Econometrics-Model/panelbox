"""Tests for panelbox.diagnostics.quantile.advanced_tests module.

Covers AdvancedDiagnostics and DiagnosticReport classes to raise coverage
from ~8.82% to 75%+.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from panelbox.diagnostics.quantile.advanced_tests import (
    AdvancedDiagnostics,
    DiagnosticReport,
    DiagnosticResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_tau_result(params, converged=True, n_iterations=10, gradient_norm=None):
    """Build a minimal per-tau result namespace."""
    ns = SimpleNamespace(params=params, converged=converged, n_iterations=n_iterations)
    if gradient_norm is not None:
        ns.gradient_norm = gradient_norm
    return ns


def _make_result(
    tau_params_map: dict[float, np.ndarray],
    X: np.ndarray | None = None,
    y: np.ndarray | None = None,
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


@pytest.fixture
def mock_result(simple_data):
    """Mock result with data and a single tau=0.5 result."""
    X, y, beta = simple_data
    return _make_result({0.5: beta}, X=X, y=y)


@pytest.fixture
def mock_result_multi_tau(simple_data):
    """Mock result with multiple quantiles (0.25, 0.5, 0.75)."""
    X, y, beta = simple_data
    return _make_result(
        {
            0.25: beta - np.array([0.0, 0.1, 0.05]),
            0.50: beta,
            0.75: beta + np.array([0.0, 0.1, 0.05]),
        },
        X=X,
        y=y,
    )


@pytest.fixture
def mock_result_no_model():
    """Mock result without model/X/y attributes."""
    beta = np.array([1.0, 2.0])
    results = {0.5: _make_tau_result(beta)}
    return SimpleNamespace(results=results)


# ---------------------------------------------------------------------------
# Tests: AdvancedDiagnostics.__init__
# ---------------------------------------------------------------------------


class TestAdvancedDiagnosticsInit:
    def test_init_with_model(self, mock_result):
        """Test initialization with a result that has a model attribute."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        assert diag.result is mock_result
        assert diag.verbose is False
        assert diag.diagnostics == []
        assert diag.X is not None
        assert diag.y is not None

    def test_init_without_model(self, mock_result_no_model):
        """Test initialization when result has no model attribute."""
        diag = AdvancedDiagnostics(mock_result_no_model, verbose=False)
        assert diag.model is None
        assert diag.X is None
        assert diag.y is None


# ---------------------------------------------------------------------------
# Tests: AdvancedDiagnostics test methods
# ---------------------------------------------------------------------------


class TestSpecificationTest:
    def test_specification_with_data(self, mock_result):
        """Test Khmaladze specification test runs and produces result."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        diag.test_specification(0.5)
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.test_name == "Khmaladze Specification Test"
        assert not np.isnan(d.statistic)
        assert 0.0 <= d.p_value <= 1.0

    def test_specification_no_data(self, mock_result_no_model):
        """Test specification test skips when data unavailable."""
        diag = AdvancedDiagnostics(mock_result_no_model, verbose=False)
        diag.test_specification(0.5)
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.status == "warning"
        assert np.isnan(d.statistic)


class TestHeteroscedasticityTest:
    def test_heteroscedasticity_with_data(self, mock_result):
        """Test He-Zhu heteroscedasticity test runs and produces result."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        diag.test_heteroscedasticity(0.5)
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.test_name == "He-Zhu Heteroscedasticity Test"
        assert not np.isnan(d.statistic)

    def test_heteroscedasticity_no_data(self, mock_result_no_model):
        """Test heteroscedasticity test skips when data unavailable."""
        diag = AdvancedDiagnostics(mock_result_no_model, verbose=False)
        diag.test_heteroscedasticity(0.5)
        d = diag.diagnostics[0]
        assert d.status == "warning"
        assert np.isnan(d.statistic)


class TestOutlierTest:
    def test_outliers_with_data(self, mock_result):
        """Test outlier detection runs and produces result."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        diag.test_outliers(0.5)
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.test_name == "Outlier Detection"
        assert d.status in ("pass", "warning", "fail")

    def test_outliers_no_data(self, mock_result_no_model):
        """Test outlier detection skips when data unavailable."""
        diag = AdvancedDiagnostics(mock_result_no_model, verbose=False)
        diag.test_outliers(0.5)
        d = diag.diagnostics[0]
        assert d.status == "warning"

    def test_outliers_mad_zero(self, rng):
        """Test outlier detection when MAD is zero (constant residuals)."""
        n = 50
        X = np.ones((n, 1))
        y = np.ones(n)  # constant -> residuals ~ 0 -> MAD = 0
        beta = np.array([1.0])
        result = _make_result({0.5: beta}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_outliers(0.5)
        assert len(diag.diagnostics) == 1


class TestInfluenceTest:
    def test_influence_with_data(self, mock_result):
        """Test influence diagnostics runs and produces result."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        diag.test_influence(0.5)
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.test_name == "Influence Diagnostics"
        assert hasattr(diag, "cooks_d")
        assert hasattr(diag, "leverage")

    def test_influence_no_data(self, mock_result_no_model):
        """Test influence diagnostics skips when data unavailable."""
        diag = AdvancedDiagnostics(mock_result_no_model, verbose=False)
        diag.test_influence(0.5)
        d = diag.diagnostics[0]
        assert d.status == "warning"


class TestConvergenceTest:
    def test_convergence_with_data(self, mock_result):
        """Test convergence check with model data available."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        diag.test_convergence(0.5)
        assert len(diag.diagnostics) == 1
        d = diag.diagnostics[0]
        assert d.test_name == "Convergence Check"

    def test_convergence_with_gradient_norm(self):
        """Test convergence when gradient_norm is provided on result."""
        tau_res = _make_tau_result(np.array([1.0]), gradient_norm=1e-8)
        result = SimpleNamespace(results={0.5: tau_res})
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_convergence(0.5)
        d = diag.diagnostics[0]
        assert d.status == "pass"

    def test_convergence_warning_gradient(self):
        """Test convergence warning when gradient norm is moderate."""
        tau_res = _make_tau_result(np.array([1.0]), gradient_norm=5e-5)
        result = SimpleNamespace(results={0.5: tau_res})
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_convergence(0.5)
        d = diag.diagnostics[0]
        assert d.status == "warning"

    def test_convergence_fail_gradient(self):
        """Test convergence failure when gradient norm is large."""
        tau_res = _make_tau_result(np.array([1.0]), converged=False, gradient_norm=1.0)
        result = SimpleNamespace(results={0.5: tau_res})
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_convergence(0.5)
        d = diag.diagnostics[0]
        assert d.status == "fail"

    def test_convergence_no_gradient_converged(self):
        """Test convergence when no gradient_norm and no data, but converged."""
        tau_res = _make_tau_result(np.array([1.0]))
        result = SimpleNamespace(results={0.5: tau_res})
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_convergence(0.5)
        d = diag.diagnostics[0]
        assert d.status == "pass"

    def test_convergence_no_gradient_not_converged(self):
        """Test convergence when not converged and no gradient info."""
        tau_res = _make_tau_result(np.array([1.0]), converged=False)
        result = SimpleNamespace(results={0.5: tau_res})
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_convergence(0.5)
        d = diag.diagnostics[0]
        assert d.status == "fail"


class TestMonotonicityTest:
    def test_monotonicity_single_tau(self, mock_result):
        """Test monotonicity with single quantile (trivially passes)."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        diag.test_monotonicity()
        d = diag.diagnostics[0]
        assert d.status == "pass"
        assert "Single quantile" in d.message

    def test_monotonicity_multi_tau_no_crossing(self, mock_result_multi_tau):
        """Test monotonicity with correctly ordered quantiles."""
        diag = AdvancedDiagnostics(mock_result_multi_tau, verbose=False)
        diag.test_monotonicity()
        d = diag.diagnostics[0]
        assert d.test_name == "Monotonicity Check"

    def test_monotonicity_with_crossing(self, simple_data):
        """Test monotonicity when quantile curves cross."""
        X, y, _beta = simple_data
        # Predictions at X_mean must be *decreasing* to create inversions.
        # X_mean[0] = 1 (intercept), X_mean[1:] close to 0.
        # So prediction ~ intercept param.  Make intercept decrease with tau.
        result = _make_result(
            {
                0.25: np.array([3.0, 0.0, 0.0]),
                0.50: np.array([2.0, 0.0, 0.0]),
                0.75: np.array([1.0, 0.0, 0.0]),
            },
            X=X,
            y=y,
        )
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_monotonicity()
        d = diag.diagnostics[0]
        # Should detect inversions
        assert d.details["inversions"] > 0

    def test_monotonicity_no_data_multi_tau(self):
        """Test monotonicity without X data but multiple taus."""
        beta = np.array([1.0, 2.0])
        results = {
            0.25: _make_tau_result(beta - np.array([0.1, 0.1])),
            0.50: _make_tau_result(beta),
            0.75: _make_tau_result(beta + np.array([0.1, 0.1])),
        }
        result = SimpleNamespace(results=results)
        diag = AdvancedDiagnostics(result, verbose=False)
        diag.test_monotonicity()
        d = diag.diagnostics[0]
        assert d.test_name == "Monotonicity Check"


class TestRunAllDiagnostics:
    def test_run_all_diagnostics(self, mock_result_multi_tau):
        """Test run_all_diagnostics executes complete battery."""
        diag = AdvancedDiagnostics(mock_result_multi_tau, verbose=False)
        report = diag.run_all_diagnostics(tau=0.5)
        assert isinstance(report, DiagnosticReport)
        assert len(report.diagnostics) >= 1

    def test_run_all_diagnostics_default_tau(self, mock_result):
        """Test run_all_diagnostics picks 0.5 by default."""
        diag = AdvancedDiagnostics(mock_result, verbose=False)
        report = diag.run_all_diagnostics()
        assert isinstance(report, DiagnosticReport)

    def test_run_all_diagnostics_non_median_default(self, simple_data):
        """Test run_all_diagnostics picks first tau when 0.5 not available."""
        X, y, beta = simple_data
        result = _make_result({0.25: beta, 0.75: beta + 0.1}, X=X, y=y)
        diag = AdvancedDiagnostics(result, verbose=False)
        report = diag.run_all_diagnostics()
        assert isinstance(report, DiagnosticReport)

    def test_run_all_verbose(self, mock_result, capsys):
        """Test run_all_diagnostics with verbose=True prints output."""
        diag = AdvancedDiagnostics(mock_result, verbose=True)
        diag.run_all_diagnostics(tau=0.5)
        captured = capsys.readouterr()
        assert "QUANTILE REGRESSION DIAGNOSTICS" in captured.out


# ---------------------------------------------------------------------------
# Tests: DiagnosticReport
# ---------------------------------------------------------------------------


class TestDiagnosticReport:
    def test_init_with_diagnostics(self):
        """Test DiagnosticReport initialization with diagnostic results."""
        diags = [
            DiagnosticResult("Test1", 1.5, 0.3, "pass", "OK"),
            DiagnosticResult("Test2", 3.0, 0.02, "fail", "Bad", "Fix it"),
        ]
        report = DiagnosticReport(diags)
        assert len(report.diagnostics) == 2
        assert hasattr(report, "health_score")
        assert hasattr(report, "health_status")

    def test_init_empty(self):
        """Test DiagnosticReport with no diagnostics."""
        report = DiagnosticReport([])
        assert report.health_score == 0.0
        assert report.health_status == "unknown"

    def test_health_score_all_pass(self):
        """Test health score is 1.0 when all tests pass."""
        diags = [
            DiagnosticResult("T1", 1.0, 0.5, "pass", "OK"),
            DiagnosticResult("T2", 1.0, 0.5, "pass", "OK"),
        ]
        report = DiagnosticReport(diags)
        assert report.health_score == 1.0
        assert report.health_status == "good"

    def test_health_score_all_fail(self):
        """Test health score is 0.0 when all tests fail."""
        diags = [
            DiagnosticResult("T1", 1.0, 0.01, "fail", "Bad"),
            DiagnosticResult("T2", 1.0, 0.01, "fail", "Bad"),
        ]
        report = DiagnosticReport(diags)
        assert report.health_score == 0.0
        assert report.health_status == "poor"

    def test_health_score_mixed(self):
        """Test health score with mix of pass/warning/fail."""
        diags = [
            DiagnosticResult("T1", 1.0, 0.5, "pass", "OK"),
            DiagnosticResult("T2", 1.0, 0.08, "warning", "Meh"),
            DiagnosticResult("T3", 1.0, 0.01, "fail", "Bad"),
        ]
        report = DiagnosticReport(diags)
        assert report.health_score == pytest.approx(0.5)
        assert report.health_status == "fair"

    def test_print_summary(self, capsys):
        """Test print_summary output format."""
        diags = [
            DiagnosticResult("Test Pass", 1.23, 0.456, "pass", "OK"),
            DiagnosticResult("Test Warn", np.nan, np.nan, "warning", "Meh", "Fix"),
        ]
        report = DiagnosticReport(diags)
        report.print_summary()
        captured = capsys.readouterr()
        assert "QUANTILE REGRESSION DIAGNOSTICS" in captured.out
        assert "Test Pass" in captured.out
        assert "RECOMMENDATIONS" in captured.out

    def test_print_summary_no_recommendations(self, capsys):
        """Test print_summary without any recommendations."""
        diags = [DiagnosticResult("Test Pass", 1.23, 0.456, "pass", "OK")]
        report = DiagnosticReport(diags)
        report.print_summary()
        captured = capsys.readouterr()
        assert "RECOMMENDATIONS" not in captured.out

    def test_to_dict(self):
        """Test conversion to dictionary format."""
        diags = [
            DiagnosticResult("T1", 1.5, 0.3, "pass", "Good", details={"a": 1}),
        ]
        report = DiagnosticReport(diags)
        d = report.to_dict()
        assert "health_score" in d
        assert "health_status" in d
        assert "tests" in d
        assert len(d["tests"]) == 1
        assert d["tests"][0]["name"] == "T1"
        assert d["tests"][0]["statistic"] == 1.5

    def test_to_html(self):
        """Test HTML report generation."""
        diags = [
            DiagnosticResult("T1", 1.5, 0.3, "pass", "Good"),
            DiagnosticResult("T2", np.nan, np.nan, "warning", "Meh", "Fix this"),
            DiagnosticResult("T3", 5.0, 0.01, "fail", "Bad", "Respec"),
        ]
        report = DiagnosticReport(diags)
        html = report.to_html()
        assert "diagnostic-report" in html
        assert "Overall Health" in html
        assert "T1" in html
        assert "Recommendations" in html

    def test_to_html_no_recommendations(self):
        """Test HTML report without recommendations."""
        diags = [DiagnosticResult("T1", 1.5, 0.3, "pass", "Good")]
        report = DiagnosticReport(diags)
        html = report.to_html()
        assert "Recommendations" not in html
