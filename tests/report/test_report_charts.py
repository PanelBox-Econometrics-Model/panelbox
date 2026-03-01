"""
Tests for report chart builders and integration with ReportManager.

Tests the chart builder modules (unit tests) and verifies that generated
reports contain interactive Plotly charts (integration tests).
"""

from __future__ import annotations

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.discrete_charts import DiscreteChartBuilder  # noqa: E402
from panelbox.report.charts.gmm_charts import GMMChartBuilder  # noqa: E402
from panelbox.report.charts.quantile_charts import QuantileChartBuilder  # noqa: E402
from panelbox.report.charts.regression_charts import RegressionChartBuilder  # noqa: E402
from panelbox.report.charts.sfa_charts import SFAChartBuilder  # noqa: E402
from panelbox.report.charts.var_charts import VARChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assert_valid_chart_html(html: str, chart_name: str = "chart") -> None:
    """Assert that HTML is a valid Plotly chart div (not a full page)."""
    assert isinstance(html, str), f"{chart_name}: expected str, got {type(html)}"
    assert "<div" in html, f"{chart_name}: missing <div> element"
    # Must NOT include plotly.js library itself (CDN is the template's job)
    assert "plotly-2.27.0.min.js" not in html, f"{chart_name}: contains Plotly CDN"
    assert len(html) < 200_000, f"{chart_name}: suspiciously large ({len(html)} bytes)"


# ===========================================================================
# GMM Chart Builder Tests
# ===========================================================================


class TestGMMChartBuilder:
    """Unit tests for GMMChartBuilder."""

    @pytest.fixture
    def full_data(self):
        return {
            "coefficients": [
                {
                    "name": "L.y",
                    "coef": 0.45,
                    "se": 0.08,
                    "tstat": 5.63,
                    "pvalue": 0.000,
                    "stars": "***",
                },
                {
                    "name": "x1",
                    "coef": 0.32,
                    "se": 0.12,
                    "tstat": 2.67,
                    "pvalue": 0.008,
                    "stars": "***",
                },
                {
                    "name": "x2",
                    "coef": -0.05,
                    "se": 0.10,
                    "tstat": -0.50,
                    "pvalue": 0.617,
                    "stars": "",
                },
            ],
            "diagnostics": {
                "hansen": {"statistic": 5.2, "pvalue": 0.39, "df": 5, "status": "PASS"},
                "ar1": {"statistic": -2.5, "pvalue": 0.012},
                "ar2": {"statistic": 0.8, "pvalue": 0.423, "status": "PASS"},
            },
            "model_info": {
                "nobs": 200,
                "n_groups": 40,
                "n_instruments": 25,
            },
        }

    def test_build_all_returns_dict(self, full_data):
        builder = GMMChartBuilder(full_data)
        charts = builder.build_all()
        assert isinstance(charts, dict)
        assert len(charts) >= 3
        for name, html in charts.items():
            _assert_valid_chart_html(html, name)

    def test_coefficient_plot_html(self, full_data):
        builder = GMMChartBuilder(full_data)
        html = builder._build_coefficient_plot()
        _assert_valid_chart_html(html, "coefficient_plot")

    def test_diagnostic_chart_html(self, full_data):
        builder = GMMChartBuilder(full_data)
        html = builder._build_diagnostic_chart()
        _assert_valid_chart_html(html, "diagnostic_chart")

    def test_instrument_chart_html(self, full_data):
        builder = GMMChartBuilder(full_data)
        html = builder._build_instrument_chart()
        _assert_valid_chart_html(html, "instrument_chart")

    def test_empty_data_graceful(self):
        builder = GMMChartBuilder({})
        result = builder.build_all()
        assert result == {}

    def test_minimal_data_coefficient_only(self):
        data = {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 0.5,
                    "se": 0.1,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
            ],
        }
        builder = GMMChartBuilder(data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        _assert_valid_chart_html(charts["coefficient_plot"], "coefficient_plot")


# ===========================================================================
# Regression Chart Builder Tests
# ===========================================================================


class TestRegressionChartBuilder:
    """Unit tests for RegressionChartBuilder."""

    @pytest.fixture
    def full_data(self):
        return {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.5,
                    "se": 0.3,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                    "ci_lower": 0.91,
                    "ci_upper": 2.09,
                },
                {
                    "name": "x2",
                    "coef": -0.8,
                    "se": 0.4,
                    "tstat": -2.0,
                    "pvalue": 0.046,
                    "stars": "**",
                    "ci_lower": -1.58,
                    "ci_upper": -0.02,
                },
            ],
            "fit_statistics": {
                "r_squared": 0.85,
                "adj_r_squared": 0.83,
                "f_statistic": 42.5,
            },
        }

    def test_build_all_returns_dict(self, full_data):
        builder = RegressionChartBuilder(full_data)
        charts = builder.build_all()
        assert isinstance(charts, dict)
        assert len(charts) >= 3
        for name, html in charts.items():
            _assert_valid_chart_html(html, name)

    def test_coefficient_plot_html(self, full_data):
        builder = RegressionChartBuilder(full_data)
        html = builder._build_coefficient_plot()
        _assert_valid_chart_html(html, "coefficient_plot")

    def test_fit_chart_html(self, full_data):
        builder = RegressionChartBuilder(full_data)
        html = builder._build_fit_chart()
        _assert_valid_chart_html(html, "fit_chart")

    def test_pvalue_chart_html(self, full_data):
        builder = RegressionChartBuilder(full_data)
        html = builder._build_pvalue_chart()
        _assert_valid_chart_html(html, "pvalue_chart")

    def test_empty_data_graceful(self):
        builder = RegressionChartBuilder({})
        result = builder.build_all()
        assert result == {}

    def test_minimal_data_coefficient_only(self):
        data = {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 0.5,
                    "se": 0.1,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
            ],
        }
        builder = RegressionChartBuilder(data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        assert "pvalue_chart" in charts


# ===========================================================================
# Discrete Chart Builder Tests
# ===========================================================================


class TestDiscreteChartBuilder:
    """Unit tests for DiscreteChartBuilder."""

    @pytest.fixture
    def full_data(self):
        return {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.20,
                    "se": 0.30,
                    "zstat": 4.00,
                    "pvalue": 0.001,
                    "stars": "***",
                },
                {
                    "name": "x2",
                    "coef": -0.50,
                    "se": 0.25,
                    "zstat": -2.00,
                    "pvalue": 0.046,
                    "stars": "**",
                },
            ],
            "fit_statistics": {
                "loglikelihood": -72.15,
                "aic": 150.30,
                "bic": 165.70,
            },
            "classification": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
            },
        }

    def test_build_all_returns_dict(self, full_data):
        builder = DiscreteChartBuilder(full_data)
        charts = builder.build_all()
        assert isinstance(charts, dict)
        assert len(charts) >= 3
        for name, html in charts.items():
            _assert_valid_chart_html(html, name)

    def test_coefficient_plot_html(self, full_data):
        builder = DiscreteChartBuilder(full_data)
        html = builder._build_coefficient_plot()
        _assert_valid_chart_html(html, "coefficient_plot")

    def test_classification_chart_html(self, full_data):
        builder = DiscreteChartBuilder(full_data)
        html = builder._build_classification_chart()
        _assert_valid_chart_html(html, "classification_chart")

    def test_ic_chart_html(self, full_data):
        builder = DiscreteChartBuilder(full_data)
        html = builder._build_ic_chart()
        _assert_valid_chart_html(html, "ic_chart")

    def test_empty_data_graceful(self):
        builder = DiscreteChartBuilder({})
        result = builder.build_all()
        assert result == {}

    def test_no_classification_skips_chart(self):
        data = {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.0,
                    "se": 0.5,
                    "zstat": 2.0,
                    "pvalue": 0.046,
                    "stars": "**",
                },
            ],
            "classification": None,
        }
        builder = DiscreteChartBuilder(data)
        charts = builder.build_all()
        assert "classification_chart" not in charts
        assert "coefficient_plot" in charts


# ===========================================================================
# SFA Chart Builder Tests
# ===========================================================================


class TestSFAChartBuilder:
    """Unit tests for SFAChartBuilder."""

    @pytest.fixture
    def full_data(self):
        return {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 0.5,
                    "se": 0.1,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
                {
                    "name": "x2",
                    "coef": -0.3,
                    "se": 0.15,
                    "tstat": -2.0,
                    "pvalue": 0.046,
                    "stars": "**",
                },
            ],
            "variance_components": {
                "sigma_v": 0.3,
                "sigma_u": 0.5,
                "gamma": 0.74,
            },
            "efficiency": {
                "mean": 0.82,
                "median": 0.85,
                "std": 0.12,
                "min": 0.31,
                "max": 0.99,
                "count": 100,
            },
        }

    def test_build_all_returns_dict(self, full_data):
        builder = SFAChartBuilder(full_data)
        charts = builder.build_all()
        assert isinstance(charts, dict)
        assert len(charts) >= 4
        for name, html in charts.items():
            _assert_valid_chart_html(html, name)

    def test_efficiency_distribution_html(self, full_data):
        builder = SFAChartBuilder(full_data)
        html = builder._build_efficiency_distribution()
        _assert_valid_chart_html(html, "efficiency_distribution")

    def test_variance_chart_html(self, full_data):
        builder = SFAChartBuilder(full_data)
        html = builder._build_variance_chart()
        _assert_valid_chart_html(html, "variance_chart")

    def test_coefficient_plot_html(self, full_data):
        builder = SFAChartBuilder(full_data)
        html = builder._build_coefficient_plot()
        _assert_valid_chart_html(html, "coefficient_plot")

    def test_empty_data_graceful(self):
        builder = SFAChartBuilder({})
        result = builder.build_all()
        assert result == {}

    def test_no_efficiency_skips_efficiency_charts(self):
        data = {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 0.5,
                    "se": 0.1,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
            ],
            "efficiency": None,
        }
        builder = SFAChartBuilder(data)
        charts = builder.build_all()
        assert "efficiency_distribution" not in charts
        assert "coefficient_plot" in charts


# ===========================================================================
# VAR Chart Builder Tests
# ===========================================================================


class TestVARChartBuilder:
    """Unit tests for VARChartBuilder."""

    @pytest.fixture
    def full_data(self):
        return {
            "equations": [
                {
                    "name": "y1",
                    "coefficients": [
                        {
                            "name": "y1.L1",
                            "coef": 0.5,
                            "se": 0.1,
                            "tstat": 5.0,
                            "pvalue": 0.001,
                            "stars": "***",
                        },
                        {
                            "name": "y2.L1",
                            "coef": -0.2,
                            "se": 0.15,
                            "tstat": -1.3,
                            "pvalue": 0.19,
                            "stars": "",
                        },
                    ],
                },
                {
                    "name": "y2",
                    "coefficients": [
                        {
                            "name": "y1.L1",
                            "coef": 0.3,
                            "se": 0.2,
                            "tstat": 1.5,
                            "pvalue": 0.13,
                            "stars": "",
                        },
                        {
                            "name": "y2.L1",
                            "coef": 0.7,
                            "se": 0.1,
                            "tstat": 7.0,
                            "pvalue": 0.001,
                            "stars": "***",
                        },
                    ],
                },
            ],
            "diagnostics": {"aic": -1500.0, "bic": -1450.0, "hqic": -1480.0},
            "stability": {"is_stable": True, "max_eigenvalue_modulus": 0.85},
        }

    def test_build_all_returns_dict(self, full_data):
        builder = VARChartBuilder(full_data)
        charts = builder.build_all()
        assert isinstance(charts, dict)
        assert len(charts) >= 3
        for name, html in charts.items():
            _assert_valid_chart_html(html, name)

    def test_stability_chart_html(self, full_data):
        builder = VARChartBuilder(full_data)
        html = builder._build_stability_chart()
        _assert_valid_chart_html(html, "stability_chart")

    def test_ic_chart_html(self, full_data):
        builder = VARChartBuilder(full_data)
        html = builder._build_ic_chart()
        _assert_valid_chart_html(html, "ic_chart")

    def test_coefficient_heatmap_html(self, full_data):
        builder = VARChartBuilder(full_data)
        html = builder._build_coefficient_heatmap()
        _assert_valid_chart_html(html, "coefficient_heatmap")

    def test_empty_data_graceful(self):
        builder = VARChartBuilder({})
        result = builder.build_all()
        assert result == {}

    def test_minimal_single_equation(self):
        data = {
            "equations": [
                {
                    "name": "y1",
                    "coefficients": [
                        {
                            "name": "y1.L1",
                            "coef": 0.5,
                            "se": 0.1,
                            "tstat": 5.0,
                            "pvalue": 0.001,
                            "stars": "***",
                        },
                    ],
                },
            ],
        }
        builder = VARChartBuilder(data)
        charts = builder.build_all()
        assert "coefficient_heatmap" in charts


# ===========================================================================
# Quantile Chart Builder Tests
# ===========================================================================


class TestQuantileChartBuilder:
    """Unit tests for QuantileChartBuilder."""

    @pytest.fixture
    def full_data(self):
        return {
            "health": {
                "score": 0.85,
                "score_pct": "85%",
                "status": "good",
                "color": "#28a745",
            },
            "tests": [
                {"name": "Shapiro-Wilk", "statistic": 0.98, "pvalue": 0.12, "status": "pass"},
                {"name": "Breusch-Pagan", "statistic": 3.5, "pvalue": 0.06, "status": "warning"},
                {"name": "Durbin-Watson", "statistic": 5.2, "pvalue": 0.02, "status": "fail"},
            ],
        }

    def test_build_all_returns_dict(self, full_data):
        builder = QuantileChartBuilder(full_data)
        charts = builder.build_all()
        assert isinstance(charts, dict)
        assert len(charts) >= 2
        for name, html in charts.items():
            _assert_valid_chart_html(html, name)

    def test_health_gauge_html(self, full_data):
        builder = QuantileChartBuilder(full_data)
        html = builder._build_health_gauge()
        _assert_valid_chart_html(html, "health_gauge")

    def test_test_results_chart_html(self, full_data):
        builder = QuantileChartBuilder(full_data)
        html = builder._build_test_results_chart()
        _assert_valid_chart_html(html, "test_results_chart")

    def test_empty_data_graceful(self):
        builder = QuantileChartBuilder({})
        result = builder.build_all()
        assert result == {}


# ===========================================================================
# Integration Tests: Reports with Charts
# ===========================================================================


@pytest.fixture
def manager():
    """Create a ReportManager instance."""
    from panelbox.report import ReportManager

    return ReportManager()


class TestGMMReportWithCharts:
    """Integration test: GMM report generates charts."""

    def test_gmm_report_contains_plotly_divs(self, manager):
        data = {
            "model_info": {
                "estimator": "System GMM",
                "nobs": 200,
                "n_groups": 40,
                "n_instruments": 25,
                "two_step": True,
            },
            "coefficients": [
                {
                    "name": "L.y",
                    "coef": 0.5,
                    "se": 0.1,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
                {
                    "name": "x1",
                    "coef": 0.3,
                    "se": 0.15,
                    "tstat": 2.0,
                    "pvalue": 0.045,
                    "stars": "**",
                },
            ],
            "diagnostics": {
                "hansen": {"statistic": 5.2, "pvalue": 0.39, "df": 5, "status": "PASS"},
                "ar1": {"statistic": -2.5, "pvalue": 0.012},
                "ar2": {"statistic": 0.8, "pvalue": 0.42, "status": "PASS"},
            },
        }
        html = manager.generate_gmm_report(data, title="GMM with Charts")
        assert "GMM with Charts" in html
        # Report should contain Plotly chart divs
        assert "plotly" in html.lower()
        assert "<div" in html


class TestRegressionReportWithCharts:
    """Integration test: Regression report generates charts."""

    def test_regression_report_contains_plotly_divs(self, manager):
        data = {
            "model_info": {
                "estimator": "Fixed Effects",
                "nobs": 500,
                "n_entities": 50,
                "n_periods": 10,
            },
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.5,
                    "se": 0.3,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                    "ci_lower": 0.91,
                    "ci_upper": 2.09,
                },
            ],
            "fit_statistics": {
                "r_squared": 0.85,
                "adj_r_squared": 0.83,
                "f_statistic": 42.5,
            },
        }
        html = manager.generate_regression_report(data, title="Regression with Charts")
        assert "Regression with Charts" in html
        assert "plotly" in html.lower()
        assert "<div" in html


class TestDiscreteReportWithCharts:
    """Integration test: Discrete report generates charts."""

    def test_discrete_report_contains_plotly_divs(self, manager):
        data = {
            "model_info": {"model_type": "Logit", "converged": True},
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.2,
                    "se": 0.3,
                    "zstat": 4.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
            ],
            "fit_statistics": {"aic": 150.3, "bic": 165.7},
            "classification": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
            },
        }
        html = manager.generate_discrete_report(data, title="Discrete with Charts")
        assert "Discrete with Charts" in html
        assert "plotly" in html.lower()
        assert "<div" in html


class TestSFAReportWithCharts:
    """Integration test: SFA report generates charts."""

    def test_sfa_report_contains_plotly_divs(self, manager):
        data = {
            "model_info": {"frontier_type": "production", "distribution": "half-normal"},
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 0.5,
                    "se": 0.1,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "stars": "***",
                },
            ],
            "variance_components": {"sigma_v": 0.3, "sigma_u": 0.5, "gamma": 0.74},
            "efficiency": {
                "mean": 0.82,
                "median": 0.85,
                "std": 0.12,
                "min": 0.31,
                "max": 0.99,
                "count": 100,
            },
            "fit_statistics": {
                "loglikelihood": -150.5,
                "aic": 309.0,
                "bic": 320.0,
            },
        }
        html = manager.generate_sfa_report(data, title="SFA with Charts")
        assert "SFA with Charts" in html
        assert "plotly" in html.lower()
        assert "<div" in html


class TestVARReportWithCharts:
    """Integration test: VAR report generates charts."""

    def test_var_report_contains_plotly_divs(self, manager):
        data = {
            "model_info": {"K": 2, "p": 1, "N": 50, "n_obs": 450},
            "equations": [
                {
                    "name": "y1",
                    "coefficients": [
                        {
                            "name": "y1.L1",
                            "coef": 0.5,
                            "se": 0.1,
                            "tstat": 5.0,
                            "pvalue": 0.001,
                            "stars": "***",
                        },
                    ],
                },
            ],
            "diagnostics": {"aic": -1500.0, "bic": -1450.0, "hqic": -1480.0},
            "stability": {"is_stable": True, "max_eigenvalue_modulus": 0.85},
        }
        html = manager.generate_var_report(data, title="VAR with Charts")
        assert "VAR with Charts" in html
        assert "plotly" in html.lower()
        assert "<div" in html


class TestQuantileReportWithCharts:
    """Integration test: Quantile report generates charts."""

    def test_quantile_report_contains_plotly_divs(self, manager):
        data = {
            "health": {
                "score": 0.85,
                "score_pct": "85%",
                "status": "good",
                "color": "#28a745",
            },
            "tests": [
                {"name": "Shapiro-Wilk", "statistic": 0.98, "pvalue": 0.12, "status": "pass"},
                {"name": "Breusch-Pagan", "statistic": 3.5, "pvalue": 0.06, "status": "warning"},
            ],
            "recommendations": ["Consider robust standard errors"],
        }
        html = manager.generate_quantile_report(data, title="Quantile with Charts")
        assert "Quantile with Charts" in html
        assert "plotly" in html.lower()
        assert "<div" in html
