"""Tests for SFAChartBuilder."""

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.sfa_charts import SFAChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_sfa_data():
    """Complete SFA transformer output for testing."""
    return {
        "coefficients": [
            {"name": "x1", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001, "stars": "***"},
            {"name": "x2", "coef": -0.3, "se": 0.15, "tstat": -2.0, "pvalue": 0.046, "stars": "**"},
            {"name": "x3", "coef": 0.05, "se": 0.2, "tstat": 0.25, "pvalue": 0.80, "stars": ""},
        ],
        "variance_components": {
            "sigma_v": 0.3,
            "sigma_u": 0.5,
            "sigma": 0.583,
            "sigma_sq": 0.34,
            "lambda_param": 1.667,
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
        "fit_statistics": {
            "loglikelihood": -150.5,
            "aic": 309.0,
            "bic": 320.0,
        },
        "model_info": {
            "frontier_type": "production",
            "distribution": "half-normal",
        },
    }


@pytest.fixture
def minimal_sfa_data():
    """Minimal data with only coefficients."""
    return {
        "coefficients": [
            {"name": "x1", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001, "stars": "***"},
        ],
    }


@pytest.fixture
def sfa_data_no_efficiency():
    """SFA data with efficiency=None."""
    return {
        "coefficients": [
            {"name": "x1", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001, "stars": "***"},
        ],
        "variance_components": {"sigma_v": 0.3, "sigma_u": 0.5, "gamma": 0.74},
        "efficiency": None,
    }


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------


class TestBuildAll:
    def test_full_data_returns_four_charts(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        charts = builder.build_all()
        assert set(charts.keys()) == {
            "efficiency_distribution",
            "efficiency_summary",
            "variance_chart",
            "coefficient_plot",
        }

    def test_empty_data_returns_empty_dict(self):
        builder = SFAChartBuilder({})
        assert builder.build_all() == {}

    def test_partial_data_returns_available_charts(self, minimal_sfa_data):
        builder = SFAChartBuilder(minimal_sfa_data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        assert "efficiency_distribution" not in charts
        assert "efficiency_summary" not in charts
        assert "variance_chart" not in charts

    def test_all_values_are_html_strings(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        charts = builder.build_all()
        for name, html in charts.items():
            assert isinstance(html, str), f"{name} is not a string"
            assert "<div" in html, f"{name} missing <div> element"

    def test_efficiency_none_skips_efficiency_charts(self, sfa_data_no_efficiency):
        builder = SFAChartBuilder(sfa_data_no_efficiency)
        charts = builder.build_all()
        assert "efficiency_distribution" not in charts
        assert "efficiency_summary" not in charts
        assert "variance_chart" in charts
        assert "coefficient_plot" in charts


# ---------------------------------------------------------------------------
# efficiency_distribution
# ---------------------------------------------------------------------------


class TestEfficiencyDistribution:
    def test_returns_html_with_div(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_efficiency_distribution()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_efficiency(self):
        builder = SFAChartBuilder({})
        assert builder._build_efficiency_distribution() is None

    def test_returns_none_for_none_efficiency(self):
        builder = SFAChartBuilder({"efficiency": None})
        assert builder._build_efficiency_distribution() is None

    def test_returns_none_for_efficiency_without_mean(self):
        builder = SFAChartBuilder({"efficiency": {"median": 0.85}})
        assert builder._build_efficiency_distribution() is None

    def test_no_plotlyjs_included(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_efficiency_distribution()
        assert len(html) < 100_000

    def test_works_with_only_mean(self):
        data = {"efficiency": {"mean": 0.75}}
        builder = SFAChartBuilder(data)
        html = builder._build_efficiency_distribution()
        assert html is not None
        assert "<div" in html


# ---------------------------------------------------------------------------
# efficiency_summary
# ---------------------------------------------------------------------------


class TestEfficiencySummary:
    def test_returns_html_with_div(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_efficiency_summary()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_efficiency(self):
        builder = SFAChartBuilder({})
        assert builder._build_efficiency_summary() is None

    def test_returns_none_for_none_efficiency(self):
        builder = SFAChartBuilder({"efficiency": None})
        assert builder._build_efficiency_summary() is None

    def test_returns_none_for_all_none_values(self):
        data = {"efficiency": {"mean": None, "median": None, "std": None}}
        builder = SFAChartBuilder(data)
        assert builder._build_efficiency_summary() is None

    def test_no_plotlyjs_included(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_efficiency_summary()
        assert len(html) < 100_000

    def test_partial_efficiency_stats(self):
        data = {"efficiency": {"mean": 0.82, "median": 0.85}}
        builder = SFAChartBuilder(data)
        html = builder._build_efficiency_summary()
        assert html is not None
        assert "<div" in html


# ---------------------------------------------------------------------------
# variance_chart
# ---------------------------------------------------------------------------


class TestVarianceChart:
    def test_returns_html_with_div(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_variance_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_variance(self):
        builder = SFAChartBuilder({})
        assert builder._build_variance_chart() is None

    def test_returns_none_for_none_variance(self):
        builder = SFAChartBuilder({"variance_components": None})
        assert builder._build_variance_chart() is None

    def test_returns_none_for_missing_sigma_values(self):
        builder = SFAChartBuilder({"variance_components": {"gamma": 0.5}})
        assert builder._build_variance_chart() is None

    def test_no_plotlyjs_included(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_variance_chart()
        assert len(html) < 100_000

    def test_works_without_gamma(self):
        data = {"variance_components": {"sigma_v": 0.3, "sigma_u": 0.5}}
        builder = SFAChartBuilder(data)
        html = builder._build_variance_chart()
        assert html is not None
        assert "<div" in html


# ---------------------------------------------------------------------------
# coefficient_plot
# ---------------------------------------------------------------------------


class TestCoefficientPlot:
    def test_returns_html_with_div(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_coefficients(self):
        builder = SFAChartBuilder({"coefficients": []})
        assert builder._build_coefficient_plot() is None

    def test_returns_none_for_missing_coefficients(self):
        builder = SFAChartBuilder({})
        assert builder._build_coefficient_plot() is None

    def test_no_plotlyjs_included(self, full_sfa_data):
        builder = SFAChartBuilder(full_sfa_data)
        html = builder._build_coefficient_plot()
        assert len(html) < 100_000

    def test_single_coefficient(self):
        data = {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.0,
                    "se": 0.5,
                    "tstat": 2.0,
                    "pvalue": 0.046,
                    "stars": "**",
                },
            ],
        }
        builder = SFAChartBuilder(data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html

    def test_uses_ci_when_available(self):
        data = {
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.0,
                    "se": 0.5,
                    "tstat": 2.0,
                    "pvalue": 0.046,
                    "stars": "**",
                    "ci_lower": 0.01,
                    "ci_upper": 1.99,
                },
            ],
        }
        builder = SFAChartBuilder(data)
        html = builder._build_coefficient_plot()
        assert html is not None
