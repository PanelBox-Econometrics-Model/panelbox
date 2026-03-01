"""Tests for RegressionChartBuilder."""

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.regression_charts import RegressionChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_regression_data():
    """Complete Regression transformer output for testing."""
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
            {
                "name": "x3",
                "coef": 0.1,
                "se": 0.5,
                "tstat": 0.2,
                "pvalue": 0.841,
                "stars": "",
                "ci_lower": -0.88,
                "ci_upper": 1.08,
            },
        ],
        "fit_statistics": {
            "r_squared": 0.85,
            "adj_r_squared": 0.83,
            "f_statistic": 42.5,
            "f_pvalue": 0.0001,
        },
        "model_info": {
            "estimator": "Fixed Effects",
            "nobs": 500,
            "n_entities": 50,
            "n_periods": 10,
        },
    }


@pytest.fixture
def minimal_regression_data():
    """Minimal data with only coefficients."""
    return {
        "coefficients": [
            {"name": "x1", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001, "stars": "***"},
        ],
    }


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------


class TestBuildAll:
    def test_full_data_returns_three_charts(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        charts = builder.build_all()
        assert set(charts.keys()) == {"coefficient_plot", "fit_chart", "pvalue_chart"}

    def test_empty_data_returns_empty_dict(self):
        builder = RegressionChartBuilder({})
        assert builder.build_all() == {}

    def test_partial_data_returns_available_charts(self, minimal_regression_data):
        builder = RegressionChartBuilder(minimal_regression_data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        assert "pvalue_chart" in charts
        assert "fit_chart" not in charts

    def test_all_values_are_html_strings(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        charts = builder.build_all()
        for name, html in charts.items():
            assert isinstance(html, str), f"{name} is not a string"
            assert "<div" in html, f"{name} missing <div> element"


# ---------------------------------------------------------------------------
# coefficient_plot
# ---------------------------------------------------------------------------


class TestCoefficientPlot:
    def test_returns_html_with_div(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_coefficients(self):
        builder = RegressionChartBuilder({"coefficients": []})
        assert builder._build_coefficient_plot() is None

    def test_returns_none_for_missing_coefficients(self):
        builder = RegressionChartBuilder({})
        assert builder._build_coefficient_plot() is None

    def test_uses_exact_ci_when_available(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        html = builder._build_coefficient_plot()
        assert html is not None

    def test_falls_back_to_computed_ci(self):
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
        builder = RegressionChartBuilder(data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html

    def test_no_plotlyjs_included(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        html = builder._build_coefficient_plot()
        assert len(html) < 100_000


# ---------------------------------------------------------------------------
# fit_chart
# ---------------------------------------------------------------------------


class TestFitChart:
    def test_returns_html_with_div(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        html = builder._build_fit_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_fit_statistics(self):
        builder = RegressionChartBuilder({"fit_statistics": {}})
        assert builder._build_fit_chart() is None

    def test_returns_none_for_missing_fit_statistics(self):
        builder = RegressionChartBuilder({})
        assert builder._build_fit_chart() is None

    def test_partial_fit_statistics_only_r_squared(self):
        data = {"fit_statistics": {"r_squared": 0.75}}
        builder = RegressionChartBuilder(data)
        html = builder._build_fit_chart()
        assert html is not None

    def test_low_r_squared_color(self):
        builder = RegressionChartBuilder({})
        color = builder._r2_color(0.2)
        assert color == "#dc3545"  # danger

    def test_medium_r_squared_color(self):
        builder = RegressionChartBuilder({})
        color = builder._r2_color(0.5)
        assert color == "#ffc107"  # warning

    def test_high_r_squared_color(self):
        builder = RegressionChartBuilder({})
        color = builder._r2_color(0.8)
        assert color == "#28a745"  # success


# ---------------------------------------------------------------------------
# pvalue_chart
# ---------------------------------------------------------------------------


class TestPvalueChart:
    def test_returns_html_with_div(self, full_regression_data):
        builder = RegressionChartBuilder(full_regression_data)
        html = builder._build_pvalue_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_coefficients(self):
        builder = RegressionChartBuilder({"coefficients": []})
        assert builder._build_pvalue_chart() is None

    def test_returns_none_for_missing_coefficients(self):
        builder = RegressionChartBuilder({})
        assert builder._build_pvalue_chart() is None

    def test_pvalue_bar_colors(self):
        builder = RegressionChartBuilder({})
        assert builder._pvalue_bar_color(0.001) == "#28a745"  # success
        assert builder._pvalue_bar_color(0.03) == "#17a2b8"  # info
        assert builder._pvalue_bar_color(0.08) == "#ffc107"  # warning
        assert builder._pvalue_bar_color(0.5) == "#dc3545"  # danger

    def test_single_coefficient_pvalues(self):
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
        html = builder._build_pvalue_chart()
        assert html is not None
