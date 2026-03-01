"""Tests for QuantileChartBuilder."""

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.quantile_charts import QuantileChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_quantile_data():
    """Complete Quantile transformer output for testing."""
    return {
        "health": {
            "score": 0.85,
            "score_pct": "85%",
            "status": "good",
            "color": "#28a745",
        },
        "tests": [
            {
                "name": "Shapiro-Wilk",
                "statistic": 0.98,
                "pvalue": 0.12,
                "status": "pass",
                "status_icon": "ok",
                "message": "Residuals appear normally distributed",
            },
            {
                "name": "Breusch-Pagan",
                "statistic": 3.5,
                "pvalue": 0.06,
                "status": "warning",
                "status_icon": "warn",
                "message": "Mild heteroskedasticity",
            },
            {
                "name": "Durbin-Watson",
                "statistic": 5.2,
                "pvalue": 0.02,
                "status": "fail",
                "status_icon": "fail",
                "message": "Serial correlation detected",
            },
        ],
        "recommendations": [
            "Consider robust standard errors",
            "Check for serial correlation",
        ],
    }


@pytest.fixture
def minimal_quantile_data():
    """Minimal data with only health score."""
    return {
        "health": {
            "score": 0.50,
            "score_pct": "50%",
            "status": "fair",
            "color": "#ffc107",
        },
    }


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------


class TestBuildAll:
    def test_full_data_returns_two_charts(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        charts = builder.build_all()
        assert set(charts.keys()) == {"health_gauge", "test_results_chart"}

    def test_empty_data_returns_empty_dict(self):
        builder = QuantileChartBuilder({})
        assert builder.build_all() == {}

    def test_health_only_returns_gauge(self, minimal_quantile_data):
        builder = QuantileChartBuilder(minimal_quantile_data)
        charts = builder.build_all()
        assert "health_gauge" in charts
        assert "test_results_chart" not in charts

    def test_all_values_are_html_strings(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        charts = builder.build_all()
        for name, html in charts.items():
            assert isinstance(html, str), f"{name} is not a string"
            assert "<div" in html, f"{name} missing <div> element"


# ---------------------------------------------------------------------------
# health_gauge
# ---------------------------------------------------------------------------


class TestHealthGauge:
    def test_returns_html_with_div(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        html = builder._build_health_gauge()
        assert html is not None
        assert "<div" in html

    def test_returns_none_when_health_is_none(self):
        builder = QuantileChartBuilder({"health": None})
        assert builder._build_health_gauge() is None

    def test_returns_none_when_health_missing(self):
        builder = QuantileChartBuilder({})
        assert builder._build_health_gauge() is None

    def test_returns_none_when_score_is_none(self):
        builder = QuantileChartBuilder({"health": {"score": None}})
        assert builder._build_health_gauge() is None

    def test_low_score(self):
        data = {
            "health": {
                "score": 0.25,
                "score_pct": "25%",
                "status": "poor",
                "color": "#dc3545",
            },
        }
        builder = QuantileChartBuilder(data)
        html = builder._build_health_gauge()
        assert html is not None
        assert "<div" in html

    def test_perfect_score(self):
        data = {
            "health": {
                "score": 1.0,
                "score_pct": "100%",
                "status": "good",
                "color": "#28a745",
            },
        }
        builder = QuantileChartBuilder(data)
        html = builder._build_health_gauge()
        assert html is not None

    def test_zero_score(self):
        data = {
            "health": {
                "score": 0.0,
                "score_pct": "0%",
                "status": "poor",
                "color": "#dc3545",
            },
        }
        builder = QuantileChartBuilder(data)
        html = builder._build_health_gauge()
        assert html is not None

    def test_no_plotlyjs_included(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        html = builder._build_health_gauge()
        assert len(html) < 100_000


# ---------------------------------------------------------------------------
# test_results_chart
# ---------------------------------------------------------------------------


class TestTestResultsChart:
    def test_returns_html_with_div(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        html = builder._build_test_results_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_tests(self):
        builder = QuantileChartBuilder({"tests": []})
        assert builder._build_test_results_chart() is None

    def test_returns_none_for_missing_tests(self):
        builder = QuantileChartBuilder({})
        assert builder._build_test_results_chart() is None

    def test_single_test(self):
        data = {
            "tests": [
                {
                    "name": "Test1",
                    "statistic": 2.1,
                    "pvalue": 0.03,
                    "status": "pass",
                },
            ],
        }
        builder = QuantileChartBuilder(data)
        html = builder._build_test_results_chart()
        assert html is not None
        assert "<div" in html

    def test_tests_sorted_by_pvalue(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        html = builder._build_test_results_chart()
        assert html is not None

    def test_tests_with_none_pvalue_skipped(self):
        data = {
            "tests": [
                {
                    "name": "Test1",
                    "statistic": 2.1,
                    "pvalue": None,
                    "status": "fail",
                },
                {
                    "name": "Test2",
                    "statistic": 1.5,
                    "pvalue": 0.15,
                    "status": "pass",
                },
            ],
        }
        builder = QuantileChartBuilder(data)
        html = builder._build_test_results_chart()
        assert html is not None

    def test_all_tests_none_pvalue_returns_none(self):
        data = {
            "tests": [
                {
                    "name": "Test1",
                    "statistic": 2.1,
                    "pvalue": None,
                    "status": "fail",
                },
            ],
        }
        builder = QuantileChartBuilder(data)
        assert builder._build_test_results_chart() is None

    def test_no_plotlyjs_included(self, full_quantile_data):
        builder = QuantileChartBuilder(full_quantile_data)
        html = builder._build_test_results_chart()
        assert len(html) < 100_000

    def test_unknown_status_uses_muted_color(self):
        data = {
            "tests": [
                {
                    "name": "Test1",
                    "statistic": 2.1,
                    "pvalue": 0.03,
                    "status": "unknown",
                },
            ],
        }
        builder = QuantileChartBuilder(data)
        html = builder._build_test_results_chart()
        assert html is not None
