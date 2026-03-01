"""Tests for DiscreteChartBuilder."""

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.discrete_charts import DiscreteChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_discrete_data():
    """Complete Discrete transformer output for testing."""
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
            {
                "name": "x3",
                "coef": 0.10,
                "se": 0.40,
                "zstat": 0.25,
                "pvalue": 0.803,
                "stars": "",
            },
        ],
        "fit_statistics": {
            "loglikelihood": -72.15,
            "aic": 150.30,
            "bic": 165.70,
            "pseudo_r_squared": 0.35,
        },
        "classification": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
        },
        "model_info": {
            "model_type": "Logit",
            "converged": True,
            "n_iter": 5,
        },
    }


@pytest.fixture
def minimal_discrete_data():
    """Minimal data with only coefficients."""
    return {
        "coefficients": [
            {
                "name": "x1",
                "coef": 0.5,
                "se": 0.1,
                "zstat": 5.0,
                "pvalue": 0.001,
                "stars": "***",
            },
        ],
    }


@pytest.fixture
def no_classification_data():
    """Discrete data without classification (e.g. Ordered Probit)."""
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
        ],
        "fit_statistics": {
            "aic": 150.30,
            "bic": 165.70,
        },
        "classification": None,
    }


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------


class TestBuildAll:
    def test_full_data_returns_three_charts(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        charts = builder.build_all()
        assert set(charts.keys()) == {
            "coefficient_plot",
            "classification_chart",
            "ic_chart",
        }

    def test_empty_data_returns_empty_dict(self):
        builder = DiscreteChartBuilder({})
        assert builder.build_all() == {}

    def test_partial_data_returns_available_charts(self, minimal_discrete_data):
        builder = DiscreteChartBuilder(minimal_discrete_data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        assert "classification_chart" not in charts
        assert "ic_chart" not in charts

    def test_no_classification_excludes_chart(self, no_classification_data):
        builder = DiscreteChartBuilder(no_classification_data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        assert "ic_chart" in charts
        assert "classification_chart" not in charts

    def test_all_values_are_html_strings(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        charts = builder.build_all()
        for name, html in charts.items():
            assert isinstance(html, str), f"{name} is not a string"
            assert "<div" in html, f"{name} missing <div> element"


# ---------------------------------------------------------------------------
# coefficient_plot
# ---------------------------------------------------------------------------


class TestCoefficientPlot:
    def test_returns_html_with_div(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_coefficients(self):
        builder = DiscreteChartBuilder({"coefficients": []})
        assert builder._build_coefficient_plot() is None

    def test_returns_none_for_missing_coefficients(self):
        builder = DiscreteChartBuilder({})
        assert builder._build_coefficient_plot() is None

    def test_no_plotlyjs_included(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        html = builder._build_coefficient_plot()
        assert len(html) < 100_000

    def test_single_coefficient(self):
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
        }
        builder = DiscreteChartBuilder(data)
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
                    "zstat": 2.0,
                    "pvalue": 0.046,
                    "stars": "**",
                    "ci_lower": 0.1,
                    "ci_upper": 1.9,
                },
            ],
        }
        builder = DiscreteChartBuilder(data)
        html = builder._build_coefficient_plot()
        assert html is not None


# ---------------------------------------------------------------------------
# classification_chart
# ---------------------------------------------------------------------------


class TestClassificationChart:
    def test_returns_html_with_div(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        html = builder._build_classification_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_when_classification_is_none(self):
        builder = DiscreteChartBuilder({"classification": None})
        assert builder._build_classification_chart() is None

    def test_returns_none_when_classification_missing(self):
        builder = DiscreteChartBuilder({})
        assert builder._build_classification_chart() is None

    def test_returns_none_for_empty_classification(self):
        builder = DiscreteChartBuilder({"classification": {}})
        assert builder._build_classification_chart() is None

    def test_partial_classification_metrics(self):
        data = {
            "classification": {
                "accuracy": 0.85,
                "f1_score": 0.80,
            },
        }
        builder = DiscreteChartBuilder(data)
        html = builder._build_classification_chart()
        assert html is not None
        assert "<div" in html

    def test_no_plotlyjs_included(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        html = builder._build_classification_chart()
        assert len(html) < 100_000


# ---------------------------------------------------------------------------
# ic_chart
# ---------------------------------------------------------------------------


class TestICChart:
    def test_returns_html_with_div(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        html = builder._build_ic_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_fit_statistics(self):
        builder = DiscreteChartBuilder({"fit_statistics": {}})
        assert builder._build_ic_chart() is None

    def test_returns_none_for_missing_fit_statistics(self):
        builder = DiscreteChartBuilder({})
        assert builder._build_ic_chart() is None

    def test_only_aic(self):
        data = {"fit_statistics": {"aic": 150.3}}
        builder = DiscreteChartBuilder(data)
        html = builder._build_ic_chart()
        assert html is not None
        assert "<div" in html

    def test_only_bic(self):
        data = {"fit_statistics": {"bic": 165.7}}
        builder = DiscreteChartBuilder(data)
        html = builder._build_ic_chart()
        assert html is not None
        assert "<div" in html

    def test_fit_stats_without_aic_bic(self):
        data = {"fit_statistics": {"loglikelihood": -72.15, "pseudo_r_squared": 0.35}}
        builder = DiscreteChartBuilder(data)
        assert builder._build_ic_chart() is None

    def test_no_plotlyjs_included(self, full_discrete_data):
        builder = DiscreteChartBuilder(full_discrete_data)
        html = builder._build_ic_chart()
        assert len(html) < 100_000
