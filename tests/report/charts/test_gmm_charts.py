"""Tests for GMMChartBuilder."""

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.gmm_charts import GMMChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_gmm_data():
    """Complete GMM transformer output for testing."""
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
            {"name": "x2", "coef": -0.05, "se": 0.10, "tstat": -0.50, "pvalue": 0.617, "stars": ""},
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
            "instrument_ratio": 0.625,
        },
    }


@pytest.fixture
def minimal_gmm_data():
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
    def test_full_data_returns_three_charts(self, full_gmm_data):
        builder = GMMChartBuilder(full_gmm_data)
        charts = builder.build_all()
        assert set(charts.keys()) == {"coefficient_plot", "diagnostic_chart", "instrument_chart"}

    def test_empty_data_returns_empty_dict(self):
        builder = GMMChartBuilder({})
        assert builder.build_all() == {}

    def test_partial_data_returns_available_charts(self, minimal_gmm_data):
        builder = GMMChartBuilder(minimal_gmm_data)
        charts = builder.build_all()
        assert "coefficient_plot" in charts
        assert "diagnostic_chart" not in charts
        assert "instrument_chart" not in charts

    def test_all_values_are_html_strings(self, full_gmm_data):
        builder = GMMChartBuilder(full_gmm_data)
        charts = builder.build_all()
        for name, html in charts.items():
            assert isinstance(html, str), f"{name} is not a string"
            assert "<div" in html, f"{name} missing <div> element"


# ---------------------------------------------------------------------------
# coefficient_plot
# ---------------------------------------------------------------------------


class TestCoefficientPlot:
    def test_returns_html_with_div(self, full_gmm_data):
        builder = GMMChartBuilder(full_gmm_data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_coefficients(self):
        builder = GMMChartBuilder({"coefficients": []})
        assert builder._build_coefficient_plot() is None

    def test_returns_none_for_missing_coefficients(self):
        builder = GMMChartBuilder({})
        assert builder._build_coefficient_plot() is None

    def test_no_plotlyjs_included(self, full_gmm_data):
        builder = GMMChartBuilder(full_gmm_data)
        html = builder._build_coefficient_plot()
        # Should not contain the plotly.js library (>3MB)
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
        builder = GMMChartBuilder(data)
        html = builder._build_coefficient_plot()
        assert html is not None
        assert "<div" in html


# ---------------------------------------------------------------------------
# diagnostic_chart
# ---------------------------------------------------------------------------


class TestDiagnosticChart:
    def test_returns_html_with_div(self, full_gmm_data):
        builder = GMMChartBuilder(full_gmm_data)
        html = builder._build_diagnostic_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_empty_diagnostics(self):
        builder = GMMChartBuilder({"diagnostics": {}})
        assert builder._build_diagnostic_chart() is None

    def test_returns_none_for_missing_diagnostics(self):
        builder = GMMChartBuilder({})
        assert builder._build_diagnostic_chart() is None

    def test_partial_diagnostics_only_hansen(self):
        data = {
            "diagnostics": {
                "hansen": {"statistic": 3.0, "pvalue": 0.55, "df": 3, "status": "PASS"},
            },
        }
        builder = GMMChartBuilder(data)
        html = builder._build_diagnostic_chart()
        assert html is not None

    def test_diagnostics_with_none_pvalues(self):
        data = {
            "diagnostics": {
                "hansen": {"statistic": 3.0, "pvalue": None},
                "ar1": {"statistic": -2.0, "pvalue": 0.01},
            },
        }
        builder = GMMChartBuilder(data)
        html = builder._build_diagnostic_chart()
        assert html is not None


# ---------------------------------------------------------------------------
# instrument_chart
# ---------------------------------------------------------------------------


class TestInstrumentChart:
    def test_returns_html_with_div(self, full_gmm_data):
        builder = GMMChartBuilder(full_gmm_data)
        html = builder._build_instrument_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_model_info(self):
        builder = GMMChartBuilder({})
        assert builder._build_instrument_chart() is None

    def test_returns_none_for_missing_instruments(self):
        builder = GMMChartBuilder({"model_info": {"n_groups": 20}})
        assert builder._build_instrument_chart() is None

    def test_returns_none_for_zero_groups(self):
        builder = GMMChartBuilder({"model_info": {"n_instruments": 10, "n_groups": 0}})
        assert builder._build_instrument_chart() is None

    def test_high_ratio(self):
        data = {"model_info": {"n_instruments": 100, "n_groups": 20}}
        builder = GMMChartBuilder(data)
        html = builder._build_instrument_chart()
        assert html is not None

    def test_low_ratio(self):
        data = {"model_info": {"n_instruments": 5, "n_groups": 50}}
        builder = GMMChartBuilder(data)
        html = builder._build_instrument_chart()
        assert html is not None
