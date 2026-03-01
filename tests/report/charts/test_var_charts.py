"""Tests for VARChartBuilder."""

import pytest

go = pytest.importorskip("plotly.graph_objects")

from panelbox.report.charts.var_charts import VARChartBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_var_data():
    """Complete VAR transformer output for testing."""
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
                    {
                        "name": "const",
                        "coef": 1.5,
                        "se": 0.3,
                        "tstat": 5.0,
                        "pvalue": 0.001,
                        "stars": "***",
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
                    {
                        "name": "const",
                        "coef": 0.8,
                        "se": 0.4,
                        "tstat": 2.0,
                        "pvalue": 0.045,
                        "stars": "*",
                    },
                ],
            },
        ],
        "diagnostics": {"aic": -1500.0, "bic": -1450.0, "hqic": -1480.0, "loglik": 770.0},
        "stability": {"is_stable": True, "max_eigenvalue_modulus": 0.85, "stability_margin": 0.15},
        "model_info": {"K": 2, "p": 1, "N": 50, "n_obs": 450, "endog_names": ["y1", "y2"]},
    }


@pytest.fixture
def minimal_var_data():
    """Minimal data with only one equation."""
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
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# build_all
# ---------------------------------------------------------------------------


class TestBuildAll:
    def test_full_data_returns_three_charts(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        charts = builder.build_all()
        assert set(charts.keys()) == {"stability_chart", "ic_chart", "coefficient_heatmap"}

    def test_empty_data_returns_empty_dict(self):
        builder = VARChartBuilder({})
        assert builder.build_all() == {}

    def test_partial_data_returns_available_charts(self, minimal_var_data):
        builder = VARChartBuilder(minimal_var_data)
        charts = builder.build_all()
        assert "coefficient_heatmap" in charts
        assert "stability_chart" not in charts
        assert "ic_chart" not in charts

    def test_all_values_are_html_strings(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        charts = builder.build_all()
        for name, html in charts.items():
            assert isinstance(html, str), f"{name} is not a string"
            assert "<div" in html, f"{name} missing <div> element"


# ---------------------------------------------------------------------------
# stability_chart
# ---------------------------------------------------------------------------


class TestStabilityChart:
    def test_returns_html_with_div(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        html = builder._build_stability_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_stability(self):
        builder = VARChartBuilder({})
        assert builder._build_stability_chart() is None

    def test_returns_none_for_none_stability(self):
        builder = VARChartBuilder({"stability": None})
        assert builder._build_stability_chart() is None

    def test_returns_none_for_missing_max_modulus(self):
        builder = VARChartBuilder({"stability": {"is_stable": True}})
        assert builder._build_stability_chart() is None

    def test_no_plotlyjs_included(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        html = builder._build_stability_chart()
        assert len(html) < 100_000

    def test_unstable_system(self):
        data = {
            "stability": {
                "is_stable": False,
                "max_eigenvalue_modulus": 1.15,
                "stability_margin": -0.15,
            },
        }
        builder = VARChartBuilder(data)
        html = builder._build_stability_chart()
        assert html is not None
        assert "<div" in html

    def test_without_is_stable_flag(self):
        data = {
            "stability": {"max_eigenvalue_modulus": 0.9},
        }
        builder = VARChartBuilder(data)
        html = builder._build_stability_chart()
        assert html is not None


# ---------------------------------------------------------------------------
# ic_chart
# ---------------------------------------------------------------------------


class TestICChart:
    def test_returns_html_with_div(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        html = builder._build_ic_chart()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_diagnostics(self):
        builder = VARChartBuilder({})
        assert builder._build_ic_chart() is None

    def test_returns_none_for_empty_diagnostics(self):
        builder = VARChartBuilder({"diagnostics": {}})
        assert builder._build_ic_chart() is None

    def test_partial_criteria(self):
        data = {"diagnostics": {"aic": -1500.0}}
        builder = VARChartBuilder(data)
        html = builder._build_ic_chart()
        assert html is not None
        assert "<div" in html

    def test_no_plotlyjs_included(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        html = builder._build_ic_chart()
        assert len(html) < 100_000

    def test_only_loglik_returns_none(self):
        data = {"diagnostics": {"loglik": 770.0}}
        builder = VARChartBuilder(data)
        assert builder._build_ic_chart() is None


# ---------------------------------------------------------------------------
# coefficient_heatmap
# ---------------------------------------------------------------------------


class TestCoefficientHeatmap:
    def test_returns_html_with_div(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        html = builder._build_coefficient_heatmap()
        assert html is not None
        assert "<div" in html

    def test_returns_none_for_missing_equations(self):
        builder = VARChartBuilder({})
        assert builder._build_coefficient_heatmap() is None

    def test_returns_none_for_empty_equations(self):
        builder = VARChartBuilder({"equations": []})
        assert builder._build_coefficient_heatmap() is None

    def test_no_plotlyjs_included(self, full_var_data):
        builder = VARChartBuilder(full_var_data)
        html = builder._build_coefficient_heatmap()
        assert len(html) < 100_000

    def test_single_equation(self, minimal_var_data):
        builder = VARChartBuilder(minimal_var_data)
        html = builder._build_coefficient_heatmap()
        assert html is not None
        assert "<div" in html

    def test_three_equations(self):
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
                {
                    "name": "y3",
                    "coefficients": [
                        {
                            "name": "y3.L1",
                            "coef": 0.9,
                            "se": 0.05,
                            "tstat": 18.0,
                            "pvalue": 0.001,
                            "stars": "***",
                        },
                    ],
                },
            ],
        }
        builder = VARChartBuilder(data)
        html = builder._build_coefficient_heatmap()
        assert html is not None
        assert "<div" in html

    def test_equations_with_no_coefficients(self):
        data = {
            "equations": [
                {"name": "y1", "coefficients": []},
            ],
        }
        builder = VARChartBuilder(data)
        assert builder._build_coefficient_heatmap() is None

    def test_handles_missing_coefficients_across_equations(self):
        """Some regressors only appear in certain equations."""
        data = {
            "equations": [
                {
                    "name": "y1",
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
                },
                {
                    "name": "y2",
                    "coefficients": [
                        {
                            "name": "x2",
                            "coef": 0.3,
                            "se": 0.2,
                            "tstat": 1.5,
                            "pvalue": 0.13,
                            "stars": "",
                        },
                    ],
                },
            ],
        }
        builder = VARChartBuilder(data)
        html = builder._build_coefficient_heatmap()
        assert html is not None
        assert "<div" in html
