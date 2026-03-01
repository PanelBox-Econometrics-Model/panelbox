"""
Tests for report/validation_transformer.py and report/report_manager.py.

Targets coverage gaps in ValidationTransformer and ReportManager.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from unittest.mock import patch

import pytest

from panelbox.report.report_manager import ReportManager
from panelbox.report.validation_transformer import ValidationTransformer


# ---------------------------------------------------------------------------
# Mock data structures
# ---------------------------------------------------------------------------
@dataclass
class MockTestResult:
    """Mimics a validation test result object."""

    statistic: float
    pvalue: float
    df: int = 1
    reject_null: bool = False
    conclusion: str = ""
    metadata: dict = field(default_factory=dict)
    alpha: float = 0.05


@dataclass
class MockValidationReport:
    """Mimics a ValidationReport object."""

    model_info: dict
    specification_tests: dict
    serial_tests: dict
    het_tests: dict
    cd_tests: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_report(
    model_info=None,
    spec_tests=None,
    serial_tests=None,
    het_tests=None,
    cd_tests=None,
):
    """Build a MockValidationReport with sensible defaults."""
    return MockValidationReport(
        model_info=model_info or {"estimator": "FixedEffects"},
        specification_tests=spec_tests or {},
        serial_tests=serial_tests or {},
        het_tests=het_tests or {},
        cd_tests=cd_tests or {},
    )


def _passing_test(stat=1.5, pval=0.45):
    return MockTestResult(statistic=stat, pvalue=pval, reject_null=False)


def _failing_test(stat=12.0, pval=0.001):
    return MockTestResult(
        statistic=stat,
        pvalue=pval,
        reject_null=True,
        conclusion="Reject H0",
    )


# ===================================================================
# ValidationTransformer tests
# ===================================================================


class TestTransformModelInfo:
    """Lines 105-113: branches on nobs / n_entities / n_periods."""

    def test_model_info_no_nobs(self):
        """model_info without 'nobs' -- line 105 False branch."""
        report = _make_report(model_info={"estimator": "FE"})
        t = ValidationTransformer(report)
        info = t._transform_model_info()
        assert "nobs_formatted" not in info

    def test_model_info_no_entities(self):
        """model_info without 'n_entities' -- line 108 False branch."""
        report = _make_report(model_info={"nobs": 500})
        t = ValidationTransformer(report)
        info = t._transform_model_info()
        assert "n_entities_formatted" not in info
        assert info["nobs_formatted"] == "500"

    def test_model_info_no_periods(self):
        """model_info without 'n_periods' -- line 111 False branch."""
        report = _make_report(model_info={"nobs": 1000, "n_entities": 50})
        t = ValidationTransformer(report)
        info = t._transform_model_info()
        assert "n_periods_formatted" not in info

    def test_model_info_periods_none(self):
        """model_info with n_periods=None -- line 111 falsy branch."""
        report = _make_report(model_info={"nobs": 1000, "n_entities": 50, "n_periods": None})
        t = ValidationTransformer(report)
        info = t._transform_model_info()
        assert "n_periods_formatted" not in info

    def test_model_info_all_fields_present(self):
        """All three formatting branches taken."""
        report = _make_report(model_info={"nobs": 10000, "n_entities": 200, "n_periods": 5})
        t = ValidationTransformer(report)
        info = t._transform_model_info()
        assert info["nobs_formatted"] == "10,000"
        assert info["n_entities_formatted"] == "200"
        assert info["n_periods_formatted"] == "5"


class TestComputeSummary:
    """Lines 207-219: overall_status branches."""

    def test_summary_excellent(self):
        """0 failed -> 'excellent'."""
        report = _make_report(
            spec_tests={"hausman": _passing_test()},
            serial_tests={"ar1": _passing_test()},
        )
        t = ValidationTransformer(report)
        summary = t._compute_summary()
        assert summary["overall_status"] == "excellent"
        assert summary["total_failed"] == 0
        assert summary["status_message"] == "All tests passed"

    def test_summary_good(self):
        """1-2 failed -> 'good'."""
        report = _make_report(
            spec_tests={"hausman": _failing_test()},
            serial_tests={"ar1": _passing_test()},
            het_tests={"bp": _failing_test()},
        )
        t = ValidationTransformer(report)
        summary = t._compute_summary()
        assert summary["overall_status"] == "good"
        assert summary["total_failed"] == 2

    def test_summary_warning(self):
        """3-4 failed -> 'warning'."""
        report = _make_report(
            spec_tests={"hausman": _failing_test()},
            serial_tests={"ar1": _failing_test(), "ar2": _failing_test()},
            het_tests={"bp": _passing_test()},
            cd_tests={"pesaran": _failing_test()},
        )
        t = ValidationTransformer(report)
        summary = t._compute_summary()
        assert summary["overall_status"] == "warning"
        assert summary["total_failed"] == 4

    def test_summary_critical(self):
        """5+ failed -> 'critical'."""
        report = _make_report(
            spec_tests={"hausman": _failing_test(), "reset": _failing_test()},
            serial_tests={"ar1": _failing_test(), "ar2": _failing_test()},
            het_tests={"bp": _failing_test()},
            cd_tests={"pesaran": _failing_test()},
        )
        t = ValidationTransformer(report)
        summary = t._compute_summary()
        assert summary["overall_status"] == "critical"
        assert summary["total_failed"] >= 5


class TestGenerateRecommendations:
    """Lines 248-326: recommendation branches for serial / het / cd / spec."""

    def test_recommendations_serial(self):
        """Serial tests that reject null -> serial recommendation."""
        report = _make_report(serial_tests={"ar1": _failing_test()})
        t = ValidationTransformer(report)
        recs = t._generate_recommendations()
        assert len(recs) == 1
        assert recs[0]["category"] == "Serial Correlation"
        assert recs[0]["severity"] == "high"

    def test_recommendations_het(self):
        """Het tests that reject null -> heteroskedasticity recommendation."""
        report = _make_report(het_tests={"bp": _failing_test()})
        t = ValidationTransformer(report)
        recs = t._generate_recommendations()
        assert len(recs) == 1
        assert recs[0]["category"] == "Heteroskedasticity"
        assert recs[0]["severity"] == "medium"

    def test_recommendations_cd(self):
        """CD tests that reject null -> cross-sectional dependence recommendation."""
        report = _make_report(cd_tests={"pesaran": _failing_test()})
        t = ValidationTransformer(report)
        recs = t._generate_recommendations()
        assert len(recs) == 1
        assert recs[0]["category"] == "Cross-Sectional Dependence"
        assert recs[0]["severity"] == "high"

    def test_recommendations_spec(self):
        """Specification tests that reject null -> spec recommendation."""
        report = _make_report(spec_tests={"hausman": _failing_test()})
        t = ValidationTransformer(report)
        recs = t._generate_recommendations()
        assert len(recs) == 1
        assert recs[0]["category"] == "Model Specification"
        assert recs[0]["severity"] == "critical"

    def test_recommendations_sorted_by_severity(self):
        """Multiple categories -> sorted critical > high > medium."""
        report = _make_report(
            spec_tests={"hausman": _failing_test()},
            serial_tests={"ar1": _failing_test()},
            het_tests={"bp": _failing_test()},
        )
        t = ValidationTransformer(report)
        recs = t._generate_recommendations()
        severities = [r["severity"] for r in recs]
        assert severities == ["critical", "high", "medium"]

    def test_no_recommendations_when_all_pass(self):
        """All tests pass -> empty recommendations."""
        report = _make_report(
            spec_tests={"hausman": _passing_test()},
            serial_tests={"ar1": _passing_test()},
        )
        t = ValidationTransformer(report)
        recs = t._generate_recommendations()
        assert recs == []


class TestPrepareChartData:
    """Lines 354-369 and 450: chart data branches."""

    def test_prepare_chart_data_legacy(self):
        """use_new_visualization=False -> _prepare_chart_data_legacy()."""
        report = _make_report(
            spec_tests={"hausman": _passing_test()},
            serial_tests={"ar1": _failing_test()},
        )
        t = ValidationTransformer(report)
        charts = t._prepare_chart_data(use_new_visualization=False)
        assert "test_overview" in charts
        assert "pvalue_distribution" in charts
        assert "test_statistics" in charts
        # Check structure
        assert "categories" in charts["test_overview"]
        assert "passed" in charts["test_overview"]
        assert "failed" in charts["test_overview"]

    def test_prepare_chart_data_no_viz(self):
        """create_validation_charts is None -> warning + fallback to legacy."""
        report = _make_report(
            spec_tests={"hausman": _passing_test()},
        )
        t = ValidationTransformer(report)
        with (
            patch(
                "panelbox.report.validation_transformer.create_validation_charts",
                None,
            ),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")
            charts = t._prepare_chart_data(use_new_visualization=True)
            # Should have issued a UserWarning
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "not available" in str(user_warnings[0].message).lower()
        # Should still return legacy chart data
        assert "test_overview" in charts

    def test_empty_category(self):
        """Empty test dict -> line 450 `if not category_tests: continue`."""
        report = _make_report(
            spec_tests={},
            serial_tests={"ar1": _passing_test()},
            het_tests={},
            cd_tests={},
        )
        t = ValidationTransformer(report)
        charts = t._prepare_chart_data_legacy()
        # Only serial_tests has data -> only 1 category in overview
        assert len(charts["test_overview"]["categories"]) == 1
        assert charts["test_overview"]["categories"][0] == "Serial Correlation"


class TestSignificanceStars:
    """Line 600 and 602: _get_significance_stars branches."""

    def test_significance_stars_all_levels(self):
        stars = ValidationTransformer._get_significance_stars
        assert stars(0.0001) == "***"  # p < 0.001
        assert stars(0.005) == "**"  # 0.001 <= p < 0.01
        assert stars(0.03) == "*"  # 0.01 <= p < 0.05 (line 600)
        assert stars(0.08) == "."  # 0.05 <= p < 0.1 (line 602)
        assert stars(0.5) == ""  # p >= 0.1


class TestTransformIncludeCharts:
    """include_charts=False -> no 'charts' key in output."""

    def test_transform_include_charts_false(self):
        report = _make_report(spec_tests={"hausman": _passing_test()})
        t = ValidationTransformer(report)
        data = t.transform(include_charts=False)
        assert "charts" not in data
        # Basic keys should still be present
        assert "model_info" in data
        assert "tests" in data
        assert "summary" in data
        assert "recommendations" in data


# ===================================================================
# ReportManager tests
# ===================================================================


@pytest.fixture
def mock_report_manager():
    """Create a ReportManager with mocked template rendering."""
    with (
        patch("panelbox.report.report_manager.TemplateManager") as MockTM,
        patch("panelbox.report.report_manager.AssetManager") as MockAM,
        patch("panelbox.report.report_manager.CSSManager") as MockCSS,
    ):
        # Configure mocks
        MockTM.return_value.render_template.return_value = "<html>rendered</html>"
        MockAM.return_value.collect_js.return_value = "// js"
        MockAM.return_value.embed_plotly.return_value = "// plotly"
        MockCSS.return_value.compile_for_report_type.return_value = "/* css */"

        mgr = ReportManager()
        yield mgr


class TestGenerateValidationReport:
    def test_generate_validation_report_static(self, mock_report_manager):
        """interactive=False -> 'validation/static/index.html' (line 204)."""
        html = mock_report_manager.generate_validation_report(
            validation_data={"model_info": {}, "tests": []},
            interactive=False,
            title="Static Report",
        )
        assert html == "<html>rendered</html>"
        # Check that render_template was called with static template
        call_args = mock_report_manager.template_manager.render_template.call_args
        assert "validation/static/index.html" in call_args[0]

    def test_generate_validation_report_interactive(self, mock_report_manager):
        """interactive=True -> 'validation/interactive/index.html'."""
        html = mock_report_manager.generate_validation_report(
            validation_data={"model_info": {}, "tests": []},
            interactive=True,
        )
        assert html == "<html>rendered</html>"
        call_args = mock_report_manager.template_manager.render_template.call_args
        assert "validation/interactive/index.html" in call_args[0]


class TestGenerateRegressionReport:
    def test_generate_regression_report(self, mock_report_manager):
        """Basic call -> lines 255-263."""
        html = mock_report_manager.generate_regression_report(
            regression_data={"coefficients": [1.0, 2.0]},
            title="Regression Results",
            subtitle="FE model",
        )
        assert html == "<html>rendered</html>"
        call_args = mock_report_manager.template_manager.render_template.call_args
        assert "regression/index.html" in call_args[0]


class TestGenerateGMMReport:
    def test_generate_gmm_report(self, mock_report_manager):
        """Basic call -> lines 295-299."""
        html = mock_report_manager.generate_gmm_report(
            gmm_data={"hansen_test": {"stat": 5.0}},
            title="GMM Results",
        )
        assert html == "<html>rendered</html>"
        call_args = mock_report_manager.template_manager.render_template.call_args
        assert "gmm/index.html" in call_args[0]


class TestGenerateResidualReport:
    def test_generate_residual_report_static(self, mock_report_manager):
        """interactive=False -> residuals/static/index.html (line 357)."""
        html = mock_report_manager.generate_residual_report(
            residual_data={"model_info": {"estimator": "FE"}},
            interactive=False,
        )
        assert html == "<html>rendered</html>"
        call_args = mock_report_manager.template_manager.render_template.call_args
        assert "residuals/static/index.html" in call_args[0]

    def test_generate_residual_report_with_charts(self, mock_report_manager):
        """With residual_charts -> line 377-378 (n_diagnostics counting)."""
        residual_data = {
            "model_info": {"estimator": "FE", "nobs": 500},
            "residual_charts": {
                "histogram": "<div>chart1</div>",
                "qq_plot": "<div>chart2</div>",
                "fitted_vs_residual": "<div>chart3</div>",
            },
        }
        html = mock_report_manager.generate_residual_report(
            residual_data=residual_data,
            title="Residual Diag",
        )
        assert html == "<html>rendered</html>"
        # Verify n_diagnostics was set in context
        call_args = mock_report_manager.template_manager.render_template.call_args
        context = call_args[0][1]
        assert context["n_diagnostics"] == 3


class TestGenerateComparisonReport:
    def test_generate_comparison_report_static(self, mock_report_manager):
        """interactive=False -> comparison/static/index.html (line 475)."""
        html = mock_report_manager.generate_comparison_report(
            comparison_data={"models_info": []},
            interactive=False,
        )
        assert html == "<html>rendered</html>"
        call_args = mock_report_manager.template_manager.render_template.call_args
        assert "comparison/static/index.html" in call_args[0]

    def test_generate_comparison_report_with_charts(self, mock_report_manager):
        """With comparison_charts -> lines 497-498 (n_charts counting)."""
        comparison_data = {
            "models_info": [
                {"name": "FE", "nobs": 1000},
                {"name": "RE", "nobs": 1000},
            ],
            "comparison_charts": {
                "coeff_comparison": "<div>chart1</div>",
                "ic_comparison": "<div>chart2</div>",
            },
        }
        html = mock_report_manager.generate_comparison_report(
            comparison_data=comparison_data,
            title="Model Comparison",
        )
        assert html == "<html>rendered</html>"
        call_args = mock_report_manager.template_manager.render_template.call_args
        context = call_args[0][1]
        assert context["n_charts"] == 2


class TestGenerateReportCustomCSSJS:
    def test_generate_report_custom_css_js(self, mock_report_manager):
        """custom_css and custom_js -> lines 134-135, 148."""
        html = mock_report_manager.generate_report(
            report_type="validation",
            template="validation/interactive/index.html",
            context={"title": "Test"},
            custom_css=["custom1.css", "custom2.css"],
            custom_js=["extra1.js", "extra2.js"],
        )
        assert html == "<html>rendered</html>"
        # Check CSS was added
        css_calls = mock_report_manager.css_manager.add_custom_css.call_args_list
        assert len(css_calls) == 2
        assert css_calls[0][0][0] == "custom1.css"
        assert css_calls[1][0][0] == "custom2.css"
        # Check JS was collected with custom files included
        js_call = mock_report_manager.asset_manager.collect_js.call_args
        js_files_arg = js_call[0][0]
        assert "extra1.js" in js_files_arg
        assert "extra2.js" in js_files_arg


class TestGenerateReportNoEmbed:
    def test_generate_report_no_embed(self, mock_report_manager):
        """embed_assets=False -> non-embedded CSS/JS paths."""
        html = mock_report_manager.generate_report(
            report_type="validation",
            template="validation/interactive/index.html",
            context={"title": "Test"},
            embed_assets=False,
        )
        assert html == "<html>rendered</html>"
        # compile_for_report_type should NOT have been called
        mock_report_manager.css_manager.compile_for_report_type.assert_not_called()
        # collect_js should NOT have been called
        mock_report_manager.asset_manager.collect_js.assert_not_called()


class TestSaveReport:
    def test_save_report_overwrite_false(self, mock_report_manager, tmp_path):
        """FileExistsError when file exists and overwrite=False."""
        output = tmp_path / "report.html"
        output.write_text("<html>old</html>")
        with pytest.raises(FileExistsError, match="already exists"):
            mock_report_manager.save_report("<html>new</html>", output, overwrite=False)

    def test_save_report_overwrite_true(self, mock_report_manager, tmp_path):
        """overwrite=True succeeds even if file exists."""
        output = tmp_path / "report.html"
        output.write_text("<html>old</html>")
        result = mock_report_manager.save_report("<html>new</html>", output, overwrite=True)
        assert result == output
        assert output.read_text() == "<html>new</html>"

    def test_save_report_creates_directories(self, mock_report_manager, tmp_path):
        """Directories are created automatically."""
        output = tmp_path / "sub" / "dir" / "report.html"
        result = mock_report_manager.save_report("<html>ok</html>", output)
        assert result == output
        assert output.exists()
