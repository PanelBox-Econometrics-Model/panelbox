"""
Tests for ReportManager.
"""

import tempfile
from pathlib import Path

import pytest

from panelbox.report import ReportManager


class TestReportManager:
    """Test ReportManager functionality."""

    @pytest.fixture
    def report_manager(self):
        """Create ReportManager instance."""
        return ReportManager(enable_cache=False)

    @pytest.fixture
    def sample_validation_data(self):
        """Create sample validation data."""
        return {
            "model_info": {
                "model_type": "Fixed Effects",
                "formula": "y ~ x1 + x2",
                "nobs": 1000,
                "n_entities": 100,
                "n_periods": 10,
            },
            "tests": [
                {
                    "category": "Specification",
                    "name": "Hausman Test",
                    "statistic": 12.5,
                    "statistic_formatted": "12.500",
                    "pvalue": 0.014,
                    "pvalue_formatted": "0.0140",
                    "df": 2,
                    "reject_null": True,
                    "result": "REJECT",
                    "result_class": "reject",
                    "conclusion": "Reject null hypothesis",
                    "significance": "**",
                    "metadata": {},
                }
            ],
            "summary": {
                "total_tests": 1,
                "total_passed": 0,
                "total_failed": 1,
                "pass_rate": 0.0,
                "pass_rate_formatted": "0.0%",
                "failed_by_category": {
                    "specification": 1,
                    "serial": 0,
                    "heteroskedasticity": 0,
                    "cross_sectional": 0,
                },
                "overall_status": "warning",
                "status_message": "Issues detected",
                "has_issues": True,
            },
            "recommendations": [],
            "charts": {
                "test_overview": {"categories": ["Specification"], "passed": [0], "failed": [1]},
                "pvalue_distribution": {"test_names": ["Hausman Test"], "pvalues": [0.014]},
                "test_statistics": {"test_names": ["Hausman Test"], "statistics": [12.5]},
            },
        }

    def test_initialization(self, report_manager):
        """Test ReportManager initialization."""
        assert report_manager is not None
        assert report_manager.template_manager is not None
        assert report_manager.asset_manager is not None
        assert report_manager.css_manager is not None

    def test_get_info(self, report_manager):
        """Test get_info method."""
        info = report_manager.get_info()

        assert "panelbox_version" in info
        assert "template_dir" in info
        assert "asset_dir" in info
        assert "templates_cached" in info
        assert "assets_cached" in info
        assert "css_layers" in info

    def test_generate_validation_report(self, report_manager, sample_validation_data):
        """Test validation report generation."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data, interactive=True, title="Test Validation Report"
        )

        assert html is not None
        assert isinstance(html, str)
        assert len(html) > 0

        # Check for key elements
        assert "Test Validation Report" in html
        assert "Hausman Test" in html
        assert "DOCTYPE html" in html

    def test_save_report(self, report_manager, sample_validation_data):
        """Test report saving."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data, title="Test Report"
        )

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"

            # Save report
            saved_path = report_manager.save_report(html, output_path)

            assert saved_path.exists()
            assert saved_path.stat().st_size > 0

            # Read and verify
            content = saved_path.read_text(encoding="utf-8")
            assert "Test Report" in content

    def test_save_report_overwrite(self, report_manager, sample_validation_data):
        """Test report saving with overwrite."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data, title="Test Report"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_report.html"

            # Save first time
            report_manager.save_report(html, output_path)

            # Should raise error without overwrite
            with pytest.raises(FileExistsError):
                report_manager.save_report(html, output_path, overwrite=False)

            # Should succeed with overwrite
            saved_path = report_manager.save_report(html, output_path, overwrite=True)
            assert saved_path.exists()

    def test_clear_cache(self, report_manager):
        """Test cache clearing."""
        # Generate a report to populate caches
        validation_data = {
            "model_info": {"model_type": "FE"},
            "tests": [],
            "summary": {
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "pass_rate": 100.0,
                "pass_rate_formatted": "100.0%",
                "has_issues": False,
                "overall_status": "excellent",
                "status_message": "All tests passed",
            },
            "recommendations": [],
        }

        report_manager.generate_validation_report(validation_data)

        # Clear caches
        report_manager.clear_cache()

        # Verify caches are cleared
        assert len(report_manager.template_manager.template_cache) == 0
        assert len(report_manager.asset_manager.asset_cache) == 0

    def test_repr(self, report_manager):
        """Test string representation."""
        repr_str = repr(report_manager)

        assert "ReportManager" in repr_str
        assert "cache=" in repr_str
        assert "minify=" in repr_str

    def test_generate_validation_report_no_plotly(self, report_manager):
        """Test validation report without Plotly (non-interactive path)."""
        data = {
            "model_info": {"model_type": "FE"},
            "tests": [],
            "summary": {
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "pass_rate": 100.0,
                "pass_rate_formatted": "100.0%",
                "has_issues": False,
                "overall_status": "excellent",
                "status_message": "OK",
            },
            "recommendations": [],
        }
        html = report_manager.generate_report(
            report_type="validation",
            template="validation/interactive/index.html",
            context={
                "report_title": "No Plotly Report",
                **data,
                **data.get("model_info", {}),
            },
            include_plotly=False,
        )
        assert isinstance(html, str)
        assert "No Plotly Report" in html

    def test_generate_validation_report_with_subtitle(self, report_manager, sample_validation_data):
        """Test validation report with subtitle."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data,
            title="Main Title",
            subtitle="Sub Title",
        )
        assert isinstance(html, str)
        assert "Main Title" in html

    def test_generate_report_embed_assets_false(self, report_manager, sample_validation_data):
        """Test report generation with embed_assets=False."""
        html = report_manager.generate_report(
            report_type="validation",
            template="validation/interactive/index.html",
            context={
                "report_title": "Test No Embed",
                **sample_validation_data,
                **sample_validation_data.get("model_info", {}),
            },
            embed_assets=False,
            include_plotly=False,
        )
        assert isinstance(html, str)

    def test_generate_report_include_plotly_false(self, report_manager, sample_validation_data):
        """Test report generation with include_plotly=False."""
        html = report_manager.generate_report(
            report_type="validation",
            template="validation/interactive/index.html",
            context={
                "report_title": "No Plotly",
                **sample_validation_data,
                **sample_validation_data.get("model_info", {}),
            },
            embed_assets=True,
            include_plotly=False,
        )
        assert isinstance(html, str)

    def test_generate_residual_report(self, report_manager):
        """Test residual diagnostics report generation."""
        data = {
            "residual_charts": {
                "histogram": "<div>Histogram</div>",
                "qq_plot": "<div>QQ Plot</div>",
            },
            "model_info": {
                "estimator": "FixedEffects",
                "nobs": 500,
                "n_entities": 50,
                "n_periods": 10,
            },
        }
        html = report_manager.generate_residual_report(
            residual_data=data, title="Residual Diagnostics"
        )
        assert isinstance(html, str)
        assert "Residual Diagnostics" in html

    def test_generate_comparison_report(self, report_manager):
        """Test model comparison report generation."""
        data = {
            "comparison_charts": {
                "coefficients": "<div>Coef Chart</div>",
                "r_squared": "<div>R2 Chart</div>",
            },
            "models_info": [
                {
                    "name": "FE",
                    "estimator": "PanelOLS",
                    "nobs": 1000,
                    "r_squared": 0.85,
                    "aic": 2300,
                    "bic": 2400,
                },
                {
                    "name": "RE",
                    "estimator": "RandomEffects",
                    "nobs": 1000,
                    "r_squared": 0.82,
                    "aic": 2350,
                    "bic": 2420,
                },
            ],
            "best_model_aic": "FE",
            "best_model_bic": "FE",
        }
        html = report_manager.generate_comparison_report(
            comparison_data=data, title="Model Comparison"
        )
        assert isinstance(html, str)
        assert "Model Comparison" in html

    def test_prepare_context(self, report_manager):
        """Test _prepare_context adds metadata."""
        context = {"report_title": "My Report"}
        result = report_manager._prepare_context("validation", context)
        assert "panelbox_version" in result
        assert "python_version" in result
        assert "report_type" in result
        assert result["report_type"] == "validation"
        assert result["report_title"] == "My Report"
        assert "generation_date" in result

    def test_get_css_files(self, report_manager):
        """Test _get_css_files returns list."""
        css_files = report_manager._get_css_files()
        assert isinstance(css_files, list)
        assert len(css_files) > 0

    def test_get_js_files(self, report_manager):
        """Test _get_js_files returns list."""
        js_files = report_manager._get_js_files()
        assert isinstance(js_files, list)
        assert "utils.js" in js_files
        assert "tab-navigation.js" in js_files

    def test_get_js_files_with_custom(self, report_manager):
        """Test _get_js_files with custom JS files."""
        js_files = report_manager._get_js_files(custom_js=["extra.js"])
        assert "extra.js" in js_files

    def test_init_with_minify(self):
        """Test initialization with minify enabled."""
        rm = ReportManager(minify=True)
        assert rm.minify is True
