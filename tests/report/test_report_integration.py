"""
Integration tests for the full report pipeline.

Tests object-based reports (SFA, VAR, Quantile), refactored templates
(Validation, Residuals, Comparison), and HTMLExporter.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from panelbox.report import ReportManager
from panelbox.report.exporters import HTMLExporter
from panelbox.report.transformers import (
    QuantileTransformer,
    SFATransformer,
    VARTransformer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create a ReportManager instance."""
    return ReportManager()


@pytest.fixture
def sfa_mock_result():
    """Mock SFResult object with required attributes."""
    params = pd.Series(
        {"const": 1.0, "x1": 0.5, "x2": -0.3, "sigma_v": 0.2, "sigma_u": 0.4},
    )
    se = pd.Series(
        {"const": 0.2, "x1": 0.1, "x2": 0.15, "sigma_v": 0.05, "sigma_u": 0.08},
    )
    tvalues = params / se
    pvalues = pd.Series(
        {"const": 0.001, "x1": 0.001, "x2": 0.045, "sigma_v": 0.001, "sigma_u": 0.001},
    )
    efficiency = pd.Series([0.85, 0.90, 0.75, 0.92, 0.88])

    return SimpleNamespace(
        frontier_type="production",
        distribution="half-normal",
        nobs=200,
        n_entities=20,
        n_periods=10,
        converged=True,
        nparams=5,
        params=params,
        se=se,
        tvalues=tvalues,
        pvalues=pvalues,
        sigma_v=0.2,
        sigma_u=0.4,
        sigma=0.45,
        sigma_sq=0.2025,
        lambda_param=2.0,
        gamma=0.8,
        efficiency_scores=efficiency,
        loglik=-120.5,
        aic=251.0,
        bic=268.3,
    )


@pytest.fixture
def var_mock_result():
    """Mock PanelVARResult object with required attributes."""
    return SimpleNamespace(
        K=2,
        p=1,
        N=50,
        n_obs=450,
        method="ols",
        cov_type="robust",
        endog_names=["gdp", "investment"],
        exog_names=["L1.gdp", "L1.investment", "const"],
        params_by_eq=[
            [0.8, 0.2, 0.5],
            [0.1, 0.6, 0.3],
        ],
        std_errors_by_eq=[
            [0.05, 0.1, 0.15],
            [0.08, 0.04, 0.12],
        ],
        aic=350.2,
        bic=370.5,
        hqic=358.1,
        loglik=-168.1,
        max_eigenvalue_modulus=0.85,
        stability_margin=0.15,
    )


@pytest.fixture
def quantile_mock_report():
    """Mock DiagnosticReport object with diagnostics."""
    diag1 = SimpleNamespace(
        test_name="Heteroscedasticity",
        statistic=12.5,
        p_value=0.001,
        status="fail",
        message="Evidence of heteroscedasticity",
        recommendation="Use robust standard errors",
    )
    diag2 = SimpleNamespace(
        test_name="Normality",
        statistic=3.2,
        p_value=0.07,
        status="warning",
        message="Marginal normality",
        recommendation="Consider quantile regression",
    )
    diag3 = SimpleNamespace(
        test_name="Stationarity",
        statistic=8.5,
        p_value=0.5,
        status="pass",
        message="Data appears stationary",
        recommendation=None,
    )
    return SimpleNamespace(
        health_score=0.65,
        health_status="fair",
        diagnostics=[diag1, diag2, diag3],
    )


@pytest.fixture
def validation_data():
    """Minimal validation report data."""
    return {
        "model_info": {
            "model_type": "Fixed Effects",
            "nobs": 500,
            "n_entities": 50,
            "n_periods": 10,
        },
        "summary": {
            "total_tests": 5,
            "total_passed": 4,
            "total_failed": 1,
            "pass_rate": 80.0,
            "pass_rate_formatted": "80.0%",
            "has_issues": True,
            "overall_status": "warning",
            "status_message": "Some tests failed",
        },
        "tests": [
            {
                "name": "Stationarity Test",
                "category": "panel",
                "statistic": 5.2,
                "pvalue": 0.022,
                "passed": True,
                "description": "Panel unit root test",
            },
        ],
        "recommendations": ["Check for unit roots"],
    }


@pytest.fixture
def residual_data():
    """Minimal residual diagnostics data."""
    return {
        "model_info": {
            "estimator": "Fixed Effects",
            "nobs": 500,
            "n_entities": 50,
            "n_periods": 10,
        },
        "residual_charts": {
            "qq_plot": "<div>QQ Plot placeholder</div>",
            "residual_vs_fitted": "<div>Residual vs Fitted placeholder</div>",
        },
    }


@pytest.fixture
def comparison_data():
    """Minimal model comparison data."""
    return {
        "models_info": [
            {
                "name": "Pooled OLS",
                "estimator": "PooledOLS",
                "nobs": 500,
                "r_squared": 0.65,
                "aic": 2500,
                "bic": 2550,
            },
            {
                "name": "Fixed Effects",
                "estimator": "PanelOLS",
                "nobs": 500,
                "r_squared": 0.78,
                "aic": 2300,
                "bic": 2400,
            },
        ],
        "best_model_aic": "Fixed Effects",
        "best_model_bic": "Fixed Effects",
        "comparison_charts": {
            "coefficient_comparison": "<div>Coefficient comparison placeholder</div>",
            "fit_comparison": "<div>Fit comparison placeholder</div>",
        },
    }


# ===========================================================================
# Etapa 3: SFA Report Pipeline
# ===========================================================================


class TestSFAReportPipeline:
    """Tests for the SFA report generation pipeline."""

    def test_sfa_transform_and_render(self, manager, sfa_mock_result):
        """Full pipeline: SFResult mock -> SFATransformer -> generate_sfa_report."""
        data = SFATransformer(sfa_mock_result).transform()
        html = manager.generate_sfa_report(data, title="SFA Analysis")
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_sfa_report_contains_title(self, manager, sfa_mock_result):
        """SFA report contains the specified title."""
        data = SFATransformer(sfa_mock_result).transform()
        html = manager.generate_sfa_report(data, title="Production Frontier")
        assert "Production Frontier" in html

    def test_sfa_report_contains_coefficients(self, manager, sfa_mock_result):
        """SFA report includes coefficient names (not variance params)."""
        data = SFATransformer(sfa_mock_result).transform()
        html = manager.generate_sfa_report(data)
        assert "const" in html
        assert "x1" in html
        assert "x2" in html

    def test_sfa_report_contains_css_inline(self, manager, sfa_mock_result):
        """SFA report embeds CSS inline."""
        data = SFATransformer(sfa_mock_result).transform()
        html = manager.generate_sfa_report(data)
        assert "<style>" in html

    def test_sfa_report_contains_branding(self, manager, sfa_mock_result):
        """SFA report includes PanelBox branding."""
        data = SFATransformer(sfa_mock_result).transform()
        html = manager.generate_sfa_report(data)
        assert "PanelBox" in html

    def test_sfa_report_self_contained(self, manager, sfa_mock_result):
        """SFA report is self-contained."""
        data = SFATransformer(sfa_mock_result).transform()
        html = manager.generate_sfa_report(data)
        assert '<link rel="stylesheet"' not in html


# ===========================================================================
# Etapa 3: VAR Report Pipeline
# ===========================================================================


class TestVARReportPipeline:
    """Tests for the Panel VAR report generation pipeline."""

    def test_var_transform_and_render(self, manager, var_mock_result):
        """Full pipeline: PanelVARResult mock -> VARTransformer -> generate_var_report."""
        data = VARTransformer(var_mock_result).transform()
        html = manager.generate_var_report(data, title="Panel VAR Analysis")
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_var_report_contains_title(self, manager, var_mock_result):
        """VAR report contains the specified title."""
        data = VARTransformer(var_mock_result).transform()
        html = manager.generate_var_report(data, title="GDP-Investment VAR")
        assert "GDP-Investment VAR" in html

    def test_var_report_contains_equations(self, manager, var_mock_result):
        """VAR report includes equation names."""
        data = VARTransformer(var_mock_result).transform()
        html = manager.generate_var_report(data)
        assert "gdp" in html
        assert "investment" in html

    def test_var_report_contains_css_inline(self, manager, var_mock_result):
        """VAR report embeds CSS inline."""
        data = VARTransformer(var_mock_result).transform()
        html = manager.generate_var_report(data)
        assert "<style>" in html

    def test_var_report_contains_branding(self, manager, var_mock_result):
        """VAR report includes PanelBox branding."""
        data = VARTransformer(var_mock_result).transform()
        html = manager.generate_var_report(data)
        assert "PanelBox" in html

    def test_var_report_self_contained(self, manager, var_mock_result):
        """VAR report is self-contained."""
        data = VARTransformer(var_mock_result).transform()
        html = manager.generate_var_report(data)
        assert '<link rel="stylesheet"' not in html


# ===========================================================================
# Etapa 3: Quantile Report Pipeline
# ===========================================================================


class TestQuantileReportPipeline:
    """Tests for the Quantile diagnostics report generation pipeline."""

    def test_quantile_transform_and_render(self, manager, quantile_mock_report):
        """Full pipeline: DiagnosticReport mock -> QuantileTransformer -> report."""
        data = QuantileTransformer(quantile_mock_report).transform()
        html = manager.generate_quantile_report(data, title="Quantile Diagnostics")
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_quantile_report_contains_title(self, manager, quantile_mock_report):
        """Quantile report contains the specified title."""
        data = QuantileTransformer(quantile_mock_report).transform()
        html = manager.generate_quantile_report(data, title="Diagnostics Report")
        assert "Diagnostics Report" in html

    def test_quantile_report_contains_tests(self, manager, quantile_mock_report):
        """Quantile report includes test names."""
        data = QuantileTransformer(quantile_mock_report).transform()
        html = manager.generate_quantile_report(data)
        assert "Heteroscedasticity" in html
        assert "Normality" in html

    def test_quantile_report_contains_css_inline(self, manager, quantile_mock_report):
        """Quantile report embeds CSS inline."""
        data = QuantileTransformer(quantile_mock_report).transform()
        html = manager.generate_quantile_report(data)
        assert "<style>" in html

    def test_quantile_report_includes_plotly(self, manager, quantile_mock_report):
        """Quantile report includes Plotly CDN for interactive charts."""
        data = QuantileTransformer(quantile_mock_report).transform()
        html = manager.generate_quantile_report(data)
        assert "plotly-2.27.0" in html

    def test_quantile_report_contains_branding(self, manager, quantile_mock_report):
        """Quantile report includes PanelBox branding."""
        data = QuantileTransformer(quantile_mock_report).transform()
        html = manager.generate_quantile_report(data)
        assert "PanelBox" in html


# ===========================================================================
# Etapa 4: Validation Report (Refactored Template)
# ===========================================================================


class TestValidationReportPipeline:
    """Tests for the Validation report with refactored base_v2 template."""

    def test_generate_validation_report_returns_html(self, manager, validation_data):
        """generate_validation_report returns valid HTML."""
        html = manager.generate_validation_report(validation_data)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_validation_report_default_title(self, manager, validation_data):
        """Validation report uses default title."""
        html = manager.generate_validation_report(validation_data)
        assert "Validation" in html

    def test_validation_report_custom_title(self, manager, validation_data):
        """Validation report uses custom title."""
        html = manager.generate_validation_report(
            validation_data,
            title="My Validation",
            subtitle="Panel checks",
        )
        assert "My Validation" in html
        assert "Panel checks" in html

    def test_validation_report_contains_css_inline(self, manager, validation_data):
        """Validation report embeds CSS inline."""
        html = manager.generate_validation_report(validation_data)
        assert "<style>" in html

    def test_validation_report_contains_branding(self, manager, validation_data):
        """Validation report includes PanelBox branding."""
        html = manager.generate_validation_report(validation_data)
        assert "PanelBox" in html

    def test_validation_report_includes_plotly(self, manager, validation_data):
        """Validation report includes Plotly (interactive mode)."""
        html = manager.generate_validation_report(validation_data, interactive=True)
        assert "plotly" in html.lower()


# ===========================================================================
# Etapa 4: Residual Report (Refactored Template)
# ===========================================================================


class TestResidualReportPipeline:
    """Tests for the Residual diagnostics report with refactored template."""

    def test_generate_residual_report_returns_html(self, manager, residual_data):
        """generate_residual_report returns valid HTML."""
        html = manager.generate_residual_report(residual_data)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_residual_report_default_title(self, manager, residual_data):
        """Residual report uses default title."""
        html = manager.generate_residual_report(residual_data)
        assert "Residual" in html

    def test_residual_report_custom_title(self, manager, residual_data):
        """Residual report uses custom title."""
        html = manager.generate_residual_report(
            residual_data,
            title="Residual Analysis",
            subtitle="FE model diagnostics",
        )
        assert "Residual Analysis" in html
        assert "FE model diagnostics" in html

    def test_residual_report_contains_css_inline(self, manager, residual_data):
        """Residual report embeds CSS inline."""
        html = manager.generate_residual_report(residual_data)
        assert "<style>" in html

    def test_residual_report_contains_branding(self, manager, residual_data):
        """Residual report includes PanelBox branding."""
        html = manager.generate_residual_report(residual_data)
        assert "PanelBox" in html

    def test_residual_report_contains_charts(self, manager, residual_data):
        """Residual report includes chart placeholders."""
        html = manager.generate_residual_report(residual_data)
        assert "QQ Plot placeholder" in html or "Residual" in html


# ===========================================================================
# Etapa 4: Comparison Report (Refactored Template)
# ===========================================================================


class TestComparisonReportPipeline:
    """Tests for the Model Comparison report with refactored template."""

    def test_generate_comparison_report_returns_html(self, manager, comparison_data):
        """generate_comparison_report returns valid HTML."""
        html = manager.generate_comparison_report(comparison_data)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_comparison_report_default_title(self, manager, comparison_data):
        """Comparison report uses default title."""
        html = manager.generate_comparison_report(comparison_data)
        assert "Comparison" in html

    def test_comparison_report_custom_title(self, manager, comparison_data):
        """Comparison report uses custom title."""
        html = manager.generate_comparison_report(
            comparison_data,
            title="Model Selection",
            subtitle="OLS vs FE",
        )
        assert "Model Selection" in html
        assert "OLS vs FE" in html

    def test_comparison_report_contains_css_inline(self, manager, comparison_data):
        """Comparison report embeds CSS inline."""
        html = manager.generate_comparison_report(comparison_data)
        assert "<style>" in html

    def test_comparison_report_contains_branding(self, manager, comparison_data):
        """Comparison report includes PanelBox branding."""
        html = manager.generate_comparison_report(comparison_data)
        assert "PanelBox" in html

    def test_comparison_report_includes_plotly(self, manager, comparison_data):
        """Comparison report includes Plotly (interactive mode)."""
        html = manager.generate_comparison_report(comparison_data, interactive=True)
        assert "plotly" in html.lower()


# ===========================================================================
# Etapa 5: HTMLExporter Tests
# ===========================================================================


class TestHTMLExporter:
    """Tests for HTMLExporter functionality."""

    def test_exporter_initializes(self):
        """HTMLExporter initializes without errors."""
        exporter = HTMLExporter()
        assert exporter is not None

    def test_exporter_repr(self):
        """HTMLExporter has a useful repr."""
        exporter = HTMLExporter()
        r = repr(exporter)
        assert "HTMLExporter" in r

    def test_export_saves_file(self, tmp_path, manager):
        """export() saves HTML content to file."""
        exporter = HTMLExporter()
        html = "<html><body>Test</body></html>"
        output = tmp_path / "test_report.html"
        result_path = exporter.export(html, output)
        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "Test" in content

    def test_export_with_metadata(self, tmp_path):
        """export() adds metadata comment."""
        exporter = HTMLExporter()
        html = "<!DOCTYPE html><html><body>Test</body></html>"
        output = tmp_path / "meta_report.html"
        result_path = exporter.export(html, output, add_metadata=True)
        content = result_path.read_text(encoding="utf-8")
        assert "PanelBox HTML Export" in content

    def test_export_without_metadata(self, tmp_path):
        """export() skips metadata when add_metadata=False."""
        exporter = HTMLExporter()
        html = "<html><body>Test</body></html>"
        output = tmp_path / "no_meta_report.html"
        result_path = exporter.export(html, output, add_metadata=False)
        content = result_path.read_text(encoding="utf-8")
        assert "PanelBox HTML Export" not in content

    def test_export_raises_on_existing_file(self, tmp_path):
        """export() raises FileExistsError if file exists and overwrite=False."""
        exporter = HTMLExporter()
        output = tmp_path / "existing.html"
        output.write_text("existing content")
        with pytest.raises(FileExistsError):
            exporter.export("<html></html>", output, overwrite=False)

    def test_export_overwrites_existing_file(self, tmp_path):
        """export() overwrites file when overwrite=True."""
        exporter = HTMLExporter()
        output = tmp_path / "overwrite.html"
        output.write_text("old content")
        exporter.export("<html>new</html>", output, overwrite=True)
        content = output.read_text(encoding="utf-8")
        assert "new" in content

    def test_export_creates_parent_dirs(self, tmp_path):
        """export() creates parent directories if they don't exist."""
        exporter = HTMLExporter()
        output = tmp_path / "subdir" / "deep" / "report.html"
        exporter.export("<html></html>", output)
        assert output.exists()

    def test_export_multiple(self, tmp_path):
        """export_multiple() saves multiple reports to a directory."""
        exporter = HTMLExporter()
        reports = {
            "report_a.html": "<html><body>Report A</body></html>",
            "report_b.html": "<html><body>Report B</body></html>",
        }
        result = exporter.export_multiple(reports, tmp_path / "multi")
        assert len(result) == 2
        for _filename, path in result.items():
            assert path.exists()

    def test_export_with_index(self, tmp_path):
        """export_with_index() creates reports plus an index page."""
        exporter = HTMLExporter()
        reports = {
            "Validation Report": "<html><body>Validation</body></html>",
            "Regression Report": "<html><body>Regression</body></html>",
        }
        result = exporter.export_with_index(reports, tmp_path / "indexed", index_title="My Reports")
        # Should include individual reports and index
        assert "_index" in result
        index_path = result["_index"]
        assert index_path.exists()
        index_content = index_path.read_text(encoding="utf-8")
        assert "My Reports" in index_content
        assert "Validation Report" in index_content
        assert "Regression Report" in index_content

    def test_get_file_size(self):
        """get_file_size() returns correct size estimates."""
        exporter = HTMLExporter()
        html = "A" * 1024  # 1 KB
        sizes = exporter.get_file_size(html)
        assert sizes["bytes"] == 1024
        assert abs(sizes["kb"] - 1.0) < 0.01

    def test_export_real_report(self, tmp_path, manager):
        """export() works with a real generated report."""
        from panelbox.report.transformers import GMMTransformer

        raw_data = {
            "model_type": "System GMM",
            "nobs": 100,
            "n_groups": 20,
            "n_instruments": 15,
            "two_step": False,
            "coefficients": [
                {"name": "L.y", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001},
            ],
            "hansen_test": {"statistic": 10.0, "pvalue": 0.6, "df": 10},
            "ar_tests": {
                "ar1": {"statistic": -2.0, "pvalue": 0.04},
                "ar2": {"statistic": 0.5, "pvalue": 0.6},
            },
        }
        transformed = GMMTransformer(raw_data).transform()
        html = manager.generate_gmm_report(transformed, title="Export Test")

        exporter = HTMLExporter()
        output = tmp_path / "gmm_export.html"
        result_path = exporter.export(html, output)
        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "Export Test" in content
        assert "<!DOCTYPE html>" in content
