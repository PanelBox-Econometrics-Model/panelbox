"""
Integration tests for ReportManager v2 infrastructure and dict-based reports.

Tests the full pipeline: Transformer -> ReportManager -> HTML output
for GMM, Regression, and Discrete report types, plus infrastructure checks.
"""

from __future__ import annotations

import pytest

from panelbox.report import ReportManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager():
    """Create a ReportManager instance."""
    return ReportManager()


@pytest.fixture
def gmm_transformed_data():
    """Minimal transformed GMM data matching GMMTransformer output."""
    return {
        "model_info": {
            "estimator": "System GMM",
            "nobs": 1000,
            "n_groups": 100,
            "n_instruments": 50,
            "two_step": True,
            "instrument_ratio": "50/100",
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
            "hansen": {"statistic": 45.3, "pvalue": 0.35, "df": 40, "status": "pass"},
            "ar1": {"statistic": -2.5, "pvalue": 0.012},
            "ar2": {"statistic": 0.8, "pvalue": 0.42, "status": "pass"},
        },
        "summary": {"overall_status": "good", "hansen_ok": True, "ar2_ok": True},
    }


@pytest.fixture
def regression_transformed_data():
    """Minimal transformed regression data matching RegressionTransformer output."""
    return {
        "model_info": {
            "estimator": "Fixed Effects",
            "formula": "y ~ x1 + x2",
            "nobs": 500,
            "n_entities": 50,
            "n_periods": 10,
            "se_type": "clustered",
        },
        "coefficients": [
            {
                "name": "x1",
                "coef": 1.5,
                "se": 0.3,
                "tstat": 5.0,
                "pvalue": 0.001,
                "stars": "***",
                "ci_lower": 0.9,
                "ci_upper": 2.1,
            },
            {
                "name": "x2",
                "coef": -0.8,
                "se": 0.4,
                "tstat": -2.0,
                "pvalue": 0.046,
                "stars": "**",
                "ci_lower": -1.6,
                "ci_upper": 0.0,
            },
        ],
        "fit_statistics": {
            "r_squared": 0.85,
            "adj_r_squared": 0.83,
            "f_statistic": 120.5,
            "f_pvalue": 0.0001,
            "aic": 1500.3,
            "bic": 1520.7,
        },
    }


@pytest.fixture
def discrete_transformed_data():
    """Minimal transformed discrete data matching DiscreteTransformer output."""
    return {
        "model_info": {
            "model_type": "Random Effects Logit",
            "distribution": "logistic",
            "nobs": 800,
            "n_entities": 80,
            "n_periods": 10,
            "converged": True,
            "n_iter": 25,
            "se_type": "robust",
        },
        "coefficients": [
            {
                "name": "x1",
                "coef": 2.1,
                "se": 0.5,
                "zstat": 4.2,
                "pvalue": 0.001,
                "stars": "***",
                "ci_lower": 1.1,
                "ci_upper": 3.1,
            },
            {
                "name": "x2",
                "coef": -0.5,
                "se": 0.25,
                "zstat": -2.0,
                "pvalue": 0.045,
                "stars": "**",
                "ci_lower": -1.0,
                "ci_upper": 0.0,
            },
        ],
        "fit_statistics": {
            "loglikelihood": -350.5,
            "aic": 709.0,
            "bic": 730.2,
            "pseudo_r_squared": 0.25,
        },
        "classification": {
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
        },
    }


# ===========================================================================
# Etapa 1: Infrastructure Tests
# ===========================================================================


class TestReportManagerInfrastructure:
    """Tests for ReportManager infrastructure components."""

    def test_report_manager_initializes(self):
        """ReportManager initializes without errors."""
        mgr = ReportManager()
        assert mgr is not None
        assert mgr.template_manager is not None
        assert mgr.asset_manager is not None
        assert mgr.css_manager is not None

    def test_report_manager_repr(self):
        """ReportManager has a useful repr."""
        mgr = ReportManager()
        r = repr(mgr)
        assert "ReportManager" in r

    def test_report_manager_get_info(self):
        """get_info returns expected keys."""
        mgr = ReportManager()
        info = mgr.get_info()
        assert "panelbox_version" in info
        assert "template_dir" in info
        assert "asset_dir" in info
        assert "css_layers" in info
        assert info["css_layers"] >= 3

    def test_report_manager_clear_cache(self):
        """clear_cache does not raise."""
        mgr = ReportManager()
        mgr.clear_cache()

    def test_css_manager_compiles_all_layers(self):
        """CSSManager compiles CSS with base, branding, components layers."""
        mgr = ReportManager()
        css = mgr.css_manager.compile()
        assert isinstance(css, str)
        assert len(css) > 0
        # Check layer headers present
        assert "BASE" in css
        assert "BRANDING" in css
        assert "COMPONENTS" in css

    def test_css_manager_compile_for_gmm(self):
        """CSSManager compiles report-type CSS for GMM."""
        mgr = ReportManager()
        css = mgr.css_manager.compile_for_report_type("gmm")
        assert isinstance(css, str)
        assert len(css) > 0

    def test_css_manager_has_four_layers(self):
        """CSSManager has 4 default layers."""
        mgr = ReportManager()
        layers = mgr.css_manager.layers
        assert "base" in layers
        assert "branding" in layers
        assert "components" in layers
        assert "custom" in layers

    def test_asset_manager_get_logo_base64(self):
        """AssetManager returns non-empty logo base64."""
        mgr = ReportManager()
        logo = mgr.asset_manager.get_logo_base64()
        assert isinstance(logo, str)
        assert len(logo) > 0
        assert logo.startswith("data:image/")

    def test_asset_manager_embed_plotly(self):
        """AssetManager returns Plotly CDN script tag."""
        mgr = ReportManager()
        plotly = mgr.asset_manager.embed_plotly()
        assert "plotly" in plotly.lower()
        assert "<script" in plotly

    def test_asset_manager_embed_plotly_disabled(self):
        """AssetManager returns empty when Plotly disabled."""
        mgr = ReportManager()
        plotly = mgr.asset_manager.embed_plotly(include_plotly=False)
        assert plotly == ""

    def test_template_manager_initializes(self):
        """TemplateManager initializes with default templates."""
        mgr = ReportManager()
        tm = mgr.template_manager
        assert tm.template_dir.exists()
        assert tm.template_exists("common/base_v2.html")

    def test_template_manager_lists_templates(self):
        """TemplateManager can list available templates."""
        mgr = ReportManager()
        templates = mgr.template_manager.list_templates()
        assert len(templates) > 0
        # Should include model-specific templates
        assert any("gmm" in t for t in templates)
        assert any("regression" in t for t in templates)


# ===========================================================================
# Etapa 2: GMM Report Pipeline
# ===========================================================================


class TestGMMReportPipeline:
    """Tests for the GMM report generation pipeline."""

    def test_generate_gmm_report_returns_html(self, manager, gmm_transformed_data):
        """generate_gmm_report returns a valid HTML string."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_gmm_report_contains_title(self, manager, gmm_transformed_data):
        """GMM report contains default title."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        assert "GMM Results" in html

    def test_gmm_report_custom_title(self, manager, gmm_transformed_data):
        """GMM report uses custom title when provided."""
        html = manager.generate_gmm_report(
            gmm_transformed_data,
            title="My GMM Analysis",
            subtitle="Testing subtitle",
        )
        assert "My GMM Analysis" in html
        assert "Testing subtitle" in html

    def test_gmm_report_contains_coefficients(self, manager, gmm_transformed_data):
        """GMM report includes coefficient names and values."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        assert "L.y" in html
        assert "x1" in html

    def test_gmm_report_contains_diagnostics(self, manager, gmm_transformed_data):
        """GMM report includes Hansen and AR test results."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        # Should contain diagnostic section
        assert "Hansen" in html or "hansen" in html.lower()

    def test_gmm_report_contains_css_inline(self, manager, gmm_transformed_data):
        """GMM report embeds CSS inline."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        assert "<style>" in html

    def test_gmm_report_contains_branding(self, manager, gmm_transformed_data):
        """GMM report includes PanelBox branding."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        assert "PanelBox" in html

    def test_gmm_report_self_contained(self, manager, gmm_transformed_data):
        """GMM report is self-contained (no external CSS/JS links, except Plotly CDN)."""
        html = manager.generate_gmm_report(gmm_transformed_data)
        # CSS should be inline, not linked
        assert '<link rel="stylesheet"' not in html
        # Plotly CDN is allowed
        assert "plotly" in html.lower()

    def test_gmm_report_via_transformer(self, manager):
        """Full pipeline: raw data -> GMMTransformer -> generate_gmm_report."""
        from panelbox.report.transformers import GMMTransformer

        raw_data = {
            "model_type": "Difference GMM",
            "nobs": 200,
            "n_groups": 40,
            "n_instruments": 20,
            "two_step": False,
            "coefficients": [
                {"name": "L.y", "coef": 0.6, "se": 0.12, "tstat": 5.0, "pvalue": 0.001},
            ],
            "hansen_test": {"statistic": 15.0, "pvalue": 0.8, "df": 15},
            "ar_tests": {
                "ar1": {"statistic": -3.0, "pvalue": 0.003},
                "ar2": {"statistic": 1.0, "pvalue": 0.3},
            },
        }
        transformed = GMMTransformer(raw_data).transform()
        html = manager.generate_gmm_report(transformed, title="Difference GMM")
        assert isinstance(html, str)
        assert "Difference GMM" in html
        assert "L.y" in html


# ===========================================================================
# Etapa 2: Regression Report Pipeline
# ===========================================================================


class TestRegressionReportPipeline:
    """Tests for the Regression report generation pipeline."""

    def test_generate_regression_report_returns_html(self, manager, regression_transformed_data):
        """generate_regression_report returns a valid HTML string."""
        html = manager.generate_regression_report(regression_transformed_data)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_regression_report_contains_title(self, manager, regression_transformed_data):
        """Regression report contains default title."""
        html = manager.generate_regression_report(regression_transformed_data)
        assert "Regression Results" in html

    def test_regression_report_custom_title(self, manager, regression_transformed_data):
        """Regression report uses custom title."""
        html = manager.generate_regression_report(
            regression_transformed_data,
            title="Fixed Effects Estimation",
            subtitle="Panel data analysis",
        )
        assert "Fixed Effects Estimation" in html
        assert "Panel data analysis" in html

    def test_regression_report_contains_coefficients(self, manager, regression_transformed_data):
        """Regression report includes coefficient names."""
        html = manager.generate_regression_report(regression_transformed_data)
        assert "x1" in html
        assert "x2" in html

    def test_regression_report_contains_css_inline(self, manager, regression_transformed_data):
        """Regression report embeds CSS inline."""
        html = manager.generate_regression_report(regression_transformed_data)
        assert "<style>" in html

    def test_regression_report_contains_branding(self, manager, regression_transformed_data):
        """Regression report includes PanelBox branding."""
        html = manager.generate_regression_report(regression_transformed_data)
        assert "PanelBox" in html

    def test_regression_report_via_transformer(self, manager):
        """Full pipeline: raw data -> RegressionTransformer -> generate_regression_report."""
        from panelbox.report.transformers import RegressionTransformer

        raw_data = {
            "model_type": "Random Effects",
            "formula": "y ~ x1",
            "nobs": 300,
            "n_entities": 30,
            "n_periods": 10,
            "se_type": "robust",
            "coefficients": [
                {
                    "name": "x1",
                    "coef": 1.0,
                    "se": 0.2,
                    "tstat": 5.0,
                    "pvalue": 0.001,
                    "ci_lower": 0.6,
                    "ci_upper": 1.4,
                },
            ],
            "r_squared": 0.75,
            "adj_r_squared": 0.74,
            "f_statistic": 80.0,
            "f_pvalue": 0.0001,
        }
        transformed = RegressionTransformer(raw_data).transform()
        html = manager.generate_regression_report(transformed, title="RE Results")
        assert isinstance(html, str)
        assert "RE Results" in html
        assert "x1" in html


# ===========================================================================
# Etapa 2: Discrete Report Pipeline
# ===========================================================================


class TestDiscreteReportPipeline:
    """Tests for the Discrete/MLE report generation pipeline."""

    def test_generate_discrete_report_returns_html(self, manager, discrete_transformed_data):
        """generate_discrete_report returns a valid HTML string."""
        html = manager.generate_discrete_report(discrete_transformed_data)
        assert isinstance(html, str)
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_discrete_report_contains_title(self, manager, discrete_transformed_data):
        """Discrete report contains default title."""
        html = manager.generate_discrete_report(discrete_transformed_data)
        assert "Discrete Model Results" in html

    def test_discrete_report_custom_title(self, manager, discrete_transformed_data):
        """Discrete report uses custom title."""
        html = manager.generate_discrete_report(
            discrete_transformed_data,
            title="Logit Model Results",
            subtitle="Binary choice analysis",
        )
        assert "Logit Model Results" in html
        assert "Binary choice analysis" in html

    def test_discrete_report_contains_coefficients(self, manager, discrete_transformed_data):
        """Discrete report includes coefficient names."""
        html = manager.generate_discrete_report(discrete_transformed_data)
        assert "x1" in html
        assert "x2" in html

    def test_discrete_report_contains_css_inline(self, manager, discrete_transformed_data):
        """Discrete report embeds CSS inline."""
        html = manager.generate_discrete_report(discrete_transformed_data)
        assert "<style>" in html

    def test_discrete_report_includes_plotly(self, manager, discrete_transformed_data):
        """Discrete report includes Plotly CDN for interactive charts."""
        html = manager.generate_discrete_report(discrete_transformed_data)
        assert "plotly-2.27.0" in html

    def test_discrete_report_contains_branding(self, manager, discrete_transformed_data):
        """Discrete report includes PanelBox branding."""
        html = manager.generate_discrete_report(discrete_transformed_data)
        assert "PanelBox" in html

    def test_discrete_report_via_transformer(self, manager):
        """Full pipeline: raw data -> DiscreteTransformer -> generate_discrete_report."""
        from panelbox.report.transformers import DiscreteTransformer

        raw_data = {
            "model_type": "Probit",
            "distribution": "normal",
            "nobs": 500,
            "n_entities": 50,
            "n_periods": 10,
            "converged": True,
            "n_iter": 15,
            "se_type": "standard",
            "coefficients": [
                {"name": "x1", "coef": 1.2, "se": 0.3, "tstat": 4.0, "pvalue": 0.001},
            ],
            "loglikelihood": -200.0,
            "aic": 404.0,
            "bic": 412.0,
            "pseudo_r_squared": 0.20,
        }
        transformed = DiscreteTransformer(raw_data).transform()
        html = manager.generate_discrete_report(transformed, title="Probit Results")
        assert isinstance(html, str)
        assert "Probit Results" in html
        assert "x1" in html

    def test_discrete_report_without_classification(self, manager):
        """Discrete report renders without classification metrics."""
        from panelbox.report.transformers import DiscreteTransformer

        raw_data = {
            "model_type": "Poisson",
            "distribution": "poisson",
            "nobs": 300,
            "converged": True,
            "coefficients": [
                {"name": "x1", "coef": 0.5, "se": 0.1, "tstat": 5.0, "pvalue": 0.001},
            ],
            "loglikelihood": -150.0,
            "aic": 304.0,
            "bic": 310.0,
        }
        transformed = DiscreteTransformer(raw_data).transform()
        html = manager.generate_discrete_report(transformed)
        assert isinstance(html, str)
        assert "x1" in html
