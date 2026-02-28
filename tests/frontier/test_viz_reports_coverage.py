"""
Deep coverage tests for frontier/visualization/reports.py.

Targets specific uncovered lines and branch partials to push
coverage from ~83.86% toward 95%+.

Uncovered items:
- Branch 51->55: include_stats=None default (already covered partially)
- Branch 67->69: caption falsy (no caption) in to_latex
- Branch 69->73: label falsy (no label) in to_latex
- Lines 78-85 branches: include_stats without certain keys
- Lines 83, 103-104: tval in include_stats
- Lines 111, 113, 115: significance star branches (p < 0.01/0.05/0.1)
- Lines 213-214: is_panel=True in to_html
- Lines 264, 266, 268: significance star branches in to_html
- Lines 321-322: exception in to_html plots
- Lines 426, 432: get_html_css themes (professional, presentation)
- Lines 473-474: is_panel=True in to_markdown
- Lines 512, 514, 516: significance star branches in to_markdown
- Lines 588-593: compare_models with latex/markdown/invalid format
- Branch 628->632: sort_by not in columns
- Branch 632->636: top_n=None
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.visualization.reports import (
    compare_models,
    efficiency_table,
    get_html_css,
    to_html,
    to_latex,
    to_markdown,
)


def _make_mock_result(
    is_panel=False,
    n_entities=10,
    n_periods=5,
    pvalue_range="mixed",
):
    """Create a mock SFResult with controllable p-values.

    Parameters
    ----------
    pvalue_range : str
        'mixed' - p-values span all significance levels
        'all_significant' - all p < 0.01
    """
    result = MagicMock()

    # Basic attributes
    result.nobs = 50
    result.loglik = -120.5
    result.aic = 250.0
    result.bic = 260.0
    result.converged = True
    result.sigma_v = 0.15
    result.sigma_u = 0.20
    result.sigma = 0.25
    result.lambda_param = 1.33
    result.gamma = 0.64

    # Panel attributes
    result.model.is_panel = is_panel
    if is_panel:
        result.model.n_entities = n_entities
        result.model.n_periods = n_periods

    # Frontier type and distribution
    result.model.frontier_type.value = "production"
    result.model.dist.value = "half_normal"

    # Parameters with varied p-values for star coverage
    param_names = ["const", "log_labor", "log_capital", "sigma_v", "sigma_u"]
    coefs = [1.0, 0.5, 0.3, 0.15, 0.20]
    ses = [0.1, 0.05, 0.15, 0.02, 0.03]

    if pvalue_range == "mixed":
        pvals = [0.005, 0.03, 0.07, 0.8, 0.9]  # ***, **, *, none, none
    else:
        pvals = [0.001, 0.001, 0.001, 0.001, 0.001]

    tvals = [c / s for c, s in zip(coefs, ses)]

    result.params = pd.Series(coefs, index=param_names)
    result.se = pd.Series(ses, index=param_names)
    result.tvalues = pd.Series(tvals, index=param_names)
    result.pvalues = pd.Series(pvals, index=param_names)

    # Mean efficiency
    result.mean_efficiency = 0.78

    # Efficiency method
    eff_values = np.random.RandomState(42).uniform(0.5, 0.95, 50)
    eff_df = pd.DataFrame({"efficiency": eff_values, "entity": range(50)})
    result.efficiency.return_value = eff_df

    return result


# --- to_latex tests ---


class TestToLatex:
    """Test to_latex coverage gaps."""

    def test_no_caption_no_label(self):
        """Cover branches 67->69, 69->73: no caption, no label."""
        result = _make_mock_result()
        latex = to_latex(result, caption=None, label=None)
        assert "\\caption" not in latex
        assert "\\label" not in latex
        assert "\\begin{table}" in latex

    def test_include_tval(self):
        """Cover lines 83, 103-104: tval in include_stats."""
        result = _make_mock_result()
        latex = to_latex(result, include_stats=["coef", "se", "tval", "pval"])
        assert "t-value" in latex

    def test_only_coef(self):
        """Cover branches 80->82, 84->87: include_stats without se/tval/pval."""
        result = _make_mock_result()
        latex = to_latex(result, include_stats=["coef"])
        assert "Coefficient" in latex
        assert "Std. Error" not in latex
        assert "p-value" not in latex

    def test_only_se_and_pval(self):
        """Cover branches 78->80, 94->98: include_stats without 'coef'."""
        result = _make_mock_result(pvalue_range="mixed")
        latex = to_latex(result, include_stats=["se", "pval"])
        assert "Std. Error" in latex
        assert "p-value" in latex
        assert "Coefficient" not in latex

    def test_significance_stars(self):
        """Cover lines 111, 113, 115: all significance star branches."""
        result = _make_mock_result(pvalue_range="mixed")
        latex = to_latex(result, include_stats=["coef", "pval"])
        # Should contain stars for different significance levels
        assert "***" in latex  # p < 0.01
        assert "**" in latex  # p < 0.05 (log_labor has p=0.03)
        assert "*" in latex  # p < 0.1 (log_capital has p=0.07)


# --- to_html tests ---


class TestToHtml:
    """Test to_html coverage gaps."""

    def test_panel_model_html(self):
        """Cover lines 213-214: is_panel=True adds entities/periods."""
        result = _make_mock_result(is_panel=True)
        html = to_html(result, include_plots=False)
        assert "Entities" in html
        assert "Time Periods" in html

    def test_significance_stars_html(self):
        """Cover lines 264, 266, 268: significance star branches in HTML."""
        result = _make_mock_result(pvalue_range="mixed")
        html = to_html(result, include_plots=False)
        assert "***" in html
        assert "**" in html

    def test_plot_exception_handler(self):
        """Cover lines 321-322: exception during plot generation."""
        result = _make_mock_result()
        # Make plot_efficiency raise an exception
        result.plot_efficiency.side_effect = RuntimeError("Plot failed")
        html = to_html(result, include_plots=True)
        assert "Could not generate plots" in html

    def test_html_professional_theme(self):
        """Cover line 426: professional theme CSS."""
        result = _make_mock_result()
        html = to_html(result, include_plots=False, theme="professional")
        assert "#3f51b5" in html  # Professional theme color

    def test_html_presentation_theme(self):
        """Cover line 432: presentation theme CSS."""
        result = _make_mock_result()
        html = to_html(result, include_plots=False, theme="presentation")
        assert "#d32f2f" in html  # Presentation theme color


# --- get_html_css tests ---


class TestGetHtmlCss:
    """Test get_html_css theme branches."""

    def test_professional_theme(self):
        """Cover line 426: professional theme."""
        css = get_html_css("professional")
        assert "#3f51b5" in css
        assert "#1a237e" in css

    def test_presentation_theme(self):
        """Cover line 432: presentation theme."""
        css = get_html_css("presentation")
        assert "#d32f2f" in css
        assert "#f44336" in css

    def test_academic_theme(self):
        """Test default academic theme (no extra CSS added)."""
        css = get_html_css("academic")
        assert "#3f51b5" not in css
        assert "#d32f2f" not in css


# --- to_markdown tests ---


class TestToMarkdown:
    """Test to_markdown coverage gaps."""

    def test_panel_model_markdown(self):
        """Cover lines 473-474: is_panel=True in markdown."""
        result = _make_mock_result(is_panel=True)
        md = to_markdown(result)
        assert "**Entities**" in md
        assert "**Time Periods**" in md

    def test_significance_stars_markdown(self):
        """Cover lines 512, 514, 516: significance star branches."""
        result = _make_mock_result(pvalue_range="mixed")
        md = to_markdown(result)
        assert "***" in md  # p < 0.01
        # The ** and * will be in the markdown table rows


# --- compare_models tests ---


class TestCompareModels:
    """Test compare_models output format branches."""

    def _make_two_models(self):
        """Create two mock models for comparison."""
        m1 = _make_mock_result()
        m2 = _make_mock_result()
        m2.aic = 240.0  # Better AIC
        m2.bic = 255.0
        m2.mean_efficiency = 0.82
        return {"Model A": m1, "Model B": m2}

    def test_latex_format(self):
        """Cover line 589: output_format='latex'."""
        models = self._make_two_models()
        result = compare_models(models, output_format="latex")
        assert isinstance(result, str)
        assert "tabular" in result or "Model A" in result

    def test_markdown_format(self):
        """Cover line 591: output_format='markdown'."""
        pytest.importorskip("tabulate")
        models = self._make_two_models()
        result = compare_models(models, output_format="markdown")
        assert isinstance(result, str)
        assert "Model A" in result

    def test_invalid_format(self):
        """Cover lines 592-593: invalid output_format."""
        models = self._make_two_models()
        with pytest.raises(ValueError, match="Unknown output_format"):
            compare_models(models, output_format="csv")


# --- efficiency_table tests ---


class TestEfficiencyTable:
    """Test efficiency_table coverage gaps."""

    def test_sort_by_nonexistent_column(self):
        """Cover branch 628->632: sort_by not in columns."""
        result = _make_mock_result()
        table = efficiency_table(result, sort_by="nonexistent_col")
        assert isinstance(table, pd.DataFrame)
        assert "rank" in table.columns

    def test_no_top_n(self):
        """Cover branch 632->636: top_n=None (no row limit)."""
        result = _make_mock_result()
        table = efficiency_table(result, top_n=None)
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 50  # All observations
