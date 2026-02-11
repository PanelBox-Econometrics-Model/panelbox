"""
Tests for Panel VAR reporting methods.

This module tests summary(), to_latex(), and to_html() methods.
"""

import numpy as np
import pytest

from panelbox.var import PanelVAR, PanelVARData


class TestReportingMethods:
    """Tests for reporting methods."""

    def test_to_latex_basic(self, simple_panel_data):
        """Test basic LaTeX export."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Generate LaTeX
        latex_output = results.to_latex()

        # Check that output is a string
        assert isinstance(latex_output, str)

        # Check for LaTeX table structure
        assert "\\begin{table}" in latex_output
        assert "\\end{table}" in latex_output
        assert "\\begin{tabular}" in latex_output
        assert "\\end{tabular}" in latex_output
        assert "\\hline" in latex_output

        # Check for variable names
        assert "y1" in latex_output
        assert "y2" in latex_output

        # Check for regressor names
        assert "L1.y1" in latex_output
        assert "L1.y2" in latex_output

    def test_to_latex_single_equation(self, simple_panel_data):
        """Test LaTeX export for single equation."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Generate LaTeX for equation 0 only
        latex_output = results.to_latex(equation=0)

        assert isinstance(latex_output, str)
        assert "\\begin{table}" in latex_output
        assert "y1" in latex_output

    def test_to_html_basic(self, simple_panel_data):
        """Test basic HTML export."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Generate HTML
        html_output = results.to_html()

        # Check that output is a string
        assert isinstance(html_output, str)

        # Check for HTML table structure
        assert "<table" in html_output
        assert "</table>" in html_output
        assert "<thead>" in html_output
        assert "<tbody>" in html_output
        assert "<tr>" in html_output
        assert "<td>" in html_output

        # Check for variable names
        assert "y1" in html_output
        assert "y2" in html_output

        # Check for regressor names
        assert "L1.y1" in html_output
        assert "L1.y2" in html_output

    def test_to_html_single_equation(self, simple_panel_data):
        """Test HTML export for single equation."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Generate HTML for equation 0 only
        html_output = results.to_html(equation=0)

        assert isinstance(html_output, str)
        assert "<table" in html_output
        assert "y1" in html_output

    def test_latex_html_consistency(self, simple_panel_data):
        """Test that LaTeX and HTML contain similar information."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        latex_output = results.to_latex()
        html_output = results.to_html()

        # Both should contain model statistics
        for stat_name in ["AIC", "BIC", "Stable"]:
            assert stat_name in latex_output
            assert stat_name in html_output

        # Both should contain all endogenous variable names
        for var_name in ["y1", "y2"]:
            assert var_name in latex_output
            assert var_name in html_output

        # Both should contain all lag regressor names
        for lag in range(1, 3):
            for var in ["y1", "y2"]:
                lag_name = f"L{lag}.{var}"
                assert lag_name in latex_output
                assert lag_name in html_output


class TestLagOrderResultReporting:
    """Tests for lag order result reporting."""

    def test_lag_order_summary(self, simple_panel_data):
        """Test lag order selection summary."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        lag_results = model.select_lag_order(max_lags=4)

        # Test summary
        summary_str = lag_results.summary()

        assert isinstance(summary_str, str)
        assert "Lag Order Selection" in summary_str
        assert "AIC" in summary_str
        assert "BIC" in summary_str
        assert "HQIC" in summary_str

    def test_lag_order_selected(self, simple_panel_data):
        """Test that selected lags are reported."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        lag_results = model.select_lag_order(max_lags=4)

        summary_str = lag_results.summary()

        # Check that selected lags are shown
        assert "Selected lags:" in summary_str
        for criterion in ["AIC", "BIC", "HQIC"]:
            if criterion in lag_results.selected:
                selected_lag = lag_results.selected[criterion]
                assert str(selected_lag) in summary_str
