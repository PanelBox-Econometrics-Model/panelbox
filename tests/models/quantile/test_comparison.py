"""
Tests for FE Quantile Regression Comparison tools.
"""

import logging

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from panelbox.core.panel_data import PanelData
from panelbox.models.quantile.comparison import ComparisonResults, FEQuantileComparison


@pytest.fixture
def comparison_panel_data():
    """Create panel data suitable for comparison tests."""
    np.random.seed(42)
    n_entities = 30
    n_time = 15
    n = n_entities * n_time

    entity_ids = np.repeat(np.arange(n_entities), n_time)
    time_ids = np.tile(np.arange(n_time), n_entities)

    X1 = np.random.randn(n)
    X2 = np.random.randn(n)

    entity_effects = np.random.randn(n_entities) * 1.5
    entity_effects_expanded = np.repeat(entity_effects, n_time)

    y = 1 + 2 * X1 + 3 * X2 + entity_effects_expanded + np.random.randn(n)

    df = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y, "X1": X1, "X2": X2})

    return PanelData(df, entity_col="entity", time_col="time")


class TestFEQuantileComparison:
    """Test FEQuantileComparison class."""

    def test_initialization(self, comparison_panel_data):
        """Test basic initialization."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        assert comp.tau == [0.5]
        assert comp.formula == "y ~ X1 + X2"
        assert comp.results == {}

    def test_initialization_multiple_quantiles(self, comparison_panel_data):
        """Test initialization with multiple quantiles."""
        comp = FEQuantileComparison(
            comparison_panel_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75]
        )
        assert comp.tau == [0.25, 0.5, 0.75]

    def test_compare_pooled_only(self, comparison_panel_data):
        """Test comparison with pooled QR only."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        result = comp.compare_all(methods=["pooled"], verbose=False)

        assert isinstance(result, ComparisonResults)
        assert "pooled" in result.estimates
        assert "pooled" in result.timing
        assert "pooled" in result.diagnostics
        assert "pseudo_r2" in result.diagnostics["pooled"]
        assert result.timing["pooled"] > 0

    def test_compare_canay_only(self, comparison_panel_data):
        """Test comparison with canay only."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        result = comp.compare_all(methods=["canay"], verbose=False)

        assert "canay" in result.estimates
        assert "canay" in result.diagnostics
        assert "pseudo_r2" in result.diagnostics["canay"]
        assert "location_shift_pval" in result.diagnostics["canay"]

    def test_compare_canay_multiple_quantiles(self, comparison_panel_data):
        """Test canay comparison with multiple quantiles."""
        comp = FEQuantileComparison(
            comparison_panel_data,
            formula="y ~ X1 + X2",
            tau=[0.25, 0.5, 0.75],
        )
        result = comp.compare_all(methods=["canay"], verbose=False)

        assert "canay" in result.estimates
        assert "location_shift_pval" in result.diagnostics["canay"]

    def test_compare_penalty_only(self, comparison_panel_data):
        """Test comparison with penalty method only."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        result = comp.compare_all(methods=["penalty"], verbose=False, lambda_fe=0.1)

        assert "penalty" in result.estimates
        assert "penalty" in result.diagnostics
        assert "pseudo_r2" in result.diagnostics["penalty"]
        assert "lambda_optimal" in result.diagnostics["penalty"]
        assert "n_zero_fe" in result.diagnostics["penalty"]

    def test_compare_all_methods(self, comparison_panel_data):
        """Test comparing all three methods."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        result = comp.compare_all(
            methods=["pooled", "canay", "penalty"], verbose=False, lambda_fe=0.1
        )

        assert len(result.estimates) == 3
        assert "pooled" in result.estimates
        assert "canay" in result.estimates
        assert "penalty" in result.estimates

    def test_compare_all_default_methods(self, comparison_panel_data):
        """Test with default methods (None)."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        result = comp.compare_all(verbose=False, lambda_fe=0.1)

        assert len(result.estimates) == 3

    def test_compare_verbose(self, comparison_panel_data, caplog):
        """Test verbose output during comparison."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        with caplog.at_level(logging.INFO):
            result = comp.compare_all(methods=["pooled"], verbose=True)
        assert result is not None

    def test_compute_diagnostics(self, comparison_panel_data):
        """Test diagnostic computation."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        result = comp.compare_all(methods=["pooled"], verbose=False)

        diagnostics = result.diagnostics["pooled"]
        assert "pseudo_r2" in diagnostics
        assert 0 <= diagnostics["pseudo_r2"] <= 1

    def test_compute_pseudo_r2_multi_quantile(self, comparison_panel_data):
        """Test pseudo-R2 with multi-tau results."""
        comp = FEQuantileComparison(
            comparison_panel_data,
            formula="y ~ X1 + X2",
            tau=[0.25, 0.5, 0.75],
        )
        result = comp.compare_all(methods=["canay"], verbose=False)
        assert "pseudo_r2" in result.diagnostics["canay"]

    def test_without_formula(self, comparison_panel_data):
        """Test comparison without formula (using column positions)."""
        comp = FEQuantileComparison(comparison_panel_data, formula=None, tau=0.5)
        result = comp.compare_all(methods=["pooled"], verbose=False)
        assert "pooled" in result.estimates


class TestComparisonResults:
    """Test ComparisonResults class."""

    @pytest.fixture
    def comparison_results(self, comparison_panel_data):
        """Generate comparison results for testing."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        return comp.compare_all(
            methods=["pooled", "canay", "penalty"],
            verbose=False,
            lambda_fe=0.1,
        )

    def test_print_summary(self, comparison_results, capsys):
        """Test print_summary output."""
        comparison_results.print_summary()
        captured = capsys.readouterr()
        assert "COEFFICIENT ESTIMATES" in captured.out
        assert "COMPUTATIONAL TIME" in captured.out
        assert "DIAGNOSTICS" in captured.out
        assert "Fastest:" in captured.out

    def test_plot_comparison(self, comparison_results):
        """Test comparison plot."""
        fig = comparison_results.plot_comparison()
        assert fig is not None
        plt.close(fig)

    def test_coefficient_correlation_matrix(self, comparison_results):
        """Test coefficient correlation matrix."""
        fig, corr_matrix = comparison_results.coefficient_correlation_matrix()
        assert fig is not None
        assert corr_matrix.shape[0] == corr_matrix.shape[1]
        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(corr_matrix), 1.0, atol=1e-10)
        plt.close(fig)

    def test_results_attributes(self, comparison_results):
        """Test that all expected attributes are present."""
        assert hasattr(comparison_results, "estimates")
        assert hasattr(comparison_results, "timing")
        assert hasattr(comparison_results, "diagnostics")
        assert hasattr(comparison_results, "tau")


class TestBootstrapComparison:
    """Test bootstrap comparison functionality."""

    def test_bootstrap_comparison_basic(self, comparison_panel_data):
        """Test bootstrap comparison with small n_boot."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        stats = comp.bootstrap_comparison(n_boot=5, methods=["canay"])

        assert "canay" in stats
        assert "mean" in stats["canay"]
        assert "std" in stats["canay"]
        assert "coverage" in stats["canay"]
        assert "ci_lower" in stats["canay"]
        assert "ci_upper" in stats["canay"]

    def test_bootstrap_comparison_default_methods(self, comparison_panel_data):
        """Test bootstrap with default methods."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        stats = comp.bootstrap_comparison(n_boot=3)

        # Default includes canay and penalty
        assert "canay" in stats or "penalty" in stats

    def test_bootstrap_coverage(self, comparison_panel_data):
        """Test that bootstrap coverage is reasonable."""
        comp = FEQuantileComparison(comparison_panel_data, formula="y ~ X1 + X2", tau=0.5)
        stats = comp.bootstrap_comparison(n_boot=5, methods=["canay"])

        coverage = stats["canay"]["coverage"]
        assert 0 < coverage <= 1
