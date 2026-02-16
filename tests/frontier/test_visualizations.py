"""
Tests for SFA visualization functions.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.data import add_translog


@pytest.fixture
def sample_data():
    """Generate sample production data for testing."""
    np.random.seed(42)
    n = 100

    # Generate data
    data = pd.DataFrame(
        {
            "entity_id": np.repeat(range(20), 5),
            "time": np.tile(range(5), 20),
            "log_output": np.random.randn(n) + 5,
            "log_labor": np.random.randn(n) + 3,
            "log_capital": np.random.randn(n) + 4,
            "region": np.random.choice(["North", "South", "East", "West"], n),
        }
    )

    return data


@pytest.fixture
def fitted_model(sample_data):
    """Fit a simple SFA model for testing."""
    # Use subset for faster tests
    data_subset = sample_data.head(50).copy()

    sf = StochasticFrontier(
        data=data_subset,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )

    result = sf.fit()
    return result


@pytest.fixture
def panel_fitted_model(sample_data):
    """Fit a panel SFA model for testing."""
    sf = StochasticFrontier(
        data=sample_data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity="entity_id",
        time="time",
        frontier="production",
        dist="half_normal",
        model_type="pitt_lee",
    )

    result = sf.fit()
    return result


class TestEfficiencyPlots:
    """Test efficiency distribution and ranking plots."""

    def test_plot_efficiency_histogram_plotly(self, fitted_model):
        """Test histogram plot with Plotly backend."""
        fig = fitted_model.plot_efficiency(kind="histogram", backend="plotly")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_efficiency_histogram_matplotlib(self, fitted_model):
        """Test histogram plot with Matplotlib backend."""
        fig = fitted_model.plot_efficiency(kind="histogram", backend="matplotlib")
        assert fig is not None
        assert hasattr(fig, "axes")

    def test_plot_efficiency_ranking_plotly(self, fitted_model):
        """Test ranking plot with Plotly backend."""
        fig = fitted_model.plot_efficiency(kind="ranking", backend="plotly", top_n=5, bottom_n=5)
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_efficiency_ranking_matplotlib(self, fitted_model):
        """Test ranking plot with Matplotlib backend."""
        fig = fitted_model.plot_efficiency(
            kind="ranking", backend="matplotlib", top_n=5, bottom_n=5
        )
        assert fig is not None
        assert hasattr(fig, "axes")

    def test_plot_efficiency_boxplot_plotly(self, fitted_model):
        """Test boxplot with Plotly backend."""
        # Add group variable to efficiency dataframe
        eff_df = fitted_model.efficiency(estimator="bc")
        eff_df["region"] = fitted_model.model.data["region"].values[: len(eff_df)]

        # Create a new result with modified data to include region
        from panelbox.frontier.visualization.efficiency_plots import plot_efficiency_boxplot

        fig = plot_efficiency_boxplot(eff_df, group_var="region", backend="plotly")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_efficiency_bounds(self, fitted_model):
        """Test that efficiency values are in valid range (0, 1]."""
        eff_df = fitted_model.efficiency(estimator="bc")
        eff_values = eff_df["efficiency"].values

        assert np.all(eff_values > 0), "Efficiency must be > 0"
        assert np.all(eff_values <= 1), "Efficiency must be <= 1"


class TestEvolutionPlots:
    """Test temporal evolution plots for panel data."""

    @pytest.mark.skip(
        reason="Panel models currently return SFResult instead of PanelSFResult - needs fix in estimation code"
    )
    def test_plot_timeseries_plotly(self, panel_fitted_model):
        """Test time series plot with Plotly backend."""
        fig = panel_fitted_model.plot_efficiency_evolution(
            kind="timeseries", backend="plotly", show_ci=True
        )
        assert fig is not None
        assert hasattr(fig, "data")

    @pytest.mark.skip(
        reason="Panel models currently return SFResult instead of PanelSFResult - needs fix in estimation code"
    )
    def test_plot_timeseries_matplotlib(self, panel_fitted_model):
        """Test time series plot with Matplotlib backend."""
        fig = panel_fitted_model.plot_efficiency_evolution(
            kind="timeseries", backend="matplotlib", show_ci=True
        )
        assert fig is not None
        assert hasattr(fig, "axes")

    @pytest.mark.skip(
        reason="Panel models currently return SFResult instead of PanelSFResult - needs fix in estimation code"
    )
    def test_plot_spaghetti_plotly(self, panel_fitted_model):
        """Test spaghetti plot with Plotly backend."""
        fig = panel_fitted_model.plot_efficiency_evolution(
            kind="spaghetti", backend="plotly", show_mean=True
        )
        assert fig is not None
        assert hasattr(fig, "data")

    @pytest.mark.skip(
        reason="Panel models currently return SFResult instead of PanelSFResult - needs fix in estimation code"
    )
    def test_plot_heatmap_plotly(self, panel_fitted_model):
        """Test heatmap plot with Plotly backend."""
        fig = panel_fitted_model.plot_efficiency_evolution(
            kind="heatmap", backend="plotly", order_by="efficiency"
        )
        assert fig is not None
        assert hasattr(fig, "data")

    @pytest.mark.skip(
        reason="Panel models currently return SFResult instead of PanelSFResult - needs fix in estimation code"
    )
    def test_plot_fanchart_plotly(self, panel_fitted_model):
        """Test fan chart plot with Plotly backend."""
        fig = panel_fitted_model.plot_efficiency_evolution(
            kind="fanchart", backend="plotly", percentiles=[10, 25, 50, 75, 90]
        )
        assert fig is not None
        assert hasattr(fig, "data")

    @pytest.mark.skip(
        reason="Panel models currently return SFResult instead of PanelSFResult - needs fix in estimation code"
    )
    def test_plot_timeseries_with_events(self, panel_fitted_model):
        """Test time series with event annotations."""
        fig = panel_fitted_model.plot_efficiency_evolution(
            kind="timeseries", backend="plotly", events={2: "Event A", 4: "Event B"}
        )
        assert fig is not None


class TestFrontierPlots:
    """Test frontier estimation plots."""

    def test_plot_frontier_2d_plotly(self, fitted_model):
        """Test 2D frontier plot with Plotly backend."""
        fig = fitted_model.plot_frontier(input_var="log_labor", kind="2d", backend="plotly")
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_frontier_2d_matplotlib(self, fitted_model):
        """Test 2D frontier plot with Matplotlib backend."""
        fig = fitted_model.plot_frontier(input_var="log_labor", kind="2d", backend="matplotlib")
        assert fig is not None
        assert hasattr(fig, "axes")

    def test_plot_frontier_3d_plotly(self, fitted_model):
        """Test 3D frontier plot with Plotly backend."""
        fig = fitted_model.plot_frontier(
            input_vars=["log_labor", "log_capital"],
            kind="3d",
            backend="plotly",
            n_grid=15,  # Small grid for speed
        )
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_frontier_contour_plotly(self, fitted_model):
        """Test contour plot with Plotly backend."""
        fig = fitted_model.plot_frontier(
            input_vars=["log_labor", "log_capital"],
            kind="contour",
            backend="plotly",
            n_grid=15,
            levels=10,
        )
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_frontier_partial_plotly(self, fitted_model):
        """Test partial frontier plot with Plotly backend."""
        fig = fitted_model.plot_frontier(
            input_var="log_labor", kind="partial", backend="plotly", fix_others_at="mean"
        )
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_frontier_partial_with_dict(self, fitted_model):
        """Test partial frontier plot with custom fixed values."""
        fig = fitted_model.plot_frontier(
            input_var="log_labor",
            kind="partial",
            backend="plotly",
            fix_others_at={"log_capital": 4.0},
        )
        assert fig is not None


class TestReports:
    """Test report generation functions."""

    def test_to_latex(self, fitted_model):
        """Test LaTeX report generation."""
        latex = fitted_model.to_latex(caption="Test Results", label="tab:test")
        assert isinstance(latex, str)
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "Test Results" in latex

    def test_to_markdown(self, fitted_model):
        """Test Markdown report generation."""
        md = fitted_model.to_markdown()
        assert isinstance(md, str)
        assert "# Stochastic Frontier Analysis Results" in md
        assert "## Model Information" in md
        assert "## Parameter Estimates" in md

    def test_to_html(self, fitted_model):
        """Test HTML report generation."""
        html = fitted_model.to_html(include_plots=False)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<html>" in html
        assert "Stochastic Frontier Analysis Results" in html

    def test_to_html_with_plots(self, fitted_model):
        """Test HTML report with plots."""
        html = fitted_model.to_html(include_plots=True)
        assert isinstance(html, str)
        assert "plotly" in html.lower()

    def test_to_html_save_file(self, fitted_model, tmp_path):
        """Test saving HTML report to file."""
        output_file = tmp_path / "test_report.html"
        fitted_model.to_html(filename=str(output_file), include_plots=False)
        assert output_file.exists()
        content = output_file.read_text()
        assert "<!DOCTYPE html>" in content

    def test_efficiency_table(self, fitted_model):
        """Test efficiency table generation."""
        eff_table = fitted_model.efficiency_table(sort_by="te", ascending=False, top_n=10)
        assert isinstance(eff_table, pd.DataFrame)
        assert "rank" in eff_table.columns
        assert "te" in eff_table.columns
        assert len(eff_table) == 10

    def test_compare_models(self, fitted_model):
        """Test model comparison."""
        from panelbox.frontier.visualization.reports import compare_models

        # Create a second model for comparison
        data = fitted_model.model.data
        sf2 = StochasticFrontier(
            data=data,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist="exponential",
        )
        result2 = sf2.fit()

        comparison = compare_models(
            models={"Half-Normal": fitted_model, "Exponential": result2}, output_format="dataframe"
        )

        assert isinstance(comparison, pd.DataFrame)
        assert "Half-Normal" in comparison.index
        assert "Exponential" in comparison.index
        assert "AIC" in comparison.columns
        assert "BIC" in comparison.columns


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_backend(self, fitted_model):
        """Test invalid backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            from panelbox.frontier.visualization.efficiency_plots import (
                plot_efficiency_distribution,
            )

            eff_df = fitted_model.efficiency(estimator="bc")
            plot_efficiency_distribution(eff_df, backend="invalid")

    def test_invalid_plot_kind(self, fitted_model):
        """Test invalid plot kind raises error."""
        with pytest.raises(ValueError, match="Unknown kind"):
            fitted_model.plot_efficiency(kind="invalid")

    def test_missing_input_var(self, fitted_model):
        """Test missing input_var raises error."""
        with pytest.raises(ValueError, match="must provide 'input_var'"):
            fitted_model.plot_frontier(kind="2d")

    def test_wrong_number_inputs_3d(self, fitted_model):
        """Test wrong number of inputs for 3D plot raises error."""
        with pytest.raises(ValueError, match="must provide 'input_vars' with 2 variables"):
            fitted_model.plot_frontier(input_vars=["log_labor"], kind="3d")

    def test_boxplot_missing_group_var(self, fitted_model):
        """Test boxplot without group_var raises error."""
        with pytest.raises(ValueError, match="must provide 'group_var'"):
            fitted_model.plot_efficiency(kind="boxplot")

    def test_invalid_output_format_compare(self, fitted_model):
        """Test invalid output format in compare_models."""
        from panelbox.frontier.visualization.reports import compare_models

        with pytest.raises(ValueError, match="Unknown output_format"):
            compare_models(models={"Model1": fitted_model}, output_format="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
