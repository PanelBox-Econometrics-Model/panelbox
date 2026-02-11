"""
Tests for sensitivity analysis visualization.
"""

import numpy as np
import pytest

from panelbox.visualization.var_plots import plot_instrument_sensitivity


class TestSensitivityPlot:
    """Test sensitivity analysis plotting function."""

    @pytest.fixture
    def mock_sensitivity_results(self):
        """Create mock sensitivity analysis results."""
        return {
            "max_instruments": [6, 12, 18, 24],
            "n_instruments_actual": [6, 12, 18, 24],
            "coefficients": {
                "coef_0": [0.50, 0.51, 0.52, 0.53],
                "coef_1": [0.30, 0.31, 0.30, 0.31],
                "coef_2": [0.10, 0.10, 0.11, 0.10],
                "coef_3": [0.20, 0.21, 0.20, 0.21],
            },
            "coefficient_changes": {
                "coef_0": 6.0,  # 6% change
                "coef_1": 3.3,
                "coef_2": 10.0,
                "coef_3": 5.0,
            },
            "max_change_overall": 10.0,
            "stable": False,
            "interpretation": "⚠ Coefficients NOT stable. Max change: 10.00%. This suggests instrument proliferation.",
        }

    def test_plot_matplotlib_returns_figure(self, mock_sensitivity_results):
        """Test matplotlib plotting returns figure when show=False."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = plot_instrument_sensitivity(
            mock_sensitivity_results, backend="matplotlib", show=False
        )

        assert fig is not None
        assert hasattr(fig, "savefig")  # Check it's a matplotlib figure

        # Clean up
        plt.close(fig)

    def test_plot_matplotlib_shows_figure(self, mock_sensitivity_results):
        """Test matplotlib plotting with show=True returns None."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        result = plot_instrument_sensitivity(
            mock_sensitivity_results, backend="matplotlib", show=True
        )

        # When show=True, should return None
        assert result is None

        # Clean up any open figures
        plt.close("all")

    def test_plot_plotly_returns_figure(self, mock_sensitivity_results):
        """Test plotly plotting returns figure when show=False."""
        try:
            import plotly.graph_objects as go
        except ImportError:
            pytest.skip("plotly not available")

        fig = plot_instrument_sensitivity(mock_sensitivity_results, backend="plotly", show=False)

        assert fig is not None
        assert hasattr(fig, "add_trace")  # Check it's a plotly figure

    def test_plot_with_custom_title(self, mock_sensitivity_results):
        """Test plotting with custom title."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        custom_title = "Custom Sensitivity Analysis"
        fig = plot_instrument_sensitivity(
            mock_sensitivity_results, title=custom_title, backend="matplotlib", show=False
        )

        # Check title is set
        assert fig.axes[0].get_title() == custom_title

        plt.close(fig)

    def test_plot_with_max_coefs_limit(self, mock_sensitivity_results):
        """Test plotting with limited number of coefficients."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        fig = plot_instrument_sensitivity(
            mock_sensitivity_results, backend="matplotlib", max_coefs_to_plot=2, show=False
        )

        # Should only plot top 2 coefficients with largest changes
        legend = fig.axes[0].get_legend()
        assert len(legend.get_texts()) == 2

        plt.close(fig)

    def test_plot_with_custom_figsize(self, mock_sensitivity_results):
        """Test matplotlib plot with custom figsize."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        custom_figsize = (10, 8)
        fig = plot_instrument_sensitivity(
            mock_sensitivity_results, backend="matplotlib", figsize=custom_figsize, show=False
        )

        # Check figure size
        assert fig.get_size_inches()[0] == pytest.approx(custom_figsize[0])
        assert fig.get_size_inches()[1] == pytest.approx(custom_figsize[1])

        plt.close(fig)

    def test_plot_with_stable_interpretation(self, mock_sensitivity_results):
        """Test plot with stable coefficients."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        # Modify to show stable results
        stable_results = mock_sensitivity_results.copy()
        stable_results["stable"] = True
        stable_results["max_change_overall"] = 2.0
        stable_results["interpretation"] = (
            "✓ Coefficients stable across 4 instrument counts. Max change: 2.00%"
        )

        fig = plot_instrument_sensitivity(stable_results, backend="matplotlib", show=False)

        # Should create figure without error
        assert fig is not None

        plt.close(fig)

    def test_plot_invalid_backend_raises_error(self, mock_sensitivity_results):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            plot_instrument_sensitivity(mock_sensitivity_results, backend="invalid_backend")

    def test_plot_plotly_without_plotly_raises_error(self, mock_sensitivity_results, monkeypatch):
        """Test that plotly backend without plotly installed raises ImportError."""
        # Mock HAS_PLOTLY to False
        import panelbox.visualization.var_plots as vp

        monkeypatch.setattr(vp, "HAS_PLOTLY", False)

        with pytest.raises(ImportError, match="Plotly is required"):
            plot_instrument_sensitivity(mock_sensitivity_results, backend="plotly")

    def test_plot_handles_missing_coefficient_changes(self):
        """Test plotting when coefficient_changes is missing."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        # Results without coefficient_changes
        results = {
            "max_instruments": [6, 12],
            "n_instruments_actual": [6, 12],
            "coefficients": {
                "coef_0": [0.50, 0.51],
                "coef_1": [0.30, 0.30],
            },
            "stable": True,
            "interpretation": "Stable",
        }

        fig = plot_instrument_sensitivity(results, backend="matplotlib", show=False)

        # Should still work, just plot first N coefficients
        assert fig is not None

        plt.close(fig)

    def test_plot_with_single_coefficient(self):
        """Test plotting with only one coefficient."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        results = {
            "max_instruments": [6, 12, 18],
            "n_instruments_actual": [6, 12, 18],
            "coefficients": {
                "coef_0": [0.50, 0.51, 0.52],
            },
            "coefficient_changes": {"coef_0": 4.0},
            "stable": True,
            "interpretation": "Stable",
        }

        fig = plot_instrument_sensitivity(results, backend="matplotlib", show=False)

        assert fig is not None
        # Should have one line
        assert len(fig.axes[0].get_lines()) == 1

        plt.close(fig)

    def test_plot_with_many_coefficients(self):
        """Test plotting with many coefficients (should limit)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        # Create 15 coefficients
        coefficients = {f"coef_{i}": [0.5 + i * 0.01, 0.51 + i * 0.01] for i in range(15)}
        coef_changes = {f"coef_{i}": float(i) for i in range(15)}

        results = {
            "max_instruments": [6, 12],
            "n_instruments_actual": [6, 12],
            "coefficients": coefficients,
            "coefficient_changes": coef_changes,
            "stable": True,
            "interpretation": "Stable",
        }

        fig = plot_instrument_sensitivity(
            results, backend="matplotlib", max_coefs_to_plot=6, show=False
        )

        # Should plot only 6 coefficients (those with largest changes)
        assert len(fig.axes[0].get_legend().get_texts()) == 6

        plt.close(fig)


class TestSensitivityPlotIntegration:
    """Integration tests with sensitivity analysis."""

    def test_plot_from_sensitivity_analysis(self):
        """Test plotting results from actual sensitivity analysis."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            pytest.skip("matplotlib not available")

        from panelbox.var.diagnostics import instrument_sensitivity_analysis

        # Mock model function
        def mock_model(max_instruments, **kwargs):
            class MockResult:
                def __init__(self, n_instr):
                    self.params_by_eq = [np.array([0.5, 0.3, 0.1])]
                    self.n_instruments = n_instr

            return MockResult(max_instruments)

        # Run sensitivity analysis
        sensitivity = instrument_sensitivity_analysis(
            model_func=mock_model, max_instruments_list=[6, 12, 18]
        )

        # Plot results
        fig = plot_instrument_sensitivity(sensitivity, backend="matplotlib", show=False)

        assert fig is not None

        plt.close(fig)
