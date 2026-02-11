"""
Tests for Panel VAR stability checks and plots.

This module tests stability verification and visualization.
"""

import numpy as np
import pytest

from panelbox.var import PanelVAR, PanelVARData


class TestStabilityPlotting:
    """Tests for stability plotting."""

    def test_plot_stability_matplotlib(self, simple_panel_data):
        """Test stability plot with matplotlib backend."""
        pytest.importorskip("matplotlib")

        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Plot stability (don't show, just create)
        fig = results.plot_stability(backend="matplotlib", show=False)

        # Check that figure was created
        assert fig is not None

        # Check figure properties
        import matplotlib.pyplot as plt

        assert isinstance(fig, plt.Figure)

        # Close figure
        plt.close(fig)

    def test_plot_stability_returns_none_when_show(self, simple_panel_data):
        """Test that plot_stability returns None when show=True."""
        pytest.importorskip("matplotlib")
        import matplotlib

        # Use non-interactive backend
        matplotlib.use("Agg")

        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Plot with show=True (but matplotlib won't actually display in test)
        result = results.plot_stability(backend="matplotlib", show=True)

        # Should return None when show=True
        assert result is None

        # Clean up
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_plot_stability_plotly_optional(self, simple_panel_data):
        """Test plotly backend (if available)."""
        plotly = pytest.importorskip("plotly", reason="Plotly not installed")

        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Plot with plotly
        fig = results.plot_stability(backend="plotly", show=False)

        # Check that figure was created
        assert fig is not None
        assert isinstance(fig, plotly.graph_objects.Figure)

    def test_plot_stability_invalid_backend(self, simple_panel_data):
        """Test that invalid backend raises error."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Invalid backend should raise ValueError
        with pytest.raises(ValueError, match="backend must be"):
            results.plot_stability(backend="invalid")


class TestStabilityProperties:
    """Tests for stability properties and methods."""

    def test_eigenvalues_property(self, simple_panel_data):
        """Test that eigenvalues property works."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Get eigenvalues
        eigs = results.eigenvalues

        # Should be array of complex numbers
        assert isinstance(eigs, np.ndarray)
        assert len(eigs) == results.K * results.p

        # Should have complex dtype
        assert np.iscomplexobj(eigs)

    def test_max_eigenvalue_modulus(self, simple_panel_data):
        """Test max eigenvalue modulus."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Get max modulus
        max_mod = results.max_eigenvalue_modulus

        # Should be a positive float
        assert isinstance(max_mod, float)
        assert max_mod >= 0

        # Should equal max of absolute values
        assert np.isclose(max_mod, np.max(np.abs(results.eigenvalues)))

    def test_is_stable_method(self, var_dgp_data):
        """Test is_stable() method."""
        df, true_params = var_dgp_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # is_stable should return bool
        stable = results.is_stable()
        assert isinstance(stable, bool)

        # Should be consistent with max_eigenvalue_modulus
        assert stable == (results.max_eigenvalue_modulus < 1.0)

    def test_stability_margin(self, simple_panel_data):
        """Test stability margin property."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Get stability margin
        margin = results.stability_margin

        # Should be float
        assert isinstance(margin, float)

        # Should equal 1 - max_modulus
        expected_margin = 1.0 - results.max_eigenvalue_modulus
        assert np.isclose(margin, expected_margin)

        # If stable, margin should be positive
        if results.is_stable():
            assert margin > 0
        else:
            assert margin <= 0
