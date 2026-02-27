"""Tests for panelbox.frontier.visualization.four_component_plots module.

Covers all 6 functions using mock FourComponentResult objects.
"""

import matplotlib

matplotlib.use("Agg")
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def mock_result():
    """Create a mock FourComponentResult with realistic attributes."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 5
    n_obs = n_entities * n_periods

    result = MagicMock()

    # Entity and time IDs (0-indexed, repeating)
    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_entities)
    result.model.entity_id = entity_ids
    result.model.time_id = time_ids
    result.model.n_entities = n_entities

    # Efficiency components
    result.eta_i = np.random.uniform(0.05, 0.5, size=n_entities)
    result.u_it = np.random.uniform(0.01, 0.3, size=n_obs)
    result.mu_i = np.random.normal(0, 0.5, size=n_entities)
    result.v_it = np.random.normal(0, 0.1, size=n_obs)

    # Variance components
    result.sigma_v = 0.15
    result.sigma_u = 0.20
    result.sigma_mu = 0.30
    result.sigma_eta = 0.25

    return result


class TestPlotEfficiencyDistributions:
    """Tests for plot_efficiency_distributions."""

    def test_smoke(self, mock_result):
        """Smoke test: efficiency distributions plot."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_efficiency_distributions,
        )

        fig = plot_efficiency_distributions(mock_result)
        assert fig is not None
        assert len(fig.axes) == 3

    def test_custom_params(self, mock_result):
        """Test with custom figsize, bins, alpha."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_efficiency_distributions,
        )

        fig = plot_efficiency_distributions(mock_result, figsize=(16, 6), bins=20, alpha=0.5)
        assert fig is not None


class TestPlotEfficiencyScatter:
    """Tests for plot_efficiency_scatter."""

    def test_smoke(self, mock_result):
        """Smoke test: persistent vs transient scatter."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_efficiency_scatter,
        )

        fig = plot_efficiency_scatter(mock_result)
        assert fig is not None

    def test_with_highlight(self, mock_result):
        """Test scatter with highlighted entities."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_efficiency_scatter,
        )

        fig = plot_efficiency_scatter(mock_result, highlight_entities=[0, 2, 5])
        assert fig is not None


class TestPlotEfficiencyEvolution:
    """Tests for plot_efficiency_evolution."""

    def test_smoke(self, mock_result):
        """Smoke test: evolution with random entities."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_efficiency_evolution,
        )

        fig = plot_efficiency_evolution(mock_result)
        assert fig is not None

    def test_specific_entities(self, mock_result):
        """Test evolution with specified entity IDs."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_efficiency_evolution,
        )

        fig = plot_efficiency_evolution(mock_result, entity_ids=[0, 3, 7])
        assert fig is not None


class TestPlotEntityDecomposition:
    """Tests for plot_entity_decomposition."""

    def test_smoke(self, mock_result):
        """Smoke test: entity decomposition (auto-select top/bottom)."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_entity_decomposition,
        )

        fig = plot_entity_decomposition(mock_result, n_entities=6)
        assert fig is not None

    def test_specific_entities(self, mock_result):
        """Test decomposition with specific entity IDs."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_entity_decomposition,
        )

        fig = plot_entity_decomposition(mock_result, entity_ids=[0, 1, 2])
        assert fig is not None


class TestPlotVarianceDecomposition:
    """Tests for plot_variance_decomposition."""

    def test_smoke(self, mock_result):
        """Smoke test: variance decomposition pie chart."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_variance_decomposition,
        )

        fig = plot_variance_decomposition(mock_result)
        assert fig is not None
        assert len(fig.axes) == 2


class TestPlotComprehensiveSummary:
    """Tests for plot_comprehensive_summary."""

    def test_smoke(self, mock_result):
        """Smoke test: comprehensive multi-panel summary."""
        from panelbox.frontier.visualization.four_component_plots import (
            plot_comprehensive_summary,
        )

        fig = plot_comprehensive_summary(mock_result, n_evolution=3)
        assert fig is not None
