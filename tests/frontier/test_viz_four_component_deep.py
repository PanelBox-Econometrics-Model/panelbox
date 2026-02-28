"""
Deep coverage tests for frontier/visualization/four_component_plots.py.

Targets the single uncovered branch partial to push coverage
from ~99.55% toward 100%.

Uncovered item:
- Branch 149->148: entity_id >= len(te_persistent) in highlight_entities
"""

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from panelbox.frontier.visualization.four_component_plots import (
    plot_efficiency_scatter,
)


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

    entity_ids = np.repeat(np.arange(n_entities), n_periods)
    time_ids = np.tile(np.arange(n_periods), n_entities)
    result.model.entity_id = entity_ids
    result.model.time_id = time_ids
    result.model.n_entities = n_entities

    result.eta_i = np.random.uniform(0.05, 0.5, size=n_entities)
    result.u_it = np.random.uniform(0.01, 0.3, size=n_obs)
    result.mu_i = np.random.normal(0, 0.5, size=n_entities)
    result.v_it = np.random.normal(0, 0.1, size=n_obs)

    result.sigma_v = 0.15
    result.sigma_u = 0.20
    result.sigma_mu = 0.30
    result.sigma_eta = 0.25

    return result


class TestPlotEfficiencyScatterBranch:
    """Test uncovered branch in plot_efficiency_scatter."""

    def test_highlight_out_of_bounds_entity(self, mock_result):
        """Cover branch 149->148: entity_id >= len(te_persistent).

        When an entity_id in highlight_entities is out of bounds,
        the inner if block should be skipped.
        """
        fig = plot_efficiency_scatter(
            mock_result,
            highlight_entities=[0, 999],  # 999 is out of bounds
        )
        assert fig is not None
