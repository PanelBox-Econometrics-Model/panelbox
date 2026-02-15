"""
Integration tests for spatial econometrics with PanelExperiment.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.experiment import PanelExperiment
from panelbox.experiment.spatial_extension import extend_panel_experiment

# Ensure spatial extension is loaded
extend_panel_experiment()


class TestSpatialIntegration:
    """Test spatial integration with PanelExperiment."""

    @pytest.fixture
    def spatial_panel_data(self):
        """Create simulated spatial panel data."""
        np.random.seed(42)

        # Panel dimensions
        n_entities = 50
        n_periods = 10
        n_obs = n_entities * n_periods

        # Create panel structure
        entity_ids = np.repeat(range(n_entities), n_periods)
        time_ids = np.tile(range(n_periods), n_entities)

        # Create spatial weight matrix
        W = np.zeros((n_entities, n_entities))
        for i in range(n_entities):
            if i > 0:
                W[i, i - 1] = 1
            if i < n_entities - 1:
                W[i, i + 1] = 1

        # Row-standardize
        W = W / W.sum(axis=1, keepdims=True)
        W[np.isnan(W)] = 0

        # Generate data
        X = np.random.randn(n_obs, 3)
        y = X @ np.array([2.0, -1.5, 0.8]) + np.random.randn(n_obs)

        # Create DataFrame
        data = pd.DataFrame(
            {
                "entity_id": entity_ids,
                "time_id": time_ids,
                "y": y,
                "x1": X[:, 0],
                "x2": X[:, 1],
                "x3": X[:, 2],
            }
        )

        return data, W

    def test_spatial_experiment_workflow(self, spatial_panel_data):
        """Test complete spatial workflow with PanelExperiment."""
        data, W = spatial_panel_data

        # Create experiment
        experiment = PanelExperiment(
            data=data, formula="y ~ x1 + x2 + x3", entity_col="entity_id", time_col="time_id"
        )

        # Step 1: Estimate baseline OLS
        experiment.fit_model("pooled_ols", name="ols")

        # Step 2: Run spatial diagnostics
        W_obj = SpatialWeights(W)
        diagnostics = experiment.run_spatial_diagnostics(W_obj, "ols")

        # Check diagnostics structure
        assert "moran" in diagnostics
        assert "recommendation" in diagnostics

        # Step 3: Add spatial models
        experiment.add_spatial_model("SAR-FE", W_obj, "sar", effects="fixed")

        # Check model was added
        assert "SAR-FE" in experiment.list_models()
