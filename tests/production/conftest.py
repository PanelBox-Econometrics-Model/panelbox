"""Shared fixtures for production module tests."""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.production.pipeline import PanelPipeline


@pytest.fixture
def sample_panel_data():
    """Create sample panel data for testing."""
    np.random.seed(42)
    n_entities, n_periods = 10, 5
    entity = np.repeat(range(n_entities), n_periods)
    time = np.tile(range(n_periods), n_entities)
    x1 = np.random.randn(n_entities * n_periods)
    x2 = np.random.randn(n_entities * n_periods)
    y = 1.5 * x1 - 0.5 * x2 + np.random.randn(n_entities * n_periods) * 0.5
    df = pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})
    return df


@pytest.fixture
def fitted_pipeline(sample_panel_data):
    """Create a fitted PanelPipeline for testing."""
    pipeline = PanelPipeline(
        model_class=PooledOLS,
        model_params={
            "formula": "y ~ x1 + x2",
            "entity_col": "entity",
            "time_col": "time",
        },
    )
    pipeline.fit(sample_panel_data)
    return pipeline
