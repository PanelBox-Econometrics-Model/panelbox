"""
Tests for predict(newdata) on Static and IV models.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.between import BetweenEstimator
from panelbox.models.static.first_difference import FirstDifferenceEstimator
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.random_effects import RandomEffects


@pytest.fixture
def panel_data():
    """Generate simple panel data for testing."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 20

    entities = np.repeat(range(n_entities), n_periods)
    times = np.tile(range(n_periods), n_entities)

    # Entity fixed effects
    entity_fe = np.repeat(np.random.randn(n_entities) * 2, n_periods)

    x1 = np.random.randn(n_entities * n_periods)
    x2 = np.random.randn(n_entities * n_periods)
    y = 1.0 + 2.0 * x1 + 3.0 * x2 + entity_fe + np.random.randn(n_entities * n_periods) * 0.5

    return pd.DataFrame(
        {
            "entity": entities,
            "time": times,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )


class TestPooledOLSPredict:
    def test_predict_no_args_returns_fitted(self, panel_data):
        model = PooledOLS("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        preds = results.predict()
        assert np.allclose(preds, results.fittedvalues)

    def test_predict_newdata(self, panel_data):
        model = PooledOLS("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        newdata = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [0, 0]})
        preds = results.predict(newdata)
        assert len(preds) == 2
        assert not np.any(np.isnan(preds))

    def test_predict_insample_close_to_fitted(self, panel_data):
        """predict(training_data) should be close to fittedvalues."""
        model = PooledOLS("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        preds = results.predict(panel_data)
        # For PooledOLS, should be very close (same data)
        assert np.allclose(preds, results.fittedvalues, atol=1e-10)


class TestFixedEffectsPredict:
    def test_predict_no_args_returns_fitted(self, panel_data):
        model = FixedEffects("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        preds = results.predict()
        assert np.allclose(preds, results.fittedvalues)

    def test_predict_newdata_known_entities(self, panel_data):
        """predict with entities that were in training data."""
        model = FixedEffects("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        newdata = pd.DataFrame(
            {
                "entity": [0, 1, 2],
                "time": [100, 100, 100],
                "x1": [1.0, 2.0, 3.0],
                "x2": [0.5, 0.5, 0.5],
                "y": [0, 0, 0],
            }
        )
        preds = results.predict(newdata)
        assert len(preds) == 3
        assert not np.any(np.isnan(preds))

    def test_predict_new_entity_uses_mean(self, panel_data):
        """predict with unknown entity falls back to overall mean."""
        model = FixedEffects("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        newdata = pd.DataFrame(
            {
                "entity": [999],  # entity not in training
                "time": [100],
                "x1": [1.0],
                "x2": [0.5],
                "y": [0],
            }
        )
        preds = results.predict(newdata)
        assert len(preds) == 1
        assert not np.isnan(preds[0])

    def test_predict_insample_close_to_fitted(self, panel_data):
        """predict(training_data) should be close to fittedvalues."""
        model = FixedEffects("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        preds = results.predict(panel_data)
        assert np.allclose(preds, results.fittedvalues, atol=1e-6)


class TestRandomEffectsPredict:
    def test_predict_newdata(self, panel_data):
        model = RandomEffects("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        newdata = pd.DataFrame({"x1": [1.0], "x2": [0.5], "y": [0]})
        preds = results.predict(newdata)
        assert len(preds) == 1
        assert not np.isnan(preds[0])


class TestFirstDifferencePredict:
    def test_predict_newdata(self, panel_data):
        model = FirstDifferenceEstimator(
            "y ~ x1 + x2", panel_data, entity_col="entity", time_col="time"
        )
        results = model.fit()
        # FD predict requires sequential data per entity
        newdata = pd.DataFrame(
            {
                "entity": [0, 0, 0],
                "time": [100, 101, 102],
                "x1": [1.0, 2.0, 3.0],
                "x2": [0.5, 0.6, 0.7],
                "y": [0, 0, 0],
            }
        )
        preds = results.predict(newdata)
        # FD drops first obs per group, so 2 predictions
        assert len(preds) == 2
        assert not np.any(np.isnan(preds))


class TestBetweenPredict:
    def test_predict_newdata(self, panel_data):
        model = BetweenEstimator("y ~ x1 + x2", panel_data, entity_col="entity", time_col="time")
        results = model.fit()
        newdata = pd.DataFrame(
            {
                "entity": [0, 0, 1, 1],
                "time": [100, 101, 100, 101],
                "x1": [1.0, 2.0, 3.0, 4.0],
                "x2": [0.5, 0.6, 0.7, 0.8],
                "y": [0, 0, 0, 0],
            }
        )
        preds = results.predict(newdata)
        # Between returns one prediction per entity
        assert len(preds) == 2  # 2 entities
