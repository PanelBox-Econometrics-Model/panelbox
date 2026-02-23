"""
Tests for GMM predict() and forecast().
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm import DifferenceGMM
from panelbox.gmm.results import GMMResults


@pytest.fixture
def gmm_data():
    """Generate simple dynamic panel data for GMM testing."""
    np.random.seed(42)
    n_entities = 50
    n_periods = 10

    entities = np.repeat(range(n_entities), n_periods)
    times = np.tile(range(n_periods), n_entities)

    # Entity FE
    fe = np.repeat(np.random.randn(n_entities), n_periods)

    # AR(1) process with exogenous
    x1 = np.random.randn(n_entities * n_periods) + fe * 0.3
    y = np.zeros(n_entities * n_periods)

    for i in range(n_entities):
        start = i * n_periods
        y[start] = fe[start] + np.random.randn()
        for t in range(1, n_periods):
            idx = start + t
            y[idx] = 0.5 * y[idx - 1] + 0.3 * x1[idx] + fe[idx] + np.random.randn() * 0.5

    return pd.DataFrame(
        {
            "id": entities,
            "time": times,
            "y": y,
            "x1": x1,
        }
    )


class TestGMMPredict:
    def test_predict_returns_valid(self, gmm_data):
        """predict() returns non-NaN predictions."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()
        preds = results.predict(gmm_data)
        # First period per entity will be NaN (no lag available)
        valid = ~np.isnan(preds)
        assert valid.sum() > 0

    def test_predict_dimensions(self, gmm_data):
        """predict() returns same length as input."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()
        preds = results.predict(gmm_data)
        assert len(preds) == len(gmm_data)

    def test_predict_metadata_stored(self, gmm_data):
        """GMMResults stores model metadata after fit."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()
        assert results.dep_var == "y"
        assert results.exog_vars == ["x1"]
        assert results.id_var == "id"
        assert results.time_var == "time"
        assert results.n_lags == 1

    def test_predict_missing_variable_raises(self, gmm_data):
        """predict() raises ValueError when variable is missing."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()
        bad_data = gmm_data.drop(columns=["x1"])
        with pytest.raises(ValueError, match="Exogenous variable 'x1' not found"):
            results.predict(bad_data)

    def test_predict_auto_computes_lags(self, gmm_data):
        """predict() computes lags automatically from dep_var column."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()
        # new_data has 'y' but not 'L1.y' — should auto-compute lag
        preds = results.predict(gmm_data)
        assert len(preds) == len(gmm_data)
        # First obs per entity should be NaN (lag not available)
        first_idx = gmm_data.groupby("id").head(1).index
        assert np.all(np.isnan(preds[first_idx]))

    def test_predict_with_time_dummies(self, gmm_data):
        """predict() handles time dummies correctly."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=True,
            collapse=True,
        )
        results = model.fit()
        preds = results.predict(gmm_data)
        assert len(preds) == len(gmm_data)
        valid = ~np.isnan(preds)
        assert valid.sum() > 0


class TestGMMForecast:
    def test_forecast_single_step(self, gmm_data):
        """forecast() 1 step ahead."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()

        last_obs = {0: [gmm_data[gmm_data["id"] == 0]["y"].iloc[-1]]}
        future_exog = pd.DataFrame(
            {
                "id": [0],
                "time": [10],
                "x1": [0.5],
            }
        )
        forecasts = results.forecast(last_obs, future_exog, steps=1)
        assert len(forecasts) == 1
        assert "forecast" in forecasts.columns
        assert not np.isnan(forecasts["forecast"].iloc[0])

    def test_forecast_multi_step(self, gmm_data):
        """forecast() multiple steps feeds back predictions."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()

        last_obs = {0: [1.0], 1: [2.0]}
        future_exog = pd.DataFrame(
            {
                "id": [0, 0, 0, 1, 1, 1],
                "time": [10, 11, 12, 10, 11, 12],
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        forecasts = results.forecast(last_obs, future_exog, steps=3)
        assert len(forecasts) == 6  # 3 steps x 2 entities
        assert not forecasts["forecast"].isna().any()

    def test_forecast_returns_correct_columns(self, gmm_data):
        """forecast() returns DataFrame with [id_var, time_var, 'forecast']."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()

        last_obs = {0: [1.0]}
        future_exog = pd.DataFrame(
            {
                "id": [0],
                "time": [10],
                "x1": [0.5],
            }
        )
        forecasts = results.forecast(last_obs, future_exog, steps=1)
        assert set(forecasts.columns) == {"id", "time", "forecast"}


class TestGMMSaveLoad:
    def test_save_load_roundtrip(self, gmm_data):
        """save/load preserves params and metadata."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = GMMResults.load(f.name)

        assert np.allclose(results.params.values, loaded.params.values)
        assert loaded.dep_var == "y"
        assert loaded.exog_vars == ["x1"]
        assert loaded.id_var == "id"
        assert loaded.time_var == "time"
        assert loaded.n_lags == 1

    def test_save_load_preserves_diagnostics(self, gmm_data):
        """save/load preserves diagnostic test results."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = GMMResults.load(f.name)

        assert loaded.hansen_j.statistic == results.hansen_j.statistic
        assert loaded.ar2_test.pvalue == results.ar2_test.pvalue
        assert loaded.nobs == results.nobs
        assert loaded.n_groups == results.n_groups

    def test_loaded_model_can_predict(self, gmm_data):
        """Loaded model retains predict capability."""
        model = DifferenceGMM(
            data=gmm_data,
            dep_var="y",
            lags=1,
            exog_vars=["x1"],
            id_var="id",
            time_var="time",
            time_dummies=False,
            collapse=True,
        )
        results = model.fit()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            results.save(f.name)
            loaded = GMMResults.load(f.name)

        preds_original = results.predict(gmm_data)
        preds_loaded = loaded.predict(gmm_data)
        assert np.allclose(
            preds_original[~np.isnan(preds_original)],
            preds_loaded[~np.isnan(preds_loaded)],
        )
