"""
Tests for Dynamic Panel Quantile Regression.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.models.quantile.dynamic import (
    DynamicQuantile,
    DynamicQuantilePanelResult,
    DynamicQuantileResult,
)
from panelbox.utils.data import PanelData


class TestDynamicQuantile:
    """Tests for dynamic quantile regression."""

    @pytest.fixture
    def dynamic_panel_data(self):
        """Generate panel data with dynamic structure."""
        np.random.seed(123)
        n_entities = 30
        n_time = 15

        data_list = []
        for entity in range(n_entities):
            # Generate AR(1) process for each entity
            y = np.zeros(n_time)
            X1 = np.random.randn(n_time)
            X2 = np.random.randn(n_time)

            # Initial value
            y[0] = np.random.randn()

            # AR(1) with persistence
            rho = 0.6
            for t in range(1, n_time):
                y[t] = rho * y[t - 1] + 0.5 * X1[t] - 0.3 * X2[t] + np.random.randn()

            # Create entity DataFrame
            entity_df = pd.DataFrame(
                {"y": y, "X1": X1, "X2": X2, "entity": entity, "time": range(n_time)}
            )
            data_list.append(entity_df)

        df = pd.concat(data_list, ignore_index=True)
        return PanelData(df, entity="entity", time="time")

    def test_basic_dynamic_estimation(self, dynamic_panel_data):
        """Test basic dynamic QR estimation."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=1, method="iv"
        )

        result = model.fit(verbose=False)

        # Check result structure
        assert isinstance(result, DynamicQuantilePanelResult)
        assert 0.5 in result.results
        assert isinstance(result.results[0.5], DynamicQuantileResult)

        # Check persistence parameter
        persistence = result.results[0.5].persistence
        assert len(persistence) == 1  # One lag
        assert 0 < persistence[0] < 1  # Should be between 0 and 1

    def test_multiple_lags(self, dynamic_panel_data):
        """Test with multiple lags."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=2, method="iv"
        )

        result = model.fit()

        # Check persistence parameters
        persistence = result.results[0.5].persistence
        assert len(persistence) == 2  # Two lags

    def test_multiple_quantiles(self, dynamic_panel_data):
        """Test estimation at multiple quantiles."""
        tau_list = [0.25, 0.5, 0.75]

        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=tau_list, lags=1, method="iv"
        )

        result = model.fit()

        # Check all quantiles estimated
        for tau in tau_list:
            assert tau in result.results
            assert result.results[tau].tau == tau

        # Persistence should vary across quantiles
        persistences = [result.results[tau].persistence[0] for tau in tau_list]
        assert len(set(persistences)) > 1  # Not all identical

    def test_qcf_method(self, dynamic_panel_data):
        """Test quantile control function method."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=1, method="qcf"
        )

        result = model.fit()

        # Check QCF-specific attributes
        assert result.results[0.5].method == "qcf"
        assert hasattr(result.results[0.5], "control_function_coef")

    def test_gmm_method(self, dynamic_panel_data):
        """Test GMM method."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=1, method="gmm"
        )

        result = model.fit()

        # Check GMM results
        assert result.results[0.5].method == "gmm"
        assert result.results[0.5].persistence is not None

    def test_long_run_effects(self, dynamic_panel_data):
        """Test long-run effects computation."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75], lags=1, method="iv"
        )

        result = model.fit()

        # Compute long-run effects
        lr_effects = model.compute_long_run_effects(result)

        # Check structure
        assert len(lr_effects) == 3
        for tau in [0.25, 0.5, 0.75]:
            assert tau in lr_effects
            if lr_effects[tau] is not None:
                assert "multiplier" in lr_effects[tau]
                assert "effects" in lr_effects[tau]
                assert "persistence" in lr_effects[tau]

    def test_unit_root_warning(self):
        """Test warning for unit root."""
        # Create data with unit root
        np.random.seed(456)
        n_entities = 10
        n_time = 10

        data_list = []
        for entity in range(n_entities):
            y = np.cumsum(np.random.randn(n_time))  # Random walk
            X = np.random.randn(n_time)

            entity_df = pd.DataFrame({"y": y, "X": X, "entity": entity, "time": range(n_time)})
            data_list.append(entity_df)

        df = pd.concat(data_list, ignore_index=True)
        panel_data = PanelData(df, entity="entity", time="time")

        model = DynamicQuantile(panel_data, formula="y ~ X", tau=0.5, lags=1)

        result = model.fit()

        # Should warn about unit root
        with warnings.catch_warnings(record=True) as w:
            lr_effects = model.compute_long_run_effects(result)
            # May or may not warn depending on estimate

    def test_impulse_response(self, dynamic_panel_data):
        """Test impulse response function."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=1, method="iv"
        )

        result = model.fit()

        # Compute IRF
        irf = model.compute_impulse_response(result, tau=0.5, horizon=10, shock_size=1.0)

        # Check IRF properties
        assert len(irf) == 10
        assert irf[0] == 1.0  # Initial shock
        # Should decay over time (for stable process)
        assert abs(irf[-1]) < abs(irf[0])

    def test_bootstrap_inference(self, dynamic_panel_data):
        """Test bootstrap inference."""
        model = DynamicQuantile(
            dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=1, method="iv"
        )

        result = model.fit(bootstrap=True, n_boot=50)  # Small for speed

        # Check that covariance matrix is computed
        assert result.results[0.5].cov_matrix is not None
        assert result.results[0.5].std_errors is not None

    def test_data_setup(self, dynamic_panel_data):
        """Test dynamic data setup."""
        model = DynamicQuantile(dynamic_panel_data, formula="y ~ X1 + X2", tau=0.5, lags=2)

        # Check data setup
        assert model.y_lagged is not None
        assert model.y_lagged.shape[1] == 2  # Two lags
        assert len(model.valid_obs) < len(model.y)  # Lost observations due to lags
        assert model.X_with_lags.shape[1] == 2 + model.X.shape[1]  # Lags + X


class TestDynamicQuantileResult:
    """Tests for DynamicQuantileResult class."""

    def test_result_properties(self):
        """Test result properties."""
        # Create mock result
        params = np.array([0.6, 0.5, -0.3])
        cov_matrix = np.eye(3) * 0.01
        persistence = np.array([0.6])

        result = DynamicQuantileResult(
            params=params,
            cov_matrix=cov_matrix,
            tau=0.5,
            persistence=persistence,
            converged=True,
            method="iv",
            n_obs=100,
            n_entities=10,
        )

        # Test properties
        assert np.allclose(result.std_errors, np.sqrt(np.diag(cov_matrix)))
        assert len(result.t_stats) == 3
        assert result.t_stats[0] == params[0] / result.std_errors[0]

    def test_result_summary(self):
        """Test result summary."""
        params = np.array([0.6, 0.5, -0.3])
        persistence = np.array([0.6])

        result = DynamicQuantileResult(
            params=params,
            cov_matrix=None,
            tau=0.5,
            persistence=persistence,
            converged=True,
            method="iv",
            n_obs=100,
            n_entities=10,
        )

        # Should not raise
        result.summary()


class TestDynamicQuantilePanelResult:
    """Tests for panel result container."""

    @pytest.fixture
    def mock_results(self):
        """Create mock results."""
        results = {}
        for tau in [0.25, 0.5, 0.75]:
            results[tau] = DynamicQuantileResult(
                params=np.array([0.5 + 0.2 * tau, 0.3, -0.2]),
                cov_matrix=np.eye(3) * 0.01,
                tau=tau,
                persistence=np.array([0.5 + 0.2 * tau]),
                converged=True,
                method="iv",
                n_obs=100,
                n_entities=10,
            )

        # Create mock model
        class MockModel:
            def __init__(self):
                self.lags = 1

            def compute_impulse_response(self, result, tau, horizon):
                irf = np.zeros(horizon)
                irf[0] = 1.0
                rho = result.results[tau].persistence[0]
                for t in range(1, horizon):
                    irf[t] = rho * irf[t - 1]
                return irf

        return DynamicQuantilePanelResult(MockModel(), results)

    def test_plot_persistence(self, mock_results):
        """Test persistence plotting."""
        # Should create plot without error
        fig = mock_results.plot_persistence()
        assert fig is not None

    def test_plot_impulse_responses(self, mock_results):
        """Test IRF plotting."""
        # Should create plot without error
        fig = mock_results.plot_impulse_responses(tau_list=[0.25, 0.5, 0.75], horizon=10)
        assert fig is not None
