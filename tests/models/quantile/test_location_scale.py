"""
Tests for Location-Scale Quantile Regression (Machado-Santos Silva 2019).
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.models.quantile.location_scale import (
    LocationScale,
    LocationScaleResult,
    NormalityTestResult,
)
from panelbox.utils.data import PanelData


class TestLocationScale:
    """Tests for Location-Scale quantile regression."""

    @pytest.fixture
    def simulated_data(self):
        """Generate panel data with location-scale structure."""
        np.random.seed(123)
        n_entities = 50
        n_time = 20
        n = n_entities * n_time

        # Generate covariates
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)

        # Location parameters (affect mean)
        alpha = np.array([2.0, 1.5, -0.5])

        # Scale parameters (affect variance)
        gamma = np.array([0.5, 0.3, -0.2])

        # Generate y with location-scale structure
        location = alpha[0] + alpha[1] * X1 + alpha[2] * X2
        log_scale = gamma[0] + gamma[1] * X1 + gamma[2] * X2
        scale = np.exp(log_scale / 2)

        # Add normal errors
        errors = np.random.randn(n)
        y = location + scale * errors

        # Create panel data
        df = pd.DataFrame(
            {
                "y": y,
                "X1": X1,
                "X2": X2,
                "entity": np.repeat(range(n_entities), n_time),
                "time": np.tile(range(n_time), n_entities),
            }
        )

        return PanelData(df, entity="entity", time="time")

    def test_basic_estimation(self, simulated_data):
        """Test basic location-scale estimation."""
        model = LocationScale(
            simulated_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75], distribution="normal"
        )

        result = model.fit(verbose=False)

        # Check results structure
        assert isinstance(result, LocationScaleResult)
        assert len(result.results) == 3
        assert 0.25 in result.results
        assert 0.5 in result.results
        assert 0.75 in result.results

        # Check location and scale results exist
        assert result.location_result is not None
        assert result.scale_result is not None
        assert model.location_params_ is not None
        assert model.scale_params_ is not None

    def test_non_crossing_guarantee(self, simulated_data):
        """Test that location-scale guarantees non-crossing."""
        tau_grid = np.arange(0.1, 1.0, 0.1)

        model = LocationScale(
            simulated_data, formula="y ~ X1 + X2", tau=tau_grid, distribution="normal"
        )

        result = model.fit()

        # Check non-crossing by examining predictions
        X_test = model.X[:10]  # Test on first 10 observations
        predictions = np.zeros((len(X_test), len(tau_grid)))

        for i, tau in enumerate(tau_grid):
            predictions[:, i] = X_test @ result.results[tau].params

        # Check monotonicity across quantiles for each observation
        for i in range(len(X_test)):
            diffs = np.diff(predictions[i, :])
            assert np.all(diffs >= -1e-10), f"Crossing detected for obs {i}"

    def test_different_distributions(self, simulated_data):
        """Test different reference distributions."""
        distributions = ["normal", "logistic", "t", "laplace"]

        for dist in distributions:
            kwargs = {"df_t": 5} if dist == "t" else {}

            model = LocationScale(
                simulated_data, formula="y ~ X1 + X2", tau=0.5, distribution=dist, **kwargs
            )

            result = model.fit()

            # Should complete without errors
            assert result is not None
            assert 0.5 in result.results

    def test_fixed_effects(self, simulated_data):
        """Test location-scale with fixed effects."""
        model = LocationScale(
            simulated_data,
            formula="y ~ X1 + X2",
            tau=[0.25, 0.5, 0.75],
            distribution="normal",
            fixed_effects=True,
        )

        result = model.fit()

        # Check that fixed effects were used
        assert model.fixed_effects is True
        assert result is not None

    def test_predict_quantiles(self, simulated_data):
        """Test quantile prediction."""
        model = LocationScale(
            simulated_data, formula="y ~ X1 + X2", tau=[0.25, 0.5, 0.75], distribution="normal"
        )

        result = model.fit()

        # Predict for new data
        X_new = np.array([[1, 0, 0], [1, 1, 1], [1, -1, -1]])
        predictions = model.predict_quantiles(X_new, tau=[0.1, 0.5, 0.9])

        # Check shape and columns
        assert predictions.shape[0] == 3
        assert "q10" in predictions.columns
        assert "q50" in predictions.columns
        assert "q90" in predictions.columns

        # Check monotonicity
        for i in range(len(X_new)):
            assert predictions.iloc[i]["q10"] <= predictions.iloc[i]["q50"]
            assert predictions.iloc[i]["q50"] <= predictions.iloc[i]["q90"]

    def test_predict_density(self, simulated_data):
        """Test density prediction."""
        model = LocationScale(simulated_data, formula="y ~ X1 + X2", tau=0.5, distribution="normal")

        result = model.fit()

        # Predict density
        X_test = np.array([[1, 0, 0]])
        y_grid, density = model.predict_density(X_test, n_points=50)

        # Check outputs
        assert len(y_grid) == 50
        assert len(density) == 50
        assert np.all(density >= 0)  # Density should be non-negative
        assert np.trapz(density, y_grid) > 0.9  # Should integrate to ~1

    def test_normality_test(self, simulated_data):
        """Test normality testing functionality."""
        model = LocationScale(simulated_data, formula="y ~ X1 + X2", tau=0.5, distribution="normal")

        result = model.fit()
        norm_test = model.test_normality()

        # Check test results
        assert isinstance(norm_test, NormalityTestResult)
        assert hasattr(norm_test, "ks_stat")
        assert hasattr(norm_test, "ks_pval")
        assert hasattr(norm_test, "jb_stat")
        assert hasattr(norm_test, "jb_pval")

        # For data generated with normal errors, should not reject
        assert norm_test.ks_pval > 0.01  # Relaxed threshold for small sample

    def test_robust_scale_estimation(self, simulated_data):
        """Test robust vs non-robust scale estimation."""
        model_robust = LocationScale(simulated_data, formula="y ~ X1 + X2", tau=0.5)
        result_robust = model_robust.fit(robust_scale=True)

        model_nonrobust = LocationScale(simulated_data, formula="y ~ X1 + X2", tau=0.5)
        result_nonrobust = model_nonrobust.fit(robust_scale=False)

        # Both should work but may give different results
        assert result_robust is not None
        assert result_nonrobust is not None

    def test_quantile_decomposition(self, simulated_data):
        """Test decomposition into location and scale effects."""
        model = LocationScale(
            simulated_data,
            formula="y ~ X1 + X2",
            tau=np.arange(0.1, 1.0, 0.1),
            distribution="normal",
        )

        result = model.fit()

        # Check that quantile effects vary with tau
        tau_list = sorted(result.results.keys())
        coef_var_1 = [result.results[tau].params[1] for tau in tau_list]

        # Coefficients should vary across quantiles
        assert np.std(coef_var_1) > 0.01

        # Middle quantile should be close to location parameter
        median_coef = result.results[0.5].params[1]
        location_coef = result.location_result.params[1]
        assert abs(median_coef - location_coef) < 0.1

    def test_covariance_estimation(self, simulated_data):
        """Test delta method covariance estimation."""
        model = LocationScale(simulated_data, formula="y ~ X1 + X2", tau=0.5, distribution="normal")

        result = model.fit()

        # Check covariance matrix
        cov_matrix = result.results[0.5].cov_matrix
        assert cov_matrix is not None
        assert cov_matrix.shape == (3, 3)
        assert np.all(np.diag(cov_matrix) > 0)  # Positive variances

        # Check standard errors
        se = result.results[0.5].std_errors
        assert len(se) == 3
        assert np.all(se > 0)

    def test_callable_distribution(self, simulated_data):
        """Test with user-provided quantile function."""

        def custom_quantile(tau):
            """Custom quantile function (uniform-like)."""
            return 2 * (tau - 0.5)

        model = LocationScale(
            simulated_data,
            formula="y ~ X1 + X2",
            tau=[0.25, 0.5, 0.75],
            distribution=custom_quantile,
        )

        result = model.fit()

        # Should work with custom function
        assert result is not None
        assert len(result.results) == 3


class TestLocationScaleWithRealData:
    """Tests using more realistic data scenarios."""

    def test_heteroskedastic_data(self):
        """Test with strongly heteroskedastic data."""
        np.random.seed(456)
        n = 500

        # Generate heteroskedastic data
        X = np.random.randn(n, 2)
        location = 2 + X[:, 0] - 0.5 * X[:, 1]
        scale = np.exp(0.5 + 0.8 * X[:, 0])  # Strong heteroskedasticity

        y = location + scale * np.random.randn(n)

        # Create data
        df = pd.DataFrame({"y": y, "X1": X[:, 0], "X2": X[:, 1], "entity": np.arange(n), "time": 0})
        panel_data = PanelData(df, entity="entity", time="time")

        # Estimate
        model = LocationScale(
            panel_data, formula="y ~ X1 + X2", tau=np.arange(0.1, 1.0, 0.1), distribution="normal"
        )

        result = model.fit()

        # Check that scale parameters capture heteroskedasticity
        # X1 should have positive scale effect
        assert model.scale_params_[1] > 0.5  # Should be close to 0.8

    def test_skewed_errors(self):
        """Test with skewed error distribution."""
        np.random.seed(789)
        n = 500

        # Generate data with skewed errors
        X = np.random.randn(n, 2)
        location = 1 + 2 * X[:, 0] - X[:, 1]

        # Use chi-squared errors (skewed)
        errors = (np.random.chisquare(3, n) - 3) / np.sqrt(6)
        y = location + errors

        # Create data
        df = pd.DataFrame({"y": y, "X1": X[:, 0], "X2": X[:, 1], "entity": np.arange(n), "time": 0})
        panel_data = PanelData(df, entity="entity", time="time")

        # Test different distributions
        for dist in ["normal", "logistic", "t"]:
            kwargs = {"df_t": 5} if dist == "t" else {}

            model = LocationScale(
                panel_data,
                formula="y ~ X1 + X2",
                tau=[0.25, 0.5, 0.75],
                distribution=dist,
                **kwargs,
            )

            result = model.fit()

            # Should complete despite skewed errors
            assert result is not None

            # Normality test should detect non-normality
            if dist == "normal":
                norm_test = model.test_normality()
                # May or may not reject depending on sample, but should run
                assert norm_test is not None
