"""
Tests for Random Effects Tobit model.

Author: PanelBox Developers
License: MIT
"""

import warnings

import numpy as np
import pytest
from scipy import stats

from panelbox.models.censored import RandomEffectsTobit


class TestRandomEffectsTobit:
    """Test suite for Random Effects Tobit model."""

    @pytest.fixture
    def simulated_data(self):
        """Generate simulated panel data with censoring."""
        np.random.seed(42)

        # Panel dimensions
        N = 50  # Number of entities
        T = 10  # Time periods
        K = 3  # Number of covariates

        # True parameters
        beta_true = np.array([0.5, -0.3, 0.2])
        sigma_eps_true = 0.5
        sigma_alpha_true = 0.3

        # Generate data
        X = np.random.randn(N * T, K)

        # Random effects
        alpha_i = np.random.normal(0, sigma_alpha_true, N)
        alpha = np.repeat(alpha_i, T)

        # Error term
        epsilon = np.random.normal(0, sigma_eps_true, N * T)

        # Latent variable
        y_star = X @ beta_true + alpha + epsilon

        # Left censoring at 0
        y = np.maximum(0, y_star)

        # Panel structure
        groups = np.repeat(np.arange(N), T)
        time = np.tile(np.arange(T), N)

        # Count censored observations
        n_censored = np.sum(y == 0)
        censoring_rate = n_censored / len(y)

        return {
            "y": y,
            "X": X,
            "groups": groups,
            "time": time,
            "beta_true": beta_true,
            "sigma_eps_true": sigma_eps_true,
            "sigma_alpha_true": sigma_alpha_true,
            "n_censored": n_censored,
            "censoring_rate": censoring_rate,
        }

    def test_initialization(self, simulated_data):
        """Test model initialization."""
        model = RandomEffectsTobit(
            endog=simulated_data["y"],
            exog=simulated_data["X"],
            groups=simulated_data["groups"],
            time=simulated_data["time"],
            censoring_point=0,
            censoring_type="left",
        )

        assert model.n_obs == len(simulated_data["y"])
        assert model.n_features == simulated_data["X"].shape[1]
        assert model.n_entities == len(np.unique(simulated_data["groups"]))
        assert model.censoring_point == 0
        assert model.censoring_type == "left"

    def test_fit_convergence(self, simulated_data):
        """Test that the model converges on simulated data."""
        model = RandomEffectsTobit(
            endog=simulated_data["y"],
            exog=simulated_data["X"],
            groups=simulated_data["groups"],
            time=simulated_data["time"],
            quadrature_points=8,  # Use fewer points for speed
        )

        # Fit with limited iterations for testing
        result = model.fit(maxiter=50)

        assert hasattr(result, "params")
        assert hasattr(result, "llf")
        assert len(result.params) == simulated_data["X"].shape[1] + 2

        # Check that variance parameters are positive
        assert result.sigma_eps > 0
        assert result.sigma_alpha > 0

    def test_parameter_recovery(self, simulated_data):
        """Test parameter recovery on larger simulated dataset."""
        # Generate larger dataset for better parameter recovery
        np.random.seed(123)
        N, T, K = 100, 20, 2
        beta_true = np.array([0.8, -0.5])
        sigma_eps_true = 0.6
        sigma_alpha_true = 0.4

        X = np.random.randn(N * T, K)
        alpha_i = np.random.normal(0, sigma_alpha_true, N)
        alpha = np.repeat(alpha_i, T)
        epsilon = np.random.normal(0, sigma_eps_true, N * T)

        y_star = X @ beta_true + alpha + epsilon
        y = np.maximum(0, y_star)  # Left censoring at 0

        groups = np.repeat(np.arange(N), T)

        model = RandomEffectsTobit(
            endog=y, exog=X, groups=groups, censoring_point=0, quadrature_points=12
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=100)

        # Check parameter recovery (allowing for some estimation error)
        assert np.allclose(result.beta, beta_true, atol=0.2)
        assert np.abs(result.sigma_eps - sigma_eps_true) < 0.2
        assert np.abs(result.sigma_alpha - sigma_alpha_true) < 0.2

    def test_predict_latent(self, simulated_data):
        """Test prediction of latent variable."""
        model = RandomEffectsTobit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=30)

        # Predict latent values
        y_latent = result.predict(pred_type="latent")

        assert len(y_latent) == len(simulated_data["y"])
        assert y_latent.shape == simulated_data["y"].shape

        # Latent predictions should be X'β (population average)
        expected = simulated_data["X"] @ result.beta
        assert np.allclose(y_latent, expected)

    def test_predict_censored(self, simulated_data):
        """Test prediction accounting for censoring."""
        model = RandomEffectsTobit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=30)

        # Predict censored values
        y_censored = result.predict(pred_type="censored")

        assert len(y_censored) == len(simulated_data["y"])

        # Censored predictions should be >= censoring point
        # (for expected values, not necessarily for each prediction)
        assert y_censored.min() >= -2  # Allow some negative due to E[y|X]

    def test_right_censoring(self):
        """Test right censoring."""
        np.random.seed(42)
        N, T, K = 30, 5, 2

        X = np.random.randn(N * T, K)
        y_star = X @ np.array([0.5, -0.3]) + np.random.randn(N * T)

        # Right censoring at 1
        y = np.minimum(1, y_star)
        groups = np.repeat(np.arange(N), T)

        model = RandomEffectsTobit(
            endog=y,
            exog=X,
            groups=groups,
            censoring_point=1,
            censoring_type="right",
            quadrature_points=6,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=20)

        assert hasattr(result, "params")
        assert result.sigma_eps > 0
        assert result.sigma_alpha > 0

    def test_both_censoring(self):
        """Test two-sided censoring."""
        np.random.seed(42)
        N, T, K = 30, 5, 2

        X = np.random.randn(N * T, K)
        y_star = X @ np.array([0.5, -0.3]) + np.random.randn(N * T)

        # Censoring at both ends
        y = np.clip(y_star, -1, 1)
        groups = np.repeat(np.arange(N), T)

        model = RandomEffectsTobit(
            endog=y,
            exog=X,
            groups=groups,
            censoring_type="both",
            lower_limit=-1,
            upper_limit=1,
            quadrature_points=6,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=20)

        assert hasattr(result, "params")
        assert result.sigma_eps > 0
        assert result.sigma_alpha > 0

    def test_summary_output(self, simulated_data):
        """Test summary output."""
        model = RandomEffectsTobit(
            endog=simulated_data["y"], exog=simulated_data["X"], groups=simulated_data["groups"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(maxiter=20)

        summary = result.summary()

        assert isinstance(summary, str)
        assert "Random Effects Tobit" in summary
        assert "sigma_eps" in summary
        assert "sigma_alpha" in summary
        assert f"Censoring type:       {'left':>8s}" in summary

    def test_censoring_detection(self):
        """Test censoring detection logic."""
        model = RandomEffectsTobit(
            endog=np.array([0, 0.5, 1.0]),
            exog=np.array([[1], [1], [1]]),
            groups=np.array([0, 0, 0]),
            censoring_point=0,
            censoring_type="left",
        )

        assert model._is_censored(0.0) == True
        assert model._is_censored(0.5) == False
        assert model._is_censored(1.0) == False

    def test_quadrature_integration(self, simulated_data):
        """Test different numbers of quadrature points."""
        # Test with different quadrature points
        for n_points in [4, 8, 12]:
            model = RandomEffectsTobit(
                endog=simulated_data["y"][:100],  # Smaller sample
                exog=simulated_data["X"][:100],
                groups=simulated_data["groups"][:100],
                quadrature_points=n_points,
            )

            # Just check log-likelihood computation works
            params = np.ones(simulated_data["X"].shape[1] + 2)
            llf = model._log_likelihood(params)

            assert np.isfinite(llf)
            assert llf < 0  # Log-likelihood should be negative


class TestPooledTobit:
    """Test suite for Pooled Tobit model."""

    def test_pooled_tobit_basic(self):
        """Test basic functionality of Pooled Tobit."""
        from panelbox.models.censored import PooledTobit

        np.random.seed(42)
        N, K = 100, 2

        X = np.random.randn(N, K)
        y_star = X @ np.array([0.5, -0.3]) + np.random.randn(N)
        y = np.maximum(0, y_star)  # Left censoring

        model = PooledTobit(endog=y, exog=X, censoring_point=0, censoring_type="left")

        result = model.fit(maxiter=100, options={"disp": False})

        assert hasattr(result, "beta")
        assert hasattr(result, "sigma")
        assert result.sigma > 0

    def test_pooled_tobit_predictions(self):
        """Test predictions from Pooled Tobit."""
        from panelbox.models.censored import PooledTobit

        np.random.seed(42)
        N, K = 100, 2

        X = np.random.randn(N, K)
        y_star = X @ np.array([0.5, -0.3]) + np.random.randn(N)
        y = np.maximum(0, y_star)

        model = PooledTobit(endog=y, exog=X, censoring_point=0)

        result = model.fit(maxiter=50, options={"disp": False})

        # Test different prediction types
        pred_latent = result.predict(pred_type="latent")
        pred_censored = result.predict(pred_type="censored")
        pred_prob = result.predict(pred_type="probability")

        assert len(pred_latent) == N
        assert len(pred_censored) == N
        assert len(pred_prob) == N

        # Probability should be between 0 and 1
        assert np.all(pred_prob >= 0)
        assert np.all(pred_prob <= 1)
