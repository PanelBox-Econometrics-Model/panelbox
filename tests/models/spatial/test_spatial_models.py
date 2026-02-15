"""
Tests for spatial panel models (SAR and SEM).
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.spatial import SpatialError, SpatialLag, SpatialWeights


def generate_spatial_panel_data(N=50, T=10, rho=0.5, lambda_param=0.0, seed=42):
    """
    Generate synthetic spatial panel data.

    Parameters
    ----------
    N : int
        Number of entities
    T : int
        Number of time periods
    rho : float
        Spatial lag parameter (for SAR)
    lambda_param : float
        Spatial error parameter (for SEM)
    seed : int
        Random seed

    Returns
    -------
    tuple
        (data, W) where data is DataFrame and W is weight matrix
    """
    np.random.seed(seed)

    # Create simple row-normalized weight matrix (rook contiguity on grid)
    W_raw = np.zeros((N, N))
    for i in range(N):
        # Simple nearest neighbor structure
        if i > 0:
            W_raw[i, i - 1] = 1
        if i < N - 1:
            W_raw[i, i + 1] = 1

    # Row-normalize
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_normalized = W_raw / row_sums

    # Generate exogenous variables
    X1 = np.random.randn(N * T)
    X2 = np.random.randn(N * T)

    # True parameters
    beta_true = np.array([1.0, -0.5])

    # Generate fixed effects
    alpha = np.random.randn(N)

    # Build panel structure
    entity_ids = []
    time_ids = []
    y_all = []

    for t in range(T):
        entity_ids.extend(range(N))
        time_ids.extend([t] * N)

        # Extract data for time t
        X1_t = X1[t * N : (t + 1) * N]
        X2_t = X2[t * N : (t + 1) * N]
        X_t = np.column_stack([X1_t, X2_t])

        # Linear prediction with fixed effects
        Xbeta = X_t @ beta_true + alpha

        # Add error
        if lambda_param != 0:
            # SEM: spatially correlated errors
            epsilon = np.random.randn(N)
            I_lambdaW = np.eye(N) - lambda_param * W_normalized
            u = np.linalg.solve(I_lambdaW, epsilon)
            y_t = Xbeta + u
        else:
            # Simple errors
            epsilon = np.random.randn(N)
            y_t = Xbeta + epsilon

        # Apply spatial lag for SAR
        if rho != 0:
            I_rhoW = np.eye(N) - rho * W_normalized
            y_t = np.linalg.solve(I_rhoW, y_t)

        y_all.extend(y_t)

    # Create DataFrame
    data = pd.DataFrame({"entity": entity_ids, "time": time_ids, "y": y_all, "x1": X1, "x2": X2})

    # Create SpatialWeights object
    W = SpatialWeights(W_normalized, normalized=True)

    return data, W


class TestSpatialLag:
    """Test Spatial Lag Model (SAR)."""

    def test_sar_initialization(self):
        """Test SAR model initialization."""
        data, W = generate_spatial_panel_data(N=10, T=5)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        assert model.model_type == "SAR"
        assert model.n_entities == 10
        assert model.n_periods == 5
        assert model.n_obs == 50

    def test_sar_fe_estimation(self):
        """Test SAR fixed effects estimation."""
        # Generate data with known parameters
        data, W = generate_spatial_panel_data(N=25, T=10, rho=0.3, lambda_param=0.0, seed=42)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        # Fit model
        result = model.fit(effects="fixed", method="qml", verbose=False)

        # Check that estimation completed
        assert model.fitted
        assert result is not None

        # Check parameter recovery (with some tolerance)
        rho_hat = result.params["rho"]
        assert abs(rho_hat - 0.3) < 0.15  # Within 0.15 of true value

        # Check that standard errors are positive
        assert all(result.bse > 0)

        # Check log-likelihood is finite
        assert np.isfinite(result.llf)

    def test_sar_pooled_estimation(self):
        """Test SAR pooled estimation."""
        data, W = generate_spatial_panel_data(N=20, T=5, rho=0.2)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        # Fit pooled model
        result = model.fit(effects="pooled", method="qml")

        # Check results
        assert "const" in result.params.index  # Should have constant
        assert "rho" in result.params.index
        assert len(result.params) == 4  # rho, const, x1, x2

    def test_sar_bounds(self):
        """Test that spatial parameter respects bounds."""
        data, W = generate_spatial_panel_data(N=15, T=8)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        bounds = model._spatial_coefficient_bounds()
        result = model.fit(effects="fixed")

        # Check rho is within bounds
        rho_hat = result.params["rho"]
        assert bounds[0] <= rho_hat <= bounds[1]

    def test_sar_prediction(self):
        """Test SAR prediction."""
        data, W = generate_spatial_panel_data(N=10, T=5)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed")

        # Generate predictions
        predictions = model.predict()

        assert len(predictions) == len(data)
        assert np.all(np.isfinite(predictions))

    def test_sar_spillover_effects(self):
        """Test computation of spillover effects."""
        data, W = generate_spatial_panel_data(N=10, T=5, rho=0.3)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed")

        # Check spillover effects are computed
        assert hasattr(result, "spillover_effects")
        assert "x1" in result.spillover_effects
        assert "x2" in result.spillover_effects

        # Check structure
        for var in ["x1", "x2"]:
            effects = result.spillover_effects[var]
            assert "direct" in effects
            assert "indirect" in effects
            assert "total" in effects

            # Total = direct + indirect
            np.testing.assert_allclose(
                effects["total"], effects["direct"] + effects["indirect"], rtol=1e-10
            )


class TestSpatialError:
    """Test Spatial Error Model (SEM)."""

    def test_sem_initialization(self):
        """Test SEM model initialization."""
        data, W = generate_spatial_panel_data(N=10, T=5)

        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        assert model.model_type == "SEM"
        assert model.n_entities == 10
        assert model.n_periods == 5

    def test_sem_gmm_fe_estimation(self):
        """Test SEM GMM fixed effects estimation."""
        # Generate data with spatial errors
        data, W = generate_spatial_panel_data(N=20, T=8, rho=0.0, lambda_param=0.4, seed=42)

        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        # Fit with GMM
        result = model.fit(effects="fixed", method="gmm", n_lags=2, verbose=False)

        # Check estimation completed
        assert model.fitted
        assert result is not None

        # Check lambda parameter
        lambda_hat = result.params["lambda"]
        assert abs(lambda_hat - 0.4) < 0.2  # Within 0.2 of true value

        # Check standard errors
        assert all(result.bse > 0)

    def test_sem_gmm_pooled_estimation(self):
        """Test SEM GMM pooled estimation."""
        data, W = generate_spatial_panel_data(N=15, T=6)

        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        # Fit pooled model
        result = model.fit(effects="pooled", method="gmm", n_lags=1)

        # Check results
        assert "lambda" in result.params.index
        assert "const" in result.params.index

    def test_sem_ml_estimation(self):
        """Test SEM maximum likelihood estimation."""
        data, W = generate_spatial_panel_data(N=10, T=5, lambda_param=0.3)

        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        # Fit with ML
        result = model.fit(effects="fixed", method="ml", verbose=False)

        # Check results
        assert "lambda" in result.params.index
        assert np.isfinite(result.llf)

    def test_sem_bounds(self):
        """Test that lambda respects bounds."""
        data, W = generate_spatial_panel_data(N=12, T=6)

        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        bounds = model._spatial_coefficient_bounds()
        result = model.fit(effects="fixed", method="gmm")

        # Check lambda is within bounds
        lambda_hat = result.params["lambda"]
        assert bounds[0] <= lambda_hat <= bounds[1]

    def test_sem_prediction(self):
        """Test SEM prediction."""
        data, W = generate_spatial_panel_data(N=10, T=4)

        model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed", method="gmm")

        # Generate predictions
        predictions = model.predict()

        assert len(predictions) == len(data)
        assert np.all(np.isfinite(predictions))


class TestSpatialModelComparison:
    """Test comparison between SAR and SEM models."""

    def test_model_comparison(self):
        """Test that SAR and SEM give different results on same data."""
        data, W = generate_spatial_panel_data(N=20, T=8, rho=0.3, lambda_param=0.0)

        # Fit SAR
        sar_model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )
        sar_result = sar_model.fit(effects="fixed")

        # Fit SEM
        sem_model = SpatialError(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )
        sem_result = sem_model.fit(effects="fixed", method="gmm")

        # Results should be different
        assert "rho" in sar_result.params.index
        assert "lambda" in sem_result.params.index

        # Coefficients might be similar but not identical
        beta_sar = sar_result.params[["x1", "x2"]].values
        beta_sem = sem_result.params[["x1", "x2"]].values

        # Should not be exactly equal
        assert not np.allclose(beta_sar, beta_sem, rtol=1e-5)

    def test_information_criteria(self):
        """Test that AIC/BIC are computed."""
        data, W = generate_spatial_panel_data(N=15, T=6)

        model = SpatialLag(
            endog=data["y"],
            exog=data[["x1", "x2"]],
            W=W,
            entity_id=data["entity"],
            time_id=data["time"],
        )

        result = model.fit(effects="fixed")

        # Check information criteria
        assert hasattr(result, "aic")
        assert hasattr(result, "bic")
        assert np.isfinite(result.aic)
        assert np.isfinite(result.bic)
        assert result.bic > result.aic  # BIC penalizes more
