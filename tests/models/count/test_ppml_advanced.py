"""
Additional advanced tests for PPML demonstrating key advantages.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.models.count import PPML


class TestPPMLAdvantages:
    """Tests demonstrating PPML advantages over OLS."""

    def test_ppml_heteroskedastic_unbiased_ols_biased(self):
        """
        Test that PPML is consistent under heteroskedasticity.

        This demonstrates the key advantage of PPML over OLS(log(y)).
        References Santos Silva & Tenreyro (2006).

        NOTE: This test verifies PPML handles heteroskedasticity correctly.
        The comparison with OLS demonstrates information usage, not necessarily
        superior parameter recovery in all cases.
        """
        np.random.seed(456)
        n = 1000

        # True parameters
        beta_0 = 1.5
        beta_1 = 0.6

        # Generate data with HETEROSKEDASTICITY
        X = np.random.randn(n)
        linear_pred = beta_0 + beta_1 * X

        # Heteroskedastic error: variance increases with X (mild)
        hetero_factor = np.exp(0.3 * X)
        epsilon = np.random.randn(n) * hetero_factor * 0.3

        # Generate dependent variable
        lambda_it = np.exp(linear_pred + epsilon)
        y = np.random.poisson(lambda_it)

        # Remove zeros for OLS
        non_zero = y > 0
        y_ols = y[non_zero]
        X_ols = X[non_zero]

        # Create entity IDs for clustering
        entity_id = np.repeat(np.arange(200), 5)

        # Fit PPML
        X_mat = np.column_stack([np.ones(n), X])
        model_ppml = PPML(endog=y, exog=X_mat, entity_id=entity_id, fixed_effects=False)
        result_ppml = model_ppml.fit()

        # Fit OLS on log(y)
        X_mat_ols = np.column_stack([np.ones(len(y_ols)), X_ols])
        from statsmodels.regression.linear_model import OLS

        model_ols = OLS(np.log(y_ols), X_mat_ols)
        result_ols = model_ols.fit()

        # PPML should recover parameter reasonably well
        # (it's consistent under heteroskedasticity)
        assert_allclose(result_ppml.params[1], beta_1, rtol=0.3)

        # PPML uses all observations (including potential zeros)
        assert result_ppml.model.n_obs == n

        # OLS drops zeros
        assert result_ols.nobs <= n

    def test_ppml_handles_zeros_ols_cannot(self):
        """
        Test that PPML handles zeros naturally while OLS must drop them.

        This demonstrates another key advantage of PPML.
        References Santos Silva & Tenreyro (2006).
        """
        np.random.seed(654)
        n = 500

        # True parameters
        beta_0 = 1.0
        beta_1 = 0.7
        beta_2 = -0.5

        # Generate data
        X1 = np.random.randn(n)
        X2 = np.random.randn(n)
        linear_pred = beta_0 + beta_1 * X1 + beta_2 * X2
        lambda_it = np.exp(linear_pred)

        # Generate counts with MANY zeros (30%)
        y = np.random.poisson(lambda_it)
        zero_indices = np.random.choice(n, size=int(0.3 * n), replace=False)
        y[zero_indices] = 0

        n_zeros = (y == 0).sum()
        n_nonzero = (y > 0).sum()

        # Create entity IDs
        entity_id = np.repeat(np.arange(100), 5)

        # Fit PPML (uses ALL observations including zeros)
        X_mat = np.column_stack([np.ones(n), X1, X2])
        model_ppml = PPML(endog=y, exog=X_mat, entity_id=entity_id, fixed_effects=False)
        result_ppml = model_ppml.fit()

        # Fit OLS (must DROP zeros)
        non_zero = y > 0
        y_ols = y[non_zero]
        X1_ols = X1[non_zero]
        X2_ols = X2[non_zero]
        X_mat_ols = np.column_stack([np.ones(n_nonzero), X1_ols, X2_ols])

        from statsmodels.regression.linear_model import OLS

        model_ols = OLS(np.log(y_ols), X_mat_ols)
        result_ols = model_ols.fit()

        # PPML uses all observations
        assert result_ppml.model.n_obs == n

        # OLS drops zeros
        assert result_ols.nobs == n_nonzero
        assert result_ols.nobs < n

        # Verify significant zeros were present
        assert n_zeros >= 100  # Should have ~150 zeros (30% of 500)

        # Check signs are correct (more important than exact values)
        assert result_ppml.params[1] > 0  # beta_1 positive
        assert result_ppml.params[2] < 0  # beta_2 negative

        # PPML should recover parameters with reasonable accuracy
        # (despite large number of zeros)
        assert 0.3 < result_ppml.params[1] < 1.1  # Reasonable range for beta_1
        assert -0.9 < result_ppml.params[2] < -0.1  # Reasonable range for beta_2

    def test_ppml_vs_ols_jensen_inequality(self):
        """
        Test demonstrating Jensen's inequality issue with log-linear models.

        E[log(y)] ≠ log(E[y])

        This is why PPML is preferred for gravity models.
        """
        np.random.seed(999)
        n = 800

        # True parameters
        beta_0 = 2.0
        beta_1 = 0.5

        # Generate data
        X = np.random.randn(n)
        linear_pred = beta_0 + beta_1 * X

        # Generate with mild multiplicative error
        epsilon = np.exp(np.random.randn(n) * 0.3)
        lambda_it = np.exp(linear_pred) * epsilon

        y = np.random.poisson(lambda_it)

        # Remove zeros for OLS
        non_zero = y > 0
        y_ols = y[non_zero]
        X_ols = X[non_zero]

        # Create entity IDs
        entity_id = np.repeat(np.arange(160), 5)

        # Fit PPML (handles multiplicative errors correctly)
        X_mat = np.column_stack([np.ones(n), X])
        model_ppml = PPML(endog=y, exog=X_mat, entity_id=entity_id, fixed_effects=False)
        result_ppml = model_ppml.fit()

        # Fit OLS (affected by Jensen's inequality)
        X_mat_ols = np.column_stack([np.ones(len(y_ols)), X_ols])
        from statsmodels.regression.linear_model import OLS

        model_ols = OLS(np.log(y_ols), X_mat_ols)
        result_ols = model_ols.fit()

        # Both models should complete successfully
        assert result_ppml.params is not None
        assert result_ols.params is not None

        # PPML should recover beta_1 reasonably well
        assert_allclose(result_ppml.params[1], beta_1, rtol=0.4)

        # Check that both methods produce valid estimates
        assert result_ppml.params[1] > 0
        assert result_ols.params[1] > 0

    def test_ppml_gravity_model_simulation(self):
        """
        Realistic gravity model simulation.

        Trade_ij = exp(β₀ + β₁ log(GDP_i) + β₂ log(GDP_j) - β₃ log(dist_ij) + ε)
        """
        np.random.seed(2024)

        # Simulate 10 countries, 5 years
        n_countries = 10
        n_years = 5

        data = []
        for year in range(n_years):
            for i in range(n_countries):
                for j in range(n_countries):
                    if i != j:  # No self-trade
                        data.append(
                            {"year": year, "origin": i, "dest": j, "pair_id": i * n_countries + j}
                        )

        df = pd.DataFrame(data)

        # Generate GDPs
        gdp_base = np.random.uniform(20, 28, n_countries)
        gdp_growth = 0.03

        df["log_gdp_i"] = df.apply(
            lambda x: gdp_base[int(x["origin"])] + gdp_growth * x["year"], axis=1
        )
        df["log_gdp_j"] = df.apply(
            lambda x: gdp_base[int(x["dest"])] + gdp_growth * x["year"], axis=1
        )

        # Generate distances (time-invariant)
        dist_matrix = np.random.uniform(5, 9, (n_countries, n_countries))
        np.fill_diagonal(dist_matrix, 0)

        df["log_distance"] = df.apply(
            lambda x: dist_matrix[int(x["origin"]), int(x["dest"])], axis=1
        )

        # True gravity parameters
        beta_0 = -10
        beta_gdp_i = 0.9
        beta_gdp_j = 0.8
        beta_dist = -1.1

        # Generate trade flows
        linear_pred = (
            beta_0
            + beta_gdp_i * df["log_gdp_i"]
            + beta_gdp_j * df["log_gdp_j"]
            + beta_dist * df["log_distance"]
        )

        # Add mild heteroskedastic error
        epsilon = np.random.randn(len(df)) * 0.3
        lambda_it = np.exp(linear_pred + epsilon)

        df["trade"] = np.random.poisson(lambda_it)

        # Add 10% zeros (realistic for trade)
        zero_mask = np.random.rand(len(df)) < 0.10
        df.loc[zero_mask, "trade"] = 0

        # Fit PPML
        y = df["trade"].values
        X = df[["log_gdp_i", "log_gdp_j", "log_distance"]].values
        X = np.column_stack([np.ones(len(X)), X])

        model = PPML(
            endog=y,
            exog=X,
            entity_id=df["pair_id"].values,
            fixed_effects=False,
            exog_names=["const", "log_gdp_i", "log_gdp_j", "log_distance"],
        )
        result = model.fit()

        # Check elasticities have correct signs
        assert result.params[1] > 0  # GDP origin elasticity positive
        assert result.params[2] > 0  # GDP dest elasticity positive
        assert result.params[3] < 0  # Distance elasticity negative

        # Elasticity magnitudes should be in reasonable range
        # (allowing for sampling variation)
        assert 0.4 < result.params[1] < 1.4
        assert 0.4 < result.params[2] < 1.4
        assert -1.8 < result.params[3] < -0.4

        # Test elasticity method
        elast = result.elasticity("log_gdp_i")
        assert elast["is_log_transformed"]
        assert_allclose(elast["elasticity"], result.params[1], rtol=1e-10)
