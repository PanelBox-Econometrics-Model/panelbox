"""
Tests for PPML (Poisson Pseudo-Maximum Likelihood).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less

from panelbox.models.count import PPML, PPMLResult


class TestPPML:
    """Tests for PPML class."""

    @pytest.fixture
    def gravity_data(self):
        """
        Create synthetic gravity model data.

        Trade flow = exp(β₀ + β₁*log_gdp_i + β₂*log_gdp_j - β₃*log_distance + ε)
        """
        np.random.seed(42)
        n_countries = 20
        n_years = 5
        n = n_countries * (n_countries - 1) * n_years  # exclude diagonal

        # Create country pairs
        pairs = []
        for year in range(n_years):
            for i in range(n_countries):
                for j in range(n_countries):
                    if i != j:  # exclude self-trade
                        pairs.append(
                            {"year": year, "origin": i, "dest": j, "pair_id": i * n_countries + j}
                        )

        df = pd.DataFrame(pairs)

        # Generate log GDP (with time trend)
        gdp_base = np.random.uniform(20, 30, n_countries)
        gdp_growth = np.random.uniform(0.01, 0.05, n_countries)

        df["log_gdp_i"] = df.apply(
            lambda row: gdp_base[int(row["origin"])] + gdp_growth[int(row["origin"])] * row["year"],
            axis=1,
        )
        df["log_gdp_j"] = df.apply(
            lambda row: gdp_base[int(row["dest"])] + gdp_growth[int(row["dest"])] * row["year"],
            axis=1,
        )

        # Generate log distance (time-invariant)
        distance_matrix = np.random.uniform(4, 9, (n_countries, n_countries))
        np.fill_diagonal(distance_matrix, 0)
        df["log_distance"] = df.apply(
            lambda row: distance_matrix[int(row["origin"]), int(row["dest"])], axis=1
        )

        # True parameters
        beta_gdp_i = 0.8
        beta_gdp_j = 0.7
        beta_distance = -1.2
        beta_0 = -15

        # Generate trade flows
        linear_pred = (
            beta_0
            + beta_gdp_i * df["log_gdp_i"]
            + beta_gdp_j * df["log_gdp_j"]
            + beta_distance * df["log_distance"]
        )

        # Add heteroskedastic error
        error = np.random.randn(len(df)) * (0.5 + 0.1 * df["log_distance"])
        lambda_it = np.exp(linear_pred + error)

        # Generate Poisson counts
        df["trade_flow"] = np.random.poisson(lambda_it)

        # Introduce some zeros (realistic for trade data)
        zero_prob = 0.1
        zeros_mask = np.random.rand(len(df)) < zero_prob
        df.loc[zeros_mask, "trade_flow"] = 0

        return {
            "df": df,
            "true_params": {
                "intercept": beta_0,
                "log_gdp_i": beta_gdp_i,
                "log_gdp_j": beta_gdp_j,
                "log_distance": beta_distance,
            },
            "n_zeros": zeros_mask.sum(),
        }

    def test_ppml_pooled_basic(self, gravity_data):
        """Test basic PPML pooled estimation."""
        df = gravity_data["df"]

        # Prepare data
        y = df["trade_flow"].values
        X = df[["log_gdp_i", "log_gdp_j", "log_distance"]].values
        X = np.column_stack([np.ones(len(X)), X])

        # Fit PPML
        model = PPML(
            endog=y,
            exog=X,
            entity_id=df["pair_id"].values,
            fixed_effects=False,
            exog_names=["const", "log_gdp_i", "log_gdp_j", "log_distance"],
        )
        result = model.fit()

        # Check result structure
        assert isinstance(result, PPMLResult)
        assert len(result.params) == 4
        assert result.cov.shape == (4, 4)

        # Check parameters are in reasonable range
        # (not exact due to heteroskedasticity and zeros)
        true_params = gravity_data["true_params"]
        assert result.params[1] > 0  # GDP elasticities positive
        assert result.params[2] > 0
        assert result.params[3] < 0  # Distance elasticity negative

    def test_ppml_fixed_effects(self, gravity_data):
        """Test PPML with fixed effects."""
        df = gravity_data["df"]

        # Prepare data
        y = df["trade_flow"].values
        X = df[["log_distance"]].values  # FE absorbs GDPs

        # Fit PPML FE
        model = PPML(
            endog=y,
            exog=X,
            entity_id=df["pair_id"].values,
            time_id=df["year"].values,
            fixed_effects=True,
            exog_names=["log_distance"],
        )
        result = model.fit()

        # Check structure
        assert isinstance(result, PPMLResult)
        assert result.fixed_effects is True

        # Distance coefficient should be negative
        assert result.params[0] < 0

    def test_ppml_handles_zeros(self):
        """Test that PPML handles zeros correctly."""
        np.random.seed(123)
        n = 200

        # Data with many zeros
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        lambda_it = np.exp(1 + 0.5 * X[:, 1])
        y = np.random.poisson(lambda_it)

        # Set 30% to zero
        zero_idx = np.random.choice(n, size=int(0.3 * n), replace=False)
        y[zero_idx] = 0

        # Create entity IDs for clustering
        entity_id = np.repeat(np.arange(40), 5)

        # Fit PPML
        model = PPML(
            endog=y, exog=X, entity_id=entity_id, fixed_effects=False, exog_names=["const", "x1"]
        )
        result = model.fit()

        # Should complete without error and recover approximate parameter
        assert len(result.params) == 2
        # Coefficient should be positive (true value is 0.5)
        assert result.params[1] > 0

    def test_ppml_elasticity_method(self, gravity_data):
        """Test elasticity computation."""
        df = gravity_data["df"]

        y = df["trade_flow"].values
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

        # Test elasticity for log-transformed variable
        elast_gdp = result.elasticity("log_gdp_i")

        assert "coefficient" in elast_gdp
        assert "elasticity" in elast_gdp
        assert "is_log_transformed" in elast_gdp
        assert elast_gdp["is_log_transformed"] is True

        # For log-transformed variable, elasticity = coefficient
        assert_allclose(elast_gdp["elasticity"], elast_gdp["coefficient"], rtol=1e-10)

    def test_ppml_elasticities_table(self, gravity_data):
        """Test elasticities table generation."""
        df = gravity_data["df"]

        y = df["trade_flow"].values
        X = df[["log_gdp_i", "log_gdp_j", "log_distance"]].values
        X = np.column_stack([np.ones(len(X)), X])

        model = PPML(
            endog=y,
            exog=X,
            fixed_effects=False,
            exog_names=["const", "log_gdp_i", "log_gdp_j", "log_distance"],
        )
        result = model.fit()

        # Get elasticities table
        elast_table = result.elasticities()

        # Check structure
        assert isinstance(elast_table, pd.DataFrame)
        assert "variable" in elast_table.columns
        assert "elasticity" in elast_table.columns
        assert len(elast_table) == 4

    def test_ppml_compare_with_ols(self, gravity_data):
        """Test comparison with OLS."""
        df = gravity_data["df"]

        # Remove zeros for OLS
        df_no_zeros = df[df["trade_flow"] > 0].copy()

        y_ppml = df["trade_flow"].values
        y_ols = np.log(df_no_zeros["trade_flow"].values)

        X_ppml = df[["log_gdp_i", "log_gdp_j", "log_distance"]].values
        X_ppml = np.column_stack([np.ones(len(X_ppml)), X_ppml])

        X_ols = df_no_zeros[["log_gdp_i", "log_gdp_j", "log_distance"]].values
        X_ols = np.column_stack([np.ones(len(X_ols)), X_ols])

        # Fit PPML
        model_ppml = PPML(
            endog=y_ppml,
            exog=X_ppml,
            entity_id=df["pair_id"].values,
            fixed_effects=False,
            exog_names=["const", "log_gdp_i", "log_gdp_j", "log_distance"],
        )
        result_ppml = model_ppml.fit()

        # Fit OLS (simple version for testing)
        from statsmodels.regression.linear_model import OLS

        model_ols = OLS(y_ols, X_ols)
        result_ols = model_ols.fit()

        # Compare
        comparison = result_ppml.compare_with_ols(result_ols)

        # Check structure
        assert isinstance(comparison, pd.DataFrame)
        assert "PPML_coef" in comparison.columns
        assert "OLS_coef" in comparison.columns
        assert "difference" in comparison.columns
        assert len(comparison) == 4

    def test_ppml_error_negative_values(self):
        """Test that PPML raises error for negative dependent variable."""
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.randn(n)  # Can be negative

        with pytest.raises(ValueError, match="non-negative"):
            PPML(endog=y, exog=X, fixed_effects=False)

    def test_ppml_error_no_entity_id_with_fe(self):
        """Test error when fixed_effects=True but no entity_id."""
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(1 + 0.5 * X[:, 1]))

        with pytest.raises(ValueError, match="entity_id required"):
            PPML(endog=y, exog=X, fixed_effects=True)

    def test_ppml_cluster_robust_ses_enforced(self, gravity_data):
        """Test that cluster-robust SEs are enforced."""
        df = gravity_data["df"]

        y = df["trade_flow"].values
        X = df[["log_gdp_i"]].values
        X = np.column_stack([np.ones(len(X)), X])

        model = PPML(endog=y, exog=X, entity_id=df["pair_id"].values, fixed_effects=False)

        # Try to use different SE type - should get warning
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.fit(se_type="homoskedastic")

            # Check that warning was issued
            assert len(w) > 0
            assert "cluster-robust" in str(w[0].message).lower()

    def test_ppml_summary_with_zero_info(self, gravity_data):
        """Test that summary includes information about zeros."""
        df = gravity_data["df"]

        y = df["trade_flow"].values
        X = df[["log_gdp_i"]].values
        X = np.column_stack([np.ones(len(X)), X])

        model = PPML(endog=y, exog=X, entity_id=df["pair_id"].values, fixed_effects=False)
        result = model.fit()

        summary = result.summary()

        # Check that summary mentions zeros
        assert isinstance(summary, str)
        assert "PPML" in summary
        if gravity_data["n_zeros"] > 0:
            assert "zeros" in summary.lower()

    def test_ppml_recovers_parameters(self):
        """
        Test that PPML recovers true parameters under correct specification.

        Uses homoskedastic DGP to check parameter recovery.
        """
        np.random.seed(789)
        n = 500

        # True parameters
        beta_0 = 1.0
        beta_1 = 0.8

        # Generate data
        X = np.random.randn(n)
        linear_pred = beta_0 + beta_1 * X
        lambda_it = np.exp(linear_pred)

        # Generate Poisson counts (homoskedastic)
        y = np.random.poisson(lambda_it)

        # Create entity IDs for clustering
        entity_id = np.repeat(np.arange(100), 5)

        # Fit PPML
        X_mat = np.column_stack([np.ones(n), X])
        model = PPML(endog=y, exog=X_mat, entity_id=entity_id, fixed_effects=False)
        result = model.fit()

        # Check parameter recovery
        # With large sample and correct specification, should be close
        assert_allclose(result.params[0], beta_0, rtol=0.15)
        assert_allclose(result.params[1], beta_1, rtol=0.15)


class TestPPMLResult:
    """Tests for PPMLResult class."""

    @pytest.fixture
    def simple_result(self):
        """Create a simple PPML result for testing."""
        np.random.seed(42)
        n = 200

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(1 + 0.5 * X[:, 1]))
        entity_id = np.repeat(np.arange(40), 5)

        model = PPML(
            endog=y, exog=X, entity_id=entity_id, fixed_effects=False, exog_names=["const", "log_x"]
        )
        return model.fit()

    def test_ppml_result_attributes(self, simple_result):
        """Test that PPMLResult has required attributes."""
        assert hasattr(simple_result, "params")
        assert hasattr(simple_result, "cov")
        assert hasattr(simple_result, "fixed_effects")
        assert hasattr(simple_result, "elasticity")
        assert hasattr(simple_result, "elasticities")

    def test_elasticity_invalid_variable(self, simple_result):
        """Test error for invalid variable name."""
        with pytest.raises(ValueError, match="not found"):
            simple_result.elasticity("nonexistent_var")

    def test_compare_with_ols_no_exog_names(self):
        """Test that compare_with_ols requires exog_names."""
        np.random.seed(42)
        n = 100

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = np.random.poisson(np.exp(1 + 0.5 * X[:, 1]))
        entity_id = np.repeat(np.arange(20), 5)

        # Fit without exog_names
        model = PPML(endog=y, exog=X, entity_id=entity_id, fixed_effects=False)
        result = model.fit()

        # Create mock OLS result
        from statsmodels.regression.linear_model import OLS

        ols_result = OLS(np.log(y + 1), X).fit()

        # Should raise error
        with pytest.raises(AttributeError, match="exog_names"):
            result.compare_with_ols(ols_result)
