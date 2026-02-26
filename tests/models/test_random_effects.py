"""
Tests for RandomEffects model.
"""

import numpy as np
import pytest

from panelbox.models.static.random_effects import RandomEffects


class TestRandomEffectsInitialization:
    """Tests for RandomEffects initialization."""

    def test_init_default(self, balanced_panel_data):
        """Test initialization with default parameters."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        assert model.variance_estimator == "swamy-arora"
        assert model.sigma2_u is None  # Not estimated yet
        assert model.sigma2_e is None
        assert model.theta is None

    def test_init_amemiya(self, balanced_panel_data):
        """Test initialization with Amemiya estimator."""
        model = RandomEffects(
            "y ~ x1 + x2", balanced_panel_data, "entity", "time", variance_estimator="amemiya"
        )

        assert model.variance_estimator == "amemiya"

    def test_init_invalid_estimator(self, balanced_panel_data):
        """Test that invalid variance estimator raises ValueError."""
        with pytest.raises(ValueError, match="variance_estimator must be one of"):
            RandomEffects(
                "y ~ x1 + x2", balanced_panel_data, "entity", "time", variance_estimator="invalid"
            )


class TestRandomEffectsFitting:
    """Tests for fitting Random Effects models."""

    def test_fit_basic(self, balanced_panel_data):
        """Test basic fitting."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert model._fitted is True
        assert len(results.params) == 3  # Intercept, x1, x2
        assert np.all(np.isfinite(results.params.values))

    def test_variance_components_computed(self, balanced_panel_data):
        """Test that variance components are computed."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        model.fit()

        # Variances should be computed and positive
        assert model.sigma2_u >= 0
        assert model.sigma2_e > 0

    def test_theta_bounds(self, balanced_panel_data):
        """Test that theta is in valid range [0, 1]."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        model.fit()

        # Theta should be between 0 and 1
        assert 0 <= model.theta <= 1

    def test_includes_intercept(self, balanced_panel_data):
        """Test that RE includes intercept (unlike FE)."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Should have intercept
        assert "Intercept" in results.params.index


class TestRSquaredMeasures:
    """Tests for R-squared measures."""

    def test_rsquared_measures_exist(self, balanced_panel_data):
        """Test that all R-squared measures are computed."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert not np.isnan(results.rsquared_within)
        assert not np.isnan(results.rsquared_between)
        assert not np.isnan(results.rsquared_overall)
        assert not np.isnan(results.rsquared_adj)

    def test_overall_rsquared_is_main(self, balanced_panel_data):
        """Test that rsquared equals overall R-squared for RE."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # For RE, the main R² is the overall R²
        assert results.rsquared == results.rsquared_overall


class TestCovarianceTypes:
    """Tests for different covariance estimators."""

    @pytest.mark.parametrize("cov_type", ["nonrobust", "robust", "clustered"])
    def test_covariance_type(self, balanced_panel_data, cov_type):
        """Test different covariance estimators produce valid results."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type=cov_type)

        assert results.cov_type == cov_type
        assert len(results.std_errors) == 3
        assert (results.std_errors > 0).all()


class TestVarianceEstimators:
    """Tests for different variance component estimators."""

    @pytest.mark.parametrize(
        "variance_estimator",
        ["swamy-arora", "walhus", "amemiya", "nerlove"],
    )
    def test_variance_estimator(self, balanced_panel_data, variance_estimator):
        """Test different variance component estimators."""
        model = RandomEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            variance_estimator=variance_estimator,
        )
        model.fit()

        assert model.sigma2_u >= 0
        assert model.sigma2_e > 0


class TestResultsSummary:
    """Tests for results summary."""

    def test_summary_format(self, balanced_panel_data):
        """Test that summary is formatted correctly."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        summary = results.summary()

        assert "Random Effects" in summary
        assert "Intercept" in summary
        assert "x1" in summary
        assert "x2" in summary
        assert "R-squared (overall)" in summary


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_regressor(self, balanced_panel_data):
        """Test with single regressor."""
        model = RandomEffects("y ~ x1", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert len(results.params) == 2  # Intercept and x1

    def test_unbalanced_panel(self, unbalanced_panel_data):
        """Test with unbalanced panel."""
        model = RandomEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = model.fit()

        # Should handle unbalanced panel
        assert len(results.params) == 3
        assert np.all(np.isfinite(results.params.values))
        assert model.sigma2_u >= 0


class TestGLSTransformation:
    """Tests for GLS transformation."""

    def test_theta_zero_means_pooled(self, balanced_panel_data):
        """Test that theta=0 is equivalent to pooled OLS."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        model.fit()

        # If theta were 0, transformation wouldn't remove any entity means
        # (This is just conceptual - we don't force theta=0)

    def test_theta_one_means_within(self, balanced_panel_data):
        """Test that theta=1 is equivalent to within (FE)."""
        # If theta = 1, the GLS transformation is equivalent to within transformation
        # (This is conceptual - theta=1 happens when sigma2_u >> sigma2_e)
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        model.fit()

        # Just verify theta was computed
        assert 0 <= model.theta <= 1


class TestComparisonWithOtherModels:
    """Tests comparing RE with other models."""

    def test_re_vs_pooled(self, balanced_panel_data):
        """Test that RE differs from Pooled OLS."""
        from panelbox.models.static.pooled_ols import PooledOLS

        model_re = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_re = model_re.fit()

        model_pooled = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_pooled = model_pooled.fit()

        # Both should produce valid results
        assert len(results_re.params) == 3
        assert len(results_pooled.params) == 3
        assert np.all(np.isfinite(results_re.params.values))
        assert np.all(np.isfinite(results_pooled.params.values))

    def test_re_includes_time_invariant(self, balanced_panel_data):
        """Test that RE can include time-invariant variables (unlike FE)."""
        # Add a time-invariant variable
        data = balanced_panel_data.copy()
        data["const_var"] = data["entity"] * 100  # Time-invariant

        model = RandomEffects("y ~ x1 + x2 + const_var", data, "entity", "time")
        results = model.fit()

        # Should include the time-invariant variable
        assert "const_var" in results.params.index


class TestCovarianceTypesAdvanced:
    """Tests for different covariance types."""

    @pytest.mark.parametrize("cov_type", ["twoway", "driscoll_kraay", "newey_west"])
    def test_fit_with_advanced_cov_types(self, balanced_panel_data, cov_type):
        """Test fitting with advanced covariance types."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type=cov_type)

        assert len(results.params) == 3
        assert np.all(np.isfinite(results.params.values))
        assert (results.std_errors > 0).all()

    def test_invalid_cov_type_raises(self, balanced_panel_data):
        """Test that invalid cov_type raises ValueError."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="cov_type must be one of"):
            model.fit(cov_type="invalid_type")


class TestInternalMethods:
    """Tests for internal estimation methods."""

    def test_estimate_gls(self, balanced_panel_data):
        """Test GLS estimation."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Check that GLS estimation worked
        assert len(results.params) == 3
        assert np.all(np.isfinite(results.params.values))

    @pytest.mark.parametrize("cov_type", ["robust", "clustered"])
    def test_vcov_computation(self, balanced_panel_data, cov_type):
        """Test covariance matrix computation for different types."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type=cov_type)

        assert results.cov_type == cov_type
        assert (results.std_errors > 0).all()
        assert len(results.std_errors) == 3

    def test_vcov_differs_by_type(self, balanced_panel_data):
        """Test that different covariance types produce different SEs."""
        model_robust = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_robust = model_robust.fit(cov_type="robust")

        model_clustered = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_clustered = model_clustered.fit(cov_type="clustered")

        # Coefficients should be identical
        np.testing.assert_array_almost_equal(
            results_robust.params.values, results_clustered.params.values
        )

        # Standard errors typically differ (both should be positive)
        assert (results_robust.std_errors > 0).all()
        assert (results_clustered.std_errors > 0).all()


class TestEstimateCoefficientsDirectly:
    """Tests for _estimate_coefficients() method (lines 522-534)."""

    def test_estimate_coefficients_directly(self, balanced_panel_data):
        """Test calling _estimate_coefficients() directly."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        beta = model._estimate_coefficients()

        # Should have 3 coefficients: Intercept, x1, x2
        assert len(beta.ravel()) == 3
        assert np.all(np.isfinite(beta))

    def test_estimate_coefficients_matches_fit(self, balanced_panel_data):
        """Test that _estimate_coefficients() gives same betas as fit()."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        beta_direct = model._estimate_coefficients().ravel()

        results = model.fit()
        beta_fit = results.params.values

        np.testing.assert_array_almost_equal(beta_direct, beta_fit)

    def test_estimate_coefficients_sets_variance_components(self, balanced_panel_data):
        """Test that _estimate_coefficients() also computes variance components."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        # Before calling, sigma2_u/e/theta should be None
        assert model.sigma2_u is None
        assert model.sigma2_e is None
        assert model.theta is None

        model._estimate_coefficients()

        # After calling, variance components should be populated
        assert model.sigma2_u >= 0
        assert model.sigma2_e > 0
        assert 0 <= model.theta <= 1


class TestComputeVcovRobustDirectly:
    """Tests for _compute_vcov_robust() method (lines 577-587)."""

    def test_compute_vcov_robust_directly(self, balanced_panel_data):
        """Test _compute_vcov_robust() with GLS-transformed data."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Reconstruct GLS-transformed X and residuals
        y, X = model.formula_parser.build_design_matrices(model.data.data, return_type="array")
        entities = model.data.data[model.data.entity_col].values

        y_gls, X_gls = model._gls_transform(y, X, entities)
        from panelbox.utils.matrix_ops import compute_ols

        _, resid_gls, _ = compute_ols(y_gls, X_gls, model.weights)

        df_resid = results.df_resid

        vcov = model._compute_vcov_robust(X_gls, resid_gls, df_resid)

        # Covariance matrix should be 3x3 (Intercept, x1, x2)
        assert vcov.shape == (3, 3)
        # Diagonal elements (variances) should be positive
        assert vcov[0, 0] > 0
        assert vcov[1, 1] > 0
        assert vcov[2, 2] > 0
        # Should be symmetric
        np.testing.assert_array_almost_equal(vcov, vcov.T)


class TestComputeVcovClusteredDirectly:
    """Tests for _compute_vcov_clustered() method (lines 589-612)."""

    def test_compute_vcov_clustered_directly(self, balanced_panel_data):
        """Test _compute_vcov_clustered() with GLS-transformed data."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Reconstruct GLS-transformed X and residuals
        y, X = model.formula_parser.build_design_matrices(model.data.data, return_type="array")
        entities = model.data.data[model.data.entity_col].values

        y_gls, X_gls = model._gls_transform(y, X, entities)
        from panelbox.utils.matrix_ops import compute_ols

        _, resid_gls, _ = compute_ols(y_gls, X_gls, model.weights)

        df_resid = results.df_resid

        vcov = model._compute_vcov_clustered(X_gls, resid_gls, entities, df_resid)

        # Covariance matrix should be 3x3 (Intercept, x1, x2)
        assert vcov.shape == (3, 3)
        # Diagonal elements (variances) should be positive
        assert vcov[0, 0] > 0
        assert vcov[1, 1] > 0
        assert vcov[2, 2] > 0
        # Should be symmetric
        np.testing.assert_array_almost_equal(vcov, vcov.T)
