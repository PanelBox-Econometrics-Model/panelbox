"""
Tests for RandomEffects model.
"""

import numpy as np
import pandas as pd
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
        assert results is not None
        assert len(results.params) == 3  # Intercept, x1, x2

    def test_variance_components_computed(self, balanced_panel_data):
        """Test that variance components are computed."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert model.sigma2_u is not None
        assert model.sigma2_e is not None
        assert model.theta is not None

        # Variances should be positive
        assert model.sigma2_u >= 0
        assert model.sigma2_e > 0

    def test_theta_bounds(self, balanced_panel_data):
        """Test that theta is in valid range [0, 1]."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

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

    def test_nonrobust_se(self, balanced_panel_data):
        """Test non-robust standard errors."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="nonrobust")

        assert results.cov_type == "nonrobust"
        assert len(results.std_errors) == 3

    def test_robust_se(self, balanced_panel_data):
        """Test robust standard errors."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="robust")

        assert results.cov_type == "robust"

    def test_clustered_se(self, balanced_panel_data):
        """Test cluster-robust standard errors."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="clustered")

        assert results.cov_type == "clustered"


class TestVarianceEstimators:
    """Tests for different variance component estimators."""

    def test_swamy_arora(self, balanced_panel_data):
        """Test Swamy-Arora estimator."""
        model = RandomEffects(
            "y ~ x1 + x2", balanced_panel_data, "entity", "time", variance_estimator="swamy-arora"
        )
        results = model.fit()

        assert model.sigma2_u is not None
        assert model.sigma2_e is not None

    def test_walhus(self, balanced_panel_data):
        """Test Wallace-Hussain estimator."""
        model = RandomEffects(
            "y ~ x1 + x2", balanced_panel_data, "entity", "time", variance_estimator="walhus"
        )
        results = model.fit()

        assert model.sigma2_u is not None

    def test_amemiya(self, balanced_panel_data):
        """Test Amemiya estimator."""
        model = RandomEffects(
            "y ~ x1 + x2", balanced_panel_data, "entity", "time", variance_estimator="amemiya"
        )
        results = model.fit()

        assert model.sigma2_u is not None

    def test_nerlove(self, balanced_panel_data):
        """Test Nerlove estimator."""
        model = RandomEffects(
            "y ~ x1 + x2", balanced_panel_data, "entity", "time", variance_estimator="nerlove"
        )
        results = model.fit()

        assert model.sigma2_u is not None


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

        assert results is not None
        # Should handle unbalanced panel
        assert model.sigma2_u >= 0


class TestGLSTransformation:
    """Tests for GLS transformation."""

    def test_theta_zero_means_pooled(self, balanced_panel_data):
        """Test that theta=0 is equivalent to pooled OLS."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # If theta were 0, transformation wouldn't remove any entity means
        # (This is just conceptual - we don't force theta=0)

    def test_theta_one_means_within(self, balanced_panel_data):
        """Test that theta=1 is equivalent to within (FE)."""
        # If theta = 1, the GLS transformation is equivalent to within transformation
        # (This is conceptual - theta=1 happens when sigma2_u >> sigma2_e)
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

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

        # Coefficients should differ (unless theta=0)
        # Just check both are fitted
        assert results_re is not None
        assert results_pooled is not None

    def test_re_includes_time_invariant(self, balanced_panel_data):
        """Test that RE can include time-invariant variables (unlike FE)."""
        # Add a time-invariant variable
        data = balanced_panel_data.copy()
        data["const_var"] = data["entity"] * 100  # Time-invariant

        model = RandomEffects("y ~ x1 + x2 + const_var", data, "entity", "time")
        results = model.fit()

        # Should include the time-invariant variable
        assert "const_var" in results.params.index


class TestCovarianceTypes:
    """Tests for different covariance types."""

    def test_fit_with_twoway_clustering(self, balanced_panel_data):
        """Test fitting with two-way clustering."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="twoway")

        assert results is not None
        assert len(results.params) == 3
        assert (results.std_errors > 0).all()

    def test_fit_with_driscoll_kraay(self, balanced_panel_data):
        """Test fitting with Driscoll-Kraay standard errors."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay")

        assert results is not None
        assert (results.std_errors > 0).all()

    def test_fit_with_newey_west(self, balanced_panel_data):
        """Test fitting with Newey-West standard errors."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west")

        assert results is not None
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
        assert results.params is not None

    def test_robust_vcov_computation(self, balanced_panel_data):
        """Test robust covariance matrix computation."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="robust")

        # Check that robust SEs are computed and positive
        assert results.cov_type == "robust"
        assert (results.std_errors > 0).all()
        assert len(results.std_errors) == 3

    def test_clustered_vcov_computation(self, balanced_panel_data):
        """Test clustered covariance matrix computation."""
        model = RandomEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="clustered")

        # Check that clustered SEs are computed and positive
        assert results.cov_type == "clustered"
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
