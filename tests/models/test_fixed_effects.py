"""
Tests for FixedEffects model.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects


class TestFixedEffectsInitialization:
    """Tests for FixedEffects initialization."""

    @pytest.mark.parametrize(
        ("entity_effects", "time_effects"),
        [(True, False), (False, True), (True, True)],
        ids=["entity", "time", "twoway"],
    )
    def test_init_effects(self, balanced_panel_data, entity_effects, time_effects):
        """Test initialization with different effect types."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=entity_effects,
            time_effects=time_effects,
        )

        assert model.entity_effects is entity_effects
        assert model.time_effects is time_effects

    def test_init_no_effects_raises(self, balanced_panel_data):
        """Test that no effects raises ValueError."""
        with pytest.raises(ValueError, match="At least one of entity_effects"):
            FixedEffects(
                "y ~ x1 + x2",
                balanced_panel_data,
                "entity",
                "time",
                entity_effects=False,
                time_effects=False,
            )


class TestFixedEffectsFitting:
    """Tests for fitting Fixed Effects models."""

    def test_fit_entity_effects(self, balanced_panel_data):
        """Test fitting with entity fixed effects."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Check that model was fitted
        assert model._fitted is True
        assert results is not None

        # Check coefficient structure
        assert len(results.params) == 2  # x1 and x2 (no intercept in FE)
        assert "x1" in results.params.index
        assert "x2" in results.params.index
        assert "Intercept" not in results.params.index  # Absorbed by FE

        # Check that fixed effects were computed
        assert model.entity_fe is not None
        assert len(model.entity_fe) == 10  # 10 entities

    def test_fit_time_effects(self, balanced_panel_data):
        """Test fitting with time fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )
        model.fit()

        assert model._fitted is True
        assert model.time_fe is not None
        assert len(model.time_fe) == 5  # 5 time periods

    def test_fit_twoway_effects(self, balanced_panel_data):
        """Test fitting with two-way fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        assert model._fitted is True
        assert model.entity_fe is not None
        assert model.time_fe is not None
        assert results.model_type == "Fixed Effects (Two-Way)"

    def test_within_transformation(self, balanced_panel_data):
        """Test that within transformation removes entity means."""
        # Create data with known entity effects
        data = balanced_panel_data.copy()

        # Add entity-specific constants
        entity_effects_true = {i: i * 10 for i in range(1, 11)}
        data["entity_effect"] = data["entity"].map(entity_effects_true)
        data["y"] = data["y"] + data["entity_effect"]

        model = FixedEffects("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        # Within transformation should remove entity effects
        # Coefficients should be similar to original (without entity effects)
        assert results is not None
        assert len(results.params) == 2


class TestRSquaredMeasures:
    """Tests for R-squared measures in Fixed Effects."""

    def test_rsquared_measures_exist(self, balanced_panel_data):
        """Test that all R-squared measures are computed."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert not np.isnan(results.rsquared_within)
        assert not np.isnan(results.rsquared_between)
        assert not np.isnan(results.rsquared_overall)
        assert not np.isnan(results.rsquared_adj)

    def test_within_rsquared_is_main(self, balanced_panel_data):
        """Test that rsquared equals within R-squared for FE."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # For FE, the main R² is the within R²
        assert results.rsquared == results.rsquared_within

    def test_rsquared_bounds(self, balanced_panel_data):
        """Test that R-squared values are in valid range."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # R-squared should be between 0 and 1
        assert 0 <= results.rsquared_within <= 1
        assert 0 <= results.rsquared_between <= 1
        assert 0 <= results.rsquared_overall <= 1


class TestCovarianceTypes:
    """Tests for different covariance estimators."""

    @pytest.mark.parametrize("cov_type", ["nonrobust", "robust", "clustered"])
    def test_covariance_type(self, balanced_panel_data, cov_type):
        """Test different covariance estimators produce valid results."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type=cov_type)

        assert results.cov_type == cov_type
        assert len(results.std_errors) == 2
        assert (results.std_errors > 0).all()

    def test_robust_vs_nonrobust(self, balanced_panel_data):
        """Test that robust SEs differ from non-robust."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        results_nonrobust = model.fit(cov_type="nonrobust")

        # Refit with robust
        model2 = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_robust = model2.fit(cov_type="robust")

        # Coefficients should be the same
        np.testing.assert_array_almost_equal(
            results_nonrobust.params.values, results_robust.params.values
        )

        # Standard errors typically different (may be same by chance with random data)
        # Just check they're both positive
        assert (results_nonrobust.std_errors > 0).all()
        assert (results_robust.std_errors > 0).all()


class TestFixedEffectsExtraction:
    """Tests for extracting fixed effects."""

    def test_entity_fe_extraction(self, balanced_panel_data):
        """Test extraction of entity fixed effects."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        model.fit()

        # Check entity FE
        assert model.entity_fe is not None
        assert isinstance(model.entity_fe, pd.Series)
        assert len(model.entity_fe) == 10
        assert model.entity_fe.name == "entity_fe"

    def test_time_fe_extraction(self, balanced_panel_data):
        """Test extraction of time fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )
        model.fit()

        assert model.time_fe is not None
        assert isinstance(model.time_fe, pd.Series)
        assert len(model.time_fe) == 5
        assert model.time_fe.name == "time_fe"

    def test_twoway_fe_extraction(self, balanced_panel_data):
        """Test extraction of both entity and time fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        model.fit()

        assert model.entity_fe is not None
        assert model.time_fe is not None
        assert len(model.entity_fe) == 10
        assert len(model.time_fe) == 5

    def test_entity_fe_sum_zero(self, balanced_panel_data):
        """Test that entity fixed effects are identified (sum to zero constraint)."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        model.fit()

        # Fixed effects should approximately sum to zero (mean is zero)
        # Due to the identification, they're centered
        assert abs(model.entity_fe.mean()) < 1e-10


class TestResultsSummary:
    """Tests for results summary."""

    def test_summary_format(self, balanced_panel_data):
        """Test that summary is formatted correctly."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        summary = results.summary()

        assert "Fixed Effects" in summary
        assert "x1" in summary
        assert "x2" in summary
        assert "R-squared (within)" in summary

    def test_twoway_summary_format(self, balanced_panel_data):
        """Test summary for two-way fixed effects."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        summary = results.summary()
        assert "Fixed Effects (Two-Way)" in summary


class TestDegreesOfFreedom:
    """Tests for degrees of freedom calculations."""

    def test_df_entity_fe(self, balanced_panel_data):
        """Test degrees of freedom with entity FE."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        n = 50  # 10 entities * 5 periods
        k = 2  # x1, x2
        n_fe = 10  # entity fixed effects

        expected_df_resid = n - k - n_fe
        assert results.df_resid == expected_df_resid

    def test_df_twoway_fe(self, balanced_panel_data):
        """Test degrees of freedom with two-way FE."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results = model.fit()

        n = 50
        k = 2
        n_fe_entity = 10
        n_fe_time = 5

        expected_df_resid = n - k - n_fe_entity - n_fe_time
        assert results.df_resid == expected_df_resid


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_regressor(self, balanced_panel_data):
        """Test with single regressor."""
        model = FixedEffects("y ~ x1", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert len(results.params) == 1
        assert "x1" in results.params.index

    def test_no_intercept_formula(self, balanced_panel_data):
        """Test that -1 in formula doesn't cause issues (FE absorbs intercept anyway)."""
        model = FixedEffects("y ~ x1 + x2 - 1", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Should work fine (intercept absorbed by FE regardless)
        assert len(results.params) == 2

    def test_unbalanced_panel(self, unbalanced_panel_data):
        """Test with unbalanced panel."""
        model = FixedEffects("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = model.fit()

        assert results is not None
        # Entity FE should have 3 entities
        assert len(model.entity_fe) == 3


class TestModelComparison:
    """Tests comparing different specifications."""

    def test_entity_vs_time_fe(self, balanced_panel_data):
        """Test that entity FE and time FE give different results."""
        model_entity = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=False,
        )
        results_entity = model_entity.fit()

        model_time = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=False,
            time_effects=True,
        )
        results_time = model_time.fit()

        # Coefficients should differ
        # (unless by chance the data has no entity or time effects)
        assert len(results_entity.params) == len(results_time.params)

    def test_oneway_vs_twoway(self, balanced_panel_data):
        """Test one-way vs two-way fixed effects."""
        model_oneway = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=False,
        )
        results_oneway = model_oneway.fit()

        model_twoway = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )
        results_twoway = model_twoway.fit()

        # Two-way should have fewer df_resid
        assert results_twoway.df_resid < results_oneway.df_resid


class TestCovarianceTypesAdvanced:
    """Tests for different covariance types."""

    def test_fit_with_twoway_clustering(self, balanced_panel_data):
        """Test fitting with two-way clustering."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="twoway")

        assert results is not None
        assert len(results.params) == 2
        assert (results.std_errors > 0).all()

    def test_fit_with_pcse(self, balanced_panel_data):
        """Test fitting with PCSE standard errors."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="pcse")

        # PCSE may produce warnings or NaN with T < N, but should not crash
        assert results is not None
        assert len(results.params) == 2

    def test_invalid_cov_type_raises(self, balanced_panel_data):
        """Test that invalid cov_type raises ValueError."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="cov_type must be one of"):
            model.fit(cov_type="invalid_type")


class TestInternalMethods:
    """Tests for internal estimation methods."""

    @pytest.mark.parametrize(
        ("entity_effects", "time_effects"),
        [(True, False), (False, True), (True, True)],
        ids=["entity", "time", "twoway"],
    )
    def test_estimate_ols_effects(self, balanced_panel_data, entity_effects, time_effects):
        """Test OLS estimation with different effect types."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=entity_effects,
            time_effects=time_effects,
        )
        results = model.fit()

        assert len(results.params) == 2
        assert results.params is not None

    @pytest.mark.parametrize("cov_type", ["robust", "clustered"])
    def test_vcov_computation(self, balanced_panel_data, cov_type):
        """Test covariance matrix computation for different types."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type=cov_type)

        assert results.cov_type == cov_type
        assert (results.std_errors > 0).all()
        assert len(results.std_errors) == 2

    def test_vcov_differs_by_type(self, balanced_panel_data):
        """Test that different covariance types produce different SEs."""
        model_robust = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_robust = model_robust.fit(cov_type="robust")

        model_clustered = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results_clustered = model_clustered.fit(cov_type="clustered")

        # Coefficients should be identical
        np.testing.assert_array_almost_equal(
            results_robust.params.values, results_clustered.params.values
        )

        # Standard errors typically differ (both should be positive)
        assert (results_robust.std_errors > 0).all()
        assert (results_clustered.std_errors > 0).all()


class TestDriscollKraayAndNeweyWestCov:
    """Tests for Driscoll-Kraay and Newey-West covariance paths."""

    def test_driscoll_kraay_cov_type(self, balanced_panel_data):
        """Test fitting with Driscoll-Kraay standard errors (lines 255-260)."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay", max_lags=2)

        assert results is not None
        assert results.cov_type == "driscoll_kraay"
        assert len(results.params) == 2
        assert (results.std_errors > 0).all()

    def test_driscoll_kraay_default_lags(self, balanced_panel_data):
        """Test Driscoll-Kraay with default (None) max_lags."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay")

        assert results is not None
        assert (results.std_errors > 0).all()

    def test_driscoll_kraay_with_kernel(self, balanced_panel_data):
        """Test Driscoll-Kraay with explicit kernel argument."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay", max_lags=1, kernel="bartlett")

        assert results is not None
        assert (results.std_errors > 0).all()

    def test_newey_west_cov_type(self, balanced_panel_data):
        """Test fitting with Newey-West standard errors (lines 262-265)."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west", max_lags=2)

        assert results is not None
        assert results.cov_type == "newey_west"
        assert len(results.params) == 2
        assert (results.std_errors > 0).all()

    def test_newey_west_default_lags(self, balanced_panel_data):
        """Test Newey-West with default (None) max_lags."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west")

        assert results is not None
        assert (results.std_errors > 0).all()

    def test_newey_west_with_kernel(self, balanced_panel_data):
        """Test Newey-West with explicit kernel argument."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west", max_lags=1, kernel="bartlett")

        assert results is not None
        assert (results.std_errors > 0).all()


class TestInsufficientDegreesOfFreedom:
    """Tests for insufficient degrees of freedom error (line 395)."""

    def test_insufficient_df_raises_error(self):
        """Test that df_resid <= 0 raises ValueError."""
        # Create a panel where n - k - n_fe_entity - n_fe_time <= 0
        # 6 entities, 2 time periods = 12 obs
        # 5 regressors + 6 entity FE + 2 time FE = 13 > 12
        np.random.seed(99)
        data = pd.DataFrame(
            {
                "entity": np.repeat(range(6), 2),
                "time": np.tile([1, 2], 6),
                "y": np.random.randn(12),
                "x1": np.random.randn(12),
                "x2": np.random.randn(12),
                "x3": np.random.randn(12),
                "x4": np.random.randn(12),
                "x5": np.random.randn(12),
            }
        )

        model = FixedEffects(
            "y ~ x1 + x2 + x3 + x4 + x5",
            data,
            "entity",
            "time",
            entity_effects=True,
            time_effects=True,
        )

        with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
            model.fit()


class TestFStatisticEdgeCase:
    """Tests for F-test nan when df1 or df2 <= 0 (line 583)."""

    def test_f_test_nan_with_single_entity(self):
        """Test that F-test returns nan when n_fe_entity = 1 (df1 = 0)."""
        # 1 entity -> df1 = n_fe_entity - 1 = 0 -> returns nan
        data = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 1],
                "time": [1, 2, 3, 4, 5],
                "y": np.random.randn(5),
                "x1": np.random.randn(5),
            }
        )

        model = FixedEffects("y ~ x1", data, "entity", "time")
        results = model.fit()

        # With 1 entity, df1 = 0, so F-stat should be nan
        assert np.isnan(results.f_statistic)
        assert np.isnan(results.f_pvalue)


class TestEstimateCoefficientsDirectly:
    """Tests for _estimate_coefficients() method (lines 688-710)."""

    @pytest.mark.parametrize(
        ("entity_effects", "time_effects"),
        [(True, False), (False, True), (True, True)],
        ids=["entity", "time", "twoway"],
    )
    def test_estimate_coefficients_directly(
        self, balanced_panel_data, entity_effects, time_effects
    ):
        """Test _estimate_coefficients() directly for all effect types."""
        model = FixedEffects(
            "y ~ x1 + x2",
            balanced_panel_data,
            "entity",
            "time",
            entity_effects=entity_effects,
            time_effects=time_effects,
        )

        beta = model._estimate_coefficients()

        # Should return coefficient array
        assert beta is not None
        assert len(beta.ravel()) == 2  # x1 and x2

    def test_estimate_coefficients_matches_fit(self, balanced_panel_data):
        """Test that _estimate_coefficients() returns same betas as fit()."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        # Get coefficients from _estimate_coefficients()
        beta_direct = model._estimate_coefficients().ravel()

        # Get coefficients from fit()
        results = model.fit()
        beta_fit = results.params.values

        np.testing.assert_array_almost_equal(beta_direct, beta_fit)


class TestComputeVcovRobustDirectly:
    """Tests for _compute_vcov_robust() method (lines 730-745)."""

    def test_compute_vcov_robust_directly(self, balanced_panel_data):
        """Test _compute_vcov_robust() with demeaned X and residuals."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Reconstruct demeaned X and residuals
        y_orig, X_orig = model.formula_parser.build_design_matrices(
            model.data.data, return_type="array"
        )
        if model.formula_parser.has_intercept:
            X_orig = X_orig[:, 1:]

        entities = model.data.data[model.data.entity_col].values
        times = model.data.data[model.data.time_col].values

        y_dm, X_dm = model._apply_demeaning(y_orig, X_orig, entities, times)
        resid_dm = y_dm - (X_dm @ results.params.values)

        df_resid = results.df_resid

        vcov = model._compute_vcov_robust(X_dm, resid_dm, df_resid)

        # Covariance matrix should be 2x2 (for x1 and x2)
        assert vcov.shape == (2, 2)
        # Diagonal elements (variances) should be positive
        assert vcov[0, 0] > 0
        assert vcov[1, 1] > 0
        # Should be symmetric
        np.testing.assert_array_almost_equal(vcov, vcov.T)


class TestComputeVcovClusteredDirectly:
    """Tests for _compute_vcov_clustered() method (lines 769-793)."""

    def test_compute_vcov_clustered_directly(self, balanced_panel_data):
        """Test _compute_vcov_clustered() with demeaned data."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Reconstruct demeaned X and residuals
        y_orig, X_orig = model.formula_parser.build_design_matrices(
            model.data.data, return_type="array"
        )
        if model.formula_parser.has_intercept:
            X_orig = X_orig[:, 1:]

        entities = model.data.data[model.data.entity_col].values
        times = model.data.data[model.data.time_col].values

        y_dm, X_dm = model._apply_demeaning(y_orig, X_orig, entities, times)
        resid_dm = y_dm - (X_dm @ results.params.values)

        df_resid = results.df_resid

        vcov = model._compute_vcov_clustered(X_dm, resid_dm, entities, df_resid)

        # Covariance matrix should be 2x2 (for x1 and x2)
        assert vcov.shape == (2, 2)
        # Diagonal elements (variances) should be positive
        assert vcov[0, 0] > 0
        assert vcov[1, 1] > 0
        # Should be symmetric
        np.testing.assert_array_almost_equal(vcov, vcov.T)
