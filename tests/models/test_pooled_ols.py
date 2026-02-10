"""
Tests for PooledOLS model.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.pooled_ols import PooledOLS


class TestPooledOLSInitialization:
    """Tests for PooledOLS initialization."""

    def test_init_basic(self, balanced_panel_data):
        """Test basic initialization."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        assert model.formula == "y ~ x1 + x2"
        assert model.data.entity_col == "entity"
        assert model.data.time_col == "time"
        assert model._fitted is False
        assert model.weights is None

    def test_init_with_weights(self, balanced_panel_data):
        """Test initialization with weights."""
        n_obs = len(balanced_panel_data)
        weights = np.random.uniform(0.5, 1.5, n_obs)
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time", weights=weights)

        assert model.weights is not None
        assert len(model.weights) == n_obs

    def test_init_with_unbalanced_data(self, unbalanced_panel_data):
        """Test initialization with unbalanced panel data."""
        model = PooledOLS("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")

        assert model.data.n_entities == 3
        assert model._fitted is False


class TestPooledOLSFitting:
    """Tests for fitting Pooled OLS models."""

    def test_fit_basic(self, balanced_panel_data):
        """Test basic model fitting."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Check that model was fitted
        assert model._fitted is True
        assert results is not None

        # Check coefficient structure
        assert len(results.params) == 3  # Intercept, x1, x2
        assert "Intercept" in results.params.index
        assert "x1" in results.params.index
        assert "x2" in results.params.index

    def test_fit_no_intercept(self, balanced_panel_data):
        """Test fitting without intercept."""
        model = PooledOLS("y ~ x1 + x2 - 1", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert "Intercept" not in results.params.index
        assert len(results.params) == 2

    def test_fit_with_unbalanced_data(self, unbalanced_panel_data):
        """Test fitting with unbalanced panel data."""
        model = PooledOLS("y ~ x1 + x2", unbalanced_panel_data, "entity", "time")
        results = model.fit()

        assert model._fitted is True
        assert results is not None
        assert len(results.params) == 3

    def test_estimate_coefficients_method(self, balanced_panel_data):
        """Test the _estimate_coefficients method."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        beta = model._estimate_coefficients()

        assert beta is not None
        assert len(beta) == 3  # Intercept, x1, x2


class TestRSquaredMeasures:
    """Tests for R-squared measures in Pooled OLS."""

    def test_rsquared_measures_exist(self, balanced_panel_data):
        """Test that all R-squared measures are computed."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert not np.isnan(results.rsquared)
        assert not np.isnan(results.rsquared_adj)
        assert not np.isnan(results.rsquared_overall)

    def test_rsquared_equals_overall(self, balanced_panel_data):
        """Test that rsquared equals overall R-squared for Pooled OLS."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # For Pooled OLS, overall R² is the main R²
        assert results.rsquared == results.rsquared_overall

    def test_within_between_rsquared_nan(self, balanced_panel_data):
        """Test that within and between R-squared are NaN for Pooled OLS."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # These don't apply to Pooled OLS
        assert np.isnan(results.rsquared_within)
        assert np.isnan(results.rsquared_between)

    def test_rsquared_bounds(self, balanced_panel_data):
        """Test that R-squared values are in valid range."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # R-squared should be between 0 and 1
        assert 0 <= results.rsquared <= 1
        assert 0 <= results.rsquared_overall <= 1

    def test_adjusted_rsquared(self, balanced_panel_data):
        """Test adjusted R-squared computation."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Adjusted R² should be less than or equal to R²
        assert results.rsquared_adj <= results.rsquared


class TestCovarianceTypes:
    """Tests for different covariance estimators."""

    def test_nonrobust_se(self, balanced_panel_data):
        """Test non-robust standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="nonrobust")

        assert results.cov_type == "nonrobust"
        assert len(results.std_errors) == 3
        assert all(results.std_errors > 0)

    def test_robust_se_hc1(self, balanced_panel_data):
        """Test robust standard errors (HC1)."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="robust")

        assert results.cov_type == "robust"
        assert len(results.std_errors) == 3

    def test_hc0_se(self, balanced_panel_data):
        """Test HC0 heteroskedasticity-robust standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="hc0")

        assert results.cov_type == "hc0"
        assert len(results.std_errors) == 3

    def test_hc2_se(self, balanced_panel_data):
        """Test HC2 heteroskedasticity-robust standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="hc2")

        assert results.cov_type == "hc2"
        assert len(results.std_errors) == 3

    def test_hc3_se(self, balanced_panel_data):
        """Test HC3 heteroskedasticity-robust standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="hc3")

        assert results.cov_type == "hc3"
        assert len(results.std_errors) == 3

    def test_clustered_se(self, balanced_panel_data):
        """Test cluster-robust standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="clustered")

        assert results.cov_type == "clustered"
        assert len(results.std_errors) == 3

    def test_twoway_cluster_se(self, balanced_panel_data):
        """Test two-way cluster-robust standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="twoway")

        assert results.cov_type == "twoway"
        assert len(results.std_errors) == 3

    def test_driscoll_kraay_se(self, balanced_panel_data):
        """Test Driscoll-Kraay standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay", max_lags=2)

        assert results.cov_type == "driscoll_kraay"
        assert results.cov_kwds["max_lags"] == 2
        assert len(results.std_errors) == 3

    def test_driscoll_kraay_se_with_kernel(self, balanced_panel_data):
        """Test Driscoll-Kraay standard errors with kernel option."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay", max_lags=3, kernel="bartlett")

        assert results.cov_type == "driscoll_kraay"
        assert results.cov_kwds["max_lags"] == 3
        assert results.cov_kwds["kernel"] == "bartlett"

    def test_driscoll_kraay_se_no_lags(self, balanced_panel_data):
        """Test Driscoll-Kraay standard errors without specifying max_lags."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay")

        assert results.cov_type == "driscoll_kraay"
        assert len(results.std_errors) == 3

    def test_newey_west_se(self, balanced_panel_data):
        """Test Newey-West HAC standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west", max_lags=2)

        assert results.cov_type == "newey_west"
        assert results.cov_kwds["max_lags"] == 2
        assert len(results.std_errors) == 3

    def test_newey_west_se_with_kernel(self, balanced_panel_data):
        """Test Newey-West standard errors with kernel option."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west", max_lags=3, kernel="bartlett")

        assert results.cov_type == "newey_west"
        assert results.cov_kwds["max_lags"] == 3
        assert results.cov_kwds["kernel"] == "bartlett"

    def test_newey_west_se_no_lags(self, balanced_panel_data):
        """Test Newey-West standard errors without specifying max_lags."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="newey_west")

        assert results.cov_type == "newey_west"
        assert len(results.std_errors) == 3

    def test_pcse(self, balanced_panel_data):
        """Test panel-corrected standard errors."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="pcse")

        assert results.cov_type == "pcse"
        assert len(results.std_errors) == 3

    def test_invalid_cov_type(self, balanced_panel_data):
        """Test that invalid cov_type raises ValueError."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="cov_type must be one of"):
            model.fit(cov_type="invalid_type")

    def test_robust_vs_nonrobust_differ(self, balanced_panel_data):
        """Test that robust SEs differ from non-robust."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        results_nonrobust = model.fit(cov_type="nonrobust")
        results_robust = model.fit(cov_type="robust")

        # Standard errors should differ
        assert not np.allclose(
            results_nonrobust.std_errors.values, results_robust.std_errors.values
        )


class TestModelInfo:
    """Tests for model information stored in results."""

    def test_model_type(self, balanced_panel_data):
        """Test model type is correctly set."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.model_type == "Pooled OLS"

    def test_formula_stored(self, balanced_panel_data):
        """Test formula is stored in results."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.formula == "y ~ x1 + x2"

    def test_cov_type_stored(self, balanced_panel_data):
        """Test covariance type is stored in results."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="robust")

        assert results.cov_type == "robust"

    def test_cov_kwds_stored(self, balanced_panel_data):
        """Test covariance kwargs are stored in results."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit(cov_type="driscoll_kraay", max_lags=3, kernel="bartlett")

        assert results.cov_kwds["max_lags"] == 3
        assert results.cov_kwds["kernel"] == "bartlett"


class TestDataInfo:
    """Tests for data information stored in results."""

    def test_nobs(self, balanced_panel_data):
        """Test number of observations."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.nobs == 50  # 10 entities * 5 periods

    def test_n_entities(self, balanced_panel_data):
        """Test number of entities."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.n_entities == 10

    def test_n_periods(self, balanced_panel_data):
        """Test number of periods."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.n_periods == 5

    def test_df_model(self, balanced_panel_data):
        """Test degrees of freedom for model."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # df_model = k - 1 (excluding intercept)
        assert results.df_model == 2

    def test_df_resid(self, balanced_panel_data):
        """Test degrees of freedom for residuals."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # df_resid = n - k
        assert results.df_resid == 47  # 50 - 3

    def test_entity_index_stored(self, balanced_panel_data):
        """Test that entity index is stored."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.entity_index is not None
        assert len(results.entity_index) == 50

    def test_time_index_stored(self, balanced_panel_data):
        """Test that time index is stored."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.time_index is not None
        assert len(results.time_index) == 50


class TestResultsAttributes:
    """Tests for results attributes."""

    def test_params_series(self, balanced_panel_data):
        """Test that params is a pandas Series."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert isinstance(results.params, pd.Series)
        assert results.params.index.tolist() == ["Intercept", "x1", "x2"]

    def test_std_errors_series(self, balanced_panel_data):
        """Test that std_errors is a pandas Series."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert isinstance(results.std_errors, pd.Series)
        assert results.std_errors.index.tolist() == ["Intercept", "x1", "x2"]

    def test_cov_params_dataframe(self, balanced_panel_data):
        """Test that cov_params is a pandas DataFrame."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert isinstance(results.cov_params, pd.DataFrame)
        assert results.cov_params.shape == (3, 3)
        assert results.cov_params.index.tolist() == ["Intercept", "x1", "x2"]
        assert results.cov_params.columns.tolist() == ["Intercept", "x1", "x2"]

    def test_residuals(self, balanced_panel_data):
        """Test residuals are computed."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.resid is not None
        assert len(results.resid) == 50

    def test_fitted_values(self, balanced_panel_data):
        """Test fitted values are computed."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results.fittedvalues is not None
        assert len(results.fittedvalues) == 50

    def test_residuals_sum_to_zero(self, balanced_panel_data):
        """Test that residuals sum to approximately zero."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        # Residuals should sum to ~0 with intercept
        assert np.abs(results.resid.sum()) < 1e-10


class TestSummaryMethods:
    """Tests for summary and display methods."""

    def test_summary_method(self, balanced_panel_data):
        """Test that summary method runs without error."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        summary = results.summary()
        assert summary is not None
        assert "Pooled OLS" in str(summary)

    def test_summary_with_different_cov_types(self, balanced_panel_data):
        """Test summary with different covariance types."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        for cov_type in ["nonrobust", "robust", "clustered"]:
            results = model.fit(cov_type=cov_type)
            summary = results.summary()
            assert summary is not None


class TestPrivateMethods:
    """Tests for private helper methods."""

    def test_compute_vcov_robust(self, balanced_panel_data):
        """Test _compute_vcov_robust method."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        # Fit to get residuals
        y, X = model.formula_parser.build_design_matrices(model.data.data, return_type="array")
        from panelbox.utils.matrix_ops import compute_ols

        beta, resid, fitted = compute_ols(y, X, None)

        # Test method
        vcov = model._compute_vcov_robust(X, resid)

        assert vcov is not None
        assert vcov.shape == (3, 3)
        assert np.allclose(vcov, vcov.T)  # Should be symmetric

    def test_compute_vcov_clustered(self, balanced_panel_data):
        """Test _compute_vcov_clustered method."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        # Fit to get residuals
        y, X = model.formula_parser.build_design_matrices(model.data.data, return_type="array")
        from panelbox.utils.matrix_ops import compute_ols

        beta, resid, fitted = compute_ols(y, X, None)

        # Test method
        vcov = model._compute_vcov_clustered(X, resid)

        assert vcov is not None
        assert vcov.shape == (3, 3)
        assert np.allclose(vcov, vcov.T)  # Should be symmetric


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_perfect_fit(self):
        """Test with perfectly fitting data (no error)."""
        np.random.seed(42)
        # Create more observations with independent x1 and x2
        data = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "time": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
                "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                "x2": [2.0, 1.0, 3.0, 2.5, 3.5, 1.5, 4.0, 2.0, 1.0, 3.0, 2.5, 4.5],
            }
        )
        # y = 1 + 2*x1 + 3*x2 (perfect fit, no noise)
        data["y"] = 1.0 + 2.0 * data["x1"] + 3.0 * data["x2"]

        model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        # Should have R² very close to 1
        assert results.rsquared > 0.999
        # Residuals should be very small
        assert np.abs(results.resid).max() < 1e-8

    def test_single_entity(self):
        """Test with single entity (multiple time periods)."""
        np.random.seed(123)
        data = pd.DataFrame(
            {
                "entity": [1] * 10,
                "time": range(10),
                "y": np.random.randn(10) * 10 + 100,
                "x1": np.random.randn(10) * 5 + 50,
                "x2": np.random.randn(10) * 3 + 30,
            }
        )

        model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        assert results is not None
        assert results.n_entities == 1

    def test_single_time_period(self):
        """Test with single time period (cross-sectional data)."""
        np.random.seed(124)
        data = pd.DataFrame(
            {
                "entity": range(10),
                "time": [1] * 10,
                "y": np.random.randn(10) * 10 + 100,
                "x1": np.random.randn(10) * 5 + 50,
                "x2": np.random.randn(10) * 3 + 30,
            }
        )

        model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        assert results is not None
        assert results.n_periods == 1

    def test_many_regressors(self, balanced_panel_data):
        """Test with multiple regressors."""
        # Add more columns
        np.random.seed(125)
        data = balanced_panel_data.copy()
        data["x3"] = np.random.randn(len(data)) * 2
        data["x4"] = np.random.randn(len(data)) * 4
        data["x5"] = np.random.randn(len(data)) * 1

        model = PooledOLS("y ~ x1 + x2 + x3 + x4 + x5", data, "entity", "time")
        results = model.fit()

        assert len(results.params) == 6  # Intercept + 5 regressors
        assert results.df_model == 5


class TestConsistency:
    """Tests for consistency across different specifications."""

    def test_cov_matrix_symmetric(self, balanced_panel_data):
        """Test that covariance matrix is symmetric."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        # Note: PCSE is excluded because balanced_panel_data has T=5 < N=10,
        # which violates PCSE requirements (T > N)
        for cov_type in ["nonrobust", "robust", "clustered"]:
            results = model.fit(cov_type=cov_type)
            cov = results.cov_params.values
            assert np.allclose(cov, cov.T), f"Covariance not symmetric for {cov_type}"

    def test_std_errors_positive(self, balanced_panel_data):
        """Test that standard errors are positive."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")

        # Note: PCSE is excluded because balanced_panel_data has T=5 < N=10,
        # which violates PCSE requirements (T > N) and can lead to NaN std errors
        for cov_type in ["nonrobust", "robust", "clustered", "twoway"]:
            results = model.fit(cov_type=cov_type)
            assert all(results.std_errors > 0), f"Negative std errors for {cov_type}"

    def test_fitted_plus_resid_equals_y(self, balanced_panel_data):
        """Test that fitted + residuals = y."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        y, _ = model.formula_parser.build_design_matrices(model.data.data, return_type="array")

        reconstructed_y = results.fittedvalues + results.resid
        assert np.allclose(y.ravel(), reconstructed_y.ravel())

    def test_results_reference_to_model(self, balanced_panel_data):
        """Test that results contains reference to model."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        assert results._model is model
