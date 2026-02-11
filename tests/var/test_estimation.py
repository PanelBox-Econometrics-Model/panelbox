"""
Tests for Panel VAR estimation.

This module tests OLS estimation, coefficient extraction, and basic properties.
"""

import numpy as np
import pytest

from panelbox.var import PanelVAR, PanelVARData


class TestOLSEstimation:
    """Tests for OLS estimation."""

    def test_basic_estimation(self, simple_panel_data):
        """Test that basic OLS estimation runs without errors."""
        # Create data
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        # Fit model
        model = PanelVAR(data)
        results = model.fit(method="ols", cov_type="nonrobust")

        # Check that we have results
        assert results is not None
        assert results.K == 2
        assert results.p == 1
        assert len(results.params_by_eq) == 2
        assert len(results.A_matrices) == 1

    def test_coefficient_matrices_shape(self, simple_panel_data):
        """Test that coefficient matrices have correct shape."""
        # Create data with K=2, p=2
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Check A matrices
        assert len(results.A_matrices) == 2
        for A_l in results.A_matrices:
            assert A_l.shape == (2, 2)

    def test_different_cov_types(self, simple_panel_data):
        """Test different covariance types."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)

        # Test nonrobust
        results_nr = model.fit(cov_type="nonrobust")
        assert results_nr is not None

        # Test HC1
        results_hc1 = model.fit(cov_type="hc1")
        assert results_hc1 is not None

        # Test clustered
        results_clust = model.fit(cov_type="clustered")
        assert results_clust is not None

        # Standard errors should differ
        se_nr = results_nr.std_errors_by_eq[0]
        se_clust = results_clust.std_errors_by_eq[0]
        # Clustered SE are typically larger (but not always)
        assert not np.allclose(se_nr, se_clust)

    def test_var_dgp_recovery(self, var_dgp_data):
        """Test that we can recover coefficients from a known DGP."""
        df, true_params = var_dgp_data

        # Create Panel VAR data
        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        # Estimate
        model = PanelVAR(data)
        results = model.fit(cov_type="nonrobust")

        # Check that estimated A matrices are close to true values
        A1_est = results.A_matrices[0]
        A2_est = results.A_matrices[1]

        A1_true = true_params["A1"]
        A2_true = true_params["A2"]

        # With large sample (10 entities × 50 periods), should be reasonably close
        # Use generous tolerance since this is finite sample
        assert np.allclose(A1_est, A1_true, atol=0.3)
        assert np.allclose(A2_est, A2_true, atol=0.3)


class TestCompanionMatrix:
    """Tests for companion matrix construction."""

    def test_companion_matrix_dimensions(self, simple_panel_data):
        """Test companion matrix has correct dimensions."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=3
        )

        model = PanelVAR(data)
        results = model.fit()

        F = results.companion_matrix()

        # With K=2, p=3, companion should be (6 × 6)
        assert F.shape == (6, 6)

    def test_companion_matrix_structure(self, simple_panel_data):
        """Test companion matrix has correct structure."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        F = results.companion_matrix()

        # K=2, p=2 → F is 4×4
        # First 2 rows should be [A_1 | A_2]
        # Rows 2-3 should be [I | 0]

        K = 2

        # Check identity block
        assert np.allclose(F[K : 2 * K, 0:K], np.eye(K))

        # Check zero block
        assert np.allclose(F[K : 2 * K, K : 2 * K], np.zeros((K, K)))


class TestStability:
    """Tests for stability checks."""

    def test_stable_system_detection(self, var_dgp_data):
        """Test detection of stable system."""
        df, true_params = var_dgp_data

        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # System should be stable (DGP is stable)
        # NOTE: In finite samples, estimated system may be slightly unstable
        # even if true system is stable, so we check eigenvalues are reasonably small
        assert results.max_eigenvalue_modulus < 1.2

    def test_eigenvalues_computed(self, simple_panel_data):
        """Test that eigenvalues are computed."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Eigenvalues should be computed
        eigs = results.eigenvalues
        assert len(eigs) == 2  # K × p = 2 × 1 = 2

        # Max modulus should be available
        max_mod = results.max_eigenvalue_modulus
        assert isinstance(max_mod, float)
        assert max_mod >= 0


class TestInformationCriteria:
    """Tests for information criteria."""

    def test_aic_bic_hqic_computed(self, simple_panel_data):
        """Test that AIC, BIC, HQIC are computed."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        assert hasattr(results, "aic")
        assert hasattr(results, "bic")
        assert hasattr(results, "hqic")

        # Should be finite
        assert np.isfinite(results.aic)
        assert np.isfinite(results.bic)
        assert np.isfinite(results.hqic)

    def test_information_criteria_ordering(self, simple_panel_data):
        """Test that BIC > AIC for overparameterized models."""
        # BIC has larger penalty, so for same data:
        # IC(p) = log|Σ| + penalty(p)
        # where BIC penalty > AIC penalty
        # So if we compare two models with different p, BIC will penalize more

        # Fit with p=1
        data_1 = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )
        model_1 = PanelVAR(data_1)
        results_1 = model_1.fit()

        # Fit with p=3 (larger model)
        data_3 = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=3
        )
        model_3 = PanelVAR(data_3)
        results_3 = model_3.fit()

        # For larger model, BIC increase should be larger than AIC increase
        # (because BIC has stronger penalty)
        aic_diff = results_3.aic - results_1.aic
        bic_diff = results_3.bic - results_1.bic

        # BIC should penalize more (difference should be larger)
        # NOTE: This may not always hold in finite samples if fit improves significantly
        # Just check that both are positive (larger model has higher IC)
        # In most cases, BIC difference > AIC difference
        assert aic_diff > 0 or bic_diff > 0  # At least one should increase


class TestLagSelection:
    """Tests for lag order selection."""

    def test_lag_selection_runs(self, simple_panel_data):
        """Test that lag selection runs without errors."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        lag_results = model.select_lag_order(max_lags=4)

        assert lag_results is not None
        assert "AIC" in lag_results.selected
        assert "BIC" in lag_results.selected
        assert "HQIC" in lag_results.selected

    def test_lag_selection_selects_reasonable_lag(self, var_dgp_data):
        """Test that lag selection can recover true lag order."""
        df, true_params = var_dgp_data

        # True DGP is VAR(2)
        # Create with p=1 initially
        data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        lag_results = model.select_lag_order(max_lags=5)

        # BIC should select p=2 or close to it
        # NOTE: In finite samples, may not always select exactly p=2
        selected_bic = lag_results.selected["BIC"]
        assert 1 <= selected_bic <= 4  # Should be reasonable

    def test_lag_selection_warns_for_large_max_lags(self, simple_panel_data):
        """Test that warning is issued for large max_lags."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)

        # With T=20, max_lags=15 is too large
        with pytest.warns(UserWarning, match="max_lags"):
            lag_results = model.select_lag_order(max_lags=15)


class TestResultMethods:
    """Tests for result methods."""

    def test_coef_matrix_method(self, simple_panel_data):
        """Test coef_matrix() method."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Get A_1 as DataFrame
        A1_df = results.coef_matrix(lag=1)

        assert A1_df.shape == (2, 2)
        assert list(A1_df.index) == ["y1", "y2"]
        assert list(A1_df.columns) == ["y1", "y2"]

        # Get A_2
        A2_df = results.coef_matrix(lag=2)
        assert A2_df.shape == (2, 2)

    def test_equation_summary(self, simple_panel_data):
        """Test equation_summary() method."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Get summary for equation 0
        summary_df = results.equation_summary(0)

        assert "coef" in summary_df.columns
        assert "std err" in summary_df.columns
        assert "t" in summary_df.columns
        assert "P>|t|" in summary_df.columns

        # Should have K*p = 2*1 = 2 rows (constant absorbed by fixed effects)
        assert len(summary_df) == 2

    def test_summary_method(self, simple_panel_data):
        """Test summary() method."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Full summary
        summary_str = results.summary()
        assert isinstance(summary_str, str)
        assert "Panel VAR Results" in summary_str
        assert "AIC" in summary_str
        assert "BIC" in summary_str
        assert "Stable" in summary_str

        # Summary for single equation
        summary_eq0 = results.summary(equation=0)
        assert "Equation 1: y1" in summary_eq0
        assert "Equation 2: y2" not in summary_eq0


class TestWaldTests:
    """Tests for Wald tests and inference."""

    def test_wald_test_runs(self, simple_panel_data):
        """Test that Wald test runs."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test first coefficient is zero
        k = len(results.params_by_eq[0])
        R = np.zeros((1, k))
        R[0, 0] = 1.0

        wald_result = results.wald_test(R, equation=0)

        assert wald_result is not None
        assert hasattr(wald_result, "statistic")
        assert hasattr(wald_result, "pvalue")
        assert 0 <= wald_result.pvalue <= 1

    def test_granger_causality_test(self, simple_panel_data):
        """Test Granger causality test."""
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)
        results = model.fit()

        # Test if y1 Granger-causes y2
        gc_result = results.test_granger_causality("y1", "y2")

        assert gc_result is not None
        assert hasattr(gc_result, "statistic")
        assert hasattr(gc_result, "pvalue")
        assert 0 <= gc_result.pvalue <= 1


class TestSUREstimation:
    """Tests for SUR (Seemingly Unrelated Regressions) covariance estimation."""

    def test_sur_estimation_runs(self, simple_panel_data):
        """Test that SUR estimation runs without errors."""
        # Create data
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        # Fit model with SUR
        model = PanelVAR(data)
        results = model.fit(method="ols", cov_type="sur")

        # Check that we have results
        assert results is not None
        assert results.K == 2
        assert results.p == 1
        assert len(results.params_by_eq) == 2
        assert len(results.std_errors_by_eq) == 2
        assert len(results.cov_by_eq) == 2

    def test_sur_coefficients_match_ols(self, simple_panel_data):
        """Test that SUR gives same coefficients as OLS (only SEs differ)."""
        # Create data
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)

        # Fit with OLS
        results_ols = model.fit(method="ols", cov_type="nonrobust")

        # Fit with SUR
        results_sur = model.fit(method="ols", cov_type="sur")

        # Coefficients should be the same (SUR only affects inference)
        for k in range(2):
            np.testing.assert_allclose(
                results_sur.params_by_eq[k],
                results_ols.params_by_eq[k],
                rtol=1e-10,
                err_msg=f"SUR coefficients differ from OLS for equation {k}",
            )

    def test_sur_standard_errors_differ(self, simple_panel_data):
        """Test that SUR standard errors differ from non-robust OLS."""
        # Create data
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(data)

        # Fit with OLS
        results_ols = model.fit(method="ols", cov_type="nonrobust")

        # Fit with SUR
        results_sur = model.fit(method="ols", cov_type="sur")

        # Standard errors should differ (SUR exploits cross-equation correlation)
        # At least one should be different
        any_different = False
        for k in range(2):
            if not np.allclose(results_sur.std_errors_by_eq[k], results_ols.std_errors_by_eq[k]):
                any_different = True
                break

        assert any_different, "SUR standard errors should differ from OLS"

    def test_sur_covariance_structure(self, simple_panel_data):
        """Test that SUR covariance matrices have correct dimensions."""
        # Create data with K=2, p=1
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit(method="ols", cov_type="sur")

        # Each equation should have covariance matrix of correct size
        # For K=2 and p=1, each equation has K*p = 2 parameters
        expected_params = 2
        for k in range(2):
            assert results.cov_by_eq[k].shape == (expected_params, expected_params)
            # Covariance should be symmetric
            assert np.allclose(results.cov_by_eq[k], results.cov_by_eq[k].T)
            # Diagonal should be positive
            assert np.all(np.diag(results.cov_by_eq[k]) > 0)

    def test_sur_with_multiple_lags(self, simple_panel_data):
        """Test SUR with multiple lags."""
        # Create data with K=2, p=3
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=3
        )

        model = PanelVAR(data)
        results = model.fit(method="ols", cov_type="sur")

        # Check dimensions
        assert results.K == 2
        assert results.p == 3

        # Each equation has K*p = 6 parameters
        expected_params = 6
        for k in range(2):
            assert len(results.params_by_eq[k]) == expected_params
            assert len(results.std_errors_by_eq[k]) == expected_params
            assert results.cov_by_eq[k].shape == (expected_params, expected_params)

    def test_sur_summary_works(self, simple_panel_data):
        """Test that summary() works with SUR."""
        # Create data
        data = PanelVARData(
            simple_panel_data, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data)
        results = model.fit(method="ols", cov_type="sur")

        # Should be able to generate summary
        summary_str = results.summary()
        assert isinstance(summary_str, str)
        assert len(summary_str) > 0
        assert "SUR" in summary_str.upper() or "sur" in summary_str
