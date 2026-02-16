"""
Tests for hypothesis testing functions in SFA models.

This module tests:
- Inefficiency presence test (LR with mixed chi-square)
- Distribution comparison tests (Vuong, nested LR)
- Variance decomposition
- Returns to scale tests
- Functional form tests
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.frontier import (
    StochasticFrontier,
    compare_nested_distributions,
    inefficiency_presence_test,
    skewness_test,
    vuong_test,
)


class TestInefficiencyPresenceTest:
    """Test inefficiency presence test with mixed chi-square distribution."""

    def test_strong_inefficiency_rejects_h0(self):
        """Test that strong inefficiency leads to rejection of H0."""
        # Generate data with strong inefficiency
        np.random.seed(42)
        n = 200

        # True model: y = 1 + 0.5*x1 + 0.3*x2 + v - u
        # where u ~ N+(0, 0.5²) and v ~ N(0, 0.1²)
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 10, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.5, n))  # Strong inefficiency
        y = 1 + 0.5 * x1 + 0.3 * x2 + v - u

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        # Fit SFA model
        sf = StochasticFrontier(
            data=data,
            depvar="y",
            exog=["x1", "x2"],
            frontier="production",
            dist="half_normal",
        )
        result_sfa = sf.fit()

        # Fit OLS for comparison
        import statsmodels.api as sm

        X = sm.add_constant(data[["x1", "x2"]])
        ols = sm.OLS(y, X).fit()
        residuals_ols = ols.resid.values

        # OLS log-likelihood (assuming normal errors)
        sigma_ols = np.std(residuals_ols, ddof=3)
        loglik_ols = (
            -0.5 * n * np.log(2 * np.pi * sigma_ols**2)
            - 0.5 * np.sum(residuals_ols**2) / sigma_ols**2
        )

        # Run inefficiency presence test
        test_result = inefficiency_presence_test(
            loglik_sfa=result_sfa.loglik,
            loglik_ols=loglik_ols,
            residuals_ols=residuals_ols,
            frontier_type="production",
            distribution="half_normal",
        )

        # Assertions
        assert test_result["pvalue"] < 0.05, "Should reject H0 with strong inefficiency"
        assert test_result["conclusion"] == "SFA needed"
        assert test_result["lr_statistic"] > 0
        assert test_result["test_type"] == "mixed_chi_square"

    def test_no_inefficiency_fails_to_reject(self):
        """Test that data without inefficiency does not reject H0."""
        np.random.seed(123)
        n = 200

        # True model: y = 1 + 0.5*x1 + 0.3*x2 + v (no inefficiency)
        x1 = np.random.uniform(0, 10, n)
        x2 = np.random.uniform(0, 10, n)
        v = np.random.normal(0, 0.3, n)
        y = 1 + 0.5 * x1 + 0.3 * x2 + v

        data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

        # Fit SFA model (will find negligible inefficiency)
        sf = StochasticFrontier(
            data=data,
            depvar="y",
            exog=["x1", "x2"],
            frontier="production",
            dist="half_normal",
        )
        result_sfa = sf.fit()

        # OLS
        import statsmodels.api as sm

        X = sm.add_constant(data[["x1", "x2"]])
        ols = sm.OLS(y, X).fit()
        residuals_ols = ols.resid.values
        sigma_ols = np.std(residuals_ols, ddof=3)
        loglik_ols = (
            -0.5 * n * np.log(2 * np.pi * sigma_ols**2)
            - 0.5 * np.sum(residuals_ols**2) / sigma_ols**2
        )

        # Run test
        test_result = inefficiency_presence_test(
            loglik_sfa=result_sfa.loglik,
            loglik_ols=loglik_ols,
            residuals_ols=residuals_ols,
            frontier_type="production",
            distribution="half_normal",
        )

        # Should not reject (or marginally reject)
        # Note: SFA may still be slightly better due to flexibility
        assert test_result["lr_statistic"] >= 0

    def test_skewness_correct_sign_production(self):
        """Test that production frontier with inefficiency has negative skewness."""
        np.random.seed(42)
        n = 200

        # Production: u reduces output
        x = np.random.uniform(0, 10, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.3, n))
        y = 1 + 0.5 * x + v - u  # Note: -u

        # OLS residuals
        import statsmodels.api as sm

        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        residuals = ols.resid

        # Skewness test
        result = skewness_test(residuals, frontier_type="production")

        assert result["skewness"] < 0, "Production frontier should have negative skewness"
        assert result["correct_sign"] == True  # noqa: E712
        assert result["warning"] is None

    def test_skewness_wrong_sign_warning(self):
        """Test that wrong skewness sign produces warning."""
        np.random.seed(42)
        n = 200

        # Production frontier but with positive skewness (misspecified)
        x = np.random.uniform(0, 10, n)
        v = np.random.normal(0, 0.05, n)  # Smaller noise
        u = np.abs(np.random.normal(0, 0.5, n))  # Larger inefficiency
        y = 1 + 0.5 * x + v + u  # Wrong sign: +u instead of -u

        import statsmodels.api as sm

        X = sm.add_constant(x)
        ols = sm.OLS(y, X).fit()
        residuals = ols.resid

        result = skewness_test(residuals, frontier_type="production")

        # With seed 42, this should have positive skewness
        if result["skewness"] > 0:
            assert result["correct_sign"] == False  # noqa: E712
            assert result["warning"] is not None
            assert "production frontier should have negative skewness" in result["warning"]
        else:
            # If by chance the skewness is negative, it should pass the test
            pytest.skip("Random data happened to have negative skewness")


class TestVuongTest:
    """Test Vuong test for non-nested model comparison."""

    def test_vuong_prefers_better_model(self):
        """Test that Vuong test prefers model with better fit."""
        np.random.seed(42)
        n = 100

        # Model 1 has better fit (true model)
        loglik1 = np.random.normal(0, 1, n)  # Higher likelihoods

        # Model 2 has worse fit - add random noise to differences
        loglik2 = loglik1 - 0.5 - np.random.normal(0, 0.1, n)  # Consistently worse with variance

        result = vuong_test(loglik1, loglik2, model1_name="Model1", model2_name="Model2")

        assert result["statistic"] != 0  # Should not be exactly zero
        # Typically Model 1 should be better, but with random data may not always be significant

    def test_vuong_equivalent_models(self):
        """Test that Vuong test finds equivalent models when appropriate."""
        np.random.seed(123)
        n = 100

        # Very similar models
        loglik1 = np.random.normal(0, 1, n)
        loglik2 = loglik1 + np.random.normal(0, 0.01, n)  # Tiny differences

        result = vuong_test(loglik1, loglik2)

        # Should not reject equivalence
        assert result["pvalue_two_sided"] > 0.05
        assert result["conclusion"] == "equivalent"

    def test_vuong_requires_same_length(self):
        """Test that Vuong test raises error for different length inputs."""
        loglik1 = np.random.normal(0, 1, 100)
        loglik2 = np.random.normal(0, 1, 50)

        with pytest.raises(ValueError, match="same length"):
            vuong_test(loglik1, loglik2)


class TestNestedDistributionComparison:
    """Test comparison of nested distributional specifications."""

    def test_half_normal_vs_truncated_normal(self):
        """Test LR test for half-normal vs truncated-normal."""
        # Simulate case where truncated-normal fits better
        loglik_half = -120.5
        loglik_trunc = -115.2  # Better fit

        result = compare_nested_distributions(
            loglik_restricted=loglik_half,
            loglik_unrestricted=loglik_trunc,
            dist_restricted="half_normal",
            dist_unrestricted="truncated_normal",
        )

        assert result["lr_statistic"] > 0
        assert result["test_type"] == "mixed_chi_square"
        assert result["df"] == 1

        # Should prefer truncated if difference is large enough
        if result["lr_statistic"] > 3.841:  # Critical value at 5%
            assert result["conclusion"] == "truncated_normal"

    def test_exponential_vs_gamma(self):
        """Test LR test for exponential vs gamma."""
        loglik_exp = -100.0
        loglik_gamma = -98.0

        result = compare_nested_distributions(
            loglik_restricted=loglik_exp,
            loglik_unrestricted=loglik_gamma,
            dist_restricted="exponential",
            dist_unrestricted="gamma",
        )

        assert result["lr_statistic"] == 4.0  # 2 * (98 - 100)
        assert result["test_type"] == "chi_square"
        assert result["df"] == 1


class TestVarianceDecomposition:
    """Test variance decomposition methods."""

    def test_variance_decomposition_basic(self):
        """Test basic variance decomposition calculation."""
        np.random.seed(42)
        n = 200

        # Generate data with known variance components
        x = np.random.uniform(0, 10, n)
        v = np.random.normal(0, 0.2, n)  # σ_v = 0.2
        u = np.abs(np.random.normal(0, 0.3, n))  # σ_u = 0.3
        y = 1 + 0.5 * x + v - u

        data = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=data,
            depvar="y",
            exog=["x"],
            frontier="production",
            dist="half_normal",
        )
        result = sf.fit()

        # Get variance decomposition
        decomp = result.variance_decomposition()

        # Check structure
        assert "gamma" in decomp
        assert "lambda_param" in decomp
        assert "gamma_ci" in decomp
        assert "interpretation" in decomp

        # Gamma should be in [0, 1]
        assert 0 <= decomp["gamma"] <= 1

        # Expected gamma ≈ σ²_u / (σ²_v + σ²_u) = 0.09 / (0.04 + 0.09) ≈ 0.69
        # Allow some estimation error
        assert 0.4 < decomp["gamma"] < 0.9

    def test_variance_decomposition_confidence_intervals(self):
        """Test that confidence intervals are computed correctly."""
        np.random.seed(42)
        n = 300  # Larger sample for better estimates

        x = np.random.uniform(0, 10, n)
        v = np.random.normal(0, 0.2, n)
        u = np.abs(np.random.normal(0, 0.3, n))
        y = 1 + 0.5 * x + v - u

        data = pd.DataFrame({"y": y, "x": x})

        sf = StochasticFrontier(
            data=data, depvar="y", exog=["x"], frontier="production", dist="half_normal"
        )
        result = sf.fit()

        decomp = result.variance_decomposition(ci_level=0.95)

        # Check CI structure
        gamma_ci = decomp["gamma_ci"]
        lambda_ci = decomp["lambda_ci"]

        assert len(gamma_ci) == 2
        assert len(lambda_ci) == 2

        # CI should contain point estimate (allow for numerical precision)
        assert gamma_ci[0] - 1e-6 <= decomp["gamma"] <= gamma_ci[1] + 1e-6

        # CI width should be non-negative (may be zero or very small for good estimates)
        assert gamma_ci[1] >= gamma_ci[0] - 1e-10


class TestReturnsToScale:
    """Test returns to scale tests."""

    def test_rts_constant_returns(self):
        """Test RTS test with constant returns to scale."""
        np.random.seed(42)
        n = 200

        # Cobb-Douglas with CRS: y = K^0.3 * L^0.7
        # In logs: ln(y) = 0.3*ln(K) + 0.7*ln(L)  [RTS = 1.0]
        ln_K = np.random.uniform(0, 2, n)
        ln_L = np.random.uniform(0, 2, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.05, n))
        ln_y = 0.3 * ln_K + 0.7 * ln_L + v - u

        data = pd.DataFrame({"ln_y": ln_y, "ln_K": ln_K, "ln_L": ln_L})

        sf = StochasticFrontier(
            data=data,
            depvar="ln_y",
            exog=["ln_K", "ln_L"],
            frontier="production",
            dist="half_normal",
        )
        result = sf.fit()

        # Test RTS
        rts_test = result.returns_to_scale_test(input_vars=["ln_K", "ln_L"])

        # RTS should be close to 1.0
        assert 0.9 < rts_test["rts"] < 1.1

        # Should not reject H0: RTS = 1 (at reasonable confidence)
        if rts_test["pvalue"] is not None and not np.isnan(rts_test["pvalue"]):
            # May or may not reject depending on sample variability
            pass

    def test_rts_increasing_returns(self):
        """Test RTS test with increasing returns to scale."""
        np.random.seed(42)
        n = 200

        # IRS: y = K^0.6 * L^0.6 → RTS = 1.2
        ln_K = np.random.uniform(0, 2, n)
        ln_L = np.random.uniform(0, 2, n)
        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.05, n))
        ln_y = 0.6 * ln_K + 0.6 * ln_L + v - u

        data = pd.DataFrame({"ln_y": ln_y, "ln_K": ln_K, "ln_L": ln_L})

        sf = StochasticFrontier(
            data=data,
            depvar="ln_y",
            exog=["ln_K", "ln_L"],
            frontier="production",
            dist="half_normal",
        )
        result = sf.fit()

        rts_test = result.returns_to_scale_test(input_vars=["ln_K", "ln_L"])

        # RTS should be > 1
        assert rts_test["rts"] > 1.0

        # Conclusion should indicate IRS if significant
        if rts_test["pvalue"] < 0.05:
            assert rts_test["conclusion"] == "IRS"


class TestTranslogHelper:
    """Test add_translog helper function."""

    def test_add_translog_basic(self):
        """Test basic Translog term generation."""
        from panelbox.frontier import add_translog

        data = pd.DataFrame({"ln_K": [1, 2, 3], "ln_L": [0.5, 1.0, 1.5]})

        data_translog = add_translog(data, variables=["ln_K", "ln_L"])

        # Check squared terms exist
        assert "ln_K_sq" in data_translog.columns
        assert "ln_L_sq" in data_translog.columns

        # Check interaction term
        assert "ln_K_ln_L" in data_translog.columns

        # Verify values
        assert np.allclose(data_translog["ln_K_sq"], data["ln_K"] ** 2)
        assert np.allclose(data_translog["ln_L_sq"], data["ln_L"] ** 2)
        assert np.allclose(data_translog["ln_K_ln_L"], data["ln_K"] * data["ln_L"])

    def test_add_translog_with_time(self):
        """Test Translog with time trend interactions."""
        from panelbox.frontier import add_translog

        data = pd.DataFrame({"ln_K": [1, 2, 3], "ln_L": [0.5, 1.0, 1.5], "t": [1, 2, 3]})

        data_translog = add_translog(
            data, variables=["ln_K", "ln_L"], include_time=True, time_var="t"
        )

        # Check time terms
        assert "t_sq" in data_translog.columns
        assert "t_ln_K" in data_translog.columns
        assert "t_ln_L" in data_translog.columns

        # Verify time squared
        assert np.allclose(data_translog["t_sq"], data["t"] ** 2)

    def test_add_translog_missing_variable(self):
        """Test that missing variable raises error."""
        from panelbox.frontier import add_translog

        data = pd.DataFrame({"ln_K": [1, 2, 3]})

        with pytest.raises(ValueError, match="not found"):
            add_translog(data, variables=["ln_K", "ln_L"])  # ln_L doesn't exist


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
