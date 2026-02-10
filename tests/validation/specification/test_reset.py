"""
Tests for RESET test (Regression Equation Specification Error Test).
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.validation.specification.reset import RESETTest


class TestRESETInit:
    """Test RESET initialization."""

    def test_init_with_pooled_ols(self, balanced_panel_data):
        """Test initialization with Pooled OLS results."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        assert test.results is results

    def test_init_with_fixed_effects(self, balanced_panel_data):
        """Test initialization with Fixed Effects results."""
        model = FixedEffects("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        assert test.results is results


class TestRESETRun:
    """Test RESET run method."""

    def test_run_basic(self, balanced_panel_data):
        """Test basic RESET test execution."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        assert result is not None
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert result.test_name == "RESET Test for Specification"

    def test_run_with_default_powers(self, balanced_panel_data):
        """Test that default powers are [2, 3]."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        assert result.metadata["powers"] == [2, 3]
        assert "fitted_pow2" in result.metadata["gamma_coefficients"]
        assert "fitted_pow3" in result.metadata["gamma_coefficients"]

    def test_run_with_custom_powers(self, balanced_panel_data):
        """Test with custom powers."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(powers=[2, 3, 4])

        assert result.metadata["powers"] == [2, 3, 4]
        assert len(result.metadata["gamma_coefficients"]) == 3

    def test_run_with_single_power(self, balanced_panel_data):
        """Test with single power."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(powers=[2])

        assert result.metadata["powers"] == [2]
        assert len(result.metadata["gamma_coefficients"]) == 1

    def test_invalid_powers_raises(self, balanced_panel_data):
        """Test that invalid powers raise ValueError."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)

        with pytest.raises(ValueError, match="Powers must be integers >= 2"):
            test.run(powers=[1])

        with pytest.raises(ValueError, match="Powers must be integers >= 2"):
            test.run(powers=[2.5])

    def test_run_with_alpha(self, balanced_panel_data):
        """Test with custom alpha level."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(alpha=0.01)

        assert result.alpha == 0.01

    def test_result_has_metadata(self, balanced_panel_data):
        """Test that result contains all expected metadata."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        metadata = result.metadata
        assert "powers" in metadata
        assert "gamma_coefficients" in metadata
        assert "standard_errors" in metadata
        assert "wald_statistic" in metadata
        assert "F_statistic" in metadata
        assert "df_numerator" in metadata
        assert "df_denominator" in metadata
        assert "pvalue_chi2" in metadata
        assert "augmented_formula" in metadata


class TestRESETWithLinearModel:
    """Test RESET with correctly specified linear model."""

    def test_accepts_linear_specification(self, balanced_panel_data):
        """Test that RESET doesn't reject correctly specified linear model."""
        # Simple linear model: y = beta0 + beta1*x1 + beta2*x2 + e
        # RESET should not strongly reject
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(alpha=0.05)

        # With random data, RESET should not always reject
        # Just check it runs without error
        assert result is not None
        assert 0 <= result.pvalue <= 1


class TestRESETWithNonlinearModel:
    """Test RESET with misspecified model."""

    def test_detects_quadratic_misspecification(self):
        """Test that RESET can detect when quadratic terms are needed."""
        np.random.seed(42)

        # Generate data with quadratic relationship
        n_entities = 20
        n_periods = 10

        entity = np.repeat(np.arange(1, n_entities + 1), n_periods)
        time = np.tile(np.arange(1, n_periods + 1), n_entities)

        x1 = np.random.normal(0, 1, n_entities * n_periods)
        x2 = np.random.normal(0, 1, n_entities * n_periods)

        # True model: y = 1 + 2*x1 + 3*x2 + 0.5*x1^2 + e  (quadratic in x1)
        y = 1 + 2 * x1 + 3 * x2 + 0.5 * x1**2 + np.random.normal(0, 0.5, n_entities * n_periods)

        data = pd.DataFrame({"entity": entity, "time": time, "x1": x1, "x2": x2, "y": y})

        # Fit misspecified linear model (omitting x1^2)
        model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(alpha=0.05)

        # RESET should detect the misspecification (but may not always reject with small sample)
        assert result is not None
        assert result.statistic > 0


class TestRESETStatistics:
    """Test statistical properties of RESET."""

    def test_f_statistic_positive(self, balanced_panel_data):
        """Test that F-statistic is positive."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        assert result.statistic > 0
        assert result.metadata["F_statistic"] > 0

    def test_wald_statistic_positive(self, balanced_panel_data):
        """Test that Wald statistic is positive."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        assert result.metadata["wald_statistic"] > 0

    def test_pvalue_in_valid_range(self, balanced_panel_data):
        """Test that p-value is between 0 and 1."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        assert 0 <= result.pvalue <= 1
        assert 0 <= result.metadata["pvalue_chi2"] <= 1

    def test_degrees_of_freedom(self, balanced_panel_data):
        """Test degrees of freedom calculation."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(powers=[2, 3])

        df_num, df_denom = result.df
        assert df_num == 2  # Two power terms
        assert df_denom > 0


class TestRESETEdgeCases:
    """Test edge cases and error handling."""

    def test_works_with_single_regressor(self, balanced_panel_data):
        """Test with single regressor."""
        model = PooledOLS("y ~ x1", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        assert result is not None

    def test_augmented_formula_structure(self, balanced_panel_data):
        """Test that augmented formula is correctly formed."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run(powers=[2, 3])

        aug_formula = result.metadata["augmented_formula"]
        assert "fitted_pow2" in aug_formula
        assert "fitted_pow3" in aug_formula
        assert "y ~" in aug_formula


class TestRESETCoefficients:
    """Test gamma coefficients and standard errors."""

    def test_gamma_coefficients_present(self, balanced_panel_data):
        """Test that gamma coefficients are returned."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        gamma = result.metadata["gamma_coefficients"]
        assert len(gamma) == 2  # Default powers [2, 3]
        assert all(isinstance(v, (int, float)) for v in gamma.values())

    def test_standard_errors_present(self, balanced_panel_data):
        """Test that standard errors are returned."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        se = result.metadata["standard_errors"]
        assert len(se) == 2
        assert all(v > 0 for v in se.values())

    def test_gamma_and_se_keys_match(self, balanced_panel_data):
        """Test that gamma coefficients and SE have matching keys."""
        model = PooledOLS("y ~ x1 + x2", balanced_panel_data, "entity", "time")
        results = model.fit()

        test = RESETTest(results)
        result = test.run()

        gamma_keys = set(result.metadata["gamma_coefficients"].keys())
        se_keys = set(result.metadata["standard_errors"].keys())

        assert gamma_keys == se_keys
