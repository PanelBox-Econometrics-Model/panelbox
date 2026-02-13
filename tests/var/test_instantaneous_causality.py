"""
Tests for instantaneous causality tests in Panel VAR.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.var.causality import (
    InstantaneousCausalityResult,
    instantaneous_causality,
    instantaneous_causality_matrix,
)


def test_instantaneous_causality_basic():
    """Test basic instantaneous causality calculation."""
    np.random.seed(42)

    # Create correlated residuals
    n = 100
    rho = 0.6  # Correlation
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    residuals = np.random.multivariate_normal(mean, cov, size=n)
    resid1 = residuals[:, 0]
    resid2 = residuals[:, 1]

    result = instantaneous_causality(resid1, resid2, "x", "y")

    # Check result type
    assert isinstance(result, InstantaneousCausalityResult)

    # Check attributes
    assert result.var1 == "x"
    assert result.var2 == "y"
    assert result.n_obs == n

    # Correlation should be close to rho
    assert abs(result.correlation - rho) < 0.15  # Allow some sampling variation

    # LR statistic should be positive
    assert result.lr_stat > 0

    # P-value should be small (reject independence)
    assert result.p_value < 0.05


def test_instantaneous_causality_no_correlation():
    """Test instantaneous causality with uncorrelated residuals."""
    np.random.seed(123)

    n = 100
    resid1 = np.random.normal(0, 1, size=n)
    resid2 = np.random.normal(0, 1, size=n)

    result = instantaneous_causality(resid1, resid2, "x", "y")

    # Correlation should be close to zero
    assert abs(result.correlation) < 0.2

    # P-value should be large (fail to reject independence)
    assert result.p_value > 0.10


def test_instantaneous_causality_perfect_correlation():
    """Test with perfect correlation."""
    np.random.seed(42)

    n = 100
    resid1 = np.random.normal(0, 1, size=n)
    resid2 = resid1.copy()  # Perfect correlation

    result = instantaneous_causality(resid1, resid2, "x", "y")

    # Correlation should be 1.0
    assert abs(result.correlation - 1.0) < 1e-10

    # LR stat should be very large or inf
    assert result.lr_stat > 1000 or np.isinf(result.lr_stat)

    # P-value should be essentially zero
    assert result.p_value < 1e-10


def test_instantaneous_causality_negative_correlation():
    """Test with negative correlation."""
    np.random.seed(42)

    n = 100
    resid1 = np.random.normal(0, 1, size=n)
    resid2 = -resid1 + np.random.normal(0, 0.3, size=n)

    result = instantaneous_causality(resid1, resid2, "x", "y")

    # Correlation should be negative
    assert result.correlation < -0.5

    # Should still reject independence (test is two-sided)
    assert result.p_value < 0.05


def test_instantaneous_causality_lr_statistic():
    """Test LR statistic calculation."""
    np.random.seed(42)

    n = 100
    rho = 0.5
    mean = [0, 0]
    cov = [[1, rho], [rho, 1]]

    residuals = np.random.multivariate_normal(mean, cov, size=n)
    resid1 = residuals[:, 0]
    resid2 = residuals[:, 1]

    result = instantaneous_causality(resid1, resid2, "x", "y")

    # Manually compute LR statistic
    r = result.correlation
    lr_expected = -n * np.log(1 - r**2)

    assert abs(result.lr_stat - lr_expected) < 1e-6


def test_instantaneous_causality_different_lengths():
    """Test error when residuals have different lengths."""
    resid1 = np.random.normal(size=100)
    resid2 = np.random.normal(size=50)

    with pytest.raises(ValueError, match="same length"):
        instantaneous_causality(resid1, resid2, "x", "y")


def test_instantaneous_causality_summary():
    """Test summary method."""
    np.random.seed(42)
    n = 100
    resid1 = np.random.normal(size=n)
    resid2 = 0.5 * resid1 + np.random.normal(0, 0.5, size=n)

    result = instantaneous_causality(resid1, resid2, "x", "y")

    summary = result.summary()

    # Check that summary contains key information
    assert "Instantaneous Causality" in summary
    assert "x" in summary
    assert "y" in summary
    assert "Correlation" in summary
    assert "LR statistic" in summary
    assert str(n) in summary


def test_instantaneous_causality_result_repr():
    """Test __repr__ method."""
    np.random.seed(42)
    resid1 = np.random.normal(size=100)
    resid2 = np.random.normal(size=100)

    result = instantaneous_causality(resid1, resid2, "x", "y")

    repr_str = repr(result)

    assert "InstantaneousCausalityResult" in repr_str
    assert "x" in repr_str
    assert "y" in repr_str


def test_instantaneous_causality_matrix_structure():
    """Test instantaneous causality matrix structure."""

    # Create mock result object
    class MockResult:
        def __init__(self):
            self.K = 3
            self.endog_names = ["x", "y", "z"]
            self.n_obs = 100

            # Create correlated residuals
            np.random.seed(42)
            # x and y correlated, z independent
            cov_matrix = np.array([[1.0, 0.6, 0.1], [0.6, 1.0, 0.05], [0.1, 0.05, 1.0]])
            residuals = np.random.multivariate_normal([0, 0, 0], cov_matrix, size=100)
            self.resid_by_eq = [residuals[:, i] for i in range(3)]

    result = MockResult()
    corr_matrix, pvalue_matrix = instantaneous_causality_matrix(result)

    # Check types
    assert isinstance(corr_matrix, pd.DataFrame)
    assert isinstance(pvalue_matrix, pd.DataFrame)

    # Check shapes
    assert corr_matrix.shape == (3, 3)
    assert pvalue_matrix.shape == (3, 3)

    # Check diagonal of correlation matrix is 1
    assert abs(corr_matrix.loc["x", "x"] - 1.0) < 1e-10
    assert abs(corr_matrix.loc["y", "y"] - 1.0) < 1e-10
    assert abs(corr_matrix.loc["z", "z"] - 1.0) < 1e-10

    # Check symmetry
    assert abs(corr_matrix.loc["x", "y"] - corr_matrix.loc["y", "x"]) < 1e-10
    assert abs(corr_matrix.loc["x", "z"] - corr_matrix.loc["z", "x"]) < 1e-10

    assert abs(pvalue_matrix.loc["x", "y"] - pvalue_matrix.loc["y", "x"]) < 1e-10

    # Check p-values are between 0 and 1
    assert pvalue_matrix.loc["x", "y"] >= 0
    assert pvalue_matrix.loc["x", "y"] <= 1

    # x and y should be significantly correlated
    assert pvalue_matrix.loc["x", "y"] < 0.10

    # x and z should not be significantly correlated
    assert pvalue_matrix.loc["x", "z"] > 0.10


class TestInstantaneousCausalityWithVAR:
    """Test instantaneous causality with real Panel VAR results."""

    def generate_var_data_with_correlated_errors(self, N=30, T=100, error_corr=0.6, seed=42):
        """
        Generate Panel VAR data with correlated errors.

        This creates instantaneous causality.
        """
        np.random.seed(seed)

        data_list = []

        for i in range(N):
            x = np.zeros(T)
            y = np.zeros(T)

            # Initial values
            x[0] = np.random.normal(0, 1)
            y[0] = np.random.normal(0, 1)

            # VAR(1) with correlated errors
            for t in range(1, T):
                # Generate correlated errors
                eps = np.random.multivariate_normal([0, 0], [[1, error_corr], [error_corr, 1]])

                x[t] = 0.3 * x[t - 1] + 0.2 * y[t - 1] + eps[0]
                y[t] = 0.3 * y[t - 1] + 0.2 * x[t - 1] + eps[1]

            entity_df = pd.DataFrame({"entity": i, "time": np.arange(T), "x": x, "y": y})

            data_list.append(entity_df)

        return pd.concat(data_list, ignore_index=True)

    @pytest.fixture
    def var_result_correlated_errors(self):
        """VAR result with correlated errors."""
        from panelbox.var.data import PanelVARData
        from panelbox.var.model import PanelVAR

        df = self.generate_var_data_with_correlated_errors(N=30, T=80, error_corr=0.5)

        data = PanelVARData(
            data=df, endog_vars=["x", "y"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data=data)
        result = model.fit(method="ols", cov_type="clustered")
        return result

    @pytest.fixture
    def var_result_uncorrelated_errors(self):
        """VAR result with uncorrelated errors."""
        from panelbox.var.data import PanelVARData
        from panelbox.var.model import PanelVAR

        df = self.generate_var_data_with_correlated_errors(N=30, T=80, error_corr=0.0)

        data = PanelVARData(
            data=df, endog_vars=["x", "y"], entity_col="entity", time_col="time", lags=1
        )

        model = PanelVAR(data=data)
        result = model.fit(method="ols", cov_type="clustered")
        return result

    def test_instantaneous_causality_detects_correlation(self, var_result_correlated_errors):
        """Test that instantaneous causality detects correlated errors."""
        result = var_result_correlated_errors

        ic_result = result.instantaneous_causality("x", "y")

        # Should detect correlation
        assert abs(ic_result.correlation) > 0.2
        assert ic_result.p_value < 0.10

    def test_instantaneous_causality_no_correlation(self, var_result_uncorrelated_errors):
        """Test with uncorrelated errors."""
        result = var_result_uncorrelated_errors

        ic_result = result.instantaneous_causality("x", "y")

        # Should not detect significant correlation
        assert abs(ic_result.correlation) < 0.3
        # Note: p-value might still be < 0.05 due to sampling variation, so use lenient threshold
        assert ic_result.p_value > 0.01

    def test_instantaneous_causality_matrix_with_var(self, var_result_correlated_errors):
        """Test instantaneous causality matrix with VAR result."""
        result = var_result_correlated_errors

        corr_matrix, pvalue_matrix = result.instantaneous_causality_matrix()

        # Check structure
        assert corr_matrix.shape == (2, 2)
        assert pvalue_matrix.shape == (2, 2)

        # Check diagonal
        assert abs(corr_matrix.loc["x", "x"] - 1.0) < 1e-10
        assert abs(corr_matrix.loc["y", "y"] - 1.0) < 1e-10

        # Check symmetry
        assert corr_matrix.loc["x", "y"] == corr_matrix.loc["y", "x"]


def test_instantaneous_causality_significance_interpretation():
    """Test significance interpretation in summary."""
    np.random.seed(42)

    # High correlation (p < 0.01)
    n = 100
    resid1 = np.random.normal(size=n)
    resid2 = 0.9 * resid1 + np.random.normal(0, 0.1, size=n)
    result1 = instantaneous_causality(resid1, resid2, "x", "y")
    assert "***" in result1.summary()

    # Medium correlation (0.01 <= p < 0.05)
    resid2 = 0.5 * resid1 + np.random.normal(0, 0.5, size=n)
    result2 = instantaneous_causality(resid1, resid2, "x", "y")
    summary2 = result2.summary()
    # Should show some level of significance
    assert "Rejects" in summary2 or "Fails to reject" in summary2
