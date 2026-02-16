"""
Test validation of LM spatial tests against R's splm package.

This module tests the Python implementation of LM tests for spatial dependence
by comparing results with R's splm package using the same test dataset.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Try statsmodels import
try:
    import patsy
    from statsmodels.regression.linear_model import OLS

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from panelbox.diagnostics.spatial_tests import (
    LMTestResult,
    lm_error_test,
    lm_lag_test,
    robust_lm_error_test,
    robust_lm_lag_test,
    run_lm_tests,
)

# Path to test fixtures
FIXTURES_PATH = Path(__file__).parent.parent / "spatial" / "fixtures"


@pytest.fixture
def spatial_test_data():
    """Load spatial test data generated for validation."""
    df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")
    W = np.loadtxt(FIXTURES_PATH / "spatial_weights.csv", delimiter=",")
    return df, W


@pytest.fixture
def r_lm_results():
    """Load R LM test results for comparison."""
    results_path = FIXTURES_PATH / "r_lm_results.json"

    if not results_path.exists():
        pytest.skip("R results file not found. Run r_lm_validation.R first to generate results.")

    with open(results_path, "r") as f:
        return json.load(f)


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
class TestLMValidation:
    """Validate LM tests against R splm package results."""

    def test_lm_lag_statistic(self, spatial_test_data, r_lm_results):
        """Test that LM-Lag statistic matches R within tolerance."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run LM-Lag test
        lm_result = lm_lag_test(ols_result.resid, ols_result.model.exog, W)

        # Compare with R (allow 10% relative tolerance)
        r_stat = r_lm_results["lm_lag_stat"]
        py_stat = lm_result.statistic

        assert np.isclose(
            py_stat, r_stat, rtol=0.10
        ), f"LM-Lag stat: Python={py_stat:.4f} vs R={r_stat:.4f}"

        # Check p-value as well
        r_pval = r_lm_results["lm_lag_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(
            py_pval, r_pval, rtol=0.15
        ), f"LM-Lag p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"

    def test_lm_error_statistic(self, spatial_test_data, r_lm_results):
        """Test that LM-Error statistic matches R within tolerance."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run LM-Error test
        lm_result = lm_error_test(ols_result.resid, ols_result.model.exog, W)

        # Compare with R
        r_stat = r_lm_results["lm_error_stat"]
        py_stat = lm_result.statistic

        assert np.isclose(
            py_stat, r_stat, rtol=0.10
        ), f"LM-Error stat: Python={py_stat:.4f} vs R={r_stat:.4f}"

        # Check p-value
        r_pval = r_lm_results["lm_error_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(
            py_pval, r_pval, rtol=0.15
        ), f"LM-Error p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"

    def test_robust_lm_lag(self, spatial_test_data, r_lm_results):
        """Test that Robust LM-Lag matches R within tolerance."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run Robust LM-Lag test
        lm_result = robust_lm_lag_test(ols_result.resid, ols_result.model.exog, W)

        # Compare with R (allow higher tolerance for robust tests)
        r_stat = r_lm_results["robust_lm_lag_stat"]
        py_stat = lm_result.statistic

        assert np.isclose(
            py_stat, r_stat, rtol=0.20
        ), f"Robust LM-Lag stat: Python={py_stat:.4f} vs R={r_stat:.4f}"

        # Check p-value
        r_pval = r_lm_results["robust_lm_lag_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(
            py_pval, r_pval, rtol=0.25
        ), f"Robust LM-Lag p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"

    def test_robust_lm_error(self, spatial_test_data, r_lm_results):
        """Test that Robust LM-Error matches R within tolerance."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run Robust LM-Error test
        lm_result = robust_lm_error_test(ols_result.resid, ols_result.model.exog, W)

        # Compare with R
        r_stat = r_lm_results["robust_lm_error_stat"]
        py_stat = lm_result.statistic

        assert np.isclose(
            py_stat, r_stat, rtol=0.20
        ), f"Robust LM-Error stat: Python={py_stat:.4f} vs R={r_stat:.4f}"

        # Check p-value
        r_pval = r_lm_results["robust_lm_error_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(
            py_pval, r_pval, rtol=0.25
        ), f"Robust LM-Error p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"

    def test_run_lm_tests_complete(self, spatial_test_data):
        """Test that run_lm_tests() executes all tests and returns proper structure."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run all tests
        results = run_lm_tests(ols_result, W)

        # Check structure
        assert "lm_lag" in results
        assert "lm_error" in results
        assert "robust_lm_lag" in results
        assert "robust_lm_error" in results
        assert "recommendation" in results
        assert "reason" in results
        assert "summary" in results

        # Check that all results are LMTestResult objects
        assert isinstance(results["lm_lag"], LMTestResult)
        assert isinstance(results["lm_error"], LMTestResult)
        assert isinstance(results["robust_lm_lag"], LMTestResult)
        assert isinstance(results["robust_lm_error"], LMTestResult)

        # Check summary is DataFrame with correct structure
        assert isinstance(results["summary"], pd.DataFrame)
        assert len(results["summary"]) == 4
        assert list(results["summary"].columns) == [
            "Test",
            "Statistic",
            "p-value",
            "Significant",
        ]

        # Check recommendation is a string
        assert isinstance(results["recommendation"], str)
        assert isinstance(results["reason"], str)

    def test_decision_tree_logic(self, spatial_test_data):
        """Test that decision tree provides correct recommendations."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run all tests
        results = run_lm_tests(ols_result, W, alpha=0.05)

        # Since we generated data with spatial lag (rho=0.5),
        # we expect LM-Lag to be significant
        assert results["lm_lag"].pvalue < 0.05, "LM-Lag should be significant"

        # Check that recommendation makes sense
        # Should recommend SAR or mention lag/spatial dependence (including SDM which has lag)
        assert (
            "SAR" in results["recommendation"]
            or "lag" in results["recommendation"].lower()
            or "Durbin" in results["recommendation"]
            or "SDM" in results["recommendation"]
            or "GNS" in results["recommendation"]
        ), f"Unexpected recommendation: {results['recommendation']}"

    def test_lm_result_summary(self, spatial_test_data):
        """Test that LMTestResult.summary() returns formatted string."""
        df, W = spatial_test_data

        # Fit pooled OLS
        y, X = patsy.dmatrices("y ~ x1 + x2 + x3", data=df, return_type="dataframe")
        ols_result = OLS(y.values.flatten(), X.values).fit()

        # Run single test
        lm_result = lm_lag_test(ols_result.resid, ols_result.model.exog, W)

        # Get summary
        summary = lm_result.summary()

        # Check that summary contains key information
        assert "LM-Lag" in summary
        assert "Statistic:" in summary
        assert "P-value:" in summary
        assert "df:" in summary
        assert "Conclusion:" in summary


@pytest.mark.skipif(not HAS_STATSMODELS, reason="statsmodels not installed")
class TestLMTestsBasic:
    """Basic tests for LM functions without R comparison."""

    def test_lm_lag_returns_result(self):
        """Test that lm_lag_test returns LMTestResult."""
        # Simple synthetic data
        n = 50
        residuals = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Simple W matrix (identity, no spatial structure)
        W = np.eye(n)
        W = W / W.sum(axis=1, keepdims=True)

        result = lm_lag_test(residuals, X, W)

        assert isinstance(result, LMTestResult)
        assert result.test_name == "LM-Lag"
        assert result.df == 1
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

    def test_lm_error_returns_result(self):
        """Test that lm_error_test returns LMTestResult."""
        n = 50
        residuals = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        W = np.eye(n)
        W = W / W.sum(axis=1, keepdims=True)

        result = lm_error_test(residuals, X, W)

        assert isinstance(result, LMTestResult)
        assert result.test_name == "LM-Error"
        assert result.df == 1
        assert 0 <= result.pvalue <= 1
        assert result.statistic >= 0

    def test_robust_tests_return_results(self):
        """Test that robust LM tests return LMTestResult."""
        n = 50
        residuals = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        W = np.eye(n)
        W = W / W.sum(axis=1, keepdims=True)

        result_lag = robust_lm_lag_test(residuals, X, W)
        result_error = robust_lm_error_test(residuals, X, W)

        assert isinstance(result_lag, LMTestResult)
        assert isinstance(result_error, LMTestResult)
        assert result_lag.test_name == "Robust LM-Lag"
        assert result_error.test_name == "Robust LM-Error"

    def test_no_spatial_dependence(self):
        """Test with data having no spatial dependence."""
        n = 50
        # Random data with no spatial structure
        residuals = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, (n, 2))])

        # W matrix with no real neighbors (very weak connections)
        W = np.random.uniform(0, 0.01, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        result_lag = lm_lag_test(residuals, X, W)
        result_error = lm_error_test(residuals, X, W)

        # With random data and weak W, tests should typically not reject H0
        # (though this is probabilistic, so we just check they run)
        assert result_lag.pvalue is not None
        assert result_error.pvalue is not None
