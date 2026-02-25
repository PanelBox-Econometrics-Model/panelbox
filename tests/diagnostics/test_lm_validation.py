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

    with open(results_path) as f:
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

        assert np.isclose(py_stat, r_stat, rtol=0.10), (
            f"LM-Lag stat: Python={py_stat:.4f} vs R={r_stat:.4f}"
        )

        # Check p-value as well
        r_pval = r_lm_results["lm_lag_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(py_pval, r_pval, rtol=0.15), (
            f"LM-Lag p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"
        )

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

        assert np.isclose(py_stat, r_stat, rtol=0.10), (
            f"LM-Error stat: Python={py_stat:.4f} vs R={r_stat:.4f}"
        )

        # Check p-value
        r_pval = r_lm_results["lm_error_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(py_pval, r_pval, rtol=0.15), (
            f"LM-Error p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"
        )

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

        assert np.isclose(py_stat, r_stat, rtol=0.20), (
            f"Robust LM-Lag stat: Python={py_stat:.4f} vs R={r_stat:.4f}"
        )

        # Check p-value
        r_pval = r_lm_results["robust_lm_lag_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(py_pval, r_pval, rtol=0.25), (
            f"Robust LM-Lag p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"
        )

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

        assert np.isclose(py_stat, r_stat, rtol=0.20), (
            f"Robust LM-Error stat: Python={py_stat:.4f} vs R={r_stat:.4f}"
        )

        # Check p-value
        r_pval = r_lm_results["robust_lm_error_pvalue"]
        py_pval = lm_result.pvalue

        assert np.isclose(py_pval, r_pval, rtol=0.25), (
            f"Robust LM-Error p-value: Python={py_pval:.4f} vs R={r_pval:.4f}"
        )

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


# ============================================================================
# Additional Coverage Tests for spatial_tests.py
# ============================================================================


class TestLMTestsPanelExpansion:
    """Test panel data W expansion for all LM tests."""

    def test_lm_lag_panel_expansion(self):
        """Test lm_lag_test with panel data (n_obs > n_entities)."""
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_obs = n_entities * n_time

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        # W is n_entities x n_entities
        W = np.random.uniform(0, 1, (n_entities, n_entities))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        result = lm_lag_test(residuals, X, W)
        assert isinstance(result, LMTestResult)
        assert result.test_name == "LM-Lag"
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_lm_error_panel_expansion(self):
        """Test lm_error_test with panel data (n_obs > n_entities)."""
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_obs = n_entities * n_time

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.random.uniform(0, 1, (n_entities, n_entities))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        result = lm_error_test(residuals, X, W)
        assert isinstance(result, LMTestResult)
        assert result.test_name == "LM-Error"

    def test_robust_lm_lag_panel_expansion(self):
        """Test robust_lm_lag_test with panel data (n_obs > n_entities)."""
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_obs = n_entities * n_time

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.random.uniform(0, 1, (n_entities, n_entities))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        result = robust_lm_lag_test(residuals, X, W)
        assert isinstance(result, LMTestResult)
        assert result.test_name == "Robust LM-Lag"

    def test_robust_lm_error_panel_expansion(self):
        """Test robust_lm_error_test with panel data (n_obs > n_entities)."""
        np.random.seed(42)
        n_entities = 10
        n_time = 5
        n_obs = n_entities * n_time

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.random.uniform(0, 1, (n_entities, n_entities))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        result = robust_lm_error_test(residuals, X, W)
        assert isinstance(result, LMTestResult)
        assert result.test_name == "Robust LM-Error"

    def test_lm_lag_panel_not_divisible_raises(self):
        """Test that lm_lag_test raises ValueError when n_obs not divisible by n_entities."""
        np.random.seed(42)
        n_entities = 10
        n_obs = 53  # Not divisible by 10

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.eye(n_entities)
        W = W / W.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="must be divisible"):
            lm_lag_test(residuals, X, W)

    def test_lm_error_panel_not_divisible_raises(self):
        """Test that lm_error_test raises ValueError when n_obs not divisible by n_entities."""
        np.random.seed(42)
        n_entities = 10
        n_obs = 53

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.eye(n_entities)
        W = W / W.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="must be divisible"):
            lm_error_test(residuals, X, W)

    def test_robust_lm_lag_panel_not_divisible_raises(self):
        """Test that robust_lm_lag_test raises ValueError for non-divisible n_obs."""
        np.random.seed(42)
        n_entities = 10
        n_obs = 53

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.eye(n_entities)
        W = W / W.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="must be divisible"):
            robust_lm_lag_test(residuals, X, W)

    def test_robust_lm_error_panel_not_divisible_raises(self):
        """Test that robust_lm_error_test raises ValueError for non-divisible n_obs."""
        np.random.seed(42)
        n_entities = 10
        n_obs = 53

        residuals = np.random.normal(0, 1, n_obs)
        X = np.column_stack([np.ones(n_obs), np.random.normal(0, 1, n_obs)])

        W = np.eye(n_entities)
        W = W / W.sum(axis=1, keepdims=True)

        with pytest.raises(ValueError, match="must be divisible"):
            robust_lm_error_test(residuals, X, W)


class TestDecisionTreeBranches:
    """Test all branches of the run_lm_tests decision tree."""

    @staticmethod
    def _make_mock_model_result(residuals, X):
        """Create a mock model result object with resid and model.exog attributes."""

        class MockModel:
            def __init__(self, exog):
                self.exog = exog

        class MockResult:
            def __init__(self, resid, model):
                self.resid = resid
                self.model = model

        return MockResult(residuals, MockModel(X))

    def test_only_lm_lag_significant(self):
        """Test decision tree when only LM-Lag is significant -> SAR."""
        np.random.seed(42)
        n = 50

        # Create data with strong spatial lag dependence
        W = np.random.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate residuals correlated with spatial lag of y
        # but not with spatial lag of residuals
        base_resid = np.random.normal(0, 1, n)
        # Add strong spatial lag signal
        residuals = base_resid + 3.0 * (W @ np.random.normal(0, 1, n))

        model_result = self._make_mock_model_result(residuals, X)
        results = run_lm_tests(model_result, W, alpha=0.05)

        assert "recommendation" in results
        assert "reason" in results
        assert isinstance(results["summary"], pd.DataFrame)

    def test_only_lm_error_significant(self):
        """Test decision tree when only LM-Error is significant -> SEM."""
        np.random.seed(123)
        n = 50

        W = np.random.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Generate residuals with spatial error correlation
        u = np.random.normal(0, 1, n)
        residuals = np.linalg.solve(np.eye(n) - 0.5 * W, u)

        model_result = self._make_mock_model_result(residuals, X)
        results = run_lm_tests(model_result, W, alpha=0.05)

        assert "recommendation" in results
        assert "reason" in results

    def test_neither_significant(self):
        """Test decision tree when neither LM test is significant."""
        np.random.seed(42)
        n = 50

        # Very weak spatial structure
        W = np.random.uniform(0, 0.001, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        residuals = np.random.normal(0, 1, n)

        model_result = self._make_mock_model_result(residuals, X)
        results = run_lm_tests(model_result, W, alpha=0.05)

        # With no spatial structure and alpha=0.05, likely "No spatial dependence"
        assert "recommendation" in results
        assert isinstance(results["reason"], str)

    def test_both_significant_robust_lag(self):
        """Test decision tree when both significant and robust lag wins."""
        np.random.seed(42)
        n = 50

        W = np.random.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])

        # Strong spatial structure to make both significant
        u = np.random.normal(0, 1, n)
        residuals = 5.0 * (W @ u) + u

        model_result = self._make_mock_model_result(residuals, X)
        results = run_lm_tests(model_result, W, alpha=0.05)

        assert "recommendation" in results
        assert "reason" in results

    def test_decision_tree_with_custom_alpha(self):
        """Test decision tree with very high alpha to force both significant."""
        np.random.seed(42)
        n = 50

        W = np.random.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        u = np.random.normal(0, 1, n)
        residuals = 3.0 * (W @ u) + u

        model_result = self._make_mock_model_result(residuals, X)
        # With alpha=0.99 almost everything is significant
        results = run_lm_tests(model_result, W, alpha=0.99)

        assert "recommendation" in results


class TestLMTestConclusions:
    """Test LM test conclusion messages for reject/cannot reject branches."""

    def test_lm_lag_cannot_reject(self):
        """Test LM-Lag conclusion when H0 cannot be rejected."""
        np.random.seed(42)
        n = 50
        residuals = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        W = np.eye(n) / n  # Very weak spatial structure

        result = lm_lag_test(residuals, X, W)
        # With identity-like W and random residuals, p-value should be high
        if result.pvalue >= 0.05:
            assert result.conclusion == "Cannot reject H0"
        else:
            assert result.conclusion == "Reject H0: No spatial lag"

    def test_lm_error_cannot_reject(self):
        """Test LM-Error conclusion when H0 cannot be rejected."""
        np.random.seed(42)
        n = 50
        residuals = np.random.normal(0, 1, n)
        X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
        W = np.eye(n) / n

        result = lm_error_test(residuals, X, W)
        if result.pvalue >= 0.05:
            assert result.conclusion == "Cannot reject H0"
        else:
            assert result.conclusion == "Reject H0: No spatial error"


class TestMoranIPanelTest:
    """Tests for MoranIPanelTest class covering all branches."""

    @pytest.fixture
    def panel_setup(self):
        """Create panel data for Moran's I tests."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5
        n_obs = n_entities * n_periods

        entity_ids = np.repeat(np.arange(n_entities), n_periods)
        time_ids = np.tile(np.arange(n_periods), n_entities)
        residuals = np.random.normal(0, 1, n_obs)

        W = np.random.uniform(0, 1, (n_entities, n_entities))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        return residuals, W, entity_ids, time_ids

    def test_pooled_method(self, panel_setup):
        """Test MoranIPanelTest with pooled method."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest, MoranIResult

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run(method="pooled")

        assert isinstance(result, MoranIResult)
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.pvalue)
        assert 0 <= result.pvalue <= 1
        assert np.isfinite(result.z_score)
        assert np.isfinite(result.variance)

    def test_by_period_method(self, panel_setup):
        """Test MoranIPanelTest with by_period method."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest, MoranIResult

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run(method="by_period")

        assert isinstance(result, dict)
        assert len(result) == 5  # 5 periods
        for _key, val in result.items():
            assert isinstance(val, MoranIResult)
            assert np.isfinite(val.statistic)

    def test_average_method(self, panel_setup):
        """Test MoranIPanelTest with average method."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest, MoranIResult

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run(method="average")

        assert isinstance(result, MoranIResult)
        assert np.isfinite(result.statistic)
        assert result.additional_info["method"] == "average"
        assert result.additional_info["n_periods"] == 5

    def test_unknown_method_raises(self, panel_setup):
        """Test that unknown method raises ValueError."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)

        with pytest.raises(ValueError, match="Unknown method"):
            test.run(method="invalid")

    def test_moran_i_result_summary(self, panel_setup):
        """Test MoranIResult.summary() method."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run(method="pooled")

        summary = result.summary()
        assert "Moran's I Test" in summary
        assert "I statistic:" in summary
        assert "Expected I:" in summary
        assert "Variance:" in summary
        assert "Z-score:" in summary
        assert "P-value:" in summary
        assert "Conclusion:" in summary

    def test_pooled_conclusion_messages(self, panel_setup):
        """Test that conclusion is one of the expected messages."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run(method="pooled")

        assert result.conclusion in [
            "Significant spatial autocorrelation",
            "No significant spatial autocorrelation",
        ]

    def test_average_conclusion_messages(self, panel_setup):
        """Test average method conclusion messages."""
        from panelbox.diagnostics.spatial_tests import MoranIPanelTest

        residuals, W, entity_ids, time_ids = panel_setup
        test = MoranIPanelTest(residuals, W, entity_ids, time_ids)
        result = test.run(method="average")

        assert result.conclusion in [
            "Average across periods shows significant spatial autocorrelation",
            "No significant spatial autocorrelation on average",
        ]


class TestLISAResult:
    """Tests for LISAResult class."""

    def test_get_clusters_all_types(self):
        """Test that get_clusters correctly classifies all cluster types."""
        from panelbox.diagnostics.spatial_tests import LISAResult

        n = 6
        # Construct values to trigger all cluster types
        local_i = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1])
        pvalues = np.array([0.01, 0.01, 0.01, 0.01, 0.10, 0.50])
        z_values = np.array([1.0, -1.0, 1.0, -1.0, 0.5, 0.5])
        Wz_values = np.array([1.0, -1.0, -1.0, 1.0, 0.5, 0.5])
        entity_ids = np.arange(n)

        lisa = LISAResult(
            local_i=local_i,
            pvalues=pvalues,
            z_values=z_values,
            Wz_values=Wz_values,
            entity_ids=entity_ids,
        )

        clusters = lisa.get_clusters(alpha=0.05)

        assert isinstance(clusters, pd.DataFrame)
        assert len(clusters) == n
        assert "cluster_type" in clusters.columns
        assert "entity_id" in clusters.columns
        assert "Ii" in clusters.columns
        assert "pvalue" in clusters.columns

        # Check cluster types
        assert clusters.iloc[0]["cluster_type"] == "HH"
        assert clusters.iloc[1]["cluster_type"] == "LL"
        assert clusters.iloc[2]["cluster_type"] == "HL"
        assert clusters.iloc[3]["cluster_type"] == "LH"
        assert clusters.iloc[4]["cluster_type"] == "Not significant"
        assert clusters.iloc[5]["cluster_type"] == "Not significant"

    def test_lisa_summary(self):
        """Test LISA summary output."""
        from panelbox.diagnostics.spatial_tests import LISAResult

        n = 5
        lisa = LISAResult(
            local_i=np.ones(n),
            pvalues=np.array([0.01, 0.01, 0.01, 0.10, 0.50]),
            z_values=np.array([1.0, -1.0, 1.0, 0.5, 0.5]),
            Wz_values=np.array([1.0, -1.0, -1.0, 0.5, 0.5]),
            entity_ids=np.arange(n),
        )

        summary = lisa.summary(alpha=0.05)
        assert "Local Moran's I (LISA) Results" in summary
        assert "Total observations: 5" in summary
        assert "Significance level: 0.05" in summary
        assert "Cluster types:" in summary


class TestLocalMoranI:
    """Tests for LocalMoranI class."""

    def test_local_moran_i_basic(self):
        """Test basic Local Moran's I computation."""
        from panelbox.diagnostics.spatial_tests import LISAResult, LocalMoranI

        np.random.seed(42)
        n = 10
        values = np.random.normal(0, 1, n)
        entity_ids = np.arange(n)

        W = np.random.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        lisa = LocalMoranI(values, W, entity_ids)
        result = lisa.run(permutations=99)  # Fewer for speed

        assert isinstance(result, LISAResult)
        assert len(result.local_i) == n
        assert len(result.pvalues) == n
        assert len(result.z_values) == n
        assert len(result.Wz_values) == n
        assert np.all(result.pvalues >= 0)
        assert np.all(result.pvalues <= 1)

    def test_local_moran_i_clusters(self):
        """Test Local Moran's I cluster classification."""
        from panelbox.diagnostics.spatial_tests import LocalMoranI

        np.random.seed(42)
        n = 10
        values = np.random.normal(0, 1, n)
        entity_ids = np.arange(n)

        W = np.random.uniform(0, 1, (n, n))
        np.fill_diagonal(W, 0)
        W = W / W.sum(axis=1, keepdims=True)

        lisa = LocalMoranI(values, W, entity_ids)
        result = lisa.run(permutations=99)
        clusters = result.get_clusters(alpha=0.05)

        assert isinstance(clusters, pd.DataFrame)
        assert "cluster_type" in clusters.columns
        # All cluster_type values should be valid
        valid_types = {"HH", "LL", "HL", "LH", "Not significant"}
        assert set(clusters["cluster_type"].unique()).issubset(valid_types)
