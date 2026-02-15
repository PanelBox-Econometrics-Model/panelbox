"""
Tests for Encompassing Tests.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS

from panelbox.diagnostics.specification import (
    EncompassingResult,
    cox_test,
    likelihood_ratio_test,
    wald_encompassing_test,
)


class TestCoxTest:
    """Tests for Cox test."""

    @pytest.fixture
    def logit_data(self):
        """Create data for logit models."""
        np.random.seed(42)
        n = 500

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # True model
        z = -1 + 2 * x1 + 1.5 * x2
        p = 1 / (1 + np.exp(-z))
        y = (np.random.rand(n) < p).astype(int)

        return {"y": y, "x1": x1, "x2": x2, "x3": x3, "n": n}

    def test_cox_test_basic(self, logit_data):
        """Test basic Cox test functionality."""
        # Model 1: correct specification
        X1 = np.column_stack([np.ones(logit_data["n"]), logit_data["x1"], logit_data["x2"]])
        model1 = Logit(logit_data["y"], X1)
        result1 = model1.fit(disp=0)

        # Model 2: includes irrelevant variable
        X2 = np.column_stack([np.ones(logit_data["n"]), logit_data["x1"], logit_data["x3"]])
        model2 = Logit(logit_data["y"], X2)
        result2 = model2.fit(disp=0)

        # Perform Cox test
        cox_result = cox_test(result1, result2, model1_name="Correct", model2_name="With x3")

        # Check result structure
        assert isinstance(cox_result, EncompassingResult)
        assert cox_result.test_name == "Cox Test for Non-nested Models"
        assert cox_result.statistic is not None
        assert 0 <= cox_result.pvalue <= 1
        assert cox_result.df is None  # Cox test doesn't use df

    def test_cox_test_interpretation(self, logit_data):
        """Test interpretation method."""
        X1 = np.column_stack([np.ones(logit_data["n"]), logit_data["x1"]])
        model1 = Logit(logit_data["y"], X1)
        result1 = model1.fit(disp=0)

        X2 = np.column_stack([np.ones(logit_data["n"]), logit_data["x1"], logit_data["x2"]])
        model2 = Logit(logit_data["y"], X2)
        result2 = model2.fit(disp=0)

        cox_result = cox_test(result1, result2)
        interpretation = cox_result.interpretation()

        # Check it returns a string with key information
        assert isinstance(interpretation, str)
        assert "Cox Test" in interpretation
        assert "p-value" in interpretation.lower()

    def test_cox_test_summary(self, logit_data):
        """Test summary table generation."""
        X1 = np.column_stack([np.ones(logit_data["n"]), logit_data["x1"]])
        model1 = Logit(logit_data["y"], X1)
        result1 = model1.fit(disp=0)

        X2 = np.column_stack([np.ones(logit_data["n"]), logit_data["x2"]])
        model2 = Logit(logit_data["y"], X2)
        result2 = model2.fit(disp=0)

        cox_result = cox_test(result1, result2)
        summary = cox_result.summary()

        # Check summary is a DataFrame
        assert isinstance(summary, pd.DataFrame)
        assert "Test" in summary.columns
        assert "Statistic" in summary.columns
        assert "p-value" in summary.columns

    def test_cox_test_no_likelihood(self):
        """Test error when models don't have log-likelihood."""

        # Create mock objects without .llf
        class MockResult:
            pass

        result1 = MockResult()
        result2 = MockResult()

        with pytest.raises(AttributeError, match="log-likelihood"):
            cox_test(result1, result2)


class TestWaldEncompassingTest:
    """Tests for Wald encompassing test."""

    @pytest.fixture
    def nested_ols_data(self):
        """Create data for nested OLS models."""
        np.random.seed(123)
        n = 200

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # True model: y = 1 + 2*x1 + 3*x2 + ε
        y = 1 + 2 * x1 + 3 * x2 + np.random.randn(n)

        return {"y": y, "x1": x1, "x2": x2, "x3": x3, "n": n}

    def test_wald_test_rejects_when_restriction_invalid(self, nested_ols_data):
        """Test that Wald test rejects when restricted model is inadequate."""
        # Restricted: y ~ x1 (missing important x2)
        X_r = np.column_stack([np.ones(nested_ols_data["n"]), nested_ols_data["x1"]])
        result_r = OLS(nested_ols_data["y"], X_r).fit()

        # Unrestricted: y ~ x1 + x2 (correct)
        X_u = np.column_stack(
            [np.ones(nested_ols_data["n"]), nested_ols_data["x1"], nested_ols_data["x2"]]
        )
        result_u = OLS(nested_ols_data["y"], X_u).fit()

        # Perform Wald test
        wald_result = wald_encompassing_test(result_r, result_u)

        # Should reject (x2 is important)
        assert wald_result.pvalue < 0.05
        assert wald_result.df == 1  # one restriction

    def test_wald_test_fails_to_reject_when_restriction_valid(self, nested_ols_data):
        """Test that Wald test doesn't reject when restriction is valid."""
        # Restricted: y ~ x1 + x2 (correct)
        X_r = np.column_stack(
            [np.ones(nested_ols_data["n"]), nested_ols_data["x1"], nested_ols_data["x2"]]
        )
        result_r = OLS(nested_ols_data["y"], X_r).fit()

        # Unrestricted: y ~ x1 + x2 + x3 (x3 is irrelevant)
        X_u = np.column_stack(
            [
                np.ones(nested_ols_data["n"]),
                nested_ols_data["x1"],
                nested_ols_data["x2"],
                nested_ols_data["x3"],
            ]
        )
        result_u = OLS(nested_ols_data["y"], X_u).fit()

        # Perform Wald test
        wald_result = wald_encompassing_test(result_r, result_u)

        # Should not reject (x3 is not important)
        assert wald_result.pvalue > 0.05
        assert wald_result.df == 1

    def test_wald_test_multiple_restrictions(self, nested_ols_data):
        """Test Wald test with multiple restrictions."""
        # Restricted: y ~ constant only
        X_r = np.ones((nested_ols_data["n"], 1))
        result_r = OLS(nested_ols_data["y"], X_r).fit()

        # Unrestricted: y ~ x1 + x2
        X_u = np.column_stack(
            [np.ones(nested_ols_data["n"]), nested_ols_data["x1"], nested_ols_data["x2"]]
        )
        result_u = OLS(nested_ols_data["y"], X_u).fit()

        # Perform Wald test
        wald_result = wald_encompassing_test(result_r, result_u)

        # Should have df = 2
        assert wald_result.df == 2
        # Should reject (both x1 and x2 are important)
        assert wald_result.pvalue < 0.05

    def test_wald_test_error_when_not_nested(self, nested_ols_data):
        """Test error when unrestricted doesn't have more parameters."""
        X1 = np.column_stack(
            [np.ones(nested_ols_data["n"]), nested_ols_data["x1"], nested_ols_data["x2"]]
        )
        result1 = OLS(nested_ols_data["y"], X1).fit()

        X2 = np.column_stack([np.ones(nested_ols_data["n"]), nested_ols_data["x1"]])
        result2 = OLS(nested_ols_data["y"], X2).fit()

        # result2 has fewer parameters - should raise error
        with pytest.raises(ValueError, match="more parameters"):
            wald_encompassing_test(result1, result2)


class TestLikelihoodRatioTest:
    """Tests for likelihood ratio test."""

    @pytest.fixture
    def logit_nested_data(self):
        """Create data for nested logit models."""
        np.random.seed(456)
        n = 400

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # True model
        z = -0.5 + 1.5 * x1 + 2 * x2
        p = 1 / (1 + np.exp(-z))
        y = (np.random.rand(n) < p).astype(int)

        return {"y": y, "x1": x1, "x2": x2, "x3": x3, "n": n}

    def test_lr_test_rejects_invalid_restriction(self, logit_nested_data):
        """Test that LR test rejects when restricted model is inadequate."""
        # Restricted: y ~ x1 (missing x2)
        X_r = np.column_stack([np.ones(logit_nested_data["n"]), logit_nested_data["x1"]])
        result_r = Logit(logit_nested_data["y"], X_r).fit(disp=0)

        # Unrestricted: y ~ x1 + x2 (correct)
        X_u = np.column_stack(
            [np.ones(logit_nested_data["n"]), logit_nested_data["x1"], logit_nested_data["x2"]]
        )
        result_u = Logit(logit_nested_data["y"], X_u).fit(disp=0)

        # Perform LR test
        lr_result = likelihood_ratio_test(result_r, result_u)

        # Should reject
        assert lr_result.pvalue < 0.05
        assert lr_result.df == 1

        # Check additional info
        assert "llf_restricted" in lr_result.additional_info
        assert "llf_unrestricted" in lr_result.additional_info
        assert (
            lr_result.additional_info["llf_unrestricted"]
            > lr_result.additional_info["llf_restricted"]
        )

    def test_lr_test_fails_to_reject_valid_restriction(self, logit_nested_data):
        """Test that LR test doesn't reject when restriction is valid."""
        # Restricted: y ~ x1 + x2 (correct)
        X_r = np.column_stack(
            [np.ones(logit_nested_data["n"]), logit_nested_data["x1"], logit_nested_data["x2"]]
        )
        result_r = Logit(logit_nested_data["y"], X_r).fit(disp=0)

        # Unrestricted: y ~ x1 + x2 + x3 (x3 irrelevant)
        X_u = np.column_stack(
            [
                np.ones(logit_nested_data["n"]),
                logit_nested_data["x1"],
                logit_nested_data["x2"],
                logit_nested_data["x3"],
            ]
        )
        result_u = Logit(logit_nested_data["y"], X_u).fit(disp=0)

        # Perform LR test
        lr_result = likelihood_ratio_test(result_r, result_u)

        # Should not reject
        assert lr_result.pvalue > 0.05
        assert lr_result.df == 1

    def test_lr_test_interpretation(self, logit_nested_data):
        """Test interpretation method."""
        X_r = np.column_stack([np.ones(logit_nested_data["n"]), logit_nested_data["x1"]])
        result_r = Logit(logit_nested_data["y"], X_r).fit(disp=0)

        X_u = np.column_stack(
            [np.ones(logit_nested_data["n"]), logit_nested_data["x1"], logit_nested_data["x2"]]
        )
        result_u = Logit(logit_nested_data["y"], X_u).fit(disp=0)

        lr_result = likelihood_ratio_test(result_r, result_u)
        interpretation = lr_result.interpretation()

        # Check interpretation contains key information
        assert isinstance(interpretation, str)
        assert "Likelihood Ratio Test" in interpretation
        assert "p-value" in interpretation.lower()

    def test_lr_test_no_likelihood_error(self):
        """Test error when models don't have log-likelihood."""
        np.random.seed(42)
        n = 100
        y = np.random.randn(n)
        X1 = np.column_stack([np.ones(n), np.random.randn(n)])
        X2 = np.column_stack([np.ones(n), np.random.randn(n), np.random.randn(n)])

        result1 = OLS(y, X1).fit()
        result2 = OLS(y, X2).fit()

        # OLS has .llf, so this should work
        # But let's test with mock objects without .llf
        class MockResult:
            def __init__(self, params):
                self.params = params

        mock1 = MockResult(np.array([1, 2]))
        mock2 = MockResult(np.array([1, 2, 3]))

        with pytest.raises(AttributeError, match="log-likelihood"):
            likelihood_ratio_test(mock1, mock2)

    def test_lr_test_summary(self, logit_nested_data):
        """Test summary table generation."""
        X_r = np.column_stack([np.ones(logit_nested_data["n"]), logit_nested_data["x1"]])
        result_r = Logit(logit_nested_data["y"], X_r).fit(disp=0)

        X_u = np.column_stack(
            [np.ones(logit_nested_data["n"]), logit_nested_data["x1"], logit_nested_data["x2"]]
        )
        result_u = Logit(logit_nested_data["y"], X_u).fit(disp=0)

        lr_result = likelihood_ratio_test(result_r, result_u)
        summary = lr_result.summary()

        # Check summary structure
        assert isinstance(summary, pd.DataFrame)
        assert "Test" in summary.columns
        assert "Statistic" in summary.columns
        assert "p-value" in summary.columns
        assert "df" in summary.columns


class TestEncompassingResult:
    """Tests for EncompassingResult class."""

    def test_encompassing_result_repr(self):
        """Test string representation."""
        result = EncompassingResult(
            test_name="Test Name",
            statistic=5.2,
            pvalue=0.023,
            df=2,
            null_hypothesis="H0 is true",
            alternative="H1 is true",
            model1_name="Model A",
            model2_name="Model B",
            additional_info={},
        )

        repr_str = repr(result)
        assert "Test Name" in repr_str
        assert "0.023" in repr_str

    def test_interpretation_reject(self):
        """Test interpretation when rejecting H0."""
        result = EncompassingResult(
            test_name="Test",
            statistic=10.5,
            pvalue=0.001,
            df=1,
            null_hypothesis="Models are equal",
            alternative="Models are different",
            model1_name="M1",
            model2_name="M2",
            additional_info={},
        )

        interp = result.interpretation(alpha=0.05)
        assert "REJECT H0" in interp
        assert "Models are different" in interp

    def test_interpretation_fail_to_reject(self):
        """Test interpretation when failing to reject H0."""
        result = EncompassingResult(
            test_name="Test",
            statistic=2.1,
            pvalue=0.15,
            df=1,
            null_hypothesis="Models are equal",
            alternative="Models are different",
            model1_name="M1",
            model2_name="M2",
            additional_info={},
        )

        interp = result.interpretation(alpha=0.05)
        assert "FAIL TO REJECT H0" in interp
        assert "Models are equal" in interp
