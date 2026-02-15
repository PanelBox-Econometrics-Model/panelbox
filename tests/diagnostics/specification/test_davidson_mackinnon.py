"""
Tests for Davidson-MacKinnon J-Test.
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_less
from statsmodels.regression.linear_model import OLS

from panelbox.diagnostics.specification import JTestResult, j_test


class TestJTest:
    """Tests for j_test function."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data with known DGP."""
        np.random.seed(42)
        n = 200

        # True model: y = 1 + 2*x1 + 3*x2 + ε
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)  # irrelevant variable

        y = 1 + 2 * x1 + 3 * x2 + np.random.randn(n)

        return {"y": y, "x1": x1, "x2": x2, "x3": x3, "n": n}

    def test_j_test_correct_model_preferred(self, simple_data):
        """Test that J-test identifies the correct model."""
        # Model 1 (correct): y ~ x1 + x2
        X1 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x2"]])
        model1 = OLS(simple_data["y"], X1)
        result1 = model1.fit()

        # Model 2 (incorrect): y ~ x1 + x3 (missing x2, includes irrelevant x3)
        X2 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x3"]])
        model2 = OLS(simple_data["y"], X2)
        result2 = model2.fit()

        # Perform J-test
        jtest = j_test(
            result1, result2, direction="both", model1_name="Correct", model2_name="Incorrect"
        )

        # Forward test: Model 2's fitted values should NOT improve Model 1
        # (because Model 1 is correct)
        assert (
            jtest.forward["pvalue"] > 0.05
        ), "Forward test should not reject when Model 1 is correct"

        # Reverse test: Model 1's fitted values SHOULD improve Model 2
        # (because Model 2 is missing important variable)
        assert (
            jtest.reverse["pvalue"] < 0.05
        ), "Reverse test should reject when Model 2 is incorrect"

    def test_j_test_forward_only(self, simple_data):
        """Test forward direction only."""
        X1 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"]])
        model1 = OLS(simple_data["y"], X1)
        result1 = model1.fit()

        X2 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x2"]])
        model2 = OLS(simple_data["y"], X2)
        result2 = model2.fit()

        jtest = j_test(result1, result2, direction="forward")

        # Should have forward results
        assert jtest.forward is not None
        assert "statistic" in jtest.forward
        assert "pvalue" in jtest.forward

        # Should not have reverse results
        assert jtest.reverse is None

    def test_j_test_reverse_only(self, simple_data):
        """Test reverse direction only."""
        X1 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"]])
        model1 = OLS(simple_data["y"], X1)
        result1 = model1.fit()

        X2 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x2"]])
        model2 = OLS(simple_data["y"], X2)
        result2 = model2.fit()

        jtest = j_test(result1, result2, direction="reverse")

        # Should have reverse results
        assert jtest.reverse is not None
        assert "statistic" in jtest.reverse
        assert "pvalue" in jtest.reverse

        # Should not have forward results
        assert jtest.forward is None

    def test_j_test_both_models_reject(self):
        """Test case where both models are inadequate."""
        np.random.seed(123)
        n = 200

        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)

        # True model includes all three
        y = 1 + 2 * x1 + 3 * x2 + 4 * x3 + np.random.randn(n)

        # Model 1: only x1, x2
        X1 = np.column_stack([np.ones(n), x1, x2])
        model1 = OLS(y, X1)
        result1 = model1.fit()

        # Model 2: only x1, x3
        X2 = np.column_stack([np.ones(n), x1, x3])
        model2 = OLS(y, X2)
        result2 = model2.fit()

        jtest = j_test(result1, result2, direction="both")

        # Both should reject (both models missing important variable)
        assert (
            jtest.forward["pvalue"] < 0.10
        ), "Forward test should reject when Model 1 is inadequate"
        assert (
            jtest.reverse["pvalue"] < 0.10
        ), "Reverse test should reject when Model 2 is inadequate"

    def test_j_test_result_summary(self, simple_data):
        """Test summary table generation."""
        X1 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"]])
        model1 = OLS(simple_data["y"], X1)
        result1 = model1.fit()

        X2 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x2"]])
        model2 = OLS(simple_data["y"], X2)
        result2 = model2.fit()

        jtest = j_test(result1, result2, direction="both")
        summary = jtest.summary()

        # Check summary is a DataFrame
        assert isinstance(summary, pd.DataFrame)

        # Check has both tests
        assert len(summary) == 2

        # Check columns
        assert "Test" in summary.columns
        assert "Test Statistic" in summary.columns
        assert "p-value" in summary.columns

    def test_j_test_interpretation(self, simple_data):
        """Test interpretation method."""
        X1 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x2"]])
        model1 = OLS(simple_data["y"], X1)
        result1 = model1.fit()

        X2 = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"], simple_data["x3"]])
        model2 = OLS(simple_data["y"], X2)
        result2 = model2.fit()

        jtest = j_test(result1, result2, direction="both", model1_name="M1", model2_name="M2")
        interpretation = jtest.interpretation()

        # Check it returns a string
        assert isinstance(interpretation, str)

        # Check it mentions both models
        assert "M1" in interpretation
        assert "M2" in interpretation

    def test_j_test_invalid_inputs(self, simple_data):
        """Test error handling for invalid inputs."""
        X = np.column_stack([np.ones(simple_data["n"]), simple_data["x1"]])
        model = OLS(simple_data["y"], X)
        result = model.fit()

        # Test with non-result object
        with pytest.raises(AttributeError):
            j_test(result, "not a result")

        with pytest.raises(AttributeError):
            j_test("not a result", result)

    def test_j_test_different_sample_sizes(self):
        """Test error when models have different sample sizes."""
        np.random.seed(42)

        # Model 1 with n=100
        n1 = 100
        y1 = np.random.randn(n1)
        X1 = np.column_stack([np.ones(n1), np.random.randn(n1)])
        model1 = OLS(y1, X1)
        result1 = model1.fit()

        # Model 2 with n=150
        n2 = 150
        y2 = np.random.randn(n2)
        X2 = np.column_stack([np.ones(n2), np.random.randn(n2)])
        model2 = OLS(y2, X2)
        result2 = model2.fit()

        # Should raise error
        with pytest.raises(ValueError, match="different sample sizes"):
            j_test(result1, result2)

    def test_nested_models_warning(self):
        """
        Test that J-test still works for nested models.

        Note: J-test is designed for non-nested models, but should still
        produce results for nested models (though interpretation differs).
        """
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.randn(n)

        # Model 1: restricted (only x1)
        X1 = np.column_stack([np.ones(n), x1])
        model1 = OLS(y, X1)
        result1 = model1.fit()

        # Model 2: unrestricted (x1 and x2)
        X2 = np.column_stack([np.ones(n), x1, x2])
        model2 = OLS(y, X2)
        result2 = model2.fit()

        # Should work without error
        jtest = j_test(result1, result2, direction="both")

        # For nested models where Model 2 nests Model 1:
        # - Forward test should reject (Model 2 improves on Model 1)
        assert jtest.forward["pvalue"] < 0.05

        # Note: The reverse test behavior can vary for nested models
        # The key is that the test completes without error
        assert jtest.reverse is not None


class TestJTestResult:
    """Tests for JTestResult class."""

    def test_jtest_result_repr(self):
        """Test string representation."""
        result = JTestResult(
            forward={"statistic": 2.5, "pvalue": 0.01, "alpha_coef": 0.5, "alpha_se": 0.2},
            reverse={"statistic": 1.2, "pvalue": 0.23, "gamma_coef": 0.3, "gamma_se": 0.25},
            model1_name="Model A",
            model2_name="Model B",
            direction="both",
        )

        repr_str = repr(result)
        assert "JTestResult" in repr_str
        assert "Model A" in repr_str
        assert "Model B" in repr_str

    def test_forward_interpretation(self):
        """Test interpretation for forward test only."""
        # Reject H0 (favor Model 2)
        result = JTestResult(
            forward={"statistic": 3.0, "pvalue": 0.001, "alpha_coef": 0.5, "alpha_se": 0.15},
            reverse=None,
            model1_name="M1",
            model2_name="M2",
            direction="forward",
        )
        interp = result.interpretation()
        assert "REJECTS H0" in interp
        assert "M2" in interp

        # Fail to reject H0 (Model 1 adequate)
        result2 = JTestResult(
            forward={"statistic": 0.5, "pvalue": 0.6, "alpha_coef": 0.1, "alpha_se": 0.2},
            reverse=None,
            model1_name="M1",
            model2_name="M2",
            direction="forward",
        )
        interp2 = result2.interpretation()
        assert "FAILS TO REJECT" in interp2
        assert "M1" in interp2

    def test_both_interpretation_prefer_model1(self):
        """Test interpretation when Model 1 is preferred."""
        result = JTestResult(
            forward={"statistic": 1.0, "pvalue": 0.32, "alpha_coef": 0.2, "alpha_se": 0.2},
            reverse={"statistic": 3.5, "pvalue": 0.001, "gamma_coef": 0.7, "gamma_se": 0.2},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "PREFER M1" in interp

    def test_both_interpretation_prefer_model2(self):
        """Test interpretation when Model 2 is preferred."""
        result = JTestResult(
            forward={"statistic": 4.0, "pvalue": 0.0001, "alpha_coef": 0.8, "alpha_se": 0.2},
            reverse={"statistic": 0.8, "pvalue": 0.42, "gamma_coef": 0.15, "gamma_se": 0.2},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "PREFER M2" in interp

    def test_both_interpretation_both_rejected(self):
        """Test interpretation when both models rejected."""
        result = JTestResult(
            forward={"statistic": 3.0, "pvalue": 0.003, "alpha_coef": 0.6, "alpha_se": 0.2},
            reverse={"statistic": 2.8, "pvalue": 0.005, "gamma_coef": 0.55, "gamma_se": 0.2},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "BOTH MODELS REJECTED" in interp

    def test_both_interpretation_both_acceptable(self):
        """Test interpretation when both models acceptable."""
        result = JTestResult(
            forward={"statistic": 1.0, "pvalue": 0.32, "alpha_coef": 0.2, "alpha_se": 0.2},
            reverse={"statistic": 0.9, "pvalue": 0.37, "gamma_coef": 0.18, "gamma_se": 0.2},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "BOTH MODELS ACCEPTABLE" in interp
