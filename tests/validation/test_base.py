"""
Tests for validation base classes.
"""

import numpy as np
import pytest

from panelbox.validation.base import ValidationTest, ValidationTestResult


class TestValidationTestResult:
    """Test ValidationTestResult class."""

    def test_init_basic(self):
        """Test basic initialization."""
        result = ValidationTestResult(
            test_name="Test Name",
            statistic=5.0,
            pvalue=0.03,
            null_hypothesis="H0: No effect",
            alternative_hypothesis="H1: Effect exists",
        )

        assert result.test_name == "Test Name"
        assert result.statistic == 5.0
        assert result.pvalue == 0.03
        assert result.null_hypothesis == "H0: No effect"
        assert result.alternative_hypothesis == "H1: Effect exists"
        assert result.alpha == 0.05  # Default

    def test_init_with_alpha(self):
        """Test initialization with custom alpha."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.01,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=0.01,
        )

        assert result.alpha == 0.01

    def test_init_with_df(self):
        """Test initialization with degrees of freedom."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            df=10,
        )

        assert result.df == 10
        assert result.metadata["df"] == 10

    def test_init_with_metadata(self):
        """Test initialization with metadata."""
        metadata = {"method": "test_method", "value": 42}
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            metadata=metadata,
        )

        assert result.metadata["method"] == "test_method"
        assert result.metadata["value"] == 42

    def test_reject_null_true(self):
        """Test that null is rejected when pvalue < alpha."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=5.0,
            pvalue=0.01,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=0.05,
        )

        assert result.reject_null is True
        assert "Reject H0" in result.conclusion

    def test_reject_null_false(self):
        """Test that null is not rejected when pvalue >= alpha."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=1.0,
            pvalue=0.10,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=0.05,
        )

        assert result.reject_null is False
        assert "Fail to reject H0" in result.conclusion

    def test_boundary_case_pvalue_equals_alpha(self):
        """Test boundary case when pvalue equals alpha."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=2.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=0.05,
        )

        # pvalue = alpha should not reject
        assert result.reject_null is False

    def test_details_property(self):
        """Test that details property is alias for metadata."""
        metadata = {"key": "value"}
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            metadata=metadata,
        )

        assert result.details is result.metadata
        assert result.details["key"] == "value"

    def test_str_method(self):
        """Test string representation."""
        result = ValidationTestResult(
            test_name="Test Name",
            statistic=5.0,
            pvalue=0.01,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        result_str = str(result)
        assert isinstance(result_str, str)
        assert "TEST NAME" in result_str
        assert "5.0" in result_str or "5.000" in result_str

    def test_repr_method(self):
        """Test repr."""
        result = ValidationTestResult(
            test_name="Test Name",
            statistic=5.123,
            pvalue=0.0145,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        result_repr = repr(result)
        assert "Test Name" in result_repr
        assert "5.123" in result_repr
        assert "0.0145" in result_repr

    def test_summary_basic(self):
        """Test summary generation."""
        result = ValidationTestResult(
            test_name="Test Name",
            statistic=5.0,
            pvalue=0.01,
            null_hypothesis="H0: No effect",
            alternative_hypothesis="H1: Effect exists",
        )

        summary = result.summary()
        assert "TEST NAME" in summary
        assert "H0: No effect" in summary
        assert "H1: Effect exists" in summary
        assert "5.0" in summary or "5.000" in summary
        assert "0.01" in summary or "0.0100" in summary

    def test_summary_with_df_int(self):
        """Test summary with integer degrees of freedom."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            df=10,
        )

        summary = result.summary()
        assert "10" in summary
        assert "Degrees of Freedom" in summary

    def test_summary_with_df_tuple(self):
        """Test summary with tuple degrees of freedom."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            df=(2, 100),
        )

        summary = result.summary()
        assert "(2, 100)" in summary

    def test_summary_with_metadata(self):
        """Test summary includes metadata."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            metadata={"method": "test_method", "iterations": 1000},
        )

        summary = result.summary()
        assert "Additional Information" in summary
        assert "method" in summary
        assert "iterations" in summary

    def test_summary_with_float_metadata(self):
        """Test summary formats float metadata correctly."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            metadata={"value": 3.14159},
        )

        summary = result.summary()
        # Should format with 4 decimal places
        assert "3.1416" in summary

    def test_conclusion_reject(self):
        """Test conclusion message when rejecting null."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=5.0,
            pvalue=0.001,
            null_hypothesis="No autocorrelation",
            alternative_hypothesis="Autocorrelation present",
            alpha=0.05,
        )

        assert "Reject H0 at 5% level" in result.conclusion
        assert "Autocorrelation present" in result.conclusion

    def test_conclusion_fail_to_reject(self):
        """Test conclusion message when failing to reject null."""
        result = ValidationTestResult(
            test_name="Test",
            statistic=1.0,
            pvalue=0.50,
            null_hypothesis="No autocorrelation",
            alternative_hypothesis="Autocorrelation present",
            alpha=0.05,
        )

        assert "Fail to reject H0 at 5% level" in result.conclusion
        assert "No autocorrelation" in result.conclusion

    def test_alpha_different_levels(self):
        """Test conclusion with different alpha levels."""
        # Test at 1% level
        result_1 = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.02,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=0.01,
        )
        assert "1% level" in result_1.conclusion

        # Test at 10% level
        result_10 = ValidationTestResult(
            test_name="Test",
            statistic=2.0,
            pvalue=0.08,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=0.10,
        )
        assert "10% level" in result_10.conclusion


class MockResults:
    """Mock PanelResults for testing ValidationTest."""

    def __init__(self):
        self.resid = np.array([1.0, -0.5, 0.3, -0.2])
        self.fittedvalues = np.array([2.0, 1.5, 1.7, 2.2])
        self.params = np.array([1.0, 0.5])
        self.nobs = 100
        self.n_entities = 10
        self.n_periods = 10
        self.model_type = "Fixed Effects"


class ConcreteValidationTest(ValidationTest):
    """Concrete implementation for testing."""

    def run(self, alpha=0.05, **kwargs):
        """Dummy implementation."""
        return ValidationTestResult(
            test_name="Concrete Test",
            statistic=1.0,
            pvalue=0.5,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            alpha=alpha,
        )


class TestValidationTest:
    """Test ValidationTest abstract base class."""

    def test_init_stores_results(self):
        """Test that initialization stores results object."""
        mock_results = MockResults()
        test = ConcreteValidationTest(mock_results)

        assert test.results is mock_results

    def test_init_extracts_attributes(self):
        """Test that initialization extracts needed attributes."""
        mock_results = MockResults()
        test = ConcreteValidationTest(mock_results)

        assert np.array_equal(test.resid, mock_results.resid)
        assert np.array_equal(test.fittedvalues, mock_results.fittedvalues)
        assert np.array_equal(test.params, mock_results.params)
        assert test.nobs == mock_results.nobs
        assert test.n_entities == mock_results.n_entities
        assert test.n_periods == mock_results.n_periods
        assert test.model_type == mock_results.model_type

    def test_run_method_exists(self):
        """Test that run method exists and is callable."""
        mock_results = MockResults()
        test = ConcreteValidationTest(mock_results)

        result = test.run()
        assert isinstance(result, ValidationTestResult)

    def test_run_method_accepts_alpha(self):
        """Test that run method accepts alpha parameter."""
        mock_results = MockResults()
        test = ConcreteValidationTest(mock_results)

        result = test.run(alpha=0.01)
        assert result.alpha == 0.01

    def test_cannot_instantiate_abstract_class(self):
        """Test that ValidationTest cannot be instantiated directly."""
        mock_results = MockResults()

        with pytest.raises(TypeError):
            ValidationTest(mock_results)
