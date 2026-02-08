"""
Tests for ValidationDataTransformer.

Tests the data transformation layer that converts ValidationReport objects
to chart-friendly format.
"""

import pytest
from unittest.mock import Mock

from panelbox.visualization.transformers.validation import ValidationDataTransformer


@pytest.fixture
def mock_test_result():
    """Mock test result object."""
    result = Mock()
    result.statistic = 15.234
    result.pvalue = 0.001
    result.df = 3
    result.conclusion = "Reject null hypothesis"
    result.reject_null = True
    result.alpha = 0.05
    result.metadata = {"test_type": "chi-square"}
    return result


@pytest.fixture
def mock_validation_report():
    """Mock ValidationReport object."""
    report = Mock()

    # Specification tests
    spec_result = Mock()
    spec_result.statistic = 12.5
    spec_result.pvalue = 0.006
    spec_result.df = 2
    spec_result.conclusion = "Use Fixed Effects"
    spec_result.reject_null = True
    spec_result.alpha = 0.05
    spec_result.metadata = {}

    report.specification_tests = {"Hausman Test": spec_result}

    # Serial correlation tests
    serial_result = Mock()
    serial_result.statistic = 3.4
    serial_result.pvalue = 0.045
    serial_result.df = 1
    serial_result.conclusion = "Serial correlation detected"
    serial_result.reject_null = True
    serial_result.alpha = 0.05
    serial_result.metadata = {}

    report.serial_tests = {"Wooldridge Test": serial_result}

    # Heteroskedasticity tests
    het_result = Mock()
    het_result.statistic = 2.1
    het_result.pvalue = 0.234
    het_result.df = 3
    het_result.conclusion = "No heteroskedasticity"
    het_result.reject_null = False
    het_result.alpha = 0.05
    het_result.metadata = {}

    report.het_tests = {"Breusch-Pagan": het_result}

    # Cross-sectional dependence tests
    cd_result = Mock()
    cd_result.statistic = 0.8
    cd_result.pvalue = 0.678
    cd_result.df = 1
    cd_result.conclusion = "No cross-sectional dependence"
    cd_result.reject_null = False
    cd_result.alpha = 0.05
    cd_result.metadata = {}

    report.cd_tests = {"Pesaran CD": cd_result}

    # Model info
    report.model_info = {
        "estimator": "FixedEffects",
        "nobs": 1000,
        "n_entities": 100,
        "n_periods": 10,
    }

    return report


class TestValidationDataTransformer:
    """Tests for ValidationDataTransformer."""

    def test_initialization(self):
        """Test transformer initialization."""
        transformer = ValidationDataTransformer()
        assert transformer is not None

    def test_transform_basic(self, mock_validation_report):
        """Test basic transformation."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        assert isinstance(result, dict)
        assert "tests" in result
        assert "categories" in result
        assert "summary" in result
        assert "model_info" in result

    def test_extract_all_tests(self, mock_validation_report):
        """Test extraction of all test types."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        tests = result["tests"]
        assert len(tests) == 4  # One from each category

        # Check test names
        test_names = [t["name"] for t in tests]
        assert "Hausman Test" in test_names
        assert "Wooldridge Test" in test_names
        assert "Breusch-Pagan" in test_names
        assert "Pesaran CD" in test_names

    def test_test_format(self, mock_validation_report):
        """Test format of individual test results."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        test = result["tests"][0]

        # Check required fields
        assert "name" in test
        assert "category" in test
        assert "statistic" in test
        assert "pvalue" in test
        assert "df" in test
        assert "conclusion" in test
        assert "passed" in test

    def test_category_assignment(self, mock_validation_report):
        """Test correct category assignment."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        # Find each test and check category
        tests_by_name = {t["name"]: t for t in result["tests"]}

        assert tests_by_name["Hausman Test"]["category"] == "Specification"
        assert tests_by_name["Wooldridge Test"]["category"] == "Serial Correlation"
        assert tests_by_name["Breusch-Pagan"]["category"] == "Heteroskedasticity"
        assert tests_by_name["Pesaran CD"]["category"] == "Cross-Sectional Dependence"

    def test_passed_field_logic(self, mock_validation_report):
        """Test passed field logic (inverted from reject_null)."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        tests_by_name = {t["name"]: t for t in result["tests"]}

        # reject_null=True means test failed (not passed)
        assert tests_by_name["Hausman Test"]["passed"] is False
        assert tests_by_name["Wooldridge Test"]["passed"] is False

        # reject_null=False means test passed
        assert tests_by_name["Breusch-Pagan"]["passed"] is True
        assert tests_by_name["Pesaran CD"]["passed"] is True

    def test_group_by_category(self, mock_validation_report):
        """Test grouping tests by category."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        categories = result["categories"]

        assert "Specification" in categories
        assert "Serial Correlation" in categories
        assert "Heteroskedasticity" in categories
        assert "Cross-Sectional Dependence" in categories

        # Check counts
        assert len(categories["Specification"]) == 1
        assert len(categories["Serial Correlation"]) == 1
        assert len(categories["Heteroskedasticity"]) == 1
        assert len(categories["Cross-Sectional Dependence"]) == 1

    def test_summary_computation(self, mock_validation_report):
        """Test summary statistics computation."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        summary = result["summary"]

        assert summary["total_tests"] == 4
        assert summary["passed"] == 2
        assert summary["failed"] == 2
        assert summary["pass_rate"] == 50.0

    def test_model_info_extraction(self, mock_validation_report):
        """Test model information extraction."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        model_info = result["model_info"]

        assert model_info["estimator"] == "FixedEffects"
        assert model_info["nobs"] == 1000
        assert model_info["n_entities"] == 100
        assert model_info["n_periods"] == 10

    def test_empty_validation_report(self):
        """Test with empty validation report."""
        report = Mock()
        report.specification_tests = {}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        result = transformer.transform(report)

        assert result["tests"] == []
        assert result["categories"] == {}
        assert result["summary"]["total_tests"] == 0
        assert result["summary"]["passed"] == 0
        assert result["summary"]["failed"] == 0
        assert result["summary"]["pass_rate"] == 0.0

    def test_partial_test_results(self):
        """Test with only some test categories."""
        report = Mock()

        # Only specification tests
        spec_result = Mock()
        spec_result.statistic = 10.0
        spec_result.pvalue = 0.01
        spec_result.df = 2
        spec_result.conclusion = "Test conclusion"
        spec_result.reject_null = True
        spec_result.alpha = 0.05
        spec_result.metadata = {}

        report.specification_tests = {"Test1": spec_result}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        result = transformer.transform(report)

        assert len(result["tests"]) == 1
        assert result["tests"][0]["category"] == "Specification"

    def test_metadata_handling(self, mock_test_result):
        """Test metadata field handling."""
        report = Mock()
        report.specification_tests = {"Test": mock_test_result}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        result = transformer.transform(report)

        test = result["tests"][0]
        assert "metadata" in test
        assert test["metadata"]["test_type"] == "chi-square"

    def test_alpha_field_handling(self, mock_validation_report):
        """Test alpha field is included."""
        transformer = ValidationDataTransformer()
        result = transformer.transform(mock_validation_report)

        # All tests should have alpha field
        for test in result["tests"]:
            assert "alpha" in test
            assert test["alpha"] == 0.05

    def test_missing_optional_attributes(self):
        """Test handling of test results missing optional attributes."""
        report = Mock()

        # Test result without metadata and alpha
        result = Mock()
        result.statistic = 5.0
        result.pvalue = 0.05
        result.df = 1
        result.conclusion = "Test"
        result.reject_null = False
        # No metadata or alpha attributes

        report.specification_tests = {"Test": result}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        transformed = transformer.transform(report)

        test = transformed["tests"][0]
        # Should not raise AttributeError
        assert test["name"] == "Test"
        assert test["passed"] is True


class TestValidationTransformerEdgeCases:
    """Edge case tests for ValidationDataTransformer."""

    def test_very_small_pvalues(self):
        """Test handling of very small p-values."""
        report = Mock()

        result = Mock()
        result.statistic = 100.0
        result.pvalue = 1e-10
        result.df = 5
        result.conclusion = "Highly significant"
        result.reject_null = True
        result.alpha = 0.05
        result.metadata = {}

        report.specification_tests = {"Test": result}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        transformed = transformer.transform(report)

        assert transformed["tests"][0]["pvalue"] == 1e-10

    def test_very_large_statistics(self):
        """Test handling of very large test statistics."""
        report = Mock()

        result = Mock()
        result.statistic = 999999.99
        result.pvalue = 0.0
        result.df = 10
        result.conclusion = "Extreme"
        result.reject_null = True
        result.alpha = 0.05
        result.metadata = {}

        report.specification_tests = {"Test": result}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        transformed = transformer.transform(report)

        assert transformed["tests"][0]["statistic"] == 999999.99

    def test_multiple_tests_same_category(self):
        """Test multiple tests in the same category."""
        report = Mock()

        result1 = Mock()
        result1.statistic = 5.0
        result1.pvalue = 0.05
        result1.df = 1
        result1.conclusion = "Test 1"
        result1.reject_null = False
        result1.alpha = 0.05
        result1.metadata = {}

        result2 = Mock()
        result2.statistic = 10.0
        result2.pvalue = 0.01
        result2.df = 2
        result2.conclusion = "Test 2"
        result2.reject_null = True
        result2.alpha = 0.05
        result2.metadata = {}

        report.specification_tests = {"Test1": result1, "Test2": result2}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.model_info = {}

        transformer = ValidationDataTransformer()
        transformed = transformer.transform(report)

        assert len(transformed["tests"]) == 2
        assert len(transformed["categories"]["Specification"]) == 2

        # Summary should reflect both tests
        assert transformed["summary"]["total_tests"] == 2
        assert transformed["summary"]["passed"] == 1
        assert transformed["summary"]["failed"] == 1
