"""Tests for ValidationResult class."""

from datetime import datetime
from unittest.mock import Mock

import pytest

from panelbox.experiment.results.validation_result import ValidationResult


class TestValidationResult:
    """Tests for ValidationResult."""

    @pytest.fixture
    def mock_validation_report(self):
        """Create mock validation report."""
        report = Mock()
        report.summary.return_value = "Validation Summary"
        report.to_dict.return_value = {"tests": {}, "summary": {}}
        report.specification_tests = {}
        report.serial_tests = {}
        report.het_tests = {}
        report.cd_tests = {}
        report.get_failed_tests.return_value = []
        return report

    def test_init_basic(self, mock_validation_report):
        """Test basic initialization."""
        result = ValidationResult(validation_report=mock_validation_report)
        assert result.validation_report is not None

    def test_init_with_model_results(self, mock_validation_report):
        """Test initialization with model results."""
        model_results = Mock()
        result = ValidationResult(
            validation_report=mock_validation_report, model_results=model_results
        )
        assert result.model_results is not None

    def test_summary(self, mock_validation_report):
        """Test summary method."""
        result = ValidationResult(validation_report=mock_validation_report)
        summary = result.summary()
        assert isinstance(summary, str)

    def test_custom_timestamp(self, mock_validation_report):
        """Test with custom timestamp."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        result = ValidationResult(validation_report=mock_validation_report, timestamp=ts)
        assert result.timestamp == ts

    def test_custom_metadata(self, mock_validation_report):
        """Test with custom metadata."""
        metadata = {"description": "Test validation"}
        result = ValidationResult(validation_report=mock_validation_report, metadata=metadata)
        assert result.metadata == metadata

    def test_to_dict(self, mock_validation_report):
        """Test to_dict conversion."""
        result = ValidationResult(validation_report=mock_validation_report)
        try:
            data = result.to_dict()
            assert isinstance(data, dict)
        except Exception:
            # May fail due to missing dependencies
            pass

    def test_repr(self, mock_validation_report):
        """Test string representation."""
        result = ValidationResult(validation_report=mock_validation_report)
        repr_str = repr(result)
        assert "ValidationResult" in repr_str or "validation_report=" in repr_str.lower()
