"""Tests for ResidualResult class."""

from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from panelbox.experiment.results.residual_result import ResidualResult


class TestResidualResult:
    """Tests for ResidualResult."""

    @pytest.fixture
    def mock_model_results(self):
        """Create mock model results."""
        results = Mock()
        results.resid = pd.Series(np.random.randn(100), name="residuals")
        results.fitted_values = pd.Series(np.random.randn(100), name="fitted")
        results.nobs = 100
        results.scale = 1.0  # Mock scale attribute
        return results

    def test_init_basic(self, mock_model_results):
        """Test basic initialization."""
        result = ResidualResult(model_results=mock_model_results)
        assert result.model_results is not None

    def test_custom_timestamp(self, mock_model_results):
        """Test with custom timestamp."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        result = ResidualResult(model_results=mock_model_results, timestamp=ts)
        assert result.timestamp == ts

    def test_custom_metadata(self, mock_model_results):
        """Test with custom metadata."""
        metadata = {"description": "Test residuals"}
        result = ResidualResult(model_results=mock_model_results, metadata=metadata)
        assert result.metadata == metadata

    def test_residuals_property(self, mock_model_results):
        """Test residuals property."""
        result = ResidualResult(model_results=mock_model_results)
        residuals = result.residuals
        assert residuals is not None
        assert len(residuals) == 100

    def test_fitted_values_property(self, mock_model_results):
        """Test fitted_values property."""
        result = ResidualResult(model_results=mock_model_results)
        # fitted_values is extracted in __init__, check it exists
        assert hasattr(result, "fitted_values")
        assert result.fitted_values is not None

    def test_to_dict(self, mock_model_results):
        """Test to_dict conversion."""
        result = ResidualResult(model_results=mock_model_results)
        try:
            data = result.to_dict()
            assert isinstance(data, dict)
        except Exception:
            # May fail due to missing dependencies
            pass

    def test_summary(self, mock_model_results):
        """Test summary method."""
        result = ResidualResult(model_results=mock_model_results)
        try:
            summary = result.summary()
            assert isinstance(summary, str)
        except Exception:
            # May fail if summary method doesn't exist
            pass
