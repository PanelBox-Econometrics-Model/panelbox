"""
Tests for ResidualDataTransformer.

Tests the data transformation layer that converts model results
to residual diagnostic chart format.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from panelbox.visualization.transformers.residuals import ResidualDataTransformer


@pytest.fixture
def mock_results():
    """Mock model results object."""
    results = Mock(spec=[
        'resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'nobs',
        'rsquared', 'rsquared_adj', 'fvalue', 'f_pvalue', 'scale', 'mse_resid', 'model'
    ])

    np.random.seed(42)
    n = 100

    # Basic residual data
    results.resid = np.random.normal(0, 1, n)
    results.fittedvalues = np.random.normal(5, 2, n)

    # Model information
    results.params = Mock()
    results.params.index = ['const', 'x1', 'x2']
    results.params.__len__ = Mock(return_value=3)

    results.df_model = 2
    results.df_resid = n - 3
    results.nobs = n
    results.rsquared = 0.75
    results.rsquared_adj = 0.74
    results.fvalue = 50.0
    results.f_pvalue = 0.001

    # Scale
    results.scale = 1.0
    results.mse_resid = 1.0

    # Model class
    results.model = Mock(spec=['__class__'])
    results.model.__class__ = Mock()
    results.model.__class__.__name__ = 'FixedEffects'

    return results


class TestResidualDataTransformer:
    """Tests for ResidualDataTransformer."""

    def test_initialization(self):
        """Test transformer initialization."""
        transformer = ResidualDataTransformer()
        assert transformer is not None

    def test_transform_basic(self, mock_results):
        """Test basic transformation."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        assert isinstance(result, dict)
        assert 'residuals' in result
        assert 'fitted' in result
        assert 'standardized_residuals' in result
        assert 'model_info' in result

    def test_extract_residuals(self, mock_results):
        """Test residual extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        residuals = result['residuals']
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == 100

    def test_extract_fitted(self, mock_results):
        """Test fitted values extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        fitted = result['fitted']
        assert isinstance(fitted, np.ndarray)
        assert len(fitted) == 100

    def test_compute_standardized_residuals(self, mock_results):
        """Test standardized residuals computation."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        std_residuals = result['standardized_residuals']
        assert isinstance(std_residuals, np.ndarray)
        assert len(std_residuals) == 100
        # Should be roughly standardized
        assert abs(np.mean(std_residuals)) < 0.2
        assert abs(np.std(std_residuals) - 1.0) < 0.2

    def test_leverage_extraction(self, mock_results):
        """Test leverage values extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        leverage = result['leverage']
        # May be None if not available
        if leverage is not None:
            assert isinstance(leverage, np.ndarray)
            assert len(leverage) == 100

    def test_cooks_distance_computation(self, mock_results):
        """Test Cook's distance computation."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        cooks_d = result['cooks_d']
        # May be None if leverage not available
        if cooks_d is not None:
            assert isinstance(cooks_d, np.ndarray)
            assert len(cooks_d) == 100

    def test_model_info_extraction(self, mock_results):
        """Test model information extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        model_info = result['model_info']
        assert isinstance(model_info, dict)
        assert model_info['nobs'] == 100
        assert model_info['df_model'] == 2
        assert model_info['df_resid'] == 97
        assert model_info['rsquared'] == 0.75
        assert model_info['model_type'] == 'FixedEffects'

    def test_prepare_qq_data(self, mock_results):
        """Test Q-Q plot data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_qq_data(mock_results)

        assert 'residuals' in data
        assert 'standardized' in data
        assert 'show_confidence' in data
        assert 'confidence_level' in data

        assert data['standardized'] is True
        assert data['show_confidence'] is True
        assert data['confidence_level'] == 0.95

    def test_prepare_residual_fitted_data(self, mock_results):
        """Test residual vs fitted data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_residual_fitted_data(mock_results)

        assert 'fitted' in data
        assert 'residuals' in data
        assert 'add_lowess' in data
        assert 'add_reference' in data

        assert data['add_lowess'] is True
        assert data['add_reference'] is True

    def test_prepare_scale_location_data(self, mock_results):
        """Test scale-location data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_scale_location_data(mock_results)

        assert 'fitted' in data
        assert 'residuals' in data
        assert 'add_lowess' in data

        assert data['add_lowess'] is True

    def test_prepare_leverage_data(self, mock_results):
        """Test leverage plot data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_leverage_data(mock_results)

        assert 'residuals' in data
        assert 'leverage' in data
        assert 'show_contours' in data

    def test_prepare_timeseries_data(self, mock_results):
        """Test time series data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_timeseries_data(mock_results)

        assert 'residuals' in data
        assert 'add_bands' in data

        assert data['add_bands'] is True

    def test_prepare_distribution_data(self, mock_results):
        """Test distribution plot data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_distribution_data(mock_results)

        assert 'residuals' in data
        assert 'bins' in data
        assert 'show_kde' in data
        assert 'show_normal' in data

        assert data['bins'] == 'auto'
        assert data['show_kde'] is True
        assert data['show_normal'] is True


class TestResidualTransformerEdgeCases:
    """Edge case tests for ResidualDataTransformer."""

    def test_missing_residuals_attribute(self):
        """Test with missing residuals attribute."""
        results = Mock(spec=[])  # No attributes

        transformer = ResidualDataTransformer()

        with pytest.raises(AttributeError, match="no residuals attribute"):
            transformer.transform(results)

    def test_alternative_residuals_attribute(self):
        """Test with alternative residuals attribute name."""
        results = Mock(spec=['residuals', 'fittedvalues', 'params', 'df_model', 'df_resid', 'scale'])
        results.residuals = np.random.normal(0, 1, 100)  # Note: 'residuals' not 'resid'
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 1.0

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert 'residuals' in result
        assert len(result['residuals']) == 100

    def test_missing_fitted_values(self):
        """Test with missing fitted values."""
        results = Mock(spec=['resid', 'params', 'df_model'])
        results.resid = np.random.normal(0, 1, 100)
        results.params = Mock()
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2

        transformer = ResidualDataTransformer()

        with pytest.raises(AttributeError, match="Cannot extract fitted values"):
            transformer.transform(results)

    def test_leverage_not_available(self, mock_results):
        """Test when leverage cannot be computed."""
        # Remove attributes needed for leverage computation
        mock_results.get_influence = Mock(side_effect=Exception())
        delattr(mock_results, 'model')

        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        assert result['leverage'] is None

    def test_cooks_distance_without_leverage(self, mock_results):
        """Test Cook's distance when leverage unavailable."""
        # Remove attributes needed for leverage
        mock_results.get_influence = Mock(side_effect=Exception())
        delattr(mock_results, 'model')

        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        # Should be None when leverage is None
        assert result['cooks_d'] is None

    def test_time_index_extraction(self):
        """Test time index extraction."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'scale', '_data'])
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 1.0

        # Add time index
        results._data = Mock(spec=['time_index'])
        results._data.time_index = np.arange(100)

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert result['time_index'] is not None
        assert len(result['time_index']) == 100

    def test_entity_id_extraction(self):
        """Test entity ID extraction."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'scale', '_data'])
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 1.0

        # Add entity IDs
        results._data = Mock(spec=['entity_id'])
        results._data.entity_id = np.repeat([1, 2, 3, 4], 25)

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert result['entity_id'] is not None
        assert len(result['entity_id']) == 100

    def test_no_time_or_entity_data(self, mock_results):
        """Test when no time/entity data available."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        # Should be None when not available
        assert result['time_index'] is None
        assert result['entity_id'] is None

    def test_model_info_missing_attributes(self):
        """Test model info with missing attributes."""
        results = Mock(spec=['resid', 'fittedvalues', 'params'])
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)

        # Only minimal attributes
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        model_info = result['model_info']
        # Should still work with minimal info
        assert isinstance(model_info, dict)

    def test_very_small_sample(self):
        """Test with very small sample size."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'scale'])
        results.resid = np.array([0.5, -0.3, 0.8])
        results.fittedvalues = np.array([5.0, 4.5, 5.5])
        results.params = Mock()
        results.params.index = ['const', 'x1']
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.df_resid = 1
        results.scale = 1.0

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert len(result['residuals']) == 3
        assert len(result['fitted']) == 3

    def test_all_zero_residuals(self):
        """Test with all zero residuals (perfect fit)."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'scale'])
        results.resid = np.zeros(100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 0.0  # Perfect fit

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert np.all(result['residuals'] == 0)

    def test_standardized_residuals_with_zero_scale(self):
        """Test standardized residuals when scale is zero."""
        results = Mock(spec=['resid', 'fittedvalues', 'params', 'df_model', 'df_resid', 'scale'])
        results.resid = np.zeros(100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ['const', 'x1', 'x2']
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 0.0

        transformer = ResidualDataTransformer()
        # Should handle division by zero gracefully
        result = transformer.transform(results)

        # With zero scale, standardized residuals may be inf or nan
        assert 'standardized_residuals' in result
