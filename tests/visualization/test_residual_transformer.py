"""
Tests for ResidualDataTransformer.

Tests the data transformation layer that converts model results
to residual diagnostic chart format.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from panelbox.visualization.transformers.residuals import ResidualDataTransformer


@pytest.fixture
def mock_results():
    """Mock model results object."""
    results = Mock(
        spec=[
            "resid",
            "fittedvalues",
            "params",
            "df_model",
            "df_resid",
            "nobs",
            "rsquared",
            "rsquared_adj",
            "fvalue",
            "f_pvalue",
            "scale",
            "mse_resid",
            "model",
        ]
    )

    np.random.seed(42)
    n = 100

    # Basic residual data
    results.resid = np.random.normal(0, 1, n)
    results.fittedvalues = np.random.normal(5, 2, n)

    # Model information
    results.params = Mock()
    results.params.index = ["const", "x1", "x2"]
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
    results.model = Mock(spec=["__class__"])
    results.model.__class__ = Mock()
    results.model.__class__.__name__ = "FixedEffects"

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
        assert "residuals" in result
        assert "fitted" in result
        assert "standardized_residuals" in result
        assert "model_info" in result

    def test_extract_residuals(self, mock_results):
        """Test residual extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        residuals = result["residuals"]
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == 100

    def test_extract_fitted(self, mock_results):
        """Test fitted values extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        fitted = result["fitted"]
        assert isinstance(fitted, np.ndarray)
        assert len(fitted) == 100

    def test_compute_standardized_residuals(self, mock_results):
        """Test standardized residuals computation."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        std_residuals = result["standardized_residuals"]
        assert isinstance(std_residuals, np.ndarray)
        assert len(std_residuals) == 100
        # Should be roughly standardized
        assert abs(np.mean(std_residuals)) < 0.2
        assert abs(np.std(std_residuals) - 1.0) < 0.2

    def test_leverage_extraction(self, mock_results):
        """Test leverage values extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        leverage = result["leverage"]
        # May be None if not available
        if leverage is not None:
            assert isinstance(leverage, np.ndarray)
            assert len(leverage) == 100

    def test_cooks_distance_computation(self, mock_results):
        """Test Cook's distance computation."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        cooks_d = result["cooks_d"]
        # May be None if leverage not available
        if cooks_d is not None:
            assert isinstance(cooks_d, np.ndarray)
            assert len(cooks_d) == 100

    def test_model_info_extraction(self, mock_results):
        """Test model information extraction."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        model_info = result["model_info"]
        assert isinstance(model_info, dict)
        assert model_info["nobs"] == 100
        assert model_info["df_model"] == 2
        assert model_info["df_resid"] == 97
        assert model_info["rsquared"] == 0.75
        assert model_info["model_type"] == "FixedEffects"

    def test_prepare_qq_data(self, mock_results):
        """Test Q-Q plot data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_qq_data(mock_results)

        assert "residuals" in data
        assert "standardized" in data
        assert "show_confidence" in data
        assert "confidence_level" in data

        assert data["standardized"] is True
        assert data["show_confidence"] is True
        assert data["confidence_level"] == 0.95

    def test_prepare_residual_fitted_data(self, mock_results):
        """Test residual vs fitted data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_residual_fitted_data(mock_results)

        assert "fitted" in data
        assert "residuals" in data
        assert "add_lowess" in data
        assert "add_reference" in data

        assert data["add_lowess"] is True
        assert data["add_reference"] is True

    def test_prepare_scale_location_data(self, mock_results):
        """Test scale-location data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_scale_location_data(mock_results)

        assert "fitted" in data
        assert "residuals" in data
        assert "add_lowess" in data

        assert data["add_lowess"] is True

    def test_prepare_leverage_data(self, mock_results):
        """Test leverage plot data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_leverage_data(mock_results)

        assert "residuals" in data
        assert "leverage" in data
        assert "show_contours" in data

    def test_prepare_timeseries_data(self, mock_results):
        """Test time series data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_timeseries_data(mock_results)

        assert "residuals" in data
        assert "add_bands" in data

        assert data["add_bands"] is True

    def test_prepare_distribution_data(self, mock_results):
        """Test distribution plot data preparation."""
        transformer = ResidualDataTransformer()
        data = transformer.prepare_distribution_data(mock_results)

        assert "residuals" in data
        assert "bins" in data
        assert "show_kde" in data
        assert "show_normal" in data

        assert data["bins"] == "auto"
        assert data["show_kde"] is True
        assert data["show_normal"] is True


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
        results = Mock(
            spec=["residuals", "fittedvalues", "params", "df_model", "df_resid", "scale"]
        )
        results.residuals = np.random.normal(0, 1, 100)  # Note: 'residuals' not 'resid'
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 1.0

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert "residuals" in result
        assert len(result["residuals"]) == 100

    def test_missing_fitted_values(self):
        """Test with missing fitted values."""
        results = Mock(spec=["resid", "params", "df_model"])
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
        delattr(mock_results, "model")

        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        assert result["leverage"] is None

    def test_cooks_distance_without_leverage(self, mock_results):
        """Test Cook's distance when leverage unavailable."""
        # Remove attributes needed for leverage
        mock_results.get_influence = Mock(side_effect=Exception())
        delattr(mock_results, "model")

        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        # Should be None when leverage is None
        assert result["cooks_d"] is None

    def test_time_index_extraction(self):
        """Test time index extraction."""
        results = Mock(
            spec=["resid", "fittedvalues", "params", "df_model", "df_resid", "scale", "_data"]
        )
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 1.0

        # Add time index
        results._data = Mock(spec=["time_index"])
        results._data.time_index = np.arange(100)

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert result["time_index"] is not None
        assert len(result["time_index"]) == 100

    def test_entity_id_extraction(self):
        """Test entity ID extraction."""
        results = Mock(
            spec=["resid", "fittedvalues", "params", "df_model", "df_resid", "scale", "_data"]
        )
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 1.0

        # Add entity IDs
        results._data = Mock(spec=["entity_id"])
        results._data.entity_id = np.repeat([1, 2, 3, 4], 25)

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert result["entity_id"] is not None
        assert len(result["entity_id"]) == 100

    def test_no_time_or_entity_data(self, mock_results):
        """Test when no time/entity data available."""
        transformer = ResidualDataTransformer()
        result = transformer.transform(mock_results)

        # Should be None when not available
        assert result["time_index"] is None
        assert result["entity_id"] is None

    def test_model_info_missing_attributes(self):
        """Test model info with missing attributes."""
        results = Mock(spec=["resid", "fittedvalues", "params"])
        results.resid = np.random.normal(0, 1, 100)
        results.fittedvalues = np.random.normal(5, 2, 100)

        # Only minimal attributes
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=3)

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        model_info = result["model_info"]
        # Should still work with minimal info
        assert isinstance(model_info, dict)

    def test_very_small_sample(self):
        """Test with very small sample size."""
        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "df_resid", "scale"])
        results.resid = np.array([0.5, -0.3, 0.8])
        results.fittedvalues = np.array([5.0, 4.5, 5.5])
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.df_resid = 1
        results.scale = 1.0

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert len(result["residuals"]) == 3
        assert len(result["fitted"]) == 3

    def test_all_zero_residuals(self):
        """Test with all zero residuals (perfect fit)."""
        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "df_resid", "scale"])
        results.resid = np.zeros(100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 0.0  # Perfect fit

        transformer = ResidualDataTransformer()
        result = transformer.transform(results)

        assert np.all(result["residuals"] == 0)

    def test_standardized_residuals_with_zero_scale(self):
        """Test standardized residuals when scale is zero."""
        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "df_resid", "scale"])
        results.resid = np.zeros(100)
        results.fittedvalues = np.random.normal(5, 2, 100)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=3)
        results.df_model = 2
        results.df_resid = 97
        results.scale = 0.0

        transformer = ResidualDataTransformer()
        # Should handle division by zero gracefully
        result = transformer.transform(results)

        # With zero scale, standardized residuals may be inf or nan
        assert "standardized_residuals" in result


class TestFittedValuesAlternativePaths:
    """Tests for fitted values extraction alternative paths."""

    def test_fitted_values_via_fitted_values_attr(self):
        """Test fitted values extraction via fitted_values attribute (line 134)."""
        np.random.seed(42)
        results = Mock(spec=["resid", "fitted_values", "params", "df_model", "scale"])
        results.resid = np.random.normal(0, 1, 50)
        results.fitted_values = np.random.randn(50)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        transformer = ResidualDataTransformer()
        fitted = transformer._extract_fitted(results)

        assert len(fitted) == 50
        assert isinstance(fitted, np.ndarray)

    def test_fitted_values_via_predict_method(self):
        """Test fitted values extraction via predict() method (line 136)."""
        np.random.seed(42)
        results = Mock(spec=["resid", "predict", "params", "df_model", "scale"])
        results.resid = np.random.normal(0, 1, 30)
        predicted = np.random.randn(30)
        results.predict = Mock(return_value=predicted)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        transformer = ResidualDataTransformer()
        fitted = transformer._extract_fitted(results)

        assert len(fitted) == 30
        results.predict.assert_called_once()

    def test_scale_via_mse_resid(self):
        """Test scale extraction via mse_resid attribute (line 160)."""
        np.random.seed(42)
        results = Mock(
            spec=["resid", "fittedvalues", "params", "df_model", "df_resid", "mse_resid"]
        )
        results.resid = np.random.normal(0, 1, 50)
        results.fittedvalues = np.random.normal(5, 2, 50)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.df_resid = 48
        results.mse_resid = 2.5

        transformer = ResidualDataTransformer()
        std_resids = transformer._compute_standardized_residuals(results.resid, results)

        # Should use mse_resid as scale: residuals / sqrt(2.5)
        expected = results.resid / np.sqrt(2.5)
        np.testing.assert_array_almost_equal(std_resids, expected)


class TestLeverageAlternativePaths:
    """Tests for leverage computation alternative paths."""

    def test_leverage_via_get_influence(self):
        """Test leverage extraction via get_influence() (lines 185-186)."""
        np.random.seed(42)
        n = 50
        results = Mock(
            spec=[
                "resid",
                "fittedvalues",
                "params",
                "df_model",
                "df_resid",
                "scale",
                "get_influence",
            ]
        )
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.df_resid = n - 2
        results.scale = 1.0

        influence = Mock()
        influence.hat_matrix_diag = np.random.uniform(0, 0.5, n)
        results.get_influence = Mock(return_value=influence)

        transformer = ResidualDataTransformer()
        leverage = transformer._compute_leverage(results)

        assert leverage is not None
        assert len(leverage) == n
        np.testing.assert_array_equal(leverage, influence.hat_matrix_diag)

    def test_leverage_manual_from_exog(self):
        """Test manual leverage computation from model.exog (lines 192-199)."""
        np.random.seed(42)
        n = 50
        k = 3
        results = Mock(spec=["resid", "fittedvalues", "params", "model", "scale"])
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=k)
        results.scale = 1.0

        # Set up model.exog for manual computation
        X = np.random.randn(n, k)
        results.model = Mock(spec=["exog"])
        results.model.exog = X

        transformer = ResidualDataTransformer()
        leverage = transformer._compute_leverage(results)

        assert leverage is not None
        assert len(leverage) == n
        # Verify against manual computation
        XtX_inv = np.linalg.inv(X.T @ X)
        expected_leverage = np.sum((X @ XtX_inv) * X, axis=1)
        np.testing.assert_array_almost_equal(leverage, expected_leverage)

    def test_cooks_distance_via_get_influence(self):
        """Test Cook's distance via get_influence() (lines 228-233)."""
        np.random.seed(42)
        n = 50
        results = Mock(
            spec=[
                "resid",
                "fittedvalues",
                "params",
                "df_model",
                "df_resid",
                "scale",
                "get_influence",
            ]
        )
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.df_resid = n - 2
        results.scale = 1.0

        expected_cooks_d = np.random.uniform(0, 1, n)
        influence = Mock()
        influence.hat_matrix_diag = np.random.uniform(0.01, 0.5, n)
        influence.cooks_distance = (expected_cooks_d, np.random.uniform(0, 1, n))
        results.get_influence = Mock(return_value=influence)

        transformer = ResidualDataTransformer()
        leverage = np.asarray(influence.hat_matrix_diag)
        std_resids = results.resid / np.sqrt(results.scale)
        cooks_d = transformer._compute_cooks_distance(results, std_resids, leverage)

        assert cooks_d is not None
        assert len(cooks_d) == n
        np.testing.assert_array_almost_equal(cooks_d, expected_cooks_d)

    def test_cooks_distance_manual_computation(self):
        """Test manual Cook's distance computation (lines 237-244)."""
        np.random.seed(42)
        n = 50
        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "scale"])
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        leverage = np.random.uniform(0.01, 0.3, n)
        std_resids = results.resid / np.sqrt(results.scale)

        transformer = ResidualDataTransformer()
        cooks_d = transformer._compute_cooks_distance(results, std_resids, leverage)

        assert cooks_d is not None
        assert len(cooks_d) == n

        # Verify manual formula: (std_resid^2 / p) * (leverage / (1 - leverage)^2)
        p = results.df_model
        expected = (std_resids**2 / p) * (leverage / (1 - leverage) ** 2)
        expected = np.where(np.isfinite(expected), expected, 0)
        np.testing.assert_array_almost_equal(cooks_d, expected)


class TestTimeEntityExtractionPaths:
    """Tests for time_index and entity_id extraction paths."""

    def test_time_index_from_multiindex(self):
        """Test time index extraction from MultiIndex (lines 263-269)."""
        import pandas as pd

        np.random.seed(42)
        n = 20
        entities = np.repeat(["A", "B", "C", "D"], 5)
        times = np.tile([2000, 2001, 2002, 2003, 2004], 4)
        idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])

        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "scale", "model"])
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        data_mock = Mock(spec=["row_labels"])
        data_mock.row_labels = idx
        results.model = Mock(spec=["data"])
        results.model.data = data_mock

        transformer = ResidualDataTransformer()
        time_index = transformer._extract_time_index(results)

        assert time_index is not None
        assert len(time_index) == n
        np.testing.assert_array_equal(time_index, times)

    def test_entity_id_from_multiindex(self):
        """Test entity ID extraction from MultiIndex (lines 294-300)."""
        import pandas as pd

        np.random.seed(42)
        n = 20
        entities = np.repeat(["A", "B", "C", "D"], 5)
        times = np.tile([2000, 2001, 2002, 2003, 2004], 4)
        idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])

        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "scale", "model"])
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        data_mock = Mock(spec=["row_labels"])
        data_mock.row_labels = idx
        results.model = Mock(spec=["data"])
        results.model.data = data_mock

        transformer = ResidualDataTransformer()
        entity_id = transformer._extract_entity_id(results)

        assert entity_id is not None
        assert len(entity_id) == n
        np.testing.assert_array_equal(entity_id, entities)

    def test_prepare_leverage_data_with_cooks_d(self):
        """Test prepare_leverage_data includes Cook's D (lines 435, 437-440)."""
        np.random.seed(42)
        n = 50
        k = 3
        results = Mock(
            spec=[
                "resid",
                "fittedvalues",
                "params",
                "df_model",
                "df_resid",
                "scale",
                "model",
            ]
        )
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1", "x2"]
        results.params.__len__ = Mock(return_value=k)
        results.df_model = k - 1
        results.df_resid = n - k
        results.scale = 1.0

        X = np.random.randn(n, k)
        results.model = Mock(spec=["exog"])
        results.model.exog = X

        transformer = ResidualDataTransformer()
        data = transformer.prepare_leverage_data(results)

        assert "residuals" in data
        assert "leverage" in data
        assert "show_contours" in data
        assert data["show_contours"] is True
        assert "cooks_d" in data
        assert len(data["cooks_d"]) == n
        assert "params" in data

    def test_prepare_timeseries_data_with_time_entity(self):
        """Test prepare_timeseries_data with time and entity (lines 463, 466)."""
        import pandas as pd

        np.random.seed(42)
        n = 20
        entities = np.repeat(["A", "B", "C", "D"], 5)
        times = np.tile([2000, 2001, 2002, 2003, 2004], 4)
        idx = pd.MultiIndex.from_arrays([entities, times], names=["entity", "time"])

        results = Mock(spec=["resid", "fittedvalues", "params", "df_model", "scale", "model"])
        results.resid = np.random.normal(0, 1, n)
        results.fittedvalues = np.random.normal(5, 2, n)
        results.params = Mock()
        results.params.index = ["const", "x1"]
        results.params.__len__ = Mock(return_value=2)
        results.df_model = 1
        results.scale = 1.0

        data_mock = Mock(spec=["row_labels"])
        data_mock.row_labels = idx
        results.model = Mock(spec=["data"])
        results.model.data = data_mock

        transformer = ResidualDataTransformer()
        data = transformer.prepare_timeseries_data(results)

        assert "residuals" in data
        assert "add_bands" in data
        assert "time_index" in data
        assert "entity_id" in data
        assert len(data["time_index"]) == n
        assert len(data["entity_id"]) == n
