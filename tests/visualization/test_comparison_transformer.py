"""
Tests for ComparisonDataTransformer.

Tests the data transformation layer for model comparison charts.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from panelbox.visualization.transformers.comparison import ComparisonDataTransformer


@pytest.fixture
def mock_results_simple():
    """Create simple mock model results."""
    results1 = Mock()
    results1.params = pd.Series({'x1': 1.2, 'x2': -0.5, 'const': 2.0})
    results1.std_errors = pd.Series({'x1': 0.3, 'x2': 0.2, 'const': 0.5})
    results1.pvalues = pd.Series({'x1': 0.001, 'x2': 0.02, 'const': 0.0001})
    results1.rsquared = 0.75
    results1.rsquared_adj = 0.73
    results1.fvalue = 45.2
    results1.llf = -150.5
    results1.aic = 305.0
    results1.bic = 315.0
    results1.hqic = 308.5

    results2 = Mock()
    results2.params = pd.Series({'x1': 1.5, 'x2': -0.3, 'const': 1.8})
    results2.std_errors = pd.Series({'x1': 0.25, 'x2': 0.18, 'const': 0.45})
    results2.pvalues = pd.Series({'x1': 0.0005, 'x2': 0.05, 'const': 0.0002})
    results2.rsquared = 0.80
    results2.rsquared_adj = 0.78
    results2.fvalue = 52.1
    results2.llf = -145.2
    results2.aic = 294.4
    results2.bic = 304.4
    results2.hqic = 297.9

    return [results1, results2]


@pytest.fixture
def mock_results_different_vars():
    """Create mock results with different variables."""
    results1 = Mock()
    results1.params = pd.Series({'x1': 1.2, 'x2': -0.5})
    results1.std_errors = pd.Series({'x1': 0.3, 'x2': 0.2})
    results1.pvalues = pd.Series({'x1': 0.001, 'x2': 0.02})
    results1.rsquared = 0.65
    results1.rsquared_adj = 0.63
    results1.fvalue = 40.0
    results1.llf = -160.0
    results1.aic = 324.0
    results1.bic = 332.0
    results1.hqic = 326.5

    results2 = Mock()
    results2.params = pd.Series({'x1': 1.5, 'x3': 0.8})  # x3 instead of x2
    results2.std_errors = pd.Series({'x1': 0.25, 'x3': 0.22})
    results2.pvalues = pd.Series({'x1': 0.0005, 'x3': 0.01})
    results2.rsquared = 0.70
    results2.rsquared_adj = 0.68
    results2.fvalue = 45.0
    results2.llf = -155.0
    results2.aic = 314.0
    results2.bic = 322.0
    results2.hqic = 316.5

    return [results1, results2]


@pytest.fixture
def mock_results_alternative_attrs():
    """Create mock results with alternative attribute names (bse instead of std_errors)."""
    results = Mock()
    results.params = pd.Series({'x1': 1.2, 'x2': -0.5})
    results.bse = pd.Series({'x1': 0.3, 'x2': 0.2})  # bse instead of std_errors
    results.pvalues = pd.Series({'x1': 0.001, 'x2': 0.02})
    results.rsquared = 0.75
    results.rsquared_adj = 0.73
    results.fvalue = 45.2
    results.llf = -150.5
    results.aic = 305.0
    results.bic = 315.0
    results.hqic = 308.5

    return [results]


class TestComparisonDataTransformer:
    """Tests for ComparisonDataTransformer main transform method."""

    def test_transform_basic(self, mock_results_simple):
        """Test basic transformation of model results."""
        transformer = ComparisonDataTransformer()
        result = transformer.transform(mock_results_simple)

        assert 'models' in result
        assert 'coefficients' in result
        assert 'std_errors' in result
        assert 'pvalues' in result
        assert 'fit_metrics' in result
        assert 'ic_values' in result

    def test_transform_with_custom_names(self, mock_results_simple):
        """Test transformation with custom model names."""
        transformer = ComparisonDataTransformer()
        names = ['Model A', 'Model B']
        result = transformer.transform(mock_results_simple, names=names)

        assert result['models'] == names

    def test_transform_default_names(self, mock_results_simple):
        """Test transformation with default model names."""
        transformer = ComparisonDataTransformer()
        result = transformer.transform(mock_results_simple)

        assert result['models'] == ['Model 1', 'Model 2']

    def test_transform_different_variables(self, mock_results_different_vars):
        """Test transformation when models have different variables."""
        transformer = ComparisonDataTransformer()
        result = transformer.transform(mock_results_different_vars)

        # Should have all unique variables
        coef_vars = set(result['coefficients'].keys())
        assert 'x1' in coef_vars
        assert 'x2' in coef_vars
        assert 'x3' in coef_vars


class TestExtractCoefficients:
    """Tests for _extract_coefficients method."""

    def test_extract_coefficients_simple(self, mock_results_simple):
        """Test extracting coefficients from simple results."""
        transformer = ComparisonDataTransformer()
        coefficients = transformer._extract_coefficients(mock_results_simple)

        assert 'x1' in coefficients
        assert 'x2' in coefficients
        assert 'const' in coefficients

        assert coefficients['x1'] == [1.2, 1.5]
        assert coefficients['x2'] == [-0.5, -0.3]
        assert coefficients['const'] == [2.0, 1.8]

    def test_extract_coefficients_different_vars(self, mock_results_different_vars):
        """Test extracting coefficients when models have different variables."""
        transformer = ComparisonDataTransformer()
        coefficients = transformer._extract_coefficients(mock_results_different_vars)

        # x1 should be in both models
        assert coefficients['x1'] == [1.2, 1.5]

        # x2 should be NaN for model 2
        assert coefficients['x2'][0] == -0.5
        assert np.isnan(coefficients['x2'][1])

        # x3 should be NaN for model 1
        assert np.isnan(coefficients['x3'][0])
        assert coefficients['x3'][1] == 0.8

    def test_extract_coefficients_missing_attr(self):
        """Test extracting coefficients when params attribute is missing."""
        results_no_params = Mock(spec=[])  # No params attribute

        transformer = ComparisonDataTransformer()
        coefficients = transformer._extract_coefficients([results_no_params])

        assert coefficients == {}


class TestExtractStdErrors:
    """Tests for _extract_std_errors method."""

    def test_extract_std_errors_simple(self, mock_results_simple):
        """Test extracting standard errors from simple results."""
        transformer = ComparisonDataTransformer()
        std_errors = transformer._extract_std_errors(mock_results_simple)

        assert 'x1' in std_errors
        assert 'x2' in std_errors
        assert 'const' in std_errors

        assert std_errors['x1'] == [0.3, 0.25]
        assert std_errors['x2'] == [0.2, 0.18]
        assert std_errors['const'] == [0.5, 0.45]

    def test_extract_std_errors_bse_attribute(self, mock_results_alternative_attrs):
        """Test extracting standard errors from bse attribute."""
        transformer = ComparisonDataTransformer()
        std_errors = transformer._extract_std_errors(mock_results_alternative_attrs)

        assert 'x1' in std_errors
        assert 'x2' in std_errors

        assert std_errors['x1'] == [0.3]
        assert std_errors['x2'] == [0.2]

    def test_extract_std_errors_missing_attr(self):
        """Test extracting standard errors when both std_errors and bse are missing."""
        results = Mock()
        results.params = pd.Series({'x1': 1.2})
        # No std_errors or bse attribute

        transformer = ComparisonDataTransformer()
        std_errors = transformer._extract_std_errors([results])

        assert 'x1' in std_errors
        assert np.isnan(std_errors['x1'][0])


class TestExtractPValues:
    """Tests for _extract_pvalues method."""

    def test_extract_pvalues_simple(self, mock_results_simple):
        """Test extracting p-values from simple results."""
        transformer = ComparisonDataTransformer()
        pvalues = transformer._extract_pvalues(mock_results_simple)

        assert 'x1' in pvalues
        assert 'x2' in pvalues
        assert 'const' in pvalues

        assert pvalues['x1'] == [0.001, 0.0005]
        assert pvalues['x2'] == [0.02, 0.05]
        assert pvalues['const'] == [0.0001, 0.0002]

    def test_extract_pvalues_missing_attr(self):
        """Test extracting p-values when pvalues attribute is missing."""
        results = Mock()
        results.params = pd.Series({'x1': 1.2})
        # No pvalues attribute

        transformer = ComparisonDataTransformer()
        pvalues = transformer._extract_pvalues([results])

        assert 'x1' in pvalues
        assert np.isnan(pvalues['x1'][0])


class TestExtractFitMetrics:
    """Tests for _extract_fit_metrics method."""

    def test_extract_fit_metrics_simple(self, mock_results_simple):
        """Test extracting fit metrics from simple results."""
        transformer = ComparisonDataTransformer()
        metrics = transformer._extract_fit_metrics(mock_results_simple)

        assert 'R²' in metrics
        assert 'Adj. R²' in metrics
        assert 'F-statistic' in metrics
        assert 'Log-Likelihood' in metrics

        assert metrics['R²'] == [0.75, 0.80]
        assert metrics['Adj. R²'] == [0.73, 0.78]
        assert metrics['F-statistic'] == [45.2, 52.1]
        assert metrics['Log-Likelihood'] == [-150.5, -145.2]

    def test_extract_fit_metrics_missing_attrs(self):
        """Test extracting fit metrics when some attributes are missing."""
        results = Mock()
        results.rsquared = 0.75
        # Missing other attributes

        transformer = ComparisonDataTransformer()
        metrics = transformer._extract_fit_metrics([results])

        assert metrics['R²'] == [0.75]
        assert np.isnan(metrics['Adj. R²'][0])
        assert np.isnan(metrics['F-statistic'][0])
        assert np.isnan(metrics['Log-Likelihood'][0])


class TestExtractICValues:
    """Tests for _extract_ic_values method."""

    def test_extract_ic_values_simple(self, mock_results_simple):
        """Test extracting information criteria from simple results."""
        transformer = ComparisonDataTransformer()
        ic_values = transformer._extract_ic_values(mock_results_simple)

        assert 'AIC' in ic_values
        assert 'BIC' in ic_values
        assert 'HQIC' in ic_values

        assert ic_values['AIC'] == [305.0, 294.4]
        assert ic_values['BIC'] == [315.0, 304.4]
        assert ic_values['HQIC'] == [308.5, 297.9]

    def test_extract_ic_values_missing_attrs(self):
        """Test extracting IC values when some attributes are missing."""
        results = Mock()
        results.aic = 305.0
        # Missing bic and hqic

        transformer = ComparisonDataTransformer()
        ic_values = transformer._extract_ic_values([results])

        assert ic_values['AIC'] == [305.0]
        assert np.isnan(ic_values['BIC'][0])
        assert np.isnan(ic_values['HQIC'][0])


class TestPrepareCoefficients:
    """Tests for prepare_coefficient_comparison method."""

    def test_prepare_coefficient_comparison_simple(self, mock_results_simple):
        """Test preparing coefficient comparison data."""
        transformer = ComparisonDataTransformer()
        data = transformer.prepare_coefficient_comparison(mock_results_simple)

        assert 'variables' in data
        assert 'models' in data
        assert 'coefficients' in data
        assert 'std_errors' in data

        # Check shapes
        assert len(data['variables']) == 3  # x1, x2, const
        assert len(data['models']) == 2
        assert data['coefficients'].shape == (3, 2)  # 3 variables, 2 models
        assert data['std_errors'].shape == (3, 2)

    def test_prepare_coefficient_comparison_with_names(self, mock_results_simple):
        """Test preparing coefficient comparison with custom names."""
        transformer = ComparisonDataTransformer()
        names = ['FE', 'RE']
        data = transformer.prepare_coefficient_comparison(mock_results_simple, names=names)

        assert data['models'] == names

    def test_prepare_coefficient_comparison_subset_variables(self, mock_results_simple):
        """Test preparing coefficient comparison with variable subset."""
        transformer = ComparisonDataTransformer()
        variables = ['x1', 'x2']
        data = transformer.prepare_coefficient_comparison(
            mock_results_simple,
            variables=variables
        )

        assert data['variables'] == variables
        assert data['coefficients'].shape == (2, 2)  # 2 variables, 2 models


class TestPrepareForestPlot:
    """Tests for prepare_forest_plot method."""

    def test_prepare_forest_plot_simple(self, mock_results_simple):
        """Test preparing forest plot data for single model."""
        transformer = ComparisonDataTransformer()
        data = transformer.prepare_forest_plot(mock_results_simple[0])

        assert 'variables' in data
        assert 'coefficients' in data
        assert 'ci_lower' in data
        assert 'ci_upper' in data
        assert 'pvalues' in data

        assert len(data['variables']) == 3
        assert len(data['coefficients']) == 3
        assert len(data['ci_lower']) == 3
        assert len(data['ci_upper']) == 3

    def test_prepare_forest_plot_confidence_level(self, mock_results_simple):
        """Test preparing forest plot with different confidence levels."""
        transformer = ComparisonDataTransformer()

        # 95% CI
        data_95 = transformer.prepare_forest_plot(
            mock_results_simple[0],
            confidence_level=0.95
        )

        # 99% CI
        data_99 = transformer.prepare_forest_plot(
            mock_results_simple[0],
            confidence_level=0.99
        )

        # 99% CI should be wider than 95% CI
        ci_width_95 = data_95['ci_upper'][0] - data_95['ci_lower'][0]
        ci_width_99 = data_99['ci_upper'][0] - data_99['ci_lower'][0]
        assert ci_width_99 > ci_width_95

    def test_prepare_forest_plot_subset_variables(self, mock_results_simple):
        """Test preparing forest plot with variable subset."""
        transformer = ComparisonDataTransformer()
        variables = ['x1']
        data = transformer.prepare_forest_plot(
            mock_results_simple[0],
            variables=variables
        )

        assert data['variables'] == variables
        assert len(data['coefficients']) == 1


class TestPrepareModelFit:
    """Tests for prepare_model_fit_comparison method."""

    def test_prepare_model_fit_simple(self, mock_results_simple):
        """Test preparing model fit comparison data."""
        transformer = ComparisonDataTransformer()
        data = transformer.prepare_model_fit_comparison(mock_results_simple)

        assert 'models' in data
        assert 'r_squared' in data
        assert 'adj_r_squared' in data
        assert 'f_statistic' in data
        assert 'log_likelihood' in data

        assert len(data['models']) == 2
        assert len(data['r_squared']) == 2
        assert data['r_squared'][0] == 0.75
        assert data['r_squared'][1] == 0.80


class TestPrepareIC:
    """Tests for prepare_ic_comparison method."""

    def test_prepare_ic_simple(self, mock_results_simple):
        """Test preparing IC comparison data."""
        transformer = ComparisonDataTransformer()
        data = transformer.prepare_ic_comparison(mock_results_simple)

        assert 'models' in data
        assert 'aic' in data
        assert 'bic' in data
        assert 'hqic' in data

        assert len(data['models']) == 2
        assert len(data['aic']) == 2
        assert data['aic'][0] == 305.0
        assert data['aic'][1] == 294.4
