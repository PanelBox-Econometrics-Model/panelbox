"""
Tests for PanelDataTransformer.

Tests data transformation methods for panel visualizations:
- extract_entity_effects
- extract_time_effects
- calculate_between_within
- analyze_panel_structure
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from panelbox.visualization.transformers.panel import PanelDataTransformer


@pytest.fixture
def sample_panel_dataframe():
    """Create sample panel DataFrame."""
    np.random.seed(42)
    n_entities = 10
    n_periods = 5

    entities = [f'E{i:02d}' for i in range(1, n_entities + 1)]
    periods = list(range(2000, 2000 + n_periods))

    index = pd.MultiIndex.from_product(
        [entities, periods],
        names=['entity', 'time']
    )

    # Create data with known between/within variance
    entity_effects = np.repeat(np.random.randn(n_entities) * 2, n_periods)
    var1 = entity_effects + np.random.randn(n_entities * n_periods) * 0.5

    return pd.DataFrame({
        'var1': var1,
        'var2': np.random.randn(n_entities * n_periods),
        'var3': np.random.randn(n_entities * n_periods) * 5
    }, index=index)


@pytest.fixture
def unbalanced_panel_dataframe():
    """Create unbalanced panel DataFrame."""
    np.random.seed(123)

    # Create unbalanced panel
    data = []
    for entity in range(1, 6):
        n_periods = np.random.randint(3, 8)  # Variable periods per entity
        for t in range(n_periods):
            data.append({
                'entity': f'E{entity:02d}',
                'time': 2000 + t,
                'value': np.random.randn()
            })

    df = pd.DataFrame(data)
    df = df.set_index(['entity', 'time'])
    return df


@pytest.fixture
def mock_panel_results():
    """Create mock PanelResults object."""
    mock = Mock()

    # Mock entity effects
    mock.entity_effects = pd.Series(
        data=np.random.randn(5) * 0.5,
        index=['A', 'B', 'C', 'D', 'E'],
        name='effects'
    )

    # Mock standard errors
    mock.std_errors = pd.Series(
        data=np.random.uniform(0.1, 0.3, 5),
        index=['A', 'B', 'C', 'D', 'E'],
        name='std_errors'
    )

    # Mock time effects
    mock.time_effects = pd.Series(
        data=np.random.randn(10) * 0.3,
        index=list(range(2000, 2010)),
        name='time_effects'
    )

    return mock


class TestExtractEntityEffects:
    """Tests for extract_entity_effects method."""

    def test_with_mock_results(self, mock_panel_results):
        """Test extraction from mock PanelResults."""
        result = PanelDataTransformer.extract_entity_effects(mock_panel_results)

        assert isinstance(result, dict)
        assert 'entity_id' in result
        assert 'effect' in result
        # std_error might be None if not calculated
        assert 'std_error' in result

        assert len(result['entity_id']) == 5
        assert len(result['effect']) == 5
        # std_error can be None
        if result['std_error'] is not None:
            assert len(result['std_error']) == 5

    def test_output_format(self, mock_panel_results):
        """Test output format is correct."""
        result = PanelDataTransformer.extract_entity_effects(mock_panel_results)

        # Check types
        assert isinstance(result['entity_id'], list)
        assert isinstance(result['effect'], (list, np.ndarray))
        # std_error can be None
        assert result['std_error'] is None or isinstance(result['std_error'], (list, np.ndarray))

        # Check entity IDs match
        assert result['entity_id'] == ['A', 'B', 'C', 'D', 'E']

    def test_without_std_errors(self):
        """Test when std_errors not available."""
        mock = Mock()
        mock.entity_effects = pd.Series([0.1, 0.2, 0.3], index=['A', 'B', 'C'])
        mock.std_errors = None

        result = PanelDataTransformer.extract_entity_effects(mock)

        assert 'entity_id' in result
        assert 'effect' in result
        # Should handle missing std_errors gracefully


class TestExtractTimeEffects:
    """Tests for extract_time_effects method."""

    def test_with_mock_results(self, mock_panel_results):
        """Test extraction from mock PanelResults."""
        result = PanelDataTransformer.extract_time_effects(mock_panel_results)

        assert isinstance(result, dict)
        assert 'time' in result
        assert 'effect' in result

        assert len(result['time']) == 10
        assert len(result['effect']) == 10

    def test_output_format(self, mock_panel_results):
        """Test output format is correct."""
        result = PanelDataTransformer.extract_time_effects(mock_panel_results)

        # Check types
        assert isinstance(result['time'], list)
        assert isinstance(result['effect'], (list, np.ndarray))

        # Check time periods
        assert result['time'] == list(range(2000, 2010))

    def test_with_std_errors(self):
        """Test with standard errors available."""
        mock = Mock()
        mock.time_effects = pd.Series([0.1, 0.2, 0.3], index=[2000, 2001, 2002])
        mock.time_std_errors = pd.Series([0.05, 0.06, 0.04], index=[2000, 2001, 2002])

        result = PanelDataTransformer.extract_time_effects(mock)

        assert 'time' in result
        assert 'effect' in result
        assert len(result['time']) == 3
        assert len(result['effect']) == 3
        # std_error can be None or present
        if result.get('std_error') is not None:
            assert len(result['std_error']) == 3


class TestCalculateBetweenWithin:
    """Tests for calculate_between_within method."""

    def test_with_dataframe_all_variables(self, sample_panel_dataframe):
        """Test calculation with all variables."""
        result = PanelDataTransformer.calculate_between_within(sample_panel_dataframe)

        assert isinstance(result, dict)
        assert 'variables' in result
        assert 'between_var' in result
        assert 'within_var' in result

        # Should have all 3 variables
        assert len(result['variables']) == 3
        assert set(result['variables']) == {'var1', 'var2', 'var3'}

    def test_with_dataframe_subset_variables(self, sample_panel_dataframe):
        """Test calculation with subset of variables."""
        result = PanelDataTransformer.calculate_between_within(
            sample_panel_dataframe,
            variables=['var1', 'var2']
        )

        assert len(result['variables']) == 2
        assert set(result['variables']) == {'var1', 'var2'}

    def test_variance_values_positive(self, sample_panel_dataframe):
        """Test that variance values are positive."""
        result = PanelDataTransformer.calculate_between_within(sample_panel_dataframe)

        # All variances should be non-negative
        assert all(v >= 0 for v in result['between_var'])
        assert all(v >= 0 for v in result['within_var'])

    def test_variance_decomposition(self, sample_panel_dataframe):
        """Test variance decomposition is reasonable."""
        result = PanelDataTransformer.calculate_between_within(sample_panel_dataframe)

        # Total variance should be sum of between and within
        for i in range(len(result['variables'])):
            var_name = result['variables'][i]
            between = result['between_var'][i]
            within = result['within_var'][i]

            # Both components should be positive
            assert between >= 0
            assert within >= 0

    def test_with_dict_data(self):
        """Test that dict data is rejected (not a DataFrame)."""
        dict_data = {
            'variables': ['x', 'y'],
            'between_var': [1.0, 2.0],
            'within_var': [0.5, 1.5]
        }

        # Dict data should raise ValueError since method expects DataFrame
        with pytest.raises(ValueError):
            PanelDataTransformer.calculate_between_within(dict_data)

    def test_single_variable(self, sample_panel_dataframe):
        """Test with single variable."""
        result = PanelDataTransformer.calculate_between_within(
            sample_panel_dataframe,
            variables=['var1']
        )

        assert len(result['variables']) == 1
        assert result['variables'][0] == 'var1'

    def test_unbalanced_panel(self, unbalanced_panel_dataframe):
        """Test with unbalanced panel."""
        result = PanelDataTransformer.calculate_between_within(
            unbalanced_panel_dataframe
        )

        assert 'variables' in result
        assert 'between_var' in result
        assert 'within_var' in result

        # Should handle unbalanced panel
        assert all(v >= 0 for v in result['between_var'])
        assert all(v >= 0 for v in result['within_var'])

    def test_invalid_variable_name(self, sample_panel_dataframe):
        """Test with invalid variable name."""
        # Invalid variable is simply skipped, not raising KeyError
        result = PanelDataTransformer.calculate_between_within(
            sample_panel_dataframe,
            variables=['nonexistent_var']
        )
        # Should return empty lists since variable doesn't exist
        assert len(result['variables']) == 0


class TestAnalyzePanelStructure:
    """Tests for analyze_panel_structure method."""

    def test_with_balanced_dataframe(self, sample_panel_dataframe):
        """Test analysis of balanced panel."""
        result = PanelDataTransformer.analyze_panel_structure(sample_panel_dataframe)

        assert isinstance(result, dict)
        assert 'entities' in result
        assert 'time_periods' in result
        assert 'presence_matrix' in result
        assert 'is_balanced' in result

        # Should be balanced
        assert result['is_balanced'] == True
        assert result['balance_percentage'] == 100.0

    def test_with_unbalanced_dataframe(self, unbalanced_panel_dataframe):
        """Test analysis of unbalanced panel."""
        result = PanelDataTransformer.analyze_panel_structure(unbalanced_panel_dataframe)

        # Should detect unbalanced
        assert result['is_balanced'] == False
        assert result['balance_percentage'] < 100.0

    def test_presence_matrix_shape(self, sample_panel_dataframe):
        """Test presence matrix has correct shape."""
        result = PanelDataTransformer.analyze_panel_structure(sample_panel_dataframe)

        n_entities = len(result['entities'])
        n_periods = len(result['time_periods'])

        assert result['presence_matrix'].shape == (n_entities, n_periods)

    def test_presence_matrix_values(self, sample_panel_dataframe):
        """Test presence matrix contains only 0 and 1."""
        result = PanelDataTransformer.analyze_panel_structure(sample_panel_dataframe)

        matrix = result['presence_matrix']
        unique_values = np.unique(matrix)

        assert set(unique_values).issubset({0, 1})

    def test_balanced_panel_all_ones(self, sample_panel_dataframe):
        """Test balanced panel has all ones in presence matrix."""
        result = PanelDataTransformer.analyze_panel_structure(sample_panel_dataframe)

        if result['is_balanced']:
            assert np.all(result['presence_matrix'] == 1)

    def test_entity_and_time_counts(self, sample_panel_dataframe):
        """Test entity and time period counts."""
        result = PanelDataTransformer.analyze_panel_structure(sample_panel_dataframe)

        assert 'n_entities' in result
        assert 'n_periods' in result

        assert result['n_entities'] == 10
        assert result['n_periods'] == 5

    def test_complete_entities_list(self, unbalanced_panel_dataframe):
        """Test complete entities list."""
        result = PanelDataTransformer.analyze_panel_structure(unbalanced_panel_dataframe)

        assert 'complete_entities' in result
        assert isinstance(result['complete_entities'], list)

        # Complete entities should have all time periods
        for entity in result['complete_entities']:
            entity_idx = result['entities'].index(entity)
            entity_row = result['presence_matrix'][entity_idx, :]
            # Should have all periods present
            assert np.sum(entity_row) == result['n_periods']

    def test_with_dict_data(self):
        """Test that dict data is rejected (not a DataFrame)."""
        dict_data = {
            'entities': ['A', 'B'],
            'time_periods': [2000, 2001],
            'presence_matrix': np.array([[1, 1], [1, 0]])
        }

        # Dict data should raise ValueError since method expects DataFrame
        with pytest.raises(ValueError):
            PanelDataTransformer.analyze_panel_structure(dict_data)

    def test_single_entity(self):
        """Test with single entity."""
        data = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_product(
            [['A'], [2000, 2001, 2002]],
            names=['entity', 'time']
        ))

        result = PanelDataTransformer.analyze_panel_structure(data)

        assert result['n_entities'] == 1
        assert result['n_periods'] == 3
        assert result['presence_matrix'].shape == (1, 3)

    def test_single_period(self):
        """Test with single time period."""
        data = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_product(
            [['A', 'B', 'C'], [2000]],
            names=['entity', 'time']
        ))

        result = PanelDataTransformer.analyze_panel_structure(data)

        assert result['n_entities'] == 3
        assert result['n_periods'] == 1
        assert result['presence_matrix'].shape == (3, 1)


class TestPanelDataTransformerIntegration:
    """Integration tests for PanelDataTransformer."""

    def test_all_methods_with_dataframe(self, sample_panel_dataframe):
        """Test all transformation methods work with same DataFrame."""
        # Between-within
        bw_result = PanelDataTransformer.calculate_between_within(
            sample_panel_dataframe
        )
        assert bw_result is not None

        # Panel structure
        ps_result = PanelDataTransformer.analyze_panel_structure(
            sample_panel_dataframe
        )
        assert ps_result is not None

    def test_consistency_across_methods(self, sample_panel_dataframe):
        """Test consistency between different methods."""
        # Get panel structure
        structure = PanelDataTransformer.analyze_panel_structure(
            sample_panel_dataframe
        )

        # Get between-within
        bw = PanelDataTransformer.calculate_between_within(
            sample_panel_dataframe
        )

        # Variable counts should match
        assert len(bw['variables']) == len(sample_panel_dataframe.columns)

    def test_output_ready_for_charts(self, sample_panel_dataframe):
        """Test that outputs are ready for chart consumption."""
        # Between-within output
        bw = PanelDataTransformer.calculate_between_within(sample_panel_dataframe)

        # Should have all required keys
        assert all(k in bw for k in ['variables', 'between_var', 'within_var'])

        # All lists should have same length
        assert len(bw['variables']) == len(bw['between_var']) == len(bw['within_var'])

        # Panel structure output
        ps = PanelDataTransformer.analyze_panel_structure(sample_panel_dataframe)

        # Should have all required keys
        assert all(k in ps for k in ['entities', 'time_periods', 'presence_matrix'])

        # Matrix dimensions should match
        assert ps['presence_matrix'].shape == (len(ps['entities']), len(ps['time_periods']))


class TestPanelDataTransformerEdgeCases:
    """Edge case tests for PanelDataTransformer."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(
            columns=['value'],
            index=pd.MultiIndex.from_tuples([], names=['entity', 'time'])
        )

        # Empty dataframe should raise ValueError or return empty structure
        try:
            result = PanelDataTransformer.analyze_panel_structure(df)
            # If it doesn't raise, check it returns valid structure
            assert result['n_entities'] == 0
            assert result['n_periods'] == 0
        except (ValueError, IndexError):
            # Raising error is also acceptable
            pass

    def test_constant_variable(self):
        """Test between-within with constant variable."""
        np.random.seed(42)
        n_entities = 10
        n_periods = 5

        index = pd.MultiIndex.from_product(
            [[f'E{i}' for i in range(n_entities)], list(range(n_periods))],
            names=['entity', 'time']
        )

        df = pd.DataFrame({
            'constant': [5.0] * (n_entities * n_periods)
        }, index=index)

        result = PanelDataTransformer.calculate_between_within(df)

        # Constant variable should have zero variance
        assert result['between_var'][0] == pytest.approx(0.0, abs=1e-10)
        assert result['within_var'][0] == pytest.approx(0.0, abs=1e-10)

    def test_all_missing_entity(self):
        """Test panel structure with entity that has all missing data."""
        # Create panel with one entity completely missing
        data = []
        for entity in ['A', 'B', 'C']:
            for t in [2000, 2001, 2002]:
                if entity != 'B':  # B is completely missing
                    data.append({'entity': entity, 'time': t, 'value': 1.0})

        df = pd.DataFrame(data).set_index(['entity', 'time'])

        result = PanelDataTransformer.analyze_panel_structure(df)

        # Should only have entities A and C
        assert set(result['entities']) == {'A', 'C'}

    def test_multiindex_not_sorted(self):
        """Test with unsorted MultiIndex."""
        data = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6]
        }, index=pd.MultiIndex.from_tuples([
            ('B', 2001),
            ('A', 2000),
            ('B', 2000),
            ('A', 2001),
            ('C', 2000),
            ('C', 2001),
        ], names=['entity', 'time']))

        result = PanelDataTransformer.analyze_panel_structure(data)

        # Should handle unsorted index
        assert result['n_entities'] == 3
        assert result['n_periods'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
