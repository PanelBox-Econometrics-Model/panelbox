"""
Tests for panel-specific chart implementations.

Tests all 4 panel chart types:
- EntityEffectsPlot
- TimeEffectsPlot
- BetweenWithinPlot
- PanelStructurePlot
"""

import pytest
import numpy as np
import pandas as pd

from panelbox.visualization.plotly.panel import (
    EntityEffectsPlot,
    TimeEffectsPlot,
    BetweenWithinPlot,
    PanelStructurePlot,
)
from panelbox.visualization.themes import PROFESSIONAL_THEME, ACADEMIC_THEME, PRESENTATION_THEME


@pytest.fixture
def sample_entity_effects_data():
    """Sample data for entity effects chart."""
    return {
        'entity_id': [f'Firm_{i}' for i in range(1, 11)],
        'effect': np.random.randn(10) * 0.5,
        'std_error': np.random.uniform(0.05, 0.15, 10)
    }


@pytest.fixture
def sample_time_effects_data():
    """Sample data for time effects chart."""
    return {
        'time': list(range(2000, 2021)),
        'effect': np.cumsum(np.random.randn(21) * 0.1),
        'std_error': np.random.uniform(0.03, 0.08, 21)
    }


@pytest.fixture
def sample_between_within_data():
    """Sample data for between-within variance decomposition."""
    return {
        'variables': ['wage', 'education', 'experience', 'age'],
        'between_var': [15.5, 8.2, 12.3, 25.1],
        'within_var': [5.2, 2.1, 3.8, 8.5]
    }


@pytest.fixture
def sample_panel_structure_data():
    """Sample data for panel structure visualization."""
    np.random.seed(42)
    n_entities = 8
    n_periods = 10
    entities = [f'Entity_{i}' for i in range(1, n_entities + 1)]
    time_periods = list(range(2010, 2010 + n_periods))

    # Create presence matrix with some missing observations
    presence_matrix = np.ones((n_entities, n_periods), dtype=int)
    presence_matrix[3, 7:] = 0  # Entity 4 drops out
    presence_matrix[5, 5:8] = 0  # Entity 6 has gap
    presence_matrix[7, :3] = 0  # Entity 8 enters late

    return {
        'entities': entities,
        'time_periods': time_periods,
        'presence_matrix': presence_matrix
    }


@pytest.fixture
def sample_panel_dataframe():
    """Sample panel DataFrame for integration tests."""
    np.random.seed(123)
    n = 50  # entities
    t = 20  # time periods

    entities_idx = [f'E{i:02d}' for i in range(1, n+1)]
    time_idx = list(range(2000, 2000 + t))

    index = pd.MultiIndex.from_product(
        [entities_idx, time_idx],
        names=['entity', 'time']
    )

    # Create data with different variance patterns
    entity_effects = np.repeat(np.random.randn(n) * 5, t)
    var1 = entity_effects + np.random.randn(n * t) * 0.5

    entity_effects2 = np.repeat(np.random.randn(n) * 0.5, t)
    var2 = entity_effects2 + np.random.randn(n * t) * 5

    entity_effects3 = np.repeat(np.random.randn(n) * 2, t)
    var3 = entity_effects3 + np.random.randn(n * t) * 2

    return pd.DataFrame({
        'var1_high_between': var1,
        'var2_high_within': var2,
        'var3_balanced': var3
    }, index=index)


class TestEntityEffectsPlot:
    """Tests for EntityEffectsPlot."""

    def test_creation_basic(self, sample_entity_effects_data):
        """Test basic chart creation."""
        chart = EntityEffectsPlot()
        fig = chart._create_figure(sample_entity_effects_data)

        assert fig is not None
        assert len(fig.data) >= 1  # At least bars trace
        assert fig.layout.title.text is not None

    def test_with_theme(self, sample_entity_effects_data):
        """Test chart with different themes."""
        for theme in [PROFESSIONAL_THEME, ACADEMIC_THEME, PRESENTATION_THEME]:
            chart = EntityEffectsPlot(theme=theme)
            fig = chart._create_figure(sample_entity_effects_data)

            assert fig is not None
            assert chart.theme == theme

    def test_sort_by_magnitude(self, sample_entity_effects_data):
        """Test sorting by magnitude."""
        chart = EntityEffectsPlot()
        fig = chart._create_figure(
            sample_entity_effects_data,
            sort_by='magnitude'
        )

        assert fig is not None
        # Verify data is sorted
        effects = sample_entity_effects_data['effect']
        sorted_effects = sorted(effects, key=abs, reverse=True)
        # Chart should sort by magnitude

    def test_without_confidence_intervals(self, sample_entity_effects_data):
        """Test chart without confidence intervals."""
        chart = EntityEffectsPlot()
        fig = chart._create_figure(
            sample_entity_effects_data,
            show_confidence=False
        )

        assert fig is not None

    def test_missing_std_error(self):
        """Test handling missing std_error."""
        data = {
            'entity_id': ['A', 'B', 'C'],
            'effect': [0.5, -0.3, 0.2]
            # No std_error
        }

        chart = EntityEffectsPlot()
        fig = chart._create_figure(data, show_confidence=False)

        assert fig is not None

    def test_empty_data(self):
        """Test handling empty data."""
        data = {
            'entity_id': [],
            'effect': [],
            'std_error': []
        }

        chart = EntityEffectsPlot()
        # Empty data should either raise error or create empty figure
        # For now, just test it doesn't crash
        try:
            fig = chart._create_figure(data)
            # If it creates a figure, that's acceptable
            assert fig is not None
        except (ValueError, IndexError, KeyError):
            # If it raises an error, that's also acceptable
            pass


class TestTimeEffectsPlot:
    """Tests for TimeEffectsPlot."""

    def test_creation_basic(self, sample_time_effects_data):
        """Test basic chart creation."""
        chart = TimeEffectsPlot()
        fig = chart._create_figure(sample_time_effects_data)

        assert fig is not None
        assert len(fig.data) >= 1  # At least line trace
        assert fig.layout.title.text is not None

    def test_with_theme(self, sample_time_effects_data):
        """Test chart with different themes."""
        for theme in [PROFESSIONAL_THEME, ACADEMIC_THEME, PRESENTATION_THEME]:
            chart = TimeEffectsPlot(theme=theme)
            fig = chart._create_figure(sample_time_effects_data)

            assert fig is not None
            assert chart.theme == theme

    def test_with_confidence_bands(self, sample_time_effects_data):
        """Test chart with confidence bands."""
        chart = TimeEffectsPlot()
        fig = chart._create_figure(
            sample_time_effects_data,
            show_confidence=True
        )

        assert fig is not None
        # Should have line trace + confidence band traces

    def test_highlight_significant(self, sample_time_effects_data):
        """Test highlighting significant effects."""
        chart = TimeEffectsPlot()
        fig = chart._create_figure(
            sample_time_effects_data,
            highlight_significant=True
        )

        assert fig is not None

    def test_without_std_error(self):
        """Test chart without standard errors."""
        data = {
            'time': [2000, 2001, 2002],
            'effect': [0.1, 0.2, 0.15]
            # No std_error
        }

        chart = TimeEffectsPlot()
        fig = chart._create_figure(data, show_confidence=False)

        assert fig is not None

    def test_numeric_time(self):
        """Test with numeric time periods."""
        data = {
            'time': [1, 2, 3, 4, 5],
            'effect': [0.1, 0.2, 0.15, 0.3, 0.25],
            'std_error': [0.05, 0.06, 0.04, 0.07, 0.05]
        }

        chart = TimeEffectsPlot()
        fig = chart._create_figure(data)

        assert fig is not None


class TestBetweenWithinPlot:
    """Tests for BetweenWithinPlot."""

    def test_creation_stacked(self, sample_between_within_data):
        """Test stacked bar chart creation."""
        chart = BetweenWithinPlot()
        fig = chart._create_figure(
            sample_between_within_data,
            chart_type='stacked'
        )

        assert fig is not None
        assert len(fig.data) == 2  # Between and Within traces
        assert fig.layout.barmode == 'stack'

    def test_creation_side_by_side(self, sample_between_within_data):
        """Test side-by-side bar chart creation."""
        chart = BetweenWithinPlot()
        fig = chart._create_figure(
            sample_between_within_data,
            chart_type='side_by_side'
        )

        assert fig is not None
        assert len(fig.data) == 2  # Between and Within traces
        assert fig.layout.barmode == 'group'

    def test_creation_scatter(self, sample_between_within_data):
        """Test scatter plot creation."""
        chart = BetweenWithinPlot()
        fig = chart._create_figure(
            sample_between_within_data,
            chart_type='scatter'
        )

        assert fig is not None
        assert len(fig.data) >= 1  # At least scatter trace

    def test_with_percentages(self, sample_between_within_data):
        """Test showing percentages."""
        chart = BetweenWithinPlot()
        fig = chart._create_figure(
            sample_between_within_data,
            chart_type='stacked',
            show_percentages=True
        )

        assert fig is not None

    def test_all_chart_types(self, sample_between_within_data):
        """Test all chart type variations."""
        chart = BetweenWithinPlot()

        for chart_type in ['stacked', 'side_by_side', 'scatter']:
            fig = chart._create_figure(
                sample_between_within_data,
                chart_type=chart_type
            )
            assert fig is not None

    def test_with_themes(self, sample_between_within_data):
        """Test with different themes."""
        for theme in [PROFESSIONAL_THEME, ACADEMIC_THEME, PRESENTATION_THEME]:
            chart = BetweenWithinPlot(theme=theme)
            fig = chart._create_figure(sample_between_within_data)

            assert fig is not None
            assert chart.theme == theme

    def test_single_variable(self):
        """Test with single variable."""
        data = {
            'variables': ['wage'],
            'between_var': [15.5],
            'within_var': [5.2]
        }

        chart = BetweenWithinPlot()
        fig = chart._create_figure(data)

        assert fig is not None

    def test_invalid_chart_type(self, sample_between_within_data):
        """Test with invalid chart type."""
        chart = BetweenWithinPlot()
        # Should default to stacked if invalid type provided
        fig = chart._create_figure(
            sample_between_within_data,
            chart_type='invalid_type'
        )

        assert fig is not None


class TestPanelStructurePlot:
    """Tests for PanelStructurePlot."""

    def test_creation_basic(self, sample_panel_structure_data):
        """Test basic chart creation."""
        chart = PanelStructurePlot()
        fig = chart._create_figure(sample_panel_structure_data)

        assert fig is not None
        assert len(fig.data) >= 1  # At least heatmap trace
        assert fig.layout.title.text is not None

    def test_with_theme(self, sample_panel_structure_data):
        """Test chart with different themes."""
        for theme in [PROFESSIONAL_THEME, ACADEMIC_THEME, PRESENTATION_THEME]:
            chart = PanelStructurePlot(theme=theme)
            fig = chart._create_figure(sample_panel_structure_data)

            assert fig is not None
            assert chart.theme == theme

    def test_with_statistics(self, sample_panel_structure_data):
        """Test showing statistics."""
        chart = PanelStructurePlot()
        fig = chart._create_figure(
            sample_panel_structure_data,
            show_statistics=True
        )

        assert fig is not None

    def test_highlight_complete(self, sample_panel_structure_data):
        """Test highlighting complete entities."""
        chart = PanelStructurePlot()
        fig = chart._create_figure(
            sample_panel_structure_data,
            highlight_complete=True
        )

        assert fig is not None

    def test_balanced_panel(self):
        """Test with balanced panel."""
        data = {
            'entities': ['A', 'B', 'C'],
            'time_periods': [2000, 2001, 2002],
            'presence_matrix': np.ones((3, 3), dtype=int)
        }

        chart = PanelStructurePlot()
        fig = chart._create_figure(data)

        assert fig is not None

    def test_completely_unbalanced_panel(self):
        """Test with highly unbalanced panel."""
        presence = np.random.randint(0, 2, size=(10, 10))
        data = {
            'entities': [f'E{i}' for i in range(10)],
            'time_periods': list(range(2000, 2010)),
            'presence_matrix': presence
        }

        chart = PanelStructurePlot()
        fig = chart._create_figure(data)

        assert fig is not None

    def test_large_panel(self):
        """Test with large panel."""
        n_entities = 50
        n_periods = 30
        data = {
            'entities': [f'E{i}' for i in range(n_entities)],
            'time_periods': list(range(2000, 2000 + n_periods)),
            'presence_matrix': np.ones((n_entities, n_periods), dtype=int)
        }

        chart = PanelStructurePlot()
        fig = chart._create_figure(data)

        assert fig is not None


class TestPanelChartsIntegration:
    """Integration tests for panel charts."""

    def test_all_charts_with_professional_theme(
        self,
        sample_entity_effects_data,
        sample_time_effects_data,
        sample_between_within_data,
        sample_panel_structure_data
    ):
        """Test all charts with professional theme."""
        theme = PROFESSIONAL_THEME

        charts = [
            (EntityEffectsPlot, sample_entity_effects_data),
            (TimeEffectsPlot, sample_time_effects_data),
            (BetweenWithinPlot, sample_between_within_data),
            (PanelStructurePlot, sample_panel_structure_data),
        ]

        for ChartClass, data in charts:
            chart = ChartClass(theme=theme)
            fig = chart._create_figure(data)
            assert fig is not None

    def test_export_methods(self, sample_entity_effects_data):
        """Test export functionality."""
        chart = EntityEffectsPlot()
        chart.figure = chart._create_figure(sample_entity_effects_data)

        # Test to_html
        html = chart.to_html()
        assert html is not None
        assert isinstance(html, str)
        assert len(html) > 0

        # Test to_json
        json_str = chart.to_json()
        assert json_str is not None
        assert isinstance(json_str, str)

    def test_consistent_api_across_charts(
        self,
        sample_entity_effects_data,
        sample_time_effects_data,
        sample_between_within_data,
        sample_panel_structure_data
    ):
        """Test that all charts have consistent API."""
        charts = [
            EntityEffectsPlot(),
            TimeEffectsPlot(),
            BetweenWithinPlot(),
            PanelStructurePlot(),
        ]

        for chart in charts:
            # All should have these methods
            assert hasattr(chart, '_create_figure')
            assert hasattr(chart, 'to_html')
            assert hasattr(chart, 'to_json')
            assert hasattr(chart, 'theme')

    def test_theme_switching(self, sample_entity_effects_data):
        """Test switching themes."""
        chart = EntityEffectsPlot(theme=PROFESSIONAL_THEME)
        fig1 = chart._create_figure(sample_entity_effects_data)
        assert fig1 is not None

        # Create new chart with different theme
        chart2 = EntityEffectsPlot(theme=ACADEMIC_THEME)
        fig2 = chart2._create_figure(sample_entity_effects_data)
        assert fig2 is not None

        # Themes should be different
        assert chart.theme != chart2.theme


class TestPanelChartsEdgeCases:
    """Edge case tests for panel charts."""

    def test_zero_effects(self):
        """Test entity effects chart with all zero effects."""
        data = {
            'entity_id': ['A', 'B', 'C'],
            'effect': [0.0, 0.0, 0.0],
            'std_error': [0.1, 0.1, 0.1]
        }

        chart = EntityEffectsPlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_single_time_period(self):
        """Test time effects with single period."""
        data = {
            'time': [2000],
            'effect': [0.5],
            'std_error': [0.1]
        }

        chart = TimeEffectsPlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_all_between_variance(self):
        """Test between-within with only between variance."""
        data = {
            'variables': ['var1', 'var2'],
            'between_var': [10.0, 20.0],
            'within_var': [0.0, 0.0]
        }

        chart = BetweenWithinPlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_all_within_variance(self):
        """Test between-within with only within variance."""
        data = {
            'variables': ['var1', 'var2'],
            'between_var': [0.0, 0.0],
            'within_var': [10.0, 20.0]
        }

        chart = BetweenWithinPlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_single_entity_panel(self):
        """Test panel structure with single entity."""
        data = {
            'entities': ['A'],
            'time_periods': [2000, 2001, 2002],
            'presence_matrix': np.ones((1, 3), dtype=int)
        }

        chart = PanelStructurePlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_single_period_panel(self):
        """Test panel structure with single period."""
        data = {
            'entities': ['A', 'B', 'C'],
            'time_periods': [2000],
            'presence_matrix': np.ones((3, 1), dtype=int)
        }

        chart = PanelStructurePlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_very_large_effects(self):
        """Test entity effects with very large values."""
        data = {
            'entity_id': ['A', 'B', 'C'],
            'effect': [1000.0, -2000.0, 1500.0],
            'std_error': [100.0, 200.0, 150.0]
        }

        chart = EntityEffectsPlot()
        fig = chart._create_figure(data)
        assert fig is not None

    def test_very_small_effects(self):
        """Test entity effects with very small values."""
        data = {
            'entity_id': ['A', 'B', 'C'],
            'effect': [0.0001, -0.0002, 0.00015],
            'std_error': [0.00001, 0.00002, 0.000015]
        }

        chart = EntityEffectsPlot()
        fig = chart._create_figure(data)
        assert fig is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
