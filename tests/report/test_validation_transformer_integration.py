"""
Tests for ValidationTransformer integration with new visualization system.

Tests the updated ValidationTransformer that uses the new visualization
system to generate pre-rendered charts.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from panelbox.report.validation_transformer import ValidationTransformer


@pytest.fixture
def mock_validation_report():
    """Create a mock ValidationReport object."""
    report = Mock()

    # Specification tests
    spec_result = Mock()
    spec_result.statistic = 15.234
    spec_result.pvalue = 0.002
    spec_result.df = 3
    spec_result.conclusion = "Reject H0: Use Fixed Effects"
    spec_result.reject_null = True
    spec_result.alpha = 0.05
    spec_result.metadata = {}

    report.specification_tests = {"Hausman Test": spec_result}

    # Serial correlation tests
    serial_result = Mock()
    serial_result.statistic = 2.345
    serial_result.pvalue = 0.128
    serial_result.df = 1
    serial_result.conclusion = "No autocorrelation detected"
    serial_result.reject_null = False
    serial_result.alpha = 0.05
    serial_result.metadata = {}

    report.serial_tests = {"Wooldridge Test": serial_result}

    # Heteroskedasticity tests
    het_result = Mock()
    het_result.statistic = 18.456
    het_result.pvalue = 0.001
    het_result.df = 3
    het_result.conclusion = "Heteroskedasticity detected"
    het_result.reject_null = True
    het_result.alpha = 0.05
    het_result.metadata = {}

    report.het_tests = {"Breusch-Pagan Test": het_result}

    # Cross-sectional dependence tests
    cd_result = Mock()
    cd_result.statistic = 3.789
    cd_result.pvalue = 0.0002
    cd_result.df = None
    cd_result.conclusion = "Cross-sectional dependence detected"
    cd_result.reject_null = True
    cd_result.alpha = 0.05
    cd_result.metadata = {}

    report.cd_tests = {"Pesaran CD Test": cd_result}

    # Model info
    report.model_info = {
        "model_type": "Fixed Effects",
        "formula": "y ~ x1 + x2 + x3",
        "nobs": 1000,
        "n_entities": 100,
        "n_periods": 10,
        "balanced": True,
    }

    return report


class TestValidationTransformerLegacyMode:
    """Tests for legacy mode (use_new_visualization=False)."""

    def test_legacy_mode_returns_data_dicts(self, mock_validation_report):
        """Test that legacy mode returns raw data dictionaries."""
        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=False)

        assert "charts" in data
        charts = data["charts"]

        # Charts should be dictionaries with data, not HTML strings
        assert isinstance(charts["test_overview"], dict)
        assert isinstance(charts["pvalue_distribution"], dict)
        assert isinstance(charts["test_statistics"], dict)

        # Check structure of test_overview
        assert "categories" in charts["test_overview"]
        assert "passed" in charts["test_overview"]
        assert "failed" in charts["test_overview"]

        assert isinstance(charts["test_overview"]["categories"], list)
        assert isinstance(charts["test_overview"]["passed"], list)
        assert isinstance(charts["test_overview"]["failed"], list)

    def test_legacy_mode_chart_data_content(self, mock_validation_report):
        """Test content of legacy mode chart data."""
        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=False)

        charts = data["charts"]

        # Test overview should have 4 categories
        assert len(charts["test_overview"]["categories"]) == 4

        # P-value distribution should have 4 test names and p-values
        assert len(charts["pvalue_distribution"]["test_names"]) == 4
        assert len(charts["pvalue_distribution"]["pvalues"]) == 4

        # Test statistics should have 4 test entries
        assert len(charts["test_statistics"]["test_names"]) == 4
        assert len(charts["test_statistics"]["statistics"]) == 4

    def test_legacy_mode_without_charts(self, mock_validation_report):
        """Test legacy mode with include_charts=False."""
        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=False, use_new_visualization=False)

        assert "charts" not in data
        assert "model_info" in data
        assert "tests" in data
        assert "summary" in data


class TestValidationTransformerNewVisualization:
    """Tests for new visualization mode (use_new_visualization=True)."""

    @patch('panelbox.report.validation_transformer.create_validation_charts')
    def test_new_mode_calls_visualization_system(self, mock_create_charts, mock_validation_report):
        """Test that new mode calls create_validation_charts."""
        # Mock chart objects with to_html method
        mock_chart1 = Mock()
        mock_chart1.to_html.return_value = '<div id="chart1">Chart 1</div>'

        mock_chart2 = Mock()
        mock_chart2.to_html.return_value = '<div id="chart2">Chart 2</div>'

        mock_chart3 = Mock()
        mock_chart3.to_html.return_value = '<div id="chart3">Chart 3</div>'

        mock_create_charts.return_value = {
            'test_overview': mock_chart1,
            'pvalue_distribution': mock_chart2,
            'test_statistics': mock_chart3
        }

        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=True)

        # Verify create_validation_charts was called
        assert mock_create_charts.called
        call_args = mock_create_charts.call_args

        # Check that it was called with correct parameters
        assert call_args[1]['theme'] == 'professional'
        assert call_args[1]['interactive'] is True
        assert call_args[1]['include_html'] is False
        assert 'test_overview' in call_args[1]['charts']
        assert 'pvalue_distribution' in call_args[1]['charts']
        assert 'test_statistics' in call_args[1]['charts']

    @patch('panelbox.report.validation_transformer.create_validation_charts')
    def test_new_mode_returns_html_strings(self, mock_create_charts, mock_validation_report):
        """Test that new mode returns HTML strings, not data dicts."""
        # Mock chart objects
        mock_chart1 = Mock()
        mock_chart1.to_html.return_value = '<div id="test-overview">Overview Chart</div>'

        mock_chart2 = Mock()
        mock_chart2.to_html.return_value = '<div id="pvalue-dist">P-value Chart</div>'

        mock_chart3 = Mock()
        mock_chart3.to_html.return_value = '<div id="test-stats">Statistics Chart</div>'

        mock_create_charts.return_value = {
            'test_overview': mock_chart1,
            'pvalue_distribution': mock_chart2,
            'test_statistics': mock_chart3
        }

        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=True)

        charts = data["charts"]

        # Charts should be HTML strings, not dicts
        assert isinstance(charts["test_overview"], str)
        assert isinstance(charts["pvalue_distribution"], str)
        assert isinstance(charts["test_statistics"], str)

        # Check HTML content
        assert '<div id="test-overview">' in charts["test_overview"]
        assert '<div id="pvalue-dist">' in charts["pvalue_distribution"]
        assert '<div id="test-stats">' in charts["test_statistics"]

    @patch('panelbox.report.validation_transformer.create_validation_charts')
    def test_new_mode_uses_prepare_visualization_data(self, mock_create_charts, mock_validation_report):
        """Test that new mode uses prepare_visualization_data method."""
        # Mock chart objects
        mock_chart = Mock()
        mock_chart.to_html.return_value = '<div>Chart</div>'

        mock_create_charts.return_value = {
            'test_overview': mock_chart,
            'pvalue_distribution': mock_chart,
            'test_statistics': mock_chart
        }

        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=True)

        # Verify that create_validation_charts was called
        assert mock_create_charts.called

        # Get the validation_data argument passed to create_validation_charts
        call_args = mock_create_charts.call_args
        validation_data = call_args[1]['validation_data']

        # Should have structure from prepare_visualization_data
        assert 'tests' in validation_data
        assert 'model_info' in validation_data
        assert isinstance(validation_data['tests'], list)

    def test_new_mode_default_behavior(self, mock_validation_report):
        """Test that new mode is the default."""
        with patch('panelbox.report.validation_transformer.create_validation_charts') as mock_create:
            mock_chart = Mock()
            mock_chart.to_html.return_value = '<div>Chart</div>'
            mock_create.return_value = {
                'test_overview': mock_chart,
                'pvalue_distribution': mock_chart,
                'test_statistics': mock_chart
            }

            transformer = ValidationTransformer(mock_validation_report)
            # Not specifying use_new_visualization should default to True
            data = transformer.transform(include_charts=True)

            # Should have called visualization system
            assert mock_create.called


class TestValidationTransformerFallback:
    """Tests for fallback behavior when visualization system fails."""

    def test_fallback_on_import_error(self, mock_validation_report):
        """Test fallback to legacy mode if visualization module not available."""
        with patch('panelbox.report.validation_transformer.create_validation_charts', side_effect=ImportError):
            transformer = ValidationTransformer(mock_validation_report)

            with pytest.warns(UserWarning, match="New visualization system not available"):
                data = transformer.transform(include_charts=True, use_new_visualization=True)

            # Should fall back to legacy mode - return data dicts
            charts = data["charts"]
            assert isinstance(charts["test_overview"], dict)
            assert "categories" in charts["test_overview"]

    def test_fallback_on_chart_generation_error(self, mock_validation_report):
        """Test fallback if chart generation fails."""
        with patch('panelbox.report.validation_transformer.create_validation_charts') as mock_create:
            mock_create.side_effect = Exception("Chart generation failed")

            transformer = ValidationTransformer(mock_validation_report)

            with pytest.warns(UserWarning, match="Failed to generate charts"):
                data = transformer.transform(include_charts=True, use_new_visualization=True)

            # Should fall back to legacy mode
            charts = data["charts"]
            assert isinstance(charts["test_overview"], dict)

    @patch('panelbox.report.validation_transformer.create_validation_charts')
    def test_fallback_on_to_html_error(self, mock_create_charts, mock_validation_report):
        """Test fallback if to_html() method fails."""
        # Mock chart object with broken to_html
        mock_chart = Mock()
        mock_chart.to_html.side_effect = Exception("HTML conversion failed")

        mock_create_charts.return_value = {
            'test_overview': mock_chart,
            'pvalue_distribution': mock_chart,
            'test_statistics': mock_chart
        }

        transformer = ValidationTransformer(mock_validation_report)

        with pytest.warns(UserWarning, match="Failed to generate charts"):
            data = transformer.transform(include_charts=True, use_new_visualization=True)

        # Should fall back to legacy mode
        charts = data["charts"]
        assert isinstance(charts["test_overview"], dict)


class TestValidationTransformerBackwardCompatibility:
    """Tests for backward compatibility."""

    def test_transform_without_parameters_still_works(self, mock_validation_report):
        """Test that calling transform() without new parameters still works."""
        with patch('panelbox.report.validation_transformer.create_validation_charts') as mock_create:
            mock_chart = Mock()
            mock_chart.to_html.return_value = '<div>Chart</div>'
            mock_create.return_value = {
                'test_overview': mock_chart,
                'pvalue_distribution': mock_chart,
                'test_statistics': mock_chart
            }

            transformer = ValidationTransformer(mock_validation_report)

            # Old code that doesn't specify use_new_visualization
            data = transformer.transform()

            # Should work and use new system by default
            assert "charts" in data
            assert "model_info" in data
            assert "tests" in data

    def test_transform_with_only_include_charts(self, mock_validation_report):
        """Test transform with only include_charts parameter."""
        transformer = ValidationTransformer(mock_validation_report)

        # Should work with just include_charts
        data1 = transformer.transform(include_charts=True)
        data2 = transformer.transform(include_charts=False)

        assert "charts" in data1  # or not, depending on import success
        assert "charts" not in data2

    def test_existing_methods_still_work(self, mock_validation_report):
        """Test that existing public methods still work."""
        transformer = ValidationTransformer(mock_validation_report)

        # prepare_visualization_data() should still work
        viz_data = transformer.prepare_visualization_data()
        assert 'tests' in viz_data
        assert 'model_info' in viz_data

        # to_dict() should still work
        dict_data = transformer.to_dict()
        assert isinstance(dict_data, dict)


class TestPrepareVisualizationDataMethod:
    """Tests for the prepare_visualization_data method."""

    def test_prepare_visualization_data_structure(self, mock_validation_report):
        """Test structure of visualization data."""
        transformer = ValidationTransformer(mock_validation_report)
        viz_data = transformer.prepare_visualization_data()

        assert 'tests' in viz_data
        assert 'model_info' in viz_data
        assert 'charts' in viz_data

        assert isinstance(viz_data['tests'], list)
        assert isinstance(viz_data['model_info'], dict)
        assert isinstance(viz_data['charts'], dict)

    def test_prepare_visualization_data_test_format(self, mock_validation_report):
        """Test format of tests in visualization data."""
        transformer = ValidationTransformer(mock_validation_report)
        viz_data = transformer.prepare_visualization_data()

        tests = viz_data['tests']
        assert len(tests) == 4  # 4 test results

        # Check first test has required fields
        test = tests[0]
        assert 'name' in test
        assert 'category' in test
        assert 'statistic' in test
        assert 'pvalue' in test
        assert 'passed' in test
        assert 'alpha' in test

    def test_prepare_visualization_data_is_used_by_new_mode(self, mock_validation_report):
        """Test that new mode actually uses prepare_visualization_data."""
        with patch('panelbox.report.validation_transformer.create_validation_charts') as mock_create:
            with patch.object(ValidationTransformer, 'prepare_visualization_data') as mock_prepare:
                mock_prepare.return_value = {
                    'tests': [],
                    'model_info': {},
                    'charts': {}
                }

                mock_chart = Mock()
                mock_chart.to_html.return_value = '<div>Chart</div>'
                mock_create.return_value = {
                    'test_overview': mock_chart,
                    'pvalue_distribution': mock_chart,
                    'test_statistics': mock_chart
                }

                transformer = ValidationTransformer(mock_validation_report)
                transformer.transform(include_charts=True, use_new_visualization=True)

                # Verify prepare_visualization_data was called
                assert mock_prepare.called


class TestTransformerIntegration:
    """Integration tests for ValidationTransformer."""

    def test_full_transform_new_mode_integration(self, mock_validation_report):
        """Test full transformation with new visualization system."""
        with patch('panelbox.report.validation_transformer.create_validation_charts') as mock_create:
            mock_chart = Mock()
            mock_chart.to_html.return_value = '<div class="plotly-chart">Interactive Chart</div>'

            mock_create.return_value = {
                'test_overview': mock_chart,
                'pvalue_distribution': mock_chart,
                'test_statistics': mock_chart
            }

            transformer = ValidationTransformer(mock_validation_report)
            data = transformer.transform(include_charts=True, use_new_visualization=True)

            # Check all expected keys
            assert 'model_info' in data
            assert 'tests' in data
            assert 'summary' in data
            assert 'recommendations' in data
            assert 'charts' in data

            # Check model info
            assert data['model_info']['model_type'] == 'Fixed Effects'

            # Check tests
            assert len(data['tests']) > 0

            # Check summary
            assert 'total_tests' in data['summary']
            assert 'pass_rate' in data['summary']

            # Check charts are HTML
            assert '<div class="plotly-chart">' in data['charts']['test_overview']

    def test_full_transform_legacy_mode_integration(self, mock_validation_report):
        """Test full transformation with legacy mode."""
        transformer = ValidationTransformer(mock_validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=False)

        # Check all expected keys
        assert 'model_info' in data
        assert 'tests' in data
        assert 'summary' in data
        assert 'recommendations' in data
        assert 'charts' in data

        # Check charts are dicts
        assert isinstance(data['charts']['test_overview'], dict)
        assert 'categories' in data['charts']['test_overview']

    def test_switching_between_modes(self, mock_validation_report):
        """Test switching between new and legacy modes."""
        with patch('panelbox.report.validation_transformer.create_validation_charts') as mock_create:
            mock_chart = Mock()
            mock_chart.to_html.return_value = '<div>Chart</div>'
            mock_create.return_value = {
                'test_overview': mock_chart,
                'pvalue_distribution': mock_chart,
                'test_statistics': mock_chart
            }

            transformer = ValidationTransformer(mock_validation_report)

            # Use new mode
            data_new = transformer.transform(include_charts=True, use_new_visualization=True)

            # Use legacy mode
            data_legacy = transformer.transform(include_charts=True, use_new_visualization=False)

            # Charts should be different types
            assert isinstance(data_new['charts']['test_overview'], str)
            assert isinstance(data_legacy['charts']['test_overview'], dict)

            # Other data should be the same
            assert data_new['model_info'] == data_legacy['model_info']
            assert len(data_new['tests']) == len(data_legacy['tests'])
