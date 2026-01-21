"""
Tests for ReportManager.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from panelbox.report import ReportManager
from panelbox.report.validation_transformer import ValidationTransformer


class TestReportManager:
    """Test ReportManager functionality."""

    @pytest.fixture
    def report_manager(self):
        """Create ReportManager instance."""
        return ReportManager(enable_cache=False)

    @pytest.fixture
    def sample_validation_data(self):
        """Create sample validation data."""
        return {
            'model_info': {
                'model_type': 'Fixed Effects',
                'formula': 'y ~ x1 + x2',
                'nobs': 1000,
                'n_entities': 100,
                'n_periods': 10
            },
            'tests': [
                {
                    'category': 'Specification',
                    'name': 'Hausman Test',
                    'statistic': 12.5,
                    'statistic_formatted': '12.500',
                    'pvalue': 0.014,
                    'pvalue_formatted': '0.0140',
                    'df': 2,
                    'reject_null': True,
                    'result': 'REJECT',
                    'result_class': 'reject',
                    'conclusion': 'Reject null hypothesis',
                    'significance': '**',
                    'metadata': {}
                }
            ],
            'summary': {
                'total_tests': 1,
                'total_passed': 0,
                'total_failed': 1,
                'pass_rate': 0.0,
                'pass_rate_formatted': '0.0%',
                'failed_by_category': {
                    'specification': 1,
                    'serial': 0,
                    'heteroskedasticity': 0,
                    'cross_sectional': 0
                },
                'overall_status': 'warning',
                'status_message': 'Issues detected',
                'has_issues': True
            },
            'recommendations': [],
            'charts': {
                'test_overview': {
                    'categories': ['Specification'],
                    'passed': [0],
                    'failed': [1]
                },
                'pvalue_distribution': {
                    'test_names': ['Hausman Test'],
                    'pvalues': [0.014]
                },
                'test_statistics': {
                    'test_names': ['Hausman Test'],
                    'statistics': [12.5]
                }
            }
        }

    def test_initialization(self, report_manager):
        """Test ReportManager initialization."""
        assert report_manager is not None
        assert report_manager.template_manager is not None
        assert report_manager.asset_manager is not None
        assert report_manager.css_manager is not None

    def test_get_info(self, report_manager):
        """Test get_info method."""
        info = report_manager.get_info()

        assert 'panelbox_version' in info
        assert 'template_dir' in info
        assert 'asset_dir' in info
        assert 'templates_cached' in info
        assert 'assets_cached' in info
        assert 'css_layers' in info

    def test_generate_validation_report(self, report_manager, sample_validation_data):
        """Test validation report generation."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data,
            interactive=True,
            title='Test Validation Report'
        )

        assert html is not None
        assert isinstance(html, str)
        assert len(html) > 0

        # Check for key elements
        assert 'Test Validation Report' in html
        assert 'Hausman Test' in html
        assert 'DOCTYPE html' in html

    def test_save_report(self, report_manager, sample_validation_data):
        """Test report saving."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data,
            title='Test Report'
        )

        # Create temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_report.html'

            # Save report
            saved_path = report_manager.save_report(html, output_path)

            assert saved_path.exists()
            assert saved_path.stat().st_size > 0

            # Read and verify
            content = saved_path.read_text()
            assert 'Test Report' in content

    def test_save_report_overwrite(self, report_manager, sample_validation_data):
        """Test report saving with overwrite."""
        html = report_manager.generate_validation_report(
            validation_data=sample_validation_data,
            title='Test Report'
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'test_report.html'

            # Save first time
            report_manager.save_report(html, output_path)

            # Should raise error without overwrite
            with pytest.raises(FileExistsError):
                report_manager.save_report(html, output_path, overwrite=False)

            # Should succeed with overwrite
            saved_path = report_manager.save_report(html, output_path, overwrite=True)
            assert saved_path.exists()

    def test_clear_cache(self, report_manager):
        """Test cache clearing."""
        # Generate a report to populate caches
        validation_data = {
            'model_info': {'model_type': 'FE'},
            'tests': [],
            'summary': {'total_tests': 0, 'has_issues': False},
            'recommendations': []
        }

        report_manager.generate_validation_report(validation_data)

        # Clear caches
        report_manager.clear_cache()

        # Verify caches are cleared
        assert len(report_manager.template_manager.template_cache) == 0
        assert len(report_manager.asset_manager.asset_cache) == 0

    def test_repr(self, report_manager):
        """Test string representation."""
        repr_str = repr(report_manager)

        assert 'ReportManager' in repr_str
        assert 'cache=' in repr_str
        assert 'minify=' in repr_str
