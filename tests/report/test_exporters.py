"""
Tests for Report Exporters.
"""

import tempfile
from pathlib import Path

import pytest

from panelbox.report.exporters import HTMLExporter, LaTeXExporter, MarkdownExporter


class TestHTMLExporter:
    """Test HTMLExporter functionality."""

    @pytest.fixture
    def html_exporter(self):
        """Create HTMLExporter instance."""
        return HTMLExporter()

    @pytest.fixture
    def sample_html(self):
        """Sample HTML content."""
        return """
<!DOCTYPE html>
<html>
<head><title>Test Report</title></head>
<body><h1>Test Report</h1></body>
</html>
"""

    def test_initialization(self, html_exporter):
        """Test HTMLExporter initialization."""
        assert html_exporter is not None
        assert html_exporter.minify is False
        assert html_exporter.pretty_print is False

    def test_export(self, html_exporter, sample_html):
        """Test HTML export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"

            exported_path = html_exporter.export(sample_html, output_path)

            assert exported_path.exists()
            assert exported_path.read_text(encoding="utf-8")

    def test_export_with_metadata(self, html_exporter, sample_html):
        """Test export with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"

            html_exporter.export(sample_html, output_path, add_metadata=True)

            content = output_path.read_text(encoding="utf-8")
            assert "PanelBox HTML Export" in content
            assert "Exported:" in content

    def test_get_file_size(self, html_exporter, sample_html):
        """Test file size estimation."""
        sizes = html_exporter.get_file_size(sample_html)

        assert "bytes" in sizes
        assert "kb" in sizes
        assert "mb" in sizes
        assert sizes["bytes"] > 0
        assert sizes["kb"] > 0


class TestLaTeXExporter:
    """Test LaTeXExporter functionality."""

    @pytest.fixture
    def latex_exporter(self):
        """Create LaTeXExporter instance."""
        return LaTeXExporter(table_style="booktabs")

    @pytest.fixture
    def sample_tests(self):
        """Sample test data."""
        return [
            {
                "category": "Specification",
                "name": "Hausman Test",
                "statistic": 12.5,
                "pvalue": 0.014,
                "df": 2,
                "result": "REJECT",
                "significance": "**",
            },
            {
                "category": "Serial Correlation",
                "name": "Wooldridge Test",
                "statistic": 3.21,
                "pvalue": 0.073,
                "df": 1,
                "result": "ACCEPT",
                "significance": ".",
            },
        ]

    def test_initialization(self, latex_exporter):
        """Test LaTeXExporter initialization."""
        assert latex_exporter is not None
        assert latex_exporter.table_style == "booktabs"
        assert latex_exporter.float_format == ".3f"

    def test_export_validation_tests(self, latex_exporter, sample_tests):
        """Test validation tests export."""
        latex = latex_exporter.export_validation_tests(
            sample_tests, caption="Test Results", label="tab:test"
        )

        assert latex is not None
        assert "\\begin{table}" in latex
        assert "\\end{table}" in latex
        assert "Hausman Test" in latex
        assert "Wooldridge Test" in latex
        assert "\\toprule" in latex
        assert "\\midrule" in latex
        assert "\\bottomrule" in latex

    def test_export_regression_table(self, latex_exporter):
        """Test regression table export."""
        coefficients = [
            {
                "variable": "x1",
                "coefficient": 1.234,
                "std_error": 0.045,
                "t_statistic": 27.4,
                "pvalue": 0.000,
            },
            {
                "variable": "x2",
                "coefficient": -0.567,
                "std_error": 0.089,
                "t_statistic": -6.37,
                "pvalue": 0.012,
            },
        ]

        model_info = {"r_squared": 0.654, "nobs": 1000, "n_entities": 100}

        latex = latex_exporter.export_regression_table(
            coefficients, model_info, caption="Regression Results"
        )

        assert latex is not None
        assert "x1" in latex
        assert "x2" in latex
        assert "1.234" in latex
        assert "R²" in latex

    def test_save(self, latex_exporter, sample_tests):
        """Test LaTeX save."""
        latex = latex_exporter.export_validation_tests(sample_tests)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"

            saved_path = latex_exporter.save(latex, output_path)

            assert saved_path.exists()
            content = saved_path.read_text(encoding="utf-8")
            assert "Hausman Test" in content

    def test_save_with_preamble(self, latex_exporter, sample_tests):
        """Test save with preamble."""
        latex = latex_exporter.export_validation_tests(sample_tests)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"

            latex_exporter.save(latex, output_path, add_preamble=True)

            content = output_path.read_text(encoding="utf-8")
            assert "\\documentclass" in content
            assert "\\begin{document}" in content
            assert "\\end{document}" in content


class TestMarkdownExporter:
    """Test MarkdownExporter functionality."""

    @pytest.fixture
    def md_exporter(self):
        """Create MarkdownExporter instance."""
        return MarkdownExporter(include_toc=True, github_flavor=True)

    @pytest.fixture
    def sample_validation_data(self):
        """Sample validation data."""
        return {
            "model_info": {
                "model_type": "Fixed Effects",
                "formula": "y ~ x1 + x2",
                "nobs": 1000,
                "n_entities": 100,
            },
            "tests": [
                {
                    "category": "Specification",
                    "name": "Hausman Test",
                    "statistic": 12.5,
                    "statistic_formatted": "12.500",
                    "pvalue": 0.014,
                    "pvalue_formatted": "0.0140",
                    "df": 2,
                    "result": "REJECT",
                    "result_class": "reject",
                    "significance": "**",
                }
            ],
            "summary": {
                "total_tests": 1,
                "total_passed": 0,
                "total_failed": 1,
                "pass_rate": 0.0,
                "pass_rate_formatted": "0.0%",
                "has_issues": True,
                "status_message": "Issues detected",
            },
            "recommendations": [],
        }

    def test_initialization(self, md_exporter):
        """Test MarkdownExporter initialization."""
        assert md_exporter is not None
        assert md_exporter.include_toc is True
        assert md_exporter.github_flavor is True

    def test_export_validation_report(self, md_exporter, sample_validation_data):
        """Test validation report export."""
        markdown = md_exporter.export_validation_report(
            sample_validation_data, title="Test Validation"
        )

        assert markdown is not None
        assert "# Test Validation" in markdown
        assert "Hausman Test" in markdown
        assert "## Summary" in markdown
        assert "## Test Results" in markdown

    def test_export_validation_tests(self, md_exporter):
        """Test validation tests export."""
        tests = [
            {
                "category": "Specification",
                "name": "Hausman Test",
                "statistic_formatted": "12.500",
                "pvalue_formatted": "0.0140",
                "df": 2,
                "result": "REJECT",
                "significance": "**",
            }
        ]

        markdown = md_exporter.export_validation_tests(tests)

        assert markdown is not None
        assert "| Category | Test |" in markdown
        assert "Hausman Test" in markdown
        assert "❌" in markdown or "REJECT" in markdown

    def test_save(self, md_exporter, sample_validation_data):
        """Test Markdown save."""
        markdown = md_exporter.export_validation_report(sample_validation_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"

            saved_path = md_exporter.save(markdown, output_path)

            assert saved_path.exists()
            content = saved_path.read_text(encoding="utf-8")
            assert "Hausman Test" in content
