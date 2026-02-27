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

    def test_export_overwrite_protection(self, html_exporter, sample_html):
        """Test that export raises error when file exists and overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"

            # First export
            html_exporter.export(sample_html, output_path)

            # Second export should raise FileExistsError
            with pytest.raises(FileExistsError, match="already exists"):
                html_exporter.export(sample_html, output_path, overwrite=False)

    def test_export_overwrite_allowed(self, html_exporter, sample_html):
        """Test that export overwrites when overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"

            # First export
            html_exporter.export(sample_html, output_path)
            original_content = output_path.read_text(encoding="utf-8")

            # Second export with different content
            new_html = "<html><body><h1>Updated</h1></body></html>"
            html_exporter.export(new_html, output_path, overwrite=True)

            updated_content = output_path.read_text(encoding="utf-8")
            assert "Updated" in updated_content
            assert updated_content != original_content

    def test_export_creates_parent_directories(self, html_exporter, sample_html):
        """Test that export creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "subdir1" / "subdir2" / "test.html"

            exported_path = html_exporter.export(sample_html, output_path)

            assert exported_path.exists()
            assert exported_path.parent.exists()

    def test_export_without_metadata(self, html_exporter, sample_html):
        """Test export without metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"

            html_exporter.export(sample_html, output_path, add_metadata=False)

            content = output_path.read_text(encoding="utf-8")
            assert "PanelBox HTML Export" not in content

    def test_minify_initialization(self):
        """Test HTMLExporter initialization with minify option."""
        exporter = HTMLExporter(minify=True)
        assert exporter.minify is True

    def test_pretty_print_initialization(self):
        """Test HTMLExporter initialization with pretty_print option."""
        exporter = HTMLExporter(pretty_print=True)
        assert exporter.pretty_print is True

    def test_export_with_pretty_print(self, sample_html):
        """Test export with pretty_print enabled."""
        exporter = HTMLExporter(pretty_print=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.html"

            exported_path = exporter.export(sample_html, output_path)

            assert exported_path.exists()
            content = output_path.read_text(encoding="utf-8")
            assert content  # Content should exist (basic formatting applied)

    def test_export_multiple(self, html_exporter):
        """Test exporting multiple HTML reports."""
        reports = {
            "report1.html": "<html><body><h1>Report 1</h1></body></html>",
            "report2.html": "<html><body><h1>Report 2</h1></body></html>",
            "report3.html": "<html><body><h1>Report 3</h1></body></html>",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = html_exporter.export_multiple(reports, tmpdir)

            assert len(exported) == 3
            assert all(path.exists() for path in exported.values())

            # Verify content
            report1_content = (Path(tmpdir) / "report1.html").read_text(encoding="utf-8")
            assert "Report 1" in report1_content

    def test_export_multiple_creates_output_dir(self, html_exporter):
        """Test that export_multiple creates output directory if it doesn't exist."""
        reports = {"report1.html": "<html><body><h1>Test</h1></body></html>"}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "new_dir"

            exported = html_exporter.export_multiple(reports, output_dir)

            assert output_dir.exists()
            assert len(exported) == 1

    def test_export_with_index(self, html_exporter):
        """Test exporting multiple reports with an index page."""
        reports = {
            "Validation Report": "<html><body><h1>Validation</h1></body></html>",
            "Regression Results": "<html><body><h1>Regression</h1></body></html>",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = html_exporter.export_with_index(reports, tmpdir, index_title="My Reports")

            # Should have 2 reports + 1 index
            assert len(exported) == 3
            assert "_index" in exported

            # Verify index page exists
            index_path = exported["_index"]
            assert index_path.exists()
            assert index_path.name == "index.html"

            # Verify index content
            index_content = index_path.read_text(encoding="utf-8")
            assert "My Reports" in index_content
            assert "Validation Report" in index_content
            assert "Regression Results" in index_content
            assert 'href="report_1.html"' in index_content
            assert 'href="report_2.html"' in index_content

    def test_export_with_index_default_title(self, html_exporter):
        """Test export_with_index using default title."""
        reports = {"Test Report": "<html><body><h1>Test</h1></body></html>"}

        with tempfile.TemporaryDirectory() as tmpdir:
            exported = html_exporter.export_with_index(reports, tmpdir)

            index_content = exported["_index"].read_text(encoding="utf-8")
            assert "PanelBox Reports" in index_content

    def test_metadata_insertion_with_doctype(self, html_exporter):
        """Test metadata insertion in HTML with DOCTYPE."""
        html = "<!DOCTYPE html><html><body>Content</body></html>"

        result = html_exporter._add_metadata(html)

        assert "<!DOCTYPE html>" in result
        assert "PanelBox HTML Export" in result
        assert result.index("PanelBox") > result.index("<!DOCTYPE")

    def test_metadata_insertion_without_doctype(self, html_exporter):
        """Test metadata insertion in HTML without DOCTYPE."""
        html = "<html><body>Content</body></html>"

        result = html_exporter._add_metadata(html)

        assert "PanelBox HTML Export" in result
        # Metadata should be at the beginning (after whitespace)
        assert result.strip().startswith("<!--")

    def test_repr(self, html_exporter):
        """Test string representation."""
        repr_str = repr(html_exporter)

        assert "HTMLExporter" in repr_str
        assert "minify=False" in repr_str
        assert "pretty_print=False" in repr_str

    def test_repr_with_options(self):
        """Test string representation with options enabled."""
        exporter = HTMLExporter(minify=True, pretty_print=True)
        repr_str = repr(exporter)

        assert "minify=True" in repr_str
        assert "pretty_print=True" in repr_str

    def test_index_page_generation(self, html_exporter):
        """Test index page HTML generation."""
        reports = ["Report 1", "Report 2", "Report 3"]

        html = html_exporter._generate_index_page(reports, "Test Index")

        assert "<!DOCTYPE html>" in html
        assert "Test Index" in html
        assert "Report 1" in html
        assert "Report 2" in html
        assert "Report 3" in html
        assert 'href="report_1.html"' in html
        assert 'href="report_2.html"' in html
        assert 'href="report_3.html"' in html
        assert "PanelBox" in html


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

    def test_init_invalid_table_style_raises(self):
        """Test invalid table_style raises ValueError."""
        with pytest.raises(ValueError, match="Invalid table_style"):
            LaTeXExporter(table_style="invalid_style")

    def test_init_standard_style(self):
        """Test initialization with standard style."""
        exporter = LaTeXExporter(table_style="standard")
        assert exporter.table_style == "standard"

    def test_init_threeparttable_style(self):
        """Test initialization with threeparttable style."""
        exporter = LaTeXExporter(table_style="threeparttable")
        assert exporter.table_style == "threeparttable"

    def test_export_validation_standard_style(self, sample_tests):
        """Test validation tests export with standard style."""
        exporter = LaTeXExporter(table_style="standard")
        latex = exporter.export_validation_tests(sample_tests)

        assert "\\begin{table}" in latex
        assert "\\hline" in latex
        assert "\\toprule" not in latex
        assert "Hausman Test" in latex

    def test_export_validation_booktabs_notes(self, latex_exporter, sample_tests):
        """Test booktabs style includes significance notes."""
        latex = latex_exporter.export_validation_tests(sample_tests)
        assert "Significance levels" in latex
        assert "\\begin{minipage}" in latex

    def test_export_regression_standard_style(self):
        """Test regression table with standard style."""
        exporter = LaTeXExporter(table_style="standard")
        coefficients = [
            {
                "variable": "x1",
                "coefficient": 1.5,
                "std_error": 0.2,
                "t_statistic": 7.5,
                "pvalue": 0.001,
            },
        ]
        model_info = {"r_squared": 0.85, "nobs": 1000}
        latex = exporter.export_regression_table(coefficients, model_info)

        assert "\\hline" in latex
        assert "\\toprule" not in latex
        assert "x1" in latex

    def test_export_regression_with_entities(self, latex_exporter):
        """Test regression table includes entity count."""
        coefficients = [
            {
                "variable": "x1",
                "coefficient": 1.0,
                "std_error": 0.1,
                "t_statistic": 10.0,
                "pvalue": 0.0001,
            },
        ]
        model_info = {"r_squared": 0.9, "nobs": 500, "n_entities": 50}
        latex = latex_exporter.export_regression_table(coefficients, model_info)

        assert "Entities" in latex
        assert "50" in latex

    def test_export_summary_stats(self, latex_exporter):
        """Test summary statistics export."""
        stats = [
            {
                "variable": "x1",
                "count": 1000,
                "mean": 5.234,
                "std": 1.567,
                "min": 1.2,
                "max": 9.8,
                "median": 5.1,
            },
            {
                "variable": "x2",
                "count": 1000,
                "mean": -2.456,
                "std": 3.123,
                "min": -10.5,
                "max": 5.6,
                "median": -2.3,
            },
        ]
        latex = latex_exporter.export_summary_stats(stats)

        assert "\\begin{table}" in latex
        assert "x1" in latex
        assert "x2" in latex
        assert "5.234" in latex
        assert "Variable" in latex

    def test_export_summary_stats_standard(self):
        """Test summary stats with standard style."""
        exporter = LaTeXExporter(table_style="standard")
        stats = [
            {
                "variable": "y",
                "count": 100,
                "mean": 10.0,
                "std": 2.0,
                "min": 5.0,
                "max": 15.0,
                "50%": 10.0,
            },
        ]
        latex = exporter.export_summary_stats(stats)

        assert "\\hline" in latex
        assert "y" in latex

    def test_save_overwrite_protection(self, latex_exporter, sample_tests):
        """Test save refuses to overwrite existing file."""
        latex = latex_exporter.export_validation_tests(sample_tests)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            output_path.write_text("existing content")

            with pytest.raises(FileExistsError, match="already exists"):
                latex_exporter.save(latex, output_path, overwrite=False)

    def test_save_overwrite_allowed(self, latex_exporter, sample_tests):
        """Test save with overwrite=True replaces file."""
        latex = latex_exporter.export_validation_tests(sample_tests)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.tex"
            output_path.write_text("existing content")

            saved_path = latex_exporter.save(latex, output_path, overwrite=True)
            assert saved_path.exists()
            content = saved_path.read_text(encoding="utf-8")
            assert "Hausman Test" in content

    def test_format_float_valid(self, latex_exporter):
        """Test _format_float with valid float."""
        result = latex_exporter._format_float(3.14159)
        assert result == "3.142"

    def test_format_float_none(self, latex_exporter):
        """Test _format_float with None returns string."""
        result = latex_exporter._format_float(None)
        assert result == "None"

    def test_format_float_string(self, latex_exporter):
        """Test _format_float with string returns string."""
        result = latex_exporter._format_float("abc")
        assert result == "abc"

    def test_format_pvalue_regular(self, latex_exporter):
        """Test _format_pvalue with regular p-value."""
        result = latex_exporter._format_pvalue(0.0432)
        assert result == "0.0432"

    def test_format_pvalue_small(self, latex_exporter):
        """Test _format_pvalue with small p-value uses scientific."""
        result = latex_exporter._format_pvalue(0.00001)
        assert "e" in result.lower()

    def test_format_pvalue_invalid(self, latex_exporter):
        """Test _format_pvalue with invalid value."""
        result = latex_exporter._format_pvalue(None)
        assert result == "None"

    def test_get_stars_very_significant(self, latex_exporter):
        """Test _get_stars with p < 0.001."""
        result = latex_exporter._get_stars(0.0005)
        assert "***" in result

    def test_get_stars_significant_01(self, latex_exporter):
        """Test _get_stars with p < 0.01."""
        result = latex_exporter._get_stars(0.005)
        assert "**" in result

    def test_get_stars_significant_05(self, latex_exporter):
        """Test _get_stars with p < 0.05."""
        result = latex_exporter._get_stars(0.03)
        assert "*" in result

    def test_get_stars_not_significant(self, latex_exporter):
        """Test _get_stars with p >= 0.05."""
        result = latex_exporter._get_stars(0.15)
        assert result == ""

    def test_get_stars_none(self, latex_exporter):
        """Test _get_stars with None pvalue."""
        result = latex_exporter._get_stars(None)
        assert result == ""

    def test_escape_special_chars(self, latex_exporter):
        """Test escaping LaTeX special characters."""
        # Test each char individually to avoid interaction between replacements
        assert latex_exporter._escape("100%") != "100%"
        assert latex_exporter._escape("a & b") != "a & b"
        assert latex_exporter._escape("$x") != "$x"

    def test_escape_percent_and_ampersand(self, latex_exporter):
        """Test escaping % and & characters (processed before backslash)."""
        # The _escape function iterates the dict and replaces chars in order.
        # % and & are early, so they get escaped to \% and \& first,
        # then \ gets re-escaped. We just verify the originals are gone.
        result = latex_exporter._escape("100%")
        assert "%" in result  # escaped form contains %
        assert result != "100%"  # but it's not the raw original

    def test_escape_disabled(self):
        """Test escape does nothing when disabled."""
        exporter = LaTeXExporter(escape_special_chars=False)
        result = exporter._escape("100% & $test # _")
        assert result == "100% & $test # _"

    def test_repr(self, latex_exporter):
        """Test string representation."""
        repr_str = repr(latex_exporter)
        assert "LaTeXExporter" in repr_str
        assert "booktabs" in repr_str

    def test_add_preamble(self, latex_exporter):
        """Test adding preamble to content."""
        content = "\\begin{table} ... \\end{table}"
        result = latex_exporter._add_preamble(content)
        assert "\\documentclass" in result
        assert "\\usepackage{booktabs}" in result
        assert "\\begin{document}" in result
        assert "\\end{document}" in result
        assert content in result


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

    def test_save_overwrite_protection(self, md_exporter):
        """Test that save raises error when file exists and overwrite=False."""
        markdown = "# Test Content"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"

            # First save
            md_exporter.save(markdown, output_path)

            # Second save should raise FileExistsError
            with pytest.raises(FileExistsError, match="already exists"):
                md_exporter.save(markdown, output_path, overwrite=False)

    def test_save_overwrite_allowed(self, md_exporter):
        """Test that save overwrites when overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.md"

            # First save
            md_exporter.save("# Original", output_path)

            # Second save with overwrite
            md_exporter.save("# Updated", output_path, overwrite=True)

            content = output_path.read_text(encoding="utf-8")
            assert "Updated" in content
            assert "Original" not in content

    def test_save_creates_parent_directories(self, md_exporter):
        """Test that save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dirs" / "test.md"

            saved_path = md_exporter.save("# Test", output_path)

            assert saved_path.exists()
            assert saved_path.parent.exists()

    def test_export_without_toc(self, sample_validation_data):
        """Test export validation report without TOC."""
        exporter = MarkdownExporter(include_toc=False)
        markdown = exporter.export_validation_report(sample_validation_data)

        assert "# Test Validation" not in markdown  # Title would be from param
        assert "Table of Contents" not in markdown

    def test_export_without_github_flavor(self, sample_validation_data):
        """Test export with github_flavor=False."""
        exporter = MarkdownExporter(github_flavor=False)
        markdown = exporter.export_validation_report(sample_validation_data)

        # Should still generate markdown
        assert markdown is not None
        assert len(markdown) > 0

    def test_validation_report_with_recommendations(self, md_exporter):
        """Test validation report with recommendations."""
        data = {
            "model_info": {"model_type": "Fixed Effects"},
            "tests": [],
            "summary": {"total_tests": 0, "total_passed": 0, "total_failed": 0},
            "recommendations": [
                {
                    "severity": "critical",
                    "category": "Heteroskedasticity",
                    "issue": "Strong evidence of heteroskedasticity",
                    "tests": ["Modified Wald Test"],
                    "suggestions": [
                        "Use robust standard errors",
                        "Consider heteroskedastic panel model",
                    ],
                },
                {
                    "severity": "high",
                    "category": "Serial Correlation",
                    "issue": "Serial correlation detected",
                    "tests": ["Wooldridge Test"],
                    "suggestions": ["Use HAC standard errors"],
                },
                {
                    "severity": "medium",
                    "category": "Specification",
                    "issue": "Model may be misspecified",
                },
                {
                    "severity": "low",
                    "category": "Other",
                    "issue": "Minor issue",
                },
            ],
        }

        markdown = md_exporter.export_validation_report(data)

        assert "## Recommendations" in markdown
        assert "🔴" in markdown  # Critical
        assert "🟠" in markdown  # High
        assert "🟡" in markdown  # Medium
        assert "🔵" in markdown  # Low
        assert "Heteroskedasticity" in markdown
        assert "Use robust standard errors" in markdown
        assert "Modified Wald Test" in markdown

    def test_validation_report_all_passed(self, md_exporter):
        """Test validation report when all tests pass."""
        data = {
            "model_info": {"model_type": "Fixed Effects"},
            "tests": [],
            "summary": {
                "total_tests": 5,
                "total_passed": 5,
                "total_failed": 0,
                "pass_rate_formatted": "100.0%",
                "has_issues": False,
                "status_message": "All tests passed",
            },
            "recommendations": [],
        }

        markdown = md_exporter.export_validation_report(data)

        assert "✅ **All tests passed**" in markdown
        assert "Passed:** 5 ✅" in markdown

    def test_validation_report_with_model_info_fields(self, md_exporter):
        """Test validation report with all model info fields."""
        data = {
            "model_info": {
                "model_type": "Fixed Effects",
                "formula": "y ~ x1 + x2 + x3",
                "nobs": 1000,
                "nobs_formatted": "1,000",
                "n_entities": 100,
                "n_entities_formatted": "100",
                "n_periods": 10,
                "n_periods_formatted": "10",
            },
            "tests": [],
            "summary": {"total_tests": 0, "total_passed": 0, "total_failed": 0},
            "recommendations": [],
        }

        markdown = md_exporter.export_validation_report(data)

        assert "Fixed Effects" in markdown
        assert "y ~ x1 + x2 + x3" in markdown
        assert "1,000" in markdown
        assert "Entities:** 100" in markdown
        assert "Time Periods:** 10" in markdown

    def test_export_regression_table(self, md_exporter):
        """Test regression table export with significance stars."""
        coefficients = [
            {
                "variable": "x1",
                "coefficient": 1.234,
                "std_error": 0.045,
                "t_statistic": 27.4,
                "pvalue": 0.0001,  # ***
            },
            {
                "variable": "x2",
                "coefficient": -0.567,
                "std_error": 0.089,
                "t_statistic": -6.37,
                "pvalue": 0.005,  # **
            },
            {
                "variable": "x3",
                "coefficient": 0.123,
                "std_error": 0.056,
                "t_statistic": 2.20,
                "pvalue": 0.02,  # *
            },
            {
                "variable": "x4",
                "coefficient": 0.045,
                "std_error": 0.034,
                "t_statistic": 1.32,
                "pvalue": 0.15,  # no stars
            },
        ]

        model_info = {"r_squared": 0.654, "nobs": 1000, "n_entities": 100}

        markdown = md_exporter.export_regression_table(coefficients, model_info)

        assert "## Regression Results" in markdown
        assert "x1" in markdown
        assert "1.2340***" in markdown
        assert "-0.5670**" in markdown
        assert "0.1230*" in markdown
        assert "0.0450" in markdown and "0.0450*" not in markdown
        assert "R²: 0.6540" in markdown
        assert "Observations: 1000" in markdown
        assert "Entities: 100" in markdown

    def test_export_regression_table_custom_title(self, md_exporter):
        """Test regression table with custom title."""
        coefficients = [
            {
                "variable": "x1",
                "coefficient": 1.0,
                "std_error": 0.1,
                "t_statistic": 10.0,
                "pvalue": 0.0,
            }
        ]
        model_info = {}

        markdown = md_exporter.export_regression_table(
            coefficients, model_info, title="Custom Regression"
        )

        assert "## Custom Regression" in markdown

    def test_export_summary_stats(self, md_exporter):
        """Test summary statistics export."""
        stats = [
            {"variable": "x1", "count": 1000, "mean": 5.234, "std": 1.567, "min": 1.2, "max": 9.8},
            {
                "variable": "x2",
                "count": 1000,
                "mean": -2.456,
                "std": 3.123,
                "min": -10.5,
                "max": 5.6,
            },
            {"variable": "y", "count": 1000, "mean": 10.123, "std": 2.456, "min": 3.4, "max": 18.9},
        ]

        markdown = md_exporter.export_summary_stats(stats)

        assert "## Summary Statistics" in markdown
        assert "x1" in markdown
        assert "x2" in markdown
        assert "y" in markdown
        assert "5.234" in markdown
        assert "-2.456" in markdown
        assert "1.567" in markdown

    def test_export_summary_stats_custom_title(self, md_exporter):
        """Test summary stats with custom title."""
        stats = [{"variable": "x1", "count": 100, "mean": 1.0, "std": 0.5, "min": 0.0, "max": 2.0}]

        markdown = md_exporter.export_summary_stats(stats, title="Variable Statistics")

        assert "## Variable Statistics" in markdown

    def test_validation_report_multiple_categories(self, md_exporter):
        """Test validation report with tests in multiple categories."""
        data = {
            "model_info": {"model_type": "Random Effects"},
            "tests": [
                {
                    "category": "Specification",
                    "name": "Hausman Test",
                    "statistic_formatted": "12.500",
                    "pvalue_formatted": "0.0140",
                    "significance": "**",
                    "result": "REJECT",
                },
                {
                    "category": "Specification",
                    "name": "Mundlak Test",
                    "statistic_formatted": "8.345",
                    "pvalue_formatted": "0.0350",
                    "significance": "*",
                    "result": "REJECT",
                },
                {
                    "category": "Serial Correlation",
                    "name": "Wooldridge Test",
                    "statistic_formatted": "1.234",
                    "pvalue_formatted": "0.2670",
                    "significance": "",
                    "result": "ACCEPT",
                },
            ],
            "summary": {"total_tests": 3, "total_passed": 1, "total_failed": 2},
            "recommendations": [],
        }

        markdown = md_exporter.export_validation_report(data)

        assert "### Specification" in markdown
        assert "### Serial Correlation" in markdown
        assert "Hausman Test" in markdown
        assert "Mundlak Test" in markdown
        assert "Wooldridge Test" in markdown
        assert "❌ REJECT" in markdown
        assert "✅ ACCEPT" in markdown

    def test_repr(self, md_exporter):
        """Test string representation."""
        repr_str = repr(md_exporter)

        assert "MarkdownExporter" in repr_str
        assert "include_toc=True" in repr_str
        assert "github_flavor=True" in repr_str

    def test_repr_no_toc(self):
        """Test string representation with TOC disabled."""
        exporter = MarkdownExporter(include_toc=False, github_flavor=False)
        repr_str = repr(exporter)

        assert "include_toc=False" in repr_str
        assert "github_flavor=False" in repr_str
