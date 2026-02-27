"""
Tests for TemplateManager.
"""

import pytest

from panelbox.report.template_manager import TemplateManager


class TestTemplateManagerInit:
    """Test TemplateManager initialization."""

    def test_init_default(self):
        """Test default initialization uses package templates."""
        tm = TemplateManager()
        assert tm.template_dir.exists()
        assert tm.enable_cache is True
        assert tm.template_cache == {}

    def test_init_cache_disabled(self):
        """Test initialization with cache disabled."""
        tm = TemplateManager(enable_cache=False)
        assert tm.enable_cache is False

    def test_init_invalid_dir_raises(self):
        """Test that invalid template dir raises ValueError."""
        with pytest.raises(ValueError, match="does not exist"):
            TemplateManager(template_dir="/nonexistent/path")

    def test_init_custom_dir(self, tmp_path):
        """Test initialization with custom template directory."""
        tm = TemplateManager(template_dir=tmp_path)
        assert tm.template_dir == tmp_path

    def test_init_registers_filters(self):
        """Test that custom filters are registered."""
        tm = TemplateManager()
        assert "number_format" in tm.env.filters
        assert "pvalue_format" in tm.env.filters
        assert "percentage" in tm.env.filters
        assert "significance_stars" in tm.env.filters
        assert "round" in tm.env.filters

    def test_init_registers_globals(self):
        """Test that custom globals are registered."""
        tm = TemplateManager()
        assert "now" in tm.env.globals
        assert "range" in tm.env.globals
        assert "len" in tm.env.globals
        assert "enumerate" in tm.env.globals
        assert "zip" in tm.env.globals


class TestTemplateManagerCaching:
    """Test template caching."""

    def test_template_caching_enabled(self):
        """Test that templates are cached when enabled."""
        tm = TemplateManager(enable_cache=True)
        t1 = tm.get_template("validation/interactive/index.html")
        t2 = tm.get_template("validation/interactive/index.html")
        assert t1 is t2
        assert "validation/interactive/index.html" in tm.template_cache

    def test_template_caching_disabled(self):
        """Test that templates are not cached when disabled."""
        tm = TemplateManager(enable_cache=False)
        tm.get_template("validation/interactive/index.html")
        assert len(tm.template_cache) == 0

    def test_clear_cache(self):
        """Test clearing template cache."""
        tm = TemplateManager(enable_cache=True)
        tm.get_template("validation/interactive/index.html")
        assert len(tm.template_cache) > 0
        tm.clear_cache()
        assert len(tm.template_cache) == 0


class TestTemplateManagerRender:
    """Test template rendering."""

    def test_render_string(self):
        """Test rendering a template from string."""
        tm = TemplateManager()
        result = tm.render_string("Hello {{ name }}!", {"name": "World"})
        assert result == "Hello World!"

    def test_render_string_with_filter(self):
        """Test rendering a template string with custom filters."""
        tm = TemplateManager()
        result = tm.render_string("{{ value|number_format(2) }}", {"value": 3.14159})
        assert result == "3.14"

    def test_render_template(self):
        """Test rendering a template from file."""
        tm = TemplateManager()
        # Use a known template
        html = tm.render_template(
            "validation/interactive/index.html",
            {
                "report_title": "Test Report",
                "tests": [],
                "summary": {
                    "total_tests": 0,
                    "total_passed": 0,
                    "total_failed": 0,
                    "pass_rate": 100.0,
                    "pass_rate_formatted": "100.0%",
                    "has_issues": False,
                    "overall_status": "excellent",
                    "status_message": "OK",
                },
                "recommendations": [],
                "charts": {},
            },
        )
        assert isinstance(html, str)
        assert "Test Report" in html


class TestTemplateManagerFilters:
    """Test custom Jinja2 filters."""

    def test_filter_number_format_basic(self):
        """Test number_format filter with basic float."""
        result = TemplateManager._filter_number_format(3.14159)
        assert result == "3.142"

    def test_filter_number_format_decimals(self):
        """Test number_format filter with custom decimals."""
        result = TemplateManager._filter_number_format(3.14159, decimals=2)
        assert result == "3.14"

    def test_filter_number_format_nan(self):
        """Test number_format filter with NaN."""
        result = TemplateManager._filter_number_format(float("nan"))
        assert result == "N/A"

    def test_filter_number_format_none(self):
        """Test number_format filter with None."""
        result = TemplateManager._filter_number_format(None)
        assert result == "N/A"

    def test_filter_number_format_invalid(self):
        """Test number_format filter with non-numeric string."""
        result = TemplateManager._filter_number_format("not_a_number")
        assert result == "not_a_number"

    def test_filter_pvalue_format_regular(self):
        """Test pvalue_format filter with regular p-value."""
        result = TemplateManager._filter_pvalue_format(0.0432)
        assert result == "0.0432"

    def test_filter_pvalue_format_small(self):
        """Test pvalue_format filter with small p-value uses scientific."""
        result = TemplateManager._filter_pvalue_format(0.00001)
        assert "e" in result.lower()

    def test_filter_pvalue_format_nan(self):
        """Test pvalue_format filter with NaN."""
        result = TemplateManager._filter_pvalue_format(float("nan"))
        assert result == "N/A"

    def test_filter_pvalue_format_none(self):
        """Test pvalue_format filter with None."""
        result = TemplateManager._filter_pvalue_format(None)
        assert result == "N/A"

    def test_filter_pvalue_format_invalid(self):
        """Test pvalue_format filter with invalid input."""
        result = TemplateManager._filter_pvalue_format("abc")
        assert result == "abc"

    def test_filter_percentage_basic(self):
        """Test percentage filter with basic value."""
        result = TemplateManager._filter_percentage(0.1234)
        assert result == "12.34%"

    def test_filter_percentage_custom_decimals(self):
        """Test percentage filter with custom decimals."""
        result = TemplateManager._filter_percentage(0.1234, decimals=1)
        assert result == "12.3%"

    def test_filter_percentage_nan(self):
        """Test percentage filter with NaN."""
        result = TemplateManager._filter_percentage(float("nan"))
        assert result == "N/A"

    def test_filter_percentage_none(self):
        """Test percentage filter with None."""
        result = TemplateManager._filter_percentage(None)
        assert result == "N/A"

    def test_filter_percentage_invalid(self):
        """Test percentage filter with invalid input."""
        result = TemplateManager._filter_percentage("abc")
        assert result == "abc"

    def test_filter_significance_stars_very_significant(self):
        """Test significance_stars filter with p < 0.001."""
        assert TemplateManager._filter_significance_stars(0.0005) == "***"

    def test_filter_significance_stars_significant_01(self):
        """Test significance_stars filter with p < 0.01."""
        assert TemplateManager._filter_significance_stars(0.005) == "**"

    def test_filter_significance_stars_significant_05(self):
        """Test significance_stars filter with p < 0.05."""
        assert TemplateManager._filter_significance_stars(0.03) == "*"

    def test_filter_significance_stars_marginal(self):
        """Test significance_stars filter with p < 0.1."""
        assert TemplateManager._filter_significance_stars(0.08) == "."

    def test_filter_significance_stars_not_significant(self):
        """Test significance_stars filter with p >= 0.1."""
        assert TemplateManager._filter_significance_stars(0.15) == ""

    def test_filter_significance_stars_none(self):
        """Test significance_stars filter with None."""
        assert TemplateManager._filter_significance_stars(None) == ""

    def test_filter_significance_stars_invalid(self):
        """Test significance_stars filter with invalid input."""
        assert TemplateManager._filter_significance_stars("abc") == ""

    def test_filter_round_basic(self):
        """Test round filter with basic value."""
        result = TemplateManager._filter_round(3.14159, decimals=2)
        assert result == 3.14

    def test_filter_round_invalid_value(self):
        """Test round filter with invalid input returns original."""
        result = TemplateManager._filter_round("not_a_number")
        assert result == "not_a_number"

    def test_filter_round_zero_decimals(self):
        """Test round filter with zero decimals."""
        result = TemplateManager._filter_round(3.7)
        assert result == 4.0


class TestTemplateManagerList:
    """Test template listing and existence checks."""

    def test_list_templates(self):
        """Test listing available templates."""
        tm = TemplateManager()
        templates = tm.list_templates()
        assert isinstance(templates, list)
        assert len(templates) > 0
        # Should find at least the validation template
        assert any("index.html" in t for t in templates)

    def test_list_templates_sorted(self):
        """Test that listed templates are sorted."""
        tm = TemplateManager()
        templates = tm.list_templates()
        assert templates == sorted(templates)

    def test_template_exists_true(self):
        """Test template_exists returns True for existing template."""
        tm = TemplateManager()
        assert tm.template_exists("validation/interactive/index.html") is True

    def test_template_exists_false(self):
        """Test template_exists returns False for nonexistent template."""
        tm = TemplateManager()
        assert tm.template_exists("nonexistent_template_xyz.html") is False

    def test_repr(self):
        """Test string representation."""
        tm = TemplateManager()
        repr_str = repr(tm)
        assert "TemplateManager" in repr_str
        assert "cache_enabled=" in repr_str
        assert "cached_templates=" in repr_str
