"""Tests for chart selector utility."""

import pytest

from panelbox.visualization.utils.chart_selector import (
    CHART_RECOMMENDATIONS,
    ChartRecommendation,
    _search_by_keywords,
    get_categories,
    list_all_charts,
    suggest_chart,
)


class TestChartRecommendation:
    """Tests for ChartRecommendation dataclass."""

    def test_chart_recommendation_str(self):
        """Test string representation of ChartRecommendation."""
        rec = ChartRecommendation(
            chart_name="Test Chart",
            display_name="Test Display Name",
            chart_type="test_chart",
            description="Test description",
            use_cases=["Use case 1", "Use case 2"],
            api_function="test_function()",
            code_example="test code",
            category="Test Category",
        )

        str_repr = str(rec)
        assert "Test Display Name" in str_repr
        assert "test_chart" in str_repr
        assert "Test Category" in str_repr
        assert "Use case 1" in str_repr
        assert "Use case 2" in str_repr


class TestSuggestChart:
    """Tests for suggest_chart function."""

    def test_suggest_chart_by_purpose(self):
        """Test suggesting chart by purpose."""
        result = suggest_chart(purpose="residual_qq_plot")
        assert isinstance(result, ChartRecommendation)
        assert result.chart_type == "residual_qq_plot"

    def test_suggest_chart_invalid_purpose(self):
        """Test suggesting chart with invalid purpose returns all charts."""
        result = suggest_chart(purpose="invalid_purpose")
        assert isinstance(result, list)
        assert len(result) == len(CHART_RECOMMENDATIONS)

    def test_suggest_chart_by_keywords(self):
        """Test suggesting chart by keywords."""
        results = suggest_chart(keywords=["residual", "normality"])
        assert isinstance(results, list)
        assert len(results) > 0
        # Check that at least one result is relevant
        assert any(
            "normality" in r.description.lower() or "qq" in r.chart_type.lower() for r in results
        )

    def test_suggest_chart_keywords_case_insensitive(self):
        """Test keyword search is case insensitive."""
        results_lower = suggest_chart(keywords=["residual"])
        results_upper = suggest_chart(keywords=["RESIDUAL"])
        assert len(results_lower) == len(results_upper)

    def test_suggest_chart_keywords_multiple(self):
        """Test multiple keywords."""
        results = suggest_chart(keywords=["panel", "effect"])
        assert isinstance(results, list)
        assert len(results) > 0

    def test_suggest_chart_default(self):
        """Test default behavior returns all charts."""
        result = suggest_chart()
        assert isinstance(result, list)
        assert len(result) == len(CHART_RECOMMENDATIONS)


class TestListAllCharts:
    """Tests for list_all_charts function."""

    def test_list_all_charts_no_filter(self):
        """Test listing all charts without filter."""
        charts = list_all_charts()
        assert len(charts) == len(CHART_RECOMMENDATIONS)

    def test_list_all_charts_with_category(self):
        """Test listing charts filtered by category."""
        charts = list_all_charts(category="Residual Diagnostics")
        assert len(charts) > 0
        assert all(c.category == "Residual Diagnostics" for c in charts)

    def test_list_all_charts_invalid_category(self):
        """Test listing charts with invalid category."""
        charts = list_all_charts(category="Invalid Category")
        assert len(charts) == 0

    def test_list_all_charts_panel_specific(self):
        """Test listing panel-specific charts."""
        charts = list_all_charts(category="Panel-Specific")
        assert len(charts) > 0
        assert all(c.category == "Panel-Specific" for c in charts)


class TestGetCategories:
    """Tests for get_categories function."""

    def test_get_categories(self):
        """Test getting all categories."""
        categories = get_categories()
        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "Residual Diagnostics" in categories
        assert "Panel-Specific" in categories

    def test_get_categories_sorted(self):
        """Test categories are sorted."""
        categories = get_categories()
        assert categories == sorted(categories)

    def test_get_categories_unique(self):
        """Test categories are unique."""
        categories = get_categories()
        assert len(categories) == len(set(categories))


class TestSearchByKeywords:
    """Tests for _search_by_keywords function."""

    def test_search_by_keywords_single(self):
        """Test search with single keyword."""
        results = _search_by_keywords(["residual"])
        assert len(results) > 0
        assert all(
            "residual" in r.description.lower()
            or "residual" in r.display_name.lower()
            or "residual" in " ".join(r.use_cases).lower()
            for r in results
        )

    def test_search_by_keywords_multiple(self):
        """Test search with multiple keywords."""
        results = _search_by_keywords(["panel", "effect"])
        assert len(results) > 0

    def test_search_by_keywords_no_match(self):
        """Test search with no matching keywords."""
        results = _search_by_keywords(["nonexistent_keyword_xyz"])
        assert len(results) == 0

    def test_search_by_keywords_category(self):
        """Test search matches category."""
        results = _search_by_keywords(["econometric"])
        assert len(results) > 0
        assert any(r.category == "Econometric Tests" for r in results)


class TestChartRecommendations:
    """Tests for CHART_RECOMMENDATIONS database."""

    def test_chart_recommendations_not_empty(self):
        """Test that CHART_RECOMMENDATIONS is not empty."""
        assert len(CHART_RECOMMENDATIONS) > 0

    def test_all_recommendations_have_required_fields(self):
        """Test all recommendations have required fields."""
        for key, rec in CHART_RECOMMENDATIONS.items():
            assert rec.chart_name
            assert rec.display_name
            assert rec.chart_type
            assert rec.description
            assert rec.use_cases
            assert len(rec.use_cases) > 0
            assert rec.api_function
            assert rec.code_example
            assert rec.category

    def test_chart_types_unique(self):
        """Test chart types are unique."""
        chart_types = [rec.chart_type for rec in CHART_RECOMMENDATIONS.values()]
        assert len(chart_types) == len(set(chart_types))

    def test_code_examples_valid(self):
        """Test code examples contain import statements."""
        for rec in CHART_RECOMMENDATIONS.values():
            assert "from panelbox" in rec.code_example or "import" in rec.code_example

    def test_residual_diagnostics_category(self):
        """Test residual diagnostics charts exist."""
        residual_charts = [
            rec for rec in CHART_RECOMMENDATIONS.values() if rec.category == "Residual Diagnostics"
        ]
        assert len(residual_charts) > 0

    def test_panel_specific_category(self):
        """Test panel-specific charts exist."""
        panel_charts = [
            rec for rec in CHART_RECOMMENDATIONS.values() if rec.category == "Panel-Specific"
        ]
        assert len(panel_charts) > 0

    def test_econometric_tests_category(self):
        """Test econometric test charts exist."""
        econ_charts = [
            rec for rec in CHART_RECOMMENDATIONS.values() if rec.category == "Econometric Tests"
        ]
        assert len(econ_charts) > 0
