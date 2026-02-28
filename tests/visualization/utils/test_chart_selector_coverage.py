"""Coverage tests for panelbox.visualization.utils.chart_selector module.

Targets uncovered lines: 502, 516-561
Focus on: suggest_chart, _search_by_keywords, list_all_charts, get_categories,
_interactive_decision_tree
"""

from unittest.mock import patch

import pytest


class TestChartRecommendation:
    """Test ChartRecommendation dataclass."""

    def test_str_representation(self):
        from panelbox.visualization.utils.chart_selector import ChartRecommendation

        rec = ChartRecommendation(
            chart_name="test",
            display_name="Test Chart",
            chart_type="test_chart",
            description="A test chart",
            use_cases=["testing"],
            api_function="test_func()",
            code_example="# example code",
            category="Test",
        )
        s = str(rec)
        assert "Test Chart" in s
        assert "test_chart" in s
        assert "testing" in s
        assert "test_func()" in s

    def test_chart_recommendations_populated(self):
        from panelbox.visualization.utils.chart_selector import (
            CHART_RECOMMENDATIONS,
        )

        assert len(CHART_RECOMMENDATIONS) > 10


class TestSuggestChart:
    """Test suggest_chart function."""

    def test_suggest_by_purpose(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart(purpose="residual_qq_plot")
        assert result.chart_name == "QQ Plot"

    def test_suggest_unknown_purpose_returns_all(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart(purpose="nonexistent_chart")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_suggest_no_args_returns_all(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart()
        assert isinstance(result, list)
        assert len(result) > 0

    def test_suggest_by_keywords(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart(keywords=["normality", "residual"])
        assert isinstance(result, list)
        assert len(result) > 0
        # Should find QQ plot
        names = [r.chart_name for r in result]
        assert "QQ Plot" in names

    def test_suggest_by_keywords_no_match(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart(keywords=["xyznonexistent123"])
        assert isinstance(result, list)
        assert len(result) == 0

    def test_suggest_entity_effects(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart(purpose="entity_effects_plot")
        assert result.chart_type == "entity_effects_plot"

    def test_suggest_cointegration(self):
        from panelbox.visualization.utils.chart_selector import suggest_chart

        result = suggest_chart(purpose="cointegration_heatmap")
        assert result.chart_type == "cointegration_heatmap"


class TestListAllCharts:
    """Test list_all_charts function."""

    def test_list_all(self):
        from panelbox.visualization.utils.chart_selector import list_all_charts

        charts = list_all_charts()
        assert len(charts) > 10
        assert all(hasattr(c, "chart_name") for c in charts)

    def test_list_by_category_residual(self):
        from panelbox.visualization.utils.chart_selector import list_all_charts

        charts = list_all_charts(category="Residual Diagnostics")
        assert len(charts) > 0
        assert all(c.category == "Residual Diagnostics" for c in charts)

    def test_list_by_category_panel(self):
        from panelbox.visualization.utils.chart_selector import list_all_charts

        charts = list_all_charts(category="Panel-Specific")
        assert len(charts) > 0

    def test_list_by_category_econometric(self):
        from panelbox.visualization.utils.chart_selector import list_all_charts

        charts = list_all_charts(category="Econometric Tests")
        assert len(charts) > 0

    def test_list_by_nonexistent_category(self):
        from panelbox.visualization.utils.chart_selector import list_all_charts

        charts = list_all_charts(category="Nonexistent Category")
        assert len(charts) == 0


class TestGetCategories:
    """Test get_categories function."""

    def test_returns_sorted_list(self):
        from panelbox.visualization.utils.chart_selector import get_categories

        cats = get_categories()
        assert isinstance(cats, list)
        assert len(cats) > 0
        assert cats == sorted(cats)

    def test_expected_categories_present(self):
        from panelbox.visualization.utils.chart_selector import get_categories

        cats = get_categories()
        assert "Residual Diagnostics" in cats
        assert "Panel-Specific" in cats


class TestDecisionTree:
    """Test DECISION_TREE structure."""

    def test_decision_tree_has_root(self):
        from panelbox.visualization.utils.chart_selector import DECISION_TREE

        assert "root" in DECISION_TREE
        assert "question" in DECISION_TREE["root"]
        assert "options" in DECISION_TREE["root"]

    def test_all_next_nodes_exist(self):
        from panelbox.visualization.utils.chart_selector import DECISION_TREE

        for node_name, node in DECISION_TREE.items():
            for key, option in node["options"].items():
                if "next" in option:
                    assert option["next"] in DECISION_TREE, (
                        f"Node {node_name} option {key} references missing node {option['next']}"
                    )


class TestSearchByKeywords:
    """Test _search_by_keywords private function."""

    def test_search_heteroskedasticity(self):
        from panelbox.visualization.utils.chart_selector import _search_by_keywords

        results = _search_by_keywords(["heteroskedasticity"])
        assert len(results) > 0

    def test_search_autocorrelation(self):
        from panelbox.visualization.utils.chart_selector import _search_by_keywords

        results = _search_by_keywords(["autocorrelation"])
        assert len(results) > 0

    def test_search_case_insensitive(self):
        from panelbox.visualization.utils.chart_selector import _search_by_keywords

        results_lower = _search_by_keywords(["normality"])
        results_upper = _search_by_keywords(["NORMALITY"])
        assert len(results_lower) == len(results_upper)

    def test_search_multiple_keywords(self):
        from panelbox.visualization.utils.chart_selector import _search_by_keywords

        results = _search_by_keywords(["panel", "structure"])
        assert len(results) > 0


class TestInteractiveDecisionTree:
    """Test _interactive_decision_tree - covers lines 502, 516-561."""

    def test_suggest_chart_interactive_calls_tree(self):
        """Cover line 502: interactive=True branch in suggest_chart."""
        from panelbox.visualization.utils.chart_selector import suggest_chart

        # Simulate user quitting immediately
        with patch("builtins.input", return_value="q"):
            result = suggest_chart(interactive=True)
        assert result is None

    def test_interactive_tree_quit(self):
        """Cover lines 516-536: entering and quitting the tree."""
        from panelbox.visualization.utils.chart_selector import _interactive_decision_tree

        with patch("builtins.input", return_value="q"):
            result = _interactive_decision_tree()
        assert result is None

    def test_interactive_tree_invalid_then_quit(self):
        """Cover lines 538-540: invalid choice then quit."""
        from panelbox.visualization.utils.chart_selector import _interactive_decision_tree

        with patch("builtins.input", side_effect=["z", "q"]):
            result = _interactive_decision_tree()
        assert result is None

    def test_interactive_tree_navigate_to_recommendation(self):
        """Cover lines 542-553: navigating to a terminal node."""
        from panelbox.visualization.utils.chart_selector import (
            DECISION_TREE,
            _interactive_decision_tree,
        )

        # Find a path from root to a terminal recommendation
        # Walk the tree to find a valid sequence of choices
        choices = []
        node_name = "root"
        while node_name:
            node = DECISION_TREE[node_name]
            # Pick the first option
            first_key = next(iter(node["options"].keys()))
            choices.append(first_key)
            option = node["options"][first_key]
            if "recommend" in option:
                break
            node_name = option.get("next")

        with patch("builtins.input", side_effect=choices):
            result = _interactive_decision_tree()
        assert result is not None
        assert hasattr(result, "chart_name")

    def test_interactive_tree_unknown_recommendation(self):
        """Cover lines 554-556: recommend key not in CHART_RECOMMENDATIONS."""
        from panelbox.visualization.utils.chart_selector import (
            DECISION_TREE,
            _interactive_decision_tree,
        )

        # Temporarily modify a terminal node to have a bad recommend key
        # Find a path to a terminal node
        node_name = "root"
        choices = []
        target_node = None
        target_key = None
        while node_name:
            node = DECISION_TREE[node_name]
            first_key = next(iter(node["options"].keys()))
            choices.append(first_key)
            option = node["options"][first_key]
            if "recommend" in option:
                target_node = node_name
                target_key = first_key
                break
            node_name = option.get("next")

        if target_node is None:
            pytest.skip("No terminal node found in decision tree")

        original = DECISION_TREE[target_node]["options"][target_key]["recommend"]
        try:
            DECISION_TREE[target_node]["options"][target_key]["recommend"] = "nonexistent_xyz"
            with patch("builtins.input", side_effect=choices):
                result = _interactive_decision_tree()
            assert result is None
        finally:
            DECISION_TREE[target_node]["options"][target_key]["recommend"] = original
