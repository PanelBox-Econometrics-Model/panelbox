"""
Chart Selection Helper - Decision Tree System.

This module provides an interactive decision tree to help users select the most
appropriate chart type for their analysis needs. It includes both programmatic
API and interactive CLI modes.

Features:
- Decision tree logic for 28+ chart types
- Interactive question-based selection
- Keyword-based search
- Usage examples for selected charts
- Integration with chart registry

Examples:
    Interactive mode:
    >>> from panelbox.visualization.utils import suggest_chart
    >>> suggest_chart(interactive=True)

    Programmatic mode:
    >>> charts = suggest_chart(purpose='residual_diagnostics', data_type='panel')
    >>> print(charts)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class ChartRecommendation:
    """Recommendation for a specific chart type."""

    chart_name: str
    display_name: str
    chart_type: str  # Registry name
    description: str
    use_cases: List[str]
    api_function: str
    code_example: str
    category: str

    def __str__(self) -> str:
        """String representation for display."""
        return f"""
📊 {self.display_name}
{'=' * 60}
Registry Type: {self.chart_type}
Category: {self.category}

Description:
  {self.description}

Best for:
{chr(10).join('  • ' + uc for uc in self.use_cases)}

API Function: {self.api_function}

Example:
{self.code_example}
"""


# Chart recommendation database
CHART_RECOMMENDATIONS = {
    # Residual Diagnostics
    "residual_qq_plot": ChartRecommendation(
        chart_name="QQ Plot",
        display_name="Q-Q Plot for Normality",
        chart_type="residual_qq_plot",
        description="Quantile-quantile plot to assess if residuals follow a normal distribution",
        use_cases=[
            "Checking normality assumption of residuals",
            "Identifying heavy-tailed or skewed distributions",
            "Detecting outliers in residuals",
        ],
        api_function="create_residual_diagnostics()",
        code_example="""from panelbox.visualization import create_residual_diagnostics

diagnostics = create_residual_diagnostics(
    results,
    charts=['qq_plot'],
    theme='academic'
)
diagnostics['qq_plot'].show()""",
        category="Residual Diagnostics",
    ),
    "residual_vs_fitted": ChartRecommendation(
        chart_name="Residual vs Fitted",
        display_name="Residual vs Fitted Values",
        chart_type="residual_vs_fitted",
        description="Scatter plot of residuals against fitted values to detect heteroskedasticity",
        use_cases=[
            "Detecting heteroskedasticity (non-constant variance)",
            "Identifying non-linear patterns",
            "Checking for specification errors",
        ],
        api_function="create_residual_diagnostics()",
        code_example="""from panelbox.visualization import create_residual_diagnostics

diagnostics = create_residual_diagnostics(
    results,
    charts=['residual_vs_fitted'],
    theme='professional'
)
diagnostics['residual_vs_fitted'].show()""",
        category="Residual Diagnostics",
    ),
    "scale_location": ChartRecommendation(
        chart_name="Scale-Location Plot",
        display_name="Scale-Location (Spread-Level) Plot",
        chart_type="scale_location",
        description="Plot of sqrt(|standardized residuals|) vs fitted values for variance homogeneity",
        use_cases=[
            "Testing homoskedasticity assumption",
            "Detecting variance changes across fitted values",
            "Comparing variance stability",
        ],
        api_function="create_residual_diagnostics()",
        code_example="""from panelbox.visualization import create_residual_diagnostics

diagnostics = create_residual_diagnostics(
    results,
    charts=['scale_location'],
    theme='academic'
)
diagnostics['scale_location'].show()""",
        category="Residual Diagnostics",
    ),
    "residual_vs_leverage": ChartRecommendation(
        chart_name="Residual vs Leverage",
        display_name="Residual vs Leverage (Influence Plot)",
        chart_type="residual_vs_leverage",
        description="Identify influential observations using leverage and standardized residuals",
        use_cases=[
            "Finding influential observations (high leverage + large residual)",
            "Detecting outliers",
            "Computing Cook's distance for influence",
        ],
        api_function="create_residual_diagnostics()",
        code_example="""from panelbox.visualization import create_residual_diagnostics

diagnostics = create_residual_diagnostics(
    results,
    charts=['residual_vs_leverage'],
    theme='professional'
)
diagnostics['residual_vs_leverage'].show()""",
        category="Residual Diagnostics",
    ),
    "acf_pacf_plot": ChartRecommendation(
        chart_name="ACF/PACF Plot",
        display_name="Autocorrelation & Partial Autocorrelation",
        chart_type="acf_pacf_plot",
        description="Dual plot showing ACF and PACF for detecting serial correlation patterns",
        use_cases=[
            "Detecting serial correlation in residuals",
            "Identifying AR/MA patterns",
            "Ljung-Box test for autocorrelation",
            "Time series model specification",
        ],
        api_function="create_acf_pacf_plot()",
        code_example="""from panelbox.visualization import create_acf_pacf_plot

chart = create_acf_pacf_plot(
    residuals=results.resid,
    max_lags=20,
    confidence_level=0.95,
    show_ljung_box=True,
    theme='academic'
)
chart.show()""",
        category="Econometric Tests",
    ),
    # Validation Charts
    "validation_dashboard": ChartRecommendation(
        chart_name="Validation Dashboard",
        display_name="Comprehensive Validation Dashboard",
        chart_type="validation_dashboard",
        chart_description="Complete overview of all validation test results in one dashboard",
        use_cases=[
            "Getting overall model validation status",
            "Executive summary of all tests",
            "Quick model health check",
        ],
        api_function="create_validation_charts()",
        code_example="""from panelbox.visualization import create_validation_charts

charts = create_validation_charts(
    validation_report,
    theme='professional',
    interactive=True
)
charts['dashboard'].show()""",
        category="Validation",
    ),
    "unit_root_test_plot": ChartRecommendation(
        chart_name="Unit Root Test",
        display_name="Unit Root Test Results",
        chart_type="unit_root_test_plot",
        description="Bar chart with color-coded significance for stationarity tests",
        use_cases=[
            "Testing for unit roots (non-stationarity)",
            "Comparing multiple tests (ADF, PP, KPSS)",
            "Panel unit root tests (IPS, LLC, Fisher)",
            "Visualizing test statistics vs critical values",
        ],
        api_function="create_unit_root_test_plot()",
        code_example="""from panelbox.visualization import create_unit_root_test_plot

results = {
    'test_names': ['ADF', 'PP', 'KPSS'],
    'test_stats': [-3.5, -3.8, 0.3],
    'critical_values': {'1%': -3.96, '5%': -3.41, '10%': -3.13},
    'pvalues': [0.008, 0.003, 0.15]
}
chart = create_unit_root_test_plot(results, theme='professional')
chart.show()""",
        category="Econometric Tests",
    ),
    # Panel-Specific Charts
    "entity_effects_plot": ChartRecommendation(
        chart_name="Entity Effects",
        display_name="Entity Fixed/Random Effects",
        chart_type="entity_effects_plot",
        description="Visualize entity-specific effects with confidence intervals",
        use_cases=[
            "Displaying fixed effects by entity",
            "Comparing entity-level heterogeneity",
            "Identifying entities with significant effects",
        ],
        api_function="create_entity_effects_plot()",
        code_example="""from panelbox.visualization import create_entity_effects_plot

chart = create_entity_effects_plot(
    panel_results,
    theme='academic',
    sort_by='effect'
)
chart.show()""",
        category="Panel-Specific",
    ),
    "time_effects_plot": ChartRecommendation(
        chart_name="Time Effects",
        display_name="Time Period Effects",
        chart_type="time_effects_plot",
        description="Visualize time-specific effects across periods",
        use_cases=[
            "Displaying time fixed effects",
            "Identifying temporal trends",
            "Detecting time-specific shocks",
        ],
        api_function="create_time_effects_plot()",
        code_example="""from panelbox.visualization import create_time_effects_plot

chart = create_time_effects_plot(
    panel_results,
    theme='professional',
    show_trend=True
)
chart.show()""",
        category="Panel-Specific",
    ),
    "between_within_plot": ChartRecommendation(
        chart_name="Between-Within Variation",
        display_name="Between-Within Variance Decomposition",
        chart_type="between_within_plot",
        description="Decompose total variance into between and within components",
        use_cases=[
            "Understanding variance sources in panel data",
            "Comparing between-entity vs within-entity variation",
            "Assessing suitability of FE vs RE models",
        ],
        api_function="create_between_within_plot()",
        code_example="""from panelbox.visualization import create_between_within_plot

chart = create_between_within_plot(
    panel_data,
    variables=['capital', 'labor', 'output'],
    theme='academic',
    style='stacked'  # or 'grouped', 'percentage'
)
chart.show()""",
        category="Panel-Specific",
    ),
    "panel_structure_plot": ChartRecommendation(
        chart_name="Panel Structure",
        display_name="Panel Structure & Balance",
        chart_type="panel_structure_plot",
        description="Heatmap showing panel balance and missing data patterns",
        use_cases=[
            "Visualizing panel structure",
            "Identifying missing observations",
            "Assessing panel balance",
            "Detecting attrition patterns",
        ],
        api_function="create_panel_structure_plot()",
        code_example="""from panelbox.visualization import create_panel_structure_plot

chart = create_panel_structure_plot(
    panel_data,
    theme='professional'
)
chart.show()""",
        category="Panel-Specific",
    ),
    # Cointegration & Cross-Sectional Dependence
    "cointegration_heatmap": ChartRecommendation(
        chart_name="Cointegration Heatmap",
        display_name="Pairwise Cointegration Matrix",
        chart_type="cointegration_heatmap",
        description="Heatmap of pairwise cointegration test p-values",
        use_cases=[
            "Testing for cointegration between variables",
            "Engle-Granger or Johansen test results",
            "Identifying long-run relationships",
        ],
        api_function="create_cointegration_heatmap()",
        code_example="""from panelbox.visualization import create_cointegration_heatmap

results = {
    'variables': ['GDP', 'Consumption', 'Investment'],
    'pvalues': [[1.0, 0.02, 0.15], [0.02, 1.0, 0.08], [0.15, 0.08, 1.0]],
    'test_name': 'Engle-Granger'
}
chart = create_cointegration_heatmap(results, theme='academic')
chart.show()""",
        category="Econometric Tests",
    ),
    "cross_sectional_dependence_plot": ChartRecommendation(
        chart_name="Cross-Sectional Dependence",
        display_name="Pesaran CD Test Visualization",
        chart_type="cross_sectional_dependence_plot",
        description="Gauge indicator showing cross-sectional dependence with optional entity breakdown",
        use_cases=[
            "Testing for cross-sectional dependence in panels",
            "Pesaran CD test results",
            "Identifying correlation across entities",
        ],
        api_function="create_cross_sectional_dependence_plot()",
        code_example="""from panelbox.visualization import create_cross_sectional_dependence_plot

results = {
    'cd_statistic': 3.45,
    'pvalue': 0.003,
    'avg_correlation': 0.28,
    'entity_correlations': [0.15, 0.32, 0.45, 0.21]  # Optional
}
chart = create_cross_sectional_dependence_plot(results, theme='professional')
chart.show()""",
        category="Econometric Tests",
    ),
    # Model Comparison
    "coefficient_comparison": ChartRecommendation(
        chart_name="Coefficient Comparison",
        display_name="Model Coefficient Comparison",
        chart_type="coefficient_comparison",
        description="Compare coefficient estimates across multiple models",
        use_cases=[
            "Comparing coefficients across models",
            "Assessing robustness of estimates",
            "Visualizing model differences",
        ],
        api_function="create_comparison_charts()",
        code_example="""from panelbox.visualization import create_comparison_charts

charts = create_comparison_charts(
    [model1_results, model2_results, model3_results],
    model_names=['OLS', 'FE', 'RE'],
    theme='professional'
)
charts['coefficients'].show()""",
        category="Model Comparison",
    ),
}


# Decision tree logic
DECISION_TREE = {
    "root": {
        "question": "What is the primary goal of your analysis?",
        "options": {
            "1": {"label": "Residual Diagnostics", "next": "residual_diagnostics"},
            "2": {"label": "Model Validation", "next": "model_validation"},
            "3": {"label": "Model Comparison", "next": "model_comparison"},
            "4": {"label": "Panel Data Analysis", "next": "panel_analysis"},
            "5": {"label": "Econometric Tests", "next": "econometric_tests"},
            "6": {"label": "Exploratory Data Analysis", "next": "eda"},
        },
    },
    "residual_diagnostics": {
        "question": "What aspect of residuals do you want to check?",
        "options": {
            "1": {"label": "Normality", "recommend": "residual_qq_plot"},
            "2": {"label": "Heteroskedasticity", "recommend": "residual_vs_fitted"},
            "3": {"label": "Variance Stability", "recommend": "scale_location"},
            "4": {"label": "Influential Observations", "recommend": "residual_vs_leverage"},
            "5": {"label": "Serial Correlation", "recommend": "acf_pacf_plot"},
            "6": {"label": "All Diagnostics", "recommend": "residual_diagnostics_all"},
        },
    },
    "model_validation": {
        "question": "What type of validation do you need?",
        "options": {
            "1": {"label": "Overall Validation Status", "recommend": "validation_dashboard"},
            "2": {"label": "Stationarity Tests", "recommend": "unit_root_test_plot"},
            "3": {"label": "Specification Tests", "recommend": "validation_dashboard"},
            "4": {"label": "P-value Distribution", "recommend": "validation_pvalue_distribution"},
        },
    },
    "model_comparison": {
        "question": "What do you want to compare across models?",
        "options": {
            "1": {"label": "Coefficients", "recommend": "coefficient_comparison"},
            "2": {"label": "Model Fit (R², AIC, BIC)", "recommend": "model_fit_comparison"},
            "3": {"label": "Information Criteria", "recommend": "information_criteria"},
            "4": {"label": "Forest Plot (Coefficients)", "recommend": "forest_plot"},
        },
    },
    "panel_analysis": {
        "question": "What panel data characteristic do you want to analyze?",
        "options": {
            "1": {"label": "Entity-Specific Effects", "recommend": "entity_effects_plot"},
            "2": {"label": "Time-Period Effects", "recommend": "time_effects_plot"},
            "3": {"label": "Between-Within Variation", "recommend": "between_within_plot"},
            "4": {"label": "Panel Structure/Balance", "recommend": "panel_structure_plot"},
            "5": {
                "label": "Cross-Sectional Dependence",
                "recommend": "cross_sectional_dependence_plot",
            },
        },
    },
    "econometric_tests": {
        "question": "What econometric property do you want to test?",
        "options": {
            "1": {"label": "Serial Correlation (ACF/PACF)", "recommend": "acf_pacf_plot"},
            "2": {"label": "Unit Roots / Stationarity", "recommend": "unit_root_test_plot"},
            "3": {"label": "Cointegration", "recommend": "cointegration_heatmap"},
            "4": {
                "label": "Cross-Sectional Dependence",
                "recommend": "cross_sectional_dependence_plot",
            },
        },
    },
    "eda": {
        "question": "What type of exploratory analysis?",
        "options": {
            "1": {"label": "Distribution", "recommend": "histogram"},
            "2": {"label": "Correlation", "recommend": "correlation_heatmap"},
            "3": {"label": "Time Series", "recommend": "panel_timeseries"},
            "4": {"label": "Group Comparison", "recommend": "box_plot"},
        },
    },
}


def suggest_chart(
    purpose: Optional[str] = None,
    data_type: Optional[str] = None,
    interactive: bool = False,
    keywords: Optional[List[str]] = None,
) -> Union[List[ChartRecommendation], ChartRecommendation]:
    """
    Suggest appropriate chart types based on analysis purpose.

    Parameters
    ----------
    purpose : str, optional
        Analysis purpose (e.g., 'residual_diagnostics', 'validation', 'comparison')
    data_type : str, optional
        Type of data ('panel', 'time_series', 'cross_section')
    interactive : bool, default False
        If True, launch interactive CLI decision tree
    keywords : list of str, optional
        Keywords to search for charts (e.g., ['normality', 'residuals'])

    Returns
    -------
    ChartRecommendation or list of ChartRecommendation
        Recommended chart(s) with usage examples

    Examples
    --------
    Interactive mode:
    >>> from panelbox.visualization.utils import suggest_chart
    >>> suggest_chart(interactive=True)

    Keyword search:
    >>> charts = suggest_chart(keywords=['residual', 'normality'])
    >>> for chart in charts:
    ...     print(chart)

    Direct purpose:
    >>> chart = suggest_chart(purpose='residual_qq_plot')
    >>> print(chart.code_example)
    """
    if interactive:
        return _interactive_decision_tree()

    if keywords:
        return _search_by_keywords(keywords)

    if purpose and purpose in CHART_RECOMMENDATIONS:
        return CHART_RECOMMENDATIONS[purpose]

    # Default: return residual diagnostics recommendation
    return list(CHART_RECOMMENDATIONS.values())


def _interactive_decision_tree() -> ChartRecommendation:
    """Run interactive CLI decision tree."""
    print("\n" + "=" * 70)
    print("📊 PanelBox Chart Selection Assistant")
    print("=" * 70)
    print("\nThis tool will help you choose the right chart for your analysis.\n")

    current_node = "root"

    while current_node:
        node = DECISION_TREE[current_node]
        print(f"\n{node['question']}\n")

        for key, option in node["options"].items():
            print(f"  [{key}] {option['label']}")

        print("\n  [q] Quit")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "q":
            print("\nExiting chart selector. Goodbye!")
            return None

        if choice not in node["options"]:
            print("❌ Invalid choice. Please try again.")
            continue

        option = node["options"][choice]

        if "recommend" in option:
            # Terminal node - return recommendation
            chart_key = option["recommend"]
            if chart_key in CHART_RECOMMENDATIONS:
                recommendation = CHART_RECOMMENDATIONS[chart_key]
                print("\n" + "=" * 70)
                print("✅ RECOMMENDATION")
                print("=" * 70)
                print(recommendation)
                return recommendation
            else:
                print(f"\n⚠️  Chart '{chart_key}' is being implemented.")
                return None
        else:
            # Continue to next node
            current_node = option["next"]

    return None


def _search_by_keywords(keywords: List[str]) -> List[ChartRecommendation]:
    """Search charts by keywords."""
    keywords_lower = [kw.lower() for kw in keywords]
    results = []

    for chart in CHART_RECOMMENDATIONS.values():
        searchable_text = (
            chart.display_name
            + " "
            + chart.description
            + " "
            + " ".join(chart.use_cases)
            + " "
            + chart.category
        ).lower()

        if any(kw in searchable_text for kw in keywords_lower):
            results.append(chart)

    return results


def list_all_charts(category: Optional[str] = None) -> List[ChartRecommendation]:
    """
    List all available charts, optionally filtered by category.

    Parameters
    ----------
    category : str, optional
        Filter by category ('Residual Diagnostics', 'Validation',
        'Model Comparison', 'Panel-Specific', 'Econometric Tests')

    Returns
    -------
    list of ChartRecommendation
        All matching charts

    Examples
    --------
    >>> from panelbox.visualization.utils import list_all_charts
    >>> panel_charts = list_all_charts(category='Panel-Specific')
    >>> for chart in panel_charts:
    ...     print(f"- {chart.display_name}")
    """
    if category:
        return [c for c in CHART_RECOMMENDATIONS.values() if c.category == category]
    return list(CHART_RECOMMENDATIONS.values())


def get_categories() -> List[str]:
    """
    Get list of all chart categories.

    Returns
    -------
    list of str
        Unique category names
    """
    return sorted(set(c.category for c in CHART_RECOMMENDATIONS.values()))


if __name__ == "__main__":
    # Run interactive mode when executed as script
    suggest_chart(interactive=True)
