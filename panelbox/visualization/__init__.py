"""
PanelBox Visualization Module.

This module provides comprehensive visualization capabilities for panel data analysis,
including interactive Plotly charts, static Matplotlib plots, and advanced diagnostics.

Main Features
-------------
- 28+ chart types for validation, diagnostics, and model comparison
- 3 professional themes (Professional, Academic, Presentation)
- Multiple export formats (JSON, HTML, PNG, SVG, PDF)
- Registry-based chart system for easy extensibility
- High-level convenience APIs for common use cases

Quick Start
-----------
>>> from panelbox.visualization import create_validation_charts
>>> charts = create_validation_charts(validation_report, theme='professional')

Architecture
------------
The visualization system follows a modular architecture:

1. **Base Classes** - Abstract base classes for all charts
2. **Registry** - Decorator-based chart registration system
3. **Factory** - Centralized chart creation with theming
4. **Themes** - Professional visual themes with design tokens
5. **Charts** - Plotly and Matplotlib implementations
6. **Transformers** - Data transformation layer

Design Patterns
---------------
- Registry Pattern: Decorator-based chart registration
- Factory Pattern: Centralized chart creation
- Strategy Pattern: Multiple rendering backends
- Template Method: Consistent chart creation workflow

Examples
--------
Create validation charts:

>>> from panelbox.validation import ValidationReport
>>> from panelbox.visualization import create_validation_charts
>>>
>>> charts = create_validation_charts(
...     validation_report,
...     theme='professional',
...     interactive=True
... )

Create residual diagnostics:

>>> from panelbox.visualization import create_residual_diagnostics
>>>
>>> diagnostics = create_residual_diagnostics(
...     results,
...     theme='academic',
...     charts=['qq_plot', 'residual_vs_fitted']
... )

Create custom chart:

>>> from panelbox.visualization import ChartFactory
>>> from panelbox.visualization.themes import PROFESSIONAL_THEME
>>>
>>> chart = ChartFactory.create(
...     chart_type='residual_qq_plot',
...     data={'residuals': residuals},
...     theme=PROFESSIONAL_THEME
... )
>>> html = chart.to_html()
"""

from .base import BaseChart, MatplotlibChartBase, PlotlyChartBase
from .factory import ChartFactory
from .registry import ChartRegistry, register_chart
from .themes import ACADEMIC_THEME, PRESENTATION_THEME, PROFESSIONAL_THEME, Theme

# Import chart implementations to trigger registration
# These imports MUST succeed for chart registration to work
try:
    from .plotly.basic import BarChart, LineChart
    from .plotly.comparison import (
        CoefficientComparisonChart,
        ForestPlotChart,
        InformationCriteriaChart,
        ModelFitComparisonChart,
    )
    from .plotly.correlation import CorrelationHeatmapChart, PairwiseCorrelationChart
    from .plotly.distribution import BoxPlotChart, HistogramChart, KDEChart, ViolinPlotChart
    from .plotly.econometric_tests import (
        ACFPACFPlot,
        CointegrationHeatmap,
        CrossSectionalDependencePlot,
        UnitRootTestPlot,
    )
    from .plotly.panel import (
        BetweenWithinPlot,
        EntityEffectsPlot,
        PanelStructurePlot,
        TimeEffectsPlot,
    )
    from .plotly.residuals import (
        PartialRegressionPlot,
        QQPlot,
        ResidualDistributionPlot,
        ResidualTimeSeriesPlot,
        ResidualVsFittedPlot,
        ResidualVsLeveragePlot,
        ScaleLocationPlot,
    )
    from .plotly.timeseries import FacetedTimeSeriesChart, PanelTimeSeriesChart, TrendLineChart
    from .plotly.validation import (
        PValueDistributionChart,
        TestComparisonHeatmap,
        TestOverviewChart,
        TestStatisticsChart,
        ValidationDashboard,
    )

    _has_plotly_charts = True
except ImportError:
    BarChart = None
    LineChart = None
    TestOverviewChart = None
    PValueDistributionChart = None
    TestStatisticsChart = None
    TestComparisonHeatmap = None
    ValidationDashboard = None
    QQPlot = None
    ResidualVsFittedPlot = None
    ScaleLocationPlot = None
    ResidualVsLeveragePlot = None
    ResidualTimeSeriesPlot = None
    ResidualDistributionPlot = None
    PartialRegressionPlot = None
    CoefficientComparisonChart = None
    ForestPlotChart = None
    ModelFitComparisonChart = None
    InformationCriteriaChart = None
    HistogramChart = None
    KDEChart = None
    ViolinPlotChart = None
    BoxPlotChart = None
    CorrelationHeatmapChart = None
    PairwiseCorrelationChart = None
    PanelTimeSeriesChart = None
    TrendLineChart = None
    FacetedTimeSeriesChart = None
    EntityEffectsPlot = None
    TimeEffectsPlot = None
    BetweenWithinPlot = None
    PanelStructurePlot = None
    ACFPACFPlot = None
    UnitRootTestPlot = None
    CointegrationHeatmap = None
    CrossSectionalDependencePlot = None
    _has_plotly_charts = False

# High-level convenience APIs (Phase 2)
try:
    from .api import (
        create_acf_pacf_plot,
        create_between_within_plot,
        create_cointegration_heatmap,
        create_comparison_charts,
        create_cross_sectional_dependence_plot,
        create_entity_effects_plot,
        create_panel_charts,
        create_panel_structure_plot,
        create_residual_diagnostics,
        create_time_effects_plot,
        create_unit_root_test_plot,
        create_validation_charts,
        export_chart,
        export_charts,
        export_charts_multiple_formats,
    )

    _has_api = True
except ImportError:
    create_validation_charts = None
    create_residual_diagnostics = None
    create_comparison_charts = None
    export_chart = None
    export_charts = None
    export_charts_multiple_formats = None
    create_panel_charts = None
    create_entity_effects_plot = None
    create_time_effects_plot = None
    create_between_within_plot = None
    create_panel_structure_plot = None
    create_acf_pacf_plot = None
    create_unit_root_test_plot = None
    create_cointegration_heatmap = None
    create_cross_sectional_dependence_plot = None
    _has_api = False

__all__ = [
    # Base classes
    "BaseChart",
    "PlotlyChartBase",
    "MatplotlibChartBase",
    # Registry and Factory
    "ChartRegistry",
    "register_chart",
    "ChartFactory",
    # Themes
    "Theme",
    "PROFESSIONAL_THEME",
    "ACADEMIC_THEME",
    "PRESENTATION_THEME",
    # Basic Charts
    "BarChart",
    "LineChart",
    # Validation Charts
    "TestOverviewChart",
    "PValueDistributionChart",
    "TestStatisticsChart",
    "TestComparisonHeatmap",
    "ValidationDashboard",
    # Residual Diagnostic Charts
    "QQPlot",
    "ResidualVsFittedPlot",
    "ScaleLocationPlot",
    "ResidualVsLeveragePlot",
    "ResidualTimeSeriesPlot",
    "ResidualDistributionPlot",
    "PartialRegressionPlot",
    # Model Comparison Charts
    "CoefficientComparisonChart",
    "ForestPlotChart",
    "ModelFitComparisonChart",
    "InformationCriteriaChart",
    # Distribution Charts
    "HistogramChart",
    "KDEChart",
    "ViolinPlotChart",
    "BoxPlotChart",
    # Correlation Charts
    "CorrelationHeatmapChart",
    "PairwiseCorrelationChart",
    # Time Series Charts
    "PanelTimeSeriesChart",
    "TrendLineChart",
    "FacetedTimeSeriesChart",
    # Panel-Specific Charts
    "EntityEffectsPlot",
    "TimeEffectsPlot",
    "BetweenWithinPlot",
    "PanelStructurePlot",
    # High-level APIs
    "create_validation_charts",
    "create_residual_diagnostics",
    "create_comparison_charts",
    "create_panel_charts",
    # Panel-specific APIs
    "create_entity_effects_plot",
    "create_time_effects_plot",
    "create_between_within_plot",
    "create_panel_structure_plot",
    # Econometric test APIs (Phase 7)
    "ACFPACFPlot",
    "UnitRootTestPlot",
    "CointegrationHeatmap",
    "CrossSectionalDependencePlot",
    "create_acf_pacf_plot",
    "create_unit_root_test_plot",
    "create_cointegration_heatmap",
    "create_cross_sectional_dependence_plot",
    # Export functions
    "export_chart",
    "export_charts",
    "export_charts_multiple_formats",
]

__version__ = "0.6.0"


def _initialize_chart_registry():
    """
    Initialize the chart registry by importing all chart modules.

    This function ensures that all chart decorators (@register_chart)
    are executed at module import time, populating the ChartRegistry.
    """
    # Charts are already imported above if plotly is available
    # This function serves as documentation and can be called explicitly if needed
    if _has_plotly_charts:
        # Verify registration happened
        registered = ChartRegistry.list_charts()
        if not registered:
            # Force re-import if registry is empty
            import importlib

            from . import plotly

            # Reload all plotly submodules to trigger registration
            for module_name in [
                "basic",
                "validation",
                "residuals",
                "comparison",
                "distribution",
                "correlation",
                "timeseries",
            ]:
                try:
                    module = importlib.import_module(
                        f".plotly.{module_name}", package="panelbox.visualization"
                    )
                    importlib.reload(module)
                except ImportError:
                    pass


# Initialize registry at import time
_initialize_chart_registry()
