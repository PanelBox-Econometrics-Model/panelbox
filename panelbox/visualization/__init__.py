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

from .base import BaseChart, PlotlyChartBase, MatplotlibChartBase
from .factory import ChartFactory
from .registry import ChartRegistry, register_chart
from .themes import ACADEMIC_THEME, PRESENTATION_THEME, PROFESSIONAL_THEME, Theme

# Import chart implementations to trigger registration (optional)
try:
    from .plotly.basic import BarChart, LineChart
    from .plotly.validation import (
        PValueDistributionChart,
        TestComparisonHeatmap,
        TestOverviewChart,
        TestStatisticsChart,
        ValidationDashboard,
    )
    from .plotly.residuals import (
        QQPlot,
        ResidualVsFittedPlot,
        ScaleLocationPlot,
        ResidualVsLeveragePlot,
        ResidualTimeSeriesPlot,
        ResidualDistributionPlot,
        PartialRegressionPlot,
    )
    from .plotly.comparison import (
        CoefficientComparisonChart,
        ForestPlotChart,
        ModelFitComparisonChart,
        InformationCriteriaChart,
    )
    from .plotly.distribution import (
        HistogramChart,
        KDEChart,
        ViolinPlotChart,
        BoxPlotChart,
    )
    from .plotly.correlation import (
        CorrelationHeatmapChart,
        PairwiseCorrelationChart,
    )
    from .plotly.timeseries import (
        PanelTimeSeriesChart,
        TrendLineChart,
        FacetedTimeSeriesChart,
    )
    from .plotly.panel import (
        EntityEffectsPlot,
        TimeEffectsPlot,
        BetweenWithinPlot,
        PanelStructurePlot,
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
    _has_plotly_charts = False

# High-level convenience APIs (Phase 2)
try:
    from .api import (
        create_comparison_charts,
        create_residual_diagnostics,
        create_validation_charts,
        export_chart,
        export_charts,
        export_charts_multiple_formats,
        create_panel_charts,
        create_entity_effects_plot,
        create_time_effects_plot,
        create_between_within_plot,
        create_panel_structure_plot,
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
    # Export functions
    "export_chart",
    "export_charts",
    "export_charts_multiple_formats",
]

__version__ = "0.5.0"
