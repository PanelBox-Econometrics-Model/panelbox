================================================================================
PANELBOX VISUALIZATION MODULE - COMPREHENSIVE SUMMARY
================================================================================

Generated: 2025-02-17
Location: /home/guhaase/projetos/panelbox/panelbox/visualization

================================================================================
1. TOP-LEVEL MODULE STRUCTURE
================================================================================

MAIN DIRECTORIES:
/home/guhaase/projetos/panelbox/panelbox/
├── visualization/                    (9 directories, 38+ modules)
│   ├── config/                       (Chart configuration)
│   ├── matplotlib/                   (Static Matplotlib implementations)
│   ├── plotly/                       (Interactive Plotly implementations)
│   ├── quantile/                     (Quantile regression visualizations)
│   ├── transformers/                 (Data transformation layer)
│   ├── utils/                        (Utility functions)
│   ├── api.py                        (High-level convenience APIs)
│   ├── base.py                       (Abstract base classes)
│   ├── factory.py                    (Factory pattern implementation)
│   ├── registry.py                   (Chart registry system)
│   ├── themes.py                     (Visual theme system)
│   ├── exceptions.py                 (Custom exceptions)
│   ├── spatial_plots.py              (Spatial model visualizations)
│   ├── var_plots.py                  (VAR model visualizations)
│   └── __init__.py                   (Module initialization)

TOTAL VISUALIZATION CODE: ~4,863 lines of Python

================================================================================
2. VISUALIZATION MODULES AND THEIR PURPOSES
================================================================================

A. CORE ARCHITECTURE MODULES:

1. api.py (44,405 bytes)
   - High-level convenience APIs for common use cases
   - Key Functions:
     * create_validation_charts()
     * create_residual_diagnostics()
     * create_comparison_charts()
     * create_panel_charts()
     * create_panel_structure_plot()
     * create_entity_effects_plot()
     * create_time_effects_plot()
     * create_between_within_plot()
     * create_acf_pacf_plot()
     * create_unit_root_test_plot()
     * create_cointegration_heatmap()
     * create_cross_sectional_dependence_plot()
     * export_chart()
     * export_charts()
     * export_charts_multiple_formats()

2. base.py (23,615 bytes)
   - Abstract base classes for all chart types
   - Classes:
     * BaseChart (Abstract)
     * PlotlyChartBase (Plotly charts)
     * MatplotlibChartBase (Matplotlib charts)
     * NumpyEncoder (JSON encoder for numpy types)
   - Template Method Pattern: Consistent chart creation workflow
   - Capabilities: HTML, JSON, image exports (PNG, SVG, PDF)

3. registry.py (8,211 bytes)
   - Registry pattern for chart management
   - Classes:
     * ChartRegistry (centralized chart storage)
   - Decorator: @register_chart() for declarative registration
   - Features: Get, list, check, unregister, get_info for charts

4. factory.py (7,106 bytes)
   - Factory pattern for centralized chart creation
   - Class: ChartFactory
   - Methods:
     * create() - Create single chart
     * create_multiple() - Batch create charts
     * list_available_charts()
     * get_chart_info()

5. themes.py (12,091 bytes)
   - Visual theming system
   - Class: Theme (dataclass)
   - Pre-built Themes:
     * PROFESSIONAL_THEME (default)
     * ACADEMIC_THEME
     * PRESENTATION_THEME
   - Customization: Color schemes, fonts, layouts, Plotly/Matplotlib templates

B. CONFIGURATION MODULES:

6. config/chart_config.py
   - ChartConfig dataclass
   - Properties: width, height, title, axes labels, legend, responsiveness
   - Usage: Per-chart customization independent of theme

7. config/color_schemes.py
   - Color palette definitions
   - Support for colorblind-friendly palettes

C. CHART IMPLEMENTATION MODULES:

8. plotly/ (7 modules)
   Modules: basic.py, comparison.py, correlation.py, distribution.py,
            econometric_tests.py, panel.py, residuals.py, timeseries.py,
            validation.py

   Contains 35 registered chart implementations:
   - Basic charts (Bar, Line)
   - Residual diagnostics (QQ plot, Residuals vs Fitted, Scale-Location, etc.)
   - Model comparison (Coefficients, Forest plot, Model fit, IC)
   - Distribution charts (Histogram, KDE, Violin, Box)
   - Correlation charts (Heatmap, Pairwise)
   - Time series charts (Panel, Trend, Faceted)
   - Panel-specific (Entity effects, Time effects, Between-within, Structure)
   - Econometric tests (ACF/PACF, Unit root, Cointegration, CD test)
   - Validation charts (Test overview, P-value distribution, Test statistics, Dashboard)

9. quantile/ (5 modules)
   Modules: __init__.py, advanced_plots.py, interactive.py,
            process_plots.py, surface_plots.py, themes.py

   Classes:
   - QuantileVisualizer
   - SurfacePlotter
   - InteractivePlotter
   - PublicationTheme

   Functions:
   - quantile_process_plot()
   - residual_plot()
   - qq_plot()

D. DATA TRANSFORMATION MODULES:

10. transformers/ (4 modules)
    Modules: validation.py, residuals.py, comparison.py, panel.py

    Purpose: Transform model results and raw data into chart-friendly format
    Classes:
    - ValidationDataTransformer
    - ResidualDataTransformer
    - ComparisonDataTransformer
    - PanelDataTransformer

E. UTILITY MODULES:

11. utils/ (3 modules)
    Modules: chart_selector.py, theme_loader.py, __init__.py

    Functions:
    - Chart selection helpers
    - Theme loading and caching

F. SPECIALIZED MODULES:

12. spatial_plots.py (25,970 bytes)
    - Spatial econometric visualizations
    - Spatial lag/error model plots

13. var_plots.py (34,913 bytes)
    - VAR (Vector Autoregression) model visualizations
    - Impulse response functions, variance decomposition, etc.

14. exceptions.py (10,200 bytes)
    - Custom exceptions for visualization module
    - Error handling classes

================================================================================
3. KEY CLASSES AND FUNCTIONS FOR VISUALIZATION
================================================================================

A. MAIN FACTORY INTERFACE:

ChartFactory
├── create(chart_type, data, theme, config, **kwargs)
├── create_multiple(chart_specs, common_theme)
├── list_available_charts()
└── get_chart_info(chart_type)

B. BASE CLASSES (for custom chart development):

BaseChart (Abstract)
├── create(data, **kwargs) - Template Method
├── _validate_data(data)
├── _preprocess_data(data)
├── _create_figure(data, **kwargs) [ABSTRACT]
├── _apply_theme(figure)
├── _finalize()
├── to_json()
├── to_html()
└── to_dict()

PlotlyChartBase (extends BaseChart)
├── to_image(format, width, height, scale)
├── save_image(file_path, format, width, height, scale)
├── to_png(width, height, scale)
├── to_svg(width, height)
└── to_pdf(width, height)

MatplotlibChartBase (extends BaseChart)
├── to_base64(format, dpi)
└── (base64-embedded exports)

C. HIGH-LEVEL CONVENIENCE APIs:

VALIDATION CHARTS:
├── create_validation_charts(validation_data, theme, interactive, charts)

RESIDUAL DIAGNOSTICS:
├── create_residual_diagnostics(results, theme, charts)

MODEL COMPARISON:
├── create_comparison_charts(results_list, names, theme, charts)

PANEL ANALYSIS:
├── create_panel_charts(panel_results, chart_types, theme)
├── create_entity_effects_plot(panel_results, theme)
├── create_time_effects_plot(panel_results, theme)
├── create_between_within_plot(panel_data, variables, theme, style)
└── create_panel_structure_plot(panel_data, theme)

ECONOMETRIC TESTS:
├── create_acf_pacf_plot(residuals, max_lags, confidence_level, theme)
├── create_unit_root_test_plot(test_results, include_series, theme)
├── create_cointegration_heatmap(cointegration_results, theme)
└── create_cross_sectional_dependence_plot(cd_results, theme)

EXPORT FUNCTIONS:
├── export_chart(chart, file_path, format, width, height, scale)
├── export_charts(charts, output_dir, format, prefix, width, height, scale)
└── export_charts_multiple_formats(charts, output_dir, formats, ...)

D. THEME MANAGEMENT:

Theme (dataclass)
├── name: str
├── color_scheme: List[str]
├── font_config: Dict
├── layout_config: Dict
├── plotly_template: str
├── matplotlib_style: str
├── success_color, warning_color, danger_color, info_color
├── get_color(index)
└── to_dict()

Pre-built Themes:
├── PROFESSIONAL_THEME (modern, accessible, business-ready)
├── ACADEMIC_THEME (publication-ready, minimal)
└── PRESENTATION_THEME (bold, high-contrast)

E. REGISTRY SYSTEM:

ChartRegistry
├── register(name, chart_class)
├── get(name) -> chart_class
├── list_charts() -> sorted list
├── is_registered(name) -> bool
├── unregister(name)
├── clear()
└── get_chart_info(name)

Decorator: @register_chart('chart_name')

================================================================================
4. AVAILABLE CHART TYPES (35 REGISTERED CHARTS)
================================================================================

BASIC CHARTS (2):
  1. bar_chart
  2. line_chart

RESIDUAL DIAGNOSTICS (7):
  3. residual_qq_plot
  4. residual_vs_fitted
  5. residual_scale_location
  6. residual_vs_leverage
  7. residual_timeseries
  8. residual_distribution
  9. residual_partial_regression

MODEL COMPARISON (4):
  10. comparison_coefficients
  11. comparison_forest_plot
  12. comparison_model_fit
  13. comparison_ic

DISTRIBUTION CHARTS (4):
  14. distribution_histogram
  15. distribution_kde
  16. distribution_violin
  17. distribution_boxplot

CORRELATION CHARTS (2):
  18. correlation_heatmap
  19. correlation_pairwise

TIME SERIES CHARTS (3):
  20. timeseries_panel
  21. timeseries_trend
  22. timeseries_faceted

PANEL-SPECIFIC CHARTS (4):
  23. panel_entity_effects
  24. panel_time_effects
  25. panel_between_within
  26. panel_structure

VALIDATION CHARTS (5):
  27. validation_test_overview
  28. validation_pvalue_distribution
  29. validation_test_statistics
  30. validation_comparison_heatmap
  31. validation_dashboard

ECONOMETRIC TEST CHARTS (4):
  32. acf_pacf_plot
  33. unit_root_test_plot
  34. cointegration_heatmap
  35. cross_sectional_dependence_plot

================================================================================
5. VISUALIZATION EXAMPLES AND DEMOS
================================================================================

A. EXAMPLE VISUALIZATION UTILITY FILES:

/home/guhaase/projetos/panelbox/examples/
├── utils/visualization/
│   ├── diagnostic_plots.py
│   ├── comparison_plots.py
│   └── panel_plots.py
├── count/utils/visualization_helpers.py
├── discrete/utils/visualization_helpers.py
├── spatial/scripts/visualization_utils.py
└── standard_errors/utils/plotting.py

B. DEVELOPMENT DOCUMENTATION:

/home/guhaase/projetos/panelbox/desenvolvimento/
├── JUPYTERS/PLANEJAMENTO/15_visualizacao/
│   └── planejamento_15_visualizacao.md
└── JUPYTERS/PROMPT/
    └── 15_visualizacao_relatorios.md

C. BENCHMARK PERFORMANCE:

/home/guhaase/projetos/panelbox/benchmarks/
└── visualization_performance.py

D. EXAMPLE NOTEBOOKS (in examples/jupyter/, examples/discrete/, etc.):

Multiple Jupyter notebooks demonstrating:
- Basic usage
- Complete workflows
- Residual diagnostics
- Model comparison
- Validation reporting
- Report generation

================================================================================
6. DESIGN PATTERNS AND ARCHITECTURE
================================================================================

A. DESIGN PATTERNS IMPLEMENTED:

1. Factory Pattern
   - ChartFactory for centralized creation
   - Single entry point for all chart types

2. Registry Pattern
   - ChartRegistry for chart management
   - @register_chart() decorator for registration
   - Dynamic chart discovery

3. Strategy Pattern
   - Multiple rendering backends (Plotly, Matplotlib)
   - Pluggable theme system

4. Template Method Pattern
   - BaseChart.create() orchestrates process
   - Subclasses implement _create_figure()
   - Consistent workflow across all charts

5. Decorator Pattern
   - Theme application as decoration
   - Custom configuration overlay

B. ARCHITECTURE LAYERS:

1. High-Level API Layer (api.py)
   └─ Easy-to-use convenience functions for common tasks

2. Factory Layer (factory.py)
   └─ Centralized chart creation with theme resolution

3. Chart Implementation Layer (plotly/*, matplotlib/*)
   └─ Specific chart implementations for each type

4. Base Classes Layer (base.py)
   └─ Abstract interfaces and common functionality

5. Data Transformation Layer (transformers/*)
   └─ Convert model results to chart-friendly format

6. Configuration Layer (themes.py, config/*)
   └─ Visual theming and per-chart configuration

7. Utility Layer (utils/*, exceptions.py)
   └─ Supporting functions and error handling

================================================================================
7. KEY FEATURES AND CAPABILITIES
================================================================================

VISUALIZATION FEATURES:
✓ 35+ registered chart types
✓ 3 professional themes (Professional, Academic, Presentation)
✓ Interactive Plotly charts with hover, zoom, pan
✓ Static Matplotlib charts for publications
✓ Responsive design support
✓ Customizable colors, fonts, layouts
✓ Colorblind-friendly palettes
✓ Light/dark theme support via Plotly templates

EXPORT CAPABILITIES:
✓ HTML (interactive, standalone)
✓ JSON (for data interchange)
✓ PNG (standard resolution)
✓ SVG (vector graphics)
✓ PDF (print-ready)
✓ JPEG (compressed)
✓ WebP (modern web format)
✓ Batch export to multiple formats
✓ Custom resolution scaling (2x for retina, etc.)

DATA SUPPORT:
✓ Panel model results
✓ Validation test results
✓ Residuals and diagnostic statistics
✓ Econometric test results
✓ Time series data
✓ Spatial data (via spatial_plots.py)
✓ VAR model outputs (via var_plots.py)
✓ Quantile regression results (via quantile/)

INTEGRATION:
✓ Seamless integration with PanelBox models
✓ Transformer layer for data preprocessing
✓ Consistent API across all chart types
✓ Theme-aware design tokens
✓ Configuration at chart and global levels

================================================================================
8. USAGE EXAMPLES (QUICK REFERENCE)
================================================================================

EXAMPLE 1: Create Validation Charts
-------
from panelbox.visualization import create_validation_charts

charts = create_validation_charts(
    validation_data,
    theme='professional',
    interactive=True,
    charts=['test_overview', 'pvalue_distribution']
)
overview_html = charts['test_overview'].to_html()

EXAMPLE 2: Create Residual Diagnostics
-------
from panelbox.visualization import create_residual_diagnostics

diagnostics = create_residual_diagnostics(
    results,
    theme='academic',
    charts=['qq_plot', 'residual_vs_fitted', 'scale_location']
)
diagnostics['qq_plot'].save_image('qq_plot.png', width=800, height=600)

EXAMPLE 3: Create Model Comparison
-------
from panelbox.visualization import create_comparison_charts

comparisons = create_comparison_charts(
    [fe_results, re_results],
    names=['Fixed Effects', 'Random Effects'],
    theme='presentation'
)

EXAMPLE 4: Create Panel Charts
-------
from panelbox.visualization import create_panel_charts

charts = create_panel_charts(
    panel_results,
    chart_types=['entity_effects', 'time_effects', 'structure'],
    theme='professional'
)

EXAMPLE 5: Export Multiple Charts
-------
from panelbox.visualization import export_charts_multiple_formats

paths = export_charts_multiple_formats(
    charts,
    output_dir='output/charts',
    formats=['png', 'svg', 'pdf'],
    width=1200,
    height=800
)

EXAMPLE 6: Use ChartFactory Directly
-------
from panelbox.visualization import ChartFactory

chart = ChartFactory.create(
    chart_type='residual_qq_plot',
    data={'residuals': model.resid},
    theme='academic',
    config={'title': 'Q-Q Plot for Normality'}
)
html = chart.to_html()

EXAMPLE 7: Create Custom Theme
-------
from panelbox.visualization import Theme, ChartFactory

custom_theme = Theme(
    name='corporate',
    color_scheme=['#003366', '#FF6600', '#00CC99'],
    font_config={'family': 'Arial', 'size': 12, 'color': '#333333'},
    layout_config={'paper_bgcolor': '#FFFFFF'}
)

chart = ChartFactory.create('qq_plot', data=data, theme=custom_theme)

EXAMPLE 8: Econometric Test Visualizations
-------
from panelbox.visualization import create_acf_pacf_plot, create_unit_root_test_plot

# ACF/PACF for serial correlation
acf_chart = create_acf_pacf_plot(
    residuals,
    max_lags=20,
    confidence_level=0.95,
    theme='academic'
)

# Unit root test results
ur_chart = create_unit_root_test_plot(
    test_results,
    include_series=True,
    theme='professional'
)

================================================================================
9. FILE SIZES AND CODE METRICS
================================================================================

Core Modules:
  api.py                        44,405 bytes
  var_plots.py                  34,913 bytes
  base.py                       23,615 bytes
  spatial_plots.py              25,970 bytes
  themes.py                     12,091 bytes
  exceptions.py                 10,200 bytes
  factory.py                     7,106 bytes
  registry.py                    8,211 bytes

  Total visualization code:    ~4,863 lines (combined)

Chart Implementation Modules:
  plotly/                     Multiple modules (~1,500 lines)
  quantile/                   Multiple modules (~500 lines)
  transformers/               4 modules (~1,200 lines)

================================================================================
10. DEPENDENCIES
================================================================================

REQUIRED:
  - plotly (interactive charts)
  - matplotlib (static charts)
  - numpy (numerical operations)
  - scipy (statistical functions)

OPTIONAL:
  - kaleido (image export: PNG, SVG, PDF, JPEG)
  - seaborn (matplotlib styling)

================================================================================
11. IMPORT CONVENTIONS
================================================================================

Standard imports in panelbox.visualization:

from panelbox.visualization import (
    # Base classes
    BaseChart, PlotlyChartBase, MatplotlibChartBase,

    # Factory and Registry
    ChartFactory, ChartRegistry, register_chart,

    # Themes
    Theme, PROFESSIONAL_THEME, ACADEMIC_THEME, PRESENTATION_THEME,

    # Chart classes (optional, use factory instead)
    QQPlot, ResidualVsFittedPlot, BarChart, LineChart, etc.,

    # High-level APIs
    create_validation_charts,
    create_residual_diagnostics,
    create_comparison_charts,
    create_panel_charts,
    export_chart, export_charts, export_charts_multiple_formats,

    # Econometric test APIs
    create_acf_pacf_plot,
    create_unit_root_test_plot,
    create_cointegration_heatmap,
    create_cross_sectional_dependence_plot
)

================================================================================
END OF SUMMARY
================================================================================
