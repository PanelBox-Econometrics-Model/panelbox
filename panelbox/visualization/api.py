"""
High-level API for chart creation.

This module provides convenient functions for creating common chart combinations
without needing to work directly with the factory or registry.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

from .factory import ChartFactory
from .themes import Theme, get_theme


def create_validation_charts(
    validation_data: Union[Dict, Any],
    theme: Union[str, Theme, None] = "professional",
    interactive: bool = True,
    charts: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create validation charts from ValidationReport or dict.

    This is the main convenience function for generating validation visualizations.
    It can create individual charts or a complete dashboard.

    Parameters
    ----------
    validation_data : dict or ValidationReport
        Validation test results. If dict, must contain:
        - 'tests': list of test results
        - 'categories': dict grouping tests by category
        - 'summary': dict with overall statistics

        If ValidationReport object, will be converted automatically.

    theme : str or Theme, default='professional'
        Visual theme ('professional', 'academic', 'presentation' or Theme object)

    interactive : bool, default=True
        If True, creates Plotly charts. If False, creates static Matplotlib charts.

    charts : list of str, optional
        Which charts to generate. Options:
        - 'test_overview': Test results by category
        - 'pvalue_distribution': P-value bar chart
        - 'test_statistics': Test statistic scatter plot
        - 'comparison_heatmap': Model comparison heatmap (requires multiple models)
        - 'dashboard': All-in-one dashboard
        If None, creates all applicable charts.

    **kwargs
        Additional options:
        - alpha : float - Significance threshold (default: 0.05)
        - include_html : bool - Return HTML strings instead of chart objects
        - config : dict - Chart configuration options

    Returns
    -------
    dict
        Dictionary mapping chart names to chart objects or HTML strings

    Examples
    --------
    Create all validation charts:

    >>> from panelbox.validation import ValidationReport
    >>> from panelbox.visualization import create_validation_charts
    >>>
    >>> # After running validation tests
    >>> validation_report = ValidationReport(...)
    >>>
    >>> charts = create_validation_charts(
    ...     validation_report,
    ...     theme='professional'
    ... )
    >>>
    >>> # Use in template
    >>> overview_html = charts['test_overview'].to_html()

    Create specific charts only:

    >>> charts = create_validation_charts(
    ...     validation_report,
    ...     theme='academic',
    ...     charts=['test_overview', 'pvalue_distribution']
    ... )

    Get HTML strings directly:

    >>> html_charts = create_validation_charts(
    ...     validation_report,
    ...     theme='presentation',
    ...     include_html=True
    ... )
    >>> # html_charts['test_overview'] is already HTML string
    """
    # Import here to avoid circular imports
    from .transformers.validation import ValidationDataTransformer

    # Convert ValidationReport to dict if needed
    if not isinstance(validation_data, dict):
        # Assume it's a ValidationReport object
        transformer = ValidationDataTransformer()
        validation_data = transformer.transform(validation_data)

    # Resolve theme
    resolved_theme = get_theme(theme) if theme else None

    # Determine which charts to create
    if charts is None:
        charts = ['test_overview', 'pvalue_distribution', 'test_statistics']
        # Add dashboard if requested or if it's a comprehensive report
        if validation_data.get('tests') and len(validation_data['tests']) > 5:
            charts.append('dashboard')

    # Extract options
    alpha = kwargs.get('alpha', 0.05)
    include_html = kwargs.get('include_html', False)
    config = kwargs.get('config', {})

    # Result dictionary
    result_charts = {}

    # Create each requested chart
    for chart_name in charts:
        if chart_name == 'test_overview':
            chart_data = _prepare_test_overview_data(validation_data)
            chart = ChartFactory.create(
                chart_type='validation_test_overview',
                data=chart_data,
                theme=resolved_theme,
                config=config.get('test_overview', {})
            )
            result_charts['test_overview'] = chart.to_html() if include_html else chart

        elif chart_name == 'pvalue_distribution':
            chart_data = _prepare_pvalue_distribution_data(validation_data, alpha)
            chart = ChartFactory.create(
                chart_type='validation_pvalue_distribution',
                data=chart_data,
                theme=resolved_theme,
                config=config.get('pvalue_distribution', {})
            )
            result_charts['pvalue_distribution'] = chart.to_html() if include_html else chart

        elif chart_name == 'test_statistics':
            chart_data = _prepare_test_statistics_data(validation_data)
            chart = ChartFactory.create(
                chart_type='validation_test_statistics',
                data=chart_data,
                theme=resolved_theme,
                config=config.get('test_statistics', {})
            )
            result_charts['test_statistics'] = chart.to_html() if include_html else chart

        elif chart_name == 'comparison_heatmap':
            # Requires multiple models
            if 'models' in validation_data:
                chart_data = _prepare_comparison_heatmap_data(validation_data)
                chart = ChartFactory.create(
                    chart_type='validation_comparison_heatmap',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('comparison_heatmap', {})
                )
                result_charts['comparison_heatmap'] = chart.to_html() if include_html else chart

        elif chart_name == 'dashboard':
            chart_data = _prepare_dashboard_data(validation_data, alpha)
            chart = ChartFactory.create(
                chart_type='validation_dashboard',
                data=chart_data,
                theme=resolved_theme,
                config=config.get('dashboard', {})
            )
            result_charts['dashboard'] = chart.to_html() if include_html else chart

    return result_charts


def _prepare_test_overview_data(validation_data: Dict) -> Dict:
    """Prepare data for test overview chart."""
    # Group tests by category
    categories = {}
    for test in validation_data.get('tests', []):
        category = test.get('category', 'Other')
        if category not in categories:
            categories[category] = {'passed': 0, 'failed': 0}

        if test.get('passed', False) or test.get('pvalue', 1.0) >= test.get('alpha', 0.05):
            categories[category]['passed'] += 1
        else:
            categories[category]['failed'] += 1

    return {
        'categories': list(categories.keys()),
        'passed': [v['passed'] for v in categories.values()],
        'failed': [v['failed'] for v in categories.values()]
    }


def _prepare_pvalue_distribution_data(validation_data: Dict, alpha: float) -> Dict:
    """Prepare data for p-value distribution chart."""
    tests = validation_data.get('tests', [])

    return {
        'test_names': [t.get('name', f"Test {i}") for i, t in enumerate(tests)],
        'pvalues': [t.get('pvalue', 1.0) for t in tests],
        'alpha': alpha
    }


def _prepare_test_statistics_data(validation_data: Dict) -> Dict:
    """Prepare data for test statistics chart."""
    tests = validation_data.get('tests', [])

    return {
        'test_names': [t.get('name', f"Test {i}") for i, t in enumerate(tests)],
        'statistics': [abs(t.get('statistic', 0.0)) for t in tests],
        'categories': [t.get('category', 'Other') for t in tests],
        'pvalues': [t.get('pvalue', 1.0) for t in tests]
    }


def _prepare_comparison_heatmap_data(validation_data: Dict) -> Dict:
    """Prepare data for comparison heatmap."""
    models = validation_data.get('models', [])
    tests = validation_data.get('test_names', [])
    matrix = validation_data.get('pvalue_matrix', [])

    return {
        'models': models,
        'tests': tests,
        'matrix': matrix
    }


def _prepare_dashboard_data(validation_data: Dict, alpha: float) -> Dict:
    """Prepare data for validation dashboard."""
    return {
        'overview': _prepare_test_overview_data(validation_data),
        'pvalues': _prepare_pvalue_distribution_data(validation_data, alpha),
        'statistics': _prepare_test_statistics_data(validation_data),
        'summary': {
            'total_tests': len(validation_data.get('tests', [])),
            'passed': sum(1 for t in validation_data.get('tests', [])
                         if t.get('passed', False) or t.get('pvalue', 1.0) >= alpha),
            'failed': sum(1 for t in validation_data.get('tests', [])
                         if not (t.get('passed', False) or t.get('pvalue', 1.0) >= alpha))
        }
    }


def create_residual_diagnostics(
    results: Any,
    theme: Union[str, Theme, None] = "professional",
    charts: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create residual diagnostic plots.

    Parameters
    ----------
    results : PanelResults
        Model estimation results
    theme : str or Theme
        Visual theme
    charts : list of str, optional
        Which diagnostic plots to generate:
        - 'qq_plot': Q-Q plot for normality
        - 'residual_vs_fitted': Residuals vs fitted values
        - 'scale_location': Scale-location plot
        - 'residual_vs_leverage': Residuals vs leverage (influence plot)
        - 'residual_timeseries': Residuals over time
        - 'residual_distribution': Histogram + KDE
        If None, creates all diagnostic plots.
    **kwargs
        Additional options:
        - include_html : bool - Return HTML strings instead of chart objects
        - config : dict - Chart configuration options

    Returns
    -------
    dict
        Dictionary mapping chart names to chart objects or HTML strings

    Examples
    --------
    >>> from panelbox import FixedEffects
    >>> from panelbox.visualization import create_residual_diagnostics
    >>>
    >>> # After model estimation
    >>> model = FixedEffects(...)
    >>> results = model.fit()
    >>>
    >>> diagnostics = create_residual_diagnostics(
    ...     results,
    ...     theme='academic'
    ... )
    >>>
    >>> # Export individual plots
    >>> diagnostics['qq_plot'].to_image('qq_plot.png', width=1200, height=900)
    """
    # Import here to avoid circular imports
    from .transformers.residuals import ResidualDataTransformer

    # Transform results to residual data
    transformer = ResidualDataTransformer()

    # Resolve theme
    resolved_theme = get_theme(theme) if theme else None

    # Determine which charts to create
    if charts is None:
        charts = [
            'qq_plot',
            'residual_vs_fitted',
            'scale_location',
            'residual_vs_leverage',
            'residual_timeseries',
            'residual_distribution'
        ]

    # Extract options
    include_html = kwargs.get('include_html', False)
    config = kwargs.get('config', {})

    # Result dictionary
    result_charts = {}

    # Create each requested chart
    for chart_name in charts:
        try:
            if chart_name == 'qq_plot':
                chart_data = transformer.prepare_qq_data(results)
                chart = ChartFactory.create(
                    chart_type='residual_qq_plot',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('qq_plot', {})
                )
                result_charts['qq_plot'] = chart.to_html() if include_html else chart

            elif chart_name == 'residual_vs_fitted':
                chart_data = transformer.prepare_residual_fitted_data(results)
                chart = ChartFactory.create(
                    chart_type='residual_vs_fitted',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('residual_vs_fitted', {})
                )
                result_charts['residual_vs_fitted'] = chart.to_html() if include_html else chart

            elif chart_name == 'scale_location':
                chart_data = transformer.prepare_scale_location_data(results)
                chart = ChartFactory.create(
                    chart_type='residual_scale_location',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('scale_location', {})
                )
                result_charts['scale_location'] = chart.to_html() if include_html else chart

            elif chart_name == 'residual_vs_leverage':
                chart_data = transformer.prepare_leverage_data(results)
                chart = ChartFactory.create(
                    chart_type='residual_vs_leverage',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('residual_vs_leverage', {})
                )
                result_charts['residual_vs_leverage'] = chart.to_html() if include_html else chart

            elif chart_name == 'residual_timeseries':
                chart_data = transformer.prepare_timeseries_data(results)
                chart = ChartFactory.create(
                    chart_type='residual_timeseries',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('residual_timeseries', {})
                )
                result_charts['residual_timeseries'] = chart.to_html() if include_html else chart

            elif chart_name == 'residual_distribution':
                chart_data = transformer.prepare_distribution_data(results)
                chart = ChartFactory.create(
                    chart_type='residual_distribution',
                    data=chart_data,
                    theme=resolved_theme,
                    config=config.get('residual_distribution', {})
                )
                result_charts['residual_distribution'] = chart.to_html() if include_html else chart

        except Exception as e:
            # Log error but continue with other charts
            import warnings
            warnings.warn(f"Failed to create {chart_name}: {str(e)}")

    return result_charts


def create_comparison_charts(
    results_list: List[Any],
    names: Optional[List[str]] = None,
    theme: Union[str, Theme, None] = "professional",
    charts: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create model comparison charts.

    Parameters
    ----------
    results_list : list of PanelResults
        Multiple model results to compare
    names : list of str, optional
        Names for each model. If None, uses 'Model 1', 'Model 2', etc.
    theme : str or Theme
        Visual theme
    charts : list of str, optional
        Which comparison charts to generate:
        - 'coefficients': Coefficient comparison plot
        - 'forest_plot': Forest plot with confidence intervals
        - 'fit_comparison': Model fit comparison (R², AIC, BIC)
        - 'ic_comparison': Information criteria comparison
        If None, creates all applicable charts.
    **kwargs
        Additional options

    Returns
    -------
    dict
        Dictionary mapping chart names to chart objects

    Examples
    --------
    >>> from panelbox import FixedEffects, RandomEffects
    >>> from panelbox.visualization import create_comparison_charts
    >>>
    >>> fe_results = FixedEffects(...).fit()
    >>> re_results = RandomEffects(...).fit()
    >>>
    >>> comparisons = create_comparison_charts(
    ...     [fe_results, re_results],
    ...     names=['Fixed Effects', 'Random Effects'],
    ...     theme='presentation'
    ... )
    """
    from .transformers.comparison import ComparisonDataTransformer
    from .factory import ChartFactory

    # Get theme
    resolved_theme = get_theme(theme) if theme else None

    # Create transformer
    transformer = ComparisonDataTransformer()

    # Determine which charts to create
    if charts is None:
        charts = ['coefficients', 'fit_comparison', 'ic_comparison']

    # Create charts
    result_charts = {}

    for chart_name in charts:
        try:
            if chart_name == 'coefficients':
                # Coefficient comparison chart
                chart_data = transformer.prepare_coefficient_comparison(
                    results_list, names=names, variables=kwargs.get('variables')
                )
                chart = ChartFactory.create('comparison_coefficients', data=chart_data, theme=resolved_theme)
                result_charts['coefficients'] = chart.to_html() if kwargs.get('include_html') else chart

            elif chart_name == 'forest_plot':
                # Forest plot (single model only)
                if len(results_list) == 1:
                    chart_data = transformer.prepare_forest_plot(
                        results_list[0], variables=kwargs.get('variables')
                    )
                    chart = ChartFactory.create('comparison_forest_plot', data=chart_data, theme=resolved_theme)
                    result_charts['forest_plot'] = chart.to_html() if kwargs.get('include_html') else chart
                else:
                    warnings.warn("Forest plot requires single model. Skipping.")

            elif chart_name == 'fit_comparison':
                # Model fit comparison
                chart_data = transformer.prepare_model_fit_comparison(results_list, names=names)
                chart = ChartFactory.create('comparison_model_fit', data=chart_data, theme=resolved_theme)
                result_charts['fit_comparison'] = chart.to_html() if kwargs.get('include_html') else chart

            elif chart_name == 'ic_comparison':
                # Information criteria comparison
                chart_data = transformer.prepare_ic_comparison(results_list, names=names)
                chart = ChartFactory.create('comparison_ic', data=chart_data, theme=resolved_theme)
                result_charts['ic_comparison'] = chart.to_html() if kwargs.get('include_html') else chart

            else:
                warnings.warn(f"Unknown chart type: {chart_name}")

        except Exception as e:
            warnings.warn(f"Failed to create {chart_name}: {str(e)}")

    return result_charts


def export_charts(
    charts: Dict[str, Any],
    output_dir: str,
    format: str = 'png',
    prefix: str = '',
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0,
    **kwargs
) -> Dict[str, str]:
    """
    Batch export multiple charts to image files.

    Exports a dictionary of charts to the specified format, automatically
    creating the output directory if needed.

    Parameters
    ----------
    charts : dict
        Dictionary mapping chart names to chart objects.
        Charts must have a `save_image()` method (PlotlyChartBase).
    output_dir : str
        Output directory path
    format : str, default='png'
        Image format: 'png', 'svg', 'jpeg', 'pdf', 'webp'
    prefix : str, default=''
        Prefix to add to all output filenames
    width : int, optional
        Image width in pixels (applies to all charts)
    height : int, optional
        Image height in pixels (applies to all charts)
    scale : float, default=1.0
        Scale factor for image resolution
    **kwargs
        Additional options passed to save_image()

    Returns
    -------
    dict
        Mapping of chart names to output file paths

    Raises
    ------
    ImportError
        If kaleido is not installed

    Examples
    --------
    >>> # Create validation charts
    >>> charts = create_validation_charts(validation_data, include_html=False)
    >>>
    >>> # Export all as PNG
    >>> paths = export_charts(
    ...     charts,
    ...     output_dir='output/charts',
    ...     format='png',
    ...     width=1200,
    ...     height=800
    ... )
    >>>
    >>> # Export with prefix
    >>> paths = export_charts(
    ...     charts,
    ...     output_dir='output/charts',
    ...     format='pdf',
    ...     prefix='validation_'
    ... )
    >>>
    >>> print(paths)
    {'test_overview': 'output/charts/validation_test_overview.pdf', ...}
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exported_paths = {}

    for chart_name, chart in charts.items():
        if chart is None:
            warnings.warn(f"Chart '{chart_name}' is None, skipping export")
            continue

        # Check if chart has save_image method
        if not hasattr(chart, 'save_image'):
            warnings.warn(
                f"Chart '{chart_name}' does not have save_image() method. "
                f"Skipping export."
            )
            continue

        try:
            # Build filename
            filename = f"{prefix}{chart_name}.{format}"
            file_path = output_path / filename

            # Export chart
            chart.save_image(
                str(file_path),
                format=format,
                width=width,
                height=height,
                scale=scale,
                **kwargs
            )

            exported_paths[chart_name] = str(file_path)

        except Exception as e:
            warnings.warn(f"Failed to export chart '{chart_name}': {str(e)}")

    return exported_paths


def export_chart(
    chart: Any,
    file_path: str,
    format: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0,
    **kwargs
) -> str:
    """
    Export a single chart to an image file.

    Convenience function for exporting a single chart.

    Parameters
    ----------
    chart : Chart
        Chart object with save_image() method
    file_path : str
        Output file path (e.g., 'chart.png')
    format : str, optional
        Image format. If None, inferred from file_path extension.
        Options: 'png', 'svg', 'jpeg', 'pdf', 'webp'
    width : int, optional
        Image width in pixels
    height : int, optional
        Image height in pixels
    scale : float, default=1.0
        Scale factor for resolution
    **kwargs
        Additional options passed to save_image()

    Returns
    -------
    str
        Output file path

    Raises
    ------
    ValueError
        If chart doesn't have save_image() method
    ImportError
        If kaleido is not installed

    Examples
    --------
    >>> # Create a chart
    >>> chart = factory.create('validation_test_overview', data=data)
    >>>
    >>> # Export as PNG
    >>> export_chart(chart, 'output/test_overview.png', width=1200)
    >>>
    >>> # Export as high-res PNG
    >>> export_chart(chart, 'output/test_overview_2x.png', scale=2.0)
    >>>
    >>> # Export as SVG
    >>> export_chart(chart, 'output/test_overview.svg')
    """
    if not hasattr(chart, 'save_image'):
        raise ValueError(
            f"Chart type {type(chart).__name__} does not have save_image() method. "
            f"Only PlotlyChartBase charts support image export."
        )

    chart.save_image(
        file_path,
        format=format,
        width=width,
        height=height,
        scale=scale,
        **kwargs
    )

    return file_path


def export_charts_multiple_formats(
    charts: Dict[str, Any],
    output_dir: str,
    formats: List[str] = ['png', 'svg'],
    prefix: str = '',
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: float = 1.0,
    **kwargs
) -> Dict[str, Dict[str, str]]:
    """
    Export multiple charts to multiple formats at once.

    Convenience function to export the same charts in different formats
    (e.g., both PNG and SVG).

    Parameters
    ----------
    charts : dict
        Dictionary mapping chart names to chart objects
    output_dir : str
        Output directory path
    formats : list of str, default=['png', 'svg']
        List of image formats to export
    prefix : str, default=''
        Prefix to add to all output filenames
    width : int, optional
        Image width in pixels (applies to all)
    height : int, optional
        Image height in pixels (applies to all)
    scale : float, default=1.0
        Scale factor for resolution
    **kwargs
        Additional options passed to save_image()

    Returns
    -------
    dict
        Nested dict: {format: {chart_name: file_path}}

    Examples
    --------
    >>> # Create charts
    >>> charts = create_validation_charts(data, include_html=False)
    >>>
    >>> # Export in multiple formats
    >>> paths = export_charts_multiple_formats(
    ...     charts,
    ...     output_dir='output/charts',
    ...     formats=['png', 'svg', 'pdf'],
    ...     width=1200,
    ...     height=800
    ... )
    >>>
    >>> print(paths['png']['test_overview'])
    'output/charts/test_overview.png'
    >>> print(paths['svg']['test_overview'])
    'output/charts/test_overview.svg'
    """
    all_paths = {}

    for fmt in formats:
        paths = export_charts(
            charts=charts,
            output_dir=output_dir,
            format=fmt,
            prefix=prefix,
            width=width,
            height=height,
            scale=scale,
            **kwargs
        )
        all_paths[fmt] = paths

    return all_paths


def create_panel_charts(
    panel_results: Any,
    chart_types: Optional[List[str]] = None,
    theme: Union[str, Theme, None] = "professional",
    include_html: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Create panel-specific charts from panel estimation results.

    This convenience function creates visualizations for panel data analysis,
    including entity effects, time effects, variance decomposition, and
    panel structure.

    Parameters
    ----------
    panel_results : PanelResults or dict
        Panel model results or panel data. Can be:
        - PanelResults object (from model estimation)
        - dict with prepared data for specific charts
        - PanelData object (for structure and variance charts)

    chart_types : list of str, optional
        Which panel charts to generate. Options:
        - 'entity_effects': Entity fixed/random effects visualization
        - 'time_effects': Time fixed effects visualization
        - 'between_within': Between-within variance decomposition
        - 'structure': Panel structure and balance heatmap

        If None, attempts to create all applicable charts based on
        available data.

    theme : str or Theme, default='professional'
        Visual theme ('professional', 'academic', 'presentation' or Theme object)

    include_html : bool, default=True
        If True, returns HTML strings. If False, returns chart objects.

    **kwargs
        Additional options:
        - sort_by : str - For entity_effects: 'magnitude', 'alphabetical', 'significance'
        - show_confidence : bool - Whether to show confidence intervals
        - significance_level : float - Significance threshold (default: 0.05)
        - max_entities : int - Maximum entities to display (for large panels)
        - chart_type : str - For between_within: 'stacked', 'side_by_side', 'scatter'
        - show_percentages : bool - Show variance percentages
        - show_statistics : bool - Show panel statistics
        - highlight_complete : bool - Highlight complete entities
        - config : dict - Chart configuration options

    Returns
    -------
    dict
        Dictionary mapping chart names to chart objects or HTML strings

    Examples
    --------
    Create all panel charts from model results:

    >>> from panelbox import PanelOLS
    >>> from panelbox.visualization import create_panel_charts
    >>>
    >>> # Estimate panel model
    >>> model = PanelOLS.from_formula('y ~ x1 + x2 + EntityEffects', data=panel_data)
    >>> results = model.fit()
    >>>
    >>> # Create all panel visualizations
    >>> charts = create_panel_charts(results, theme='academic')
    >>>
    >>> # Access individual charts
    >>> charts['entity_effects'].show()
    >>> charts['time_effects'].show()

    Create specific charts only:

    >>> charts = create_panel_charts(
    ...     results,
    ...     chart_types=['entity_effects', 'structure'],
    ...     theme='presentation',
    ...     max_entities=20  # Limit for large panels
    ... )

    Create variance decomposition from data:

    >>> charts = create_panel_charts(
    ...     panel_data,
    ...     chart_types=['between_within', 'structure'],
    ...     chart_type='scatter'  # For between_within
    ... )

    Get HTML strings directly:

    >>> html_charts = create_panel_charts(
    ...     results,
    ...     theme='professional',
    ...     include_html=True
    ... )
    >>> # html_charts['entity_effects'] is already HTML string

    Use in templates:

    >>> # In Jinja2 template:
    >>> # {{ charts.entity_effects|safe }}
    """
    from .transformers.panel import PanelDataTransformer

    # Default to all chart types if none specified
    if chart_types is None:
        # Try to detect what charts are possible
        chart_types = []

        # Check if we can create effects charts (requires model results)
        if hasattr(panel_results, 'entity_effects') or hasattr(panel_results, 'params'):
            chart_types.extend(['entity_effects', 'time_effects'])

        # Check if we can create variance/structure charts (requires data)
        if hasattr(panel_results, 'dataframe') or hasattr(panel_results, 'model'):
            chart_types.extend(['between_within', 'structure'])

        # Fallback: try all
        if not chart_types:
            chart_types = ['entity_effects', 'time_effects', 'between_within', 'structure']

    # Map chart type names to registry names
    chart_map = {
        'entity_effects': 'panel_entity_effects',
        'time_effects': 'panel_time_effects',
        'between_within': 'panel_between_within',
        'structure': 'panel_structure',
    }

    # Get theme
    if isinstance(theme, str):
        theme_obj = get_theme(theme)
    else:
        theme_obj = theme

    result_charts = {}

    for chart_type in chart_types:
        registry_name = chart_map.get(chart_type)
        if not registry_name:
            warnings.warn(f"Unknown chart type: '{chart_type}', skipping")
            continue

        try:
            # Create chart
            chart = ChartFactory.create(
                registry_name,
                data=panel_results,
                theme=theme_obj,
                **kwargs
            )

            # Convert to HTML if requested
            if include_html:
                result_charts[chart_type] = chart.to_html()
            else:
                result_charts[chart_type] = chart

        except Exception as e:
            warnings.warn(f"Failed to create {chart_type}: {str(e)}")

    return result_charts


def create_entity_effects_plot(
    panel_results: Any,
    theme: Union[str, Theme, None] = "professional",
    **kwargs
) -> Any:
    """
    Create entity effects visualization.

    Convenience function to create a single entity effects chart.

    Parameters
    ----------
    panel_results : PanelResults or dict
        Panel model results or dict with entity effects data
    theme : str or Theme
        Visual theme
    **kwargs
        Additional chart options (sort_by, show_confidence, etc.)

    Returns
    -------
    Chart
        Entity effects chart object

    Examples
    --------
    >>> from panelbox.visualization import create_entity_effects_plot
    >>>
    >>> chart = create_entity_effects_plot(
    ...     panel_results,
    ...     theme='academic',
    ...     sort_by='magnitude',
    ...     max_entities=20
    ... )
    >>> chart.show()
    """
    if isinstance(theme, str):
        theme = get_theme(theme)

    # Transform PanelResults to entity effects data if needed
    if not isinstance(panel_results, dict):
        from .transformers.panel import PanelDataTransformer
        panel_results = PanelDataTransformer.extract_entity_effects(panel_results)

    return ChartFactory.create(
        'panel_entity_effects',
        data=panel_results,
        theme=theme,
        **kwargs
    )


def create_time_effects_plot(
    panel_results: Any,
    theme: Union[str, Theme, None] = "professional",
    **kwargs
) -> Any:
    """
    Create time effects visualization.

    Convenience function to create a single time effects chart.

    Parameters
    ----------
    panel_results : PanelResults or dict
        Panel model results or dict with time effects data
    theme : str or Theme
        Visual theme
    **kwargs
        Additional chart options (show_confidence, highlight_significant, etc.)

    Returns
    -------
    Chart
        Time effects chart object

    Examples
    --------
    >>> from panelbox.visualization import create_time_effects_plot
    >>>
    >>> chart = create_time_effects_plot(
    ...     panel_results,
    ...     theme='professional',
    ...     show_confidence=True,
    ...     highlight_significant=True
    ... )
    >>> chart.show()
    """
    if isinstance(theme, str):
        theme = get_theme(theme)

    # Transform PanelResults to time effects data if needed
    if not isinstance(panel_results, dict):
        from .transformers.panel import PanelDataTransformer
        panel_results = PanelDataTransformer.extract_time_effects(panel_results)

    return ChartFactory.create(
        'panel_time_effects',
        data=panel_results,
        theme=theme,
        **kwargs
    )


def create_between_within_plot(
    panel_data: Any,
    variables: Optional[List[str]] = None,
    theme: Union[str, Theme, None] = "professional",
    style: str = 'stacked',
    **kwargs
) -> Any:
    """
    Create between-within variance decomposition chart.

    Convenience function to create variance decomposition visualization.

    Parameters
    ----------
    panel_data : PanelData or DataFrame or dict
        Panel data or dict with variance decomposition
    variables : list of str, optional
        Variables to decompose (if using PanelData/DataFrame)
    theme : str or Theme
        Visual theme
    style : str, default='stacked'
        Chart style: 'stacked', 'side_by_side', or 'scatter'
    **kwargs
        Additional chart options (show_percentages, etc.)

    Returns
    -------
    Chart
        Between-within chart object

    Examples
    --------
    >>> from panelbox.visualization import create_between_within_plot
    >>>
    >>> chart = create_between_within_plot(
    ...     panel_data,
    ...     variables=['wage', 'education'],
    ...     theme='academic',
    ...     style='stacked',
    ...     show_percentages=True
    ... )
    >>> chart.show()
    """
    if isinstance(theme, str):
        theme = get_theme(theme)

    # Prepare data if needed
    if variables is not None and not isinstance(panel_data, dict):
        from .transformers.panel import PanelDataTransformer
        panel_data = PanelDataTransformer.calculate_between_within(
            panel_data,
            variables=variables
        )

    # Pass style as config to avoid chart_type parameter conflict
    config = kwargs.pop('config', {})
    config['chart_type'] = style

    return ChartFactory.create(
        'panel_between_within',
        data=panel_data,
        theme=theme,
        config=config,
        **kwargs
    )


def create_panel_structure_plot(
    panel_data: Any,
    theme: Union[str, Theme, None] = "professional",
    **kwargs
) -> Any:
    """
    Create panel structure visualization.

    Convenience function to create panel structure/balance heatmap.

    Parameters
    ----------
    panel_data : PanelData or DataFrame or dict
        Panel data or dict with structure info
    theme : str or Theme
        Visual theme
    **kwargs
        Additional chart options (show_statistics, highlight_complete, etc.)

    Returns
    -------
    Chart
        Panel structure chart object

    Examples
    --------
    >>> from panelbox.visualization import create_panel_structure_plot
    >>>
    >>> chart = create_panel_structure_plot(
    ...     panel_data,
    ...     theme='presentation',
    ...     show_statistics=True,
    ...     highlight_complete=True
    ... )
    >>> chart.show()
    """
    if isinstance(theme, str):
        theme = get_theme(theme)

    # Transform DataFrame to structure data if needed
    if not isinstance(panel_data, dict):
        from .transformers.panel import PanelDataTransformer
        panel_data = PanelDataTransformer.analyze_panel_structure(panel_data)

    return ChartFactory.create(
        'panel_structure',
        data=panel_data,
        theme=theme,
        **kwargs
    )
