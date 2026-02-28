"""Coverage tests for panelbox.visualization.__init__.py module.

Targets uncovered lines: 116-152, 175-191, 288-307
The __init__.py has conditional imports and registry initialization.
"""

import importlib.util


class TestVisualizationModuleImports:
    """Test that the visualization module imports correctly."""

    def test_import_base_classes(self):
        from panelbox.visualization import (
            BaseChart,
            MatplotlibChartBase,
            PlotlyChartBase,
        )

        assert BaseChart is not None
        assert PlotlyChartBase is not None
        assert MatplotlibChartBase is not None

    def test_import_factory_registry(self):
        from panelbox.visualization import ChartFactory, ChartRegistry, register_chart

        assert ChartFactory is not None
        assert ChartRegistry is not None
        assert register_chart is not None

    def test_import_themes(self):
        from panelbox.visualization import (
            ACADEMIC_THEME,
            PRESENTATION_THEME,
            PROFESSIONAL_THEME,
            Theme,
        )

        assert PROFESSIONAL_THEME is not None
        assert ACADEMIC_THEME is not None
        assert PRESENTATION_THEME is not None
        assert Theme is not None

    def test_plotly_charts_available(self):
        from panelbox.visualization import _has_plotly_charts

        # Should be True if plotly is installed
        assert isinstance(_has_plotly_charts, bool)
        if _has_plotly_charts:
            from panelbox.visualization import BarChart, LineChart

            assert BarChart is not None
            assert LineChart is not None

    def test_api_available(self):
        from panelbox.visualization import _has_api

        assert isinstance(_has_api, bool)
        if _has_api:
            from panelbox.visualization import (
                create_validation_charts,
                export_chart,
                export_charts,
            )

            assert create_validation_charts is not None
            assert export_chart is not None
            assert export_charts is not None

    def test_all_exports_defined(self):
        import panelbox.visualization as viz

        assert hasattr(viz, "__all__")
        assert len(viz.__all__) > 40

    def test_version_defined(self):
        import panelbox.visualization as viz

        assert hasattr(viz, "__version__")

    def test_chart_registry_populated(self):
        from panelbox.visualization import ChartRegistry

        charts = ChartRegistry.list_charts()
        assert len(charts) > 0

    def test_plotly_chart_classes(self):
        """Test all plotly chart classes are importable."""
        from panelbox.visualization import (
            EntityEffectsPlot,
            QQPlot,
        )

        # If plotly is installed, these should not be None
        if importlib.util.find_spec("plotly") is not None:
            assert QQPlot is not None
            assert EntityEffectsPlot is not None

    def test_validation_chart_classes(self):
        """Test validation chart imports."""
        from panelbox.visualization import (
            TestOverviewChart,
        )

        if importlib.util.find_spec("plotly") is not None:
            assert TestOverviewChart is not None

    def test_econometric_test_chart_classes(self):
        """Test econometric test chart imports."""
        from panelbox.visualization import (
            ACFPACFPlot,
        )

        if importlib.util.find_spec("plotly") is not None:
            assert ACFPACFPlot is not None

    def test_timeseries_chart_classes(self):
        """Test timeseries chart imports."""
        from panelbox.visualization import (
            PanelTimeSeriesChart,
        )

        if importlib.util.find_spec("plotly") is not None:
            assert PanelTimeSeriesChart is not None

    def test_between_within_plot_import(self):
        from panelbox.visualization import BetweenWithinPlot

        if importlib.util.find_spec("plotly") is not None:
            assert BetweenWithinPlot is not None

    def test_api_functions_import(self):
        from panelbox.visualization import (
            create_acf_pacf_plot,
            create_between_within_plot,
            create_cointegration_heatmap,
            create_cross_sectional_dependence_plot,
            create_entity_effects_plot,
            create_panel_charts,
            create_panel_structure_plot,
            create_time_effects_plot,
            create_unit_root_test_plot,
            export_charts_multiple_formats,
        )

        assert create_panel_charts is not None
        assert create_entity_effects_plot is not None
        assert create_time_effects_plot is not None
        assert create_between_within_plot is not None
        assert create_panel_structure_plot is not None
        assert create_acf_pacf_plot is not None
        assert create_unit_root_test_plot is not None
        assert create_cointegration_heatmap is not None
        assert create_cross_sectional_dependence_plot is not None
        assert export_charts_multiple_formats is not None

    def test_initialize_chart_registry(self):
        from panelbox.visualization import _initialize_chart_registry

        # Should not raise
        _initialize_chart_registry()

    def test_partial_regression_plot_import(self):
        from panelbox.visualization import PartialRegressionPlot

        if importlib.util.find_spec("plotly") is not None:
            assert PartialRegressionPlot is not None
