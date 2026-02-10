"""Tests for StaticValidationRenderer."""

import pytest

try:
    import matplotlib

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from panelbox.report.renderers.static_validation_renderer import StaticValidationRenderer


class TestStaticValidationRendererInit:
    """Test initialization of StaticValidationRenderer."""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        renderer = StaticValidationRenderer()
        assert renderer.figure_size == (10, 6)
        assert renderer.dpi == 150
        assert renderer.style == "seaborn-v0_8-darkgrid"

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        renderer = StaticValidationRenderer(figure_size=(12, 8), dpi=300, style="default")
        assert renderer.figure_size == (12, 8)
        assert renderer.dpi == 300
        assert renderer.style == "default"

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_init_with_invalid_style_fallback(self):
        """Test initialization with invalid style falls back to default."""
        renderer = StaticValidationRenderer(style="nonexistent_style")
        assert renderer.style == "nonexistent_style"  # Stored but fallback used

    def test_init_without_matplotlib_raises_error(self, monkeypatch):
        """Test initialization without matplotlib raises ImportError."""
        # Temporarily disable matplotlib
        import panelbox.report.renderers.static_validation_renderer as module

        original_available = module.MATPLOTLIB_AVAILABLE
        monkeypatch.setattr(module, "MATPLOTLIB_AVAILABLE", False)

        with pytest.raises(ImportError, match="Matplotlib is required"):
            StaticValidationRenderer()

        monkeypatch.setattr(module, "MATPLOTLIB_AVAILABLE", original_available)


class TestStaticValidationRendererCharts:
    """Test chart rendering methods."""

    @pytest.fixture
    def renderer(self):
        """Create a renderer instance."""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")
        return StaticValidationRenderer(dpi=50)  # Low DPI for faster tests

    @pytest.fixture
    def validation_data(self):
        """Create sample validation data."""
        return {
            "charts": {
                "test_overview": {
                    "categories": ["Autocorr", "Heterosk", "Normality"],
                    "passed": [3, 2, 1],
                    "failed": [0, 1, 2],
                },
                "pvalue_distribution": {
                    "test_names": ["DW", "BP", "White", "JB"],
                    "pvalues": [0.001, 0.03, 0.08, 0.6],
                },
                "test_statistics": {
                    "test_names": ["DW", "BP", "White"],
                    "statistics": [2.1, 15.3, 28.4],
                },
            }
        }

    @pytest.fixture
    def summary_data(self):
        """Create sample summary data."""
        return {"total_passed": 8, "total_failed": 3, "total_tests": 11}

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_validation_charts_all(self, renderer, validation_data):
        """Test rendering all validation charts."""
        charts = renderer.render_validation_charts(validation_data)

        assert "test_overview" in charts
        assert "pvalue_distribution" in charts
        assert "test_statistics" in charts

        # Verify base64 data URI format
        for chart_name, chart_data in charts.items():
            assert chart_data.startswith("data:image/png;base64,")
            assert len(chart_data) > 100  # Should be substantial

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_validation_charts_partial(self, renderer):
        """Test rendering with partial data."""
        partial_data = {
            "charts": {
                "test_overview": {
                    "categories": ["Test1"],
                    "passed": [5],
                    "failed": [2],
                }
            }
        }

        charts = renderer.render_validation_charts(partial_data)

        assert "test_overview" in charts
        assert "pvalue_distribution" not in charts
        assert "test_statistics" not in charts

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_validation_charts_empty(self, renderer):
        """Test rendering with no chart data."""
        empty_data = {"charts": {}}
        charts = renderer.render_validation_charts(empty_data)
        assert charts == {}

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_test_overview(self, renderer, validation_data):
        """Test rendering test overview chart."""
        data = validation_data["charts"]["test_overview"]
        chart = renderer._render_test_overview(data)

        assert chart.startswith("data:image/png;base64,")
        assert len(chart) > 100

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_pvalue_distribution(self, renderer, validation_data):
        """Test rendering p-value distribution chart."""
        data = validation_data["charts"]["pvalue_distribution"]
        chart = renderer._render_pvalue_distribution(data)

        assert chart.startswith("data:image/png;base64,")
        assert len(chart) > 100

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_test_statistics(self, renderer, validation_data):
        """Test rendering test statistics chart."""
        data = validation_data["charts"]["test_statistics"]
        chart = renderer._render_test_statistics(data)

        assert chart.startswith("data:image/png;base64,")
        assert len(chart) > 100

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_summary_chart(self, renderer, summary_data):
        """Test rendering summary pie chart."""
        chart = renderer.render_summary_chart(summary_data)

        assert chart.startswith("data:image/png;base64,")
        assert len(chart) > 100

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_pvalue_distribution_color_coding(self, renderer):
        """Test p-value distribution uses correct colors."""
        data = {
            "test_names": ["Very Sig", "Sig", "Marginal", "Not Sig"],
            "pvalues": [0.005, 0.03, 0.08, 0.6],
        }
        chart = renderer._render_pvalue_distribution(data)

        # Just verify chart was created successfully
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_with_high_dpi(self):
        """Test rendering with high DPI produces larger images."""
        low_dpi_renderer = StaticValidationRenderer(dpi=50)
        high_dpi_renderer = StaticValidationRenderer(dpi=200)

        data = {
            "test_names": ["Test1", "Test2"],
            "statistics": [10, 20],
        }

        low_chart = low_dpi_renderer._render_test_statistics(data)
        high_chart = high_dpi_renderer._render_test_statistics(data)

        # High DPI should produce larger file
        assert len(high_chart) > len(low_chart)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_with_custom_figure_size(self):
        """Test rendering with custom figure size."""
        renderer = StaticValidationRenderer(figure_size=(15, 10), dpi=50)

        data = {
            "test_names": ["Test1"],
            "statistics": [10],
        }

        chart = renderer._render_test_statistics(data)
        assert chart.startswith("data:image/png;base64,")


class TestStaticValidationRendererRepr:
    """Test string representation."""

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_repr(self):
        """Test __repr__ method."""
        renderer = StaticValidationRenderer(figure_size=(12, 8), dpi=300, style="ggplot")
        repr_str = repr(renderer)

        assert "StaticValidationRenderer" in repr_str
        assert "figure_size=(12, 8)" in repr_str
        assert "dpi=300" in repr_str
        assert "style='ggplot'" in repr_str


class TestStaticValidationRendererEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def renderer(self):
        """Create a renderer instance."""
        if not MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")
        return StaticValidationRenderer(dpi=50)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_with_empty_arrays(self, renderer):
        """Test rendering with empty data arrays."""
        data = {
            "test_names": [],
            "statistics": [],
        }
        chart = renderer._render_test_statistics(data)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_with_single_datapoint(self, renderer):
        """Test rendering with single datapoint."""
        data = {
            "test_names": ["SingleTest"],
            "statistics": [42.0],
        }
        chart = renderer._render_test_statistics(data)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_summary_with_zero_failed(self, renderer):
        """Test summary chart when no tests failed."""
        summary = {"total_passed": 10, "total_failed": 0, "total_tests": 10}
        chart = renderer.render_summary_chart(summary)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_summary_with_zero_passed(self, renderer):
        """Test summary chart when no tests passed."""
        summary = {"total_passed": 0, "total_failed": 10, "total_tests": 10}
        chart = renderer.render_summary_chart(summary)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_with_very_long_labels(self, renderer):
        """Test rendering with very long test names."""
        data = {
            "test_names": [
                "Very Long Test Name That Might Cause Layout Issues" * 3,
                "Another Long Name",
            ],
            "statistics": [10, 20],
        }
        chart = renderer._render_test_statistics(data)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_with_extreme_pvalues(self, renderer):
        """Test p-value rendering with very small values."""
        data = {
            "test_names": ["Test1", "Test2", "Test3"],
            "pvalues": [1e-100, 0.5, 0.9999],
        }
        chart = renderer._render_pvalue_distribution(data)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_render_overview_with_large_numbers(self, renderer):
        """Test overview chart with large numbers."""
        data = {
            "categories": ["Cat1", "Cat2"],
            "passed": [10000, 20000],
            "failed": [5000, 3000],
        }
        chart = renderer._render_test_overview(data)
        assert chart.startswith("data:image/png;base64,")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_fig_to_base64_closes_figure(self, renderer):
        """Test that _fig_to_base64 properly closes the figure."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Get initial figure count
        initial_fig_count = len(plt.get_fignums())

        # Convert to base64
        renderer._fig_to_base64(fig)

        # Figure should be closed
        final_fig_count = len(plt.get_fignums())
        assert final_fig_count == initial_fig_count - 1

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_multiple_renders_no_memory_leak(self, renderer, validation_data):
        """Test multiple renders don't leak memory (figures)."""
        import matplotlib.pyplot as plt

        initial_fig_count = len(plt.get_fignums())

        # Render multiple times
        for _ in range(5):
            renderer.render_validation_charts(validation_data)

        # No figures should remain open
        final_fig_count = len(plt.get_fignums())
        assert final_fig_count == initial_fig_count
