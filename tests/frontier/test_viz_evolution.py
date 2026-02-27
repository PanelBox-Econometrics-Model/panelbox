"""Tests for panelbox.frontier.visualization.evolution_plots module.

Covers all 4 functions with both plotly and matplotlib backends,
plus error paths and optional parameters.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def efficiency_df():
    """Create mock efficiency DataFrame for panel data."""
    np.random.seed(42)
    entities = ["A", "B", "C", "D", "E"]
    periods = list(range(2010, 2020))
    rows = []
    for e in entities:
        for t in periods:
            rows.append({"entity": e, "time": t, "efficiency": np.random.uniform(0.5, 0.95)})
    return pd.DataFrame(rows)


# --- plot_efficiency_timeseries ---


class TestPlotEfficiencyTimeseries:
    """Tests for plot_efficiency_timeseries."""

    def test_plotly_default(self, efficiency_df):
        """Smoke test: timeseries with plotly backend (default)."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        fig = plot_efficiency_timeseries(efficiency_df, backend="plotly")
        assert fig is not None

    def test_matplotlib_default(self, efficiency_df):
        """Smoke test: timeseries with matplotlib backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        fig = plot_efficiency_timeseries(efficiency_df, backend="matplotlib")
        assert fig is not None

    def test_plotly_all_options(self, efficiency_df):
        """Test plotly with show_ci, show_range, show_median, events."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        fig = plot_efficiency_timeseries(
            efficiency_df,
            backend="plotly",
            show_ci=True,
            show_range=True,
            show_median=True,
            events={2012: "Event A", 2015: "Event B"},
            title="Custom Title",
        )
        assert fig is not None

    def test_matplotlib_all_options(self, efficiency_df):
        """Test matplotlib with show_ci, show_range, show_median, events."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        fig = plot_efficiency_timeseries(
            efficiency_df,
            backend="matplotlib",
            show_ci=True,
            show_range=True,
            show_median=True,
            events={2012: "Event A", 2015: "Event B"},
            title="Custom Title",
        )
        assert fig is not None

    def test_no_ci_no_range(self, efficiency_df):
        """Test with show_ci=False and show_range=False."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        fig = plot_efficiency_timeseries(
            efficiency_df, backend="plotly", show_ci=False, show_range=False
        )
        assert fig is not None

    def test_invalid_backend(self, efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_timeseries(efficiency_df, backend="invalid")

    def test_missing_time_column(self):
        """Test missing 'time' column raises ValueError."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_timeseries,
        )

        df = pd.DataFrame({"efficiency": [0.5, 0.6]})
        with pytest.raises(ValueError, match="must have 'time' column"):
            plot_efficiency_timeseries(df)


# --- plot_efficiency_spaghetti ---


class TestPlotEfficiencySpaghetti:
    """Tests for plot_efficiency_spaghetti."""

    def test_plotly_default(self, efficiency_df):
        """Smoke test: spaghetti with plotly backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_spaghetti,
        )

        fig = plot_efficiency_spaghetti(efficiency_df, backend="plotly")
        assert fig is not None

    def test_matplotlib_default(self, efficiency_df):
        """Smoke test: spaghetti with matplotlib backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_spaghetti,
        )

        fig = plot_efficiency_spaghetti(efficiency_df, backend="matplotlib")
        assert fig is not None

    def test_plotly_highlight(self, efficiency_df):
        """Test plotly with highlighted entities."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_spaghetti,
        )

        fig = plot_efficiency_spaghetti(
            efficiency_df,
            backend="plotly",
            highlight=["A", "C"],
            show_mean=True,
            title="Custom Spaghetti",
        )
        assert fig is not None

    def test_matplotlib_highlight(self, efficiency_df):
        """Test matplotlib with highlighted entities and no mean."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_spaghetti,
        )

        fig = plot_efficiency_spaghetti(
            efficiency_df,
            backend="matplotlib",
            highlight=["A"],
            show_mean=False,
        )
        assert fig is not None

    def test_invalid_backend(self, efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_spaghetti,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_spaghetti(efficiency_df, backend="bad")

    def test_missing_entity_col(self, efficiency_df):
        """Test missing entity column raises ValueError."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_spaghetti,
        )

        with pytest.raises(ValueError, match="must have 'nonexistent' column"):
            plot_efficiency_spaghetti(efficiency_df, entity_col="nonexistent")


# --- plot_efficiency_heatmap ---


class TestPlotEfficiencyHeatmap:
    """Tests for plot_efficiency_heatmap."""

    def test_plotly_default(self, efficiency_df):
        """Smoke test: heatmap with plotly backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_heatmap,
        )

        fig = plot_efficiency_heatmap(efficiency_df, backend="plotly")
        assert fig is not None

    def test_matplotlib_default(self, efficiency_df):
        """Smoke test: heatmap with matplotlib backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_heatmap,
        )

        fig = plot_efficiency_heatmap(efficiency_df, backend="matplotlib")
        assert fig is not None

    def test_order_by_alphabetical(self, efficiency_df):
        """Test heatmap ordered alphabetically."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_heatmap,
        )

        fig = plot_efficiency_heatmap(efficiency_df, backend="plotly", order_by="alphabetical")
        assert fig is not None

    def test_order_by_none(self, efficiency_df):
        """Test heatmap with no ordering."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_heatmap,
        )

        fig = plot_efficiency_heatmap(efficiency_df, backend="plotly", order_by="none")
        assert fig is not None

    def test_invalid_backend(self, efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_heatmap,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_heatmap(efficiency_df, backend="xyz")


# --- plot_efficiency_fanchart ---


class TestPlotEfficiencyFanchart:
    """Tests for plot_efficiency_fanchart."""

    def test_plotly_default(self, efficiency_df):
        """Smoke test: fanchart with plotly backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_fanchart,
        )

        fig = plot_efficiency_fanchart(efficiency_df, backend="plotly")
        assert fig is not None

    def test_matplotlib_default(self, efficiency_df):
        """Smoke test: fanchart with matplotlib backend."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_fanchart,
        )

        fig = plot_efficiency_fanchart(efficiency_df, backend="matplotlib")
        assert fig is not None

    def test_custom_percentiles(self, efficiency_df):
        """Test fanchart with custom percentiles."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_fanchart,
        )

        fig = plot_efficiency_fanchart(
            efficiency_df,
            backend="plotly",
            percentiles=[5, 25, 50, 75, 95],
            title="Custom Fanchart",
        )
        assert fig is not None

    def test_invalid_backend(self, efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.evolution_plots import (
            plot_efficiency_fanchart,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_fanchart(efficiency_df, backend="bad")
