"""
Deep coverage tests for frontier/visualization/evolution_plots.py.

Targets specific uncovered lines and branch partials from the existing
test suite to push coverage from ~94% toward 100%.

Uncovered items:
- Line 290: ValueError for missing 'time' column in spaghetti
- Lines 433, 435: ValueError for missing 'time'/'entity' in heatmap
- Line 532: ValueError for missing 'time' in fanchart
- Branch 170->169: events with event_time NOT in time_points (plotly)
- Branch 203->211: show_ci=False in matplotlib timeseries
- Branch 229->228: events with event_time NOT in time_points.values (mpl)
- Branch 336->349: show_mean=False in plotly spaghetti
- Branch 390->392: highlight=None and show_mean=False in matplotlib spaghetti
- Branch 449->452: order_by=None (else branch) in heatmap
- Branch 596->600: 50 not in percentiles for fanchart matplotlib
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.visualization.evolution_plots import (
    plot_efficiency_fanchart,
    plot_efficiency_heatmap,
    plot_efficiency_spaghetti,
    plot_efficiency_timeseries,
)


@pytest.fixture(autouse=True)
def close_figs():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def efficiency_df():
    """Create efficiency DataFrame with entity/time/efficiency columns."""
    np.random.seed(42)
    entities = [f"firm_{i}" for i in range(5)]
    times = list(range(2018, 2023))
    rows = []
    for entity in entities:
        for t in times:
            rows.append(
                {
                    "entity": entity,
                    "time": t,
                    "efficiency": np.random.uniform(0.5, 0.95),
                }
            )
    return pd.DataFrame(rows)


# --- Error path tests (ValueError raises) ---


class TestErrorPaths:
    """Test ValueError raises for missing columns."""

    def test_spaghetti_missing_time_column(self):
        """Cover line 290: missing 'time' in spaghetti plot."""
        df = pd.DataFrame({"entity": ["a", "b"], "efficiency": [0.8, 0.9]})
        with pytest.raises(ValueError, match="must have 'time' column"):
            plot_efficiency_spaghetti(df)

    def test_heatmap_missing_time_column(self):
        """Cover line 433: missing 'time' in heatmap."""
        df = pd.DataFrame({"entity": ["a", "b"], "efficiency": [0.8, 0.9]})
        with pytest.raises(ValueError, match="must have 'time' column"):
            plot_efficiency_heatmap(df)

    def test_heatmap_missing_entity_column(self):
        """Cover line 435: missing entity col in heatmap."""
        df = pd.DataFrame({"time": [1, 2], "efficiency": [0.8, 0.9]})
        with pytest.raises(ValueError, match="must have 'entity' column"):
            plot_efficiency_heatmap(df)

    def test_fanchart_missing_time_column(self):
        """Cover line 532: missing 'time' in fanchart."""
        df = pd.DataFrame({"efficiency": [0.8, 0.9]})
        with pytest.raises(ValueError, match="must have 'time' column"):
            plot_efficiency_fanchart(df)


# --- Branch coverage tests ---


class TestTimeseriesBranches:
    """Test branch partials in plot_efficiency_timeseries."""

    def test_plotly_events_not_in_time_points(self, efficiency_df):
        """Cover branch 170->169: event_time NOT in time_points."""
        events = {1900: "Ancient Event", 2050: "Future Event"}
        fig = plot_efficiency_timeseries(
            efficiency_df,
            backend="plotly",
            events=events,
        )
        assert fig is not None

    def test_matplotlib_no_ci(self, efficiency_df):
        """Cover branch 203->211: show_ci=False in matplotlib."""
        fig = plot_efficiency_timeseries(
            efficiency_df,
            backend="matplotlib",
            show_ci=False,
        )
        assert fig is not None

    def test_matplotlib_events_not_in_time(self, efficiency_df):
        """Cover branch 229->228: event_time NOT in time_points.values."""
        events = {1900: "Not Found Event"}
        fig = plot_efficiency_timeseries(
            efficiency_df,
            backend="matplotlib",
            events=events,
        )
        assert fig is not None


class TestSpaghettiBranches:
    """Test branch partials in plot_efficiency_spaghetti."""

    def test_plotly_no_mean(self, efficiency_df):
        """Cover branch 336->349: show_mean=False in plotly."""
        fig = plot_efficiency_spaghetti(
            efficiency_df,
            backend="plotly",
            show_mean=False,
        )
        assert fig is not None

    def test_matplotlib_no_highlight_no_mean(self, efficiency_df):
        """Cover branch 390->392: highlight=None and show_mean=False."""
        fig = plot_efficiency_spaghetti(
            efficiency_df,
            backend="matplotlib",
            highlight=None,
            show_mean=False,
        )
        assert fig is not None


class TestHeatmapBranches:
    """Test branch partials in plot_efficiency_heatmap."""

    def test_heatmap_order_by_none(self, efficiency_df):
        """Cover branch 449->452: order_by=None (else branch) + custom title."""
        fig = plot_efficiency_heatmap(
            efficiency_df,
            backend="plotly",
            order_by=None,
            title="Custom Heatmap Title",
        )
        assert fig is not None


class TestFanchartBranches:
    """Test branch partials in plot_efficiency_fanchart."""

    def test_matplotlib_no_median_in_percentiles(self, efficiency_df):
        """Cover branch 596->600: 50 not in percentiles list."""
        fig = plot_efficiency_fanchart(
            efficiency_df,
            backend="matplotlib",
            percentiles=[10, 25, 75, 90],
        )
        assert fig is not None
