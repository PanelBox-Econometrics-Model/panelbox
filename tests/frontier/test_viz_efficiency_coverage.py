"""Tests for panelbox.frontier.visualization.efficiency_plots module.

Covers matplotlib backends, show_stats paths, ranking, boxplot with
statistical tests (ANOVA, Kruskal-Wallis), and error paths.
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
    """Create mock efficiency DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "entity": [f"firm_{i}" for i in range(n)],
            "efficiency": np.random.uniform(0.4, 0.98, n),
        }
    )


@pytest.fixture
def grouped_efficiency_df():
    """Create mock efficiency DataFrame with grouping variable."""
    np.random.seed(42)
    n = 120
    groups = np.random.choice(["North", "South", "East", "West"], n)
    return pd.DataFrame(
        {
            "entity": [f"firm_{i}" for i in range(n)],
            "efficiency": np.random.uniform(0.4, 0.98, n),
            "region": groups,
        }
    )


# --- plot_efficiency_distribution ---


class TestPlotEfficiencyDistribution:
    """Tests for plot_efficiency_distribution."""

    def test_matplotlib_default(self, efficiency_df):
        """Test distribution with matplotlib backend."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_distribution,
        )

        fig = plot_efficiency_distribution(efficiency_df, backend="matplotlib")
        assert fig is not None

    def test_matplotlib_with_stats(self, efficiency_df):
        """Test distribution with matplotlib and show_stats=True."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_distribution,
        )

        fig = plot_efficiency_distribution(
            efficiency_df, backend="matplotlib", show_stats=True, show_kde=True
        )
        assert fig is not None

    def test_matplotlib_no_stats_no_kde(self, efficiency_df):
        """Test distribution with matplotlib, no stats, no KDE."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_distribution,
        )

        fig = plot_efficiency_distribution(
            efficiency_df,
            backend="matplotlib",
            show_stats=False,
            show_kde=False,
        )
        assert fig is not None

    def test_plotly_with_stats(self, efficiency_df):
        """Test distribution with plotly and show_stats=True."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_distribution,
        )

        fig = plot_efficiency_distribution(
            efficiency_df,
            backend="plotly",
            show_stats=True,
            show_kde=True,
            title="Custom Distribution",
        )
        assert fig is not None

    def test_plotly_no_stats(self, efficiency_df):
        """Test distribution with plotly, show_stats=False."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_distribution,
        )

        fig = plot_efficiency_distribution(
            efficiency_df,
            backend="plotly",
            show_stats=False,
            show_kde=False,
        )
        assert fig is not None

    def test_invalid_backend(self, efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_distribution,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_distribution(efficiency_df, backend="bad")


# --- plot_efficiency_ranking ---


class TestPlotEfficiencyRanking:
    """Tests for plot_efficiency_ranking."""

    def test_matplotlib_backend(self, efficiency_df):
        """Test ranking with matplotlib backend."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_ranking,
        )

        fig = plot_efficiency_ranking(efficiency_df, backend="matplotlib", top_n=5, bottom_n=5)
        assert fig is not None

    def test_plotly_backend(self, efficiency_df):
        """Test ranking with plotly backend."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_ranking,
        )

        fig = plot_efficiency_ranking(
            efficiency_df,
            backend="plotly",
            top_n=5,
            bottom_n=5,
            title="Custom Ranking",
        )
        assert fig is not None

    def test_invalid_backend(self, efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_ranking,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_ranking(efficiency_df, backend="bad")


# --- plot_efficiency_boxplot ---


class TestPlotEfficiencyBoxplot:
    """Tests for plot_efficiency_boxplot."""

    def test_plotly_no_test(self, grouped_efficiency_df):
        """Test boxplot with plotly, no statistical test."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        fig = plot_efficiency_boxplot(grouped_efficiency_df, group_var="region", backend="plotly")
        assert fig is not None

    def test_plotly_anova(self, grouped_efficiency_df):
        """Test boxplot with plotly and ANOVA test."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        fig = plot_efficiency_boxplot(
            grouped_efficiency_df,
            group_var="region",
            backend="plotly",
            test="anova",
        )
        assert fig is not None

    def test_plotly_kruskal(self, grouped_efficiency_df):
        """Test boxplot with plotly and Kruskal-Wallis test."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        fig = plot_efficiency_boxplot(
            grouped_efficiency_df,
            group_var="region",
            backend="plotly",
            test="kruskal",
        )
        assert fig is not None

    def test_matplotlib_no_test(self, grouped_efficiency_df):
        """Test boxplot with matplotlib, no statistical test."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        fig = plot_efficiency_boxplot(
            grouped_efficiency_df, group_var="region", backend="matplotlib"
        )
        assert fig is not None

    def test_matplotlib_anova(self, grouped_efficiency_df):
        """Test boxplot with matplotlib and ANOVA test."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        fig = plot_efficiency_boxplot(
            grouped_efficiency_df,
            group_var="region",
            backend="matplotlib",
            test="anova",
        )
        assert fig is not None

    def test_matplotlib_kruskal(self, grouped_efficiency_df):
        """Test boxplot with matplotlib and Kruskal-Wallis test."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        fig = plot_efficiency_boxplot(
            grouped_efficiency_df,
            group_var="region",
            backend="matplotlib",
            test="kruskal",
        )
        assert fig is not None

    def test_invalid_group_var(self, grouped_efficiency_df):
        """Test invalid group_var raises ValueError."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        with pytest.raises(ValueError, match="not found"):
            plot_efficiency_boxplot(grouped_efficiency_df, group_var="nonexistent")

    def test_invalid_test(self, grouped_efficiency_df):
        """Test invalid test type raises ValueError."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        with pytest.raises(ValueError, match="Unknown test"):
            plot_efficiency_boxplot(
                grouped_efficiency_df,
                group_var="region",
                test="invalid_test",
            )

    def test_invalid_backend(self, grouped_efficiency_df):
        """Test invalid backend raises ValueError."""
        from panelbox.frontier.visualization.efficiency_plots import (
            plot_efficiency_boxplot,
        )

        with pytest.raises(ValueError, match="Unknown backend"):
            plot_efficiency_boxplot(grouped_efficiency_df, group_var="region", backend="bad")
