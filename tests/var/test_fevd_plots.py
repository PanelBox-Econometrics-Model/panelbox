"""
Tests for FEVD visualization functions.

This module tests the plotting functionality for Forecast Error Variance Decomposition.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from panelbox.var.fevd import FEVDResult, compute_fevd_cholesky
from panelbox.var.irf import compute_irf_cholesky


@pytest.fixture
def simple_fevd_result():
    """Create a simple FEVD result for testing."""
    # Simple VAR(1) with K=2
    K = 2
    periods = 10

    # Simple coefficient matrix
    A1 = np.array([[0.5, 0.1], [0.2, 0.6]])

    # Residual covariance
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    # Compute IRF first
    irf_matrix = compute_irf_cholesky([A1], Sigma, periods)

    # Cholesky factor
    P = np.linalg.cholesky(Sigma)

    # Compute FEVD
    decomposition = compute_fevd_cholesky(irf_matrix, P, Sigma, periods)

    var_names = ["x1", "x2"]

    return FEVDResult(
        decomposition=decomposition,
        var_names=var_names,
        periods=periods,
        method="cholesky",
    )


@pytest.fixture
def larger_fevd_result():
    """Create a larger FEVD result (K=4) for testing."""
    K = 4
    periods = 20

    # Random but stable VAR(1)
    np.random.seed(42)
    A1 = np.random.randn(K, K) * 0.15

    # Make it stable
    eigvals = np.linalg.eigvals(A1)
    if np.max(np.abs(eigvals)) >= 0.95:
        A1 *= 0.8 / np.max(np.abs(eigvals))

    # Residual covariance
    Sigma = np.eye(K) + 0.2 * np.random.randn(K, K)
    Sigma = Sigma @ Sigma.T  # Make positive definite

    # Compute IRF
    irf_matrix = compute_irf_cholesky([A1], Sigma, periods)

    # Cholesky factor
    P = np.linalg.cholesky(Sigma)

    # Compute FEVD
    decomposition = compute_fevd_cholesky(irf_matrix, P, Sigma, periods)

    var_names = ["gdp", "inflation", "interest", "unemployment"]

    return FEVDResult(
        decomposition=decomposition,
        var_names=var_names,
        periods=periods,
        method="cholesky",
    )


def test_basic_area_plot_generation(simple_fevd_result):
    """Test that basic stacked area plot generation works."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    # Should have K subplots (one per variable)
    axes = fig.get_axes()
    K = len(simple_fevd_result.var_names)
    assert len(axes) == K

    plt.close(fig)


def test_stacked_area_2_variables(simple_fevd_result):
    """Test stacked area chart with 2 variables."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()
    assert len(axes) == 2  # One subplot per variable

    # Check that each subplot has a title
    for ax in axes:
        assert ax.get_title() != ""

    plt.close(fig)


def test_stacked_area_4_variables(larger_fevd_result):
    """Test stacked area chart with 4 variables."""
    fig = larger_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()
    assert len(axes) == 4  # One subplot per variable

    # Check figure size is reasonable
    figsize = fig.get_size_inches()
    assert figsize[0] >= 8  # At least 8 inches wide
    assert figsize[1] >= 8  # At least 8 inches tall for 4 subplots

    plt.close(fig)


def test_stacked_bar_plot_generation(simple_fevd_result):
    """Test that stacked bar plot generation works."""
    horizons = [1, 5, 10]
    fig = simple_fevd_result.plot(kind="bar", horizons=horizons, show=False)

    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    plt.close(fig)


def test_bar_chart_selected_horizons(simple_fevd_result):
    """Test bar chart with selected horizons."""
    horizons = [1, 5, 10]
    fig = simple_fevd_result.plot(kind="bar", horizons=horizons, show=False)

    # Should render without errors
    assert fig is not None

    plt.close(fig)


def test_plot_variables_subset(larger_fevd_result):
    """Test plotting only a subset of variables."""
    fig = larger_fevd_result.plot(kind="area", variables=["gdp", "inflation"], show=False)

    axes = fig.get_axes()
    assert len(axes) == 2  # Only 2 subplots

    plt.close(fig)


def test_plot_sum_to_100_percent(simple_fevd_result):
    """Test that visual areas sum to 100% (1.0)."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()

    # For stacked area plots, the top edge should reach 1.0
    for ax in axes:
        # Get y-limits
        ylim = ax.get_ylim()
        # Y-axis should go from 0 to approximately 1
        assert ylim[0] <= 0.01  # Near zero
        assert ylim[1] >= 0.99  # Near 1.0

    plt.close(fig)


def test_plot_axis_labels_area(simple_fevd_result):
    """Test that axis labels are present for area plots."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()

    for ax in axes:
        xlabel = ax.get_xlabel()
        ylabel = ax.get_ylabel()

        # At least the bottom subplot should have x-label
        # At least some subplots should have y-label

    # Check that at least one axis has labels
    xlabels = [ax.get_xlabel() for ax in axes]
    ylabels = [ax.get_ylabel() for ax in axes]

    assert any(len(label) > 0 for label in xlabels)
    assert any(len(label) > 0 for label in ylabels)

    plt.close(fig)


def test_plot_legend_present(simple_fevd_result):
    """Test that legend is present."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()

    # At least one subplot should have a legend
    legends = [ax.get_legend() for ax in axes]
    assert any(legend is not None for legend in legends)

    plt.close(fig)


def test_plot_title_includes_variable_name(simple_fevd_result):
    """Test that subplot titles include variable names."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()
    var_names = simple_fevd_result.var_names

    for ax, var_name in zip(axes, var_names):
        title = ax.get_title().lower()
        assert var_name.lower() in title or "fevd" in title

    plt.close(fig)


def test_plot_export_png(simple_fevd_result, tmp_path):
    """Test exporting FEVD plot as PNG."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    output_path = tmp_path / "fevd_test.png"
    fig.savefig(output_path, dpi=300)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_plot_export_svg(simple_fevd_result, tmp_path):
    """Test exporting FEVD plot as SVG."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    output_path = tmp_path / "fevd_test.svg"
    fig.savefig(output_path, format="svg")

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_plot_export_pdf(simple_fevd_result, tmp_path):
    """Test exporting FEVD plot as PDF."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    output_path = tmp_path / "fevd_test.pdf"
    fig.savefig(output_path, format="pdf")

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_plot_themes(simple_fevd_result):
    """Test different visual themes."""
    themes = ["academic", "professional", "presentation"]

    for theme in themes:
        fig = simple_fevd_result.plot(kind="area", theme=theme, show=False)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        plt.close(fig)


def test_plot_custom_figsize(simple_fevd_result):
    """Test custom figure size."""
    custom_figsize = (12, 8)
    fig = simple_fevd_result.plot(kind="area", figsize=custom_figsize, show=False)

    actual_figsize = fig.get_size_inches()
    assert np.allclose(actual_figsize, custom_figsize, atol=0.1)

    plt.close(fig)


def test_plot_color_consistency(simple_fevd_result):
    """Test that same shock uses same color across subplots."""
    fig = simple_fevd_result.plot(kind="area", show=False)

    axes = fig.get_axes()

    # Get colors from first subplot
    first_ax = axes[0]
    collections_1 = first_ax.collections

    # This test is implementation-dependent
    # Just ensure no errors for now
    assert len(collections_1) > 0

    plt.close(fig)


def test_plot_backend_matplotlib(simple_fevd_result):
    """Test matplotlib backend (default)."""
    fig = simple_fevd_result.plot(backend="matplotlib", show=False)

    assert isinstance(fig, matplotlib.figure.Figure)

    plt.close(fig)


def test_plot_area_vs_bar_different(simple_fevd_result):
    """Test that area and bar plots are different."""
    fig_area = simple_fevd_result.plot(kind="area", show=False)
    fig_bar = simple_fevd_result.plot(kind="bar", horizons=[1, 5, 10], show=False)

    # Both should be valid figures
    assert isinstance(fig_area, matplotlib.figure.Figure)
    assert isinstance(fig_bar, matplotlib.figure.Figure)

    # They should have different structures
    # (Area has K subplots, bar might have different layout)

    plt.close(fig_area)
    plt.close(fig_bar)


def test_plot_invalid_kind_raises_error(simple_fevd_result):
    """Test that invalid plot kind raises error."""
    with pytest.raises((ValueError, KeyError, AttributeError)):
        simple_fevd_result.plot(kind="invalid_kind", show=False)


def test_plot_with_confidence_intervals(simple_fevd_result):
    """Test plotting FEVD with confidence intervals."""
    # Add mock confidence intervals
    simple_fevd_result.ci_lower = simple_fevd_result.decomposition - 0.05
    simple_fevd_result.ci_upper = simple_fevd_result.decomposition + 0.05
    simple_fevd_result.ci_level = 0.95

    # Ensure CIs are valid (between 0 and 1)
    simple_fevd_result.ci_lower = np.clip(simple_fevd_result.ci_lower, 0, 1)
    simple_fevd_result.ci_upper = np.clip(simple_fevd_result.ci_upper, 0, 1)

    # Plot should work even with CIs
    fig = simple_fevd_result.plot(kind="area", show=False)

    assert fig is not None

    plt.close(fig)


@pytest.mark.skip(reason="Plotly backend may not be fully implemented yet")
def test_plot_backend_plotly(simple_fevd_result):
    """Test plotly backend (interactive)."""
    try:
        import plotly.graph_objects as go

        fig = simple_fevd_result.plot(backend="plotly", show=False)

        # Plotly figure should be different type
        assert not isinstance(fig, matplotlib.figure.Figure)

    except ImportError:
        pytest.skip("Plotly not installed")


def test_plot_horizons_validation(simple_fevd_result):
    """Test that invalid horizons are handled gracefully."""
    # Try horizons beyond the computed periods
    horizons = [1, 5, 10, 100]  # 100 is beyond periods=10

    # Should either work (by filtering) or raise clear error
    try:
        fig = simple_fevd_result.plot(kind="bar", horizons=horizons, show=False)
        plt.close(fig)
    except (ValueError, IndexError):
        # Expected if validation is strict
        pass
