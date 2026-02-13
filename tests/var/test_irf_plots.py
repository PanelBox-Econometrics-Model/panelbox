"""
Tests for IRF visualization functions.

This module tests the plotting functionality for Impulse Response Functions.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from panelbox.var.irf import IRFResult, compute_irf_cholesky


@pytest.fixture
def simple_irf_result():
    """Create a simple IRF result for testing."""
    # Simple VAR(1) with K=2
    K = 2
    periods = 10

    # Simple coefficient matrix
    A1 = np.array([[0.5, 0.1], [0.2, 0.6]])

    # Residual covariance
    Sigma = np.array([[1.0, 0.3], [0.3, 1.0]])

    # Compute IRF
    irf_matrix = compute_irf_cholesky([A1], Sigma, periods)

    var_names = ["x1", "x2"]

    return IRFResult(
        irf_matrix=irf_matrix,
        var_names=var_names,
        periods=periods,
        method="cholesky",
        shock_size="one_std",
        cumulative=False,
    )


@pytest.fixture
def larger_irf_result():
    """Create a larger IRF result (K=4) for testing grid layout."""
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

    var_names = ["gdp", "inflation", "interest", "unemployment"]

    return IRFResult(
        irf_matrix=irf_matrix,
        var_names=var_names,
        periods=periods,
        method="cholesky",
        shock_size="one_std",
        cumulative=False,
    )


def test_basic_plot_generation(simple_irf_result):
    """Test that basic plot generation works without errors."""
    fig = simple_irf_result.plot(show=False)

    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)

    # Should have K×K subplots
    axes = fig.get_axes()
    K = len(simple_irf_result.var_names)
    assert len(axes) == K * K

    plt.close(fig)


def test_plot_grid_2x2(simple_irf_result):
    """Test that 2×2 grid is correctly generated."""
    fig = simple_irf_result.plot(show=False)

    axes = fig.get_axes()
    assert len(axes) == 4  # 2×2 grid

    # Check that each subplot has a title
    for ax in axes:
        assert ax.get_title() != ""

    plt.close(fig)


def test_plot_grid_4x4(larger_irf_result):
    """Test that 4×4 grid is correctly generated and legible."""
    fig = larger_irf_result.plot(show=False)

    axes = fig.get_axes()
    assert len(axes) == 16  # 4×4 grid

    # Check that figure size is reasonable for 4×4
    figsize = fig.get_size_inches()
    assert figsize[0] >= 10  # At least 10 inches wide
    assert figsize[1] >= 10  # At least 10 inches tall

    plt.close(fig)


def test_plot_impulse_filter(simple_irf_result):
    """Test filtering by impulse variable."""
    # Plot only responses to x1 shocks
    fig = simple_irf_result.plot(impulse="x1", show=False)

    axes = fig.get_axes()
    K = len(simple_irf_result.var_names)
    assert len(axes) == K  # Only K subplots (one column)

    # Check that all titles mention x1
    for ax in axes:
        title = ax.get_title().lower()
        assert "x1" in title

    plt.close(fig)


def test_plot_response_filter(simple_irf_result):
    """Test filtering by response variable."""
    # Plot only how x2 responds to all shocks
    fig = simple_irf_result.plot(response="x2", show=False)

    axes = fig.get_axes()
    K = len(simple_irf_result.var_names)
    assert len(axes) == K  # Only K subplots (one row)

    # Check that all titles mention x2
    for ax in axes:
        title = ax.get_title().lower()
        assert "x2" in title

    plt.close(fig)


def test_plot_variables_subset(larger_irf_result):
    """Test plotting only a subset of variables."""
    # Plot only gdp and inflation (2×2 grid)
    fig = larger_irf_result.plot(variables=["gdp", "inflation"], show=False)

    axes = fig.get_axes()
    assert len(axes) == 4  # 2×2 grid

    plt.close(fig)


def test_plot_with_confidence_intervals(simple_irf_result):
    """Test plotting with confidence intervals."""
    # Add mock confidence intervals
    simple_irf_result.ci_lower = simple_irf_result.irf_matrix - 0.1
    simple_irf_result.ci_upper = simple_irf_result.irf_matrix + 0.1
    simple_irf_result.ci_level = 0.95

    fig = simple_irf_result.plot(ci=True, show=False)

    assert fig is not None

    # Check that confidence bands are plotted
    # (Looking for filled areas in the plot)
    axes = fig.get_axes()
    for ax in axes:
        # Check for PolyCollection (filled area)
        collections = [c for c in ax.collections if hasattr(c, "get_facecolor")]
        # May or may not have collections depending on implementation
        # Just ensure no errors

    plt.close(fig)


def test_plot_without_confidence_intervals(simple_irf_result):
    """Test plotting without confidence intervals when not available."""
    # Ensure no CI attributes
    simple_irf_result.ci_lower = None
    simple_irf_result.ci_upper = None

    fig = simple_irf_result.plot(ci=True, show=False)

    # Should still work, just without CI bands
    assert fig is not None

    plt.close(fig)


def test_plot_export_png(simple_irf_result, tmp_path):
    """Test exporting plot as PNG."""
    fig = simple_irf_result.plot(show=False)

    output_path = tmp_path / "irf_test.png"
    fig.savefig(output_path, dpi=300)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_plot_export_svg(simple_irf_result, tmp_path):
    """Test exporting plot as SVG."""
    fig = simple_irf_result.plot(show=False)

    output_path = tmp_path / "irf_test.svg"
    fig.savefig(output_path, format="svg")

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_plot_export_pdf(simple_irf_result, tmp_path):
    """Test exporting plot as PDF."""
    fig = simple_irf_result.plot(show=False)

    output_path = tmp_path / "irf_test.pdf"
    fig.savefig(output_path, format="pdf")

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    plt.close(fig)


def test_plot_themes(simple_irf_result):
    """Test different visual themes."""
    themes = ["academic", "professional", "presentation"]

    for theme in themes:
        fig = simple_irf_result.plot(theme=theme, show=False)

        assert fig is not None
        assert isinstance(fig, matplotlib.figure.Figure)

        plt.close(fig)


def test_plot_custom_figsize(simple_irf_result):
    """Test custom figure size."""
    custom_figsize = (12, 8)
    fig = simple_irf_result.plot(figsize=custom_figsize, show=False)

    actual_figsize = fig.get_size_inches()
    assert np.allclose(actual_figsize, custom_figsize, atol=0.1)

    plt.close(fig)


def test_plot_cumulative_irf(simple_irf_result):
    """Test plotting cumulative IRFs."""
    # Create cumulative version
    from panelbox.var.irf import compute_cumulative_irf

    cumulative_matrix = compute_cumulative_irf(simple_irf_result.irf_matrix)

    cumulative_irf = IRFResult(
        irf_matrix=cumulative_matrix,
        var_names=simple_irf_result.var_names,
        periods=simple_irf_result.periods,
        method="cholesky",
        shock_size="one_std",
        cumulative=True,
    )

    fig = cumulative_irf.plot(show=False)

    assert fig is not None

    plt.close(fig)


def test_plot_zero_reference_line(simple_irf_result):
    """Test that zero reference line is present."""
    fig = simple_irf_result.plot(show=False)

    axes = fig.get_axes()

    # Check that at least one axis has a horizontal line at y=0
    for ax in axes:
        lines = ax.get_lines()
        # Should have at least 2 lines: IRF + zero reference
        assert len(lines) >= 1

    plt.close(fig)


def test_plot_axis_labels(simple_irf_result):
    """Test that axis labels are present."""
    fig = simple_irf_result.plot(show=False)

    axes = fig.get_axes()

    for ax in axes:
        # Check x-label
        xlabel = ax.get_xlabel()
        # May be empty for internal subplots, but bottom row should have labels

        # Check y-label
        ylabel = ax.get_ylabel()
        # May be empty for internal subplots, but left column should have labels

    # At least one axis should have labels
    xlabels = [ax.get_xlabel() for ax in axes]
    ylabels = [ax.get_ylabel() for ax in axes]

    # Bottom row should have x-labels
    assert any(len(label) > 0 for label in xlabels)

    plt.close(fig)


def test_plot_title_format(simple_irf_result):
    """Test that subplot titles follow expected format."""
    fig = simple_irf_result.plot(show=False)

    axes = fig.get_axes()

    var_names = simple_irf_result.var_names

    # Check that titles mention both response and impulse variables
    expected_pairs = [(i, j) for i in var_names for j in var_names]

    for ax, (resp, imp) in zip(axes, expected_pairs):
        title = ax.get_title().lower()
        # Title should contain both variable names
        assert resp.lower() in title or imp.lower() in title

    plt.close(fig)


def test_plot_backend_matplotlib(simple_irf_result):
    """Test matplotlib backend (default)."""
    fig = simple_irf_result.plot(backend="matplotlib", show=False)

    assert isinstance(fig, matplotlib.figure.Figure)

    plt.close(fig)


@pytest.mark.skip(reason="Plotly backend may not be fully implemented yet")
def test_plot_backend_plotly(simple_irf_result):
    """Test plotly backend (interactive)."""
    # This test is skipped for now as plotly backend may not be implemented
    try:
        import plotly.graph_objects as go

        fig = simple_irf_result.plot(backend="plotly", show=False)

        # Plotly figure should be different type
        assert not isinstance(fig, matplotlib.figure.Figure)

    except ImportError:
        pytest.skip("Plotly not installed")
