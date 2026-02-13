"""
Tests for causality network visualization.

This module tests the Granger causality network visualization functionality.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality_network import (
    _build_causality_graph,
    _get_layout_positions,
    plot_causality_network,
)

# Skip tests if networkx is not installed
pytest.importorskip("networkx")


@pytest.fixture
def simple_granger_matrix():
    """
    Simple 3x3 Granger causality matrix with clear patterns.

    Pattern:
    - x1 → x2 (p=0.01, strong)
    - x2 → x3 (p=0.04, moderate)
    - x3 → x1 (p=0.08, weak, not significant at 0.05)
    """
    data = np.array(
        [
            [1.0, 0.01, 0.08],  # x1: caused by x2 (sig), x3 (not sig)
            [0.20, 1.0, 0.04],  # x2: caused by x3 (sig), not by x1
            [0.50, 0.60, 1.0],  # x3: not caused by x1 or x2
        ]
    )
    return pd.DataFrame(data, index=["x1", "x2", "x3"], columns=["x1", "x2", "x3"])


@pytest.fixture
def no_causality_matrix():
    """Matrix with no significant causality relationships."""
    data = np.array(
        [
            [1.0, 0.50, 0.60],
            [0.70, 1.0, 0.80],
            [0.90, 0.85, 1.0],
        ]
    )
    return pd.DataFrame(data, index=["y1", "y2", "y3"], columns=["y1", "y2", "y3"])


@pytest.fixture
def bidirectional_causality_matrix():
    """Matrix with bidirectional causality."""
    data = np.array(
        [
            [1.0, 0.01, 0.50],  # x1 caused by x2
            [0.02, 1.0, 0.40],  # x2 caused by x1 (bidirectional!)
            [0.30, 0.25, 1.0],
        ]
    )
    return pd.DataFrame(data, index=["x1", "x2", "x3"], columns=["x1", "x2", "x3"])


class TestBuildCausalityGraph:
    """Tests for _build_causality_graph function."""

    def test_simple_unidirectional_causality(self, simple_granger_matrix):
        """Test building graph with unidirectional causality (x1→x2)."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)

        # Check nodes
        assert len(G.nodes()) == 3
        assert set(G.nodes()) == {"x1", "x2", "x3"}

        # Check edges (only x1→x2 and x2→x3 should be significant)
        assert G.number_of_edges() == 2
        assert G.has_edge("x2", "x1")  # x2 causes x1 (p=0.01)
        assert G.has_edge("x3", "x2")  # x3 causes x2 (p=0.04)
        assert not G.has_edge("x3", "x1")  # x3→x1 not significant (p=0.08)

        # Check edge attributes
        assert "p_value" in G["x2"]["x1"]
        assert "weight" in G["x2"]["x1"]
        assert G["x2"]["x1"]["p_value"] == 0.01

    def test_no_causality(self, no_causality_matrix):
        """Test graph with no significant causality (empty graph)."""
        with pytest.warns(UserWarning, match="No significant Granger causality"):
            G = _build_causality_graph(no_causality_matrix, threshold=0.05)

        # Should have nodes but no edges
        assert len(G.nodes()) == 3
        assert G.number_of_edges() == 0

    def test_bidirectional_causality(self, bidirectional_causality_matrix):
        """Test graph with bidirectional causality."""
        G = _build_causality_graph(bidirectional_causality_matrix, threshold=0.05)

        # Should have edges in both directions
        assert G.has_edge("x2", "x1")  # x2 → x1
        assert G.has_edge("x1", "x2")  # x1 → x2 (bidirectional)

    def test_no_self_loops(self, simple_granger_matrix):
        """Test that diagonal (self-causality) is not included."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)

        # No self-loops
        for node in G.nodes():
            assert not G.has_edge(node, node)

    def test_threshold_filtering(self, simple_granger_matrix):
        """Test that threshold correctly filters edges."""
        # With threshold=0.05: 2 edges (p=0.01, p=0.04)
        G1 = _build_causality_graph(simple_granger_matrix, threshold=0.05)
        assert G1.number_of_edges() == 2

        # With threshold=0.10: 3 edges (p=0.01, p=0.04, p=0.08)
        G2 = _build_causality_graph(simple_granger_matrix, threshold=0.10)
        assert G2.number_of_edges() == 3


class TestGetLayoutPositions:
    """Tests for _get_layout_positions function."""

    def test_circular_layout(self, simple_granger_matrix):
        """Test circular layout."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, layout="circular")

        # All nodes should have positions
        assert len(pos) == 3
        assert all(node in pos for node in G.nodes())

        # Positions should be 2D tuples
        for node, (x, y) in pos.items():
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))

    def test_spring_layout(self, simple_granger_matrix):
        """Test spring layout."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, layout="spring")

        assert len(pos) == 3
        assert all(node in pos for node in G.nodes())

    def test_kamada_kawai_layout(self, simple_granger_matrix):
        """Test Kamada-Kawai layout."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, layout="kamada_kawai")

        assert len(pos) == 3

    def test_shell_layout(self, simple_granger_matrix):
        """Test shell layout."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)
        pos = _get_layout_positions(G, layout="shell")

        assert len(pos) == 3

    def test_single_node_layout(self):
        """Test layout with single node."""
        import networkx as nx

        G = nx.DiGraph()
        G.add_node("x1")

        # Kamada-Kawai should handle single node gracefully
        pos = _get_layout_positions(G, layout="kamada_kawai")
        assert len(pos) == 1
        assert "x1" in pos

    def test_invalid_layout_raises(self, simple_granger_matrix):
        """Test that invalid layout raises ValueError."""
        G = _build_causality_graph(simple_granger_matrix, threshold=0.05)

        with pytest.raises(ValueError, match="Unknown layout"):
            _get_layout_positions(G, layout="invalid_layout")


class TestPlotCausalityNetwork:
    """Tests for plot_causality_network function."""

    def test_matplotlib_backend_returns_figure(self, simple_granger_matrix):
        """Test that matplotlib backend returns a Figure."""
        matplotlib = pytest.importorskip("matplotlib")

        fig = plot_causality_network(
            simple_granger_matrix, threshold=0.05, backend="matplotlib", show=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plotly_backend_returns_figure(self, simple_granger_matrix):
        """Test that plotly backend returns a Figure."""
        plotly = pytest.importorskip("plotly")

        fig = plot_causality_network(
            simple_granger_matrix, threshold=0.05, backend="plotly", show=False
        )

        # Plotly Figure
        assert hasattr(fig, "data")
        assert hasattr(fig, "layout")

    def test_invalid_backend_raises(self, simple_granger_matrix):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            plot_causality_network(simple_granger_matrix, backend="invalid_backend", show=False)

    def test_invalid_input_type_raises(self):
        """Test that non-DataFrame input raises ValueError."""
        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            plot_causality_network(np.array([[1, 0], [0, 1]]), show=False)  # Not a DataFrame

    def test_non_square_matrix_raises(self):
        """Test that non-square matrix raises ValueError."""
        df = pd.DataFrame(np.random.rand(3, 2))

        with pytest.raises(ValueError, match="must be square"):
            plot_causality_network(df, show=False)

    def test_custom_parameters(self, simple_granger_matrix):
        """Test custom visualization parameters."""
        matplotlib = pytest.importorskip("matplotlib")

        fig = plot_causality_network(
            simple_granger_matrix,
            threshold=0.10,
            layout="circular",
            node_size=5000,
            edge_width_range=(2.0, 10.0),
            show_pvalues=True,
            title="Custom Network",
            figsize=(12, 10),
            backend="matplotlib",
            show=False,
        )

        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.get_size_inches()[0] == 12
        assert fig.get_size_inches()[1] == 10

    def test_no_causality_warning(self, no_causality_matrix):
        """Test that no causality triggers warning and still produces plot."""
        matplotlib = pytest.importorskip("matplotlib")

        with pytest.warns(UserWarning, match="No significant Granger causality"):
            fig = plot_causality_network(
                no_causality_matrix, threshold=0.05, backend="matplotlib", show=False
            )

        # Should still produce a figure (with nodes but no edges)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_k_equals_5_layout(self):
        """Test layout with K=5 variables (readability check)."""
        matplotlib = pytest.importorskip("matplotlib")

        # Create 5x5 matrix
        data = np.ones((5, 5))
        for i in range(5):
            data[i, i] = 1.0  # Diagonal
            for j in range(5):
                if i != j:
                    data[i, j] = 0.01 * (i + j + 1)  # Some pattern

        df = pd.DataFrame(
            data, index=[f"x{i}" for i in range(1, 6)], columns=[f"x{i}" for i in range(1, 6)]
        )

        fig = plot_causality_network(
            df, threshold=0.15, layout="spring", backend="matplotlib", show=False
        )

        assert isinstance(fig, matplotlib.figure.Figure)


class TestNetworkIntegration:
    """Integration tests with actual PanelVAR results."""

    def test_with_synthetic_var_result(self):
        """Test network plot with synthetic VAR result."""
        from panelbox.var import PanelVAR, PanelVARData

        # Generate synthetic data
        np.random.seed(42)
        N, T, K = 30, 20, 3
        data_list = []

        for i in range(N):
            entity_data = {
                "entity": i,
                "time": np.arange(T),
                "x1": np.random.randn(T).cumsum(),
                "x2": np.random.randn(T).cumsum(),
                "x3": np.random.randn(T).cumsum(),
            }
            data_list.append(pd.DataFrame(entity_data))

        data = pd.concat(data_list, ignore_index=True)

        # Estimate VAR
        pvar_data = PanelVARData(
            data, endog_vars=["x1", "x2", "x3"], entity_col="entity", time_col="time", lags=2
        )

        model = PanelVAR(pvar_data)
        result = model.fit(method="ols")

        # Get Granger causality matrix
        granger_mat = result.granger_causality_matrix()

        # Plot (should not raise)
        matplotlib = pytest.importorskip("matplotlib")

        fig = plot_causality_network(granger_mat, threshold=0.10, backend="matplotlib", show=False)

        assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
