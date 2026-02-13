"""
Causality network visualization for Panel VAR models.

This module provides functions to visualize Granger causality relationships
as directed network graphs using NetworkX, Matplotlib, and Plotly.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def plot_causality_network(
    granger_matrix: pd.DataFrame,
    threshold: float = 0.05,
    layout: str = "spring",
    node_size: float = 3000,
    edge_width_range: Tuple[float, float] = (1.0, 5.0),
    show_pvalues: bool = False,
    backend: str = "plotly",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    **kwargs,
):
    """
    Plot Granger causality relationships as a directed network graph.

    Creates a network where:
    - Nodes represent variables
    - Directed edges represent significant Granger causality (p < threshold)
    - Edge thickness represents strength of causality (inverse of p-value)
    - Edge color can represent sign of net effect (if result object provided)

    Parameters
    ----------
    granger_matrix : pd.DataFrame
        Granger causality p-value matrix (K×K).
        Element [i, j] is the p-value for "j causes i"
    threshold : float, default=0.05
        Significance threshold. Only relationships with p < threshold are shown.
    layout : str, default='spring'
        Network layout algorithm:
        - 'circular': Circular layout
        - 'spring': Spring force-directed layout
        - 'kamada_kawai': Kamada-Kawai layout
        - 'shell': Shell layout
    node_size : float, default=3000
        Size of nodes
    edge_width_range : tuple, default=(1.0, 5.0)
        Range for edge widths (min, max)
    show_pvalues : bool, default=False
        Whether to annotate edges with p-values
    backend : str, default='plotly'
        Plotting backend: 'plotly' or 'matplotlib'
    title : str, optional
        Plot title. Default: "Granger Causality Network"
    figsize : tuple, default=(10, 8)
        Figure size (only for matplotlib backend)
    show : bool, default=True
        Whether to display the plot
    **kwargs
        Additional arguments passed to plotting functions

    Returns
    -------
    fig
        Plotly Figure or Matplotlib Figure object

    Raises
    ------
    ImportError
        If networkx is not installed
    ValueError
        If backend is not supported or granger_matrix is invalid

    Examples
    --------
    >>> result = pvar.fit(method='ols', lags=2)
    >>> granger_mat = result.granger_causality_matrix(significance_level=0.05)
    >>> fig = plot_causality_network(granger_mat, threshold=0.05, backend='plotly')
    >>>
    >>> # Customize
    >>> fig = plot_causality_network(
    ...     granger_mat,
    ...     threshold=0.10,
    ...     layout='circular',
    ...     node_size=4000,
    ...     show_pvalues=True,
    ...     backend='matplotlib'
    ... )

    Notes
    -----
    - Requires networkx package: `pip install networkx`
    - Self-loops are not shown (variable cannot Granger-cause itself)
    - If no significant relationships exist, an empty graph is shown with a warning
    """
    if not HAS_NETWORKX:
        raise ImportError(
            "networkx is required for causality network plots. Install with: pip install networkx"
        )

    # Validate inputs
    if not isinstance(granger_matrix, pd.DataFrame):
        raise ValueError("granger_matrix must be a pandas DataFrame")

    if granger_matrix.shape[0] != granger_matrix.shape[1]:
        raise ValueError("granger_matrix must be square (K×K)")

    # Build directed graph
    G = _build_causality_graph(granger_matrix, threshold)

    # Get positions based on layout
    pos = _get_layout_positions(G, layout)

    # Plot
    if backend == "plotly":
        if not HAS_PLOTLY:
            raise ImportError("plotly is required. Install with: pip install plotly")
        return _plot_network_plotly(
            G,
            pos,
            granger_matrix,
            threshold,
            node_size,
            edge_width_range,
            show_pvalues,
            title,
            show,
            **kwargs,
        )
    elif backend == "matplotlib":
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required. Install with: pip install matplotlib")
        return _plot_network_matplotlib(
            G,
            pos,
            granger_matrix,
            threshold,
            node_size,
            edge_width_range,
            show_pvalues,
            title,
            figsize,
            show,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown backend='{backend}'. Use 'plotly' or 'matplotlib'")


def _build_causality_graph(granger_matrix: pd.DataFrame, threshold: float) -> "nx.DiGraph":
    """
    Build directed graph from Granger causality matrix.

    Parameters
    ----------
    granger_matrix : pd.DataFrame
        Granger causality p-value matrix
    threshold : float
        Significance threshold

    Returns
    -------
    nx.DiGraph
        Directed graph with nodes and edges for significant causality
    """
    var_names = granger_matrix.index.tolist()
    K = len(var_names)

    # Create directed graph
    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    # Add edges for significant causality
    n_edges = 0
    for i in range(K):
        for j in range(K):
            if i == j:
                continue  # Skip self-loops

            cause = var_names[j]
            effect = var_names[i]
            p_value = granger_matrix.iloc[i, j]

            if p_value < threshold:
                # Significant causality: j → i
                weight = 1.0 / max(p_value, 1e-10)  # Inverse of p-value
                G.add_edge(cause, effect, p_value=p_value, weight=weight)
                n_edges += 1

    if n_edges == 0:
        warnings.warn(
            f"No significant Granger causality relationships found at threshold={threshold}. "
            "Try increasing the threshold or check your model specification.",
            UserWarning,
        )

    return G


def _get_layout_positions(G: "nx.DiGraph", layout: str) -> Dict[str, Tuple[float, float]]:
    """
    Compute node positions using specified layout algorithm.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph
    layout : str
        Layout algorithm name

    Returns
    -------
    dict
        Dictionary mapping node names to (x, y) positions
    """
    n_nodes = len(G.nodes())

    if layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "spring":
        # Spring layout with deterministic seed for reproducibility
        pos = nx.spring_layout(G, seed=42, k=1.5 / np.sqrt(n_nodes) if n_nodes > 0 else 1.0)
    elif layout == "kamada_kawai":
        if n_nodes > 1:
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = {list(G.nodes())[0]: (0.5, 0.5)} if n_nodes == 1 else {}
    elif layout == "shell":
        pos = nx.shell_layout(G)
    else:
        raise ValueError(
            f"Unknown layout='{layout}'. Use 'circular', 'spring', 'kamada_kawai', or 'shell'"
        )

    return pos


def _plot_network_matplotlib(
    G: "nx.DiGraph",
    pos: Dict,
    granger_matrix: pd.DataFrame,
    threshold: float,
    node_size: float,
    edge_width_range: Tuple[float, float],
    show_pvalues: bool,
    title: Optional[str],
    figsize: Tuple[int, int],
    show: bool,
    **kwargs,
) -> plt.Figure:
    """Plot network using Matplotlib."""
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightblue",
        node_size=node_size,
        alpha=0.9,
        ax=ax,
        edgecolors="black",
        linewidths=2,
    )

    # Draw node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", ax=ax)

    # Draw edges with varying thickness
    if G.number_of_edges() > 0:
        # Normalize edge weights to width range
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        if len(weights) > 0:
            weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
            widths = edge_width_range[0] + weights_normalized * (
                edge_width_range[1] - edge_width_range[0]
            )
        else:
            widths = [edge_width_range[0]] * len(G.edges())

        # Draw edges
        for (u, v), width in zip(G.edges(), widths):
            p_value = G[u][v]["p_value"]
            # Color based on significance level
            color = "darkgreen" if p_value < 0.01 else "green" if p_value < 0.05 else "orange"

            nx.draw_networkx_edges(
                G,
                pos,
                [(u, v)],
                edge_color=color,
                width=width,
                alpha=0.7,
                arrowsize=20,
                arrowstyle="->",
                ax=ax,
                connectionstyle="arc3,rad=0.1",
            )

            # Add p-value labels if requested
            if show_pvalues:
                # Compute label position (midpoint of edge)
                x = (pos[u][0] + pos[v][0]) / 2
                y = (pos[u][1] + pos[v][1]) / 2
                ax.text(
                    x,
                    y,
                    f"p={p_value:.3f}",
                    fontsize=8,
                    ha="center",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="gray"
                    ),
                )

    ax.set_title(title or "Granger Causality Network", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="darkgreen", linewidth=3, label=f"p < 0.01"),
        Line2D([0], [0], color="green", linewidth=3, label=f"0.01 ≤ p < 0.05"),
        Line2D([0], [0], color="orange", linewidth=3, label=f"0.05 ≤ p < {threshold}"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    plt.tight_layout()

    if show:
        plt.show()

    return fig


def _plot_network_plotly(
    G: "nx.DiGraph",
    pos: Dict,
    granger_matrix: pd.DataFrame,
    threshold: float,
    node_size: float,
    edge_width_range: Tuple[float, float],
    show_pvalues: bool,
    title: Optional[str],
    show: bool,
    **kwargs,
) -> go.Figure:
    """Plot network using Plotly (interactive)."""
    # Create edge traces
    edge_traces = []

    if G.number_of_edges() > 0:
        # Normalize weights for edge widths
        weights = np.array([G[u][v]["weight"] for u, v in G.edges()])
        if len(weights) > 0:
            weights_normalized = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
            widths = edge_width_range[0] + weights_normalized * (
                edge_width_range[1] - edge_width_range[0]
            )
        else:
            widths = [edge_width_range[0]] * len(G.edges())

        for (u, v), width in zip(G.edges(), widths):
            p_value = G[u][v]["p_value"]
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Color based on p-value
            if p_value < 0.01:
                color = "darkgreen"
            elif p_value < 0.05:
                color = "green"
            else:
                color = "orange"

            # Edge line
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="text",
                text=f"{u} → {v}<br>p-value: {p_value:.4f}",
                showlegend=False,
            )
            edge_traces.append(edge_trace)

            # Arrow annotation
            # Plotly doesn't have native arrows in scatter, use annotations
            # Calculate arrow position (80% along the edge to avoid overlapping with node)
            arrow_x = x0 + 0.8 * (x1 - x0)
            arrow_y = y0 + 0.8 * (y1 - y0)

    # Create node trace
    node_x = []
    node_y = []
    node_text = []
    node_names = list(G.nodes())

    for node in node_names:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(size=12, color="black", family="Arial Black"),
        hoverinfo="text",
        hovertext=[f"{name}<br>Degree: {G.degree(name)}" for name in node_names],
        marker=dict(
            size=node_size / 30,  # Scale down for Plotly
            color="lightblue",
            line=dict(color="black", width=2),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    # Add arrows as annotations
    if G.number_of_edges() > 0:
        annotations = []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]

            # Arrow at 80% of edge
            ax = x0 + 0.8 * (x1 - x0)
            ay = y0 + 0.8 * (y1 - y0)

            annotations.append(
                dict(
                    ax=x0 + 0.6 * (x1 - x0),
                    ay=y0 + 0.6 * (y1 - y0),
                    x=ax,
                    y=ay,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=fig.data[list(G.edges()).index((u, v))].line.color,
                )
            )

        fig.update_layout(annotations=annotations)

    # Update layout
    fig.update_layout(
        title=dict(text=title or "Granger Causality Network (Interactive)", font=dict(size=16)),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        **kwargs,
    )

    if show:
        fig.show()

    return fig
