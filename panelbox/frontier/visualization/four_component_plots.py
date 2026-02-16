"""Visualization functions for four-component SFA models.

This module provides specialized plotting functions for visualizing
the results of four-component stochastic frontier analysis, including:
    - Distribution comparisons of persistent vs transient efficiency
    - Scatter plots showing the relationship between efficiency components
    - Time evolution of efficiency measures
    - Entity-specific decomposition plots
    - Variance component pie charts
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


def plot_efficiency_distributions(
    result,
    figsize: Tuple[float, float] = (14, 5),
    bins: int = 30,
    alpha: float = 0.7,
) -> plt.Figure:
    """Plot distributions of persistent, transient, and overall efficiency.

    Parameters:
        result: FourComponentResult object
        figsize: Figure size (width, height)
        bins: Number of histogram bins
        alpha: Transparency level

    Returns:
        Matplotlib figure object
    """
    # Get efficiency measures
    te_persistent = np.exp(-result.eta_i)
    te_transient = np.exp(-result.u_it)
    te_overall = te_persistent[result.model.entity_id] * te_transient

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Persistent efficiency
    axes[0].hist(te_persistent, bins=bins, alpha=alpha, color="steelblue", edgecolor="black")
    axes[0].axvline(
        te_persistent.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {te_persistent.mean():.3f}",
    )
    axes[0].set_xlabel("Persistent Efficiency")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Persistent Technical Efficiency\n(Structural, Long-run)")
    axes[0].legend()
    axes[0].set_xlim(0, 1)

    # Transient efficiency
    axes[1].hist(te_transient, bins=bins, alpha=alpha, color="coral", edgecolor="black")
    axes[1].axvline(
        te_transient.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {te_transient.mean():.3f}",
    )
    axes[1].set_xlabel("Transient Efficiency")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Transient Technical Efficiency\n(Managerial, Short-run)")
    axes[1].legend()
    axes[1].set_xlim(0, 1)

    # Overall efficiency
    axes[2].hist(te_overall, bins=bins, alpha=alpha, color="mediumseagreen", edgecolor="black")
    axes[2].axvline(
        te_overall.mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {te_overall.mean():.3f}",
    )
    axes[2].set_xlabel("Overall Efficiency")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Overall Technical Efficiency\n(Persistent × Transient)")
    axes[2].legend()
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    return fig


def plot_efficiency_scatter(
    result,
    figsize: Tuple[float, float] = (10, 8),
    alpha: float = 0.6,
    highlight_entities: Optional[list] = None,
) -> plt.Figure:
    """Plot scatter of persistent vs transient efficiency.

    Parameters:
        result: FourComponentResult object
        figsize: Figure size (width, height)
        alpha: Point transparency
        highlight_entities: List of entity IDs to highlight

    Returns:
        Matplotlib figure object
    """
    # Get efficiency measures (average transient for each entity)
    te_persistent = np.exp(-result.eta_i)

    # Calculate average transient efficiency per entity
    te_transient_mean = np.zeros(len(te_persistent))
    for i in range(len(te_persistent)):
        mask = result.model.entity_id == i
        te_transient_mean[i] = np.exp(-result.u_it[mask]).mean()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    scatter = ax.scatter(
        te_persistent,
        te_transient_mean,
        s=80,
        alpha=alpha,
        c=te_persistent * te_transient_mean,
        cmap="RdYlGn",
        edgecolors="black",
        linewidths=0.5,
    )

    # Highlight specific entities if requested
    if highlight_entities:
        for entity_id in highlight_entities:
            if entity_id < len(te_persistent):
                ax.scatter(
                    te_persistent[entity_id],
                    te_transient_mean[entity_id],
                    s=200,
                    marker="*",
                    c="red",
                    edgecolors="black",
                    linewidths=2,
                    zorder=10,
                    label=f"Entity {entity_id}",
                )

    # Add diagonal reference line (equal efficiency)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=2, label="TE_p = TE_t")

    # Add quadrant lines
    ax.axhline(te_transient_mean.mean(), color="gray", linestyle=":", alpha=0.5)
    ax.axvline(te_persistent.mean(), color="gray", linestyle=":", alpha=0.5)

    # Labels and formatting
    ax.set_xlabel("Persistent Efficiency (Structural)", fontsize=12)
    ax.set_ylabel("Transient Efficiency (Managerial)", fontsize=12)
    ax.set_title(
        "Persistent vs Transient Technical Efficiency\n(Average transient efficiency per entity)",
        fontsize=14,
    )
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Overall Efficiency", fontsize=10)

    # Add quadrant labels
    ax.text(
        0.25, 0.95, "Low Persistent\nHigh Transient", ha="center", va="top", fontsize=9, alpha=0.6
    )
    ax.text(0.75, 0.95, "High Both", ha="center", va="top", fontsize=9, alpha=0.6)
    ax.text(0.25, 0.05, "Low Both", ha="center", va="bottom", fontsize=9, alpha=0.6)
    ax.text(
        0.75,
        0.05,
        "High Persistent\nLow Transient",
        ha="center",
        va="bottom",
        fontsize=9,
        alpha=0.6,
    )

    plt.tight_layout()
    return fig


def plot_efficiency_evolution(
    result,
    entity_ids: Optional[list] = None,
    n_entities: int = 5,
    figsize: Tuple[float, float] = (14, 8),
) -> plt.Figure:
    """Plot time evolution of efficiency for selected entities.

    Parameters:
        result: FourComponentResult object
        entity_ids: Specific entity IDs to plot (if None, select random)
        n_entities: Number of entities to plot if entity_ids is None
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    # Select entities
    if entity_ids is None:
        np.random.seed(42)
        entity_ids = np.random.choice(
            result.model.n_entities,
            size=min(n_entities, result.model.n_entities),
            replace=False,
        )

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Get unique time periods
    unique_times = np.unique(result.model.time_id)

    # Plot each entity
    colors = plt.cm.tab10(np.linspace(0, 1, len(entity_ids)))

    for idx, entity_id in enumerate(entity_ids):
        # Get data for this entity
        mask = result.model.entity_id == entity_id

        # Persistent efficiency (constant over time)
        te_persistent = np.exp(-result.eta_i[entity_id])

        # Transient efficiency (varies over time)
        te_transient = np.exp(-result.u_it[mask])
        time_periods = result.model.time_id[mask]

        # Plot transient efficiency
        axes[0].plot(
            time_periods,
            te_transient,
            marker="o",
            color=colors[idx],
            label=f"Entity {entity_id}",
            linewidth=2,
            markersize=6,
        )

        # Plot overall efficiency
        te_overall = te_persistent * te_transient
        axes[1].plot(
            time_periods,
            te_overall,
            marker="s",
            color=colors[idx],
            label=f"Entity {entity_id}",
            linewidth=2,
            markersize=6,
        )

        # Add persistent level as horizontal line
        axes[1].axhline(
            te_persistent,
            color=colors[idx],
            linestyle="--",
            alpha=0.3,
            linewidth=1,
        )

    # Formatting
    axes[0].set_xlabel("Time Period", fontsize=11)
    axes[0].set_ylabel("Transient Efficiency", fontsize=11)
    axes[0].set_title(
        "Transient Technical Efficiency Over Time\n(Managerial efficiency)", fontsize=13
    )
    axes[0].legend(loc="best", ncol=2)
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Time Period", fontsize=11)
    axes[1].set_ylabel("Overall Efficiency", fontsize=11)
    axes[1].set_title(
        "Overall Technical Efficiency Over Time\n(Dashed lines show persistent efficiency levels)",
        fontsize=13,
    )
    axes[1].legend(loc="best", ncol=2)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_entity_decomposition(
    result,
    entity_ids: Optional[list] = None,
    n_entities: int = 10,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Plot component decomposition for selected entities.

    Shows the four components (μ_i, η_i, u_it_mean, v_it_mean) for each entity.

    Parameters:
        result: FourComponentResult object
        entity_ids: Specific entity IDs to plot (if None, select top/bottom)
        n_entities: Number of entities to plot if entity_ids is None
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    # Select entities (top and bottom by overall efficiency)
    if entity_ids is None:
        te_persistent = np.exp(-result.eta_i)
        te_overall_mean = np.zeros(result.model.n_entities)

        for i in range(result.model.n_entities):
            mask = result.model.entity_id == i
            te_transient = np.exp(-result.u_it[mask])
            te_overall_mean[i] = (te_persistent[i] * te_transient).mean()

        # Get top and bottom entities
        n_top = n_entities // 2
        n_bottom = n_entities - n_top
        top_entities = np.argsort(te_overall_mean)[-n_top:][::-1]
        bottom_entities = np.argsort(te_overall_mean)[:n_bottom]
        entity_ids = np.concatenate([top_entities, bottom_entities])

    # Prepare data
    n = len(entity_ids)
    mu_i = result.mu_i[entity_ids]
    eta_i = result.eta_i[entity_ids]

    # Average transient components per entity
    u_it_mean = np.zeros(n)
    v_it_mean = np.zeros(n)

    for idx, entity_id in enumerate(entity_ids):
        mask = result.model.entity_id == entity_id
        u_it_mean[idx] = result.u_it[mask].mean()
        v_it_mean[idx] = result.v_it[mask].mean()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar positions
    x = np.arange(n)
    width = 0.2

    # Plot bars
    bars1 = ax.bar(
        x - 1.5 * width,
        mu_i,
        width,
        label="μ_i (Heterogeneity)",
        color="skyblue",
        edgecolor="black",
    )
    bars2 = ax.bar(
        x - 0.5 * width,
        -eta_i,
        width,
        label="-η_i (Persistent Ineff.)",
        color="indianred",
        edgecolor="black",
    )
    bars3 = ax.bar(
        x + 0.5 * width,
        v_it_mean,
        width,
        label="v_it (Noise, avg)",
        color="lightgreen",
        edgecolor="black",
    )
    bars4 = ax.bar(
        x + 1.5 * width,
        -u_it_mean,
        width,
        label="-u_it (Transient Ineff., avg)",
        color="orange",
        edgecolor="black",
    )

    # Formatting
    ax.set_xlabel("Entity ID", fontsize=11)
    ax.set_ylabel("Component Value", fontsize=11)
    ax.set_title(
        "Four-Component Decomposition by Entity\n(Top and bottom entities by overall efficiency)",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{eid}" for eid in entity_ids])
    ax.legend(loc="best", ncol=2)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_variance_decomposition(
    result,
    figsize: Tuple[float, float] = (10, 8),
    explode: Tuple[float, ...] = (0.05, 0.05, 0.05, 0.05),
) -> plt.Figure:
    """Plot pie chart of variance component shares.

    Parameters:
        result: FourComponentResult object
        figsize: Figure size (width, height)
        explode: Tuple of explosion values for each slice

    Returns:
        Matplotlib figure object
    """
    # Variance components
    var_v = result.sigma_v**2
    var_u = result.sigma_u**2
    var_mu = result.sigma_mu**2
    var_eta = result.sigma_eta**2

    total_var = var_v + var_u + var_mu + var_eta

    # Calculate shares
    shares = np.array([var_v, var_u, var_mu, var_eta]) / total_var * 100

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Pie chart
    labels = [
        f"Noise (σ²_v)\n{shares[0]:.1f}%",
        f"Transient Ineff. (σ²_u)\n{shares[1]:.1f}%",
        f"Heterogeneity (σ²_μ)\n{shares[2]:.1f}%",
        f"Persistent Ineff. (σ²_η)\n{shares[3]:.1f}%",
    ]

    colors = ["lightgreen", "orange", "skyblue", "indianred"]

    wedges, texts, autotexts = ax1.pie(
        shares,
        labels=labels,
        colors=colors,
        autopct="",
        startangle=90,
        explode=explode,
        shadow=True,
        textprops={"fontsize": 10},
    )

    ax1.set_title("Variance Decomposition\n(Share of total variance)", fontsize=13, pad=20)

    # Bar chart with absolute values
    components = [
        "Noise\n(v_it)",
        "Transient\nIneff.\n(u_it)",
        "Heterog.\n(μ_i)",
        "Persistent\nIneff.\n(η_i)",
    ]
    variances = [var_v, var_u, var_mu, var_eta]

    bars = ax2.bar(components, variances, color=colors, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for bar, var in zip(bars, variances):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{var:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax2.set_ylabel("Variance", fontsize=11)
    ax2.set_title("Variance Components\n(Absolute values)", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def plot_comprehensive_summary(
    result,
    entity_highlight: Optional[list] = None,
    n_evolution: int = 3,
    figsize: Tuple[float, float] = (16, 12),
) -> plt.Figure:
    """Create comprehensive summary plot with multiple panels.

    Parameters:
        result: FourComponentResult object
        entity_highlight: Entities to highlight in scatter plot
        n_evolution: Number of entities for evolution plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Get efficiency measures
    te_persistent = np.exp(-result.eta_i)
    te_transient = np.exp(-result.u_it)
    te_overall = te_persistent[result.model.entity_id] * te_transient

    # 1. Distribution of persistent efficiency
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(te_persistent, bins=25, alpha=0.7, color="steelblue", edgecolor="black")
    ax1.axvline(te_persistent.mean(), color="red", linestyle="--", linewidth=2)
    ax1.set_xlabel("Persistent Efficiency")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Persistent TE Distribution")
    ax1.set_xlim(0, 1)

    # 2. Distribution of transient efficiency
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(te_transient, bins=25, alpha=0.7, color="coral", edgecolor="black")
    ax2.axvline(te_transient.mean(), color="red", linestyle="--", linewidth=2)
    ax2.set_xlabel("Transient Efficiency")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Transient TE Distribution")
    ax2.set_xlim(0, 1)

    # 3. Variance decomposition
    ax3 = fig.add_subplot(gs[0, 2])
    var_v = result.sigma_v**2
    var_u = result.sigma_u**2
    var_mu = result.sigma_mu**2
    var_eta = result.sigma_eta**2
    total_var = var_v + var_u + var_mu + var_eta
    shares = np.array([var_v, var_u, var_mu, var_eta]) / total_var * 100
    colors = ["lightgreen", "orange", "skyblue", "indianred"]
    labels = ["Noise", "Trans.\nIneff.", "Heterog.", "Pers.\nIneff."]
    ax3.pie(shares, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax3.set_title("Variance Decomposition")

    # 4. Scatter persistent vs transient (spans 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    te_transient_mean = np.zeros(len(te_persistent))
    for i in range(len(te_persistent)):
        mask = result.model.entity_id == i
        te_transient_mean[i] = np.exp(-result.u_it[mask]).mean()

    scatter = ax4.scatter(
        te_persistent,
        te_transient_mean,
        s=60,
        alpha=0.6,
        c=te_persistent * te_transient_mean,
        cmap="RdYlGn",
        edgecolors="black",
        linewidths=0.5,
    )
    ax4.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1.5)
    ax4.set_xlabel("Persistent Efficiency")
    ax4.set_ylabel("Transient Efficiency (avg)")
    ax4.set_title("Persistent vs Transient Efficiency")
    ax4.set_xlim(0, 1.05)
    ax4.set_ylim(0, 1.05)

    # 5. Summary statistics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    summary_text = f"""
    EFFICIENCY SUMMARY

    Persistent:
      Mean: {te_persistent.mean():.3f}
      Std:  {te_persistent.std():.3f}
      Min:  {te_persistent.min():.3f}
      Max:  {te_persistent.max():.3f}

    Transient:
      Mean: {te_transient.mean():.3f}
      Std:  {te_transient.std():.3f}
      Min:  {te_transient.min():.3f}
      Max:  {te_transient.max():.3f}

    Overall:
      Mean: {te_overall.mean():.3f}
      Std:  {te_overall.std():.3f}
      Min:  {te_overall.min():.3f}
      Max:  {te_overall.max():.3f}
    """

    ax5.text(0.1, 0.5, summary_text, fontsize=9, verticalalignment="center", family="monospace")

    # 6. Time evolution (spans all bottom columns)
    ax6 = fig.add_subplot(gs[2, :])

    # Select random entities for evolution
    np.random.seed(42)
    entity_ids = np.random.choice(
        result.model.n_entities, size=min(n_evolution, result.model.n_entities), replace=False
    )
    colors_evo = plt.cm.tab10(np.linspace(0, 1, len(entity_ids)))

    for idx, entity_id in enumerate(entity_ids):
        mask = result.model.entity_id == entity_id
        te_persistent_i = np.exp(-result.eta_i[entity_id])
        te_transient_i = np.exp(-result.u_it[mask])
        te_overall_i = te_persistent_i * te_transient_i
        time_periods = result.model.time_id[mask]

        ax6.plot(
            time_periods,
            te_overall_i,
            marker="o",
            color=colors_evo[idx],
            label=f"Entity {entity_id}",
            linewidth=2,
        )
        ax6.axhline(te_persistent_i, color=colors_evo[idx], linestyle="--", alpha=0.3, linewidth=1)

    ax6.set_xlabel("Time Period")
    ax6.set_ylabel("Overall Efficiency")
    ax6.set_title("Overall Efficiency Over Time (dashed lines = persistent levels)")
    ax6.legend(loc="best", ncol=3)
    ax6.set_ylim(0, 1.05)
    ax6.grid(True, alpha=0.3)

    fig.suptitle(
        "Four-Component SFA Model: Comprehensive Summary", fontsize=16, fontweight="bold", y=0.995
    )

    return fig
