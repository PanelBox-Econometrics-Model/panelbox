"""
SFA Visualization Gallery

This script demonstrates all visualization capabilities of the SFA module:
1. Efficiency distribution plots
2. Efficiency rankings
3. Temporal evolution (panel data)
4. Frontier estimation plots
5. Report generation

Run this script to generate an HTML gallery of all visualizations.
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.visualization import compare_models

# Set random seed for reproducibility
np.random.seed(42)


def generate_sample_data(n_entities=20, n_periods=5):
    """Generate synthetic production data."""
    n = n_entities * n_periods

    # True production function with inefficiency
    # y = β₀ + β₁·log(L) + β₂·log(K) + v - u
    data = pd.DataFrame(
        {
            "entity_id": np.repeat(range(n_entities), n_periods),
            "time": np.tile(range(n_periods), n_entities),
            "log_labor": np.random.uniform(2, 5, n),
            "log_capital": np.random.uniform(3, 6, n),
            "region": np.random.choice(["North", "South", "East", "West"], n),
        }
    )

    # True parameters
    beta_0 = 2.0
    beta_L = 0.6
    beta_K = 0.3

    # Generate output with noise and inefficiency
    y_frontier = beta_0 + beta_L * data["log_labor"] + beta_K * data["log_capital"]

    v = np.random.normal(0, 0.1, n)  # Noise
    u = np.abs(np.random.normal(0, 0.15, n))  # Inefficiency (always non-negative)

    data["log_output"] = y_frontier + v - u  # Production frontier

    return data


def example_1_efficiency_distribution():
    """Example 1: Efficiency distribution plots."""
    print("\n=== Example 1: Efficiency Distribution ===")

    # Generate data
    data = generate_sample_data(n_entities=50, n_periods=1)

    # Fit model
    sf = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )
    result = sf.fit()

    # Plot 1: Histogram with KDE
    print("Generating histogram plot...")
    fig1 = result.plot_efficiency(
        kind="histogram",
        backend="plotly",
        bins=25,
        show_kde=True,
        show_stats=True,
        title="Efficiency Distribution: Banking Sector",
    )
    fig1.write_html("sfa_efficiency_histogram.html")
    print("  → Saved to: sfa_efficiency_histogram.html")

    # Plot 2: Rankings
    print("Generating ranking plot...")
    fig2 = result.plot_efficiency(
        kind="ranking",
        backend="plotly",
        top_n=15,
        bottom_n=15,
        title="Efficiency Rankings: Top & Bottom Performers",
    )
    fig2.write_html("sfa_efficiency_ranking.html")
    print("  → Saved to: sfa_efficiency_ranking.html")

    # Plot 3: Box plot by region
    print("Generating box plot by region...")
    eff_df = result.efficiency(estimator="bc")
    eff_df["region"] = data["region"].values

    from panelbox.frontier.visualization.efficiency_plots import plot_efficiency_boxplot

    fig3 = plot_efficiency_boxplot(
        eff_df,
        group_var="region",
        backend="plotly",
        test="kruskal",
        title="Efficiency by Region (Kruskal-Wallis Test)",
    )
    fig3.write_html("sfa_efficiency_boxplot.html")
    print("  → Saved to: sfa_efficiency_boxplot.html")

    return result


def example_2_panel_evolution():
    """Example 2: Temporal evolution plots (panel data)."""
    print("\n=== Example 2: Temporal Evolution (Panel Data) ===")

    # Generate panel data
    data = generate_sample_data(n_entities=20, n_periods=10)

    # Fit panel model
    sf = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity_col="entity_id",
        time_col="time",
        frontier="production",
        dist="half_normal",
        is_panel=True,
        model_type="pitt_lee",
    )
    result = sf.fit()

    # Plot 1: Time series
    print("Generating time series plot...")
    fig1 = result.plot_efficiency_evolution(
        kind="timeseries",
        backend="plotly",
        show_ci=True,
        show_median=True,
        events={2: "Reform", 7: "Crisis"},
        title="Mean Efficiency Over Time",
    )
    fig1.write_html("sfa_timeseries.html")
    print("  → Saved to: sfa_timeseries.html")

    # Plot 2: Spaghetti plot
    print("Generating spaghetti plot...")
    fig2 = result.plot_efficiency_evolution(
        kind="spaghetti",
        backend="plotly",
        highlight=[0, 1, 2],  # Highlight first 3 entities
        alpha=0.2,
        show_mean=True,
        title="Individual Efficiency Trajectories",
    )
    fig2.write_html("sfa_spaghetti.html")
    print("  → Saved to: sfa_spaghetti.html")

    # Plot 3: Heatmap
    print("Generating heatmap...")
    fig3 = result.plot_efficiency_evolution(
        kind="heatmap",
        backend="plotly",
        order_by="efficiency",
        title="Efficiency Heatmap (Entity × Time)",
    )
    fig3.write_html("sfa_heatmap.html")
    print("  → Saved to: sfa_heatmap.html")

    # Plot 4: Fan chart
    print("Generating fan chart...")
    fig4 = result.plot_efficiency_evolution(
        kind="fanchart",
        backend="plotly",
        percentiles=[10, 25, 50, 75, 90],
        title="Efficiency Fan Chart (Percentile Evolution)",
    )
    fig4.write_html("sfa_fanchart.html")
    print("  → Saved to: sfa_fanchart.html")

    return result


def example_3_frontier_plots():
    """Example 3: Frontier estimation plots."""
    print("\n=== Example 3: Frontier Estimation Plots ===")

    # Generate data
    data = generate_sample_data(n_entities=100, n_periods=1)

    # Fit model
    sf = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )
    result = sf.fit()

    # Plot 1: 2D frontier (1 input)
    print("Generating 2D frontier plot...")
    fig1 = result.plot_frontier(
        input_var="log_labor",
        kind="2d",
        backend="plotly",
        show_distance=True,
        n_observations=50,
        title="Production Frontier: Labor Input",
    )
    fig1.write_html("sfa_frontier_2d.html")
    print("  → Saved to: sfa_frontier_2d.html")

    # Plot 2: 3D frontier surface
    print("Generating 3D frontier plot...")
    fig2 = result.plot_frontier(
        input_vars=["log_labor", "log_capital"],
        kind="3d",
        backend="plotly",
        n_grid=20,
        title="Production Frontier Surface",
    )
    fig2.write_html("sfa_frontier_3d.html")
    print("  → Saved to: sfa_frontier_3d.html")

    # Plot 3: Contour plot
    print("Generating contour plot...")
    fig3 = result.plot_frontier(
        input_vars=["log_labor", "log_capital"],
        kind="contour",
        backend="plotly",
        n_grid=30,
        levels=15,
        title="Frontier Contour Plot",
    )
    fig3.write_html("sfa_frontier_contour.html")
    print("  → Saved to: sfa_frontier_contour.html")

    # Plot 4: Partial frontier
    print("Generating partial frontier plot...")
    fig4 = result.plot_frontier(
        input_var="log_labor",
        kind="partial",
        backend="plotly",
        fix_others_at="mean",
        title="Partial Frontier: Labor (Capital Fixed at Mean)",
    )
    fig4.write_html("sfa_frontier_partial.html")
    print("  → Saved to: sfa_frontier_partial.html")

    return result


def example_4_model_comparison():
    """Example 4: Comparing different distributional specifications."""
    print("\n=== Example 4: Model Comparison ===")

    # Generate data
    data = generate_sample_data(n_entities=50, n_periods=1)

    # Fit models with different distributions
    print("Fitting Half-Normal model...")
    sf_hn = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )
    result_hn = sf_hn.fit()

    print("Fitting Exponential model...")
    sf_exp = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="exponential",
    )
    result_exp = sf_exp.fit()

    print("Fitting Truncated Normal model...")
    sf_tn = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="truncated_normal",
    )
    result_tn = sf_tn.fit()

    # Compare models
    print("\nModel Comparison:")
    comparison = compare_models(
        models={"Half-Normal": result_hn, "Exponential": result_exp, "Truncated Normal": result_tn},
        output_format="dataframe",
    )
    print(comparison)

    # Save comparison to LaTeX
    latex_comparison = compare_models(
        models={"Half-Normal": result_hn, "Exponential": result_exp, "Truncated Normal": result_tn},
        output_format="latex",
    )

    with open("sfa_model_comparison.tex", "w") as f:
        f.write(latex_comparison)
    print("\n  → LaTeX comparison saved to: sfa_model_comparison.tex")

    return result_hn


def example_5_reports():
    """Example 5: Generating comprehensive reports."""
    print("\n=== Example 5: Report Generation ===")

    # Generate data
    data = generate_sample_data(n_entities=50, n_periods=1)

    # Fit model
    sf = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )
    result = sf.fit()

    # 1. HTML Report
    print("Generating HTML report...")
    result.to_html(filename="sfa_report.html", include_plots=True, theme="academic")
    print("  → Saved to: sfa_report.html")

    # 2. LaTeX Table
    print("Generating LaTeX table...")
    latex = result.to_latex(
        caption="Production Frontier Analysis Results",
        label="tab:sfa_results",
        include_stats=["coef", "se", "pval"],
    )
    with open("sfa_table.tex", "w") as f:
        f.write(latex)
    print("  → Saved to: sfa_table.tex")

    # 3. Markdown Report
    print("Generating Markdown report...")
    md = result.to_markdown()
    with open("sfa_report.md", "w") as f:
        f.write(md)
    print("  → Saved to: sfa_report.md")

    # 4. Efficiency Rankings Table
    print("Generating efficiency rankings table...")
    eff_table = result.efficiency_table(sort_by="te", ascending=False, top_n=20, estimator="bc")
    eff_table.to_excel("efficiency_rankings.xlsx", index=False)
    print("  → Saved to: efficiency_rankings.xlsx")

    # Print summary
    print("\n" + "=" * 80)
    print(result.summary())
    print("=" * 80)

    return result


def main():
    """Run all examples."""
    print("=" * 80)
    print("SFA VISUALIZATION GALLERY".center(80))
    print("=" * 80)

    # Run all examples
    result1 = example_1_efficiency_distribution()
    result2 = example_2_panel_evolution()
    result3 = example_3_frontier_plots()
    result4 = example_4_model_comparison()
    result5 = example_5_reports()

    print("\n" + "=" * 80)
    print("Gallery generation complete!".center(80))
    print("=" * 80)
    print("\nGenerated files:")
    print("  Efficiency Plots:")
    print("    - sfa_efficiency_histogram.html")
    print("    - sfa_efficiency_ranking.html")
    print("    - sfa_efficiency_boxplot.html")
    print("\n  Evolution Plots (Panel):")
    print("    - sfa_timeseries.html")
    print("    - sfa_spaghetti.html")
    print("    - sfa_heatmap.html")
    print("    - sfa_fanchart.html")
    print("\n  Frontier Plots:")
    print("    - sfa_frontier_2d.html")
    print("    - sfa_frontier_3d.html")
    print("    - sfa_frontier_contour.html")
    print("    - sfa_frontier_partial.html")
    print("\n  Reports:")
    print("    - sfa_report.html")
    print("    - sfa_table.tex")
    print("    - sfa_report.md")
    print("    - sfa_model_comparison.tex")
    print("    - efficiency_rankings.xlsx")
    print("\nOpen the HTML files in your browser to view the interactive visualizations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
