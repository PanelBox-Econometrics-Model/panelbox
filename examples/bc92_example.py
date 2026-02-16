"""
Example: Battese & Coelli (1992) Time-Varying Inefficiency Model

This example demonstrates the Battese & Coelli (1992) stochastic frontier
model with time-varying inefficiency.

Model:
    y_it = x_it'β + v_it - u_it

where:
    u_it = exp[-η(t - T_i)] · u_i
    u_i ~ N⁺(0, σ²_u)
    v_it ~ N(0, σ²_v)

The time-decay parameter η controls how inefficiency changes over time:
- η > 0: Inefficiency decreases over time (learning effect)
- η < 0: Inefficiency increases over time (degradation)
- η = 0: Time-invariant inefficiency (reduces to Pitt-Lee model)

References:
    Battese, G. E., & Coelli, T. J. (1992).
    Frontier production functions, technical efficiency and panel data:
    with application to paddy farmers in India.
    Journal of Productivity Analysis, 3(1-2), 153-169.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier


def generate_bc92_data(N=30, T=10, eta_true=0.05, seed=42):
    """
    Generate synthetic data from BC92 model.

    Parameters:
        N: Number of firms
        T: Number of time periods
        eta_true: True time-decay parameter
        seed: Random seed for reproducibility

    Returns:
        DataFrame with simulated data
    """
    np.random.seed(seed)

    # True parameters
    beta_true = np.array([5.0, 0.6, 0.4])  # [const, capital, labor]
    sigma_v_true = 0.1
    sigma_u_true = 0.2

    data = []

    for i in range(N):
        # Draw firm-specific inefficiency
        u_i = np.abs(np.random.normal(0, sigma_u_true))

        for t in range(T):
            # Time-varying inefficiency
            u_it = np.exp(-eta_true * (t - (T - 1))) * u_i

            # Generate X variables
            lnk = np.random.normal(2.5, 0.3)
            lnl = np.random.normal(1.7, 0.2)

            # Generate y
            X = np.array([1.0, lnk, lnl])
            v_it = np.random.normal(0, sigma_v_true)
            y_it = X @ beta_true + v_it - u_it

            data.append(
                {
                    "firm": i,
                    "year": t,
                    "lny": y_it,
                    "lnk": lnk,
                    "lnl": lnl,
                    "true_u": u_it,
                    "true_eff": np.exp(-u_it),
                }
            )

    return pd.DataFrame(data)


def main():
    """Run BC92 example."""
    print("=" * 70)
    print("Battese & Coelli (1992) Time-Varying Inefficiency Model")
    print("=" * 70)

    # Generate data
    print("\n1. Generating synthetic data...")
    eta_true = 0.05  # Positive = learning effect
    df = generate_bc92_data(N=30, T=10, eta_true=eta_true)

    print(f"   - Number of firms: {df['firm'].nunique()}")
    print(f"   - Number of periods: {df['year'].nunique()}")
    print(f"   - Total observations: {len(df)}")
    print(f"   - True eta (time-decay): {eta_true:.4f}")

    # Estimate BC92 model
    print("\n2. Estimating BC92 model...")
    model = StochasticFrontier(
        data=df,
        depvar="lny",
        exog=["lnk", "lnl"],
        entity="firm",
        time="year",
        frontier="production",
        dist="half_normal",
        model_type="bc92",
    )

    result = model.fit()

    # Display results
    print("\n3. Estimation Results:")
    print("-" * 70)
    print(result.summary())

    # Extract parameters
    eta_est = result.params.iloc[-1]
    print("\n4. Time-Decay Parameter Interpretation:")
    print("-" * 70)
    print(f"   Estimated eta: {eta_est:.4f}")
    print(f"   True eta:      {eta_true:.4f}")

    if eta_est > 0:
        print(f"\n   → Positive eta = {eta_est:.4f}")
        print("   → Inefficiency DECREASES over time")
        print("   → Interpretation: Firms are learning")
    elif eta_est < 0:
        print(f"\n   → Negative eta = {eta_est:.4f}")
        print("   → Inefficiency INCREASES over time")
        print("   → Interpretation: Technology degradation or obsolescence")
    else:
        print(f"\n   → Eta ≈ 0")
        print("   → Time-invariant inefficiency (like Pitt-Lee model)")

    # Get efficiency estimates
    print("\n5. Efficiency Estimates:")
    print("-" * 70)
    eff_df = result.efficiency()

    # Merge with original data
    df_with_eff = df.merge(
        eff_df[["entity", "efficiency"]], left_on="firm", right_on="entity", how="left"
    )

    print(f"\n   Average efficiency: {eff_df['efficiency'].mean():.3f}")
    print(f"   Min efficiency:     {eff_df['efficiency'].min():.3f}")
    print(f"   Max efficiency:     {eff_df['efficiency'].max():.3f}")

    # Analyze efficiency trends
    print("\n6. Efficiency Trends Over Time:")
    print("-" * 70)

    # Calculate average efficiency by year
    avg_eff_by_year = df_with_eff.groupby("year")["efficiency"].mean()

    print("\n   Year    Avg Efficiency   Change from Previous")
    print("   " + "-" * 50)
    for year in sorted(df_with_eff["year"].unique()):
        eff = avg_eff_by_year[year]
        if year == 0:
            print(f"   {year:4d}      {eff:.4f}              -")
        else:
            change = eff - avg_eff_by_year[year - 1]
            print(f"   {year:4d}      {eff:.4f}        {change:+.4f}")

    # Visualize results
    print("\n7. Creating visualizations...")
    create_visualizations(df_with_eff, eta_est, eta_true)

    # Test if eta is significantly different from 0
    print("\n8. Statistical Test:")
    print("-" * 70)
    # TODO: Implement LR test for eta = 0
    print("   To test if eta = 0, compare BC92 with Pitt-Lee model using LR test")
    print("   LR statistic = 2 * (loglik_BC92 - loglik_PittLee)")
    print("   Under H0: eta = 0, LR ~ χ²(1)")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


def create_visualizations(df, eta_est, eta_true):
    """Create visualization plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Efficiency over time (average across firms)
    ax1 = axes[0, 0]
    avg_eff_by_year = df.groupby("year")["efficiency"].mean()
    avg_true_eff_by_year = df.groupby("year")["true_eff"].mean()

    ax1.plot(avg_eff_by_year.index, avg_eff_by_year.values, "o-", label="Estimated")
    ax1.plot(
        avg_true_eff_by_year.index,
        avg_true_eff_by_year.values,
        "s--",
        label="True",
        alpha=0.7,
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Average Efficiency")
    ax1.set_title("Efficiency Evolution Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Efficiency distribution by year (boxplot)
    ax2 = axes[0, 1]
    years = sorted(df["year"].unique())
    eff_by_year = [df[df["year"] == year]["efficiency"].values for year in years]
    ax2.boxplot(eff_by_year, labels=years)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Efficiency")
    ax2.set_title("Efficiency Distribution by Year")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Individual firm efficiency trajectories (sample)
    ax3 = axes[1, 0]
    sample_firms = np.random.choice(
        df["firm"].unique(), size=min(10, df["firm"].nunique()), replace=False
    )
    for firm in sample_firms:
        firm_data = df[df["firm"] == firm].sort_values("year")
        ax3.plot(firm_data["year"], firm_data["efficiency"], alpha=0.6, linewidth=1)
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Efficiency")
    ax3.set_title(f"Efficiency Trajectories (Sample of {len(sample_firms)} Firms)")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Scatter plot of estimated vs true efficiency
    ax4 = axes[1, 1]
    ax4.scatter(df["true_eff"], df["efficiency"], alpha=0.5)
    ax4.plot([0, 1], [0, 1], "r--", label="45° line")
    ax4.set_xlabel("True Efficiency")
    ax4.set_ylabel("Estimated Efficiency")
    ax4.set_title("Estimated vs True Efficiency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(
        f"BC92 Model Analysis (η_true={eta_true:.3f}, η_est={eta_est:.3f})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save figure
    plt.savefig("bc92_analysis.png", dpi=300, bbox_inches="tight")
    print(f"   Saved figure: bc92_analysis.png")


if __name__ == "__main__":
    main()
