"""
Basic Usage Examples for Stochastic Frontier Analysis (SFA)

This script demonstrates the basic usage of the SFA module in PanelBox,
including estimation of production and cost frontiers with different
distributional assumptions.

Examples:
    1. Cross-sectional production frontier with half-normal distribution
    2. Cost frontier with exponential distribution
    3. Comparing multiple distributional specifications
    4. Extracting and analyzing efficiency estimates
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier import DistributionType, FrontierType, StochasticFrontier


def simulate_production_data(n=500, seed=42):
    """Simulate data for a Cobb-Douglas production frontier.

    Model:
        ln(Y) = β₀ + β₁*ln(L) + β₂*ln(K) + v - u

    where:
        v ~ N(0, σ²_v) is random noise
        u ~ N⁺(0, σ²_u) is technical inefficiency

    Parameters:
        n: Number of observations
        seed: Random seed

    Returns:
        DataFrame with columns: log_output, log_labor, log_capital
    """
    np.random.seed(seed)

    # True parameters
    beta_0 = 2.0
    beta_1 = 0.6  # Labor elasticity
    beta_2 = 0.3  # Capital elasticity
    sigma_v = 0.1  # Noise std dev
    sigma_u = 0.2  # Inefficiency std dev

    # Generate inputs (log scale)
    log_labor = np.random.uniform(0, 3, n)
    log_capital = np.random.uniform(0, 3, n)

    # Generate errors
    v = np.random.normal(0, sigma_v, n)
    u = np.abs(np.random.normal(0, sigma_u, n))  # Half-normal

    # Generate output (log scale)
    log_output = beta_0 + beta_1 * log_labor + beta_2 * log_capital + v - u

    # Create DataFrame
    data = pd.DataFrame(
        {
            "log_output": log_output,
            "log_labor": log_labor,
            "log_capital": log_capital,
            "true_efficiency": np.exp(-u),
        }
    )

    print(f"\nSimulated {n} observations")
    print(f"True parameters:")
    print(f"  β₀ = {beta_0:.2f}, β₁ = {beta_1:.2f}, β₂ = {beta_2:.2f}")
    print(f"  σ_v = {sigma_v:.2f}, σ_u = {sigma_u:.2f}")
    print(f"  True mean efficiency: {np.exp(-u).mean():.4f}")

    return data


def example_1_basic_production_frontier():
    """Example 1: Basic production frontier with half-normal distribution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Production Frontier with Half-Normal Distribution")
    print("=" * 70)

    # Simulate data
    data = simulate_production_data(n=500)

    # Specify model
    sf = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )

    print(f"\nModel specification:")
    print(sf)

    # Estimate via MLE
    print("\nEstimating via MLE...")
    result = sf.fit(method="mle", verbose=True)

    # Display summary
    print("\n" + result.summary())

    # Extract efficiency estimates
    print("\nComputing efficiency estimates...")
    eff_jlms = result.efficiency(estimator="jlms")
    eff_bc = result.efficiency(estimator="bc")

    print("\nEfficiency Statistics (JLMS):")
    print(eff_jlms["efficiency"].describe())

    print("\nEfficiency Statistics (BC):")
    print(eff_bc["efficiency"].describe())

    # Compare with true efficiency
    if "true_efficiency" in data.columns:
        correlation = np.corrcoef(data["true_efficiency"], eff_bc["efficiency"])[0, 1]
        print(f"\nCorrelation with true efficiency: {correlation:.4f}")

    return result, eff_bc


def example_2_cost_frontier():
    """Example 2: Cost frontier with exponential distribution."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Cost Frontier with Exponential Distribution")
    print("=" * 70)

    # Simulate cost data
    np.random.seed(123)
    n = 500

    # True parameters for cost function
    beta_0 = 1.0
    beta_1 = 0.4
    beta_2 = 0.5
    sigma_v = 0.1
    sigma_u = 0.15

    # Generate data
    log_labor = np.random.uniform(0, 2, n)
    log_capital = np.random.uniform(0, 2, n)

    v = np.random.normal(0, sigma_v, n)
    u = np.random.exponential(sigma_u, n)  # Exponential

    # Cost function: higher u means higher cost
    log_cost = beta_0 + beta_1 * log_labor + beta_2 * log_capital + v + u

    data = pd.DataFrame({"log_cost": log_cost, "log_labor": log_labor, "log_capital": log_capital})

    # Estimate cost frontier
    sf = StochasticFrontier(
        data=data,
        depvar="log_cost",
        exog=["log_labor", "log_capital"],
        frontier="cost",
        dist="exponential",
    )

    print(f"\nModel specification:")
    print(sf)

    result = sf.fit(method="mle", verbose=False)

    print("\n" + result.summary())

    # Cost efficiency
    eff = result.efficiency(estimator="bc")
    print("\nCost Efficiency Statistics:")
    print(eff["efficiency"].describe())
    print("\nNote: Cost efficiency > 1 means actual cost exceeds minimum cost")

    return result, eff


def example_3_compare_distributions():
    """Example 3: Compare different distributional assumptions."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Comparing Distributional Assumptions")
    print("=" * 70)

    # Simulate data
    data = simulate_production_data(n=500, seed=999)

    # Estimate with different distributions
    distributions = ["half_normal", "exponential", "truncated_normal"]
    results = {}

    for dist in distributions:
        print(f"\nEstimating with {dist} distribution...")

        sf = StochasticFrontier(
            data=data,
            depvar="log_output",
            exog=["log_labor", "log_capital"],
            frontier="production",
            dist=dist,
        )

        result = sf.fit(method="mle", verbose=False)
        results[dist] = result

        print(f"  Log-likelihood: {result.loglik:.4f}")
        print(f"  AIC: {result.aic:.4f}")
        print(f"  BIC: {result.bic:.4f}")

    # Compare models
    print("\nModel Comparison:")
    comparison = results["half_normal"].compare_distributions(
        [results["exponential"], results["truncated_normal"]]
    )
    print(comparison.to_string(index=False))

    # Plot efficiency distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, dist in enumerate(distributions):
        eff = results[dist].efficiency(estimator="bc")
        axes[i].hist(eff["efficiency"], bins=30, alpha=0.7, edgecolor="black")
        axes[i].set_title(f'{dist.replace("_", " ").title()}')
        axes[i].set_xlabel("Technical Efficiency")
        axes[i].set_ylabel("Frequency")
        axes[i].axvline(
            eff["efficiency"].mean(),
            color="red",
            linestyle="--",
            label=f'Mean: {eff["efficiency"].mean():.3f}',
        )
        axes[i].legend()

    plt.tight_layout()
    plt.savefig("sfa_distribution_comparison.png", dpi=150)
    print("\nEfficiency distributions saved to: sfa_distribution_comparison.png")

    return results, comparison


def example_4_efficiency_analysis():
    """Example 4: Detailed efficiency analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Detailed Efficiency Analysis")
    print("=" * 70)

    # Simulate data
    data = simulate_production_data(n=500)

    # Estimate model
    sf = StochasticFrontier(
        data=data,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="half_normal",
    )

    result = sf.fit(method="mle", verbose=False)

    # Get efficiency estimates
    eff = result.efficiency(estimator="bc", ci_level=0.95)

    print("\nEfficiency Summary:")
    print(eff.describe())

    # Identify most and least efficient units
    print("\nTop 5 Most Efficient Units:")
    print(eff.nlargest(5, "efficiency")[["efficiency", "ci_lower", "ci_upper"]])

    print("\nTop 5 Least Efficient Units:")
    print(eff.nsmallest(5, "efficiency")[["efficiency", "ci_lower", "ci_upper"]])

    # Plot efficiency with confidence intervals
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(eff["efficiency"], bins=30, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Technical Efficiency")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Technical Efficiency")
    ax1.axvline(
        eff["efficiency"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean: {eff["efficiency"].mean():.3f}',
    )
    ax1.legend()

    # Efficiency with CIs (sorted)
    eff_sorted = eff.sort_values("efficiency").reset_index(drop=True)
    indices = eff_sorted.index[::10]  # Sample every 10th for readability

    ax2.errorbar(
        indices,
        eff_sorted.loc[indices, "efficiency"],
        yerr=[
            eff_sorted.loc[indices, "efficiency"] - eff_sorted.loc[indices, "ci_lower"],
            eff_sorted.loc[indices, "ci_upper"] - eff_sorted.loc[indices, "efficiency"],
        ],
        fmt="o",
        markersize=3,
        capsize=3,
        alpha=0.6,
    )
    ax2.set_xlabel("Observation (sorted by efficiency)")
    ax2.set_ylabel("Technical Efficiency")
    ax2.set_title("Efficiency Estimates with 95% Confidence Intervals")
    ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig("sfa_efficiency_analysis.png", dpi=150)
    print("\nEfficiency analysis saved to: sfa_efficiency_analysis.png")

    return result, eff


if __name__ == "__main__":
    """Run all examples."""
    print("\n" + "=" * 70)
    print("STOCHASTIC FRONTIER ANALYSIS - BASIC USAGE EXAMPLES")
    print("=" * 70)

    # Run examples
    try:
        result1, eff1 = example_1_basic_production_frontier()
        result2, eff2 = example_2_cost_frontier()
        results3, comparison3 = example_3_compare_distributions()
        result4, eff4 = example_4_efficiency_analysis()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
