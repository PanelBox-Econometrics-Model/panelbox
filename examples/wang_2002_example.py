"""
Example: Bank Efficiency with Heteroscedastic Inefficiency (Wang 2002)

This example demonstrates the Wang (2002) model for analyzing bank efficiency
with inefficiency determinants affecting both location and scale.

Research Question:
------------------
How do bank characteristics (size, age, capitalization) affect:
1. The average level of inefficiency? (Location effect)
2. The variability of inefficiency? (Scale effect)

Why Wang (2002)?
----------------
Traditional two-stage approaches are inconsistent:
1. Stage 1: Estimate frontier, obtain efficiency scores
2. Stage 2: Regress efficiency on determinants
Problem: Efficiency scores from stage 1 are estimated with error

Wang (2002) provides single-step consistent estimation!

Dataset:
--------
Simulated data for 200 banks with:
- Outputs: Loans, Securities
- Inputs: Labor, Capital
- Determinants: Bank size, age, capital ratio
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox.frontier import StochasticFrontier

# Set random seed for reproducibility
np.random.seed(42)


def generate_bank_data(n_banks=200):
    """Generate simulated bank data with heteroscedastic inefficiency.

    Parameters:
        n_banks: Number of banks

    Returns:
        DataFrame with bank production data
    """
    print("Generating simulated bank data...")
    print(f"  Number of banks: {n_banks}")

    # Bank characteristics (determinants of inefficiency)
    bank_size = np.random.uniform(10, 100, n_banks)  # Total assets (billions)
    bank_age = np.random.uniform(1, 50, n_banks)  # Years in operation
    capital_ratio = np.random.uniform(0.05, 0.15, n_banks)  # Capital/Assets

    # Standardize for better numerical stability
    bank_size_std = (bank_size - bank_size.mean()) / bank_size.std()
    bank_age_std = (bank_age - bank_age.mean()) / bank_age.std()
    capital_ratio_std = (capital_ratio - capital_ratio.mean()) / capital_ratio.std()

    # Inputs (in logs)
    log_labor = np.random.normal(4, 0.5, n_banks)  # Number of employees
    log_capital = np.random.normal(5, 0.6, n_banks)  # Physical capital

    # True production function: Cobb-Douglas
    # log(output) = β0 + β1*log(labor) + β2*log(capital) + v - u
    beta_0 = 2.0  # Constant
    beta_labor = 0.6  # Labor elasticity
    beta_capital = 0.3  # Capital elasticity

    # Inefficiency structure (Wang 2002)
    # Location (mean inefficiency): μ_i = δ0 + δ1*age + δ2*capital_ratio
    # Older banks and highly capitalized banks tend to be MORE inefficient
    delta_0 = 0.2
    delta_age = 0.15  # Older → more bureaucratic → less efficient
    delta_capital = 0.10  # Higher capital ratio → more conservative → less efficient

    # Create Z matrix (location determinants)
    Z = np.column_stack([np.ones(n_banks), bank_age_std, capital_ratio_std])
    mu_i = Z @ np.array([delta_0, delta_age, delta_capital])

    # Scale (variance of inefficiency): ln(σ²_u,i) = γ0 + γ1*size
    # Larger banks have MORE variable inefficiency (diverse operations)
    gamma_0 = -1.5
    gamma_size = 0.3  # Larger → more heterogeneous operations → more variance

    # Create W matrix (scale determinants)
    W = np.column_stack([np.ones(n_banks), bank_size_std])
    ln_sigma_u_sq_i = W @ np.array([gamma_0, gamma_size])
    sigma_u_i = np.sqrt(np.exp(ln_sigma_u_sq_i))

    # Generate inefficiency: u_i ~ N⁺(μ_i, σ²_u,i)
    u = np.abs(np.random.normal(mu_i, sigma_u_i))

    # Random noise: v ~ N(0, σ²_v)
    sigma_v = 0.15
    v = np.random.normal(0, sigma_v, n_banks)

    # Output (in logs)
    log_output = (
        beta_0
        + beta_labor * log_labor
        + beta_capital * log_capital
        + v
        - u  # Production frontier: inefficiency reduces output
    )

    # Create DataFrame
    df = pd.DataFrame(
        {
            "bank_id": range(1, n_banks + 1),
            "log_output": log_output,
            "log_labor": log_labor,
            "log_capital": log_capital,
            "bank_size": bank_size,
            "bank_age": bank_age,
            "capital_ratio": capital_ratio,
            "bank_size_std": bank_size_std,
            "bank_age_std": bank_age_std,
            "capital_ratio_std": capital_ratio_std,
            "true_inefficiency": u,
            "true_efficiency": np.exp(-u),
        }
    )

    print(f"\nData summary:")
    print(f"  Mean output (log): {log_output.mean():.3f}")
    print(f"  Mean inefficiency: {u.mean():.3f}")
    print(f"  Mean efficiency: {np.exp(-u).mean():.3f}")

    return df


def estimate_wang_model(df):
    """Estimate Wang (2002) model with heteroscedastic inefficiency.

    Parameters:
        df: DataFrame with bank data

    Returns:
        SFResult object
    """
    print("\n" + "=" * 70)
    print("WANG (2002) MODEL: HETEROSCEDASTIC INEFFICIENCY")
    print("=" * 70)

    # Specify model
    model = StochasticFrontier(
        data=df,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        frontier="production",
        dist="truncated_normal",
        inefficiency_vars=["bank_age_std", "capital_ratio_std"],  # Z: location
        het_vars=["bank_size_std"],  # W: scale
    )

    print("\nModel specification:")
    print(f"  Frontier: Production")
    print(f"  Distribution: Truncated Normal")
    print(f"  Location determinants (Z): bank_age, capital_ratio")
    print(f"  Scale determinants (W): bank_size")

    # Estimate
    print("\nEstimating model...")
    result = model.fit(verbose=True)

    if not result.converged:
        print("\nWARNING: Model did not converge!")
        return result

    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    print(result.summary())

    return result


def analyze_marginal_effects(result):
    """Analyze and interpret marginal effects.

    Parameters:
        result: SFResult from Wang (2002) model
    """
    print("\n" + "=" * 70)
    print("MARGINAL EFFECTS ANALYSIS")
    print("=" * 70)

    # Location effects (mean inefficiency)
    print("\n1. LOCATION EFFECTS (∂E[u_i]/∂z_k)")
    print("   How characteristics affect AVERAGE inefficiency:")
    print("-" * 70)

    me_location = result.marginal_effects(method="location")
    print(me_location.to_string(index=False))

    print("\nInterpretation (Location):")
    for idx, row in me_location.iterrows():
        var = row["variable"]
        me = row["marginal_effect"]
        pval = row["p_value"]
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))

        if me > 0:
            direction = "INCREASES"
        else:
            direction = "DECREASES"

        print(f"  - {var}: {direction} average inefficiency by {abs(me):.4f} {sig}")

    # Scale effects (variance of inefficiency)
    print("\n2. SCALE EFFECTS (∂σ_u,i/∂w_k)")
    print("   How characteristics affect VARIANCE of inefficiency:")
    print("-" * 70)

    me_scale = result.marginal_effects(method="scale")
    print(me_scale.to_string(index=False))

    print("\nInterpretation (Scale):")
    for idx, row in me_scale.iterrows():
        var = row["variable"]
        me = row["marginal_effect"]
        pval = row["p_value"]
        sig = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))

        if me > 0:
            direction = "INCREASES"
        else:
            direction = "DECREASES"

        print(f"  - {var}: {direction} variability of inefficiency {sig}")


def visualize_results(result, df):
    """Create visualizations of Wang (2002) results.

    Parameters:
        result: SFResult from Wang (2002) model
        df: Original data
    """
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    # Get efficiency estimates
    eff_df = result.efficiency(estimator="bc")
    df = df.copy()
    df["estimated_efficiency"] = eff_df["efficiency"].values

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Wang (2002) Model: Bank Efficiency Analysis", fontsize=16, fontweight="bold")

    # 1. Efficiency distribution
    ax1 = axes[0, 0]
    ax1.hist(df["estimated_efficiency"], bins=30, edgecolor="black", alpha=0.7)
    ax1.axvline(
        df["estimated_efficiency"].mean(),
        color="red",
        linestyle="--",
        label=f'Mean = {df["estimated_efficiency"].mean():.3f}',
    )
    ax1.set_xlabel("Efficiency")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Efficiency Scores")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Efficiency vs. Bank Age
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df["bank_age"], df["estimated_efficiency"], c=df["capital_ratio"], cmap="viridis", alpha=0.6
    )
    ax2.set_xlabel("Bank Age (years)")
    ax2.set_ylabel("Efficiency")
    ax2.set_title("Efficiency vs. Bank Age")
    plt.colorbar(scatter, ax=ax2, label="Capital Ratio")
    ax2.grid(alpha=0.3)

    # Add trend line
    z = np.polyfit(df["bank_age"], df["estimated_efficiency"], 1)
    p = np.poly1d(z)
    ax2.plot(df["bank_age"], p(df["bank_age"]), "r--", alpha=0.8, label="Trend")
    ax2.legend()

    # 3. Efficiency vs. Bank Size
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        df["bank_size"], df["estimated_efficiency"], c=df["bank_age"], cmap="plasma", alpha=0.6
    )
    ax3.set_xlabel("Bank Size (total assets, billions)")
    ax3.set_ylabel("Efficiency")
    ax3.set_title("Efficiency vs. Bank Size")
    plt.colorbar(scatter, ax=ax3, label="Bank Age")
    ax3.grid(alpha=0.3)

    # 4. True vs. Estimated Efficiency
    ax4 = axes[1, 1]
    ax4.scatter(df["true_efficiency"], df["estimated_efficiency"], alpha=0.5)
    ax4.plot([0, 1], [0, 1], "r--", label="45° line")
    ax4.set_xlabel("True Efficiency")
    ax4.set_ylabel("Estimated Efficiency")
    ax4.set_title("True vs. Estimated Efficiency")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add correlation
    corr = np.corrcoef(df["true_efficiency"], df["estimated_efficiency"])[0, 1]
    ax4.text(
        0.05,
        0.95,
        f"Correlation: {corr:.3f}",
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig("wang_2002_analysis.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved as 'wang_2002_analysis.png'")

    # Show plot
    plt.show()


def main():
    """Main analysis workflow."""
    print("=" * 70)
    print("WANG (2002) HETEROSCEDASTIC INEFFICIENCY MODEL")
    print("Application: Bank Efficiency Analysis")
    print("=" * 70)

    # 1. Generate data
    df = generate_bank_data(n_banks=200)

    # 2. Estimate Wang model
    result = estimate_wang_model(df)

    # 3. Analyze marginal effects
    analyze_marginal_effects(result)

    # 4. Visualize results
    visualize_results(result, df)

    # 5. Key takeaways
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. SINGLE-STEP ESTIMATION: Wang (2002) avoids bias from two-stage methods

    2. LOCATION EFFECTS (δ):
       - Positive: Variable INCREASES average inefficiency
       - Example: Older banks tend to be more inefficient on average

    3. SCALE EFFECTS (γ):
       - Positive: Variable INCREASES variance of inefficiency
       - Example: Larger banks have more variable efficiency
         (some very efficient, some very inefficient)

    4. POLICY IMPLICATIONS:
       - Location effects → Target intervention to specific bank types
       - Scale effects → Understand heterogeneity in treatment effects

    5. ADVANTAGES OVER BC95:
       - BC95: Only models μ_i (location), assumes constant σ²_u
       - Wang (2002): Models BOTH μ_i and σ²_u,i
       - Richer understanding of inefficiency sources
    """
    )


if __name__ == "__main__":
    main()
