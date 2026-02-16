"""
Example: Manufacturing Efficiency with Time-Varying Inefficiency (Kumbhakar 1990)

This example demonstrates the Kumbhakar (1990) model for analyzing firm efficiency
with time-varying inefficiency following a flexible logistic pattern.

Research Question:
------------------
How does firm inefficiency evolve over time?
- Learning: Firms become more efficient as they gain experience
- Degradation: Firms become less efficient due to aging capital
- Non-monotonic: Efficiency first improves, then deteriorates

Why Kumbhakar (1990)?
---------------------
Traditional panel models assume time-invariant inefficiency:
- Pitt-Lee (1981): u_i constant over time
Problem: Unrealistic for many applications!

Kumbhakar (1990) allows flexible time patterns via logistic function:
- B(t) = 1 / [1 + exp(b*t + c*t²)]
- Non-monotonic patterns (U-shape, inverted-U)
- Only 2 parameters (parsimonious!)

Dataset:
--------
Simulated panel data for 100 manufacturing firms over 10 years with:
- Output: Value added
- Inputs: Labor, capital, materials
- Time pattern: Learning effect (inefficiency decreases over time)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox.frontier import StochasticFrontier

# Set random seed for reproducibility
np.random.seed(42)


def generate_panel_data_learning(n_firms=100, n_periods=10):
    """Generate panel data with learning pattern (inefficiency decreases).

    Parameters:
        n_firms: Number of firms
        n_periods: Number of time periods

    Returns:
        DataFrame with panel data
    """
    print("Generating panel data with LEARNING pattern...")
    print(f"  Number of firms: {n_firms}")
    print(f"  Time periods: {n_periods}")

    # True production function parameters
    beta_0 = 3.0  # Constant
    beta_labor = 0.4
    beta_capital = 0.3
    beta_materials = 0.25

    # Time pattern parameters (LEARNING)
    # b < 0: inefficiency weight increases over time
    # c > 0: growth slows down (convex pattern)
    b_true = -0.3  # Negative: learning effect
    c_true = 0.02  # Positive: learning slows down

    # Inefficiency parameters
    mu_true = 0.5  # Mean base inefficiency
    sigma_u_true = 0.3  # Std of base inefficiency
    sigma_v_true = 0.15  # Noise

    print(f"\nTrue time pattern parameters:")
    print(f"  b = {b_true:.3f} (< 0: learning)")
    print(f"  c = {c_true:.3f} (> 0: learning slows down)")

    data = []

    for i in range(n_firms):
        # Firm-specific base inefficiency (time-invariant component)
        u_i = np.abs(np.random.normal(mu_true, sigma_u_true))

        for t in range(n_periods):
            # Inputs (in logs)
            log_labor = np.random.normal(4, 0.5)
            log_capital = np.random.normal(5, 0.6)
            log_materials = np.random.normal(4.5, 0.5)

            # Time-varying inefficiency weight
            # B(t) = 1 / [1 + exp(b*t + c*t²)]
            B_t = 1.0 / (1.0 + np.exp(b_true * t + c_true * t**2))

            # Time-varying inefficiency
            u_it = B_t * u_i

            # Random noise
            v_it = np.random.normal(0, sigma_v_true)

            # Output (in logs)
            log_output = (
                beta_0
                + beta_labor * log_labor
                + beta_capital * log_capital
                + beta_materials * log_materials
                + v_it
                - u_it  # Production frontier: inefficiency reduces output
            )

            data.append(
                {
                    "firm": i,
                    "time": t,
                    "log_output": log_output,
                    "log_labor": log_labor,
                    "log_capital": log_capital,
                    "log_materials": log_materials,
                    "true_B_t": B_t,
                    "true_u_it": u_it,
                    "true_efficiency": np.exp(-u_it),
                }
            )

    df = pd.DataFrame(data)

    print(f"\nData summary:")
    print(f"  Total observations: {len(df)}")
    print(f"  Mean efficiency (period 0): {df[df['time']==0]['true_efficiency'].mean():.3f}")
    print(
        f"  Mean efficiency (period {n_periods-1}): {df[df['time']==n_periods-1]['true_efficiency'].mean():.3f}"
    )
    print(
        f"  Improvement: {(df[df['time']==n_periods-1]['true_efficiency'].mean() - df[df['time']==0]['true_efficiency'].mean()):.3f}"
    )

    return df


def estimate_kumbhakar_model(df):
    """Estimate Kumbhakar (1990) model.

    Parameters:
        df: Panel DataFrame

    Returns:
        SFResult object
    """
    print("\n" + "=" * 70)
    print("KUMBHAKAR (1990) MODEL: TIME-VARYING INEFFICIENCY")
    print("=" * 70)

    # Specify model
    model = StochasticFrontier(
        data=df,
        depvar="log_output",
        exog=["log_labor", "log_capital", "log_materials"],
        entity="firm",
        time="time",
        frontier="production",
        model_type="kumbhakar_1990",
    )

    print("\nModel specification:")
    print(f"  Model: Kumbhakar (1990)")
    print(f"  Frontier: Production")
    print(f"  Time pattern: B(t) = 1 / [1 + exp(b*t + c*t²)]")

    # Estimate
    print("\nEstimating model (this may take a minute)...")
    result = model.fit(verbose=True, maxiter=300)

    if not result.converged:
        print("\nWARNING: Model did not converge!")
        return result

    print("\n" + "=" * 70)
    print("ESTIMATION RESULTS")
    print("=" * 70)
    print(result.summary())

    return result


def interpret_time_pattern(result):
    """Interpret the time pattern from estimated parameters.

    Parameters:
        result: SFResult from Kumbhakar model
    """
    print("\n" + "=" * 70)
    print("TIME PATTERN INTERPRETATION")
    print("=" * 70)

    # Extract time pattern parameters
    b_est = result.params["b"]
    c_est = result.params["c"]

    print(f"\nEstimated parameters:")
    print(f"  b = {b_est:.4f}")
    print(f"  c = {c_est:.4f}")

    # Interpretation
    print(f"\nInterpretation:")

    if abs(b_est) < 0.1 and abs(c_est) < 0.1:
        print("  → Time pattern is approximately FLAT (time-invariant)")
        print("    Similar to Pitt-Lee (1981) model")
    elif b_est < 0:
        print("  → LEARNING pattern detected!")
        print("    Inefficiency weight B(t) increases over time")
        print("    → Firms become more efficient as they gain experience")
        if c_est > 0:
            print("    → Learning slows down over time (convex pattern)")
        else:
            print("    → Learning accelerates over time (concave pattern)")
    else:
        print("  → DEGRADATION pattern detected!")
        print("    Inefficiency weight B(t) decreases over time")
        print("    → Firms become less efficient (aging capital, organizational decay)")
        if c_est < 0:
            print("    → Degradation slows down over time")
        else:
            print("    → Degradation accelerates over time")

    # Compute B(t) over time
    time_range = np.arange(10)
    B_t = 1.0 / (1.0 + np.exp(b_est * time_range + c_est * time_range**2))

    print(f"\nB(t) evolution:")
    for t in time_range:
        print(f"  Period {t}: B(t) = {B_t[t]:.4f}")


def visualize_results(result, df):
    """Create visualizations of Kumbhakar (1990) results.

    Parameters:
        result: SFResult from Kumbhakar model
        df: Original data
    """
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

    # Extract parameters
    b_est = result.params["b"]
    c_est = result.params["c"]

    # Get efficiency estimates
    from panelbox.frontier.efficiency import estimate_panel_efficiency

    eff_df = estimate_panel_efficiency(result, estimator="bc", by_period=True)

    # Merge with original data
    df = df.merge(
        eff_df[["entity", "time", "efficiency"]],
        left_on=["firm", "time"],
        right_on=["entity", "time"],
        how="left",
    )
    df.rename(columns={"efficiency": "estimated_efficiency"}, inplace=True)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Kumbhakar (1990) Model: Time-Varying Efficiency Analysis", fontsize=16, fontweight="bold"
    )

    # 1. B(t) function
    ax1 = axes[0, 0]
    time_range = np.arange(10)
    B_t_est = 1.0 / (1.0 + np.exp(b_est * time_range + c_est * time_range**2))
    B_t_true = df.groupby("time")["true_B_t"].mean().values

    ax1.plot(time_range, B_t_true, "o-", label="True B(t)", linewidth=2)
    ax1.plot(time_range, B_t_est, "s--", label="Estimated B(t)", linewidth=2)
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("B(t) - Inefficiency Weight")
    ax1.set_title("Time Pattern: B(t) Evolution")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Average efficiency over time
    ax2 = axes[0, 1]
    eff_by_time_true = df.groupby("time")["true_efficiency"].mean()
    eff_by_time_est = df.groupby("time")["estimated_efficiency"].mean()

    ax2.plot(
        eff_by_time_true.index, eff_by_time_true.values, "o-", label="True Efficiency", linewidth=2
    )
    ax2.plot(
        eff_by_time_est.index,
        eff_by_time_est.values,
        "s--",
        label="Estimated Efficiency",
        linewidth=2,
    )
    ax2.set_xlabel("Time Period")
    ax2.set_ylabel("Average Efficiency")
    ax2.set_title("Average Efficiency Over Time")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Efficiency distribution by period
    ax3 = axes[1, 0]
    periods_to_plot = [0, 4, 9]
    for period in periods_to_plot:
        data_period = df[df["time"] == period]["estimated_efficiency"]
        ax3.hist(data_period, alpha=0.5, bins=20, label=f"Period {period}", edgecolor="black")
    ax3.set_xlabel("Efficiency")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Efficiency Distribution Across Periods")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. True vs. Estimated Efficiency
    ax4 = axes[1, 1]
    ax4.scatter(
        df["true_efficiency"], df["estimated_efficiency"], alpha=0.3, c=df["time"], cmap="viridis"
    )
    ax4.plot([0, 1], [0, 1], "r--", label="45° line")
    ax4.set_xlabel("True Efficiency")
    ax4.set_ylabel("Estimated Efficiency")
    ax4.set_title("True vs. Estimated Efficiency")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add correlation
    corr = np.corrcoef(df["true_efficiency"].dropna(), df["estimated_efficiency"].dropna())[0, 1]
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
    plt.savefig("kumbhakar_1990_analysis.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved as 'kumbhakar_1990_analysis.png'")

    # Show plot
    plt.show()


def main():
    """Main analysis workflow."""
    print("=" * 70)
    print("KUMBHAKAR (1990) TIME-VARYING INEFFICIENCY MODEL")
    print("Application: Manufacturing Firm Efficiency with Learning")
    print("=" * 70)

    # 1. Generate data
    df = generate_panel_data_learning(n_firms=100, n_periods=10)

    # 2. Estimate Kumbhakar model
    result = estimate_kumbhakar_model(df)

    # 3. Interpret time pattern
    interpret_time_pattern(result)

    # 4. Visualize results
    visualize_results(result, df)

    # 5. Key takeaways
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. FLEXIBLE TIME PATTERNS: Kumbhakar (1990) allows non-monotonic evolution
       - Learning: b < 0 (inefficiency decreases)
       - Degradation: b > 0 (inefficiency increases)
       - U-shape or inverted-U with quadratic term c

    2. PARSIMONIOUS: Only 2 time parameters (b, c) for entire panel
       - Common pattern across all firms
       - More restrictive than firm-specific patterns (e.g., CSS)
       - But much easier to estimate!

    3. NESTS PITT-LEE: When b = c = 0, reduces to time-invariant model
       - Can test H₀: b = c = 0 using LR test

    4. INTERPRETATION:
       - B(t) = inefficiency weight at time t
       - u_it = B(t) × u_i
       - Higher B(t) → more inefficiency
       - TE_it = exp(-u_it) = exp(-B(t)×u_i)

    5. WHEN TO USE:
       - Believe inefficiency evolves over time
       - Want common time pattern (not firm-specific)
       - Have moderate T (at least 5-6 periods)
       - Want to capture learning or degradation effects

    6. ALTERNATIVES:
       - Battese-Coelli (1992): Monotonic exponential decay
       - Lee-Schmidt (1993): Time dummies (more flexible, less parsimonious)
       - CSS (1990): Firm-specific time patterns (very flexible!)
    """
    )


if __name__ == "__main__":
    main()
