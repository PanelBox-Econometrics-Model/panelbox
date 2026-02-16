"""
Example: Manufacturing Efficiency with Flexible Time Pattern (Lee & Schmidt 1993)

This example demonstrates the Lee & Schmidt (1993) model for analyzing firm efficiency
with time-varying inefficiency via time-specific scale factors (time dummies).

Research Question:
------------------
How does inefficiency evolve flexibly over time without imposing parametric restrictions?
- Does inefficiency follow a learning pattern?
- Are there structural breaks or non-monotonic patterns?
- Do different time periods have systematically different inefficiency levels?

Why Lee & Schmidt (1993)?
--------------------------
Existing models impose strong restrictions:
- Pitt-Lee (1981): Time-invariant (too restrictive!)
- Battese-Coelli (1992): Monotonic exponential decay
- Kumbhakar (1990): Logistic pattern (2 parameters)

Lee & Schmidt (1993) offers maximum flexibility:
- u_it = δ_i × ξ_t
- δ_i: Firm-specific inefficiency level (N parameters)
- ξ_t: Time-specific scale factor (T-1 parameters after normalization)
- NO parametric restrictions on time pattern!
- Can capture ANY time pattern (learning, degradation, cycles, breaks)

Trade-off:
- More flexible than Kumbhakar (1990)
- But requires many parameters (N + T-1)
- Need sufficient observations (moderate to large T)

Dataset:
--------
Simulated panel data for 80 firms over 8 years with:
- Output: Value added
- Inputs: Labor, capital
- Time pattern: Non-monotonic (U-shaped inefficiency)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier

# Set random seed for reproducibility
np.random.seed(42)


def generate_panel_data_nonmonotonic(n_firms=80, n_periods=8):
    """Generate panel data with non-monotonic time pattern (U-shape).

    Parameters:
        n_firms: Number of firms
        n_periods: Number of time periods

    Returns:
        DataFrame with panel data
    """
    print("Generating panel data with NON-MONOTONIC time pattern...")
    print(f"  Number of firms: {n_firms}")
    print(f"  Time periods: {n_periods}")

    # True production function parameters
    beta_0 = 3.5
    beta_labor = 0.5
    beta_capital = 0.4

    # Non-monotonic time pattern (U-shape)
    # ξ_t: first decreases (learning), then increases (degradation)
    # Normalized so ξ_0 = 1.0
    xi_t_true = np.array([1.0, 0.85, 0.72, 0.65, 0.68, 0.78, 0.92, 1.05])

    print(f"\nTrue time pattern ξ_t (normalized ξ_0=1):")
    for t, xi in enumerate(xi_t_true):
        print(f"  Period {t}: ξ_t = {xi:.3f}")

    # Inefficiency parameters
    sigma_u_true = 0.4
    mu_true = 0.3
    sigma_v_true = 0.12

    data = []

    for i in range(n_firms):
        # Firm-specific inefficiency level
        delta_i = np.abs(np.random.normal(mu_true, sigma_u_true))

        for t in range(n_periods):
            # Inputs (in logs)
            log_labor = np.random.normal(4, 0.6)
            log_capital = np.random.normal(5, 0.7)

            # Time-varying inefficiency
            # u_it = δ_i × ξ_t
            u_it = delta_i * xi_t_true[t]

            # Random noise
            v_it = np.random.normal(0, sigma_v_true)

            # Output (in logs)
            log_output = beta_0 + beta_labor * log_labor + beta_capital * log_capital + v_it - u_it

            data.append(
                {
                    "firm": i,
                    "time": t,
                    "log_output": log_output,
                    "log_labor": log_labor,
                    "log_capital": log_capital,
                    "true_delta_i": delta_i,
                    "true_xi_t": xi_t_true[t],
                    "true_u_it": u_it,
                    "true_efficiency": np.exp(-u_it),
                }
            )

    df = pd.DataFrame(data)

    print(f"\nData summary:")
    print(f"  Total observations: {len(df)}")
    print(f"  Mean efficiency (period 0): {df[df['time']==0]['true_efficiency'].mean():.3f}")
    print(f"  Mean efficiency (period 3): {df[df['time']==3]['true_efficiency'].mean():.3f}")
    print(f"  Mean efficiency (period 7): {df[df['time']==7]['true_efficiency'].mean():.3f}")
    print(f"\nTime pattern: U-shaped (learning then degradation)")

    return df, xi_t_true


def estimate_lee_schmidt_model(df):
    """Estimate Lee & Schmidt (1993) model.

    Parameters:
        df: Panel DataFrame

    Returns:
        SFResult object
    """
    print("\n" + "=" * 70)
    print("LEE & SCHMIDT (1993) MODEL: FLEXIBLE TIME-VARYING INEFFICIENCY")
    print("=" * 70)

    # Specify model
    model = StochasticFrontier(
        data=df,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity="firm",
        time="time",
        frontier="production",
        model_type="lee_schmidt_1993",
    )

    print("\nModel specification:")
    print(f"  Model: Lee & Schmidt (1993)")
    print(f"  Frontier: Production")
    print(f"  Structure: u_it = δ_i × ξ_t")
    print(f"  Normalization: ξ_0 = 1 (first period)")

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


def interpret_time_pattern(result, xi_t_true):
    """Interpret the estimated time pattern.

    Parameters:
        result: SFResult from Lee-Schmidt model
        xi_t_true: True time pattern
    """
    print("\n" + "=" * 70)
    print("TIME PATTERN INTERPRETATION")
    print("=" * 70)

    # Extract estimated time pattern
    # Look for parameters named 'xi_t_1', 'xi_t_2', etc.
    xi_params = [p for p in result.params.index if p.startswith("xi_")]

    if len(xi_params) == 0:
        print("\nNote: Time pattern parameters not found in params.")
        print("The model may store them separately in temporal_params.")
        if hasattr(result, "temporal_params"):
            print(f"Temporal params: {result.temporal_params}")
        return

    # Reconstruct full time pattern (including normalized ξ_0 = 1)
    n_periods = len(xi_params) + 1
    xi_t_est = np.ones(n_periods)
    for i, param in enumerate(xi_params):
        xi_t_est[i + 1] = result.params[param]

    print(f"\nEstimated time pattern ξ_t:")
    print(f"{'Period':<10}{'True ξ_t':<15}{'Estimated ξ_t':<15}{'Difference':<15}")
    print("-" * 60)
    for t in range(n_periods):
        true_val = xi_t_true[t] if t < len(xi_t_true) else np.nan
        est_val = xi_t_est[t]
        diff = est_val - true_val if not np.isnan(true_val) else np.nan
        print(f"{t:<10}{true_val:<15.4f}{est_val:<15.4f}{diff:<15.4f}")

    # Interpretation
    print(f"\nInterpretation:")

    # Check if pattern is monotonic
    diffs = np.diff(xi_t_est)
    is_increasing = np.all(diffs > 0)
    is_decreasing = np.all(diffs < 0)

    if is_decreasing:
        print("  → LEARNING pattern: ξ_t decreases monotonically")
        print("    Firms become more efficient over time")
    elif is_increasing:
        print("  → DEGRADATION pattern: ξ_t increases monotonically")
        print("    Firms become less efficient over time")
    else:
        print("  → NON-MONOTONIC pattern detected!")

        # Find minimum
        min_idx = np.argmin(xi_t_est)
        print(f"    Minimum ξ_t at period {min_idx}: ξ_{min_idx} = {xi_t_est[min_idx]:.4f}")

        if min_idx > 0 and min_idx < n_periods - 1:
            print("    → U-SHAPED pattern:")
            print(f"      Periods 0-{min_idx}: Learning (ξ_t decreases)")
            print(f"      Periods {min_idx}-{n_periods-1}: Degradation (ξ_t increases)")
            print("      Interpretation: Initial learning, then organizational decay")

    # Statistical test for time variation
    print(f"\n  Statistical test: H₀: ξ_1 = ξ_2 = ... = ξ_T (time-invariant)")
    print(f"  If time pattern is flat → Use Pitt-Lee instead (more efficient)")


def compare_with_kumbhakar(result_ls, df):
    """Compare Lee-Schmidt with Kumbhakar (1990).

    Parameters:
        result_ls: Lee-Schmidt result
        df: Original data
    """
    print("\n" + "=" * 70)
    print("COMPARISON: LEE-SCHMIDT vs KUMBHAKAR (1990)")
    print("=" * 70)

    print("\nEstimating Kumbhakar (1990) for comparison...")

    model_k90 = StochasticFrontier(
        data=df,
        depvar="log_output",
        exog=["log_labor", "log_capital"],
        entity="firm",
        time="time",
        frontier="production",
        model_type="kumbhakar_1990",
    )

    result_k90 = model_k90.fit(verbose=False, maxiter=300)

    print(f"\nModel comparison:")
    print(f"{'Metric':<30}{'Lee-Schmidt':<20}{'Kumbhakar':<20}")
    print("-" * 70)
    print(f"{'Log-likelihood':<30}{result_ls.loglik:<20.4f}{result_k90.loglik:<20.4f}")
    print(f"{'AIC':<30}{result_ls.aic:<20.4f}{result_k90.aic:<20.4f}")
    print(f"{'BIC':<30}{result_ls.bic:<20.4f}{result_k90.bic:<20.4f}")

    # Number of parameters
    n_params_ls = len(result_ls.params)
    n_params_k90 = len(result_k90.params)
    print(f"{'Number of parameters':<30}{n_params_ls:<20}{n_params_k90:<20}")

    # Vuong test or LR test
    print(f"\nInterpretation:")
    if result_ls.aic < result_k90.aic:
        print("  → Lee-Schmidt has lower AIC (better fit, accounting for complexity)")
        print("    Non-monotonic pattern is important for this data!")
    else:
        print("  → Kumbhakar has lower AIC (more parsimonious)")
        print("    Logistic pattern is sufficient for this data")

    print(f"\nTrade-offs:")
    print(f"  Lee-Schmidt: Very flexible, but requires {n_params_ls} parameters")
    print(f"  Kumbhakar: Parsimonious ({n_params_k90} params), but assumes logistic pattern")


def visualize_results(result, df, xi_t_true):
    """Create visualizations of Lee & Schmidt (1993) results.

    Parameters:
        result: SFResult from Lee-Schmidt model
        df: Original data
        xi_t_true: True time pattern
    """
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70)

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

    # Extract estimated time pattern
    xi_params = [p for p in result.params.index if p.startswith("xi_")]
    n_periods = len(xi_params) + 1
    xi_t_est = np.ones(n_periods)
    for i, param in enumerate(xi_params):
        xi_t_est[i + 1] = result.params[param]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Lee & Schmidt (1993) Model: Flexible Time-Varying Efficiency Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Time pattern ξ_t
    ax1 = axes[0, 0]
    time_range = np.arange(n_periods)

    ax1.plot(time_range, xi_t_true, "o-", label="True ξ_t", linewidth=2, markersize=8)
    ax1.plot(time_range, xi_t_est, "s--", label="Estimated ξ_t", linewidth=2, markersize=8)
    ax1.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5, label="Normalized level (ξ_0=1)")
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("ξ_t - Time-Specific Scale Factor")
    ax1.set_title("Time Pattern: ξ_t Evolution (Non-monotonic)")
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
    ax2.set_title("Average Efficiency Over Time (Inverted U-shape)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Efficiency distribution by period (violin plot)
    ax3 = axes[1, 0]
    periods_to_plot = [0, 3, 7]
    data_for_violin = [
        df[df["time"] == period]["estimated_efficiency"].dropna() for period in periods_to_plot
    ]

    parts = ax3.violinplot(data_for_violin, positions=periods_to_plot, widths=0.7, showmeans=True)
    ax3.set_xlabel("Time Period")
    ax3.set_ylabel("Efficiency")
    ax3.set_title("Efficiency Distribution Across Periods")
    ax3.set_xticks(periods_to_plot)
    ax3.set_xticklabels([f"t={t}" for t in periods_to_plot])
    ax3.grid(alpha=0.3, axis="y")

    # 4. True vs. Estimated Efficiency
    ax4 = axes[1, 1]
    scatter = ax4.scatter(
        df["true_efficiency"], df["estimated_efficiency"], alpha=0.4, c=df["time"], cmap="viridis"
    )
    ax4.plot([0, 1], [0, 1], "r--", label="45° line", linewidth=2)
    ax4.set_xlabel("True Efficiency")
    ax4.set_ylabel("Estimated Efficiency")
    ax4.set_title("True vs. Estimated Efficiency")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax4, label="Time Period")

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
    plt.savefig("lee_schmidt_1993_analysis.png", dpi=150, bbox_inches="tight")
    print("\nFigure saved as 'lee_schmidt_1993_analysis.png'")

    # Show plot
    plt.show()


def main():
    """Main analysis workflow."""
    print("=" * 70)
    print("LEE & SCHMIDT (1993) FLEXIBLE TIME-VARYING INEFFICIENCY MODEL")
    print("Application: Manufacturing with Non-Monotonic Efficiency Pattern")
    print("=" * 70)

    # 1. Generate data
    df, xi_t_true = generate_panel_data_nonmonotonic(n_firms=80, n_periods=8)

    # 2. Estimate Lee-Schmidt model
    result = estimate_lee_schmidt_model(df)

    # 3. Interpret time pattern
    interpret_time_pattern(result, xi_t_true)

    # 4. Compare with Kumbhakar
    compare_with_kumbhakar(result, df)

    # 5. Visualize results
    visualize_results(result, df, xi_t_true)

    # 6. Key takeaways
    print("\n" + "=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print(
        """
    1. MAXIMUM FLEXIBILITY: Lee & Schmidt (1993) imposes NO parametric form
       - Can capture ANY time pattern (learning, degradation, cycles, breaks)
       - ξ_t are time dummies (T-1 free parameters after normalization)
       - Only restriction: common pattern across firms

    2. DECOMPOSITION: u_it = δ_i × ξ_t
       - δ_i: Firm-specific inefficiency LEVEL (time-invariant)
       - ξ_t: Common time-specific SCALE FACTOR
       - Efficient firms (low δ_i) vary proportionally with inefficient firms

    3. NORMALIZATION: Typically ξ_0 = 1 or ξ_T = 1
       - Necessary for identification
       - Interpretation: inefficiency relative to base period

    4. ADVANTAGES:
       - No parametric restrictions (unlike Kumbhakar's logistic)
       - Can detect structural breaks, cycles, non-monotonic patterns
       - Provides diagnostic for time pattern (plot ξ_t)

    5. DISADVANTAGES:
       - MANY parameters: K (frontier) + 2 (variances) + N (firms) + (T-1) (time)
       - Requires large N and moderate T
       - Less parsimonious than Kumbhakar (1990)
       - May overfit with small samples

    6. WHEN TO USE:
       - Believe time pattern is complex (non-monotonic, breaks)
       - Have sufficient data (N > 30, T > 5)
       - Want to discover time pattern empirically
       - Willing to trade parsimony for flexibility

    7. ALTERNATIVES:
       - Pitt-Lee (1981): Time-invariant (most parsimonious)
       - Battese-Coelli (1992): Exponential decay (monotonic learning)
       - Kumbhakar (1990): Logistic pattern (flexible but parametric)
       - CSS (1990): Firm-specific time patterns (MOST flexible, MANY params!)

    8. MODEL SELECTION:
       - Use AIC/BIC to compare with Kumbhakar
       - Plot ξ_t to diagnose time pattern
       - If ξ_t is approximately flat → Use Pitt-Lee
       - If ξ_t is monotonic → Consider Kumbhakar or BC92
       - If ξ_t is complex (U-shape, breaks) → Lee-Schmidt is appropriate!
    """
    )


if __name__ == "__main__":
    main()
