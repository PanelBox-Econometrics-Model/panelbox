"""
Example: Cornwell, Schmidt & Sickles (1990) Distribution-Free Model

This example demonstrates the use of the CSS distribution-free model
for estimating technical efficiency without distributional assumptions.

The CSS model is robust to misspecification of the inefficiency distribution
and allows for time-varying efficiency through flexible time trends.

References:
    Cornwell, C., Schmidt, P., & Sickles, R. C. (1990).
        Production frontiers with cross-sectional and time-series variation
        in efficiency levels. Journal of Econometrics, 46(1-2), 185-200.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier


# Generate synthetic panel data
def generate_panel_data(N=50, T=15, seed=42):
    """Generate panel data with time-varying inefficiency."""
    np.random.seed(seed)

    data = []
    for i in range(N):
        # Firm-specific productivity trend
        alpha_base = 5 + np.random.normal(0, 0.3)
        trend_linear = np.random.normal(0.1, 0.03)
        trend_quad = np.random.normal(-0.008, 0.002)

        for t in range(T):
            # Time-varying productivity (inverse of inefficiency)
            alpha_it = alpha_base + trend_linear * t + trend_quad * (t**2)

            # Production inputs
            ln_labor = np.random.normal(2.5, 0.5)
            ln_capital = np.random.normal(3.0, 0.6)

            # Output
            ln_output = (
                alpha_it + 0.6 * ln_labor + 0.4 * ln_capital + np.random.normal(0, 0.15)  # noise
            )

            data.append(
                {
                    "firm_id": i,
                    "year": t,
                    "ln_output": ln_output,
                    "ln_labor": ln_labor,
                    "ln_capital": ln_capital,
                }
            )

    return pd.DataFrame(data)


# Generate data
df = generate_panel_data(N=50, T=15)

print("=" * 70)
print("CSS Distribution-Free Model Example")
print("=" * 70)
print(
    f"\nData: {len(df)} observations, {df['firm_id'].nunique()} firms, {df['year'].nunique()} years"
)
print(f"Balanced panel: {df.groupby('firm_id').size().nunique() == 1}")

# ============================================================================
# Model 1: CSS with Quadratic Time Trend (Recommended)
# ============================================================================
print("\n" + "=" * 70)
print("Model 1: CSS with Quadratic Time Trend")
print("=" * 70)

css_quad = StochasticFrontier(
    data=df,
    depvar="ln_output",
    exog=["ln_labor", "ln_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    model_type="css",
    css_time_trend="quadratic",  # θ_i1 + θ_i2·t + θ_i3·t²
)

result_quad = css_quad.fit()

print("\nParameter Estimates:")
print(result_quad.params)

print(f"\nModel Fit:")
print(f"  R² = {result_quad._r_squared:.4f}")
print(f"  σ_v (noise) = {result_quad._css_result.sigma_v:.4f}")
print(f"  Number of parameters: {result_quad.nparams}")

print(f"\nEfficiency Summary:")
eff_quad = result_quad._efficiency_it
print(f"  Mean efficiency: {np.mean(eff_quad):.4f}")
print(f"  Min efficiency:  {np.min(eff_quad):.4f}")
print(f"  Max efficiency:  {np.max(eff_quad):.4f}")
print(f"  Std efficiency:  {np.std(eff_quad):.4f}")

# ============================================================================
# Model 2: CSS with Linear Time Trend
# ============================================================================
print("\n" + "=" * 70)
print("Model 2: CSS with Linear Time Trend")
print("=" * 70)

css_linear = StochasticFrontier(
    data=df,
    depvar="ln_output",
    exog=["ln_labor", "ln_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    model_type="css",
    css_time_trend="linear",  # θ_i1 + θ_i2·t
)

result_linear = css_linear.fit()

print("\nParameter Estimates:")
print(result_linear.params)

print(f"\nModel Fit:")
print(f"  R² = {result_linear._r_squared:.4f}")
print(f"  Mean efficiency: {np.mean(result_linear._efficiency_it):.4f}")

# ============================================================================
# Model 3: CSS with No Time Trend (Pure Fixed Effects)
# ============================================================================
print("\n" + "=" * 70)
print("Model 3: CSS with No Time Trend (Pure FE)")
print("=" * 70)

css_none = StochasticFrontier(
    data=df,
    depvar="ln_output",
    exog=["ln_labor", "ln_capital"],
    entity="firm_id",
    time="year",
    frontier="production",
    model_type="css",
    css_time_trend="none",  # θ_i (time-invariant)
)

result_none = css_none.fit()

print("\nParameter Estimates:")
print(result_none.params)

print(f"\nModel Fit:")
print(f"  R² = {result_none._r_squared:.4f}")
print(f"  Mean efficiency: {np.mean(result_none._efficiency_it):.4f}")

# ============================================================================
# Compare Models
# ============================================================================
print("\n" + "=" * 70)
print("Model Comparison")
print("=" * 70)

comparison = pd.DataFrame(
    {
        "Model": ["Quadratic", "Linear", "None (FE)"],
        "R²": [
            result_quad._r_squared,
            result_linear._r_squared,
            result_none._r_squared,
        ],
        "Mean Efficiency": [
            np.mean(result_quad._efficiency_it),
            np.mean(result_linear._efficiency_it),
            np.mean(result_none._efficiency_it),
        ],
        "Std Efficiency": [
            np.std(result_quad._efficiency_it),
            np.std(result_linear._efficiency_it),
            np.std(result_none._efficiency_it),
        ],
    }
)

print("\n", comparison.to_string(index=False))
print("\nRecommendation: Quadratic time trend has highest R²")

# ============================================================================
# Efficiency Analysis
# ============================================================================
print("\n" + "=" * 70)
print("Efficiency Analysis (Quadratic Model)")
print("=" * 70)

# Efficiency by entity
eff_by_entity = result_quad._css_result.efficiency_by_entity()

print("\nTop 5 Most Efficient Firms (average over time):")
print(
    eff_by_entity.nlargest(5, "mean_efficiency")[
        ["entity", "mean_efficiency", "std_efficiency", "trend"]
    ].to_string(index=False)
)

print("\nBottom 5 Least Efficient Firms (average over time):")
print(
    eff_by_entity.nsmallest(5, "mean_efficiency")[
        ["entity", "mean_efficiency", "std_efficiency", "trend"]
    ].to_string(index=False)
)

# Efficiency by period
eff_by_period = result_quad._css_result.efficiency_by_period()

print("\nEfficiency Over Time:")
print(eff_by_period[["period", "mean_efficiency", "std_efficiency"]].to_string(index=False))

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "=" * 70)
print("Creating Visualizations...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficiency distribution
ax1 = axes[0, 0]
ax1.hist(eff_quad.flatten(), bins=30, edgecolor="black", alpha=0.7)
ax1.set_xlabel("Technical Efficiency")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Technical Efficiency (CSS Quadratic)")
ax1.axvline(np.mean(eff_quad), color="red", linestyle="--", label=f"Mean = {np.mean(eff_quad):.3f}")
ax1.legend()

# Plot 2: Efficiency evolution over time
ax2 = axes[0, 1]
for i in range(min(10, len(eff_quad))):  # Plot first 10 firms
    ax2.plot(range(result_quad._css_result.n_periods), eff_quad[i, :], alpha=0.5)
ax2.plot(
    eff_by_period["period"],
    eff_by_period["mean_efficiency"],
    color="red",
    linewidth=2,
    label="Average",
)
ax2.set_xlabel("Time Period")
ax2.set_ylabel("Technical Efficiency")
ax2.set_title("Efficiency Evolution Over Time (Sample of Firms)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Mean efficiency by entity
ax3 = axes[1, 0]
ax3.bar(range(len(eff_by_entity)), eff_by_entity["mean_efficiency"].sort_values())
ax3.set_xlabel("Firm (sorted by efficiency)")
ax3.set_ylabel("Mean Technical Efficiency")
ax3.set_title("Mean Efficiency by Firm")
ax3.grid(True, alpha=0.3, axis="y")

# Plot 4: Efficiency trend comparison
ax4 = axes[1, 1]
periods = range(len(eff_by_period))
ax4.plot(periods, eff_by_period["mean_efficiency"], marker="o", label="Quadratic", linewidth=2)

# Add comparison models
eff_linear_by_period = result_linear._css_result.efficiency_by_period()
eff_none_by_period = result_none._css_result.efficiency_by_period()

ax4.plot(periods, eff_linear_by_period["mean_efficiency"], marker="s", label="Linear", linewidth=2)
ax4.plot(periods, eff_none_by_period["mean_efficiency"], marker="^", label="None (FE)", linewidth=2)

ax4.set_xlabel("Time Period")
ax4.set_ylabel("Mean Technical Efficiency")
ax4.set_title("Model Comparison: Mean Efficiency Over Time")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("css_efficiency_analysis.png", dpi=150, bbox_inches="tight")
print("Saved: css_efficiency_analysis.png")

plt.show()

# ============================================================================
# Key Insights
# ============================================================================
print("\n" + "=" * 70)
print("Key Insights")
print("=" * 70)

print(
    """
1. CSS Model Advantages:
   - No distributional assumptions about inefficiency
   - Allows for flexible time-varying efficiency
   - Robust to misspecification

2. Time Trend Specification:
   - Quadratic: Most flexible, allows U-shaped or inverted-U patterns
   - Linear: Monotonic trends only
   - None: Time-invariant efficiency (pure FE)

3. Recommendations:
   - Use quadratic for datasets with T ≥ 10
   - Use linear for smaller T or when theory suggests monotonic trends
   - Compare models using R² or F-tests

4. Interpretation:
   - Efficiency ranges from 0 to 1
   - Entity with highest α_it in period t has efficiency = 1 (frontier)
   - Other entities: efficiency = exp(α_it - max(α_jt))

5. Requirements:
   - Balanced or unbalanced panel data
   - T ≥ 5 (minimum), T ≥ 10 (recommended)
   - No missing values in key variables
"""
)

print("=" * 70)
print("Example completed successfully!")
print("=" * 70)
