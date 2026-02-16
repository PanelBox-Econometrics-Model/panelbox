"""
Example: Normal-Gamma Stochastic Frontier Model

This script demonstrates the use of the gamma distribution for inefficiency
in stochastic frontier analysis. The gamma distribution is more flexible than
exponential and half-normal distributions.

Key features:
- Flexible shape parameter P allows various distribution shapes
- When P=1, reduces to exponential distribution
- Can capture different degrees of skewness
- Uses Simulated Maximum Likelihood (SML) for estimation

Author: PanelBox Development Team
Date: 2024-02-15
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# Example 1: Basic Gamma Model
# ============================================================================

print("=" * 80)
print("Example 1: Basic Gamma Stochastic Frontier Model")
print("=" * 80)

# Generate synthetic data
n = 200
P_true = 2.5  # Shape parameter
theta_true = 2.0  # Rate parameter
sigma_v_true = 0.3

# Inputs: labor and capital
X = np.column_stack(
    [
        np.ones(n),
        np.random.normal(5, 0.5, n),  # log(labor)
        np.random.normal(10, 1.0, n),  # log(capital)
    ]
)

beta_true = np.array([2.0, 0.6, 0.4])  # Production function coefficients

# Generate inefficiency and noise
u = np.random.gamma(P_true, 1 / theta_true, n)
v = np.random.normal(0, sigma_v_true, n)

# Generate output (production frontier)
y = X @ beta_true + v - u

# Create DataFrame
df = pd.DataFrame(
    {
        "log_output": y,
        "log_labor": X[:, 1],
        "log_capital": X[:, 2],
    }
)

print(f"\nData generated:")
print(f"  Sample size: {n}")
print(f"  True P (shape): {P_true}")
print(f"  True θ (rate): {theta_true}")
print(f"  True E[u] = P/θ: {P_true/theta_true:.4f}")
print(f"  True σ_v: {sigma_v_true}")

# Estimate gamma model
print("\nEstimating gamma model...")
model_gamma = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="gamma",
)

result_gamma = model_gamma.fit(maxiter=200, verbose=False)

print(f"\nEstimation Results:")
print(f"  Converged: {result_gamma.converged}")
print(f"  Log-likelihood: {result_gamma.loglik:.4f}")
print(f"\nEstimated Parameters:")
print(f"  P (shape): {result_gamma.gamma_P:.4f} (true: {P_true})")
print(f"  θ (rate): {result_gamma.gamma_theta:.4f} (true: {theta_true})")
print(
    f"  E[u] = P/θ: {result_gamma.gamma_P/result_gamma.gamma_theta:.4f} (true: {P_true/theta_true:.4f})"
)
print(f"  σ_v: {result_gamma.sigma_v:.4f} (true: {sigma_v_true})")

# Display full summary
print("\n" + result_gamma.summary())

# ============================================================================
# Example 2: Compare Gamma with Half-Normal and Exponential
# ============================================================================

print("\n" + "=" * 80)
print("Example 2: Comparing Gamma with Other Distributions")
print("=" * 80)

# Estimate half-normal model
print("\nEstimating half-normal model...")
model_hn = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="half_normal",
)
result_hn = model_hn.fit(maxiter=200, verbose=False)

# Estimate exponential model
print("Estimating exponential model...")
model_exp = StochasticFrontier(
    data=df,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="exponential",
)
result_exp = model_exp.fit(maxiter=200, verbose=False)

# Compare log-likelihoods
print("\nModel Comparison:")
print(f"{'Model':<20} {'Log-Likelihood':<18} {'AIC':<12} {'BIC':<12}")
print("-" * 65)
print(
    f"{'Gamma':<20} {result_gamma.loglik:<18.4f} {result_gamma.aic:<12.2f} {result_gamma.bic:<12.2f}"
)
print(
    f"{'Half-Normal':<20} {result_hn.loglik:<18.4f} {result_hn.aic:<12.2f} {result_hn.bic:<12.2f}"
)
print(
    f"{'Exponential':<20} {result_exp.loglik:<18.4f} {result_exp.aic:<12.2f} {result_exp.bic:<12.2f}"
)

# Likelihood ratio test: Gamma vs Half-Normal
lr_stat_hn = 2 * (result_gamma.loglik - result_hn.loglik)
print(f"\nLR Test: Gamma vs Half-Normal")
print(f"  LR statistic: {lr_stat_hn:.4f}")
print(f"  (Gamma has 2 extra params: P and θ vs σ_u)")

# Likelihood ratio test: Gamma vs Exponential
lr_stat_exp = 2 * (result_gamma.loglik - result_exp.loglik)
print(f"\nLR Test: Gamma vs Exponential")
print(f"  LR statistic: {lr_stat_exp:.4f}")
print(f"  (Gamma has 1 extra param: P)")

# ============================================================================
# Example 3: Efficiency Estimation
# ============================================================================

print("\n" + "=" * 80)
print("Example 3: Efficiency Estimation")
print("=" * 80)

# Compute efficiency using BC estimator
print("\nComputing efficiency scores using BC estimator...")
from panelbox.frontier.efficiency import estimate_efficiency

eff_df = estimate_efficiency(result_gamma, estimator="bc", ci_level=0.95)

print(f"\nEfficiency Statistics:")
print(f"  Mean efficiency: {eff_df['efficiency'].mean():.4f}")
print(f"  Median efficiency: {eff_df['efficiency'].median():.4f}")
print(f"  Min efficiency: {eff_df['efficiency'].min():.4f}")
print(f"  Max efficiency: {eff_df['efficiency'].max():.4f}")
print(f"  Std. dev.: {eff_df['efficiency'].std():.4f}")

# Display top 10 most efficient firms
print("\nTop 10 Most Efficient Observations:")
top_10 = eff_df.nlargest(10, "efficiency")
print(top_10[["efficiency", "inefficiency"]].to_string())

# Display bottom 10 least efficient firms
print("\nBottom 10 Least Efficient Observations:")
bottom_10 = eff_df.nsmallest(10, "efficiency")
print(bottom_10[["efficiency", "inefficiency"]].to_string())

# ============================================================================
# Example 4: Visualizations
# ============================================================================

print("\n" + "=" * 80)
print("Example 4: Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Efficiency Distribution
ax1 = axes[0, 0]
ax1.hist(eff_df["efficiency"], bins=30, alpha=0.7, edgecolor="black", color="skyblue")
ax1.axvline(eff_df["efficiency"].mean(), color="red", linestyle="--", linewidth=2, label="Mean")
ax1.axvline(
    eff_df["efficiency"].median(), color="green", linestyle="--", linewidth=2, label="Median"
)
ax1.set_xlabel("Efficiency Score", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title("Distribution of Technical Efficiency\n(Gamma Model)", fontsize=13, fontweight="bold")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Theoretical Gamma Distribution
ax2 = axes[0, 1]
u_range = np.linspace(0, 3, 1000)
from scipy.stats import gamma as gamma_dist

gamma_pdf = gamma_dist.pdf(u_range, a=result_gamma.gamma_P, scale=1 / result_gamma.gamma_theta)
ax2.plot(
    u_range,
    gamma_pdf,
    linewidth=2,
    color="darkblue",
    label=f"Gamma(P={result_gamma.gamma_P:.2f}, θ={result_gamma.gamma_theta:.2f})",
)
ax2.hist(u, bins=30, density=True, alpha=0.4, color="orange", label="True u (simulated)")
ax2.set_xlabel("Inefficiency (u)", fontsize=12)
ax2.set_ylabel("Density", fontsize=12)
ax2.set_title(
    "Theoretical Gamma Distribution\nvs True Inefficiency", fontsize=13, fontweight="bold"
)
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Residuals vs Fitted Values
ax3 = axes[1, 0]
fitted = result_gamma.fitted_values
residuals = result_gamma.residuals
ax3.scatter(fitted, residuals, alpha=0.5, s=30, color="purple")
ax3.axhline(0, color="black", linestyle="--", linewidth=1.5)
ax3.set_xlabel("Fitted Values", fontsize=12)
ax3.set_ylabel("Residuals", fontsize=12)
ax3.set_title("Residuals vs Fitted Values", fontsize=13, fontweight="bold")
ax3.grid(alpha=0.3)

# Plot 4: Efficiency vs Inefficiency
ax4 = axes[1, 1]
ax4.scatter(eff_df["inefficiency"], eff_df["efficiency"], alpha=0.5, s=30, color="teal")
ax4.set_xlabel("Inefficiency (u)", fontsize=12)
ax4.set_ylabel("Efficiency (exp(-u))", fontsize=12)
ax4.set_title("Efficiency vs Inefficiency", fontsize=13, fontweight="bold")
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("gamma_frontier_analysis.png", dpi=300, bbox_inches="tight")
print("\nPlots saved as 'gamma_frontier_analysis.png'")

# ============================================================================
# Example 5: Interpretation of Gamma Parameters
# ============================================================================

print("\n" + "=" * 80)
print("Example 5: Interpretation of Gamma Parameters")
print("=" * 80)

print(
    f"""
The Gamma distribution is characterized by two parameters:

1. **Shape Parameter P = {result_gamma.gamma_P:.4f}**
   - Controls the shape of the distribution
   - P < 1: highly skewed, mode at 0
   - P = 1: exponential distribution
   - P > 1: unimodal distribution
   - P → ∞: approaches normal distribution

2. **Rate Parameter θ = {result_gamma.gamma_theta:.4f}**
   - Controls the scale (rate) of the distribution
   - Higher θ → smaller inefficiency on average
   - Lower θ → larger inefficiency on average

Expected Inefficiency: E[u] = P/θ = {result_gamma.gamma_P/result_gamma.gamma_theta:.4f}
Variance of Inefficiency: Var[u] = P/θ² = {result_gamma.gamma_P/(result_gamma.gamma_theta**2):.4f}

Interpretation for this model:
- The estimated P = {result_gamma.gamma_P:.2f} suggests {'a fairly skewed' if result_gamma.gamma_P < 2 else 'a moderately symmetric'}
  inefficiency distribution.
- The average inefficiency is {result_gamma.gamma_P/result_gamma.gamma_theta:.2f}, meaning firms
  operate on average at {np.exp(-(result_gamma.gamma_P/result_gamma.gamma_theta))*100:.1f}% of the frontier.
"""
)

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
print(
    f"""
Summary of Results:
- The gamma distribution provides a flexible model for inefficiency
- Estimated parameters recover true values reasonably well
- The model {'performs better' if result_gamma.aic < result_hn.aic and result_gamma.aic < result_exp.aic else 'has comparable fit'}
  compared to half-normal and exponential
- Efficiency scores range from {eff_df['efficiency'].min():.3f} to {eff_df['efficiency'].max():.3f}
- Mean efficiency: {eff_df['efficiency'].mean():.3f}

When to use Gamma distribution:
✓ When inefficiency distribution may not be half-normal or exponential
✓ When you want flexibility in the shape of inefficiency
✓ When you have sufficient data (gamma requires more observations)
✓ When computational cost is acceptable (SML is slower)
"""
)
