"""
Example: Normal-Gamma Stochastic Frontier Model

This example demonstrates the use of the gamma distribution for modeling
inefficiency in stochastic frontier analysis. The gamma distribution is more
flexible than half-normal or exponential distributions.

The model is:
    y_i = x_i'β + v_i - u_i    (production frontier)

where:
    v_i ~ N(0, σ²_v)           (statistical noise)
    u_i ~ Gamma(P, θ)          (technical inefficiency)

Gamma distribution properties:
    - Shape parameter P > 0 controls the distribution shape
    - Rate parameter θ > 0 controls the scale
    - E[u] = P/θ
    - Var[u] = P/θ²
    - When P=1, reduces to exponential distribution

References:
    Greene, W. H. (1990). A gamma-distributed stochastic frontier model.
        Journal of Econometrics, 46(1-2), 141-163.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.efficiency import estimate_efficiency

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Generate Synthetic Data
# ============================================================================

print("=" * 70)
print("EXAMPLE: Normal-Gamma Stochastic Frontier Model")
print("=" * 70)

# Sample size
n = 200

# True parameters
beta_true = np.array([2.0, 0.8, -0.5])
sigma_v_true = 0.2
P_true = 2.5  # Shape parameter
theta_true = 2.0  # Rate parameter

# Generate covariates
X = np.column_stack(
    [
        np.ones(n),  # Intercept
        np.random.uniform(0, 10, n),  # Capital
        np.random.uniform(0, 10, n),  # Labor
    ]
)

# Generate inefficiency from gamma distribution
u = np.random.gamma(P_true, 1 / theta_true, n)

# Generate noise
v = np.random.normal(0, sigma_v_true, n)

# Generate output (production frontier)
y = X @ beta_true + v - u

# Create DataFrame
data = pd.DataFrame(
    {
        "output": y,
        "capital": X[:, 1],
        "labor": X[:, 2],
    }
)

print(f"\nData generated:")
print(f"  Sample size: {n}")
print(f"  True β: {beta_true}")
print(f"  True σ_v: {sigma_v_true:.3f}")
print(f"  True P (shape): {P_true:.3f}")
print(f"  True θ (rate): {theta_true:.3f}")
print(f"  True E[u] = P/θ: {P_true/theta_true:.3f}")
print(f"  True Var[u] = P/θ²: {P_true/(theta_true**2):.3f}")

# ============================================================================
# 2. Estimate Gamma Model
# ============================================================================

print(f"\n{'-'*70}")
print("Estimating Normal-Gamma Stochastic Frontier Model...")
print(f"{'-'*70}")

model = StochasticFrontier(
    data=data,
    depvar="output",
    exog=["capital", "labor"],
    frontier="production",
    dist="gamma",
)

# Fit model (may take a while due to SML)
result = model.fit(maxiter=200, verbose=False)

# Display results
print("\n" + result.summary())

# ============================================================================
# 3. Compare Parameters
# ============================================================================

print(f"\n{'-'*70}")
print("Parameter Recovery:")
print(f"{'-'*70}")

beta_est = result.params[:3].values
P_est = result.gamma_P
theta_est = result.gamma_theta

print(f"\n{'Parameter':<20} {'True':<12} {'Estimated':<12} {'Error':<12}")
print(f"{'-'*60}")
print(
    f"{'β_0 (Intercept)':<20} {beta_true[0]:>11.3f} {beta_est[0]:>11.3f} {abs(beta_est[0]-beta_true[0]):>11.3f}"
)
print(
    f"{'β_1 (Capital)':<20} {beta_true[1]:>11.3f} {beta_est[1]:>11.3f} {abs(beta_est[1]-beta_true[1]):>11.3f}"
)
print(
    f"{'β_2 (Labor)':<20} {beta_true[2]:>11.3f} {beta_est[2]:>11.3f} {abs(beta_est[2]-beta_true[2]):>11.3f}"
)
print(
    f"{'σ_v':<20} {sigma_v_true:>11.3f} {result.sigma_v:>11.3f} {abs(result.sigma_v-sigma_v_true):>11.3f}"
)
print(f"{'P (shape)':<20} {P_true:>11.3f} {P_est:>11.3f} {abs(P_est-P_true):>11.3f}")
print(f"{'θ (rate)':<20} {theta_true:>11.3f} {theta_est:>11.3f} {abs(theta_est-theta_true):>11.3f}")
print(
    f"{'E[u] = P/θ':<20} {P_true/theta_true:>11.3f} {P_est/theta_est:>11.3f} {abs(P_est/theta_est-P_true/theta_true):>11.3f}"
)

# ============================================================================
# 4. Estimate Technical Efficiency
# ============================================================================

print(f"\n{'-'*70}")
print("Estimating Technical Efficiency...")
print(f"{'-'*70}")

# Battese-Coelli estimator
eff_bc = estimate_efficiency(result, estimator="bc")

print(f"\nEfficiency Statistics (BC estimator):")
print(f"  Mean efficiency: {eff_bc['efficiency'].mean():.4f}")
print(f"  Std efficiency: {eff_bc['efficiency'].std():.4f}")
print(f"  Min efficiency: {eff_bc['efficiency'].min():.4f}")
print(f"  Max efficiency: {eff_bc['efficiency'].max():.4f}")

# True efficiency
true_eff = np.exp(-u)
print(f"\nTrue Efficiency (for comparison):")
print(f"  Mean: {true_eff.mean():.4f}")
print(f"  Std: {true_eff.std():.4f}")

# ============================================================================
# 5. Visualization: Gamma Distribution
# ============================================================================

print(f"\n{'-'*70}")
print("Creating visualization...")
print(f"{'-'*70}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Estimated vs True Gamma PDF
ax1 = axes[0, 0]
u_grid = np.linspace(0, 5, 200)
pdf_true = gamma_dist.pdf(u_grid, a=P_true, scale=1 / theta_true)
pdf_est = gamma_dist.pdf(u_grid, a=P_est, scale=1 / theta_est)

ax1.plot(u_grid, pdf_true, "b-", linewidth=2, label=f"True: P={P_true:.2f}, θ={theta_true:.2f}")
ax1.plot(u_grid, pdf_est, "r--", linewidth=2, label=f"Est: P={P_est:.2f}, θ={theta_est:.2f}")
ax1.hist(u, bins=30, density=True, alpha=0.3, color="gray", label="Observed u")
ax1.set_xlabel("Inefficiency (u)")
ax1.set_ylabel("Density")
ax1.set_title("Gamma Distribution: True vs Estimated")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Estimated vs True Efficiency
ax2 = axes[0, 1]
ax2.scatter(true_eff, eff_bc["efficiency"], alpha=0.5, s=20)
ax2.plot([0, 1], [0, 1], "r--", linewidth=1, label="45° line")
ax2.set_xlabel("True Efficiency")
ax2.set_ylabel("Estimated Efficiency (BC)")
ax2.set_title("Efficiency: Estimated vs True")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

# Plot 3: Residuals histogram
ax3 = axes[1, 0]
residuals = result.residuals
ax3.hist(residuals, bins=30, density=True, alpha=0.7, edgecolor="black")
ax3.set_xlabel("Residuals (ε = v - u)")
ax3.set_ylabel("Density")
ax3.set_title("Residuals Distribution")
ax3.grid(True, alpha=0.3)

# Plot 4: Efficiency distribution
ax4 = axes[1, 1]
ax4.hist(eff_bc["efficiency"], bins=30, alpha=0.7, edgecolor="black", label="Estimated")
ax4.hist(true_eff, bins=30, alpha=0.5, edgecolor="black", color="red", label="True")
ax4.set_xlabel("Technical Efficiency")
ax4.set_ylabel("Frequency")
ax4.set_title("Efficiency Distribution")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("gamma_frontier_example.png", dpi=150)
print(f"\nPlot saved as: gamma_frontier_example.png")

# ============================================================================
# 6. Model Comparison
# ============================================================================

print(f"\n{'-'*70}")
print("Model Comparison: Gamma vs Half-Normal vs Exponential")
print(f"{'-'*70}")

# Estimate half-normal model
model_hn = StochasticFrontier(
    data=data,
    depvar="output",
    exog=["capital", "labor"],
    frontier="production",
    dist="half_normal",
)
result_hn = model_hn.fit(verbose=False)

# Estimate exponential model
model_exp = StochasticFrontier(
    data=data,
    depvar="output",
    exog=["capital", "labor"],
    frontier="production",
    dist="exponential",
)
result_exp = model_exp.fit(verbose=False)

print(f"\n{'Model':<20} {'Log-Lik':<12} {'AIC':<12} {'BIC':<12}")
print(f"{'-'*60}")
print(f"{'Gamma':<20} {result.loglik:>11.2f} {result.aic:>11.2f} {result.bic:>11.2f}")
print(
    f"{'Half-Normal':<20} {result_hn.loglik:>11.2f} {result_hn.aic:>11.2f} {result_hn.bic:>11.2f}"
)
print(
    f"{'Exponential':<20} {result_exp.loglik:>11.2f} {result_exp.aic:>11.2f} {result_exp.bic:>11.2f}"
)

# Best model by AIC
best_model = min(
    [("Gamma", result.aic), ("Half-Normal", result_hn.aic), ("Exponential", result_exp.aic)],
    key=lambda x: x[1],
)

print(f"\nBest model by AIC: {best_model[0]}")

print(f"\n{'='*70}")
print("Analysis Complete!")
print(f"{'='*70}")
