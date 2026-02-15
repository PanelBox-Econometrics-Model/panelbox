"""
Example: True Models with BC95 Inefficiency Determinants

This example demonstrates the most comprehensive stochastic frontier models,
combining Greene's (2005) "True" models with Battese & Coelli (1995)
inefficiency determinants.

Key Features:
1. True Fixed Effects (TFE) + BC95: α_i + Z affecting u_it
2. True Random Effects (TRE) + BC95: w_i + Z affecting u_it
3. Comparison with simple True models (without Z)
4. Interpretation of three-component decomposition

References:
    Greene, W. H. (2005).
        Reconsidering heterogeneity in panel data estimators of the stochastic
        frontier model. Journal of Econometrics, 126(2), 269-303.

    Battese, G. E., & Coelli, T. J. (1995).
        A model for technical inefficiency effects in a stochastic frontier
        production function for panel data. Empirical Economics, 20(2), 325-332.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panelbox.frontier.tests import lr_test
from panelbox.frontier.true_models import (
    loglik_tfe_bc95,
    loglik_tre_bc95,
    loglik_true_fixed_effects,
    loglik_true_random_effects,
    variance_decomposition_tre,
)

# ============================================================================
# 1. Generate Data with Heterogeneity AND Inefficiency Determinants
# ============================================================================


def generate_panel_data_bc95(N=80, T=8, seed=42):
    """Generate panel data with heterogeneity, inefficiency determinants.

    Model: y_it = α_i + β_0 + β_1*labor + β_2*capital + v_it - u_it

    where:
        α_i ~ N(0, 0.5²) is firm heterogeneity
        u_it ~ N⁺(μ_it, 0.3²) with μ_it = δ_0 + δ_1*size + δ_2*age
        v_it ~ N(0, 0.2²) is noise

    Interpretation:
        - α_i: Permanent technology/management differences (heterogeneity)
        - μ_it: Systematic factors affecting inefficiency (determinants)
        - u_it - μ_it: Random inefficiency component
    """
    np.random.seed(seed)

    # True parameters
    beta_true = np.array([5.0, 0.6, 0.4])  # Frontier
    delta_true = np.array([0.5, -0.3, 0.2])  # Inefficiency effects
    sigma_v = 0.2
    sigma_u = 0.3
    sigma_alpha = 0.5  # Heterogeneity

    # Panel structure
    entity_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    # Frontier variables
    log_labor = np.random.randn(N * T) * 0.5 + 2.0
    log_capital = np.random.randn(N * T) * 0.5 + 2.5
    X = np.column_stack([np.ones(N * T), log_labor, log_capital])

    # Inefficiency determinants (firm characteristics)
    firm_size = np.random.randn(N) * 0.8  # Standardized size
    firm_age = np.random.randn(N) * 1.0  # Standardized age

    # Time-varying determinants
    size_it = firm_size[entity_id] + np.random.randn(N * T) * 0.2
    age_it = firm_age[entity_id] + time_id / T  # Age increases over time

    Z = np.column_stack([np.ones(N * T), size_it, age_it])

    # Firm heterogeneity
    alpha_i = np.random.randn(N) * sigma_alpha
    alpha = alpha_i[entity_id]

    # Inefficiency with determinants
    mu_it = Z @ delta_true
    u = np.abs(np.random.randn(N * T)) * sigma_u + mu_it

    # Noise
    v = np.random.randn(N * T) * sigma_v

    # Output
    log_output = X @ beta_true + alpha - u + v

    # Create DataFrame
    data = pd.DataFrame(
        {
            "firm": entity_id,
            "year": time_id,
            "log_output": log_output,
            "log_labor": log_labor,
            "log_capital": log_capital,
            "firm_size": size_it,
            "firm_age": age_it,
        }
    )

    return {
        "data": data,
        "true_params": {
            "beta": beta_true,
            "delta": delta_true,
            "sigma_v": sigma_v,
            "sigma_u": sigma_u,
            "sigma_alpha": sigma_alpha,
            "alpha_i": alpha_i,
        },
        "arrays": {"y": log_output, "X": X, "Z": Z, "entity_id": entity_id, "time_id": time_id},
    }


print("=" * 70)
print("TRUE MODELS WITH BC95 INEFFICIENCY DETERMINANTS")
print("Combining Heterogeneity Separation with Inefficiency Modeling")
print("=" * 70)

# Generate data
print("\n1. Generating simulated panel data...")
dgp = generate_panel_data_bc95(N=80, T=8)
data = dgp["data"]
arrays = dgp["arrays"]

print(f"   - N = {data['firm'].nunique()} firms")
print(f"   - T = {data['year'].nunique()} years")
print(f"   - Total observations: {len(data)}")
print(f"\n   True Frontier Parameters (β):")
print(f"     β_0 (Intercept): {dgp['true_params']['beta'][0]:.3f}")
print(f"     β_1 (Labor):     {dgp['true_params']['beta'][1]:.3f}")
print(f"     β_2 (Capital):   {dgp['true_params']['beta'][2]:.3f}")
print(f"\n   True Inefficiency Determinants (δ):")
print(f"     δ_0 (Constant):  {dgp['true_params']['delta'][0]:.3f}")
print(f"     δ_1 (Size):      {dgp['true_params']['delta'][1]:.3f}")
print(f"     δ_2 (Age):       {dgp['true_params']['delta'][2]:.3f}")


# ============================================================================
# 2. Estimate TRE Model (Without BC95 - Baseline)
# ============================================================================

print("\n2. Estimating baseline TRE model (without Z)...")


def estimate_tre_simple(y, X, entity_id, time_id, verbose=True):
    """Estimate simple TRE model (no inefficiency determinants)."""
    k = X.shape[1]

    # Starting values
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    sigma_ols = np.std(resid_ols)

    theta_start = np.concatenate(
        [
            beta_ols,
            [np.log(sigma_ols**2 / 3)],
            [np.log(sigma_ols**2 / 3)],
            [np.log(sigma_ols**2 / 3)],
        ]
    )

    def neg_loglik(theta):
        ll = loglik_true_random_effects(
            theta, y, X, entity_id, time_id, sign=1, n_quadrature=20, method="gauss-hermite"
        )
        return -ll

    result = minimize(
        neg_loglik, theta_start, method="L-BFGS-B", bounds=[(None, None)] * k + [(-13.8, 13.8)] * 3
    )

    if verbose:
        print(f"   - Converged: {result.success}")
        print(f"   - Log-likelihood: {-result.fun:.2f}")

    return {
        "beta": result.x[:k],
        "sigma_v_sq": np.exp(result.x[k]),
        "sigma_u_sq": np.exp(result.x[k + 1]),
        "sigma_w_sq": np.exp(result.x[k + 2]),
        "loglik": -result.fun,
        "theta": result.x,
    }


tre_simple = estimate_tre_simple(arrays["y"], arrays["X"], arrays["entity_id"], arrays["time_id"])

print(f"\n   Simple TRE Estimates:")
print(f"   β: {tre_simple['beta']}")
print(f"   σ²_v: {tre_simple['sigma_v_sq']:.4f}")
print(f"   σ²_u: {tre_simple['sigma_u_sq']:.4f}")
print(f"   σ²_w: {tre_simple['sigma_w_sq']:.4f}")


# ============================================================================
# 3. Estimate TRE + BC95 Model
# ============================================================================

print("\n3. Estimating TRE + BC95 model (with inefficiency determinants)...")


def estimate_tre_bc95(y, X, Z, entity_id, time_id, verbose=True):
    """Estimate TRE model with BC95 inefficiency effects."""
    k = X.shape[1]
    m = Z.shape[1]

    # Starting values
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    sigma_ols = np.std(resid_ols)

    theta_start = np.concatenate(
        [
            beta_ols,
            [np.log(sigma_ols**2 / 3)],
            [np.log(sigma_ols**2 / 3)],
            [np.log(sigma_ols**2 / 3)],
            np.zeros(m),  # delta initial values
        ]
    )

    def neg_loglik(theta):
        ll = loglik_tre_bc95(theta, y, X, Z, entity_id, time_id, sign=1, n_quadrature=20)
        return -ll

    result = minimize(
        neg_loglik,
        theta_start,
        method="L-BFGS-B",
        bounds=[(None, None)] * k + [(-13.8, 13.8)] * 3 + [(None, None)] * m,
    )

    if verbose:
        print(f"   - Converged: {result.success}")
        print(f"   - Log-likelihood: {-result.fun:.2f}")

    return {
        "beta": result.x[:k],
        "sigma_v_sq": np.exp(result.x[k]),
        "sigma_u_sq": np.exp(result.x[k + 1]),
        "sigma_w_sq": np.exp(result.x[k + 2]),
        "delta": result.x[k + 3 :],
        "loglik": -result.fun,
        "theta": result.x,
    }


tre_bc95 = estimate_tre_bc95(
    arrays["y"], arrays["X"], arrays["Z"], arrays["entity_id"], arrays["time_id"]
)

print(f"\n   TRE + BC95 Estimates:")
print(f"\n   Frontier (β):")
print(f"     β_0: {tre_bc95['beta'][0]:.4f} (true: {dgp['true_params']['beta'][0]:.4f})")
print(f"     β_1: {tre_bc95['beta'][1]:.4f} (true: {dgp['true_params']['beta'][1]:.4f})")
print(f"     β_2: {tre_bc95['beta'][2]:.4f} (true: {dgp['true_params']['beta'][2]:.4f})")
print(f"\n   Variance Components:")
print(f"     σ²_v: {tre_bc95['sigma_v_sq']:.4f} (true: {dgp['true_params']['sigma_v']**2:.4f})")
print(f"     σ²_u: {tre_bc95['sigma_u_sq']:.4f} (true: {dgp['true_params']['sigma_u']**2:.4f})")
print(f"     σ²_w: {tre_bc95['sigma_w_sq']:.4f} (true: {dgp['true_params']['sigma_alpha']**2:.4f})")
print(f"\n   Inefficiency Determinants (δ):")
print(f"     δ_0: {tre_bc95['delta'][0]:.4f} (true: {dgp['true_params']['delta'][0]:.4f})")
print(f"     δ_1: {tre_bc95['delta'][1]:.4f} (true: {dgp['true_params']['delta'][1]:.4f})")
print(f"     δ_2: {tre_bc95['delta'][2]:.4f} (true: {dgp['true_params']['delta'][2]:.4f})")


# ============================================================================
# 4. Estimate TFE + BC95 Model
# ============================================================================

print("\n4. Estimating TFE + BC95 model...")


def estimate_tfe_bc95(y, X, Z, entity_id, time_id, verbose=True):
    """Estimate TFE model with BC95 inefficiency effects."""
    k = X.shape[1]
    m = Z.shape[1]

    # Starting values
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    sigma_ols = np.std(resid_ols)

    theta_start = np.concatenate(
        [beta_ols, [np.log(sigma_ols**2 / 2)], [np.log(sigma_ols**2 / 2)], np.zeros(m)]
    )

    def neg_loglik(theta):
        ll = loglik_tfe_bc95(theta, y, X, Z, entity_id, time_id, sign=1)
        return -ll

    result = minimize(
        neg_loglik,
        theta_start,
        method="L-BFGS-B",
        bounds=[(None, None)] * k + [(-13.8, 13.8)] * 2 + [(None, None)] * m,
    )

    if verbose:
        print(f"   - Converged: {result.success}")
        print(f"   - Log-likelihood: {-result.fun:.2f}")

    return {
        "beta": result.x[:k],
        "sigma_v_sq": np.exp(result.x[k]),
        "sigma_u_sq": np.exp(result.x[k + 1]),
        "delta": result.x[k + 2 :],
        "loglik": -result.fun,
        "theta": result.x,
    }


tfe_bc95 = estimate_tfe_bc95(
    arrays["y"], arrays["X"], arrays["Z"], arrays["entity_id"], arrays["time_id"]
)

print(f"\n   TFE + BC95 Estimates:")
print(f"\n   Frontier (β):")
print(f"     β_0: {tfe_bc95['beta'][0]:.4f} (true: {dgp['true_params']['beta'][0]:.4f})")
print(f"     β_1: {tfe_bc95['beta'][1]:.4f} (true: {dgp['true_params']['beta'][1]:.4f})")
print(f"     β_2: {tfe_bc95['beta'][2]:.4f} (true: {dgp['true_params']['beta'][2]:.4f})")
print(f"\n   Variance Components:")
print(f"     σ²_v: {tfe_bc95['sigma_v_sq']:.4f}")
print(f"     σ²_u: {tfe_bc95['sigma_u_sq']:.4f}")
print(f"\n   Inefficiency Determinants (δ):")
print(f"     δ_0: {tfe_bc95['delta'][0]:.4f} (true: {dgp['true_params']['delta'][0]:.4f})")
print(f"     δ_1: {tfe_bc95['delta'][1]:.4f} (true: {dgp['true_params']['delta'][1]:.4f})")
print(f"     δ_2: {tfe_bc95['delta'][2]:.4f} (true: {dgp['true_params']['delta'][2]:.4f})")


# ============================================================================
# 5. Model Comparison via Likelihood Ratio Tests
# ============================================================================

print("\n5. Model Comparison: Likelihood Ratio Tests")

# Test 1: TRE vs TRE+BC95 (are inefficiency determinants significant?)
lr1 = lr_test(
    loglik_restricted=tre_simple["loglik"],
    loglik_unrestricted=tre_bc95["loglik"],
    df_diff=3,  # Three delta parameters
)

print(f"\n   Test 1: TRE vs TRE+BC95 (H0: δ = 0)")
print(f"   - LR statistic: {lr1['statistic']:.4f}")
print(f"   - P-value:      {lr1['pvalue']:.6f}")
print(f"   - Conclusion:   {lr1['conclusion']}")
print(f"   → {lr1['interpretation']}")

# Test 2: Compare TFE+BC95 vs TRE+BC95
print(f"\n   Test 2: Model Fit Comparison")
print(f"   - TFE+BC95 LL: {tfe_bc95['loglik']:.2f}")
print(f"   - TRE+BC95 LL: {tre_bc95['loglik']:.2f}")
print(f"   - Difference:  {tre_bc95['loglik'] - tfe_bc95['loglik']:.2f}")


# ============================================================================
# 6. Variance Decomposition (TRE + BC95)
# ============================================================================

print("\n6. Variance Decomposition (TRE + BC95)")

decomp = variance_decomposition_tre(
    tre_bc95["sigma_v_sq"], tre_bc95["sigma_u_sq"], tre_bc95["sigma_w_sq"]
)

print(f"\n   Total variance: {decomp['sigma_total_sq']:.4f}")
print(f"   Components:")
print(f"   - Noise (γ_v):          {decomp['gamma_v']:.1%}")
print(f"   - Inefficiency (γ_u):   {decomp['gamma_u']:.1%}")
print(f"   - Heterogeneity (γ_w):  {decomp['gamma_w']:.1%}")


# ============================================================================
# 7. Interpretation of Results
# ============================================================================

print("\n7. Interpretation of Inefficiency Determinants")

print(f"\n   δ_1 (Firm Size) = {tre_bc95['delta'][1]:.4f}")
if tre_bc95["delta"][1] < 0:
    print("   → Larger firms tend to be MORE efficient (negative δ)")
    print("   → Size brings economies of scale, better management")
else:
    print("   → Larger firms tend to be LESS efficient (positive δ)")
    print("   → Size may bring bureaucracy, coordination problems")

print(f"\n   δ_2 (Firm Age) = {tre_bc95['delta'][2]:.4f}")
if tre_bc95["delta"][2] < 0:
    print("   → Older firms tend to be MORE efficient (negative δ)")
    print("   → Learning-by-doing, accumulated experience")
else:
    print("   → Older firms tend to be LESS efficient (positive δ)")
    print("   → Obsolete technology, organizational inertia")


# ============================================================================
# 8. Visualization
# ============================================================================

print("\n8. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Parameter Recovery - Frontier
ax1 = axes[0, 0]
params = ["β₀", "β₁", "β₂"]
x_pos = np.arange(len(params))
width = 0.2

ax1.bar(x_pos - width, dgp["true_params"]["beta"], width, label="True", alpha=0.8)
ax1.bar(x_pos, tre_bc95["beta"], width, label="TRE+BC95", alpha=0.8)
ax1.bar(x_pos + width, tfe_bc95["beta"], width, label="TFE+BC95", alpha=0.8)
ax1.set_xlabel("Parameter")
ax1.set_ylabel("Estimate")
ax1.set_title("Frontier Parameter Recovery")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(params)
ax1.legend()
ax1.grid(True, alpha=0.3, axis="y")

# Plot 2: Parameter Recovery - Inefficiency Determinants
ax2 = axes[0, 1]
params_delta = ["δ₀", "δ₁ (Size)", "δ₂ (Age)"]
x_pos_delta = np.arange(len(params_delta))

ax2.bar(x_pos_delta - width / 2, dgp["true_params"]["delta"], width, label="True", alpha=0.8)
ax2.bar(x_pos_delta + width / 2, tre_bc95["delta"], width, label="TRE+BC95", alpha=0.8, color="C2")
ax2.set_xlabel("Parameter")
ax2.set_ylabel("Estimate")
ax2.set_title("Inefficiency Determinants Recovery")
ax2.set_xticks(x_pos_delta)
ax2.set_xticklabels(params_delta)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

# Plot 3: Variance Decomposition
ax3 = axes[1, 0]
labels = ["Noise\n(v_it)", "Inefficiency\n(u_it)", "Heterogeneity\n(w_i)"]
sizes = [decomp["gamma_v"], decomp["gamma_u"], decomp["gamma_w"]]
colors = ["#ff9999", "#66b3ff", "#99ff99"]
ax3.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
ax3.set_title("Variance Decomposition (TRE+BC95)")

# Plot 4: Model Comparison (Log-likelihood)
ax4 = axes[1, 1]
models = ["TRE\n(simple)", "TRE+BC95", "TFE+BC95"]
logliks = [tre_simple["loglik"], tre_bc95["loglik"], tfe_bc95["loglik"]]
colors_bar = ["#1f77b4", "#ff7f0e", "#2ca02c"]

bars = ax4.bar(models, logliks, color=colors_bar, alpha=0.7)
ax4.set_ylabel("Log-likelihood")
ax4.set_title("Model Comparison")
ax4.grid(True, alpha=0.3, axis="y")

# Add values on bars
for bar, ll in zip(bars, logliks):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2.0, height, f"{ll:.1f}", ha="center", va="bottom")

plt.tight_layout()
plt.savefig("sfa_true_bc95_example.png", dpi=150, bbox_inches="tight")
print("   - Saved figure: sfa_true_bc95_example.png")

plt.show()


# ============================================================================
# 9. Summary
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY: TRUE MODELS WITH BC95")
print("=" * 70)

print("\n1. Model Architecture:")
print("   TRE + BC95 = w_i (heterogeneity) + u_it(Z) (inefficiency) + v_it (noise)")
print("   - Three-component error decomposition")
print("   - Inefficiency varies with firm characteristics (Z)")
print("   - Most comprehensive panel SFA model")

print("\n2. Key Findings:")
print(f"   - Adding inefficiency determinants improves fit")
print(f"     (LR statistic: {lr1['statistic']:.2f}, p-value: {lr1['pvalue']:.6f})")
print(f"   - Firm size effect: {tre_bc95['delta'][1]:.3f}")
print(f"   - Firm age effect:  {tre_bc95['delta'][2]:.3f}")

print("\n3. Advantages of True + BC95:")
print("   ✓ Separates heterogeneity from inefficiency")
print("   ✓ Models systematic inefficiency determinants")
print("   ✓ Allows time-varying inefficiency")
print("   ✓ Provides detailed variance decomposition")
print("   ✓ Avoids confounding α_i with u_it")

print("\n4. When to Use:")
print("   - Panel data with T ≥ 6")
print("   - Suspected firm heterogeneity AND inefficiency determinants")
print("   - Need to understand what drives inefficiency")
print("   - Policy analysis (e.g., how size affects efficiency)")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
