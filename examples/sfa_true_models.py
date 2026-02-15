"""
Example: True Fixed Effects and True Random Effects Models (Greene 2005)

This example demonstrates how to use Greene's (2005) "True" models that
properly separate firm heterogeneity from technical inefficiency.

Key Features:
1. True Fixed Effects (TFE): Separates α_i from u_it
2. True Random Effects (TRE): Three-component error (w_i, u_it, v_it)
3. Hausman test for model selection
4. Bias correction for TFE
5. Variance decomposition for TRE

References:
    Greene, W. H. (2005).
        Reconsidering heterogeneity in panel data estimators of the stochastic
        frontier model. Journal of Econometrics, 126(2), 269-303.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panelbox.frontier.tests import hausman_test_tfe_tre, heterogeneity_significance_test, lr_test
from panelbox.frontier.true_models import (
    bias_correct_tfe_analytical,
    loglik_true_fixed_effects,
    loglik_true_random_effects,
    variance_decomposition_tre,
)

# ============================================================================
# 1. Generate Simulated Data
# ============================================================================


def generate_panel_data(N=100, T=8, seed=42):
    """Generate panel data with heterogeneity and inefficiency.

    Model: y_it = α_i + β_0 + β_1*labor + β_2*capital + v_it - u_it

    where:
        α_i ~ N(0, 0.5²) is firm heterogeneity (technology differences)
        u_it ~ half-normal(0.3²) is time-varying inefficiency
        v_it ~ N(0, 0.2²) is noise
    """
    np.random.seed(seed)

    # True parameters
    beta_true = np.array([5.0, 0.6, 0.4])  # Intercept, labor, capital
    sigma_v = 0.2
    sigma_u = 0.3
    sigma_alpha = 0.5  # Heterogeneity std

    # Panel structure
    entity_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    # Inputs (in logs)
    log_labor = np.random.randn(N * T) * 0.5 + 2.0
    log_capital = np.random.randn(N * T) * 0.5 + 2.5

    X = np.column_stack([np.ones(N * T), log_labor, log_capital])

    # Firm heterogeneity (persistent technology differences)
    alpha_i = np.random.randn(N) * sigma_alpha
    alpha = alpha_i[entity_id]

    # Time-varying inefficiency
    u = np.abs(np.random.randn(N * T)) * sigma_u

    # Noise
    v = np.random.randn(N * T) * sigma_v

    # Output (in logs)
    log_output = X @ beta_true + alpha - u + v

    # Create DataFrame
    data = pd.DataFrame(
        {
            "firm": entity_id,
            "year": time_id,
            "log_output": log_output,
            "log_labor": log_labor,
            "log_capital": log_capital,
        }
    )

    return {
        "data": data,
        "true_params": {
            "beta": beta_true,
            "sigma_v": sigma_v,
            "sigma_u": sigma_u,
            "sigma_alpha": sigma_alpha,
            "alpha_i": alpha_i,
        },
        "arrays": {"y": log_output, "X": X, "entity_id": entity_id, "time_id": time_id},
    }


print("=" * 70)
print("TRUE FIXED EFFECTS AND TRUE RANDOM EFFECTS MODELS")
print("Greene (2005) - Separating Heterogeneity from Inefficiency")
print("=" * 70)

# Generate data
print("\n1. Generating simulated panel data...")
dgp = generate_panel_data(N=100, T=8)
data = dgp["data"]
arrays = dgp["arrays"]

print(f"   - N = {data['firm'].nunique()} firms")
print(f"   - T = {data['year'].nunique()} years")
print(f"   - Total observations: {len(data)}")
print(f"   - True β: {dgp['true_params']['beta']}")
print(f"   - True σ_v: {dgp['true_params']['sigma_v']:.3f}")
print(f"   - True σ_u: {dgp['true_params']['sigma_u']:.3f}")


# ============================================================================
# 2. Estimate True Fixed Effects (TFE) Model
# ============================================================================

print("\n2. Estimating True Fixed Effects (TFE) model...")


def estimate_tfe(y, X, entity_id, time_id, verbose=True):
    """Estimate TFE model via MLE."""
    k = X.shape[1]

    # Starting values (from OLS)
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    sigma_ols = np.std(resid_ols)

    theta_start = np.concatenate(
        [beta_ols, [np.log(sigma_ols**2 / 2)], [np.log(sigma_ols**2 / 2)]]  # σ²_v  # σ²_u
    )

    # Negative log-likelihood
    def neg_loglik(theta):
        ll = loglik_true_fixed_effects(theta, y, X, entity_id, time_id, sign=1)
        return -ll

    # Optimize
    result = minimize(
        neg_loglik,
        theta_start,
        method="L-BFGS-B",
        bounds=[(None, None)] * k + [(-13.8, 13.8), (-13.8, 13.8)],
    )

    if verbose:
        print(f"   - Converged: {result.success}")
        print(f"   - Log-likelihood: {-result.fun:.2f}")

    # Extract estimates
    beta_hat = result.x[:k]
    sigma_v_sq_hat = np.exp(result.x[k])
    sigma_u_sq_hat = np.exp(result.x[k + 1])

    # Get alpha estimates
    alpha_result = loglik_true_fixed_effects(
        result.x, y, X, entity_id, time_id, sign=1, return_alpha=True
    )

    N = len(np.unique(entity_id))
    alpha_hat = np.array([alpha_result["alpha"][i] for i in range(N)])

    return {
        "beta": beta_hat,
        "sigma_v_sq": sigma_v_sq_hat,
        "sigma_u_sq": sigma_u_sq_hat,
        "alpha": alpha_hat,
        "loglik": -result.fun,
        "theta": result.x,
        "vcov": (
            np.linalg.inv(result.hess_inv.todense())
            if hasattr(result.hess_inv, "todense")
            else None
        ),
    }


tfe_result = estimate_tfe(arrays["y"], arrays["X"], arrays["entity_id"], arrays["time_id"])

print(f"\n   TFE Parameter Estimates:")
print(
    f"   β_0 (Intercept):  {tfe_result['beta'][0]:.4f} (true: {dgp['true_params']['beta'][0]:.4f})"
)
print(
    f"   β_1 (Labor):      {tfe_result['beta'][1]:.4f} (true: {dgp['true_params']['beta'][1]:.4f})"
)
print(
    f"   β_2 (Capital):    {tfe_result['beta'][2]:.4f} (true: {dgp['true_params']['beta'][2]:.4f})"
)
print(
    f"   σ²_v:             {tfe_result['sigma_v_sq']:.4f} (true: {dgp['true_params']['sigma_v']**2:.4f})"
)
print(
    f"   σ²_u:             {tfe_result['sigma_u_sq']:.4f} (true: {dgp['true_params']['sigma_u']**2:.4f})"
)

# Bias correction for TFE
print("\n3. Applying bias correction to TFE fixed effects...")
N = len(tfe_result["alpha"])
T = len(np.unique(arrays["time_id"]))

alpha_corrected = bias_correct_tfe_analytical(
    tfe_result["alpha"], T, tfe_result["sigma_v_sq"], tfe_result["sigma_u_sq"]
)

# Compare with true alpha
correlation_uncorrected = np.corrcoef(tfe_result["alpha"], dgp["true_params"]["alpha_i"])[0, 1]
correlation_corrected = np.corrcoef(alpha_corrected, dgp["true_params"]["alpha_i"])[0, 1]

print(f"   - Correlation (uncorrected): {correlation_uncorrected:.4f}")
print(f"   - Correlation (corrected):   {correlation_corrected:.4f}")
print(f"   - Mean bias correction:      {np.mean(alpha_corrected - tfe_result['alpha']):.4f}")


# ============================================================================
# 4. Estimate True Random Effects (TRE) Model
# ============================================================================

print("\n4. Estimating True Random Effects (TRE) model...")


def estimate_tre(y, X, entity_id, time_id, n_quad=32, verbose=True):
    """Estimate TRE model via MLE with Gauss-Hermite quadrature."""
    k = X.shape[1]

    # Starting values
    beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    resid_ols = y - X @ beta_ols
    sigma_ols = np.std(resid_ols)

    theta_start = np.concatenate(
        [
            beta_ols,
            [np.log(sigma_ols**2 / 3)],  # σ²_v
            [np.log(sigma_ols**2 / 3)],  # σ²_u
            [np.log(sigma_ols**2 / 3)],  # σ²_w
        ]
    )

    # Negative log-likelihood
    def neg_loglik(theta):
        ll = loglik_true_random_effects(
            theta, y, X, entity_id, time_id, sign=1, n_quadrature=n_quad, method="gauss-hermite"
        )
        return -ll

    # Optimize
    result = minimize(
        neg_loglik, theta_start, method="L-BFGS-B", bounds=[(None, None)] * k + [(-13.8, 13.8)] * 3
    )

    if verbose:
        print(f"   - Converged: {result.success}")
        print(f"   - Log-likelihood: {-result.fun:.2f}")
        print(f"   - Quadrature points: {n_quad}")

    # Extract estimates
    beta_hat = result.x[:k]
    sigma_v_sq_hat = np.exp(result.x[k])
    sigma_u_sq_hat = np.exp(result.x[k + 1])
    sigma_w_sq_hat = np.exp(result.x[k + 2])

    return {
        "beta": beta_hat,
        "sigma_v_sq": sigma_v_sq_hat,
        "sigma_u_sq": sigma_u_sq_hat,
        "sigma_w_sq": sigma_w_sq_hat,
        "loglik": -result.fun,
        "theta": result.x,
        "vcov": None,  # Would need numerical Hessian
    }


tre_result = estimate_tre(
    arrays["y"], arrays["X"], arrays["entity_id"], arrays["time_id"], n_quad=32
)

print(f"\n   TRE Parameter Estimates:")
print(
    f"   β_0 (Intercept):  {tre_result['beta'][0]:.4f} (true: {dgp['true_params']['beta'][0]:.4f})"
)
print(
    f"   β_1 (Labor):      {tre_result['beta'][1]:.4f} (true: {dgp['true_params']['beta'][1]:.4f})"
)
print(
    f"   β_2 (Capital):    {tre_result['beta'][2]:.4f} (true: {dgp['true_params']['beta'][2]:.4f})"
)
print(
    f"   σ²_v:             {tre_result['sigma_v_sq']:.4f} (true: {dgp['true_params']['sigma_v']**2:.4f})"
)
print(
    f"   σ²_u:             {tre_result['sigma_u_sq']:.4f} (true: {dgp['true_params']['sigma_u']**2:.4f})"
)
print(
    f"   σ²_w:             {tre_result['sigma_w_sq']:.4f} (true: {dgp['true_params']['sigma_alpha']**2:.4f})"
)


# ============================================================================
# 5. Variance Decomposition for TRE
# ============================================================================

print("\n5. Variance decomposition (TRE model)...")

decomp = variance_decomposition_tre(
    tre_result["sigma_v_sq"], tre_result["sigma_u_sq"], tre_result["sigma_w_sq"]
)

print(f"\n   Total variance: {decomp['sigma_total_sq']:.4f}")
print(f"   - Noise (γ_v):          {decomp['gamma_v']:.1%}")
print(f"   - Inefficiency (γ_u):   {decomp['gamma_u']:.1%}")
print(f"   - Heterogeneity (γ_w):  {decomp['gamma_w']:.1%}")


# ============================================================================
# 6. Model Comparison: Likelihood Ratio Test
# ============================================================================

print("\n6. Likelihood Ratio Test: TRE vs Pooled SFA")
print("   (Tests H0: σ²_w = 0)")

# For this example, we assume we have a pooled SFA result
# In practice, you would estimate a pooled model
# Here we use TFE as a proxy (it's similar to pooled with N fixed effects)

lr_result = lr_test(
    loglik_restricted=tfe_result["loglik"],
    loglik_unrestricted=tre_result["loglik"],
    df_diff=1,  # One parameter difference: σ²_w
)

print(f"   - LR statistic: {lr_result['statistic']:.4f}")
print(f"   - P-value:      {lr_result['pvalue']:.4f}")
print(f"   - Conclusion:   {lr_result['conclusion']}")
print(f"   - {lr_result['interpretation']}")


# ============================================================================
# 7. Hausman Test: TFE vs TRE (Conceptual)
# ============================================================================

print("\n7. Hausman Test: TFE vs TRE")
print("   (Tests if w_i is correlated with X)")

# Note: This would require proper variance-covariance matrices
# Here we provide a conceptual demonstration

print("   - For a proper Hausman test, we need:")
print("     1. Variance-covariance matrices from both models")
print("     2. Only frontier parameters (β) are compared")
print("   - If p-value < 0.05: prefer TFE (correlation present)")
print("   - If p-value ≥ 0.05: prefer TRE (more efficient)")

# In this simulated data:
# - Data was generated with uncorrelated α_i and X
# - So TRE should be preferred (more efficient)


# ============================================================================
# 8. Visualization
# ============================================================================

print("\n8. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: True vs Estimated α_i (TFE)
axes[0, 0].scatter(dgp["true_params"]["alpha_i"], tfe_result["alpha"], alpha=0.6)
axes[0, 0].plot([-2, 2], [-2, 2], "r--", label="45° line")
axes[0, 0].set_xlabel("True α_i")
axes[0, 0].set_ylabel("Estimated α_i (TFE)")
axes[0, 0].set_title(f"TFE Fixed Effects\n(Correlation: {correlation_uncorrected:.3f})")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Bias correction effect
axes[0, 1].scatter(tfe_result["alpha"], alpha_corrected, alpha=0.6)
axes[0, 1].plot([-2, 2], [-2, 2], "r--", label="45° line")
axes[0, 1].set_xlabel("Uncorrected α_i")
axes[0, 1].set_ylabel("Bias-corrected α_i")
axes[0, 1].set_title("Effect of Bias Correction (TFE)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Variance decomposition (TRE)
labels = ["Noise\n(v_it)", "Inefficiency\n(u_it)", "Heterogeneity\n(w_i)"]
sizes = [decomp["gamma_v"], decomp["gamma_u"], decomp["gamma_w"]]
colors = ["#ff9999", "#66b3ff", "#99ff99"]
axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
axes[1, 0].set_title("Variance Decomposition (TRE)")

# Plot 4: Parameter comparison
param_names = ["β_0", "β_1", "β_2"]
true_vals = dgp["true_params"]["beta"]
tfe_vals = tfe_result["beta"]
tre_vals = tre_result["beta"]

x_pos = np.arange(len(param_names))
width = 0.25

axes[1, 1].bar(x_pos - width, true_vals, width, label="True", alpha=0.8)
axes[1, 1].bar(x_pos, tfe_vals, width, label="TFE", alpha=0.8)
axes[1, 1].bar(x_pos + width, tre_vals, width, label="TRE", alpha=0.8)
axes[1, 1].set_xlabel("Parameter")
axes[1, 1].set_ylabel("Estimate")
axes[1, 1].set_title("Parameter Estimates: TFE vs TRE")
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(param_names)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("sfa_true_models_example.png", dpi=150, bbox_inches="tight")
print("   - Saved figure: sfa_true_models_example.png")

plt.show()


# ============================================================================
# 9. Summary and Recommendations
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 70)

print("\n1. Model Performance:")
print(f"   - TFE log-likelihood: {tfe_result['loglik']:.2f}")
print(f"   - TRE log-likelihood: {tre_result['loglik']:.2f}")
print(f"   - Difference:         {tre_result['loglik'] - tfe_result['loglik']:.2f}")

print("\n2. When to use each model:")
print("   - Use TFE when:")
print("     • You expect firm heterogeneity correlates with inputs")
print("     • You have sufficient time periods (T ≥ 10)")
print("     • You can apply bias correction for T < 10")
print("\n   - Use TRE when:")
print("     • Heterogeneity is uncorrelated with inputs")
print("     • You want more efficient estimates")
print("     • You need variance decomposition")

print("\n3. Key advantages of True models:")
print("   - Separate heterogeneity from inefficiency")
print("   - Allow time-varying inefficiency")
print("   - More flexible than classical panel models")
print("   - Proper inference via Hausman test")

print("\n" + "=" * 70)
print("Example completed successfully!")
print("=" * 70)
