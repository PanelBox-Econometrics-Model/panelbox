"""
Battese & Coelli (1995) - Inefficiency Effects Model

This example demonstrates the BC95 model, which is the most widely used
panel SFA model in applied research. It allows inefficiency to depend on
observed characteristics (determinants).

Model:
    y_{it} = X_{it}β + v_{it} - u_{it}

    where:
        u_{it} ~ N⁺(μ_{it}, σ²_u)
        μ_{it} = Z_{it}δ  (heterogeneous mean)

Key features:
- Single-step MLE estimation (NOT two-step!)
- Inefficiency determinants directly in likelihood
- Interpretation: δ_j > 0 → variable j INCREASES inefficiency
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panelbox.frontier.panel_likelihoods import loglik_battese_coelli_95


def simulate_bc95_data(N=100, T=10, seed=42):
    """Simulate panel data with inefficiency determinants.

    Model:
        y_{it} = β₀ + β₁*x1 + β₂*x2 + v_{it} - u_{it}
        u_{it} ~ N⁺(δ₀ + δ₁*z1 + δ₂*z2, σ²_u)

    where:
        z1, z2 are inefficiency determinants
    """
    np.random.seed(seed)

    # True parameters
    beta_0 = 2.0
    beta_1 = 0.6
    beta_2 = 0.3
    sigma_v = 0.1
    sigma_u = 0.2

    # Determinants parameters
    delta_0 = 0.1  # Base inefficiency
    delta_1 = 0.05  # Firm size effect (smaller firms less efficient)
    delta_2 = -0.03  # Education effect (more educated → more efficient)

    n = N * T

    # Entity and time IDs
    entity_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    # Frontier variables (vary over time)
    x1 = np.random.uniform(0, 3, n)
    x2 = np.random.uniform(0, 3, n)

    # Determinants of inefficiency
    # z1: Firm size (log of employees) - varies slowly over time
    size_base = np.random.uniform(1, 5, N)  # Base size per firm
    size_growth = np.random.uniform(-0.1, 0.1, N)  # Growth rate
    z1 = np.concatenate([size_base[i] + size_growth[i] * np.arange(T) for i in range(N)])

    # z2: Manager education level (constant per firm)
    education = np.random.uniform(10, 18, N)  # Years of education
    z2 = np.repeat(education, T)

    # z3: Market concentration (constant in cross-section, varies over time)
    concentration = 0.5 + 0.1 * (np.arange(T) / T)  # Increases over time
    z3 = np.tile(concentration, N)

    # Compute μ_{it} = Z_{it}δ
    mu_it = delta_0 + delta_1 * z1 + delta_2 * z2

    # Generate errors
    v = np.random.normal(0, sigma_v, n)

    # Generate u_{it} ~ N⁺(μ_{it}, σ²_u)
    u = np.abs(np.random.normal(mu_it, sigma_u, n))

    # Ensure u ≥ 0
    u = np.maximum(u, 0)

    # Generate output
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + v - u

    # True efficiency
    efficiency = np.exp(-u)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "entity": entity_id,
            "time": time_id,
            "output": y,
            "x1": x1,
            "x2": x2,
            "firm_size": z1,
            "education": z2,
            "concentration": z3,
            "true_u": u,
            "true_efficiency": efficiency,
            "true_mu": mu_it,
        }
    )

    print(f"\nSimulated BC95 data: N={N}, T={T}, n={n}")
    print(f"\nTrue FRONTIER parameters:")
    print(f"  β = [{beta_0:.2f}, {beta_1:.2f}, {beta_2:.2f}]")
    print(f"  σ_v = {sigma_v:.2f}, σ_u = {sigma_u:.2f}")

    print(f"\nTrue INEFFICIENCY parameters:")
    print(f"  δ₀ (intercept) = {delta_0:.3f}")
    print(f"  δ₁ (firm size) = {delta_1:.3f}  [> 0: larger firms MORE efficient]")
    print(f"  δ₂ (education) = {delta_2:.3f}  [< 0: more education → LESS inefficiency]")

    print(f"\nDescriptive statistics:")
    print(f"  Mean efficiency: {efficiency.mean():.4f}")
    print(f"  Mean firm size: {z1.mean():.2f}")
    print(f"  Mean education: {z2.mean():.2f} years")

    return data


def estimate_bc95(data, Z_vars):
    """Estimate Battese-Coelli 1995 model.

    Parameters:
        data: Panel DataFrame
        Z_vars: List of determinant variable names

    Returns:
        Optimization result
    """
    # Prepare arrays
    y = data["output"].values
    X = np.column_stack([np.ones(len(data)), data["x1"].values, data["x2"].values])
    Z = np.column_stack([np.ones(len(data))] + [data[z].values for z in Z_vars])

    entity_id = data["entity"].values
    time_id = data["time"].values

    k = X.shape[1]
    m = Z.shape[1]

    # Define negative log-likelihood
    def negloglik(theta):
        ll = loglik_battese_coelli_95(theta, y, X, Z, entity_id, time_id, sign=1)
        return -ll if np.isfinite(ll) else 1e10

    # Starting values
    from sklearn.linear_model import LinearRegression

    # OLS for frontier
    ols = LinearRegression(fit_intercept=False)
    ols.fit(X, y)
    beta_init = ols.coef_
    resid = y - X @ beta_init

    # Initial variances
    sigma_sq_init = np.var(resid)

    # OLS for inefficiency determinants (very rough)
    abs_resid = np.abs(resid)
    ols_z = LinearRegression(fit_intercept=False)
    ols_z.fit(Z, abs_resid)
    delta_init = ols_z.coef_ * 0.1  # Scale down

    theta_init = np.concatenate(
        [beta_init, [np.log(sigma_sq_init / 2)], [np.log(sigma_sq_init / 2)], delta_init]
    )

    print(f"\nStarting values:")
    print(f"  β_init = {beta_init}")
    print(f"  δ_init = {delta_init}")

    print("\nOptimizing...")
    result = minimize(
        negloglik, theta_init, method="L-BFGS-B", options={"maxiter": 1000, "disp": True}
    )

    return result, k, m


def example_bc95_full():
    """Full BC95 example with interpretation."""
    print("\n" + "=" * 70)
    print("BATTESE-COELLI (1995) - INEFFICIENCY EFFECTS MODEL")
    print("=" * 70)

    # Simulate data
    data = simulate_bc95_data(N=100, T=8, seed=123)

    # Estimate model with determinants
    print("\n" + "-" * 70)
    print("Estimating BC95 model with determinants...")
    print("-" * 70)

    Z_vars = ["firm_size", "education"]
    result, k, m = estimate_bc95(data, Z_vars)

    if result.success:
        print("\n" + "=" * 70)
        print("ESTIMATION SUCCESSFUL")
        print("=" * 70)

        # Extract parameters
        beta_hat = result.x[:k]
        ln_sigma_v_sq_hat = result.x[k]
        ln_sigma_u_sq_hat = result.x[k + 1]
        delta_hat = result.x[k + 2 : k + 2 + m]

        sigma_v_hat = np.sqrt(np.exp(ln_sigma_v_sq_hat))
        sigma_u_hat = np.sqrt(np.exp(ln_sigma_u_sq_hat))

        # Display results
        print("\n1. FRONTIER PARAMETERS (β)")
        print("-" * 70)
        print(f"  Constant:  {beta_hat[0]:.4f}")
        print(f"  x1:        {beta_hat[1]:.4f}")
        print(f"  x2:        {beta_hat[2]:.4f}")

        print("\n2. VARIANCE COMPONENTS")
        print("-" * 70)
        print(f"  σ_v (noise):          {sigma_v_hat:.4f}")
        print(f"  σ_u (inefficiency):   {sigma_u_hat:.4f}")
        print(f"  λ = σ_u/σ_v:          {sigma_u_hat/sigma_v_hat:.4f}")

        sigma_sq = sigma_v_hat**2 + sigma_u_hat**2
        gamma = sigma_u_hat**2 / sigma_sq
        print(f"  γ = σ²_u/σ²:          {gamma:.4f}")

        print("\n3. INEFFICIENCY DETERMINANTS (δ)")
        print("-" * 70)
        print(f"  Intercept:    {delta_hat[0]:>8.4f}")
        print(f"  Firm size:    {delta_hat[1]:>8.4f}  ", end="")
        if delta_hat[1] > 0:
            print("(Larger firms MORE inefficient)")
        else:
            print("(Larger firms LESS inefficient)")

        print(f"  Education:    {delta_hat[2]:>8.4f}  ", end="")
        if delta_hat[2] > 0:
            print("(More education → MORE inefficiency)")
        else:
            print("(More education → LESS inefficiency)")

        print("\n4. MODEL FIT")
        print("-" * 70)
        print(f"  Log-likelihood: {-result.fun:.4f}")
        print(f"  Converged:      {result.success}")

        # Marginal effects
        print("\n5. MARGINAL EFFECTS")
        print("-" * 70)
        print("\nEffect of determinants on E[u|Z]:")

        # Sample marginal effect calculation
        Z_mean = np.column_stack([np.ones(1), data[["firm_size", "education"]].mean().values])
        mu_mean = Z_mean @ delta_hat

        print(f"\nAt mean values of Z:")
        print(f"  Mean predicted μ = {mu_mean[0]:.4f}")

        # Approximate marginal effects
        print(f"\nMarginal effects (∂E[u]/∂z ≈ δ):")
        print(f"  ∂E[u]/∂(firm_size)  ≈ {delta_hat[1]:.4f}")
        print(f"  ∂E[u]/∂(education)  ≈ {delta_hat[2]:.4f}")

        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Firm size vs inefficiency
        ax = axes[0, 0]
        scatter_data = data.groupby("entity").agg({"firm_size": "mean", "true_u": "mean"})
        ax.scatter(scatter_data["firm_size"], scatter_data["true_u"], alpha=0.5, s=30)

        # Fitted line
        size_range = np.linspace(
            scatter_data["firm_size"].min(), scatter_data["firm_size"].max(), 50
        )
        Z_plot = np.column_stack(
            [
                np.ones(len(size_range)),
                size_range,
                np.full(len(size_range), data["education"].mean()),
            ]
        )
        mu_plot = Z_plot @ delta_hat
        ax.plot(size_range, mu_plot, "r-", linewidth=2, label="Fitted μ(Z)")

        ax.set_xlabel("Firm Size (log employees)")
        ax.set_ylabel("Inefficiency (u)")
        ax.set_title("Effect of Firm Size on Inefficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Education vs inefficiency
        ax = axes[0, 1]
        scatter_data2 = data.groupby("entity").agg({"education": "mean", "true_u": "mean"})
        ax.scatter(scatter_data2["education"], scatter_data2["true_u"], alpha=0.5, s=30)

        # Fitted line
        edu_range = np.linspace(
            scatter_data2["education"].min(), scatter_data2["education"].max(), 50
        )
        Z_plot2 = np.column_stack(
            [np.ones(len(edu_range)), np.full(len(edu_range), data["firm_size"].mean()), edu_range]
        )
        mu_plot2 = Z_plot2 @ delta_hat
        ax.plot(edu_range, mu_plot2, "r-", linewidth=2, label="Fitted μ(Z)")

        ax.set_xlabel("Manager Education (years)")
        ax.set_ylabel("Inefficiency (u)")
        ax.set_title("Effect of Education on Inefficiency")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Distribution of predicted μ
        ax = axes[1, 0]
        Z_full = np.column_stack(
            [np.ones(len(data)), data["firm_size"].values, data["education"].values]
        )
        mu_pred = Z_full @ delta_hat

        ax.hist(mu_pred, bins=30, alpha=0.7, edgecolor="black")
        ax.axvline(
            mu_pred.mean(),
            color="r",
            linestyle="--",
            linewidth=2,
            label=f"Mean = {mu_pred.mean():.3f}",
        )
        ax.set_xlabel("Predicted μ_{it} = Z_{it}δ")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Predicted Inefficiency Mean")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: True vs predicted μ
        ax = axes[1, 1]
        ax.scatter(data["true_mu"], mu_pred, alpha=0.3, s=10)
        ax.plot(
            [data["true_mu"].min(), data["true_mu"].max()],
            [data["true_mu"].min(), data["true_mu"].max()],
            "r--",
            linewidth=2,
            label="45° line",
        )

        from scipy.stats import pearsonr

        corr, _ = pearsonr(data["true_mu"], mu_pred)

        ax.set_xlabel("True μ_{it}")
        ax.set_ylabel("Predicted μ_{it}")
        ax.set_title(f"True vs Predicted μ (corr = {corr:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("bc95_determinants_example.png", dpi=150)
        print(f"\nPlots saved to: bc95_determinants_example.png")

        return result, data

    else:
        print("\nEstimation failed!")
        print(f"Message: {result.message}")
        return None, data


def example_two_step_warning():
    """Warning about two-step estimation bias."""
    print("\n" + "=" * 70)
    print("WARNING: TWO-STEP ESTIMATION IS BIASED!")
    print("=" * 70)

    print("\nThe two-step approach:")
    print("  Step 1: Estimate frontier, obtain û_i")
    print("  Step 2: Regress û_i on Z")

    print("\nWhy it's WRONG (Wang & Schmidt 2002):")
    print("  • û_i depends on estimated β (not true β)")
    print("  • Ignores correlation between β and δ")
    print("  • Standard errors are INCORRECT")
    print("  • Hypothesis tests are INVALID")

    print("\nBC95 single-step MLE:")
    print("  ✓ Estimates β and δ JOINTLY")
    print("  ✓ Correct standard errors")
    print("  ✓ Asymptotically efficient")

    print("\nRECOMMENDATION:")
    print("  Always use single-step MLE for BC95 model.")
    print("  Never use two-step estimation in research!")


if __name__ == "__main__":
    """Run examples."""
    print("\n" + "=" * 70)
    print("BATTESE-COELLI (1995) EXAMPLES")
    print("=" * 70)

    try:
        # Main example
        result, data = example_bc95_full()

        # Warning about two-step
        example_two_step_warning()

        print("\n" + "=" * 70)
        print("EXAMPLES COMPLETED")
        print("=" * 70)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
