"""
Panel Stochastic Frontier Analysis Usage Examples

This script demonstrates the usage of panel SFA models in PanelBox:
1. Pitt & Lee (1981) - Time-invariant efficiency
2. Battese & Coelli (1992) - Time-varying efficiency
3. Battese & Coelli (1995) - Inefficiency determinants
4. Cornwell-Schmidt-Sickles (1990) - Distribution-free
5. Kumbhakar (1990) - Flexible time pattern
6. Lee & Schmidt (1993) - Time dummies

These models exploit the panel dimension to:
- Obtain more precise efficiency estimates
- Model time-varying efficiency patterns
- Include determinants of inefficiency
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# CSS model
from panelbox.frontier.css import estimate_css_model

# Panel likelihood functions
from panelbox.frontier.panel_likelihoods import (
    loglik_battese_coelli_92,
    loglik_battese_coelli_95,
    loglik_kumbhakar_1990,
    loglik_pitt_lee_half_normal,
)


def simulate_panel_data(N=100, T=10, seed=42):
    """Simulate panel data for production frontier.

    Model:
        y_{it} = β₀ + β₁*x1_{it} + β₂*x2_{it} + v_{it} - u_i

    where:
        v_{it} ~ N(0, σ²_v) is noise
        u_i ~ N⁺(0, σ²_u) is time-invariant inefficiency

    Parameters:
        N: Number of entities (firms)
        T: Number of time periods
        seed: Random seed

    Returns:
        DataFrame with panel data
    """
    np.random.seed(seed)

    # True parameters
    beta_0 = 2.0
    beta_1 = 0.6
    beta_2 = 0.3
    sigma_v = 0.1
    sigma_u = 0.2

    # Total observations
    n = N * T

    # Entity and time IDs
    entity_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)

    # Generate inputs (log scale)
    x1 = np.random.uniform(0, 3, n)
    x2 = np.random.uniform(0, 3, n)

    # Generate errors
    v = np.random.normal(0, sigma_v, n)

    # Time-invariant inefficiency (one per entity, repeated over time)
    u_i = np.abs(np.random.normal(0, sigma_u, N))
    u = np.repeat(u_i, T)

    # True efficiency
    efficiency = np.exp(-u)

    # Generate output
    y = beta_0 + beta_1 * x1 + beta_2 * x2 + v - u

    # Create DataFrame
    data = pd.DataFrame(
        {
            "entity": entity_id,
            "time": time_id,
            "output": y,
            "x1": x1,
            "x2": x2,
            "true_efficiency": efficiency,
        }
    )

    print(f"\nSimulated panel data: N={N}, T={T}, n={n}")
    print(f"True parameters:")
    print(f"  β = [{beta_0:.2f}, {beta_1:.2f}, {beta_2:.2f}]")
    print(f"  σ_v = {sigma_v:.2f}, σ_u = {sigma_u:.2f}")
    print(f"  Mean true efficiency: {efficiency.mean():.4f}")

    return data


def example_1_pitt_lee():
    """Example 1: Pitt & Lee (1981) model - time-invariant efficiency."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Pitt & Lee (1981) - Time-Invariant Efficiency")
    print("=" * 70)

    # Simulate data
    data = simulate_panel_data(N=100, T=10)

    # Prepare arrays
    y = data["output"].values
    X = np.column_stack([np.ones(len(data)), data["x1"].values, data["x2"].values])
    entity_id = data["entity"].values
    time_id = data["time"].values

    # Define negative log-likelihood (for minimization)
    def negloglik(theta):
        ll = loglik_pitt_lee_half_normal(theta, y, X, entity_id, time_id, sign=1)
        return -ll if np.isfinite(ll) else 1e10

    # Starting values (OLS + simple variance estimates)
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_init

    # Initial variances
    sigma_sq_init = np.var(resid)
    ln_sigma_v_sq_init = np.log(sigma_sq_init / 2)
    ln_sigma_u_sq_init = np.log(sigma_sq_init / 2)

    theta_init = np.concatenate([beta_init, [ln_sigma_v_sq_init], [ln_sigma_u_sq_init]])

    print("\nEstimating via MLE...")
    print(f"Starting values: β={beta_init[:3]}")

    # Optimize
    result = minimize(
        negloglik, theta_init, method="L-BFGS-B", options={"maxiter": 1000, "disp": True}
    )

    if result.success:
        print("\nEstimation successful!")

        # Extract estimates
        k = X.shape[1]
        beta_hat = result.x[:k]
        ln_sigma_v_sq_hat = result.x[k]
        ln_sigma_u_sq_hat = result.x[k + 1]

        sigma_v_hat = np.sqrt(np.exp(ln_sigma_v_sq_hat))
        sigma_u_hat = np.sqrt(np.exp(ln_sigma_u_sq_hat))

        print(f"\nParameter estimates:")
        print(f"  β = {beta_hat}")
        print(f"  σ_v = {sigma_v_hat:.4f}")
        print(f"  σ_u = {sigma_u_hat:.4f}")
        print(f"  λ = σ_u/σ_v = {sigma_u_hat/sigma_v_hat:.4f}")
        print(f"  Log-likelihood = {-result.fun:.4f}")

        return result, data
    else:
        print("\nEstimation failed!")
        return None, data


def example_2_battese_coelli_92():
    """Example 2: Battese & Coelli (1992) - time-varying efficiency."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Battese & Coelli (1992) - Time-Varying Efficiency")
    print("=" * 70)

    print("\nThis model allows efficiency to change over time via:")
    print("  u_{it} = exp[-η(t - T)] * u_i")
    print("\nWhere:")
    print("  η > 0: efficiency improves over time")
    print("  η < 0: efficiency worsens over time")
    print("  η = 0: efficiency is constant (reduces to Pitt-Lee)")

    # Simulate data with time-varying efficiency
    data = simulate_panel_data(N=50, T=10)

    y = data["output"].values
    X = np.column_stack([np.ones(len(data)), data["x1"].values, data["x2"].values])
    entity_id = data["entity"].values
    time_id = data["time"].values

    def negloglik_bc92(theta):
        ll = loglik_battese_coelli_92(theta, y, X, entity_id, time_id, sign=1)
        return -ll if np.isfinite(ll) else 1e10

    # Starting values (from Pitt-Lee + η=0)
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_init

    sigma_sq_init = np.var(resid)

    theta_init = np.concatenate(
        [
            beta_init,
            [np.log(sigma_sq_init / 2)],  # ln(σ²_v)
            [np.log(sigma_sq_init / 2)],  # ln(σ²_u)
            [0.0],  # μ
            [0.01],  # η (start near 0)
        ]
    )

    print("\nEstimating BC92 model...")
    result = minimize(
        negloglik_bc92, theta_init, method="L-BFGS-B", options={"maxiter": 500, "disp": False}
    )

    if result.success:
        print("\nEstimation successful!")

        k = X.shape[1]
        beta_hat = result.x[:k]
        ln_sigma_v_sq_hat = result.x[k]
        ln_sigma_u_sq_hat = result.x[k + 1]
        mu_hat = result.x[k + 2]
        eta_hat = result.x[k + 3]

        sigma_v_hat = np.sqrt(np.exp(ln_sigma_v_sq_hat))
        sigma_u_hat = np.sqrt(np.exp(ln_sigma_u_sq_hat))

        print(f"\nParameter estimates:")
        print(f"  β = {beta_hat}")
        print(f"  σ_v = {sigma_v_hat:.4f}")
        print(f"  σ_u = {sigma_u_hat:.4f}")
        print(f"  μ = {mu_hat:.4f}")
        print(f"  η = {eta_hat:.4f}")

        if eta_hat > 0:
            print(f"\n  → Efficiency IMPROVES over time (η > 0)")
        elif eta_hat < 0:
            print(f"\n  → Efficiency WORSENS over time (η < 0)")
        else:
            print(f"\n  → Efficiency is CONSTANT over time (η ≈ 0)")

        return result, data
    else:
        print("\nEstimation failed!")
        return None, data


def example_3_css():
    """Example 3: Cornwell-Schmidt-Sickles (1990) - distribution-free."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Cornwell-Schmidt-Sickles (1990) - Distribution-Free")
    print("=" * 70)

    print("\nThis model does NOT assume a distribution for inefficiency.")
    print("Instead, it uses time-varying intercepts:")
    print("  α_i(t) = θ_{i1} + θ_{i2}*t + θ_{i3}*t²")

    # Simulate data
    data = simulate_panel_data(N=50, T=10)

    y = data["output"].values
    X_no_const = np.column_stack([data["x1"].values, data["x2"].values])
    entity_id = data["entity"].values
    time_id = data["time"].values

    print("\nEstimating CSS model with quadratic time trend...")

    result = estimate_css_model(
        y=y,
        X=X_no_const,
        entity_id=entity_id,
        time_id=time_id,
        time_trend="quadratic",
        frontier_type="production",
    )

    print(f"\nEstimation complete!")
    print(f"  N = {result.n_entities} entities")
    print(f"  T = {result.n_periods} periods")
    print(f"  R² = {result.r_squared:.4f}")
    print(f"  σ_v = {result.sigma_v:.4f}")

    # Efficiency summary
    print(f"\nEfficiency estimates:")
    print(f"  Mean: {result.efficiency_it.mean():.4f}")
    print(f"  Std:  {result.efficiency_it.std():.4f}")
    print(f"  Min:  {result.efficiency_it.min():.4f}")
    print(f"  Max:  {result.efficiency_it.max():.4f} (should be 1.0)")

    # Get efficiency by entity
    eff_by_entity = result.efficiency_by_entity()
    print(f"\nTop 5 most efficient entities (average over time):")
    print(eff_by_entity.nlargest(5, "mean_efficiency")[["entity", "mean_efficiency", "trend"]])

    # Plot efficiency evolution
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Efficiency over time for first 10 entities
    for i in range(min(10, result.n_entities)):
        ax1.plot(
            range(result.n_periods), result.efficiency_it[i, :], alpha=0.6, label=f"Entity {i}"
        )
    ax1.set_xlabel("Time Period")
    ax1.set_ylabel("Technical Efficiency")
    ax1.set_title("Efficiency Evolution (First 10 Entities)")
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3)

    # Panel B: Distribution of efficiency in first and last period
    ax2.hist(result.efficiency_it[:, 0], bins=20, alpha=0.5, label=f"Period 0", edgecolor="black")
    ax2.hist(
        result.efficiency_it[:, -1],
        bins=20,
        alpha=0.5,
        label=f"Period {result.n_periods-1}",
        edgecolor="black",
    )
    ax2.set_xlabel("Technical Efficiency")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Efficiency Distribution: First vs Last Period")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sfa_panel_css_example.png", dpi=150)
    print(f"\nPlot saved to: sfa_panel_css_example.png")

    return result, data


def example_4_compare_models():
    """Example 4: Compare different panel specifications."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Comparing Panel SFA Specifications")
    print("=" * 70)

    print("\nComparing CSS with different time trend specifications:")
    print("  - none: Fixed effects only (efficiency constant over time)")
    print("  - linear: Linear time trend")
    print("  - quadratic: Quadratic time trend")

    # Simulate data
    data = simulate_panel_data(N=80, T=10)

    y = data["output"].values
    X_no_const = np.column_stack([data["x1"].values, data["x2"].values])
    entity_id = data["entity"].values
    time_id = data["time"].values

    results = {}
    for trend in ["none", "linear", "quadratic"]:
        print(f"\nEstimating with time_trend='{trend}'...")
        result = estimate_css_model(
            y=y,
            X=X_no_const,
            entity_id=entity_id,
            time_id=time_id,
            time_trend=trend,
            frontier_type="production",
        )
        results[trend] = result

    # Compare
    print("\n" + "=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print(f"{'Specification':<15} {'R²':<10} {'σ_v':<10} {'Mean Eff':<12}")
    print("-" * 70)

    for trend, result in results.items():
        print(
            f"{trend:<15} {result.r_squared:>8.4f}  {result.sigma_v:>8.4f}  "
            f"{result.efficiency_it.mean():>10.4f}"
        )

    return results


if __name__ == "__main__":
    """Run all examples."""
    print("\n" + "=" * 70)
    print("PANEL STOCHASTIC FRONTIER ANALYSIS - EXAMPLES")
    print("=" * 70)

    try:
        # Example 1: Pitt-Lee
        result1, data1 = example_1_pitt_lee()

        # Example 2: Battese-Coelli 1992
        result2, data2 = example_2_battese_coelli_92()

        # Example 3: CSS
        result3, data3 = example_3_css()

        # Example 4: Compare models
        results4 = example_4_compare_models()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 70)

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()
