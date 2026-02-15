"""
Example: Spatial Durbin Model (SDM) and Effects Decomposition

This example demonstrates:
1. Estimating a Spatial Durbin Model with panel data
2. Computing direct, indirect, and total effects
3. Visualizing spatial effects decomposition
4. Comparing SAR vs SDM specifications

The SDM extends the SAR model by including spatial lags of explanatory variables:
y = ρWy + Xβ + WXθ + α + ε

This allows for richer spatial spillover patterns where neighboring regions'
characteristics (WX) directly affect outcomes.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.spatial_weights import SpatialWeights
from panelbox.effects.spatial_effects import compute_spatial_effects

# PanelBox imports
from panelbox.models.spatial import SpatialDurbin, SpatialLag
from panelbox.visualization.spatial_plots import (
    plot_direct_vs_indirect,
    plot_effects_comparison,
    plot_spatial_effects,
)


def generate_spatial_panel_data(N=50, T=10, rho=0.4, beta=None, theta=None, seed=42):
    """
    Generate simulated panel data with SDM structure.

    Parameters
    ----------
    N : int
        Number of spatial units
    T : int
        Number of time periods
    rho : float
        Spatial autoregressive parameter
    beta : array-like
        Direct effects coefficients
    theta : array-like
        Spatial spillover coefficients (if None, generates SAR data)
    seed : int
        Random seed

    Returns
    -------
    dict
        Data dictionary with panel DataFrame and spatial weights
    """
    np.random.seed(seed)

    if beta is None:
        beta = np.array([1.5, -0.8, 0.3])

    K = len(beta)

    # Generate spatial weights matrix (nearest neighbors)
    W = np.zeros((N, N))
    for i in range(N):
        # Connect to adjacent units (circular)
        neighbors = [(i - 1) % N, (i + 1) % N]
        for j in neighbors:
            W[i, j] = 1

    # Row normalize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums

    # Generate exogenous variables
    X = np.random.randn(N * T, K)

    # Add some persistence
    for k in range(K):
        X[:, k] = 0.7 * X[:, k] + 0.3 * np.random.randn(N * T)

    # Entity fixed effects
    alpha = np.random.randn(N) * 2
    alpha_panel = np.repeat(alpha, T)

    # Generate dependent variable
    y = np.zeros(N * T)
    epsilon = np.random.randn(N * T)

    for t in range(T):
        start_idx = t * N
        end_idx = (t + 1) * N

        # Compute spatial multiplier
        I_rhoW_inv = np.linalg.inv(np.eye(N) - rho * W)

        # Direct effects: Xβ
        Xbeta = X[start_idx:end_idx] @ beta

        # Spatial spillovers: WXθ (if SDM)
        if theta is not None:
            WX = W @ X[start_idx:end_idx]
            WXtheta = WX @ theta
        else:
            WXtheta = 0

        # Generate y_t
        y_t = I_rhoW_inv @ (Xbeta + WXtheta + alpha + epsilon[start_idx:end_idx])
        y[start_idx:end_idx] = y_t

    # Create panel DataFrame
    entity_ids = np.repeat(np.arange(N), T)
    time_ids = np.tile(np.arange(T), N)

    data = pd.DataFrame(
        {
            "entity": entity_ids,
            "time": time_ids,
            "y": y,
            "population": X[:, 0],
            "income": X[:, 1],
            "education": X[:, 2],
        }
    )

    # Set MultiIndex
    data = data.set_index(["entity", "time"])

    # Add some interpretable patterns
    data["population"] = (data["population"] - data["population"].min() + 1) * 1000
    data["income"] = (data["income"] - data["income"].min() + 10) * 5000
    data["education"] = (data["education"] - data["education"].min() + 5) * 2

    return {"data": data, "W": W, "true_params": {"rho": rho, "beta": beta, "theta": theta}}


def main():
    """Run the SDM example analysis."""

    print("=" * 80)
    print("Spatial Durbin Model (SDM) Example")
    print("=" * 80)

    # 1. Generate data with spatial spillovers
    print("\n1. Generating simulated panel data with spatial spillovers...")

    # True parameters
    beta_true = np.array([0.5, 0.3, -0.2])
    theta_true = np.array([0.2, 0.1, -0.1])  # Spatial spillovers

    data_dict = generate_spatial_panel_data(
        N=50, T=10, rho=0.35, beta=beta_true, theta=theta_true, seed=42
    )

    data = data_dict["data"]
    W = data_dict["W"]

    print(
        f"Panel dimensions: N={len(data.index.get_level_values('entity').unique())}, "
        f"T={len(data.index.get_level_values('time').unique())}"
    )
    print(f"Total observations: {len(data)}")
    print(f"\nSpatial weight matrix: {W.shape}")
    print(f"Average number of neighbors: {(W > 0).sum(axis=1).mean():.1f}")

    # 2. Estimate SAR model (nested model without WX)
    print("\n2. Estimating SAR model (without spatial lags of X)...")

    sar_model = SpatialLag(
        formula="y ~ population + income + education",
        data=data,
        entity_col="entity",
        time_col="time",
        W=W,
    )

    sar_result = sar_model.fit(effects="fixed", method="qml")

    print("\nSAR Model Results:")
    print("-" * 40)
    print(f"Spatial parameter (ρ): {sar_result.params['rho']:.4f}")
    print(f"Population effect: {sar_result.params['population']:.4f}")
    print(f"Income effect: {sar_result.params['income']:.4f}")
    print(f"Education effect: {sar_result.params['education']:.4f}")
    print(f"Log-likelihood: {sar_result.llf:.2f}")

    # 3. Estimate SDM (full model with WX)
    print("\n3. Estimating SDM (with spatial lags of X)...")

    sdm_model = SpatialDurbin(
        formula="y ~ population + income + education",
        data=data,
        entity_col="entity",
        time_col="time",
        W=W,
        effects="fixed",
    )

    sdm_result = sdm_model.fit(method="qml")

    print("\nSDM Model Results:")
    print("-" * 40)
    print(f"Spatial parameter (ρ): {sdm_result.params['rho']:.4f}")
    print("\nDirect coefficients (β):")
    print(f"  Population: {sdm_result.params['population']:.4f}")
    print(f"  Income: {sdm_result.params['income']:.4f}")
    print(f"  Education: {sdm_result.params['education']:.4f}")
    print("\nSpatial spillover coefficients (θ):")
    print(f"  W*Population: {sdm_result.params.get('W*population', 0):.4f}")
    print(f"  W*Income: {sdm_result.params.get('W*income', 0):.4f}")
    print(f"  W*Education: {sdm_result.params.get('W*education', 0):.4f}")
    print(f"\nLog-likelihood: {sdm_result.llf:.2f}")

    # 4. Model comparison
    print("\n4. Model Comparison (SAR vs SDM):")
    print("-" * 40)

    # Likelihood ratio test
    lr_stat = 2 * (sdm_result.llf - sar_result.llf)
    df_diff = len(sdm_result.params) - len(sar_result.params)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)

    print(f"Likelihood Ratio Test:")
    print(f"  LR statistic: {lr_stat:.3f}")
    print(f"  Degrees of freedom: {df_diff}")
    print(f"  P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("  → SDM is preferred (spatial spillovers are significant)")
    else:
        print("  → SAR is sufficient (spatial spillovers not significant)")

    # Information criteria
    print(f"\nInformation Criteria:")
    print(f"  SAR AIC: {sar_result.aic:.2f}")
    print(f"  SDM AIC: {sdm_result.aic:.2f}")
    print(f"  SAR BIC: {sar_result.bic:.2f}")
    print(f"  SDM BIC: {sdm_result.bic:.2f}")

    # 5. Compute spatial effects decomposition
    print("\n5. Computing Spatial Effects Decomposition...")

    # For SAR model
    sar_effects = compute_spatial_effects(
        sar_result, n_simulations=1000, confidence_level=0.95, method="simulation"
    )

    # For SDM model
    sdm_effects = compute_spatial_effects(
        sdm_result, n_simulations=1000, confidence_level=0.95, method="simulation"
    )

    print("\nSAR Effects Decomposition:")
    print(sar_effects.summary())

    print("\nSDM Effects Decomposition:")
    print(sdm_effects.summary())

    # 6. Visualizations
    print("\n6. Creating visualizations...")

    fig = plt.figure(figsize=(15, 10))

    # 6.1 SDM effects decomposition
    ax1 = fig.add_subplot(2, 3, 1)
    plot_spatial_effects(sdm_effects, ax=ax1, show_ci=True)
    ax1.set_title("SDM: Spatial Effects Decomposition")

    # 6.2 Direct vs Indirect effects
    ax2 = fig.add_subplot(2, 3, 2)
    plot_direct_vs_indirect(sdm_effects, ax=ax2)
    ax2.set_title("SDM: Direct vs Indirect Effects")

    # 6.3 Model comparison
    ax3 = fig.add_subplot(2, 3, 3)
    plot_effects_comparison([sar_effects, sdm_effects], ["SAR", "SDM"], effect_type="total", ax=ax3)
    ax3.set_title("Total Effects: SAR vs SDM")

    # 6.4 Spatial weight matrix structure
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(W[:20, :20], cmap="Blues", aspect="equal")
    ax4.set_title("Spatial Weights Matrix (subset)")
    ax4.set_xlabel("Unit j")
    ax4.set_ylabel("Unit i")
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # 6.5 Parameter estimates comparison
    ax5 = fig.add_subplot(2, 3, 5)
    params_compare = pd.DataFrame(
        {
            "SAR": [sar_result.params.get(p, 0) for p in ["population", "income", "education"]],
            "SDM_direct": [
                sdm_result.params.get(p, 0) for p in ["population", "income", "education"]
            ],
            "SDM_spillover": [
                sdm_result.params.get(f"W*{p}", 0) for p in ["population", "income", "education"]
            ],
        },
        index=["Population", "Income", "Education"],
    )

    params_compare.plot(kind="bar", ax=ax5)
    ax5.set_title("Parameter Estimates Comparison")
    ax5.set_ylabel("Coefficient Value")
    ax5.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax5.legend(title="Model/Type")

    # 6.6 Effects magnitude comparison
    ax6 = fig.add_subplot(2, 3, 6)

    # Extract total effects for comparison
    variables = ["population", "income", "education"]
    sar_total = [sar_effects.effects[v]["total"] for v in variables]
    sdm_total = [sdm_effects.effects[v]["total"] for v in variables]

    x = np.arange(len(variables))
    width = 0.35

    ax6.bar(x - width / 2, sar_total, width, label="SAR", color="steelblue")
    ax6.bar(x + width / 2, sdm_total, width, label="SDM", color="coral")

    ax6.set_xlabel("Variable")
    ax6.set_ylabel("Total Effect")
    ax6.set_title("Total Effects Comparison")
    ax6.set_xticks(x)
    ax6.set_xticklabels(["Population", "Income", "Education"])
    ax6.legend()
    ax6.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("sdm_effects_analysis.png", dpi=150, bbox_inches="tight")
    print("\nVisualization saved as 'sdm_effects_analysis.png'")

    # 7. Interpretation
    print("\n7. Interpretation of Results:")
    print("=" * 80)

    # Compare direct and indirect effects
    for var in variables:
        sdm_eff = sdm_effects.effects[var]
        print(f"\n{var.capitalize()}:")
        print(f"  Direct effect: {sdm_eff['direct']:.4f} (SE: {sdm_eff['direct_se']:.4f})")
        print(f"  Indirect effect: {sdm_eff['indirect']:.4f} (SE: {sdm_eff['indirect_se']:.4f})")
        print(f"  Total effect: {sdm_eff['total']:.4f} (SE: {sdm_eff['total_se']:.4f})")

        # Interpretation
        spillover_pct = (
            abs(sdm_eff["indirect"] / sdm_eff["total"]) * 100 if sdm_eff["total"] != 0 else 0
        )
        print(f"  → Spillovers account for {spillover_pct:.1f}% of total effect")

    # Summary insights
    print("\n" + "=" * 80)
    print("Key Insights:")
    print("-" * 40)
    print("1. The SDM captures both direct effects (β) and spatial spillovers (θ)")
    print("2. Total effects = Direct + Indirect, accounting for feedback loops")
    print("3. Indirect effects arise from both ρ (SAR) and θ (SDM spillovers)")
    print("4. Significance of θ parameters indicates presence of local spillovers")
    print("5. Effects decomposition helps quantify the importance of spatial interactions")

    print("\nAnalysis complete!")

    return {
        "sar_result": sar_result,
        "sdm_result": sdm_result,
        "sar_effects": sar_effects,
        "sdm_effects": sdm_effects,
    }


if __name__ == "__main__":
    results = main()
