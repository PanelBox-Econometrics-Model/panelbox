"""
Integrated Productivity Analysis Example.

This example demonstrates a complete productivity analysis workflow combining:
1. TFP decomposition (technical change, efficiency change, scale effects)
2. Marginal effects of firm characteristics on inefficiency
3. Joint interpretation for policy recommendations

Application: Manufacturing sector productivity analysis

References:
    Kumbhakar & Lovell (2000). Stochastic Frontier Analysis. Chapter 7.
    Wang & Schmidt (2002). One-step and two-step estimation. JPA 18:129-144.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.frontier.utils import TFPDecomposition, marginal_effects

# ============================================================================
# 1. GENERATE SYNTHETIC DATA
# ============================================================================


def generate_manufacturing_data(n_firms=100, n_years=5, seed=42):
    """Generate synthetic manufacturing panel data.

    Parameters
    ----------
    n_firms : int
        Number of firms
    n_years : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns
    -------
    data : pd.DataFrame
        Panel data with production and firm characteristics
    true_params : dict
        True parameter values (for validation)
    """
    np.random.seed(seed)

    # Panel structure
    firms = np.repeat(range(1, n_firms + 1), n_years)
    years = np.tile(range(2015, 2015 + n_years), n_firms)
    n_obs = n_firms * n_years

    # True production function parameters
    # ln(Y) = β₀ + β_K ln(K) + β_L ln(L) + v - u
    beta_0 = 3.0
    beta_K = 0.35  # Capital elasticity
    beta_L = 0.50  # Labor elasticity
    # RTS = 0.85 (DRS)

    # Firm characteristics affecting inefficiency
    # u_i ~ N⁺(z'δ, σ²_u)
    # z = [firm_age, manager_education, export_share]
    delta_age = 0.015  # Older firms slightly less efficient
    delta_education = -0.025  # Education improves efficiency
    delta_export = -0.020  # Exporters more efficient

    sigma_v = 0.08  # Noise std dev
    sigma_u = 0.15  # Inefficiency std dev

    # Technical progress
    tech_progress_rate = 0.025  # 2.5% per year

    # Generate firm characteristics (time-invariant)
    firm_age = np.random.uniform(5, 50, n_firms)  # Years
    manager_education = np.random.uniform(10, 20, n_firms)  # Years
    export_share = np.random.uniform(0, 0.6, n_firms)  # Share of output

    # Expand to panel
    firm_age_panel = np.repeat(firm_age, n_years)
    manager_education_panel = np.repeat(manager_education, n_years)
    export_share_panel = np.repeat(export_share, n_years)

    # Generate inputs (in logs)
    log_K_base = np.random.uniform(4, 6, n_firms)
    log_L_base = np.random.uniform(3, 5, n_firms)

    # Input growth rates
    K_growth = np.random.uniform(0.02, 0.06, n_firms)
    L_growth = np.random.uniform(0.01, 0.04, n_firms)

    log_K = np.zeros(n_obs)
    log_L = np.zeros(n_obs)
    u = np.zeros(n_obs)

    for i in range(n_firms):
        firm_id = i + 1
        mask = firms == firm_id
        t = np.arange(n_years)

        # Inputs grow over time
        log_K[mask] = log_K_base[i] + K_growth[i] * t
        log_L[mask] = log_L_base[i] + L_growth[i] * t

        # Inefficiency with determinants
        mu_i = (
            delta_age * firm_age[i]
            + delta_education * manager_education[i]
            + delta_export * export_share[i]
        )

        # Time-varying inefficiency (AR process)
        u_base = max(0.01, np.random.normal(mu_i, sigma_u))
        u_temp = np.zeros(n_years)
        u_temp[0] = u_base

        for t_idx in range(1, n_years):
            u_temp[t_idx] = 0.7 * u_temp[t_idx - 1] + 0.3 * mu_i + np.random.normal(0, 0.03)
            u_temp[t_idx] = max(0.01, u_temp[t_idx])

        u[mask] = u_temp

    # Technical change
    tech_level = tech_progress_rate * (years - 2015)

    # Noise
    v = np.random.normal(0, sigma_v, n_obs)

    # Output
    log_Y = beta_0 + beta_K * log_K + beta_L * log_L + tech_level + v - u

    # Create DataFrame
    data = pd.DataFrame(
        {
            "firm": firms,
            "year": years,
            "log_output": log_Y,
            "log_capital": log_K,
            "log_labor": log_L,
            "firm_age": firm_age_panel,
            "manager_education": manager_education_panel,
            "export_share": export_share_panel,
            "true_inefficiency": u,
            "true_efficiency": np.exp(-u),
        }
    )

    true_params = {
        "beta_0": beta_0,
        "beta_K": beta_K,
        "beta_L": beta_L,
        "delta_age": delta_age,
        "delta_education": delta_education,
        "delta_export": delta_export,
        "sigma_v": sigma_v,
        "sigma_u": sigma_u,
        "tech_progress_rate": tech_progress_rate,
    }

    return data, true_params


# ============================================================================
# 2. MOCK SFA MODEL (For demonstration without full SFA implementation)
# ============================================================================


class MockModel:
    """Mock SFA model for demonstration."""

    def __init__(self, data):
        self.data = data
        self.entity = "firm"
        self.time = "year"
        self.depvar = "log_output"
        self.exog = ["log_capital", "log_labor"]
        self.n_exog = 2
        self.inefficiency_vars = ["firm_age", "manager_education", "export_share"]
        self.ineff_var_names = self.inefficiency_vars


class MockResult:
    """Mock SFA result."""

    def __init__(self, model, params):
        self.model = model
        self.params = params
        k = len(params)
        self.vcov = np.eye(k) * 0.001  # Mock variance-covariance matrix

    def efficiency(self, estimator="bc"):
        """Return efficiency scores."""
        return self.model.data[["firm", "year", "true_efficiency"]].rename(
            columns={"firm": "entity", "year": "time", "true_efficiency": "efficiency"}
        )


# ============================================================================
# 3. MAIN ANALYSIS
# ============================================================================


def main():
    """Run integrated productivity analysis."""
    print("=" * 80)
    print("INTEGRATED PRODUCTIVITY ANALYSIS".center(80))
    print("Manufacturing Sector Example".center(80))
    print("=" * 80)
    print()

    # Generate data
    print("1. Generating synthetic data...")
    data, true_params = generate_manufacturing_data(n_firms=100, n_years=5)
    print(
        f"   ✓ Panel: {data['firm'].nunique()} firms, "
        f"{data['year'].nunique()} years, "
        f"{len(data)} observations"
    )
    print()

    # Create mock model and result
    print("2. Estimating SFA model...")
    model = MockModel(data)
    params = np.array(
        [
            true_params["beta_K"],
            true_params["beta_L"],
            true_params["sigma_v"] ** 2,
            true_params["sigma_u"] ** 2,
            true_params["delta_age"],
            true_params["delta_education"],
            true_params["delta_export"],
        ]
    )
    result = MockResult(model, params)
    print(f"   ✓ Estimated parameters:")
    print(f"     β_K = {params[0]:.3f}, β_L = {params[1]:.3f}")
    print(f"     RTS = {params[0] + params[1]:.3f}")
    print()

    # ========================================================================
    # A. TFP DECOMPOSITION
    # ========================================================================

    print("=" * 80)
    print("A. TFP DECOMPOSITION ANALYSIS".center(80))
    print("=" * 80)
    print()

    tfp = TFPDecomposition(result, periods=(2015, 2019))
    decomp = tfp.decompose()

    # Print summary
    print(tfp.summary())
    print()

    # Aggregate results
    agg = tfp.aggregate_decomposition()

    # Identify sources of growth
    print("Key Findings:")
    print("-" * 80)

    if agg["mean_delta_tfp"] > 0:
        print(
            f"✓ Positive aggregate TFP growth: {agg['mean_delta_tfp']:.3f} "
            f"(~{100 * agg['mean_delta_tfp']:.1f}%)"
        )
    else:
        print(f"✗ Negative aggregate TFP growth: {agg['mean_delta_tfp']:.3f}")

    # Dominant component
    components = [
        ("Technical Change", agg["mean_delta_tc"], agg["pct_from_tc"]),
        ("Efficiency Change", agg["mean_delta_te"], agg["pct_from_te"]),
        ("Scale Effects", agg["mean_delta_se"], agg["pct_from_se"]),
    ]
    dominant = max(components, key=lambda x: abs(x[1]))

    print(f"\nDominant driver: {dominant[0]} ({abs(dominant[2]):.1f}% of total)")
    print()

    # ========================================================================
    # B. MARGINAL EFFECTS ANALYSIS
    # ========================================================================

    print("=" * 80)
    print("B. MARGINAL EFFECTS ANALYSIS".center(80))
    print("=" * 80)
    print()

    print("Effects of firm characteristics on inefficiency:")
    print()

    me = marginal_effects(result, method="mean")
    print(me.to_string(index=False))
    print()
    print("Note: Positive ME = increases inefficiency (decreases efficiency)")
    print("      Negative ME = decreases inefficiency (increases efficiency)")
    print()

    # ========================================================================
    # C. INTEGRATED INTERPRETATION
    # ========================================================================

    print("=" * 80)
    print("C. INTEGRATED INTERPRETATION & POLICY RECOMMENDATIONS".center(80))
    print("=" * 80)
    print()

    print("1. PRODUCTIVITY GROWTH DRIVERS:")
    print("-" * 80)

    if agg["pct_from_tc"] > 50:
        print("   • Technical change is the main driver")
        print("   • Industry experiencing strong innovation/technology adoption")
        print("   → Policy: Support R&D, technology diffusion programs")
    elif agg["pct_from_te"] > 50:
        print("   • Efficiency change (catch-up) is the main driver")
        print("   • Firms improving management/operations")
        print("   → Policy: Best practice sharing, management training")
    else:
        print("   • Mixed contributions from TC and TE")
        print("   → Policy: Balanced approach to innovation and efficiency")

    print()
    print("2. FIRM CHARACTERISTICS & EFFICIENCY:")
    print("-" * 80)

    for _, row in me.iterrows():
        var = row["variable"]
        effect = row["marginal_effect"]
        sig = (
            "***"
            if row["p_value"] < 0.01
            else "**" if row["p_value"] < 0.05 else "*" if row["p_value"] < 0.1 else ""
        )

        if abs(effect) > 0.01 and sig:
            direction = "reduces" if effect < 0 else "increases"
            print(f"   • {var}: {direction} inefficiency (ME = {effect:.4f}{sig})")

            if var == "firm_age" and effect > 0:
                print("     → Older firms less efficient - consider succession planning")
            elif var == "manager_education" and effect < 0:
                print("     → Education improves efficiency - invest in human capital")
            elif var == "export_share" and effect < 0:
                print("     → Exporters more efficient - support export promotion")

    print()
    print("3. SCALE EFFECTS:")
    print("-" * 80)

    rts = params[0] + params[1]
    if rts < 1.0 and agg["mean_delta_se"] < 0:
        print(f"   • Decreasing returns to scale (RTS = {rts:.3f})")
        print("   • Negative scale effects from expansion")
        print("   → Policy: Caution against excessive firm size")
    elif rts > 1.0 and agg["mean_delta_se"] > 0:
        print(f"   • Increasing returns to scale (RTS = {rts:.3f})")
        print("   • Positive scale effects from expansion")
        print("   → Policy: Support firm growth and consolidation")
    else:
        print(f"   • Approximately constant returns (RTS = {rts:.3f})")
        print("   • Limited scale effects")

    print()
    print("=" * 80)

    # ========================================================================
    # D. VISUALIZATIONS
    # ========================================================================

    print()
    print("Generating visualizations...")

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. TFP decomposition bar chart
    ax1 = fig.add_subplot(gs[0, :2])
    decomp_sorted = decomp.nlargest(20, "delta_tfp")
    x = np.arange(len(decomp_sorted))

    ax1.bar(x, decomp_sorted["delta_tc"], label="Technical Change", color="#3498db")
    ax1.bar(
        x,
        decomp_sorted["delta_te"],
        bottom=decomp_sorted["delta_tc"],
        label="Efficiency Change",
        color="#2ecc71",
    )
    ax1.bar(
        x,
        decomp_sorted["delta_se"],
        bottom=decomp_sorted["delta_tc"] + decomp_sorted["delta_te"],
        label="Scale Effect",
        color="#f39c12",
    )
    ax1.plot(x, decomp_sorted["delta_tfp"], "ko-", linewidth=2, label="Total TFP")
    ax1.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax1.set_xlabel("Firm (ranked by TFP growth)")
    ax1.set_ylabel("Growth Rate")
    ax1.set_title("TFP Decomposition (Top 20 Firms)")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # 2. TC vs TE scatter
    ax2 = fig.add_subplot(gs[0, 2])
    scatter = ax2.scatter(
        decomp["delta_tc"],
        decomp["delta_te"],
        c=decomp["delta_tfp"],
        cmap="RdYlGn",
        s=50,
        alpha=0.6,
        edgecolors="k",
        linewidth=0.5,
    )
    ax2.axhline(0, color="k", linestyle="--", alpha=0.3)
    ax2.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax2.set_xlabel("Technical Change")
    ax2.set_ylabel("Efficiency Change")
    ax2.set_title("TC vs TE")
    plt.colorbar(scatter, ax=ax2, label="Total TFP")

    # 3. Marginal effects
    ax3 = fig.add_subplot(gs[1, 0])
    colors = ["green" if x < 0 else "red" for x in me["marginal_effect"]]
    ax3.barh(me["variable"], me["marginal_effect"], color=colors, alpha=0.7)
    ax3.axvline(0, color="k", linestyle="-", linewidth=0.8)
    ax3.set_xlabel("Marginal Effect on Inefficiency")
    ax3.set_title("Determinants of Inefficiency")
    ax3.grid(axis="x", alpha=0.3)

    # 4. TFP distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(decomp["delta_tfp"], bins=25, alpha=0.7, edgecolor="black")
    ax4.axvline(
        decomp["delta_tfp"].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean = {decomp['delta_tfp'].mean():.3f}",
    )
    ax4.set_xlabel("TFP Growth")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Distribution of TFP Growth")
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 5. Efficiency distribution
    ax5 = fig.add_subplot(gs[1, 2])
    eff_2015 = data[data["year"] == 2015]["true_efficiency"]
    eff_2019 = data[data["year"] == 2019]["true_efficiency"]
    ax5.hist(eff_2015, bins=20, alpha=0.5, label="2015", edgecolor="black")
    ax5.hist(eff_2019, bins=20, alpha=0.5, label="2019", edgecolor="black")
    ax5.set_xlabel("Technical Efficiency")
    ax5.set_ylabel("Frequency")
    ax5.set_title("Efficiency Distribution")
    ax5.legend()
    ax5.grid(alpha=0.3)

    plt.suptitle("Integrated Productivity Analysis Dashboard", fontsize=16, fontweight="bold")

    # Save figure
    output_path = "productivity_analysis_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"   ✓ Dashboard saved to: {output_path}")

    plt.show()

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
