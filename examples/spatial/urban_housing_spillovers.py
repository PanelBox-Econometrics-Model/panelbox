"""
Example: Urban Housing Spillovers
==================================

This example demonstrates spatial spillover effects in urban housing markets.
We analyze how house prices in one neighborhood affect prices in adjacent neighborhoods.

Key questions:
- Do house price changes spill over to neighboring areas?
- What percentage of price changes is due to spillover effects?
- How should housing policies account for spatial dependencies?

Dataset: Simulated Baltimore-like housing market data (50 neighborhoods, 10 years)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox import PanelExperiment, SpatialWeights
from panelbox.experiment.spatial_extension import extend_panel_experiment

# Ensure spatial extension is loaded
extend_panel_experiment()

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
pd.options.display.float_format = "{:.2f}".format


def generate_housing_data():
    """
    Generate synthetic housing market data with spatial structure.

    Simulates 50 neighborhoods over 10 years with:
    - Crime rates affecting prices
    - School quality as amenity
    - Distance to CBD
    - Spatial spillovers in prices
    """
    np.random.seed(123)

    n_neighborhoods = 50
    n_years = 10
    years = range(2010, 2020)

    # Create neighborhood grid (approximate Baltimore structure)
    neighborhoods = []
    for i in range(n_neighborhoods):
        neighborhoods.append(
            {
                "id": i,
                "name": f"Neighborhood_{i:02d}",
                "distance_cbd": np.sqrt(i) * 2,  # Distance to Central Business District
                "lat": 39.3 + (i % 10) * 0.01,
                "lon": -76.6 + (i // 10) * 0.01,
            }
        )

    nbh_df = pd.DataFrame(neighborhoods)

    # Create spatial weight matrix (contiguity-based)
    W = np.zeros((n_neighborhoods, n_neighborhoods))
    for i in range(n_neighborhoods):
        # Connect to adjacent neighborhoods in grid
        if i % 10 != 0:  # Not leftmost
            W[i, i - 1] = 1
        if i % 10 != 9:  # Not rightmost
            W[i, i + 1] = 1
        if i >= 10:  # Not top row
            W[i, i - 10] = 1
        if i < 40:  # Not bottom row
            W[i, i + 10] = 1

    # Make symmetric and row-standardize
    W = np.maximum(W, W.T)
    W = W / W.sum(axis=1, keepdims=True)
    W[np.isnan(W)] = 0

    # Generate panel data
    panel_data = []

    # Spatial correlation parameter
    rho_true = 0.35  # Moderate spatial spillover

    for year in years:
        # Annual shocks
        year_effect = (year - 2010) * 5000  # General appreciation

        # Generate neighborhood characteristics for this year
        crime_rate = np.random.exponential(5, n_neighborhoods)  # Crime per 1000 residents
        school_quality = np.random.normal(70, 10, n_neighborhoods)  # Test scores
        unemployment = np.random.normal(5, 2, n_neighborhoods)  # Unemployment rate

        # Price equation with spatial lag
        # price = rho * W @ price + X @ beta + epsilon
        X = np.column_stack(
            [
                -crime_rate * 2000,  # Crime reduces prices
                school_quality * 500,  # Good schools increase prices
                -nbh_df["distance_cbd"].values * 3000,  # CBD proximity premium
                -unemployment * 1000,  # Economic conditions
            ]
        )

        beta = np.array([1, 1, 1, 1])
        epsilon = np.random.normal(0, 10000, n_neighborhoods)

        # Solve for equilibrium prices: (I - rho*W)^{-1} @ (X @ beta + epsilon)
        I = np.eye(n_neighborhoods)
        A_inv = np.linalg.inv(I - rho_true * W)
        prices = A_inv @ (X @ beta + epsilon + year_effect + 200000)  # Base price 200k

        # Add to panel
        for i in range(n_neighborhoods):
            panel_data.append(
                {
                    "neighborhood": nbh_df.iloc[i]["name"],
                    "year": year,
                    "price": prices[i],
                    "crime_rate": crime_rate[i],
                    "school_quality": school_quality[i],
                    "unemployment": unemployment[i],
                    "distance_cbd": nbh_df.iloc[i]["distance_cbd"],
                }
            )

    return pd.DataFrame(panel_data), W, nbh_df


def main():
    print("=" * 80)
    print("URBAN HOUSING SPILLOVERS ANALYSIS")
    print("Spatial Panel Models for Real Estate Markets")
    print("=" * 80)

    # Generate data
    print("\n1. Loading Baltimore housing market data...")
    data, W_array, neighborhoods = generate_housing_data()

    print(f"   Dataset: {neighborhoods.shape[0]} neighborhoods × {data['year'].nunique()} years")
    print(f"   Total observations: {len(data)}")
    print(f"\n   Summary statistics:")
    print(data[["price", "crime_rate", "school_quality", "unemployment"]].describe())

    # Create spatial weights object
    W = SpatialWeights(W_array)
    W = W.standardize("row")
    print(f"\n   Spatial structure: Average {W.mean_neighbors:.1f} neighbors per neighborhood")

    # Initialize experiment
    print("\n2. Setting up spatial panel analysis...")
    experiment = PanelExperiment(
        data=data,
        formula="price ~ crime_rate + school_quality + unemployment + distance_cbd",
        entity_col="neighborhood",
        time_col="year",
    )

    # Baseline OLS
    print("\n3. Estimating baseline model (ignoring spatial effects)...")
    ols = experiment.fit_model("pooled_ols", name="OLS")
    print("\n   OLS Results (biased if spatial correlation exists):")
    print(ols.summary())

    # Spatial diagnostics
    print("\n4. Testing for spatial autocorrelation...")
    diagnostics = experiment.run_spatial_diagnostics(W, "OLS")

    print(f"\n   Moran's I Test:")
    print(f"   Statistic: {diagnostics['moran']['statistic']:.4f}")
    print(f"   P-value: {diagnostics['moran']['pvalue']:.6f}")

    if diagnostics["moran"]["pvalue"] < 0.01:
        print("   *** Strong evidence of spatial spillovers in housing prices ***")

    print(f"\n   Model recommendation: {diagnostics['recommendation']}")

    # Estimate spatial models
    print("\n5. Estimating spatial models...")

    # SAR - Direct spillover in prices
    print("\n   SAR Model (price spillovers)...")
    sar = experiment.add_spatial_model("SAR", W, "sar", effects="fixed")
    print(f"   Spatial lag (ρ): {sar.rho:.3f} (p={sar.rho_pvalue:.4f})")
    print(f"   → A 1% increase in neighboring prices leads to {sar.rho:.1%} increase locally")

    # SDM - Allow for spatially lagged variables too
    print("\n   SDM Model (price and characteristic spillovers)...")
    sdm = experiment.add_spatial_model("SDM", W, "sdm", effects="fixed")

    # Model comparison
    print("\n6. Model Comparison:")
    comparison = experiment.compare_spatial_models()
    print(comparison[["Model", "AIC", "BIC", "ρ"]].to_string(index=False))

    # Effects decomposition
    print("\n7. Decomposing Spatial Effects (SDM):")
    print("-" * 60)
    effects = experiment.decompose_spatial_effects("SDM")

    for var in ["crime_rate", "school_quality", "unemployment"]:
        direct = effects["direct"][var]
        indirect = effects["indirect"][var]
        total = effects["total"][var]
        spillover_pct = (indirect / total * 100) if total != 0 else 0

        print(f"\n   {var.upper()}:")
        print(f"   Direct effect:   ${direct:,.0f}")
        print(f"   Spillover effect: ${indirect:,.0f}")
        print(f"   Total effect:     ${total:,.0f}")
        print(f"   → {abs(spillover_pct):.0f}% of impact spills over to neighbors")

    # Policy implications
    print("\n" + "=" * 80)
    print("POLICY IMPLICATIONS")
    print("=" * 80)

    print("\n1. Crime Reduction Programs:")
    crime_total = abs(effects["total"]["crime_rate"])
    crime_spillover = abs(effects["indirect"]["crime_rate"])
    print(f"   - Reducing crime by 1 per 1000 residents increases prices by ${crime_total:,.0f}")
    print(f"   - ${crime_spillover:,.0f} of this benefit spills over to adjacent neighborhoods")
    print(f"   - Implication: Crime reduction has regional benefits beyond target area")

    print("\n2. School Quality Improvements:")
    school_total = effects["total"]["school_quality"]
    school_spillover = effects["indirect"]["school_quality"]
    print(f"   - 10-point improvement in test scores increases prices by ${school_total*10:,.0f}")
    print(f"   - ${school_spillover*10:,.0f} benefits neighboring areas")
    print(f"   - Implication: School investments create positive regional spillovers")

    print("\n3. Spatial Multiplier Effect:")
    multiplier = 1 / (1 - sdm.rho)
    print(f"   - Spatial multiplier: {multiplier:.2f}")
    print(f"   - Any local shock is amplified {multiplier:.2f}x through spatial feedback")
    print(f"   - Implication: Local policies have larger regional impacts than expected")

    print("\n4. Targeted vs. Regional Policies:")
    print("   - High spillovers (>30%) suggest regional coordination is beneficial")
    print("   - Targeted neighborhood interventions may have limited local impact")
    print("   - Regional approaches internalize positive externalities")

    # Visualizations
    print("\n8. Creating visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Price trends by neighborhood group
    ax1 = axes[0, 0]
    for i in range(0, 50, 10):
        nbh_name = f"Neighborhood_{i:02d}"
        nbh_data = data[data["neighborhood"] == nbh_name]
        ax1.plot(
            nbh_data["year"], nbh_data["price"] / 1000, marker="o", label=f"Nbh {i}", alpha=0.7
        )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Price ($1000s)")
    ax1.set_title("Housing Price Trends")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spatial weight matrix structure
    ax2 = axes[0, 1]
    im = ax2.imshow(W_array, cmap="Blues", aspect="equal")
    ax2.set_title("Spatial Weight Matrix")
    ax2.set_xlabel("Neighborhood ID")
    ax2.set_ylabel("Neighborhood ID")
    plt.colorbar(im, ax=ax2, label="Weight")

    # Plot 3: Effects decomposition
    ax3 = axes[1, 0]
    vars = list(effects["direct"].index)
    x_pos = np.arange(len(vars))
    width = 0.25

    ax3.bar(x_pos - width, effects["direct"].values, width, label="Direct", color="#2ecc71")
    ax3.bar(x_pos, effects["indirect"].values, width, label="Indirect", color="#f39c12")
    ax3.bar(x_pos + width, effects["total"].values, width, label="Total", color="#3498db")

    ax3.set_xlabel("Variable")
    ax3.set_ylabel("Effect on Price ($)")
    ax3.set_title("Spatial Effects Decomposition")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(vars, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    # Plot 4: Spillover percentages
    ax4 = axes[1, 1]
    spillover_pcts = []
    for var in vars:
        if effects["total"][var] != 0:
            pct = abs(effects["indirect"][var] / effects["total"][var] * 100)
        else:
            pct = 0
        spillover_pcts.append(pct)

    bars = ax4.bar(
        vars, spillover_pcts, color=["#e74c3c" if p > 30 else "#95a5a6" for p in spillover_pcts]
    )
    ax4.set_ylabel("Spillover (%)")
    ax4.set_title("Percentage of Effect That Spills Over")
    ax4.axhline(y=30, color="r", linestyle="--", alpha=0.5, label="30% threshold")
    ax4.set_xticklabels(vars, rotation=45, ha="right")
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, pct in zip(bars, spillover_pcts):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0, height, f"{pct:.0f}%", ha="center", va="bottom"
        )

    plt.tight_layout()
    plt.savefig("housing_spillovers_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n   Plots saved to: housing_spillovers_analysis.png")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("1. Significant spatial spillovers exist in urban housing markets")
    print("2. Crime and school quality have substantial neighborhood spillover effects")
    print("3. Ignoring spatial effects underestimates total policy impacts")
    print("4. Regional coordination can improve policy effectiveness")


if __name__ == "__main__":
    main()
