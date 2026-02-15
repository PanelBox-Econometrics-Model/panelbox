"""
Tutorial completo de Econometria Espacial com PanelBox.
Executar este script para testar funcionalidades espaciais.
"""

import numpy as np
import pandas as pd

import panelbox as pb
from panelbox import PanelExperiment, SpatialWeights
from panelbox.experiment.spatial_extension import extend_panel_experiment

# Ensure spatial methods are available
extend_panel_experiment()


def create_spatial_data():
    """Create simulated spatial panel data."""
    np.random.seed(42)

    # Panel dimensions
    n_states = 48
    n_years = 20
    years = range(2000, 2020)

    # Create panel structure
    panel_data = []
    for state_id in range(n_states):
        for year in years:
            income = 40000 + state_id * 500 + year * 100 + np.random.randn() * 5000
            population = 1e6 + state_id * 1e4 + (year - 2000) * 5000 + np.random.randn() * 1e5
            unemployment = 5 + np.random.randn() * 2
            price = (
                0.5 * income
                + 0.00001 * population
                - 1000 * unemployment
                + np.random.randn() * 10000
            )

            panel_data.append(
                {
                    "state": f"state_{state_id:02d}",
                    "year": year,
                    "price": price,
                    "income": income,
                    "population": population,
                    "unemployment": unemployment,
                }
            )

    data = pd.DataFrame(panel_data)

    # Create spatial weight matrix
    W_matrix = np.zeros((n_states, n_states))
    for i in range(n_states):
        if i > 0:
            W_matrix[i, i - 1] = 1
        if i < n_states - 1:
            W_matrix[i, i + 1] = 1

    W_matrix = np.maximum(W_matrix, W_matrix.T)
    W = SpatialWeights(W_matrix)
    W = W.standardize("row")

    return data, W


def main():
    print("=" * 80)
    print("PANELBOX SPATIAL ECONOMETRICS TUTORIAL")
    print("=" * 80)

    # Create data
    print("\n1. Creating spatial panel data...")
    data, W = create_spatial_data()
    print(f"   Panel: {data['state'].nunique()} states × {data['year'].nunique()} years")
    print(f"   Total observations: {len(data)}")

    # Create experiment
    print("\n2. Setting up PanelExperiment...")
    experiment = PanelExperiment(
        data=data,
        formula="price ~ income + population + unemployment",
        entity_col="state",
        time_col="year",
    )

    # Estimate OLS
    print("\n3. Estimating baseline OLS...")
    ols_result = experiment.fit_model("pooled_ols", name="OLS")

    # Spatial diagnostics
    print("\n4. Running spatial diagnostics...")
    diagnostics = experiment.run_spatial_diagnostics(W, "OLS")

    print(f"   Moran's I: {diagnostics['moran']['statistic']:.4f}")
    print(f"   P-value: {diagnostics['moran']['pvalue']:.6f}")

    if diagnostics["moran"]["pvalue"] < 0.05:
        print("   ⚠ Significant spatial autocorrelation detected!")

    print(f"\n   Recommendation: {diagnostics['recommendation']}")

    # Estimate spatial models
    print("\n5. Estimating spatial models...")

    print("   - SAR-FE...")
    sar = experiment.add_spatial_model("SAR-FE", W, "sar", effects="fixed")
    print(f"     ρ = {sar.rho:.4f} (p = {sar.rho_pvalue:.4f})")

    print("   - SEM-FE...")
    sem = experiment.add_spatial_model("SEM-FE", W, "sem", effects="fixed")
    print(f"     λ = {sem.lambda_:.4f} (p = {sem.lambda_pvalue:.4f})")

    print("   - SDM-FE...")
    sdm = experiment.add_spatial_model("SDM-FE", W, "sdm", effects="fixed")
    print(f"     ρ = {sdm.rho:.4f}")

    # Compare models
    print("\n6. Comparing models...")
    comparison = experiment.compare_spatial_models()
    print("\n", comparison[["Model", "Type", "AIC", "BIC"]].to_string(index=False))

    # Effects decomposition
    print("\n7. Decomposing spatial effects (SDM)...")
    effects = experiment.decompose_spatial_effects("SDM-FE")

    print("\n   Effects Decomposition:")
    for var in effects["direct"].index:
        print(f"\n   {var}:")
        print(f"     Direct:   {effects['direct'][var]:.6f}")
        print(f"     Indirect: {effects['indirect'][var]:.6f}")
        print(f"     Total:    {effects['total'][var]:.6f}")

    print("\n" + "=" * 80)
    print("TUTORIAL COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
