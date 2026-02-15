"""
Basic example of spatial panel models in PanelBox.

This script demonstrates:
1. Creating spatial weight matrices
2. Estimating Spatial Lag (SAR) models
3. Estimating Spatial Error (SEM) models
4. Interpreting spillover effects
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.models.spatial import SpatialError, SpatialLag, SpatialWeights


def generate_example_data():
    """Generate example spatial panel data."""
    np.random.seed(42)

    # Parameters
    N = 30  # Number of regions
    T = 15  # Number of time periods

    # Create a simple spatial weight matrix (chain structure)
    # Each region is connected to its neighbors
    W_raw = np.zeros((N, N))
    for i in range(N):
        if i > 0:
            W_raw[i, i - 1] = 1
        if i < N - 1:
            W_raw[i, i + 1] = 1

    # Create panel data
    data_list = []
    for t in range(T):
        for i in range(N):
            # Generate covariates
            population = np.random.uniform(5, 20) + np.random.randn()
            income = np.random.uniform(20, 50) + np.random.randn()

            # Add spatial correlation in the dependent variable
            # (this simulates spatial spillovers)
            neighbors_effect = 0
            if i > 0:
                neighbors_effect += 0.3 * np.random.randn()
            if i < N - 1:
                neighbors_effect += 0.3 * np.random.randn()

            # Dependent variable (e.g., housing prices)
            price = 10 + 0.5 * population + 0.3 * income + neighbors_effect + np.random.randn()

            data_list.append(
                {"region": i, "time": t, "price": price, "population": population, "income": income}
            )

    data = pd.DataFrame(data_list)

    return data, W_raw


def main():
    """Run spatial panel model examples."""
    print("=" * 70)
    print("PanelBox Spatial Panel Models Example")
    print("=" * 70)

    # Generate example data
    print("\n1. Generating example spatial panel data...")
    data, W_raw = generate_example_data()
    print(f"   - Number of regions: {len(data['region'].unique())}")
    print(f"   - Number of periods: {len(data['time'].unique())}")
    print(f"   - Total observations: {len(data)}")

    # Create spatial weights object
    print("\n2. Creating spatial weight matrix...")
    W = SpatialWeights(W_raw)
    print(f"   - Matrix dimension: {W.n}×{W.n}")
    print(f"   - Sum of weights (before normalization): {W.s0:.2f}")

    # Row-standardize the weight matrix
    W.standardize("row")
    print(f"   - Row-standardized: Each row now sums to 1")
    print(f"   - Spatial coefficient bounds: {W.get_bounds()}")

    # Print summary statistics
    W.summary()

    # =========================================================================
    # Spatial Lag Model (SAR)
    # =========================================================================
    print("\n" + "=" * 70)
    print("3. Spatial Lag Model (SAR)")
    print("=" * 70)
    print("\nModel: y = ρWy + Xβ + α + ε")
    print("where ρ captures spatial spillovers in the dependent variable")

    # Estimate SAR with fixed effects
    sar_model = SpatialLag(
        endog=data["price"],
        exog=data[["population", "income"]],
        W=W,
        entity_id=data["region"],
        time_id=data["time"],
    )

    print("\nEstimating SAR with fixed effects...")
    sar_result = sar_model.fit(effects="fixed", method="qml", verbose=True)

    # Print results
    print("\nSAR Estimation Results:")
    sar_result.summary()

    # Interpret spatial parameter
    rho = sar_result.params["rho"]
    print(f"\nSpatial autoregressive parameter (ρ): {rho:.4f}")
    if rho > 0:
        print("   → Positive spatial spillovers: higher prices in neighboring regions")
        print("     are associated with higher prices in a given region")
    elif rho < 0:
        print("   → Negative spatial spillovers: higher prices in neighboring regions")
        print("     are associated with lower prices in a given region")
    else:
        print("   → No significant spatial spillovers detected")

    # Spillover effects
    if hasattr(sar_result, "spillover_effects"):
        print("\nSpillover Effects:")
        for var, effects in sar_result.spillover_effects.items():
            if var != "rho":
                print(f"\n{var}:")
                print(f"   Direct effect:   {effects['direct']:.4f}")
                print(f"   Indirect effect: {effects['indirect']:.4f}")
                print(f"   Total effect:    {effects['total']:.4f}")
                print(f"   % Indirect:      {100*effects['indirect']/effects['total']:.1f}%")

    # =========================================================================
    # Spatial Error Model (SEM)
    # =========================================================================
    print("\n" + "=" * 70)
    print("4. Spatial Error Model (SEM)")
    print("=" * 70)
    print("\nModel: y = Xβ + α + u, where u = λWu + ε")
    print("where λ captures spatial correlation in the error term")

    # Estimate SEM with fixed effects
    sem_model = SpatialError(
        endog=data["price"],
        exog=data[["population", "income"]],
        W=W,
        entity_id=data["region"],
        time_id=data["time"],
    )

    print("\nEstimating SEM with GMM...")
    sem_result = sem_model.fit(effects="fixed", method="gmm", n_lags=2, verbose=True)

    # Print results
    print("\nSEM Estimation Results:")
    sem_result.summary()

    # Interpret spatial error parameter
    lambda_param = sem_result.params["lambda"]
    print(f"\nSpatial error parameter (λ): {lambda_param:.4f}")
    if lambda_param > 0:
        print("   → Positive spatial error correlation: unobserved shocks")
        print("     affecting one region also affect neighboring regions")
    elif lambda_param < 0:
        print("   → Negative spatial error correlation")
    else:
        print("   → No significant spatial error correlation")

    # =========================================================================
    # Model Comparison
    # =========================================================================
    print("\n" + "=" * 70)
    print("5. Model Comparison")
    print("=" * 70)

    # Compare information criteria
    print("\nInformation Criteria:")
    print(f"SAR - AIC: {sar_result.aic:.2f}, BIC: {sar_result.bic:.2f}")
    print(f"SEM - AIC: {sem_result.aic:.2f}, BIC: {sem_result.bic:.2f}")

    if sar_result.aic < sem_result.aic:
        print("\n→ SAR model preferred by AIC")
    else:
        print("\n→ SEM model preferred by AIC")

    # Compare coefficient estimates
    print("\nCoefficient Comparison:")
    print(f"{'Variable':<15} {'SAR':<10} {'SEM':<10} {'Difference':<10}")
    print("-" * 45)
    for var in ["population", "income"]:
        sar_coef = sar_result.params[var]
        sem_coef = sem_result.params[var]
        diff = sar_coef - sem_coef
        print(f"{var:<15} {sar_coef:<10.4f} {sem_coef:<10.4f} {diff:<10.4f}")

    # =========================================================================
    # Predictions
    # =========================================================================
    print("\n" + "=" * 70)
    print("6. Generating Predictions")
    print("=" * 70)

    # Generate predictions from SAR model
    sar_predictions = sar_model.predict()
    print(f"\nSAR predictions generated: {len(sar_predictions)} values")

    # Generate predictions from SEM model
    sem_predictions = sem_model.predict()
    print(f"SEM predictions generated: {len(sem_predictions)} values")

    # Compare predictions
    correlation = np.corrcoef(sar_predictions, sem_predictions)[0, 1]
    print(f"\nCorrelation between SAR and SEM predictions: {correlation:.4f}")

    # Plot actual vs predicted for one time period
    t = 0  # First time period
    idx = data["time"] == t
    actual = data.loc[idx, "price"].values
    sar_pred_t = sar_predictions[idx]
    sem_pred_t = sem_predictions[idx]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(actual, sar_pred_t, alpha=0.6)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("SAR Predicted Price")
    plt.title(f"SAR Model Predictions (t={t})")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(actual, sem_pred_t, alpha=0.6)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
    plt.xlabel("Actual Price")
    plt.ylabel("SEM Predicted Price")
    plt.title(f"SEM Model Predictions (t={t})")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spatial_predictions.png", dpi=150, bbox_inches="tight")
    print("\nPrediction plots saved to 'spatial_predictions.png'")

    # =========================================================================
    # Conclusion
    # =========================================================================
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"1. Spatial spillovers (SAR ρ): {sar_result.params['rho']:.4f}")
    print(f"2. Spatial error correlation (SEM λ): {sem_result.params['lambda']:.4f}")
    print(f"3. Both models show significant spatial dependence")
    print(
        f"4. Model selection: {'SAR' if sar_result.aic < sem_result.aic else 'SEM'} preferred by AIC"
    )

    return data, W, sar_result, sem_result


if __name__ == "__main__":
    data, W, sar_result, sem_result = main()
