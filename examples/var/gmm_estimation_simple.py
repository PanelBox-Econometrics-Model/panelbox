"""
Simple Panel VAR GMM Estimation Example

This example demonstrates basic GMM estimation for Panel VAR using the
low-level panelbox.var.gmm API.

NOTE: This uses the low-level GMM estimation function. The high-level
PanelVAR API does not yet include GMM support.

Author: PanelBox Team
Date: 2025-02-12
"""

import numpy as np
import pandas as pd

from panelbox.var.gmm import estimate_panel_var_gmm


def generate_simple_panel_data(N=50, T=20, seed=42):
    """
    Generate simulated panel VAR(1) data for demonstration.

    The DGP is:
    y1_t = 0.5*y1_{t-1} + 0.2*y2_{t-1} + alpha_i + u1_t
    y2_t = 0.1*y1_{t-1} + 0.6*y2_{t-1} + alpha_i + u2_t

    Parameters
    ----------
    N : int
        Number of entities (cross-sectional units)
    T : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Panel data with columns: entity, time, y1, y2
    """
    np.random.seed(seed)

    # True VAR(1) coefficients
    A = np.array([[0.5, 0.2], [0.1, 0.6]])  # y1 equation  # y2 equation

    data_list = []

    for i in range(N):
        # Entity-specific fixed effect
        alpha_i = np.random.randn(2) * 0.5

        # Initialize
        y = np.zeros((T, 2))
        y[0] = alpha_i + np.random.randn(2) * 0.5

        # Generate VAR(1) process
        for t in range(1, T):
            y[t] = alpha_i + A @ y[t - 1] + np.random.randn(2) * 0.3

        # Store as rows
        for t in range(T):
            data_list.append({"entity": i, "time": t, "y1": y[t, 0], "y2": y[t, 1]})

    return pd.DataFrame(data_list)


def main():
    """Run basic GMM estimation example."""
    print("=" * 80)
    print("Simple Panel VAR GMM Estimation Example")
    print("=" * 80)

    # Generate data
    print("\n[1] Generating data...")
    print("-" * 60)
    data = generate_simple_panel_data(N=50, T=20, seed=42)
    print(f"Data shape: {data.shape}")
    print(f"Entities: {data['entity'].nunique()}")
    print(f"Time periods per entity: ~{data.groupby('entity').size().mean():.0f}")

    # Estimate with GMM
    print("\n[2] Estimating VAR(1) with GMM...")
    print("-" * 60)
    print("Settings:")
    print("  - Transformation: Forward Orthogonal Deviations (FOD)")
    print("  - GMM Step: Two-step with Windmeijer correction")
    print("  - Instruments: Collapsed")
    print("  - Max instruments: 10")

    result = estimate_panel_var_gmm(
        data=data,
        var_lags=1,
        value_cols=["y1", "y2"],
        entity_col="entity",
        time_col="time",
        transform="fod",
        gmm_step="two-step",
        instrument_type="collapsed",
        max_instruments=10,
        windmeijer_correction=True,
    )

    # Display results
    print("\n[3] Estimation Results")
    print("-" * 60)
    print(f"GMM Step: {result.gmm_step}")
    print(f"Transform: {result.transform}")
    print(f"Instrument Type: {result.instrument_type}")
    print(f"Windmeijer Corrected: {result.windmeijer_corrected}")
    print()
    print(f"Number of Observations: {result.n_obs}")
    print(f"Number of Entities: {result.n_entities}")
    print(f"Number of Instruments: {result.n_instruments}")
    print(f"Instrument/Entity Ratio: {result.n_instruments / result.n_entities:.2f}")

    # Display coefficients
    print("\n[4] Estimated Coefficients")
    print("-" * 60)
    print(f"Shape: {result.coefficients.shape} (rows=params per eq, cols=equations)")
    print(f"\nCoefficient matrix:")
    print(result.coefficients)

    print(f"\nStandard Errors:")
    print(result.standard_errors)

    # Compute t-statistics
    t_stats = result.coefficients / result.standard_errors
    print(f"\nt-statistics:")
    print(t_stats)

    # Interpretation guide
    print("\n[5] Interpretation Guide")
    print("-" * 60)
    print("The coefficient matrix has shape (K*p, K) where:")
    print("  K = number of variables (2 in this case: y1, y2)")
    print("  p = number of lags (1 in this case)")
    print()
    print("Each column represents one equation:")
    print("  Column 0: y1 equation coefficients")
    print("  Column 1: y2 equation coefficients")
    print()
    print("Each row represents one regressor:")
    print("  Row 0: Lag 1 of y1")
    print("  Row 1: Lag 1 of y2")
    print()
    print("Example interpretation:")
    print(f"  y1 equation - effect of y1_{{t-1}}: {result.coefficients[0, 0]:.4f}")
    print(f"  y1 equation - effect of y2_{{t-1}}: {result.coefficients[1, 0]:.4f}")
    print(f"  y2 equation - effect of y1_{{t-1}}: {result.coefficients[0, 1]:.4f}")
    print(f"  y2 equation - effect of y2_{{t-1}}: {result.coefficients[1, 1]:.4f}")

    # Compare with true values
    print("\n[6] Comparison with True DGP")
    print("-" * 60)
    true_A = np.array([[0.5, 0.2], [0.1, 0.6]])
    print("True coefficient matrix (transposed to match output):")
    print(true_A.T)  # Transpose to match (K*p, K) format

    print("\nEstimated vs True:")
    print(
        f"  y1←y1_{{t-1}}: {result.coefficients[0,0]:.4f} vs 0.5000 (error: {abs(result.coefficients[0,0]-0.5):.4f})"
    )
    print(
        f"  y1←y2_{{t-1}}: {result.coefficients[1,0]:.4f} vs 0.2000 (error: {abs(result.coefficients[1,0]-0.2):.4f})"
    )
    print(
        f"  y2←y1_{{t-1}}: {result.coefficients[0,1]:.4f} vs 0.1000 (error: {abs(result.coefficients[0,1]-0.1):.4f})"
    )
    print(
        f"  y2←y2_{{t-1}}: {result.coefficients[1,1]:.4f} vs 0.6000 (error: {abs(result.coefficients[1,1]-0.6):.4f})"
    )

    # Check instrument proliferation
    print("\n[7] Instrument Diagnostics")
    print("-" * 60)
    if result.n_instruments > result.n_entities:
        print(f"⚠  WARNING: Instrument proliferation detected!")
        print(f"   #Instruments ({result.n_instruments}) > #Entities ({result.n_entities})")
        print(f"   Recommendation: Use 'collapsed' instruments or reduce max_instruments")
    else:
        print(f"✓  OK: #Instruments ({result.n_instruments}) ≤ #Entities ({result.n_entities})")
        print(f"   Instrument count is acceptable")

    if result.n_instruments / result.n_entities > 1.0:
        print(f"⚠  Ratio is high ({result.n_instruments / result.n_entities:.2f})")
    elif result.n_instruments / result.n_entities > 0.5:
        print(f"✓  Ratio is moderate ({result.n_instruments / result.n_entities:.2f})")
    else:
        print(f"✓  Ratio is conservative ({result.n_instruments / result.n_entities:.2f})")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Try with different sample sizes (N, T)")
    print("  2. Compare 'fod' vs 'fd' transformations")
    print("  3. Compare 'one-step' vs 'two-step' GMM")
    print("  4. Experiment with different instrument settings")
    print("  5. Add more variables (K > 2) or lags (p > 1)")

    return result


if __name__ == "__main__":
    result = main()
