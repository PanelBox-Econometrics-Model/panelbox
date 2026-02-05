"""
Basic Difference GMM Example
=============================

Simple example demonstrating Difference GMM estimation with synthetic data.

This example:
1. Generates synthetic panel data
2. Estimates a Difference GMM model
3. Displays results

Author: PanelBox Development Team
Date: January 2026
"""

import os
import sys

import numpy as np
import pandas as pd

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from panelbox.gmm import DifferenceGMM


def generate_synthetic_data(n_groups=100, n_periods=10, seed=42):
    """
    Generate synthetic panel data for GMM estimation.

    Data Generating Process:
    y_{it} = 0.5 * y_{i,t-1} + 0.3 * x_{it} + η_i + ε_{it}

    Parameters
    ----------
    n_groups : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Synthetic panel data
    """
    np.random.seed(seed)

    # Parameters
    gamma = 0.5  # AR(1) coefficient
    beta = 0.3  # Effect of x on y

    # Generate panel structure
    data = []

    for i in range(1, n_groups + 1):
        # Fixed effect
        eta_i = np.random.normal(0, 1)

        # Exogenous variable (varying over time)
        x_it = np.random.normal(2, 0.5, n_periods)

        # Initialize y
        y_it = np.zeros(n_periods)
        y_it[0] = eta_i + np.random.normal(0, 0.5)

        # Generate y recursively
        for t in range(1, n_periods):
            epsilon_it = np.random.normal(0, 0.5)
            y_it[t] = gamma * y_it[t - 1] + beta * x_it[t] + eta_i + epsilon_it

        # Store data
        for t in range(n_periods):
            data.append({"id": i, "year": t + 1, "y": y_it[t], "x": x_it[t]})

    df = pd.DataFrame(data)

    print(f"Generated synthetic panel data:")
    print(f"  - N = {n_groups} groups")
    print(f"  - T = {n_periods} periods")
    print(f"  - Total observations = {len(df)}")
    print(f"  - True parameters: gamma={gamma}, beta={beta}")
    print()

    return df


def main():
    """Run basic Difference GMM example."""
    print("=" * 70)
    print("Basic Difference GMM Example")
    print("=" * 70)
    print()

    # Generate synthetic data
    data = generate_synthetic_data(n_groups=100, n_periods=10)

    # Estimate Difference GMM
    print("Estimating Difference GMM...")
    print()

    model = DifferenceGMM(
        data=data,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )

    # Fit the model
    results = model.fit()

    # Display results
    print(results.summary())
    print()

    # Check key diagnostics
    print("=" * 70)
    print("Key Diagnostics")
    print("=" * 70)
    print()
    print(f"Instrument ratio: {results.instrument_ratio:.3f}")
    print(f"  (should be < 1.0 to avoid instrument proliferation)")
    print()
    print(f"Hansen J-test p-value: {results.hansen_j.pvalue:.4f}")
    print(f"  (0.10 < p < 0.25 indicates valid instruments)")
    print()
    print(f"AR(2) test p-value: {results.ar2_test.pvalue:.4f}")
    print(f"  (should be > 0.10 for valid moment conditions)")
    print()

    # Compare with true parameters
    print("=" * 70)
    print("Parameter Comparison (True vs Estimated)")
    print("=" * 70)
    print()
    print(f"{'Parameter':<15} {'True':>10} {'Estimated':>12} {'Std.Err':>10}")
    print("-" * 50)
    print(
        f"{'L1.y':<15} {0.5:>10.3f} {results.params['L1.y']:>12.3f} "
        f"{results.std_errors['L1.y']:>10.3f}"
    )
    print(
        f"{'x':<15} {0.3:>10.3f} {results.params['x']:>12.3f} " f"{results.std_errors['x']:>10.3f}"
    )
    print()

    print("✓ Example completed successfully!")


if __name__ == "__main__":
    main()
