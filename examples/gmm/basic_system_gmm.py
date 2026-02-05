"""
Basic System GMM Example
=========================

Example demonstrating System GMM estimation (Blundell-Bond, 1998).

System GMM combines difference and level equations to improve
efficiency, especially for persistent series.

Author: PanelBox Development Team
Date: January 2026
"""

import os
import sys

import numpy as np
import pandas as pd

# Add panelbox to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from panelbox.gmm import DifferenceGMM, SystemGMM


def generate_persistent_series(n_groups=100, n_periods=10, seed=42):
    """
    Generate synthetic panel data with persistent series.

    Data Generating Process:
    y_{it} = 0.8 * y_{i,t-1} + 0.2 * x_{it} + η_i + ε_{it}

    High persistence (γ = 0.8) makes System GMM advantageous over Difference GMM.

    Parameters
    ----------
    n_groups : int
        Number of cross-sectional units
    n_periods : int
        Number of time periods
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Synthetic panel data
    """
    np.random.seed(seed)

    # Parameters (high persistence)
    gamma = 0.8  # AR(1) coefficient - highly persistent
    beta = 0.2  # Effect of x on y

    data = []

    for i in range(1, n_groups + 1):
        # Fixed effect
        eta_i = np.random.normal(0, 1)

        # Exogenous variable
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

    print(f"Generated persistent panel data:")
    print(f"  - N = {n_groups} groups")
    print(f"  - T = {n_periods} periods")
    print(f"  - Total observations = {len(df)}")
    print(f"  - True parameters: gamma={gamma} (persistent!), beta={beta}")
    print()

    return df


def main():
    """Run System GMM example and compare with Difference GMM."""
    print("=" * 70)
    print("System GMM vs Difference GMM Comparison")
    print("=" * 70)
    print()

    # Generate persistent series
    data = generate_persistent_series(n_groups=100, n_periods=10)

    # ========================================================================
    # Model 1: Difference GMM (Arellano-Bond)
    # ========================================================================
    print("=" * 70)
    print("Model 1: Difference GMM (Arellano-Bond, 1991)")
    print("=" * 70)
    print()

    diff_gmm = DifferenceGMM(
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

    diff_results = diff_gmm.fit()
    print(diff_results.summary())
    print()

    # ========================================================================
    # Model 2: System GMM (Blundell-Bond)
    # ========================================================================
    print("=" * 70)
    print("Model 2: System GMM (Blundell-Bond, 1998)")
    print("=" * 70)
    print()

    sys_gmm = SystemGMM(
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
        level_instruments={"max_lags": 1},
    )

    sys_results = sys_gmm.fit()
    print(sys_results.summary())
    print()

    # ========================================================================
    # Comparison
    # ========================================================================
    print("=" * 70)
    print("Model Comparison")
    print("=" * 70)
    print()

    comparison = pd.DataFrame(
        {
            "True": [0.8, 0.2],
            "Difference GMM": [diff_results.params["L1.y"], diff_results.params["x"]],
            "System GMM": [sys_results.params["L1.y"], sys_results.params["x"]],
            "Diff GMM SE": [diff_results.std_errors["L1.y"], diff_results.std_errors["x"]],
            "Sys GMM SE": [sys_results.std_errors["L1.y"], sys_results.std_errors["x"]],
        },
        index=["L1.y (γ)", "x (β)"],
    )

    print(comparison.to_string())
    print()

    # Calculate relative efficiency
    se_ratio_gamma = diff_results.std_errors["L1.y"] / sys_results.std_errors["L1.y"]
    se_ratio_beta = diff_results.std_errors["x"] / sys_results.std_errors["x"]

    print("Relative Efficiency (Diff GMM SE / Sys GMM SE):")
    print(f"  γ (L1.y): {se_ratio_gamma:.3f}x")
    print(f"  β (x):    {se_ratio_beta:.3f}x")
    print()

    if se_ratio_gamma > 1.1:
        print("✓ System GMM is MORE EFFICIENT for the AR coefficient")
        print("  (expected for persistent series)")
    else:
        print("⚠ Efficiency gains are modest")
    print()

    # Diagnostics comparison
    print("=" * 70)
    print("Diagnostic Tests Comparison")
    print("=" * 70)
    print()

    diagnostics = pd.DataFrame(
        {
            "Difference GMM": [
                diff_results.n_instruments,
                diff_results.instrument_ratio,
                diff_results.hansen_j.pvalue,
                diff_results.sargan.pvalue,
                (
                    diff_results.ar2_test.pvalue
                    if diff_results.ar2_test.pvalue is not np.nan
                    else "N/A"
                ),
            ],
            "System GMM": [
                sys_results.n_instruments,
                sys_results.instrument_ratio,
                sys_results.hansen_j.pvalue,
                sys_results.sargan.pvalue,
                sys_results.ar2_test.pvalue if sys_results.ar2_test.pvalue is not np.nan else "N/A",
            ],
        },
        index=[
            "Instruments",
            "Instrument Ratio",
            "Hansen J p-value",
            "Sargan p-value",
            "AR(2) p-value",
        ],
    )

    print(diagnostics.to_string())
    print()

    print("Note: System GMM uses additional level instruments,")
    print("      which increases instrument count but improves precision")
    print("      for persistent series.")
    print()

    print("=" * 70)
    print("✓ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
