"""
Validation script for panel unit root tests.

Compares PanelBox results with R's plm package.
"""

import numpy as np
import pandas as pd

from panelbox.diagnostics.unit_root import breitung_test, hadri_test, panel_unit_root_test


def generate_stationary_panel(n_entities=10, n_time=100, rho=0.6, seed=42):
    """
    Generate stationary AR(1) panel data.

    Parameters
    ----------
    rho : float
        AR(1) coefficient (must be < 1 for stationarity).
    """
    np.random.seed(seed)
    data = []

    for i in range(n_entities):
        y = np.zeros(n_time)
        y[0] = np.random.randn()

        for t in range(1, n_time):
            y[t] = rho * y[t - 1] + np.random.randn()

        for t in range(n_time):
            data.append({"entity": i, "time": t, "y": y[t]})

    return pd.DataFrame(data)


def generate_unit_root_panel(n_entities=10, n_time=100, seed=123):
    """Generate panel data with unit roots (random walks)."""
    np.random.seed(seed)
    data = []

    for i in range(n_entities):
        y = np.random.randn(n_time).cumsum()

        for t in range(n_time):
            data.append({"entity": i, "time": t, "y": y[t]})

    return pd.DataFrame(data)


def main():
    print("=" * 80)
    print("Panel Unit Root Test Validation (Python/PanelBox)")
    print("=" * 80)
    print()

    # ========================================================================
    # TEST 1: Stationary Panel Data
    # ========================================================================
    print("TEST 1: Stationary Panel Data (ρ = 0.6)")
    print("-" * 80)

    df_stat = generate_stationary_panel(n_entities=10, n_time=100, rho=0.6)

    print("\nHadri Test (H0: Stationarity):")
    hadri_stat = hadri_test(
        df_stat, "y", entity_col="entity", time_col="time", trend="c", robust=True
    )
    print(f"  Statistic: {hadri_stat.statistic:.4f}")
    print(f"  P-value:   {hadri_stat.pvalue:.4f}")
    print(f"  Decision:  {'REJECT' if hadri_stat.reject else 'FAIL TO REJECT'} H0")

    print("\nBreitung Test (H0: Unit Root):")
    breitung_stat = breitung_test(df_stat, "y", entity_col="entity", time_col="time", trend="ct")
    print(f"  Statistic: {breitung_stat.statistic:.4f}")
    print(f"  P-value:   {breitung_stat.pvalue:.4f}")
    print(f"  Decision:  {'REJECT' if breitung_stat.reject else 'FAIL TO REJECT'} H0")

    print("\nUnified Test Summary:")
    result_stat = panel_unit_root_test(
        df_stat, "y", entity_col="entity", time_col="time", test="all", trend="c"
    )
    print(result_stat.summary_table())

    print()
    print("=" * 80)
    print()

    # ========================================================================
    # TEST 2: Unit Root Panel Data
    # ========================================================================
    print("TEST 2: Unit Root Panel Data (Random Walk)")
    print("-" * 80)

    df_ur = generate_unit_root_panel(n_entities=10, n_time=100)

    print("\nHadri Test (H0: Stationarity):")
    hadri_ur = hadri_test(df_ur, "y", entity_col="entity", time_col="time", trend="c", robust=True)
    print(f"  Statistic: {hadri_ur.statistic:.4f}")
    print(f"  P-value:   {hadri_ur.pvalue:.4f}")
    print(f"  Decision:  {'REJECT' if hadri_ur.reject else 'FAIL TO REJECT'} H0")

    print("\nBreitung Test (H0: Unit Root):")
    breitung_ur = breitung_test(df_ur, "y", entity_col="entity", time_col="time", trend="ct")
    print(f"  Statistic: {breitung_ur.statistic:.4f}")
    print(f"  P-value:   {breitung_ur.pvalue:.4f}")
    print(f"  Decision:  {'REJECT' if breitung_ur.reject else 'FAIL TO REJECT'} H0")

    print("\nUnified Test Summary:")
    result_ur = panel_unit_root_test(
        df_ur, "y", entity_col="entity", time_col="time", test="all", trend="c"
    )
    print(result_ur.summary_table())

    print()
    print("=" * 80)
    print()

    # ========================================================================
    # VALIDATION SUMMARY
    # ========================================================================
    print("=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print()
    print("Expected Results:")
    print("  Stationary Data:")
    print("    - Hadri: Should NOT reject H0 (p-value > 0.05)")
    print("    - Breitung: Should REJECT H0 (p-value < 0.05)")
    print()
    print("  Unit Root Data:")
    print("    - Hadri: Should REJECT H0 (p-value < 0.05)")
    print("    - Breitung: Should NOT reject H0 (p-value > 0.05)")
    print()
    print("Actual Results:")
    print("  Stationary Data:")
    print(
        f"    - Hadri p-value:    {hadri_stat.pvalue:.4f} "
        f"({'✓' if hadri_stat.pvalue > 0.05 else '✗'})"
    )
    print(
        f"    - Breitung p-value: {breitung_stat.pvalue:.4f} "
        f"({'✓' if breitung_stat.pvalue < 0.05 else '✗'})"
    )
    print()
    print("  Unit Root Data:")
    print(
        f"    - Hadri p-value:    {hadri_ur.pvalue:.4f} ({'✓' if hadri_ur.pvalue < 0.05 else '✗'})"
    )
    print(
        f"    - Breitung p-value: {breitung_ur.pvalue:.4f} "
        f"({'✓' if breitung_ur.pvalue > 0.05 else '✗'})"
    )
    print()
    print("=" * 80)
    print()
    print("Note: Compare these results with R output from unit_root_r.R")
    print("Test statistics should match within ±0.01 tolerance.")
    print()


if __name__ == "__main__":
    main()
