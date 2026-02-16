"""Debug script for SAR RE implementation."""

from pathlib import Path

import numpy as np
import pandas as pd

from panelbox.models.spatial import SpatialLag

FIXTURES_PATH = Path(__file__).parent / "fixtures"


def test_sar_re_debug():
    """Debug SAR RE estimation."""

    # Load data
    df = pd.read_csv(FIXTURES_PATH / "spatial_test_data.csv")
    W = np.loadtxt(FIXTURES_PATH / "spatial_weights.csv", delimiter=",")

    print(f"\nData shape: {df.shape}")
    print(f"W shape: {W.shape}")
    print(f"\nData columns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head(10))

    print(f"\nData statistics:")
    print(df.describe())

    print(f"\nUnique entities: {df['entity'].nunique()}")
    print(f"Unique time periods: {df['time'].nunique()}")

    # Create model
    model = SpatialLag(
        formula="y ~ x1 + x2 + x3", data=df, entity_col="entity", time_col="time", W=W
    )

    print(f"\nModel attributes:")
    print(f"  n_entities: {model.n_entities}")
    print(f"  n_periods: {model.n_periods}")
    print(f"  endog shape: {model.endog.shape}")
    print(f"  exog shape: {model.exog.shape}")

    # Fit model
    print(f"\n{'='*60}")
    print("Fitting SAR RE model...")
    print(f"{'='*60}")

    result = model.fit(effects="random", method="ml", verbose=True, maxiter=200)

    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Rho: {result.rho if hasattr(result, 'rho') else result.params.get('rho', 'N/A')}")
    print(f"Log-likelihood: {result.llf}")
    print(f"Convergence: {result.convergence_info}")

    if hasattr(result, "variance_components"):
        print(f"\nVariance components:")
        print(f"  sigma_alpha2: {result.variance_components['sigma_alpha2']}")
        print(f"  sigma_epsilon2: {result.variance_components['sigma_epsilon2']}")
        print(f"  theta: {result.variance_components.get('theta', 'N/A')}")

    print(f"\nParameters:")
    print(result.params)

    # Compare with R results
    import json

    with open(FIXTURES_PATH / "r_sar_re_results.json", "r") as f:
        r_results = json.load(f)

    print(f"\n{'='*60}")
    print("Comparison with R:")
    print(f"{'='*60}")

    py_rho = result.rho if hasattr(result, "rho") else result.params.get("rho", np.nan)
    r_rho = r_results["sar_re"]["rho"]
    print(f"Rho: Python={py_rho:.6f}, R={r_rho:.6f}, diff={py_rho - r_rho:.6f}")

    py_llf = result.llf
    r_llf = r_results["sar_re"]["logLik"]
    print(f"Log-likelihood: Python={py_llf:.2f}, R={r_llf:.2f}, diff={py_llf - r_llf:.2f}")

    if hasattr(result, "variance_components"):
        py_sigma_alpha2 = result.variance_components["sigma_alpha2"]
        r_sigma_alpha2 = r_results["sar_re"]["sigma_alpha2"]
        print(f"sigma_alpha2: Python={py_sigma_alpha2:.6f}, R={r_sigma_alpha2:.6f}")

        py_sigma_eps2 = result.variance_components["sigma_epsilon2"]
        r_sigma_eps2 = r_results["sar_re"]["sigma_epsilon2"]
        print(f"sigma_epsilon2: Python={py_sigma_eps2:.6f}, R={r_sigma_eps2:.6f}")


if __name__ == "__main__":
    test_sar_re_debug()
