"""
Test bootstrap performance for SFA models

Requirements:
- 999 bootstrap replications for N=100, T=10 should complete in < 120s
"""

import time

import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.bootstrap import SFABootstrap


def generate_panel_data(n=100, t=10, seed=42):
    """Generate simulated panel data for testing"""
    np.random.seed(seed)

    # Create panel structure
    entities = np.repeat(np.arange(1, n + 1), t)
    times = np.tile(np.arange(1, t + 1), n)

    # Generate covariates
    x1 = np.random.normal(5, 1, n * t)
    x2 = np.random.normal(3, 0.5, n * t)

    # True parameters
    beta0 = 2.0
    beta1 = 0.5
    beta2 = 0.3
    sigma_v = 0.1
    sigma_u = 0.2

    # Generate error components
    v = np.random.normal(0, sigma_v, n * t)  # noise
    u = np.abs(np.random.normal(0, sigma_u, n * t))  # inefficiency (half-normal)

    # Generate output
    y = beta0 + beta1 * x1 + beta2 * x2 + v - u

    df = pd.DataFrame({"entity": entities, "time": times, "y": y, "x1": x1, "x2": x2})

    return df


def test_bootstrap_performance_999_replications():
    """Test that 999 bootstrap replications complete within 120 seconds"""
    print("\n" + "=" * 70)
    print("Bootstrap Performance Test")
    print("=" * 70)

    # Generate data: N=100, T=10
    df = generate_panel_data(n=100, t=10)
    print(f"Dataset: N=100, T=10, total obs={len(df)}")

    # Estimate model
    print("Estimating SFA model...")
    sf = StochasticFrontier(
        data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="half_normal"
    )

    result = sf.fit(method="mle")
    print(f"  Log-likelihood: {result.loglik:.4f}")
    print(f"  σ_v: {result.sigma_v:.4f}")
    print(f"  σ_u: {result.sigma_u:.4f}")

    # Bootstrap test
    print("\nRunning bootstrap with 999 replications...")
    start_time = time.time()

    bootstrap = SFABootstrap(
        result,
        n_boot=999,
        method="parametric",
        n_jobs=4,  # parallel
        seed=42,
    )

    boot_results = bootstrap.bootstrap_parameters()
    elapsed = time.time() - start_time

    print(f"\n✓ Bootstrap completed in {elapsed:.2f} seconds")
    print("  Target: < 120 seconds")
    print(f"  Performance: {999 / elapsed:.1f} bootstrap/second")
    print(f"  Speedup: {elapsed / 120:.2f}× (< 1.0 is good)")

    # Check results
    assert boot_results is not None, "Bootstrap returned None"
    boot_result = boot_results["results_df"]
    assert "ci_lower" in boot_result.columns, "No CI lower bound"
    assert "ci_upper" in boot_result.columns, "No CI upper bound"

    # Performance criterion
    assert elapsed < 120, f"Bootstrap too slow: {elapsed:.2f}s > 120s"

    print(f"\n✓ PASSED: Performance criterion met ({elapsed:.2f}s < 120s)")

    return boot_result, elapsed


def test_bootstrap_performance_100_replications():
    """Quick test with 100 replications for CI validation"""
    print("\n" + "=" * 70)
    print("Bootstrap CI Validation Test (100 reps)")
    print("=" * 70)

    # Generate data with known parameters
    n, t = 50, 5
    df = generate_panel_data(n=n, t=t)

    # Estimate
    sf = StochasticFrontier(
        data=df, depvar="y", exog=["x1", "x2"], frontier="production", dist="half_normal"
    )

    result = sf.fit(method="mle")

    # Bootstrap
    print("Running bootstrap (100 reps for quick test)...")
    boot_result = result.bootstrap(
        n_boot=100, method="parametric", ci_level=0.95, n_jobs=2, seed=42
    )

    print("\nBootstrap CI Results:")
    print(boot_result)

    # Check that CIs are reasonable
    for param in ["sigma_v", "sigma_u"]:
        if param in boot_result.index:
            ci_width = boot_result.loc[param, "ci_upper"] - boot_result.loc[param, "ci_lower"]
            point_est = boot_result.loc[param, "estimate"]
            print(
                f"  {param}: {point_est:.4f} [{boot_result.loc[param, 'ci_lower']:.4f}, {boot_result.loc[param, 'ci_upper']:.4f}] (width: {ci_width:.4f})"
            )

            # CI should not be degenerate
            assert ci_width > 0, f"{param} CI is degenerate"
            assert ci_width < 2 * point_est, f"{param} CI is unreasonably wide"

    print("\n✓ PASSED: CIs are reasonable")

    return boot_result


if __name__ == "__main__":
    # Run performance test
    try:
        boot_res, elapsed = test_bootstrap_performance_999_replications()
        print(f"\n{'=' * 70}")
        print(f"PERFORMANCE TEST: {'PASSED' if elapsed < 120 else 'FAILED'}")
        print(f"{'=' * 70}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()

    # Run CI validation test
    try:
        ci_res = test_bootstrap_performance_100_replications()
        print(f"\n{'=' * 70}")
        print("CI VALIDATION TEST: PASSED")
        print(f"{'=' * 70}")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
