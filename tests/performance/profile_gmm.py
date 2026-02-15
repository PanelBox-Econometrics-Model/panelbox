"""
Profiling script for GMM methods to identify bottlenecks.
"""

import cProfile
import io
import pstats

import numpy as np
import pandas as pd


def generate_gmm_data(n=500, t=10, k=3, n_instruments=5):
    """Generate synthetic data for GMM estimation."""
    np.random.seed(42)

    # Panel structure
    ids = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)

    # Generate instruments
    z = np.random.randn(n * t, n_instruments)

    # Generate endogenous variable with correlation
    epsilon = np.random.randn(n * t)
    x_endo = z @ np.random.randn(n_instruments) + 0.5 * epsilon + np.random.randn(n * t)

    # Generate exogenous variables
    x_exo = np.random.randn(n * t, k - 1)

    # Combine
    X = np.column_stack([x_endo, x_exo])

    # Generate outcome
    beta_true = np.array([1.0, 0.5, -0.3])
    y = X @ beta_true + epsilon

    # Create DataFrame
    df = pd.DataFrame(
        {
            "entity": ids,
            "time": times,
            "y": y,
            "x1": X[:, 0],
            "x2": X[:, 1],
            "x3": X[:, 2],
        }
    )

    # Add instruments
    for i in range(n_instruments):
        df[f"z{i+1}"] = z[:, i]

    return df


def profile_cue_gmm():
    """Profile CUE-GMM estimation."""
    print("Profiling CUE-GMM...")

    # Generate data
    data = generate_gmm_data(n=500, t=10)

    # Setup profiler
    profiler = cProfile.Profile()

    # Profile CUE-GMM
    profiler.enable()

    from panelbox.gmm.cue_gmm import ContinuousUpdatedGMM

    model = ContinuousUpdatedGMM(
        dependent="y", exog=["x2", "x3"], endog=["x1"], instruments=["z1", "z2", "z3", "z4", "z5"]
    )
    result = model.fit(data, maxiter=100)

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)  # Top 20 functions

    print(s.getvalue())
    print(f"\nCUE-GMM converged: {result.converged}")
    print(f"Iterations: {result.iterations}")

    return s.getvalue()


def profile_bias_corrected():
    """Profile Bias-Corrected GMM."""
    print("\n" + "=" * 80)
    print("Profiling Bias-Corrected GMM...")

    # Generate data
    data = generate_gmm_data(n=200, t=20)

    # Setup profiler
    profiler = cProfile.Profile()

    # Profile BC-GMM
    profiler.enable()

    from panelbox.gmm.bias_corrected import BiasCorrectedGMM

    model = BiasCorrectedGMM(
        dependent="y", exog=["x2", "x3"], endog=["x1"], instruments=["z1", "z2", "z3", "z4", "z5"]
    )
    result = model.fit(data)

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print(s.getvalue())
    print(f"\nBC-GMM converged: {result.converged}")

    return s.getvalue()


if __name__ == "__main__":
    # Run profiling
    cue_stats = profile_cue_gmm()
    bc_stats = profile_bias_corrected()

    # Save to file
    with open("gmm_profiling_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CUE-GMM PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(cue_stats)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("BIAS-CORRECTED GMM PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(bc_stats)

    print("\n" + "=" * 80)
    print("Profiling results saved to gmm_profiling_results.txt")
