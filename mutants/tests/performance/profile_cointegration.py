"""
Profiling script for Cointegration tests to identify bottlenecks.
"""

import cProfile
import io
import pstats

import numpy as np
import pandas as pd


def generate_cointegration_data(n=50, t=30):
    """Generate panel data with cointegration."""
    np.random.seed(42)

    data_list = []
    for i in range(n):
        # Random walk for X
        x = np.cumsum(np.random.randn(t))

        # Cointegrated Y with I(0) error
        y = 0.5 * x + np.random.randn(t) * 0.5

        df_i = pd.DataFrame({"entity": i, "time": np.arange(t), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


def profile_westerlund():
    """Profile Westerlund test with bootstrap."""
    print("Profiling Westerlund Test...")

    # Generate data
    data = generate_cointegration_data(n=50, t=30)

    # Setup profiler
    profiler = cProfile.Profile()

    # Profile
    profiler.enable()

    from panelbox.diagnostics.cointegration import westerlund_test

    result = westerlund_test(
        data,
        "y",
        ["x"],
        entity_col="entity",
        time_col="time",
        test_type="gt",
        bootstrap=True,
        n_bootstrap=500,  # Reduced for profiling
    )

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print(s.getvalue())
    print(f"\nWesterlund statistic: {result['statistic']:.4f}")
    print(f"Bootstrap p-value: {result['pvalue']:.4f}")

    return s.getvalue()


def profile_pedroni():
    """Profile Pedroni test."""
    print("\n" + "=" * 80)
    print("Profiling Pedroni Test...")

    # Generate data
    data = generate_cointegration_data(n=50, t=30)

    # Setup profiler
    profiler = cProfile.Profile()

    # Profile
    profiler.enable()

    from panelbox.diagnostics.cointegration import pedroni_test

    result = pedroni_test(
        data, "y", ["x"], entity_col="entity", time_col="time", test_type="panel_v"
    )

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print(s.getvalue())
    print(f"\nPedroni statistic: {result['statistic']:.4f}")

    return s.getvalue()


if __name__ == "__main__":
    # Run profiling
    westerlund_stats = profile_westerlund()
    pedroni_stats = profile_pedroni()

    # Save to file
    with open("cointegration_profiling_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("WESTERLUND TEST PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(westerlund_stats)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("PEDRONI TEST PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(pedroni_stats)

    print("\n" + "=" * 80)
    print("Profiling results saved to cointegration_profiling_results.txt")
