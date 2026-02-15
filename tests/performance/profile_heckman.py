"""
Profiling script for Panel Heckman to identify bottlenecks.
"""

import cProfile
import io
import pstats

import numpy as np
import pandas as pd


def generate_heckman_data(n=200, t=10):
    """Generate synthetic data for Heckman estimation."""
    np.random.seed(42)

    # Panel structure
    ids = np.repeat(np.arange(n), t)
    times = np.tile(np.arange(t), n)

    # Correlated errors
    rho = 0.6
    cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
    errors = np.random.multivariate_normal([0, 0], cov_matrix, size=n * t)
    u = errors[:, 0]  # Selection error
    eps = errors[:, 1]  # Outcome error

    # Selection equation
    z1 = np.random.randn(n * t)
    z2 = np.random.randn(n * t)
    z_star = 0.5 * z1 - 0.3 * z2 + u
    selected = (z_star > 0).astype(int)

    # Outcome equation (observed only if selected)
    x1 = np.random.randn(n * t)
    x2 = np.random.randn(n * t)
    y_star = 1.0 * x1 + 0.5 * x2 + eps
    y = np.where(selected == 1, y_star, np.nan)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "entity": ids,
            "time": times,
            "selected": selected,
            "y": y,
            "x1": x1,
            "x2": x2,
            "z1": z1,
            "z2": z2,
        }
    )

    return df


def profile_heckman_twostep():
    """Profile Heckman two-step."""
    print("Profiling Panel Heckman Two-Step...")

    # Generate data
    data = generate_heckman_data(n=200, t=10)

    # Setup profiler
    profiler = cProfile.Profile()

    # Profile
    profiler.enable()

    from panelbox.models.selection import PanelHeckman

    model = PanelHeckman(
        selection_formula="selected ~ z1 + z2",
        outcome_formula="y ~ x1 + x2",
        entity_col="entity",
        time_col="time",
    )
    result = model.fit(data, method="two-step")

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print(s.getvalue())
    print(f"\nTwo-step rho: {result.rho:.4f}")

    return s.getvalue()


def profile_heckman_mle():
    """Profile Heckman MLE with quadrature."""
    print("\n" + "=" * 80)
    print("Profiling Panel Heckman MLE...")

    # Generate smaller data for MLE
    data = generate_heckman_data(n=100, t=10)

    # Setup profiler
    profiler = cProfile.Profile()

    # Profile
    profiler.enable()

    from panelbox.models.selection import PanelHeckman

    model = PanelHeckman(
        selection_formula="selected ~ z1 + z2",
        outcome_formula="y ~ x1 + x2",
        entity_col="entity",
        time_col="time",
    )
    result = model.fit(data, method="mle", quadrature_points=10, maxiter=50)

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(20)

    print(s.getvalue())
    print(f"\nMLE converged: {result.converged}")
    print(f"MLE rho: {result.rho:.4f}")

    return s.getvalue()


if __name__ == "__main__":
    # Run profiling
    twostep_stats = profile_heckman_twostep()
    mle_stats = profile_heckman_mle()

    # Save to file
    with open("heckman_profiling_results.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HECKMAN TWO-STEP PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(twostep_stats)
        f.write("\n\n")
        f.write("=" * 80 + "\n")
        f.write("HECKMAN MLE PROFILING RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(mle_stats)

    print("\n" + "=" * 80)
    print("Profiling results saved to heckman_profiling_results.txt")
