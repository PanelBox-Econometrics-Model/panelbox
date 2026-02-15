#!/usr/bin/env python
"""
Quick test for censored models implementation.
"""

import warnings

import numpy as np

from panelbox.models.censored import HonoreTrimmedEstimator, PooledTobit, RandomEffectsTobit


def test_tobit_basic():
    """Test basic Tobit functionality."""
    print("\n=== Testing Random Effects Tobit ===")

    np.random.seed(42)

    # Generate simple test data
    N, T, K = 30, 5, 2
    beta_true = np.array([0.5, -0.3])
    sigma_eps = 0.4
    sigma_alpha = 0.3

    # Generate panel data
    X = np.random.randn(N * T, K)
    groups = np.repeat(np.arange(N), T)

    # Random effects
    alpha_i = np.random.normal(0, sigma_alpha, N)
    alpha = np.repeat(alpha_i, T)

    # Generate outcome with censoring
    y_star = X @ beta_true + alpha + np.random.normal(0, sigma_eps, N * T)
    y = np.maximum(0, y_star)  # Left censoring at 0

    print(f"Data: N={N}, T={T}, K={K}")
    print(f"Censoring rate: {100 * np.mean(y == 0):.1f}%")

    # Fit Random Effects Tobit
    model = RandomEffectsTobit(
        endog=y, exog=X, groups=groups, censoring_point=0, quadrature_points=8
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(maxiter=50, options={"disp": False})

    print("\nRandom Effects Tobit Results:")
    print(f"β estimates: {result.beta}")
    print(f"True β:      {beta_true}")
    print(f"σ_ε estimate: {result.sigma_eps:.3f} (true: {sigma_eps})")
    print(f"σ_α estimate: {result.sigma_alpha:.3f} (true: {sigma_alpha})")
    print(f"Converged: {result.converged}")

    # Test predictions
    y_pred = result.predict(pred_type="censored")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")

    return result


def test_pooled_tobit():
    """Test Pooled Tobit."""
    print("\n=== Testing Pooled Tobit ===")

    np.random.seed(123)

    N, K = 100, 2
    beta_true = np.array([0.6, -0.4])
    sigma_true = 0.5

    X = np.random.randn(N, K)
    y_star = X @ beta_true + np.random.normal(0, sigma_true, N)
    y = np.maximum(0, y_star)

    print(f"Data: N={N}, K={K}")
    print(f"Censoring rate: {100 * np.mean(y == 0):.1f}%")

    model = PooledTobit(endog=y, exog=X, censoring_point=0)

    result = model.fit(maxiter=100, options={"disp": False})

    print("\nPooled Tobit Results:")
    print(f"β estimates: {result.beta}")
    print(f"True β:      {beta_true}")
    print(f"σ estimate: {result.sigma:.3f} (true: {sigma_true})")
    print(f"Converged: {result.converged}")

    return result


def test_honore():
    """Test Honoré estimator (basic)."""
    print("\n=== Testing Honoré Trimmed Estimator ===")

    np.random.seed(456)

    # Small dataset for quick testing
    N, T, K = 10, 3, 2
    beta_true = np.array([0.4, -0.2])

    X = np.random.randn(N * T, K)
    groups = np.repeat(np.arange(N), T)
    time = np.tile(np.arange(T), N)

    # Fixed effects
    alpha_i = np.random.randn(N)
    alpha = np.repeat(alpha_i, T)

    # Generate outcome
    y_star = X @ beta_true + alpha + 0.3 * np.random.randn(N * T)
    y = np.maximum(0, y_star)

    print(f"Data: N={N}, T={T}, K={K}")
    print(f"Censoring rate: {100 * np.mean(y == 0):.1f}%")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = HonoreTrimmedEstimator(endog=y, exog=X, groups=groups, time=time, censoring_point=0)

        results = model.fit(maxiter=20, verbose=False)

    print("\nHonoré Estimator Results:")
    print(f"β estimates: {results.params}")
    print(f"True β:      {beta_true}")
    print(f"Converged: {results.converged}")
    print(f"Observations trimmed: {results.n_trimmed}")

    return results


if __name__ == "__main__":
    print("Testing Censored Models Implementation")
    print("=" * 50)

    try:
        # Test Random Effects Tobit
        re_tobit = test_tobit_basic()

        # Test Pooled Tobit
        pooled_tobit = test_pooled_tobit()

        # Test Honoré
        honore = test_honore()

        print("\n" + "=" * 50)
        print("All tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
