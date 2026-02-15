#!/usr/bin/env python
"""
Quick test for ordered choice models and their marginal effects.
"""

import warnings

import numpy as np
import pandas as pd

from panelbox.marginal_effects.discrete_me import compute_ordered_ame, compute_ordered_mem
from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit


def test_ordered_logit():
    """Test Ordered Logit model."""
    print("\n=== Testing Ordered Logit ===")

    np.random.seed(42)

    # Generate data
    N, K = 200, 3
    J = 4  # Categories: 0, 1, 2, 3

    beta_true = np.array([0.5, -0.3, 0.2])
    cutpoints_true = np.array([-1, 0, 1])

    X = np.random.randn(N, K)
    linear_pred = X @ beta_true

    # Generate ordered outcomes
    y = np.zeros(N, dtype=int)
    for i in range(N):
        y_star = linear_pred[i] + np.random.logistic(0, 1)
        if y_star <= cutpoints_true[0]:
            y[i] = 0
        elif y_star <= cutpoints_true[1]:
            y[i] = 1
        elif y_star <= cutpoints_true[2]:
            y[i] = 2
        else:
            y[i] = 3

    groups = np.arange(N)

    print(f"Data: N={N}, K={K}, Categories={J}")
    print(f"Category distribution: {np.bincount(y)}")

    # Fit model
    model = OrderedLogit(endog=y, exog=X, groups=groups, n_categories=J)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(maxiter=100, options={"disp": False})

    print("\nOrdered Logit Results:")
    print(f"β estimates: {result.beta}")
    print(f"True β:      {beta_true}")
    print(f"Cutpoints:   {result.cutpoints}")
    print(f"True κ:      {cutpoints_true}")
    print(f"Converged:   {result.converged}")

    # Test predictions
    probs = result.predict_proba()
    print(f"Probability shape: {probs.shape}")
    print(
        f"Sum of probabilities: min={probs.sum(axis=1).min():.4f}, max={probs.sum(axis=1).max():.4f}"
    )

    return result


def test_ordered_probit():
    """Test Ordered Probit model."""
    print("\n=== Testing Ordered Probit ===")

    np.random.seed(123)

    N, K = 150, 2
    J = 3

    beta_true = np.array([0.4, -0.2])
    cutpoints_true = np.array([-0.5, 0.5])

    X = np.random.randn(N, K)
    linear_pred = X @ beta_true

    # Generate with normal errors
    y = np.zeros(N, dtype=int)
    for i in range(N):
        y_star = linear_pred[i] + np.random.normal(0, 1)
        if y_star <= cutpoints_true[0]:
            y[i] = 0
        elif y_star <= cutpoints_true[1]:
            y[i] = 1
        else:
            y[i] = 2

    groups = np.arange(N)

    print(f"Data: N={N}, K={K}, Categories={J}")
    print(f"Category distribution: {np.bincount(y)}")

    # Fit model
    model = OrderedProbit(endog=y, exog=X, groups=groups)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(maxiter=100, options={"disp": False})

    print("\nOrdered Probit Results:")
    print(f"β estimates: {result.beta}")
    print(f"True β:      {beta_true}")
    print(f"Cutpoints:   {result.cutpoints}")
    print(f"True κ:      {cutpoints_true}")
    print(f"Converged:   {result.converged}")

    return result


def test_marginal_effects():
    """Test marginal effects for ordered models."""
    print("\n=== Testing Marginal Effects ===")

    np.random.seed(456)

    # Create simple ordered logit model
    N, K = 100, 2
    J = 3

    X = np.random.randn(N, K)
    y = np.random.randint(0, J, N)
    groups = np.arange(N)

    model = OrderedLogit(endog=y, exog=X, groups=groups)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(maxiter=50, options={"disp": False})

    print(f"Fitted model with {J} categories and {K} variables")

    # Compute AME
    print("\nComputing Average Marginal Effects (AME)...")
    ame = compute_ordered_ame(result)

    print("AME Results:")
    print(ame.marginal_effects)

    # Check sum-to-zero property
    print(f"\nSum-to-zero property verified: {ame.verify_sum_to_zero()}")

    # Show sums for each variable (should be close to 0)
    sums = ame.marginal_effects.sum(axis=0)
    print("Sums across categories for each variable:")
    for i, s in enumerate(sums):
        print(f"  Variable {i}: {s:.6f}")

    # Compute MEM
    print("\nComputing Marginal Effects at Means (MEM)...")
    mem = compute_ordered_mem(result)

    print("MEM Results:")
    print(mem.marginal_effects)

    print(f"MEM sum-to-zero property verified: {mem.verify_sum_to_zero()}")

    return ame, mem


def test_random_effects_ordered():
    """Test Random Effects Ordered Logit."""
    print("\n=== Testing Random Effects Ordered Logit ===")

    np.random.seed(789)

    # Panel data
    N, T, K = 20, 4, 2
    J = 3

    beta_true = np.array([0.3, -0.2])
    cutpoints_true = np.array([-0.3, 0.3])
    sigma_alpha_true = 0.4

    X = np.random.randn(N * T, K)
    groups = np.repeat(np.arange(N), T)

    # Random effects
    alpha_i = np.random.normal(0, sigma_alpha_true, N)
    alpha = np.repeat(alpha_i, T)

    # Generate outcomes
    y = np.zeros(N * T, dtype=int)
    for i in range(N * T):
        y_star = X[i] @ beta_true + alpha[i] + np.random.logistic(0, 1)
        if y_star <= cutpoints_true[0]:
            y[i] = 0
        elif y_star <= cutpoints_true[1]:
            y[i] = 1
        else:
            y[i] = 2

    print(f"Panel data: N={N}, T={T}, K={K}, Categories={J}")
    print(f"Category distribution: {np.bincount(y)}")

    # Fit RE model
    model = RandomEffectsOrderedLogit(
        endog=y, exog=X, groups=groups, quadrature_points=6  # Few points for speed
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit(maxiter=20, options={"disp": False})  # Few iterations for testing

    print("\nRandom Effects Ordered Logit Results:")
    print(f"β estimates:  {result.beta}")
    print(f"True β:       {beta_true}")
    print(f"Cutpoints:    {result.cutpoints}")
    print(f"True κ:       {cutpoints_true}")
    print(f"σ_α estimate: {result.sigma_alpha:.3f} (true: {sigma_alpha_true})")
    print(f"Converged:    {result.converged}")

    return result


if __name__ == "__main__":
    print("Testing Ordered Choice Models Implementation")
    print("=" * 60)

    try:
        # Test Ordered Logit
        ologit = test_ordered_logit()

        # Test Ordered Probit
        oprobit = test_ordered_probit()

        # Test Marginal Effects
        ame, mem = test_marginal_effects()

        # Test Random Effects
        re_ologit = test_random_effects_ordered()

        print("\n" + "=" * 60)
        print("All ordered models tests completed successfully!")

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
