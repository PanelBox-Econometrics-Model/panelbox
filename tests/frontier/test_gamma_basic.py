"""
Basic test for gamma distribution to ensure implementation works.
"""

import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier


def test_gamma_basic_fit():
    """Test that gamma model can fit without errors."""
    np.random.seed(42)
    n = 50

    # Generate simple data
    X = np.column_stack([np.ones(n), np.random.normal(0, 1, n)])
    beta_true = np.array([1.0, 0.5])
    u = np.random.gamma(2.0, 1 / 1.5, n)
    v = np.random.normal(0, 0.3, n)
    y = X @ beta_true + v - u

    df = pd.DataFrame({"y": y, "x1": X[:, 1]})

    # Estimate
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x1"],
        frontier="production",
        dist="gamma",
    )

    result = model.fit(maxiter=10, verbose=True)  # Very few iterations

    # Just check that it runs and returns results
    assert result is not None
    assert hasattr(result, "gamma_P")
    assert hasattr(result, "gamma_theta")
    assert result.gamma_P is not None
    assert result.gamma_theta is not None
    print(f"\nGamma P: {result.gamma_P}")
    print(f"Gamma theta: {result.gamma_theta}")
    print(f"Log-likelihood: {result.loglik}")


if __name__ == "__main__":
    test_gamma_basic_fit()
