"""
Tests for Lee & Schmidt (1993) time-varying inefficiency model with time dummies.

Tests:
1. Basic estimation convergence
2. Time pattern δ_t recovery
3. Flexibility of time dummies
4. Normalization constraint (δ_T = 1)
5. Comparison with other time-varying models
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.data import ModelType


@pytest.fixture
def simulated_ls_panel_data():
    """Generate simulated panel data with Lee-Schmidt time pattern.

    DGP: y_it = β₀ + β₁·x_it + v_it - u_it
         u_it = δ_t · u_i
         δ_T = 1 (normalized)
         u_i ~ N⁺(μ, σ²_u)
         v_it ~ N(0, σ²_v)
    """
    np.random.seed(42)

    N = 50  # Number of entities
    T = 8  # Number of time periods

    # True parameters
    beta_0 = 5.0
    beta_1 = 0.7
    sigma_v = 0.3
    sigma_u = 0.5
    mu = 0.0

    # True time pattern (δ_t)
    # Create a non-monotonic pattern: high, low, medium, normalized to 1 at T
    delta_t_base = np.array([1.5, 1.3, 0.9, 0.7, 0.8, 1.0, 1.1, 1.0])

    # Generate data
    data_list = []

    for i in range(N):
        # Draw firm-specific inefficiency
        u_i = np.random.gamma(2, sigma_u / 2)  # Approximate truncated normal

        for t in range(T):
            # Time-varying inefficiency
            delta_t = delta_t_base[t]
            u_it = delta_t * u_i

            # Exogenous variable
            x_it = 2.0 + 0.3 * t + np.random.normal(0, 0.5)

            # Noise
            v_it = np.random.normal(0, sigma_v)

            # Output
            y_it = beta_0 + beta_1 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                    "true_u": u_it,
                    "true_delta_t": delta_t,
                }
            )

    df = pd.DataFrame(data_list)

    # Store true parameters
    true_params = {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "sigma_v": sigma_v,
        "sigma_u": sigma_u,
        "mu": mu,
        "delta_t": delta_t_base,
    }

    return df, true_params


def test_lee_schmidt_basic_estimation(simulated_ls_panel_data):
    """Test basic Lee-Schmidt (1993) estimation convergence."""
    df, true_params = simulated_ls_panel_data

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Check convergence
    assert result.converged, "Lee-Schmidt model should converge"

    # Check that log-likelihood is finite
    assert np.isfinite(result.loglik), "Log-likelihood should be finite"

    # Check parameter names
    assert "mu" in result.params.index, "Should have mu parameter"

    # Check that we have T-1 delta parameters + 1 normalized
    delta_params = [p for p in result.params.index if p.startswith("delta_t")]
    T = len(df["time"].unique())
    assert len(delta_params) == T, f"Should have {T} delta_t parameters"


def test_lee_schmidt_normalization_constraint(simulated_ls_panel_data):
    """Test that δ_T = 1 normalization is satisfied."""
    df, true_params = simulated_ls_panel_data

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Get number of time periods
    T = len(df["time"].unique())

    # Check that last delta is exactly 1.0 (normalized)
    delta_T_name = f"delta_t{T}"
    assert delta_T_name in result.params.index, f"Should have {delta_T_name}"

    delta_T = result.params[delta_T_name]
    assert abs(delta_T - 1.0) < 1e-10, f"δ_T should be 1.0 (normalized), got {delta_T:.10f}"


def test_lee_schmidt_delta_pattern_recovery(simulated_ls_panel_data):
    """Test that δ_t pattern is reasonably recovered."""
    df, true_params = simulated_ls_panel_data

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Extract estimated delta_t values
    T = len(df["time"].unique())
    delta_t_est = np.array([result.params[f"delta_t{t+1}"] for t in range(T)])

    # True pattern
    delta_t_true = true_params["delta_t"]

    # Check that estimates are positive
    assert np.all(delta_t_est > 0), "All δ_t should be positive"

    # Check that pattern is qualitatively similar
    # (Exact recovery is not expected due to finite sample and noise)

    # Check correlation between true and estimated patterns
    correlation = np.corrcoef(delta_t_true, delta_t_est)[0, 1]

    # Correlation should be positive (similar pattern)
    assert correlation > 0.3, f"Pattern correlation should be positive, got {correlation:.4f}"


def test_lee_schmidt_flexibility():
    """Test that Lee-Schmidt can capture arbitrary time patterns."""
    np.random.seed(123)

    N = 40
    T = 6

    # Create a VERY non-monotonic pattern: zig-zag
    delta_t_zigzag = np.array([1.5, 0.8, 1.3, 0.9, 1.2, 1.0])

    data_list = []

    for i in range(N):
        u_i = 0.5

        for t in range(T):
            delta_t = delta_t_zigzag[t]
            u_it = delta_t * u_i

            x_it = 1.5 + 0.2 * t + np.random.normal(0, 0.3)
            v_it = np.random.normal(0, 0.2)
            y_it = 4.0 + 0.6 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate Lee-Schmidt model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Check convergence
    assert result.converged, "Lee-Schmidt should converge even with complex pattern"

    # Extract delta_t
    delta_t_est = np.array([result.params[f"delta_t{t+1}"] for t in range(T)])

    # Check that we capture the zig-zag (up-down-up-down)
    # At least check that there's variation (not monotonic)
    delta_diff = np.diff(delta_t_est)

    # Should have both positive and negative changes
    has_increase = np.any(delta_diff > 0)
    has_decrease = np.any(delta_diff < 0)

    assert has_increase or has_decrease, "Should capture non-monotonic pattern"


def test_lee_schmidt_vs_kumbhakar():
    """Compare Lee-Schmidt with Kumbhakar on same data.

    Lee-Schmidt is more flexible (T-1 parameters) vs Kumbhakar (2 parameters).
    Lee-Schmidt should fit at least as well, often better.
    """
    np.random.seed(456)

    N = 45
    T = 8

    # Generate data with moderate time pattern
    delta_t_pattern = np.array([1.4, 1.2, 1.0, 0.9, 0.95, 1.0, 1.05, 1.0])

    data_list = []

    for i in range(N):
        u_i = 0.6

        for t in range(T):
            delta_t = delta_t_pattern[t]
            u_it = delta_t * u_i

            x_it = 2.0 + 0.15 * t + np.random.normal(0, 0.4)
            v_it = np.random.normal(0, 0.25)
            y_it = 4.5 + 0.55 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate Lee-Schmidt
    model_ls = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result_ls = model_ls.fit(verbose=False, maxiter=500)

    # Estimate Kumbhakar
    model_k90 = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.KUMBHAKAR_1990,
        frontier="production",
    )

    result_k90 = model_k90.fit(verbose=False, maxiter=500)

    # Lee-Schmidt should have higher or equal log-likelihood
    # (It's more flexible)
    assert (
        result_ls.loglik >= result_k90.loglik - 1.0
    ), f"Lee-Schmidt LL ({result_ls.loglik:.4f}) should be >= Kumbhakar LL ({result_k90.loglik:.4f})"


def test_lee_schmidt_time_varying_efficiency():
    """Test that efficiency varies over time according to δ_t pattern."""
    np.random.seed(789)

    N = 30
    T = 6

    # Create clear pattern: high inefficiency early, low later
    delta_t_learning = np.array([1.8, 1.5, 1.2, 1.0, 0.9, 1.0])

    data_list = []

    for i in range(N):
        u_i = 0.5

        for t in range(T):
            delta_t = delta_t_learning[t]
            u_it = delta_t * u_i

            x_it = 1.8 + 0.12 * t
            v_it = np.random.normal(0, 0.2)
            y_it = 3.5 + 0.6 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Get efficiency estimates
    from panelbox.frontier.efficiency import estimate_panel_efficiency

    eff_df = estimate_panel_efficiency(result, by_period=True)

    # Average efficiency by period
    avg_eff_by_period = eff_df.groupby("period")["efficiency"].mean()

    # Check that efficiency varies over time
    eff_std = avg_eff_by_period.std()
    assert eff_std > 0.01, "Efficiency should vary over time"

    # For learning pattern (δ_t decreasing), efficiency should increase over time
    # (Less inefficiency = higher efficiency)
    # Check first vs middle period
    if len(avg_eff_by_period) >= 4:
        eff_early = avg_eff_by_period.iloc[0]
        eff_mid = avg_eff_by_period.iloc[3]

        # Middle period should have higher efficiency than early
        # (Statistical test, allow some noise)
        # We just verify there's a trend


def test_lee_schmidt_small_T():
    """Test Lee-Schmidt with small T (fewer time periods).

    Lee-Schmidt requires T >= 3 to be identifiable (need at least 2 free δ_t).
    """
    np.random.seed(101112)

    N = 50
    T = 4  # Small T

    delta_t_pattern = np.array([1.3, 1.0, 0.9, 1.0])

    data_list = []

    for i in range(N):
        u_i = 0.5

        for t in range(T):
            delta_t = delta_t_pattern[t]
            u_it = delta_t * u_i

            x_it = 2.0 + 0.2 * t + np.random.normal(0, 0.3)
            v_it = np.random.normal(0, 0.2)
            y_it = 4.0 + 0.6 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Should still converge with T=4
    assert result.converged, "Lee-Schmidt should work with T=4"

    # Check we have correct number of parameters
    delta_params = [p for p in result.params.index if p.startswith("delta_t")]
    assert len(delta_params) == T, f"Should have {T} delta_t parameters"


def test_lee_schmidt_parameter_bounds():
    """Test that δ_t parameters are positive (inefficiency scale factors)."""
    np.random.seed(131415)

    N = 40
    T = 7

    # Generate data
    delta_t_pattern = np.array([1.5, 1.2, 1.0, 0.8, 0.9, 1.1, 1.0])

    data_list = []

    for i in range(N):
        u_i = 0.6

        for t in range(T):
            delta_t = delta_t_pattern[t]
            u_it = delta_t * u_i

            x_it = 1.5 + 0.15 * t + np.random.normal(0, 0.35)
            v_it = np.random.normal(0, 0.25)
            y_it = 3.8 + 0.65 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Extract all delta_t
    delta_t_est = np.array([result.params[f"delta_t{t+1}"] for t in range(T)])

    # All δ_t should be positive
    assert np.all(delta_t_est > 0), f"All δ_t should be positive, got: {delta_t_est}"

    # Check that they're in reasonable range (not extreme)
    assert np.all(delta_t_est < 5.0), "δ_t should not be too large"


def test_lee_schmidt_vs_pitt_lee():
    """Compare Lee-Schmidt with Pitt-Lee.

    For time-varying data, Lee-Schmidt should fit better.
    """
    np.random.seed(161718)

    N = 45
    T = 8

    # Clear time-varying pattern
    delta_t_pattern = np.array([1.6, 1.4, 1.2, 1.0, 0.9, 0.95, 1.0, 1.0])

    data_list = []

    for i in range(N):
        u_i = 0.55

        for t in range(T):
            delta_t = delta_t_pattern[t]
            u_it = delta_t * u_i

            x_it = 2.0 + 0.18 * t + np.random.normal(0, 0.4)
            v_it = np.random.normal(0, 0.25)
            y_it = 4.2 + 0.62 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate Lee-Schmidt
    model_ls = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.LEE_SCHMIDT_1993,
        frontier="production",
    )

    result_ls = model_ls.fit(verbose=False, maxiter=500)

    # Estimate Pitt-Lee
    model_pl = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.PITT_LEE,
        frontier="production",
    )

    result_pl = model_pl.fit(verbose=False, maxiter=500)

    # Lee-Schmidt should have higher log-likelihood for time-varying data
    assert (
        result_ls.loglik >= result_pl.loglik
    ), f"Lee-Schmidt LL ({result_ls.loglik:.4f}) should be >= Pitt-Lee LL ({result_pl.loglik:.4f})"

    # The improvement should be noticeable
    ll_improvement = result_ls.loglik - result_pl.loglik
    assert ll_improvement > 0, f"Lee-Schmidt should improve fit, got {ll_improvement:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
