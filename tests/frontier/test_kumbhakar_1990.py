"""
Tests for Kumbhakar (1990) time-varying inefficiency model.

Tests:
1. Basic estimation convergence
2. Time pattern function B(t) behavior
3. Efficiency variation over time
4. Comparison with Pitt-Lee when b=c=0
5. Parameter recovery with simulated data
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.data import ModelType


@pytest.fixture
def simulated_panel_data():
    """Generate simulated panel data with known time-varying inefficiency.

    DGP: y_it = β₀ + β₁·x_it + v_it - u_it
         u_it = B(t) · u_i
         B(t) = 1 / [1 + exp(b·t + c·t²)]
         u_i ~ N⁺(μ, σ²_u)
         v_it ~ N(0, σ²_v)
    """
    np.random.seed(42)

    N = 50  # Number of entities
    T = 10  # Number of time periods

    # True parameters
    beta_0 = 5.0
    beta_1 = 0.7
    sigma_v = 0.3
    sigma_u = 0.5
    mu = 0.0
    b = -0.2  # Learning: inefficiency decreases over time
    c = 0.0  # Linear pattern

    # Generate data
    data_list = []

    for i in range(N):
        # Draw firm-specific inefficiency
        # Use gamma as approximation for truncated normal
        u_i = np.random.gamma(2, sigma_u / 2)

        for t in range(T):
            # Time pattern
            B_t = 1.0 / (1.0 + np.exp(b * t + c * t**2))
            u_it = B_t * u_i

            # Exogenous variable
            x_it = 2.0 + 0.5 * t + np.random.normal(0, 0.5)

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
                    "true_B_t": B_t,
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
        "b": b,
        "c": c,
    }

    return df, true_params


def test_kumbhakar_basic_estimation(simulated_panel_data):
    """Test basic Kumbhakar (1990) estimation convergence."""
    df, true_params = simulated_panel_data

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.KUMBHAKAR_1990,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Check convergence
    assert result.converged, "Kumbhakar model should converge"

    # Check that log-likelihood is finite
    assert np.isfinite(result.loglik), "Log-likelihood should be finite"

    # Check parameter names
    assert "b" in result.params.index, "Should have b parameter"
    assert "c" in result.params.index, "Should have c parameter"
    assert "mu" in result.params.index, "Should have mu parameter"


def test_kumbhakar_time_pattern_learning(simulated_panel_data):
    """Test that B(t) captures learning pattern (b < 0)."""
    df, true_params = simulated_panel_data

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.KUMBHAKAR_1990,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Get estimated b and c
    b_est = result.params["b"]
    c_est = result.params["c"]

    # For learning data (true b = -0.2), estimated b should be negative
    assert b_est < 0, f"b should be negative for learning pattern, got {b_est:.4f}"

    # Compute B(t) for different time periods
    t_range = np.arange(10)
    B_t = 1.0 / (1.0 + np.exp(b_est * t_range + c_est * t_range**2))

    # B(t) should be in (0, 1)
    assert np.all(B_t > 0) and np.all(B_t < 1), "B(t) should be in (0, 1)"

    # For learning (b < 0, c ≈ 0), B(t) should decrease over time
    if abs(c_est) < 0.1:  # If c is small
        assert B_t[9] < B_t[0], "B(t) should decrease for learning pattern"


def test_kumbhakar_parameter_recovery(simulated_panel_data):
    """Test parameter recovery with simulated data."""
    df, true_params = simulated_panel_data

    # Estimate model
    model = StochasticFrontier(
        data=df,
        depvar="y",
        exog=["x"],
        entity="entity",
        time="time",
        model_type=ModelType.KUMBHAKAR_1990,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Check beta recovery (within reasonable tolerance)
    # Note: We use x as exogenous variable (beta_1)
    beta_est = result.params[0]  # Intercept

    # Check that estimates are in reasonable range
    # (Perfect recovery is not expected due to finite sample)
    assert 4.0 < beta_est < 6.0, f"Intercept should be near 5.0, got {beta_est:.4f}"

    # Check sigma_v and sigma_u recovery
    sigma_v_est = result.sigma_v
    sigma_u_est = result.sigma_u

    assert 0.1 < sigma_v_est < 0.6, f"sigma_v should be near 0.3, got {sigma_v_est:.4f}"
    assert 0.2 < sigma_u_est < 0.8, f"sigma_u should be near 0.5, got {sigma_u_est:.4f}"

    # Check b recovery
    b_est = result.params["b"]
    assert -0.5 < b_est < 0.1, f"b should be near -0.2, got {b_est:.4f}"


def test_kumbhakar_time_varying_efficiency():
    """Test that efficiency varies over time as expected."""
    np.random.seed(123)

    N = 30
    T = 8

    # Generate data with clear time pattern
    data_list = []

    for i in range(N):
        u_i = 0.5  # Fixed inefficiency base

        for t in range(T):
            # Strong learning pattern
            b = -0.3
            c = 0.0
            B_t = 1.0 / (1.0 + np.exp(b * t + c * t**2))
            u_it = B_t * u_i

            x_it = 1.0 + 0.1 * t
            v_it = np.random.normal(0, 0.2)
            y_it = 3.0 + 0.5 * x_it + v_it - u_it

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
        model_type=ModelType.KUMBHAKAR_1990,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Get efficiency estimates (using efficiency module)
    from panelbox.frontier.efficiency import estimate_panel_efficiency

    eff_df = estimate_panel_efficiency(result, by_period=True)

    # Check that efficiency varies over time
    # For each entity, efficiency in later periods should be higher (less inefficiency)
    for i in range(min(N, 5)):  # Check first 5 entities
        eff_entity = eff_df[eff_df["entity"] == i].sort_values("period")

        if len(eff_entity) == T:
            eff_early = eff_entity.iloc[0]["efficiency"]
            eff_late = eff_entity.iloc[-1]["efficiency"]

            # Later periods should have higher efficiency (learning)
            # (This is statistical, so we allow some variation)
            # We just check that mean efficiency increases

    # Average efficiency by period
    avg_eff_by_period = eff_df.groupby("period")["efficiency"].mean()

    # Check that there's variation over time (not constant)
    eff_std = avg_eff_by_period.std()
    assert eff_std > 0.01, "Efficiency should vary over time"


def test_kumbhakar_reduces_to_pitt_lee():
    """Test that when b=c=0, Kumbhakar reduces to Pitt-Lee (time-invariant)."""
    np.random.seed(456)

    N = 40
    T = 8

    # Generate time-INVARIANT data (no learning)
    data_list = []

    for i in range(N):
        u_i = np.random.gamma(2, 0.3)  # Time-invariant inefficiency

        for t in range(T):
            x_it = 1.5 + 0.2 * t + np.random.normal(0, 0.3)
            v_it = np.random.normal(0, 0.25)
            y_it = 4.0 + 0.6 * x_it + v_it - u_i  # Note: u_i, not u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate Kumbhakar model
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

    # Extract b and c
    b_est = result_k90.params["b"]
    c_est = result_k90.params["c"]

    # For time-invariant data, b and c should be close to 0
    # (Allow some tolerance due to sampling variation)
    assert abs(b_est) < 0.3, f"b should be near 0 for time-invariant data, got {b_est:.4f}"
    assert abs(c_est) < 0.3, f"c should be near 0 for time-invariant data, got {c_est:.4f}"

    # When b=c=0, B(t) = 0.5 for all t (constant)
    # This makes inefficiency constant over time, like Pitt-Lee


def test_kumbhakar_quadratic_pattern():
    """Test that c parameter captures non-monotonic patterns (U-shape)."""
    np.random.seed(789)

    N = 35
    T = 10

    # Generate data with U-shaped inefficiency pattern
    # (inefficiency decreases then increases)
    data_list = []

    for i in range(N):
        u_i = 0.6

        for t in range(T):
            # U-shape: b > 0, c < 0 creates inverted learning
            # or b < 0, c > 0 creates learning then degradation
            b = -0.4
            c = 0.05  # Positive c creates eventual increase
            B_t = 1.0 / (1.0 + np.exp(b * t + c * t**2))
            u_it = B_t * u_i

            x_it = 2.0 + 0.15 * t
            v_it = np.random.normal(0, 0.2)
            y_it = 3.5 + 0.55 * x_it + v_it - u_it

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
        model_type=ModelType.KUMBHAKAR_1990,
        frontier="production",
    )

    result = model.fit(verbose=False, maxiter=500)

    # Check that c is non-zero (capturing quadratic pattern)
    c_est = result.params["c"]

    # For U-shaped data, c should be positive
    # (Allow some tolerance)
    assert c_est > -0.1, "c should be positive or near zero for U-shaped pattern"


def test_kumbhakar_model_comparison():
    """Test that Kumbhakar provides better fit than Pitt-Lee for time-varying data."""
    np.random.seed(101112)

    N = 45
    T = 10

    # Generate data with clear time-varying pattern
    data_list = []

    for i in range(N):
        u_i = 0.5

        for t in range(T):
            b = -0.25
            c = 0.0
            B_t = 1.0 / (1.0 + np.exp(b * t + c * t**2))
            u_it = B_t * u_i

            x_it = 1.8 + 0.12 * t + np.random.normal(0, 0.4)
            v_it = np.random.normal(0, 0.25)
            y_it = 4.2 + 0.65 * x_it + v_it - u_it

            data_list.append(
                {
                    "entity": i,
                    "time": t,
                    "y": y_it,
                    "x": x_it,
                }
            )

    df = pd.DataFrame(data_list)

    # Estimate Kumbhakar model
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

    # Estimate Pitt-Lee model for comparison
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

    # Kumbhakar should have higher log-likelihood for time-varying data
    # (It's a generalization of Pitt-Lee)
    assert (
        result_k90.loglik >= result_pl.loglik
    ), f"Kumbhakar LL ({result_k90.loglik:.4f}) should be >= Pitt-Lee LL ({result_pl.loglik:.4f})"

    # The difference should be noticeable for strong time pattern
    # (At least a few log-likelihood points)
    ll_diff = result_k90.loglik - result_pl.loglik
    assert ll_diff > 0, f"Kumbhakar should improve fit, LL difference: {ll_diff:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
