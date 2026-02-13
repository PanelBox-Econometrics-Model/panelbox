"""
Monte Carlo simulation to validate Dumitrescu-Hurlin test implementation.

This test replicates key results from Dumitrescu & Hurlin (2012) Table 1,
testing the size and power of the test under different configurations.

Reference:
Dumitrescu, E. I., & Hurlin, C. (2012). Testing for Granger non-causality in
heterogeneous panels. Economic modelling, 29(4), 1450-1460.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from panelbox.var.causality import dumitrescu_hurlin_test


def generate_panel_var_homogeneous(N, T, lags, beta_cause=0.0, seed=None):
    """
    Generate panel VAR data with HOMOGENEOUS coefficients.

    Parameters
    ----------
    N : int
        Number of entities
    T : int
        Number of time periods
    lags : int
        Number of lags
    beta_cause : float
        Causal effect coefficient (0 = no causality)
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data
    """
    if seed is not None:
        np.random.seed(seed)

    data_list = []
    for i in range(N):
        x1 = np.zeros(T)
        x2 = np.zeros(T)

        # Initialize
        for t in range(lags):
            x1[t] = np.random.randn()
            x2[t] = np.random.randn()

        # Generate data
        for t in range(lags, T):
            # x1: AR(1) process
            x1[t] = 0.5 * x1[t - 1] + np.random.randn() * 0.5

            # x2: depends on lagged x1 (if beta_cause != 0)
            x2[t] = 0.5 * x2[t - 1] + beta_cause * x1[t - 1] + np.random.randn() * 0.5

        entity_data = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "x1": x1,
                "x2": x2,
            }
        )
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


def generate_panel_var_heterogeneous(N, T, lags, beta_cause_share=0.0, beta_value=0.3, seed=None):
    """
    Generate panel VAR data with HETEROGENEOUS coefficients.

    Parameters
    ----------
    N : int
        Number of entities
    T : int
        Number of time periods
    lags : int
        Number of lags
    beta_cause_share : float
        Share of entities with causal effect (0 to 1)
    beta_value : float
        Causal effect coefficient for entities with causality
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data
    """
    if seed is not None:
        np.random.seed(seed)

    # Determine which entities have causality
    n_causal = int(N * beta_cause_share)
    causal_entities = np.random.choice(N, size=n_causal, replace=False)

    data_list = []
    for i in range(N):
        x1 = np.zeros(T)
        x2 = np.zeros(T)

        # Initialize
        for t in range(lags):
            x1[t] = np.random.randn()
            x2[t] = np.random.randn()

        # Determine beta for this entity
        beta_i = beta_value if i in causal_entities else 0.0

        # Generate data
        for t in range(lags, T):
            # x1: AR(1) process
            x1[t] = 0.5 * x1[t - 1] + np.random.randn() * 0.5

            # x2: depends on lagged x1 (heterogeneous)
            x2[t] = 0.5 * x2[t - 1] + beta_i * x1[t - 1] + np.random.randn() * 0.5

        entity_data = pd.DataFrame(
            {
                "entity": i,
                "time": range(T),
                "x1": x1,
                "x2": x2,
            }
        )
        data_list.append(entity_data)

    return pd.concat(data_list, ignore_index=True)


@pytest.mark.slow
def test_dh_size_h0_monte_carlo():
    """
    Test the SIZE of DH test under H0 (no causality).

    Under H0, rejection rate should be ≈ significance level (5%).

    Configuration: N=50, T=20, lags=1, beta=0 (no causality)
    Expected: ~5% rejection rate at 5% significance level
    """
    np.random.seed(42)
    n_simulations = 100  # Use more (500+) for production
    N = 50
    T = 20
    lags = 1
    alpha = 0.05

    rejections_Z_tilde = 0
    rejections_Z_bar = 0

    for sim in range(n_simulations):
        # Generate data under H0 (no causality)
        data = generate_panel_var_homogeneous(N, T, lags, beta_cause=0.0, seed=sim + 1000)

        # Run DH test
        result = dumitrescu_hurlin_test(
            data=data, cause="x1", effect="x2", lags=lags, entity_col="entity", time_col="time"
        )

        # Check rejection
        if result.Z_tilde_pvalue < alpha:
            rejections_Z_tilde += 1
        if result.Z_bar_pvalue < alpha:
            rejections_Z_bar += 1

    rejection_rate_Z_tilde = rejections_Z_tilde / n_simulations
    rejection_rate_Z_bar = rejections_Z_bar / n_simulations

    print(f"\nMonte Carlo Size Test (H0, N={N}, T={T}, lags={lags})")
    print(f"Simulations: {n_simulations}")
    print(f"Rejection rate Z̃: {rejection_rate_Z_tilde:.3f} (expected ≈{alpha})")
    print(f"Rejection rate Z̄: {rejection_rate_Z_bar:.3f} (expected ≈{alpha})")

    # Size should be reasonably close to nominal level
    # Allow for Monte Carlo error (binomial SE ≈ sqrt(0.05*0.95/100) ≈ 0.022)
    # Use wider tolerance for small number of simulations
    assert rejection_rate_Z_tilde < 0.15, "Z̃ test size too large"
    assert rejection_rate_Z_bar < 0.15, "Z̄ test size too large"

    # Optionally check if not too conservative (but this is less critical)
    # assert rejection_rate_Z_tilde > 0.01, "Z̃ test might be too conservative"


@pytest.mark.slow
def test_dh_power_homogeneous_causality():
    """
    Test the POWER of DH test under H1 with HOMOGENEOUS causality.

    When causality is present, test should reject H0 frequently.

    Configuration: N=50, T=20, lags=1, beta=0.3 (moderate causality, ALL entities)
    Expected: High rejection rate (>> 50%)
    """
    np.random.seed(42)
    n_simulations = 100  # Use more (500+) for production
    N = 50
    T = 20
    lags = 1
    beta_cause = 0.3  # Moderate causal effect
    alpha = 0.05

    rejections_Z_tilde = 0
    rejections_Z_bar = 0

    for sim in range(n_simulations):
        # Generate data under H1 (homogeneous causality)
        data = generate_panel_var_homogeneous(N, T, lags, beta_cause=beta_cause, seed=sim + 2000)

        # Run DH test
        result = dumitrescu_hurlin_test(
            data=data, cause="x1", effect="x2", lags=lags, entity_col="entity", time_col="time"
        )

        # Check rejection
        if result.Z_tilde_pvalue < alpha:
            rejections_Z_tilde += 1
        if result.Z_bar_pvalue < alpha:
            rejections_Z_bar += 1

    rejection_rate_Z_tilde = rejections_Z_tilde / n_simulations
    rejection_rate_Z_bar = rejections_Z_bar / n_simulations

    print(f"\nMonte Carlo Power Test (H1 homogeneous, N={N}, T={T}, beta={beta_cause})")
    print(f"Simulations: {n_simulations}")
    print(f"Rejection rate (power) Z̃: {rejection_rate_Z_tilde:.3f}")
    print(f"Rejection rate (power) Z̄: {rejection_rate_Z_bar:.3f}")

    # Power should be high for moderate effect
    assert rejection_rate_Z_tilde > 0.50, "Z̃ test power too low"
    assert rejection_rate_Z_bar > 0.50, "Z̄ test power too low"


@pytest.mark.slow
def test_dh_power_heterogeneous_causality():
    """
    Test the POWER of DH test under H1 with HETEROGENEOUS causality.

    When some entities have causality, test should detect it.

    Configuration: N=50, T=20, lags=1, 50% entities with beta=0.3
    Expected: Moderate to high rejection rate (depends on share of causal entities)
    """
    np.random.seed(42)
    n_simulations = 100
    N = 50
    T = 20
    lags = 1
    beta_cause_share = 0.5  # 50% of entities have causality
    beta_value = 0.4  # Moderate to strong effect
    alpha = 0.05

    rejections_Z_tilde = 0
    rejections_Z_bar = 0

    for sim in range(n_simulations):
        # Generate data under H1 (heterogeneous causality)
        data = generate_panel_var_heterogeneous(
            N, T, lags, beta_cause_share=beta_cause_share, beta_value=beta_value, seed=sim + 3000
        )

        # Run DH test
        result = dumitrescu_hurlin_test(
            data=data, cause="x1", effect="x2", lags=lags, entity_col="entity", time_col="time"
        )

        # Check rejection
        if result.Z_tilde_pvalue < alpha:
            rejections_Z_tilde += 1
        if result.Z_bar_pvalue < alpha:
            rejections_Z_bar += 1

    rejection_rate_Z_tilde = rejections_Z_tilde / n_simulations
    rejection_rate_Z_bar = rejections_Z_bar / n_simulations

    print(
        f"\nMonte Carlo Power Test (H1 heterogeneous, {beta_cause_share*100:.0f}% entities with beta={beta_value})"
    )
    print(f"Simulations: {n_simulations}")
    print(f"Rejection rate (power) Z̃: {rejection_rate_Z_tilde:.3f}")
    print(f"Rejection rate (power) Z̄: {rejection_rate_Z_bar:.3f}")

    # Power should be reasonable (> 30% for 50% entities with moderate effect)
    assert rejection_rate_Z_tilde > 0.30, "Z̃ test power too low for heterogeneous causality"
    assert rejection_rate_Z_bar > 0.30, "Z̄ test power too low for heterogeneous causality"


@pytest.mark.slow
def test_dh_moments_accuracy():
    """
    Test that the moments (E[W̄], Var[W̄]) are accurately computed.

    Under H0, W̄ should have mean ≈ E[W̄] and variance ≈ Var[W̄].
    """
    np.random.seed(42)
    n_simulations = 100
    N = 50
    T = 20
    lags = 1

    W_bar_values = []

    for sim in range(n_simulations):
        # Generate data under H0
        data = generate_panel_var_homogeneous(N, T, lags, beta_cause=0.0, seed=sim + 4000)

        # Run DH test
        result = dumitrescu_hurlin_test(
            data=data, cause="x1", effect="x2", lags=lags, entity_col="entity", time_col="time"
        )

        W_bar_values.append(result.W_bar)

    # Empirical moments
    empirical_mean = np.mean(W_bar_values)
    empirical_var = np.var(W_bar_values, ddof=1)

    # Theoretical moments (approximate)
    # Under H0, E[W_i] ≈ p (number of restrictions)
    # W̄ is the AVERAGE of N individual statistics, so:
    # E[W̄] = E[W_i] = p
    # Var[W̄] = Var[W_i] / N = 2p / N (since W_i ~ χ²(p) approximately)
    p = lags
    theoretical_mean = p
    theoretical_var_individual = 2 * p  # Variance of individual W_i
    theoretical_var_average = theoretical_var_individual / N  # Variance of W̄

    print(f"\nMonte Carlo Moments Test (N={N}, T={T}, lags={lags})")
    print(f"Simulations: {n_simulations}")
    print(f"Empirical E[W̄]: {empirical_mean:.3f} (theoretical ≈{theoretical_mean})")
    print(f"Empirical Var[W̄]: {empirical_var:.3f} (theoretical ≈{theoretical_var_average:.3f})")

    # Check that empirical moments are reasonably close to theoretical
    # Allow for sampling variability
    assert abs(empirical_mean - theoretical_mean) < 0.5, "W̄ mean differs from expected"
    assert abs(empirical_var - theoretical_var_average) < 0.05, "W̄ variance differs from expected"
