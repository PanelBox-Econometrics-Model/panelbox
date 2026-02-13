"""
Tests for Dumitrescu-Hurlin (2012) panel Granger causality test.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality import (
    DumitrescuHurlinResult,
    dumitrescu_hurlin_moments,
    dumitrescu_hurlin_test,
)


def test_dumitrescu_hurlin_moments():
    """Test moment calculation for DH test."""
    T = 20
    p = 2
    K = 2

    E_W, Var_W = dumitrescu_hurlin_moments(T, p, K)

    # First moment should equal p
    assert E_W == p

    # Variance should be positive
    assert Var_W > 0

    # For large T, variance should approach 2p
    T_large = 1000
    E_W_large, Var_W_large = dumitrescu_hurlin_moments(T_large, p, K)
    assert abs(Var_W_large - 2 * p) < 0.5  # Should be close to asymptotic


def test_dumitrescu_hurlin_moments_insufficient_df():
    """Test error when insufficient degrees of freedom."""
    # T too small
    T = 5
    p = 2
    K = 2

    # df = T - K*p - 1 = 5 - 4 - 1 = 0
    with pytest.raises(ValueError, match="Insufficient degrees of freedom"):
        dumitrescu_hurlin_moments(T, p, K)


def generate_panel_var_data(
    N=30, T=50, rho_x_to_y=0.5, rho_y_to_x=0.0, heterogeneous=False, seed=42
):
    """
    Generate panel VAR(1) data with known causality structure.

    Parameters
    ----------
    N : int
        Number of entities
    T : int
        Time periods
    rho_x_to_y : float
        Average coefficient of x_{t-1} in y_t equation
    rho_y_to_x : float
        Average coefficient of y_{t-1} in x_t equation
    heterogeneous : bool
        If True, add entity-specific variation to coefficients
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Panel data with columns ['entity', 'time', 'x', 'y']
    """
    np.random.seed(seed)

    data_list = []

    for i in range(N):
        # Entity-specific coefficients (if heterogeneous)
        if heterogeneous:
            beta_x_to_y = rho_x_to_y + np.random.normal(0, 0.1)
            beta_y_to_x = rho_y_to_x + np.random.normal(0, 0.1)
        else:
            beta_x_to_y = rho_x_to_y
            beta_y_to_x = rho_y_to_x

        x = np.zeros(T)
        y = np.zeros(T)

        # Initial values
        x[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)

        # VAR(1) dynamics
        for t in range(1, T):
            x[t] = 0.3 * x[t - 1] + beta_y_to_x * y[t - 1] + np.random.normal(0, 0.5)
            y[t] = 0.3 * y[t - 1] + beta_x_to_y * x[t - 1] + np.random.normal(0, 0.5)

        # Create DataFrame
        entity_df = pd.DataFrame({"entity": i, "time": np.arange(T), "x": x, "y": y})

        data_list.append(entity_df)

    return pd.concat(data_list, ignore_index=True)


def test_dumitrescu_hurlin_basic_structure():
    """Test basic structure and properties of DH test result."""
    # Generate data with causality
    data = generate_panel_var_data(N=20, T=30, rho_x_to_y=0.4, rho_y_to_x=0.0)

    result = dumitrescu_hurlin_test(
        data=data, cause="x", effect="y", lags=1, entity_col="entity", time_col="time"
    )

    # Check result type
    assert isinstance(result, DumitrescuHurlinResult)

    # Check basic attributes
    assert result.cause == "x"
    assert result.effect == "y"
    assert result.N == 20
    assert result.lags == 1

    # Check that we have N individual statistics
    assert len(result.individual_W) == 20

    # All individual W should be non-negative
    assert np.all(result.individual_W >= 0)

    # W_bar should be the mean
    assert abs(result.W_bar - np.mean(result.individual_W)) < 1e-10

    # Check that both statistics are computed
    assert not np.isnan(result.Z_tilde_stat)
    assert not np.isnan(result.Z_bar_stat)

    # P-values should be between 0 and 1
    assert 0 <= result.Z_tilde_pvalue <= 1
    assert 0 <= result.Z_bar_pvalue <= 1

    # Recommended stat should be one of the two
    assert result.recommended_stat in ["Z_tilde", "Z_bar"]


def test_dumitrescu_hurlin_detects_causality_homogeneous():
    """Test DH detects causality in homogeneous panel."""
    # Generate data where x causes y for all entities
    data = generate_panel_var_data(N=25, T=40, rho_x_to_y=0.5, rho_y_to_x=0.0, heterogeneous=False)

    result = dumitrescu_hurlin_test(data=data, cause="x", effect="y", lags=1)

    # Should reject H0 (detect causality)
    assert result.Z_tilde_pvalue < 0.05, "Should detect x→y causality (Z_tilde)"
    assert result.Z_bar_pvalue < 0.05, "Should detect x→y causality (Z_bar)"


def test_dumitrescu_hurlin_detects_causality_heterogeneous():
    """Test DH detects causality in heterogeneous panel."""
    # Generate heterogeneous data where x causes y
    data = generate_panel_var_data(N=25, T=40, rho_x_to_y=0.5, rho_y_to_x=0.0, heterogeneous=True)

    result = dumitrescu_hurlin_test(data=data, cause="x", effect="y", lags=1)

    # Should still detect causality even with heterogeneity
    # At least one of the statistics should reject
    assert min(result.Z_tilde_pvalue, result.Z_bar_pvalue) < 0.10


def test_dumitrescu_hurlin_no_causality():
    """Test DH does not reject when there is no causality."""
    # Generate data with NO causality
    data = generate_panel_var_data(N=25, T=40, rho_x_to_y=0.0, rho_y_to_x=0.0, heterogeneous=False)

    result = dumitrescu_hurlin_test(data=data, cause="x", effect="y", lags=1)

    # Should NOT reject H0
    # Due to randomness, we use a lenient threshold
    assert result.Z_tilde_pvalue > 0.01
    assert result.Z_bar_pvalue > 0.01


def test_dumitrescu_hurlin_partial_causality():
    """Test DH with partial causality (some entities, not all)."""
    # Generate data where only SOME entities have causality
    np.random.seed(123)
    N = 30
    T = 40

    data_list = []

    for i in range(N):
        # First half: x causes y
        # Second half: no causality
        if i < N // 2:
            rho_x_to_y = 0.6
        else:
            rho_x_to_y = 0.0

        x = np.zeros(T)
        y = np.zeros(T)
        x[0] = np.random.normal(0, 1)
        y[0] = np.random.normal(0, 1)

        for t in range(1, T):
            x[t] = 0.3 * x[t - 1] + np.random.normal(0, 0.5)
            y[t] = 0.3 * y[t - 1] + rho_x_to_y * x[t - 1] + np.random.normal(0, 0.5)

        entity_df = pd.DataFrame({"entity": i, "time": np.arange(T), "x": x, "y": y})
        data_list.append(entity_df)

    data = pd.concat(data_list, ignore_index=True)

    result = dumitrescu_hurlin_test(data, "x", "y", lags=1)

    # Should still reject H0 even though only some entities have causality
    # This is the power of the DH test
    assert min(result.Z_tilde_pvalue, result.Z_bar_pvalue) < 0.10


def test_dumitrescu_hurlin_recommended_stat():
    """Test that recommended statistic depends on T."""
    # Small T: should recommend Z_tilde
    data_small_T = generate_panel_var_data(N=30, T=8, rho_x_to_y=0.5)
    result_small_T = dumitrescu_hurlin_test(data_small_T, "x", "y", lags=1)
    assert result_small_T.recommended_stat == "Z_tilde"

    # Large T: should recommend Z_bar
    data_large_T = generate_panel_var_data(N=30, T=50, rho_x_to_y=0.5)
    result_large_T = dumitrescu_hurlin_test(data_large_T, "x", "y", lags=1)
    assert result_large_T.recommended_stat == "Z_bar"


def test_dumitrescu_hurlin_summary():
    """Test summary method."""
    data = generate_panel_var_data(N=20, T=30, rho_x_to_y=0.4)

    result = dumitrescu_hurlin_test(data, "x", "y", lags=1)

    summary = result.summary()

    # Check that summary contains key information
    assert "Dumitrescu-Hurlin" in summary
    assert "x" in summary
    assert "y" in summary
    assert "W̄" in summary or "W_bar" in summary
    assert "Z̃" in summary or "Z_tilde" in summary
    assert str(result.N) in summary


def test_dumitrescu_hurlin_insufficient_observations():
    """Test error when entity has too few observations."""
    # Create data where one entity has too few obs
    data_list = []

    # First entity: sufficient
    entity1 = pd.DataFrame(
        {
            "entity": 0,
            "time": np.arange(20),
            "x": np.random.normal(size=20),
            "y": np.random.normal(size=20),
        }
    )
    data_list.append(entity1)

    # Second entity: insufficient (only 3 obs, need > lags+1 = 2)
    entity2 = pd.DataFrame(
        {
            "entity": 1,
            "time": np.arange(2),
            "x": np.random.normal(size=2),
            "y": np.random.normal(size=2),
        }
    )
    data_list.append(entity2)

    data = pd.concat(data_list, ignore_index=True)

    with pytest.raises(ValueError, match="insufficient observations"):
        dumitrescu_hurlin_test(data, "x", "y", lags=1)


def test_dumitrescu_hurlin_missing_variables():
    """Test error when variables not found."""
    data = generate_panel_var_data(N=10, T=20)

    with pytest.raises(ValueError, match="not found"):
        dumitrescu_hurlin_test(data, "nonexistent", "y", lags=1)


def test_dumitrescu_hurlin_result_repr():
    """Test __repr__ method."""
    data = generate_panel_var_data(N=15, T=25, rho_x_to_y=0.3)
    result = dumitrescu_hurlin_test(data, "x", "y", lags=1)

    repr_str = repr(result)

    assert "DumitrescuHurlinResult" in repr_str
    assert "x" in repr_str
    assert "y" in repr_str


def test_dumitrescu_hurlin_individual_statistics_positive():
    """Test that individual Wald statistics are non-negative."""
    data = generate_panel_var_data(N=20, T=30)

    result = dumitrescu_hurlin_test(data, "x", "y", lags=1)

    # All individual W_i should be >= 0
    assert np.all(result.individual_W >= 0)

    # W_bar should be the mean
    assert abs(result.W_bar - np.mean(result.individual_W)) < 1e-6
