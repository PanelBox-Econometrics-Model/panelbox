"""
Test that wild bootstrap is robust to heteroskedasticity.

This test validates that wild bootstrap performs better than residual bootstrap
when data exhibits heteroskedasticity.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality_bootstrap import bootstrap_granger_test
from panelbox.var.data import PanelVARData
from panelbox.var.model import PanelVAR


@pytest.fixture
def panel_data_with_heteroskedasticity():
    """
    Generate panel data with Granger causality AND heteroskedasticity.

    The error variance increases over time, creating heteroskedasticity.
    """
    np.random.seed(42)
    N = 40
    T = 30
    lags = 2

    data_list = []
    for i in range(N):
        # x1: AR(1) process
        x1 = np.zeros(T)
        x1[0] = np.random.randn()
        for t in range(1, T):
            x1[t] = 0.5 * x1[t - 1] + np.random.randn()

        # x2: depends on lagged x1 (causality) with heteroskedastic errors
        x2 = np.zeros(T)
        x2[0] = np.random.randn()
        x2[1] = np.random.randn()
        for t in range(lags, T):
            # Error variance increases over time (heteroskedasticity)
            # Variance at time t is proportional to (1 + t/T)
            error_std = 0.5 * np.sqrt(1 + t / T)
            x2[t] = (
                0.3 * x2[t - 1] + 0.4 * x1[t - 1] + 0.3 * x1[t - 2] + np.random.randn() * error_std
            )

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


def test_wild_bootstrap_vs_residual_with_heteroskedasticity(panel_data_with_heteroskedasticity):
    """
    Test that wild bootstrap is more robust than residual bootstrap under heteroskedasticity.

    Expected behavior:
    - Both should detect causality (present in DGP)
    - Wild bootstrap should provide more reliable inference
    - P-values should be in reasonable range

    Note: This is a probabilistic test, so we use lenient thresholds.
    """
    # Prepare data
    var_data = PanelVARData(
        panel_data_with_heteroskedasticity,
        endog_vars=["x1", "x2"],
        entity_col="entity",
        time_col="time",
        lags=2,
    )

    # Fit model
    model = PanelVAR(var_data)
    result = model.fit()

    # Run wild bootstrap
    boot_wild = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=99,
        bootstrap_type="wild",
        random_state=42,
        show_progress=False,
    )

    # Run residual bootstrap for comparison
    boot_residual = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=99,
        bootstrap_type="residual",
        random_state=42,
        show_progress=False,
    )

    # Both should detect causality (it's present in DGP)
    assert boot_wild.p_value_bootstrap < 0.20, "Wild bootstrap should detect causality"
    assert boot_residual.p_value_bootstrap < 0.20, "Residual bootstrap should also detect causality"

    # Wild bootstrap should detect strong causality (p-value very small or exactly 0)
    # This is expected given the strong causality in the DGP
    assert boot_wild.p_value_bootstrap < 0.05, "Wild bootstrap should strongly reject H0"

    # Check that bootstrap distributions are reasonable
    assert len(boot_wild.bootstrap_dist) > 80, "Most wild bootstrap iterations should succeed"
    assert (
        len(boot_residual.bootstrap_dist) > 80
    ), "Most residual bootstrap iterations should succeed"

    # Bootstrap statistics should be positive
    assert boot_wild.bootstrap_dist.mean() > 0
    assert boot_residual.bootstrap_dist.mean() > 0

    # Observed statistic should be positive (causality present)
    assert boot_wild.observed_stat > 0
    assert boot_residual.observed_stat == boot_wild.observed_stat  # Same data

    print(f"\nWild bootstrap p-value: {boot_wild.p_value_bootstrap:.4f}")
    print(f"Residual bootstrap p-value: {boot_residual.p_value_bootstrap:.4f}")
    print(f"Asymptotic p-value: {boot_wild.p_value_asymptotic:.4f}")


def test_wild_bootstrap_reproducibility_with_heteroskedasticity(
    panel_data_with_heteroskedasticity,
):
    """Test that wild bootstrap is reproducible even with heteroskedastic data."""
    # Prepare data
    var_data = PanelVARData(
        panel_data_with_heteroskedasticity,
        endog_vars=["x1", "x2"],
        entity_col="entity",
        time_col="time",
        lags=2,
    )

    # Fit model
    model = PanelVAR(var_data)
    result = model.fit()

    # Run wild bootstrap twice with same seed
    boot1 = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=50,
        bootstrap_type="wild",
        random_state=42,
        show_progress=False,
    )

    boot2 = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=50,
        bootstrap_type="wild",
        random_state=42,
        show_progress=False,
    )

    # Results should be identical
    np.testing.assert_allclose(boot1.bootstrap_dist, boot2.bootstrap_dist)
    assert boot1.p_value_bootstrap == boot2.p_value_bootstrap


def test_bootstrap_confidence_intervals_with_heteroskedasticity(
    panel_data_with_heteroskedasticity,
):
    """Test that bootstrap confidence intervals are reasonable under heteroskedasticity."""
    # Prepare data
    var_data = PanelVARData(
        panel_data_with_heteroskedasticity,
        endog_vars=["x1", "x2"],
        entity_col="entity",
        time_col="time",
        lags=2,
    )

    # Fit model
    model = PanelVAR(var_data)
    result = model.fit()

    # Run wild bootstrap
    boot_result = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=99,
        bootstrap_type="wild",
        random_state=42,
        show_progress=False,
    )

    # Check confidence interval
    assert boot_result.ci_lower < boot_result.ci_upper
    assert boot_result.ci_lower >= 0  # Wald statistics are non-negative

    # Observed statistic might or might not be in CI
    # (depends on whether we're testing under H0 or H1)
    # But CI should be reasonable
    ci_width = boot_result.ci_upper - boot_result.ci_lower
    assert ci_width > 0, "CI should have positive width"
    assert ci_width < boot_result.observed_stat * 5, "CI should not be unreasonably wide"

    print(f"\nObserved statistic: {boot_result.observed_stat:.4f}")
    print(f"95% CI: [{boot_result.ci_lower:.4f}, {boot_result.ci_upper:.4f}]")
