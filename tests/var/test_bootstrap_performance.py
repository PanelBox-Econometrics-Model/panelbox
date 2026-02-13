"""
Test bootstrap performance.

Tests that bootstrap can complete 999 iterations in reasonable time.
"""

import time

import numpy as np
import pandas as pd
import pytest

from panelbox.var.causality_bootstrap import bootstrap_granger_test
from panelbox.var.data import PanelVARData
from panelbox.var.model import PanelVAR


@pytest.fixture
def panel_data_var1_k2_n50_t20():
    """
    Generate panel data: VAR(1), K=2, N=50, T=20.

    This is the benchmark case from FASE_3.md.
    """
    np.random.seed(42)
    N = 50
    T = 20
    lags = 1

    data_list = []
    for i in range(N):
        # Simple VAR(1) with two variables
        x1 = np.zeros(T)
        x2 = np.zeros(T)

        x1[0] = np.random.randn()
        x2[0] = np.random.randn()

        for t in range(1, T):
            x1[t] = 0.3 * x1[t - 1] + 0.2 * x2[t - 1] + np.random.randn() * 0.5
            x2[t] = 0.1 * x1[t - 1] + 0.4 * x2[t - 1] + np.random.randn() * 0.5

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
def test_bootstrap_performance_999_iterations(panel_data_var1_k2_n50_t20):
    """
    Test that bootstrap completes 999 iterations in < 60 seconds.

    Benchmark: VAR(1), K=2, N=50, T=20
    Target: < 60 seconds for 999 bootstrap iterations
    """
    # Prepare data
    var_data = PanelVARData(
        panel_data_var1_k2_n50_t20,
        endog_vars=["x1", "x2"],
        entity_col="entity",
        time_col="time",
        lags=1,
    )

    # Fit model
    model = PanelVAR(var_data)
    result = model.fit()

    # Time bootstrap test
    start_time = time.time()

    boot_result = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=999,
        bootstrap_type="wild",
        random_state=42,
        show_progress=True,  # Show progress to see what's happening
    )

    elapsed_time = time.time() - start_time

    # Check performance
    assert elapsed_time < 60.0, f"Bootstrap took {elapsed_time:.2f}s, should be < 60s"

    # Check that most iterations succeeded
    assert len(boot_result.bootstrap_dist) > 900, "Most bootstrap iterations should succeed"

    print(f"\nBootstrap performance: {elapsed_time:.2f} seconds for 999 iterations")
    print(f"Successful iterations: {len(boot_result.bootstrap_dist)}/999")
    print(f"Average time per iteration: {elapsed_time/999:.4f} seconds")


def test_bootstrap_performance_100_iterations(panel_data_var1_k2_n50_t20):
    """
    Test bootstrap with 100 iterations (faster for regular testing).

    This should complete very quickly.
    """
    # Prepare data
    var_data = PanelVARData(
        panel_data_var1_k2_n50_t20,
        endog_vars=["x1", "x2"],
        entity_col="entity",
        time_col="time",
        lags=1,
    )

    # Fit model
    model = PanelVAR(var_data)
    result = model.fit()

    # Time bootstrap test
    start_time = time.time()

    boot_result = bootstrap_granger_test(
        result,
        causing_var="x1",
        caused_var="x2",
        n_bootstrap=100,
        bootstrap_type="wild",
        random_state=42,
        show_progress=False,
    )

    elapsed_time = time.time() - start_time

    # Should complete in < 10 seconds
    assert elapsed_time < 10.0, f"100 iterations took {elapsed_time:.2f}s, should be < 10s"

    # Check that most iterations succeeded
    assert len(boot_result.bootstrap_dist) > 90, "Most bootstrap iterations should succeed"

    print(f"\n100 iterations: {elapsed_time:.2f} seconds")
    print(f"Extrapolated to 999: {elapsed_time * 9.99:.2f} seconds")
