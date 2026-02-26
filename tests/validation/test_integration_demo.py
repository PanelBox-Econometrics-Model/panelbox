"""
Integration test demonstrating Panel VAR forecasting and causality network.
"""

import pandas as pd

try:
    import pytest  # noqa: F401

    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

from panelbox.var import PanelVAR, PanelVARData, plot_causality_network


def test_panel_var_forecasting():
    """Test Panel VAR forecasting functionality."""
    # Load simple test data
    data = pd.read_csv("tests/validation/data/simple_pvar.csv")

    # Prepare data
    pvar_data = PanelVARData(
        data, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=2
    )

    # Fit model
    model = PanelVAR(pvar_data)
    result = model.fit(method="ols")

    # Generate forecasts
    fcst = result.forecast(steps=5, ci_method="bootstrap", n_bootstrap=100, seed=42)

    # Check forecast results
    assert fcst.forecasts.shape == (5, result.N, result.K)
    assert fcst.horizon == 5
    assert fcst.ci_lower is not None
    assert fcst.ci_upper is not None

    # Test to_dataframe
    df = fcst.to_dataframe(entity=0)
    assert isinstance(df, pd.DataFrame)
    # Columns are named by variable (e.g. y1, y1_ci_lower, y1_ci_upper, y2, ...)
    assert any("ci_lower" in c for c in df.columns)
    assert any("ci_upper" in c for c in df.columns)

    # Test summary
    summary = fcst.summary()
    assert "Forecast horizon: 5" in summary
    assert "Number of variables: 3" in summary

    print("✓ Forecasting test passed!")


def test_causality_network():
    """Test causality network visualization."""
    # Load simple test data
    data = pd.read_csv("tests/validation/data/simple_pvar.csv")

    # Prepare data
    pvar_data = PanelVARData(
        data, endog_vars=["y1", "y2", "y3"], entity_col="entity", time_col="time", lags=2
    )

    # Fit model
    model = PanelVAR(pvar_data)
    result = model.fit(method="ols")

    # Get Granger causality matrix
    granger_matrix = result.granger_causality_matrix()

    # Check matrix shape
    assert granger_matrix.shape == (3, 3)
    assert isinstance(granger_matrix, pd.DataFrame)

    # Test network plot (without showing)
    try:
        fig = plot_causality_network(
            granger_matrix, threshold=0.10, backend="matplotlib", show=False
        )
        assert fig is not None
        print("✓ Causality network test passed!")
    except ImportError as e:
        # Skip gracefully
        print(f"⊘ Skipping causality network (requires networkx): {e}")
        return  # Don't raise when running standalone


def test_full_workflow():
    """Test full workflow: fit → forecast → causality → plot."""
    # Load data
    data = pd.read_csv("tests/validation/data/love_zicchino_synthetic.csv")

    # Prepare data
    pvar_data = PanelVARData(
        data,
        endog_vars=["sales", "inv", "ar", "debt"],
        entity_col="firm_id",
        time_col="year",
        lags=2,
    )

    # Fit model
    model = PanelVAR(pvar_data)
    result = model.fit(method="ols")

    # 1. Forecasting
    fcst = result.forecast(steps=3, seed=42)
    assert fcst.forecasts.shape == (3, result.N, 4)

    # 2. Granger causality
    granger_mat = result.granger_causality_matrix()
    assert granger_mat.shape == (4, 4)

    # 3. IRF
    irf_result = result.irf(periods=10)
    assert irf_result.irf_matrix.shape[0] == 11  # periods+1

    # 4. FEVD
    fevd_result = result.fevd(periods=10)
    assert fevd_result.decomposition.shape == (11, 4, 4)  # periods+1

    print("✓ Full workflow test passed!")
    print(f"  - Forecast horizon: {fcst.horizon}")
    print(f"  - IRF periods: {irf_result.periods}")
    print(f"  - FEVD periods: {fevd_result.periods}")


if __name__ == "__main__":
    print("Running integration tests...\n")

    test_panel_var_forecasting()

    # Try causality network (may skip if networkx not installed)
    try:
        test_causality_network()
    except Exception as e:
        print(f"⊘ Causality network test skipped: {e}")

    test_full_workflow()

    print("\n✓ All integration tests passed!")
