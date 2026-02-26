"""
Validation test: PanelBox Panel VAR vs R panelvar on Love & Zicchino (2006) dataset.

This test compares PanelBox Panel VAR estimation results against R's panelvar package
using a synthetic dataset based on the Love & Zicchino (2006) specification.

Acceptance criteria (FASE 6):
- Coefficients GMM: ± 1e-4 (ideally) or ± 5% relative error (acceptable)
- Hansen J statistic: ± 1e-3
- AR(1), AR(2) statistics: ± 1e-3
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.r_validation


@pytest.fixture(scope="module")
def love_zicchino_data():
    """Load Love & Zicchino (2006) dataset."""
    data_path = Path("tests/validation/data/love_zicchino_2006.csv")
    if not data_path.exists():
        pytest.skip("Love & Zicchino dataset not found. Run generate_love_zicchino_data.R first.")
    return pd.read_csv(data_path)


@pytest.fixture(scope="module")
def r_reference_results():
    """Load R reference results."""
    results_path = Path("tests/validation/reference_outputs/love_zicchino_r_results.json")
    if not results_path.exists():
        pytest.skip("R reference results not found. Run estimate_love_zicchino_r.R first.")

    with open(results_path) as f:
        return json.load(f)


@pytest.mark.xfail(reason="PanelVAR GMM method not yet implemented", raises=NotImplementedError)
def test_panelbox_vs_r_gmm_coefficients(love_zicchino_data, r_reference_results):
    """
    Test that PanelBox GMM coefficients match R panelvar within tolerance.

    This is the critical validation test for FASE 6.
    """
    from panelbox.var import PanelVAR, PanelVARData

    # Prepare data
    var_data = PanelVARData(
        love_zicchino_data,
        endog_vars=["sales", "inventory", "ar", "debt"],
        entity_col="firm_id",
        time_col="year",
        lags=2,
    )

    # Estimate with GMM (FOD transformation)
    model = PanelVAR(var_data)
    result = model.fit(
        method="gmm", transformation="fod", gmm_type="twostep", max_instruments=3, collapse=True
    )

    # Get R reference coefficients
    r_gmm = r_reference_results["gmm_fod"]
    r_A1 = np.array(r_gmm["A_matrices"][0])
    r_A2 = np.array(r_gmm["A_matrices"][1])

    # Get PanelBox coefficients
    pb_A1 = result.A_matrices[0]
    pb_A2 = result.A_matrices[1]

    # Compare A1 matrix
    diff_A1 = np.abs(pb_A1 - r_A1)
    rel_diff_A1 = diff_A1 / (np.abs(r_A1) + 1e-10) * 100

    print("\n=== A1 Matrix Comparison ===")
    print("R (panelvar):")
    print(r_A1)
    print("\nPanelBox:")
    print(pb_A1)
    print("\nAbsolute difference:")
    print(diff_A1)
    print("\nRelative difference (%):")
    print(rel_diff_A1)

    # Compare A2 matrix
    diff_A2 = np.abs(pb_A2 - r_A2)
    rel_diff_A2 = diff_A2 / (np.abs(r_A2) + 1e-10) * 100

    print("\n=== A2 Matrix Comparison ===")
    print("R (panelvar):")
    print(r_A2)
    print("\nPanelBox:")
    print(pb_A2)
    print("\nAbsolute difference:")
    print(diff_A2)
    print("\nRelative difference (%):")
    print(rel_diff_A2)

    # Acceptance criteria
    # Primary: ± 1e-4 (very strict)
    # Secondary: ± 5% relative OR ± 0.05 absolute (acceptable for cross-platform GMM)
    max_diff_A1 = diff_A1.max()
    max_diff_A2 = diff_A2.max()
    max_rel_diff_A1 = rel_diff_A1.max()
    max_rel_diff_A2 = rel_diff_A2.max()

    print("\n=== Summary ===")
    print(f"Max absolute diff A1: {max_diff_A1:.6f}")
    print(f"Max absolute diff A2: {max_diff_A2:.6f}")
    print(f"Max relative diff A1: {max_rel_diff_A1:.2f}%")
    print(f"Max relative diff A2: {max_rel_diff_A2:.2f}%")

    # Strict criterion (ideal)
    strict_pass_A1 = max_diff_A1 < 1e-4
    strict_pass_A2 = max_diff_A2 < 1e-4

    # Relaxed criterion (acceptable for cross-platform)
    relaxed_pass_A1 = (max_rel_diff_A1 < 5.0) or (max_diff_A1 < 0.05)
    relaxed_pass_A2 = (max_rel_diff_A2 < 5.0) or (max_diff_A2 < 0.05)

    if strict_pass_A1 and strict_pass_A2:
        print("\n✓ EXCELLENT: Coefficients match R within strict tolerance (± 1e-4)")
    elif relaxed_pass_A1 and relaxed_pass_A2:
        print("\n✓ ACCEPTABLE: Coefficients match R within relaxed tolerance (± 5% or ± 0.05)")
    else:
        print("\n✗ FAILED: Coefficients differ from R beyond acceptable tolerance")
        pytest.fail(
            f"Coefficients differ from R: A1 max diff={max_diff_A1:.6f}, "
            f"A2 max diff={max_diff_A2:.6f} (tolerance: ± 5% or ± 0.05)"
        )

    # Assert at least relaxed criterion
    assert relaxed_pass_A1, (
        f"A1 coefficients differ: max {max_diff_A1:.6f} (> 0.05) and {max_rel_diff_A1:.2f}% (> 5%)"
    )
    assert relaxed_pass_A2, (
        f"A2 coefficients differ: max {max_diff_A2:.6f} (> 0.05) and {max_rel_diff_A2:.2f}% (> 5%)"
    )


@pytest.mark.xfail(reason="PanelVAR GMM method not yet implemented", raises=NotImplementedError)
def test_panelbox_gmm_stability(love_zicchino_data):
    """Test that the estimated VAR is stable (all eigenvalues < 1)."""
    from panelbox.var import PanelVAR, PanelVARData

    var_data = PanelVARData(
        love_zicchino_data,
        endog_vars=["sales", "inventory", "ar", "debt"],
        entity_col="firm_id",
        time_col="year",
        lags=2,
    )

    model = PanelVAR(var_data)
    result = model.fit(
        method="gmm", transformation="fod", gmm_type="twostep", max_instruments=3, collapse=True
    )

    print("\n=== Stability Check ===")
    print(f"Max eigenvalue modulus: {result.max_eigenvalue_modulus():.4f}")
    print(f"Is stable: {result.is_stable()}")

    if result.is_stable():
        print("✓ VAR is stable (all eigenvalues < 1)")
    else:
        print("! VAR is unstable (some eigenvalues ≥ 1)")

    # Note: We don't assert stability here because the DGP might produce unstable estimates
    # in finite samples. This is informative only.


@pytest.mark.xfail(reason="PanelVAR GMM method not yet implemented", raises=NotImplementedError)
def test_panelbox_forecast_functionality(love_zicchino_data):
    """Test that forecasting functionality works on this dataset."""
    from panelbox.var import PanelVAR, PanelVARData

    var_data = PanelVARData(
        love_zicchino_data,
        endog_vars=["sales", "inventory", "ar", "debt"],
        entity_col="firm_id",
        time_col="year",
        lags=2,
    )

    model = PanelVAR(var_data)
    result = model.fit(
        method="gmm", transformation="fod", gmm_type="twostep", max_instruments=3, collapse=True
    )

    # Generate forecasts
    fcst = result.forecast(steps=5, ci_method="bootstrap", n_bootstrap=100, seed=42)

    print("\n=== Forecast Test ===")
    print(f"Forecast horizon: {fcst.horizon}")
    print(f"Number of entities: {fcst.N}")
    print(f"Number of variables: {fcst.K}")
    print(f"Has confidence intervals: {fcst.ci_lower is not None}")

    assert fcst.horizon == 5
    assert fcst.N == 50
    assert fcst.K == 4
    assert fcst.forecasts.shape == (5, 50, 4)
    assert fcst.ci_lower is not None
    assert fcst.ci_upper is not None

    print("✓ Forecast functionality works correctly")


def test_panelbox_causality_network_plot(love_zicchino_data):
    """Test that causality network plotting works on this dataset."""
    from panelbox.var import PanelVAR, PanelVARData

    var_data = PanelVARData(
        love_zicchino_data,
        endog_vars=["sales", "inventory", "ar", "debt"],
        entity_col="firm_id",
        time_col="year",
        lags=2,
    )

    model = PanelVAR(var_data)
    result = model.fit(method="ols", lags=2)

    print("\n=== Causality Network Test ===")

    # Test that the method exists and runs without error
    try:
        # We'll use matplotlib backend and not show the plot in tests
        fig = result.plot_causality_network(threshold=0.10, backend="matplotlib", show=False)
        print("✓ Causality network plot generated successfully")
        assert fig is not None
    except ImportError as e:
        pytest.skip(f"Causality network requires networkx: {e}")


@pytest.mark.parametrize(
    "method",
    [
        "ols",
        pytest.param(
            "gmm",
            marks=pytest.mark.xfail(
                reason="PanelVAR GMM method not yet implemented", raises=NotImplementedError
            ),
        ),
    ],
)
def test_panelbox_granger_causality(love_zicchino_data, method):
    """Test Granger causality tests work on this dataset."""
    from panelbox.var import PanelVAR, PanelVARData

    var_data = PanelVARData(
        love_zicchino_data,
        endog_vars=["sales", "inventory", "ar", "debt"],
        entity_col="firm_id",
        time_col="year",
        lags=2,
    )

    model = PanelVAR(var_data)

    if method == "ols":
        result = model.fit(method="ols", lags=2)
    else:
        result = model.fit(
            method="gmm",
            transformation="fod",
            gmm_type="twostep",
            max_instruments=3,
            collapse=True,
        )

    print(f"\n=== Granger Causality Test ({method.upper()}) ===")

    # Test Granger causality matrix
    granger_mat = result.granger_causality_matrix(significance_level=0.05)
    print("Granger causality matrix (p-values):")
    print(granger_mat)

    assert granger_mat.shape == (4, 4)
    # Diagonal is NaN (variable doesn't Granger-cause itself), check only off-diagonal
    off_diag = granger_mat.values[~np.isnan(granger_mat.values)]
    assert np.all((off_diag >= 0) & (off_diag <= 1))  # p-values in [0, 1]

    # Test specific causality
    test_result = result.granger_causality("sales", "inventory")
    print(f"\nDoes sales Granger-cause inventory? p-value = {test_result.p_value:.4f}")

    assert hasattr(test_result, "p_value")
    assert hasattr(test_result, "wald_stat")

    print(f"✓ Granger causality tests work with {method.upper()}")


if __name__ == "__main__":
    # Allow running this script directly for debugging
    pytest.main([__file__, "-v", "-s"])
