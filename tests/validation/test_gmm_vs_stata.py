"""
Validation tests for Panel VAR GMM against Stata's pvar command.

This test suite compares PanelBox's GMM estimation with Stata's implementation
by Abrigo & Love (2016), which is the reference implementation for Panel VAR GMM.

Reference:
- Abrigo, M. R., & Love, I. (2016). Estimation of panel vector autoregression in Stata.
  The Stata Journal, 16(3), 778-804.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.var.gmm import estimate_panel_var_gmm


@pytest.fixture(scope="module")
def test_data():
    """
    Generate test data for validation.

    Uses a simple DGP with known properties to ensure both implementations
    should converge to similar estimates.

    Returns
    -------
    pd.DataFrame
        Panel data with columns: entity, time, y1, y2
    """
    np.random.seed(42)

    n_entities = 50
    n_time = 10

    # True VAR(1) parameters
    A1_true = np.array([[0.5, 0.2], [0.1, 0.6]])

    data_list = []

    for entity in range(1, n_entities + 1):
        # Initial values
        y = np.random.randn(2)

        for t in range(1, n_time + 1):
            # VAR(1) process: y_t = A1 @ y_{t-1} + epsilon_t
            epsilon = np.random.randn(2) * 0.5
            y = A1_true @ y + epsilon

            data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})

    df = pd.DataFrame(data_list)

    # Save for Stata
    data_path = Path("/tmp/pvar_gmm_test_data.dta")
    df.to_stata(data_path, write_index=False)

    return df


@pytest.fixture(scope="module")
def stata_results(test_data):
    """
    Run Stata pvar command and return results.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test data (triggers data generation)

    Returns
    -------
    dict
        Dictionary containing Stata estimation results including:
        - coefficients: coefficient estimates
        - std_errors: standard errors
        - n_instruments: number of instruments
        - hansen_j: Hansen J statistic
        - hansen_j_p: Hansen J p-value
        - ar1_p: AR(1) test p-value
        - ar2_p: AR(2) test p-value
    """
    # Path to Stata script
    script_path = Path(__file__).parent / "test_gmm_vs_stata.do"

    # Create Stata script if it doesn't exist
    if not script_path.exists():
        stata_code = """
* Validation script for Panel VAR GMM
* Compares with PanelBox implementation

clear all
set more off

* Load data
use "/tmp/pvar_gmm_test_data.dta", clear

* Declare panel structure
xtset entity time

* Install pvar if needed
*ssc install pvar

* Estimate Panel VAR using GMM with Forward Orthogonal Deviations
* Following Abrigo & Love (2016)
* Note: pvar uses forward orthogonal deviations by default

* VAR(1) with two variables
pvar y1 y2, lags(1) fod instlag(1/5)

* Extract results
matrix B = e(b)
matrix V = e(V)

* Standard errors from variance-covariance matrix
mata: V = st_matrix("V")
mata: se = sqrt(diagonal(V))
mata: st_matrix("SE", se)

* Get test statistics
scalar hansen_j = e(j)
scalar hansen_j_p = e(jp)

* Get number of instruments
scalar n_instruments = e(ninstr)

* Export results to JSON
capture file close results_file
file open results_file using "/tmp/pvar_gmm_stata_results.json", write replace

file write results_file "{" _n
file write results_file `"  "method": "pvar_fod_gmm","' _n
file write results_file `"  "lags": 1,"' _n
file write results_file `"  "n_instruments": "' %9.0f (n_instruments) `","' _n
file write results_file `"  "hansen_j": "' %12.6f (hansen_j) `","' _n
file write results_file `"  "hansen_j_p": "' %12.6f (hansen_j_p) `","' _n

* Extract coefficients
local ncol = colsof(B)
file write results_file `"  "coefficients": ["' _n
forval i = 1/`ncol' {
    local val = B[1, `i']
    file write results_file `"    "' %12.8f (`val')
    if `i' < `ncol' {
        file write results_file `","'
    }
    file write results_file _n
}
file write results_file `"  ],"' _n

* Extract standard errors
local nrow = rowsof(SE)
file write results_file `"  "std_errors": ["' _n
forval i = 1/`nrow' {
    mata: st_numscalar("se_val", st_matrix("SE")[`i', 1])
    local val = se_val
    file write results_file `"    "' %12.8f (`val')
    if `i' < `nrow' {
        file write results_file `","'
    }
    file write results_file _n
}
file write results_file `"  ]"' _n

file write results_file "}" _n
file close results_file

* Display results for debugging
display "Hansen J: " hansen_j " (p=" hansen_j_p ")"
display "N instruments: " n_instruments

exit, clear
"""
        script_path.write_text(stata_code)

    # Run Stata script
    try:
        result = subprocess.run(
            ["stata", "-b", "do", str(script_path)], capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            # Try alternative Stata executable names
            for stata_cmd in ["stata-mp", "stata-se", "xstata"]:
                try:
                    result = subprocess.run(
                        [stata_cmd, "-b", "do", str(script_path)],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                    if result.returncode == 0:
                        break
                except FileNotFoundError:
                    continue
            else:
                pytest.skip(f"Stata not available or script failed: {result.stderr}")

    except FileNotFoundError:
        pytest.skip("Stata not installed or not in PATH")

    # Load results
    results_path = Path("/tmp/pvar_gmm_stata_results.json")
    if not results_path.exists():
        pytest.skip("Stata results file not found")

    with open(results_path) as f:
        stata_results = json.load(f)

    return stata_results


@pytest.fixture(scope="module")
def python_results(test_data):
    """
    Estimate Panel VAR GMM using PanelBox.

    Parameters
    ----------
    test_data : pd.DataFrame
        Test data

    Returns
    -------
    GMMEstimationResult
        Estimation results from PanelBox
    """
    result = estimate_panel_var_gmm(
        data=test_data,
        var_lags=1,
        value_cols=["y1", "y2"],
        entity_col="entity",
        time_col="time",
        transform="fod",
        gmm_step="two-step",
        instrument_type="all",
        max_instruments=5,
        windmeijer_correction=True,
    )

    return result


class TestGMMvsStata:
    """Test suite for GMM validation against Stata pvar."""

    def test_coefficients_match_stata(self, python_results, stata_results):
        """
        CRITICAL TEST: Coefficients must match Stata within 1e-4.

        This is the formal acceptance criterion for FASE 2.
        """
        # Get Python coefficients (flatten to 1D for comparison)
        py_coefs = python_results.coefficients.flatten()

        # Get Stata coefficients
        stata_coefs = np.array(stata_results["coefficients"])

        # ACCEPTANCE CRITERION: |Δ| ≤ 1e-4
        np.testing.assert_allclose(
            py_coefs,
            stata_coefs,
            rtol=1e-4,
            atol=1e-4,
            err_msg="GMM coefficients do not match Stata within required tolerance (1e-4)",
        )

    def test_standard_errors_reasonable(self, python_results, stata_results):
        """
        Test that standard errors are in reasonable range compared to Stata.

        Note: Exact SE matching is harder due to Windmeijer correction details,
        but they should be in similar magnitude.
        """
        # Get Python SEs (flatten to 1D)
        py_ses = python_results.standard_errors.flatten()

        # Get Stata SEs
        stata_ses = np.array(stata_results["std_errors"])

        # Check magnitude is similar (within 20%)
        np.testing.assert_allclose(
            py_ses,
            stata_ses,
            rtol=0.20,
            atol=1e-3,
            err_msg="Standard errors differ significantly from Stata",
        )

    def test_hansen_j_matches_stata(self, python_results, stata_results):
        """
        Test that Hansen J statistic matches Stata within 1e-3.

        This validates that the GMM moment conditions are computed correctly.

        Note: This test is pending implementation of Hansen J test in diagnostics.py
        """
        pytest.skip("Hansen J test not yet implemented in diagnostics.py (US-2.2)")

    def test_number_of_instruments_matches(self, python_results, stata_results):
        """
        Test that instrument count matches between implementations.
        """
        assert python_results.n_instruments == stata_results["n_instruments"], (
            f"Instrument count mismatch: Python={python_results.n_instruments}, "
            f"Stata={stata_results['n_instruments']}"
        )

    def test_windmeijer_correction_applied(self, python_results):
        """
        Test that Windmeijer correction was applied for two-step GMM.
        """
        assert python_results.windmeijer_corrected, (
            "Windmeijer correction should be applied for two-step GMM"
        )

    def test_transform_is_fod(self, python_results):
        """
        Test that Forward Orthogonal Deviations was used.
        """
        assert python_results.transform == "fod", (
            f"Expected FOD transform, got {python_results.transform}"
        )


class TestGMMCoefficientsStructure:
    """Test coefficient structure and interpretation."""

    def test_coefficient_shape(self, python_results):
        """
        Test that coefficient matrix has correct shape.

        For VAR(1) with K=2 variables:
        - Should be (K*p × K) = (2 × 2)
        - Each column is one equation
        - Each row is one lagged variable's coefficient
        """
        expected_shape = (2, 2)  # (K*p, K) for p=1, K=2
        assert python_results.coefficients.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {python_results.coefficients.shape}"
        )

    def test_coefficients_are_stable(self, python_results):
        """
        Test that estimated VAR is stable (eigenvalues < 1).

        For a stable VAR, companion matrix eigenvalues must be < 1 in magnitude.
        """
        # For VAR(1), companion matrix is just A1
        A1 = python_results.coefficients

        eigenvalues = np.linalg.eigvals(A1)
        max_eigenvalue = np.max(np.abs(eigenvalues))

        assert max_eigenvalue < 1.0, f"VAR is unstable: max eigenvalue = {max_eigenvalue:.4f} >= 1"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
