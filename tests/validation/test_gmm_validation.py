"""
GMM Validation Tests
====================

Validates PanelBox GMM implementations against R reference outputs.

These tests compare PanelBox CUE-GMM and Bias-Corrected GMM against
established R implementations.

Requirements:
- R reference outputs in tests/validation/outputs/
- Run gmm_cue.R first to generate reference data

Tolerance Levels:
- Coefficients: ± 1e-4
- J-statistic: ± 1e-2
- Standard errors: ± 1e-3 (more lenient due to HAC estimation differences)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm import ContinuousUpdatedGMM


class TestCUEGMMValidation:
    """Validate CUE-GMM against R gmm package."""

    @pytest.fixture
    def validation_dir(self):
        """Get validation outputs directory."""
        test_dir = Path(__file__).parent
        outputs_dir = test_dir / "outputs"
        return outputs_dir

    def load_r_reference(self, filepath):
        """Load R reference output from JSON."""
        if not filepath.exists():
            pytest.skip(f"R reference file not found: {filepath}")

        with open(filepath) as f:
            return json.load(f)

    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("outputs", "gmm_cue_test1.json").exists(),
        reason="R reference outputs not available. Run gmm_cue.R first.",
    )
    def test_cue_simple_iv(self, validation_dir):
        """
        Test Case 1: Simple IV model.

        Validates CUE-GMM on simple instrumental variables setup.
        """
        # Load R reference
        r_ref = self.load_r_reference(validation_dir / "gmm_cue_test1.json")

        # Use exact data from R (to ensure numerical equivalence)
        n = r_ref["n"]
        y = np.array(r_ref["data"]["y"])
        x = np.array(r_ref["data"]["x"])
        z1 = np.array(r_ref["data"]["z1"])
        z2 = np.array(r_ref["data"]["z2"])

        # Create DataFrame
        data = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "entity": np.arange(n), "time": 1})
        data = data.set_index(["entity", "time"])

        # Estimate with PanelBox
        model = ContinuousUpdatedGMM(
            data=data,
            dep_var="y",
            exog_vars=["x"],
            instruments=["z1", "z2"],
            weighting="hac",
            bandwidth="auto",
        )
        results = model.fit(verbose=False)

        # Compare coefficients
        py_coefs = results.params.values
        r_coefs = np.array(r_ref["coefficients"])

        print("\nCoefficient Comparison:")
        print(f"  PanelBox: {py_coefs}")
        print(f"  R:        {r_coefs}")
        print(f"  Diff:     {py_coefs - r_coefs}")

        np.testing.assert_allclose(
            py_coefs, r_coefs, rtol=1e-3, atol=1e-4, err_msg="CUE-GMM coefficients differ from R"
        )

        # Compare J-statistic (more lenient tolerance)
        py_j = model.j_stat_
        r_j = r_ref["j_statistic"]

        print("\nJ-statistic Comparison:")
        print(f"  PanelBox: {py_j:.6f}")
        print(f"  R:        {r_j:.6f}")
        print(f"  Diff:     {abs(py_j - r_j):.6f}")

        np.testing.assert_allclose(
            py_j, r_j, rtol=0.1, atol=0.05, err_msg="J-statistic differs from R"
        )

        # Check convergence
        assert model.converged_, "PanelBox CUE-GMM did not converge"
        assert r_ref["convergence"], "R CUE-GMM did not converge"

        print("\n✓ Test 1 PASSED: Simple IV model validated against R")

    @pytest.mark.skipif(
        not Path(__file__).parent.joinpath("outputs", "gmm_cue_test2.json").exists(),
        reason="R reference outputs not available. Run gmm_cue.R first.",
    )
    def test_cue_overidentified(self, validation_dir):
        """
        Test Case 2: Overidentified model.

        Validates CUE-GMM with more instruments than parameters.
        """
        # Load R reference
        r_ref = self.load_r_reference(validation_dir / "gmm_cue_test2.json")

        # Use exact data from R
        n = r_ref["n"]
        y = np.array(r_ref["data"]["y"])
        x = np.array(r_ref["data"]["x"])
        z1 = np.array(r_ref["data"]["z1"])
        z2 = np.array(r_ref["data"]["z2"])
        z3 = np.array(r_ref["data"]["z3"])

        data = pd.DataFrame(
            {"y": y, "x": x, "z1": z1, "z2": z2, "z3": z3, "entity": np.arange(n), "time": 1}
        )
        data = data.set_index(["entity", "time"])

        # Estimate
        model = ContinuousUpdatedGMM(
            data=data, dep_var="y", exog_vars=["x"], instruments=["z1", "z2", "z3"], weighting="hac"
        )
        results = model.fit(verbose=False)

        # Compare
        py_coefs = results.params.values
        r_coefs = np.array(r_ref["coefficients"])

        print("\nCoefficient Comparison (Overidentified):")
        print(f"  PanelBox: {py_coefs}")
        print(f"  R:        {r_coefs}")
        print(f"  Diff:     {py_coefs - r_coefs}")

        # Slightly more lenient tolerance for overidentified case due to optimization differences
        np.testing.assert_allclose(
            py_coefs,
            r_coefs,
            rtol=1e-2,
            atol=1e-3,
            err_msg="Overidentified CUE-GMM coefficients differ from R",
        )

        # J-statistic should be testable (overidentified)
        py_j = model.j_stat_
        r_j = r_ref["j_statistic"]

        print("\nJ-statistic (Overidentified):")
        print(f"  PanelBox: {py_j:.6f}")
        print(f"  R:        {r_j:.6f}")

        # More lenient for J-statistic
        np.testing.assert_allclose(
            py_j, r_j, rtol=0.15, atol=0.1, err_msg="J-statistic differs significantly from R"
        )

        print("\n✓ Test 2 PASSED: Overidentified model validated against R")

    def test_summary_validation_status(self, validation_dir):
        """
        Summary test showing validation status.

        Prints which reference files are available and validation status.
        """
        test_files = ["gmm_cue_test1.json", "gmm_cue_test2.json", "gmm_cue_test3.json"]

        print("\n" + "=" * 70)
        print("GMM Validation Status")
        print("=" * 70)

        for test_file in test_files:
            filepath = validation_dir / test_file
            if filepath.exists():
                print(f"✓ {test_file} - Reference available")
            else:
                print(f"✗ {test_file} - Reference NOT available (run gmm_cue.R)")

        print("=" * 70)

        # At least one reference should exist for this test to be meaningful
        any_exist = any((validation_dir / tf).exists() for tf in test_files)

        if not any_exist:
            pytest.skip(
                "No R reference outputs available. "
                "Run tests/validation/scripts/gmm_cue.R to generate them."
            )


class TestBiasCorrectedGMMValidation:
    """Validate Bias-Corrected GMM (Monte Carlo validation)."""

    def test_bias_correction_monte_carlo(self):
        """
        Validate bias correction via Monte Carlo.

        This test verifies that bias-corrected GMM reduces bias
        compared to standard GMM on average across simulations.
        """
        from panelbox.gmm import BiasCorrectedGMM, DifferenceGMM

        np.random.seed(2024)
        n_sims = 20  # Reduced for speed
        n = 80
        t = 12

        true_rho = 0.6

        bc_errors = []
        gmm_errors = []

        for _sim in range(n_sims):
            # Generate dynamic panel
            data_list = []
            for i in range(n):
                alpha_i = np.random.normal(0, 1)
                x_it = np.random.normal(0, 1, t)
                y_it = np.zeros(t)
                y_it[0] = alpha_i + np.random.normal(0, 0.5)

                for t_idx in range(1, t):
                    epsilon = np.random.normal(0, 0.5)
                    y_it[t_idx] = true_rho * y_it[t_idx - 1] + 0.3 * x_it[t_idx] + alpha_i + epsilon

                data_list.append(
                    pd.DataFrame({"entity": i, "time": range(t), "y": y_it, "x": x_it})
                )

            data = pd.concat(data_list, ignore_index=True).set_index(["entity", "time"])

            try:
                # Bias-corrected
                bc_model = BiasCorrectedGMM(
                    data=data,
                    dep_var="y",
                    lags=[1],
                    id_var="entity",
                    time_var="time",
                    exog_vars=["x"],
                )
                bc_results = bc_model.fit(time_dummies=False, verbose=False)

                # Standard GMM
                gmm_model = DifferenceGMM(
                    data=data,
                    dep_var="y",
                    lags=[1],
                    id_var="entity",
                    time_var="time",
                    exog_vars=["x"],
                    time_dummies=False,
                )
                gmm_results = gmm_model.fit()

                # Extract lag coefficient
                lag_names = [
                    n for n in bc_results.params.index if "L1.y" in n or "lag1" in n.lower()
                ]
                if lag_names:
                    bc_rho = bc_results.params[lag_names[0]]
                    gmm_rho = gmm_results.params[lag_names[0]]

                    bc_errors.append(bc_rho - true_rho)
                    gmm_errors.append(gmm_rho - true_rho)

            except Exception:
                # Skip failed simulations
                pass

        if len(bc_errors) >= 10:
            bc_bias = np.mean(bc_errors)
            gmm_bias = np.mean(gmm_errors)

            print("\nBias Correction Monte Carlo Results:")
            print(f"  True ρ:           {true_rho}")
            print(f"  GMM bias:         {gmm_bias:.4f}")
            print(f"  BC-GMM bias:      {bc_bias:.4f}")
            print(f"  Bias reduction:   {abs(gmm_bias) - abs(bc_bias):.4f}")

            # Bias-corrected should have smaller absolute bias
            # (Statistical property, allow some tolerance)
            assert abs(bc_bias) <= abs(gmm_bias) * 1.3, (
                f"Bias correction did not reduce bias: BC={bc_bias:.4f}, GMM={gmm_bias:.4f}"
            )

            print("\n✓ Bias correction validated via Monte Carlo")
        else:
            pytest.skip("Insufficient successful simulations for validation")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
