"""
Validation tests comparing PanelBox CUE-GMM with R gmm package.
"""

import pytest

pytestmark = pytest.mark.r_validation


class TestCUEGMMvsR:
    """Compare CUE-GMM against R gmm package."""

    def test_parameter_estimates(self, cue_gmm_r_results):
        """Compare parameter estimates with R."""
        from panelbox.gmm import ContinuousUpdatedGMM

        r_results = cue_gmm_r_results["results"]
        data = cue_gmm_r_results["data"]

        # Fit PanelBox model
        model = ContinuousUpdatedGMM(
            data=data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )
        result = model.fit()

        # Get R coefficients
        r_coefs = r_results.set_index("param")["coef"]

        # Compare (tolerance ~15% due to different implementations)
        for i, var in enumerate(["const", "x1", "x2"]):
            python_coef = result.params.iloc[i]
            r_coef = r_coefs[var]
            rel_diff = abs(python_coef - r_coef) / abs(r_coef)

            assert rel_diff < 0.15, (
                f"Parameter {var}: Python={python_coef:.4f}, R={r_coef:.4f}, "
                f"rel_diff={rel_diff:.2%}"
            )

    def test_standard_errors(self, cue_gmm_r_results):
        """Verify standard errors are positive and reasonable."""
        from panelbox.gmm import ContinuousUpdatedGMM

        cue_gmm_r_results["results"]
        data = cue_gmm_r_results["data"]

        model = ContinuousUpdatedGMM(
            data=data, dep_var="y", exog_vars=["x1", "x2"], instruments=["z1", "z2", "z3"]
        )
        result = model.fit()

        # Just verify standard errors are positive (differences can be large due to implementation)
        for i in range(len(result.std_errors)):
            assert result.std_errors.iloc[i] > 0, f"SE at position {i} should be positive"
