"""
Validation tests comparing PanelBox Heckman with R sampleSelection.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.r_validation


class TestHeckmanvsR:
    """Compare Heckman against R sampleSelection package."""

    def test_selection_parameters_2step(self, heckman_r_results):
        """Compare selection equation parameters (two-step)."""
        from panelbox.models.selection import PanelHeckman

        data = heckman_r_results["data"]
        r_results = heckman_r_results["results_2step"]

        # Add panel structure
        n = len(data)
        data["entity"] = np.repeat(range(n // 10), 10)[:n]
        data["time"] = np.tile(range(10), n // 10)[:n]

        # Fit PanelBox model
        model = PanelHeckman(
            data=data,
            selection_depvar="s",
            outcome_depvar="y",
            selection_exog=["z1", "z2"],
            outcome_exog=["x1"],
            entity_col="entity",
            time_col="time",
        )
        result = model.fit(method="2step")

        # Get R selection coefficients
        r_sel = r_results[r_results["equation"] == "selection"].set_index("param")["coef"]

        # Compare selection parameters
        for var in ["z1", "z2"]:
            if var in result.selection_params.index:
                python_coef = result.selection_params[var]
                r_coef = r_sel[var]
                rel_diff = abs(python_coef - r_coef) / abs(r_coef)

                assert rel_diff < 0.15, f"Selection {var}: Python={python_coef:.4f}, R={r_coef:.4f}"

    def test_outcome_parameters_2step(self, heckman_r_results):
        """Compare outcome equation parameters (two-step)."""
        from panelbox.models.selection import PanelHeckman

        data = heckman_r_results["data"]
        r_results = heckman_r_results["results_2step"]

        n = len(data)
        data["entity"] = np.repeat(range(n // 10), 10)[:n]
        data["time"] = np.tile(range(10), n // 10)[:n]

        model = PanelHeckman(
            data=data,
            selection_depvar="s",
            outcome_depvar="y",
            selection_exog=["z1", "z2"],
            outcome_exog=["x1"],
            entity_col="entity",
            time_col="time",
        )
        result = model.fit(method="2step")

        r_out = r_results[r_results["equation"] == "outcome"].set_index("param")["coef"]

        # Compare x1 coefficient
        if "x1" in result.outcome_params.index:
            python_coef = result.outcome_params["x1"]
            r_coef = r_out["x1"]
            rel_diff = abs(python_coef - r_coef) / abs(r_coef)

            assert rel_diff < 0.15, f"Outcome x1: Python={python_coef:.4f}, R={r_coef:.4f}"
