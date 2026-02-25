"""
Validation tests comparing PanelBox Heckman with R sampleSelection.
"""

from __future__ import annotations

import numpy as np


class TestHeckmanvsR:
    """Compare Heckman against R sampleSelection package."""

    def test_selection_parameters_2step(self, heckman_r_results):
        """Compare selection equation parameters (two-step)."""
        from panelbox.models.selection import PanelHeckman

        data = heckman_r_results["data"]
        r_results = heckman_r_results["results_2step"]

        # Prepare arrays for PanelHeckman
        # Outcome variable (observed only when selected)
        y = data["y"].values
        # Selection indicator
        s = data["s"].values
        # Outcome exogenous (with intercept)
        X = np.column_stack([np.ones(len(data)), data["x1"].values])
        # Selection exogenous (with intercept)
        Z = np.column_stack([np.ones(len(data)), data["z1"].values, data["z2"].values])

        # Fit PanelBox model
        model = PanelHeckman(endog=y, exog=X, selection=s, exog_selection=Z, method="two_step")
        result = model.fit()

        # Get R selection coefficients
        r_sel = r_results[r_results["equation"] == "selection"].set_index("param")["coef"]

        # Compare selection parameters (intercept, z1, z2)
        # PanelHeckman stores selection params as selection_params_
        if hasattr(result, "selection_params_"):
            sel_params = result.selection_params_

            # Compare intercept
            rel_diff_intercept = abs(sel_params[0] - r_sel["(Intercept)"]) / abs(
                r_sel["(Intercept)"]
            )
            assert rel_diff_intercept < 0.15, (
                f"Selection intercept: Python={sel_params[0]:.4f}, R={r_sel['(Intercept)']:.4f}"
            )

            # Compare z1
            rel_diff_z1 = abs(sel_params[1] - r_sel["z1"]) / abs(r_sel["z1"])
            assert rel_diff_z1 < 0.15, (
                f"Selection z1: Python={sel_params[1]:.4f}, R={r_sel['z1']:.4f}"
            )

            # Compare z2
            rel_diff_z2 = abs(sel_params[2] - r_sel["z2"]) / abs(r_sel["z2"])
            assert rel_diff_z2 < 0.15, (
                f"Selection z2: Python={sel_params[2]:.4f}, R={r_sel['z2']:.4f}"
            )

    def test_outcome_parameters_2step(self, heckman_r_results):
        """Compare outcome equation parameters (two-step)."""
        from panelbox.models.selection import PanelHeckman

        data = heckman_r_results["data"]
        r_results = heckman_r_results["results_2step"]

        # Prepare arrays
        y = data["y"].values
        s = data["s"].values
        X = np.column_stack([np.ones(len(data)), data["x1"].values])
        Z = np.column_stack([np.ones(len(data)), data["z1"].values, data["z2"].values])

        model = PanelHeckman(endog=y, exog=X, selection=s, exog_selection=Z, method="two_step")
        result = model.fit()

        r_out = r_results[r_results["equation"] == "outcome"].set_index("param")["coef"]

        # Compare outcome parameters
        if hasattr(result, "params"):
            # Intercept
            rel_diff_intercept = abs(result.params[0] - r_out["(Intercept)"]) / abs(
                r_out["(Intercept)"]
            )
            assert rel_diff_intercept < 0.15, (
                f"Outcome intercept: Python={result.params[0]:.4f}, R={r_out['(Intercept)']:.4f}"
            )

            # x1 coefficient
            rel_diff_x1 = abs(result.params[1] - r_out["x1"]) / abs(r_out["x1"])
            assert rel_diff_x1 < 0.15, (
                f"Outcome x1: Python={result.params[1]:.4f}, R={r_out['x1']:.4f}"
            )
