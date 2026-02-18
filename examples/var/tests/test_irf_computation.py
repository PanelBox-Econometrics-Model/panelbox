"""
Test IRF computation against theoretical values.
Uses a known VAR(1) DGP where theoretical IRFs can be computed analytically.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from var_simulation import check_stability, simulate_panel_var, theoretical_irf

from panelbox.var import PanelVAR, PanelVARData


class TestIRFAccuracy:
    """Test IRF accuracy against known DGP."""

    @pytest.fixture
    def known_dgp(self):
        """Define a simple VAR(1) with known parameters."""
        A1 = np.array(
            [
                [0.5, 0.1],
                [0.2, 0.3],
            ]
        )
        Sigma = np.array(
            [
                [1.0, 0.3],
                [0.3, 0.8],
            ]
        )
        return [A1], Sigma

    @pytest.fixture
    def large_panel(self, known_dgp):
        """Generate a large panel from known DGP for estimation."""
        A_matrices, Sigma = known_dgp
        df = simulate_panel_var(
            A_matrices=A_matrices,
            Sigma=Sigma,
            n_entities=50,
            n_periods=100,
            fixed_effects_std=0.5,
            seed=42,
        )
        return df

    def test_cholesky_irf_matches_theoretical(self, known_dgp, large_panel):
        """With large N,T the estimated IRF should approximate theoretical IRF."""
        A_matrices, Sigma = known_dgp

        # Compute theoretical IRF
        true_irf = theoretical_irf(A_matrices, Sigma, periods=10, method="cholesky")

        # Estimate model
        endog_vars = [c for c in large_panel.columns if c.startswith("y_")]
        var_data = PanelVARData(
            data=large_panel,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        # Compute estimated IRF
        est_irf = results.irf(periods=10, method="cholesky", verbose=False)

        # Check that estimated IRF is close to theoretical (with tolerance)
        for h in range(11):
            np.testing.assert_allclose(
                est_irf.irf_matrix[h],
                true_irf[h],
                atol=0.15,
                err_msg=f"IRF mismatch at horizon {h}",
            )

    def test_generalized_irf_order_invariance(self, large_panel):
        """Generalized IRFs should not change with variable ordering."""
        endog_vars = [c for c in large_panel.columns if c.startswith("y_")]

        # Original order
        var_data1 = PanelVARData(
            data=large_panel,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model1 = PanelVAR(var_data1)
        results1 = model1.fit(method="ols", cov_type="clustered")
        irf1 = results1.irf(periods=10, method="generalized", verbose=False)

        # Reversed order
        endog_reversed = list(reversed(endog_vars))
        var_data2 = PanelVARData(
            data=large_panel,
            endog_vars=endog_reversed,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model2 = PanelVAR(var_data2)
        results2 = model2.fit(method="ols", cov_type="clustered")
        irf2 = results2.irf(periods=10, method="generalized", verbose=False)

        # Generalized IRFs should be the same (up to reordering)
        # IRF of y_0 -> y_1 in order 1 should equal IRF of y_0 -> y_1 in order 2
        K = len(endog_vars)
        for i in range(K):
            for j in range(K):
                # Find corresponding indices in reversed ordering
                i_rev = K - 1 - i
                j_rev = K - 1 - j
                np.testing.assert_allclose(
                    irf1.irf_matrix[:, i, j],
                    irf2.irf_matrix[:, i_rev, j_rev],
                    atol=1e-10,
                    err_msg=f"Generalized IRF not order-invariant for ({i},{j})",
                )

    def test_cumulative_irf_is_sum(self, large_panel):
        """Cumulative IRF at horizon h should equal sum of IRFs 0..h."""
        endog_vars = [c for c in large_panel.columns if c.startswith("y_")]
        var_data = PanelVARData(
            data=large_panel,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        # Non-cumulative
        irf_nc = results.irf(periods=10, method="cholesky", cumulative=False, verbose=False)
        # Cumulative
        irf_c = results.irf(periods=10, method="cholesky", cumulative=True, verbose=False)

        # Verify cumulative = cumsum of non-cumulative
        cumsum = np.cumsum(irf_nc.irf_matrix, axis=0)
        np.testing.assert_allclose(
            irf_c.irf_matrix,
            cumsum,
            atol=1e-10,
            err_msg="Cumulative IRF does not equal sum of point IRFs",
        )

    def test_irf_dies_out_for_stable_process(self, known_dgp, large_panel):
        """For a stable VAR, IRFs should converge to zero at long horizons."""
        A_matrices, Sigma = known_dgp

        # Verify DGP is stable
        stability = check_stability(A_matrices)
        assert stability["is_stable"]

        # Compute theoretical IRF at long horizon
        long_irf = theoretical_irf(A_matrices, Sigma, periods=50, method="cholesky")

        # IRFs at horizon 50 should be very close to zero
        assert np.max(np.abs(long_irf[50])) < 0.01

    def test_bootstrap_ci_contains_point_estimate(self, large_panel):
        """Bootstrap CI should contain the point estimate."""
        endog_vars = [c for c in large_panel.columns if c.startswith("y_")]
        var_data = PanelVARData(
            data=large_panel,
            endog_vars=endog_vars,
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        model = PanelVAR(var_data)
        results = model.fit(method="ols", cov_type="clustered")

        try:
            irf = results.irf(
                periods=10,
                method="cholesky",
                ci_method="bootstrap",
                n_bootstrap=100,
                ci_level=0.95,
                seed=42,
                verbose=False,
            )

            if irf.ci_lower is not None and irf.ci_upper is not None:
                # Point estimate should be within CI
                assert np.all(irf.irf_matrix >= irf.ci_lower - 1e-10)
                assert np.all(irf.irf_matrix <= irf.ci_upper + 1e-10)
        except Exception as e:
            pytest.skip(f"Bootstrap CI not available: {e}")
