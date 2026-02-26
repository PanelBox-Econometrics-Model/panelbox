"""
Tests for Panel VAR GMM estimation -- coverage-focused.

This module contains tests targeting specific uncovered lines in
panelbox/var/gmm.py.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.var.gmm import (
    GMMEstimationResult,
    estimate_panel_var_gmm,
    gmm_one_step,
    gmm_two_step,
    windmeijer_correction_matrix,
)


class TestGMMVARCoverage:
    """Tests targeting specific uncovered lines in panelbox/var/gmm.py."""

    # ------------------------------------------------------------------ #
    # Helper to build panel data
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_panel(n_entities=10, n_periods=20, vars_list=None, seed=42):
        """Create simple balanced panel DataFrame."""
        if vars_list is None:
            vars_list = ["y1", "y2"]
        np.random.seed(seed)
        rows = []
        for i in range(1, n_entities + 1):
            for t in range(1, n_periods + 1):
                row = {"entity": i, "time": t}
                for v in vars_list:
                    row[v] = np.random.randn()
                rows.append(row)
        return pd.DataFrame(rows)

    # ================================================================== #
    # Line 109: y = y.reshape(-1, 1) in gmm_one_step when y is 1D
    # ================================================================== #
    def test_gmm_one_step_1d_y(self):
        """Line 109: gmm_one_step reshapes 1D y to 2D."""
        np.random.seed(42)
        n_obs, n_params, n_instr = 50, 2, 4

        X = np.random.randn(n_obs, n_params)
        Z = np.random.randn(n_obs, n_instr)
        y_1d = np.random.randn(n_obs)  # 1D input

        assert y_1d.ndim == 1

        beta, vcov, resid = gmm_one_step(y_1d, X, Z)

        assert beta.shape == (n_params, 1)
        assert vcov.shape == (n_params, n_params)
        assert resid.shape == (n_obs, 1)

    def test_gmm_one_step_1d_matches_2d(self):
        """Line 109: 1D and 2D y produce identical results."""
        np.random.seed(99)
        n_obs, n_params, n_instr = 60, 3, 5

        X = np.random.randn(n_obs, n_params)
        Z = np.random.randn(n_obs, n_instr)
        y_1d = np.random.randn(n_obs)
        y_2d = y_1d.reshape(-1, 1)

        beta_1d, vcov_1d, resid_1d = gmm_one_step(y_1d, X, Z)
        beta_2d, vcov_2d, resid_2d = gmm_one_step(y_2d, X, Z)

        np.testing.assert_allclose(beta_1d, beta_2d, rtol=1e-12)
        np.testing.assert_allclose(vcov_1d, vcov_2d, rtol=1e-12)
        np.testing.assert_allclose(resid_1d, resid_2d, rtol=1e-12)

    # ================================================================== #
    # Lines 128-139: LinAlgError handling in gmm_one_step
    # ================================================================== #
    def test_gmm_one_step_singular_matrix_raises_value_error(self):
        """Lines 128-129: singular X'Z W Z'X raises ValueError."""
        np.random.seed(42)
        n_obs = 20

        # Zero X makes X'Z W Z'X = 0, which is singular
        X = np.zeros((n_obs, 3))
        Z = np.random.randn(n_obs, 4)
        y = np.random.randn(n_obs, 1)

        with pytest.raises(ValueError, match="GMM estimation failed"):
            gmm_one_step(y, X, Z)

    def test_gmm_one_step_collinear_x_raises_value_error(self):
        """Lines 128-129: collinear X columns cause singular matrix."""
        np.random.seed(42)
        n_obs = 30

        col = np.random.randn(n_obs)
        # Two identical columns -> rank-deficient
        X = np.column_stack([col, col])
        Z = np.random.randn(n_obs, 4)
        y = np.random.randn(n_obs, 1)

        with pytest.raises(ValueError, match="GMM estimation failed"):
            gmm_one_step(y, X, Z)

    def test_gmm_one_step_vcov_nan_on_inv_failure(self):
        """Lines 137-139: when inv fails, vcov is filled with NaN."""
        from unittest.mock import patch

        np.random.seed(42)
        n_obs, n_params, n_instr = 50, 2, 4

        X = np.random.randn(n_obs, n_params)
        Z = np.random.randn(n_obs, n_instr)
        y = np.random.randn(n_obs, 1)

        def inv_raises(mat):
            raise np.linalg.LinAlgError("mocked singular")

        with (
            patch("numpy.linalg.inv", side_effect=inv_raises),
            warnings.catch_warnings(record=True) as caught,
        ):
            warnings.simplefilter("always")
            _beta, vcov, _resid = gmm_one_step(y, X, Z)

        assert np.all(np.isnan(vcov)), "vcov should be all NaN on inv failure"
        vcov_warns = [w for w in caught if "variance-covariance" in str(w.message).lower()]
        assert len(vcov_warns) >= 1

    # ================================================================== #
    # Line 177: y = y.reshape(-1, 1) in gmm_two_step when y is 1D
    # ================================================================== #
    def test_gmm_two_step_1d_y(self):
        """Line 177: gmm_two_step reshapes 1D y to 2D."""
        np.random.seed(42)
        n_obs, n_params, n_instr = 80, 2, 5

        X = np.random.randn(n_obs, n_params)
        Z = np.random.randn(n_obs, n_instr)
        y_1d = np.random.randn(n_obs)

        assert y_1d.ndim == 1

        beta, vcov, resid, corrected = gmm_two_step(y_1d, X, Z)

        assert beta.shape == (n_params, 1)
        assert vcov.shape == (n_params, n_params)
        assert resid.shape == (n_obs, 1)
        assert isinstance(corrected, bool)

    def test_gmm_two_step_1d_matches_2d(self):
        """Line 177: 1D and 2D y produce identical two-step results."""
        np.random.seed(77)
        n_obs, n_params, n_instr = 60, 2, 4

        X = np.random.randn(n_obs, n_params)
        Z = np.random.randn(n_obs, n_instr)
        y_1d = np.random.randn(n_obs)
        y_2d = y_1d.reshape(-1, 1)

        beta_1d, vcov_1d, resid_1d, corr_1d = gmm_two_step(y_1d, X, Z)
        beta_2d, vcov_2d, resid_2d, corr_2d = gmm_two_step(y_2d, X, Z)

        np.testing.assert_allclose(beta_1d, beta_2d, rtol=1e-12)
        np.testing.assert_allclose(vcov_1d, vcov_2d, rtol=1e-12)
        np.testing.assert_allclose(resid_1d, resid_2d, rtol=1e-12)
        assert corr_1d == corr_2d

    def test_gmm_two_step_1d_y_no_windmeijer(self):
        """Line 177: 1D y with Windmeijer correction disabled."""
        np.random.seed(42)
        n_obs, n_params, n_instr = 80, 2, 5

        X = np.random.randn(n_obs, n_params)
        Z = np.random.randn(n_obs, n_instr)
        y_1d = np.random.randn(n_obs)

        beta, _vcov, _resid, corrected = gmm_two_step(y_1d, X, Z, windmeijer_correction=False)
        assert beta.shape == (n_params, 1)
        assert corrected is False

    # ================================================================== #
    # Lines 279-281: Windmeijer correction exception -> warning
    # ================================================================== #
    def test_windmeijer_correction_failure_returns_none(self):
        """Lines 279-281: exception inside correction -> warning + None."""
        np.random.seed(42)

        # Pass None as X to trigger AttributeError inside try block
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = windmeijer_correction_matrix(
                X=None,  # will cause AttributeError on .shape
                Z=np.random.randn(10, 3),
                residuals=np.random.randn(10, 1),
                beta_1=np.random.randn(2, 1),
                beta_2=np.random.randn(2, 1),
                vcov_2=np.eye(2),
                W_2=np.eye(3),
            )

        assert result is None
        windmeijer_warns = [w for w in caught if "Windmeijer correction failed" in str(w.message)]
        assert len(windmeijer_warns) >= 1

    def test_windmeijer_correction_failure_string_input(self):
        """Lines 279-281: passing string X triggers exception path."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = windmeijer_correction_matrix(
                X="not_an_array",
                Z=np.random.randn(10, 3),
                residuals=np.random.randn(10, 1),
                beta_1=np.random.randn(2, 1),
                beta_2=np.random.randn(2, 1),
                vcov_2=np.eye(2),
                W_2=np.eye(3),
            )

        assert result is None
        windmeijer_warns = [w for w in caught if "Windmeijer correction failed" in str(w.message)]
        assert len(windmeijer_warns) >= 1

    # ================================================================== #
    # Lines 379-405: estimate_panel_var_gmm lagging and concatenation
    # ================================================================== #
    def test_estimate_panel_var_gmm_fod_one_step(self):
        """Lines 379-405: FOD transform + one-step GMM."""
        df = self._make_panel(n_entities=10, n_periods=20)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            transform="fod",
            gmm_step="one-step",
        )

        assert isinstance(result, GMMEstimationResult)
        assert result.coefficients.shape == (2, 2)
        assert result.gmm_step == "one-step"
        assert result.transform == "fod"
        assert result.n_obs > 0
        assert result.n_entities == 10

    def test_estimate_panel_var_gmm_fd_one_step(self):
        """Lines 379-405: FD transform + one-step GMM."""
        df = self._make_panel(n_entities=10, n_periods=20)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            transform="fd",
            gmm_step="one-step",
        )

        assert isinstance(result, GMMEstimationResult)
        assert result.transform == "fd"
        assert result.n_obs > 0

    def test_estimate_panel_var_gmm_fod_two_step(self):
        """Lines 379-405: FOD transform + two-step GMM with Windmeijer."""
        df = self._make_panel(n_entities=10, n_periods=25)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            transform="fod",
            gmm_step="two-step",
            windmeijer_correction=True,
        )

        assert isinstance(result, GMMEstimationResult)
        assert result.gmm_step == "two-step"
        assert result.windmeijer_corrected is True

    def test_estimate_panel_var_gmm_fd_two_step(self):
        """Lines 379-405: FD transform + two-step GMM."""
        df = self._make_panel(n_entities=10, n_periods=25)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            transform="fd",
            gmm_step="two-step",
            windmeijer_correction=False,
        )

        assert isinstance(result, GMMEstimationResult)
        assert result.gmm_step == "two-step"
        assert result.transform == "fd"

    def test_estimate_panel_var_gmm_var2(self):
        """Lines 379-405: VAR(2) with lagging logic."""
        df = self._make_panel(n_entities=8, n_periods=25)

        result = estimate_panel_var_gmm(
            df,
            var_lags=2,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            transform="fod",
            gmm_step="one-step",
        )

        assert isinstance(result, GMMEstimationResult)
        # VAR(2) with 2 vars: 4 params per equation
        assert result.coefficients.shape == (4, 2)

    def test_estimate_panel_var_gmm_three_variables(self):
        """Lines 379-405: three endogenous variables."""
        df = self._make_panel(n_entities=10, n_periods=25, vars_list=["y1", "y2", "y3"])

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2", "y3"],
            entity_col="entity",
            time_col="time",
            transform="fod",
            gmm_step="one-step",
        )

        assert isinstance(result, GMMEstimationResult)
        # VAR(1) with 3 vars: 3 params per equation, 3 equations
        assert result.coefficients.shape == (3, 3)

    def test_estimate_panel_var_gmm_short_entities_skipped(self):
        """Lines 378-379: entities with insufficient obs are skipped."""
        np.random.seed(42)
        rows = []
        # 8 entities with enough data
        for i in range(1, 9):
            for t in range(1, 21):
                rows.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        # 2 entities with only 2 periods (will be skipped after transform)
        for i in range(9, 11):
            for t in range(1, 3):
                rows.append(
                    {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                )
        df = pd.DataFrame(rows)

        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            transform="fd",
            gmm_step="one-step",
        )

        assert isinstance(result, GMMEstimationResult)
        assert result.n_obs > 0
