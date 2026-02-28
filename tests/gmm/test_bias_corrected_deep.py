"""
Deep coverage tests for panelbox/gmm/bias_corrected.py.

Targets two remaining uncovered branches:
1. Line 235: The ``n_entities > 5000`` warning in ``fit()``.
2. Line 381->379: The ``param_idx >= k`` branch (False side of
   ``if param_idx < k``) inside ``_compute_first_order_bias``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm import BiasCorrectedGMM


def _make_dynamic_panel(n=100, t=15, rho=0.5, beta_x=0.3, seed=42):
    """
    Generate dynamic panel data with MultiIndex AND id/time columns.

    Parameters
    ----------
    n : int
        Number of entities.
    t : int
        Number of time periods per entity.
    rho : float
        Autoregressive coefficient.
    beta_x : float
        Coefficient on exogenous variable.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        Panel data with MultiIndex (entity_idx, time_idx) and columns
        id, year, y, x.
    """
    np.random.seed(seed)
    data_list = []
    for i in range(n):
        alpha_i = np.random.normal(0, 1.0)
        x_it = np.random.normal(0, 1, t)
        y_it = np.zeros(t)
        y_it[0] = alpha_i + np.random.normal(0, 0.5)
        for j in range(1, t):
            y_it[j] = rho * y_it[j - 1] + beta_x * x_it[j] + alpha_i + np.random.normal(0, 0.5)
        entity_data = pd.DataFrame({"id": i, "year": range(t), "y": y_it, "x": x_it})
        data_list.append(entity_data)
    data = pd.concat(data_list, ignore_index=True)
    data.index = pd.MultiIndex.from_arrays(
        [data["id"], data["year"]], names=["entity_idx", "time_idx"]
    )
    return data


class TestLargeNWarning:
    """Cover the ``n_entities > 5000`` warning branch in ``fit()`` (line 235)."""

    def test_large_n_warning_fires(self):
        """fit() emits a UserWarning when the panel has more than 5000 entities.

        We build a lightweight DataFrame with 5001 entities (1 time period each),
        then patch the underlying DifferenceGMM so no real estimation runs.
        The warning is issued *before* the GMM model is constructed, so it
        will fire even if the rest of fit() is interrupted.
        """
        n = 5001
        t = 12
        np.random.seed(77)
        ids = np.repeat(np.arange(n), t)
        times = np.tile(np.arange(t), n)
        y = np.random.normal(size=n * t)
        x = np.random.normal(size=n * t)

        data = pd.DataFrame({"id": ids, "year": times, "y": y, "x": x})
        data.index = pd.MultiIndex.from_arrays(
            [data["id"], data["year"]], names=["entity_idx", "time_idx"]
        )

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1],
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            bias_order=1,
            min_n=50,
            min_t=10,
        )

        # Build a fake GMMResults to be returned by the patched DifferenceGMM.fit()
        n_params = 2  # lag + exog
        fake_results = MagicMock()
        fake_results.params = pd.Series([0.5, 0.3], index=["y_lag1", "x"])
        fake_results.vcov = np.eye(n_params) * 0.01
        fake_results.nobs = n * t
        fake_results.n_groups = n
        fake_results.n_instruments = 10
        fake_results.n_params = n_params
        fake_results.hansen_j = MagicMock()
        fake_results.sargan = MagicMock()
        fake_results.ar1_test = MagicMock()
        fake_results.ar2_test = MagicMock()
        fake_results.diff_hansen = MagicMock()
        fake_results.weight_matrix = np.eye(10)
        fake_results.converged = True
        fake_results.two_step = False
        fake_results.windmeijer_corrected = False
        fake_results.model_type = "Difference GMM"
        fake_results.transformation = "first_difference"
        fake_results.residuals = np.zeros(n * t)
        fake_results.fitted_values = np.zeros(n * t)

        # Patch DifferenceGMM so we avoid running the real estimator
        with patch("panelbox.gmm.bias_corrected.DifferenceGMM") as MockDiffGMM:
            mock_instance = MagicMock()
            mock_instance.fit.return_value = fake_results
            mock_instance.Z_transformed = np.zeros((1, 1))
            MockDiffGMM.return_value = mock_instance

            with pytest.warns(
                UserWarning,
                match=r"Bias-corrected GMM with N>5,000 may take considerable time",
            ):
                model.fit(time_dummies=False)


class TestParamIdxOutOfRange:
    """Cover the ``param_idx >= k`` branch in ``_compute_first_order_bias``
    (line 381->379).

    This branch fires when ``len(self.lags) > len(params)``, i.e. the model
    has more lag specifications than the parameter vector has elements.
    Normally the GMM estimator always produces enough params, but calling
    ``_compute_first_order_bias`` directly with a short params vector
    exercises the guard.
    """

    def test_param_idx_ge_k_branch(self):
        """_compute_first_order_bias skips bias for lags beyond params length."""
        data = _make_dynamic_panel(n=60, t=15, rho=0.5, seed=55)

        model = BiasCorrectedGMM(
            data=data,
            dep_var="y",
            lags=[1, 2, 3],
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            bias_order=1,
        )

        # Provide a params vector shorter than len(self.lags) = 3.
        # Only 2 elements, so param_idx=2 will fail ``param_idx < k``.
        short_params = np.array([0.5, 0.3])

        bias = model._compute_first_order_bias(short_params)

        # k = 2, so only indices 0 and 1 get bias corrections
        avg_t = data.groupby(level=0).size().mean()
        expected_0 = -(1 + 0.5) / (avg_t - 1)
        expected_1 = -(1 + 0.3) / (avg_t - 1)

        assert bias.shape == (2,)
        np.testing.assert_allclose(bias[0], expected_0, rtol=1e-10)
        np.testing.assert_allclose(bias[1], expected_1, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
