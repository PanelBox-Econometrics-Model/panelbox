"""
Tests for Anderson-Hsiao IV estimator for dynamic panel data.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.models.dynamic.anderson_hsiao import AndersonHsiao


def _make_ar1_panel(
    N=20,
    T=15,
    rho=0.5,
    beta=None,
    sigma2=1.0,
    sigma_alpha=1.0,
    seed=42,
):
    """
    Generate AR(1) panel data.

    DGP: y_it = rho * y_{i,t-1} + beta' x_it + alpha_i + eps_it
    """
    rng = np.random.default_rng(seed)

    n_exog = 0 if beta is None else len(beta)
    alpha = rng.normal(0, sigma_alpha, N)
    sigma = np.sqrt(sigma2)

    rows = []
    for i in range(N):
        y_prev = rng.normal(0, 2)  # initial y
        for t in range(T):
            eps = rng.normal(0, sigma)
            x_vals = rng.normal(0, 1, n_exog)
            x_beta = float(x_vals @ beta) if beta is not None else 0.0
            y = rho * y_prev + x_beta + alpha[i] + eps
            row = {"entity": i + 1, "time": t + 1, "y": y}
            for j in range(n_exog):
                row[f"x{j + 1}"] = x_vals[j]
            rows.append(row)
            y_prev = y

    return pd.DataFrame(rows)


class TestAndersonHsiao:
    """Tests for Anderson-Hsiao estimator."""

    @pytest.fixture
    def ar1_data(self):
        """Simple AR(1) panel data with rho=0.5."""
        return _make_ar1_panel(N=30, T=15, rho=0.5, seed=42)

    @pytest.fixture
    def ar1_data_with_exog(self):
        """AR(1) panel with exogenous variables."""
        return _make_ar1_panel(N=30, T=15, rho=0.5, beta=np.array([1.0, -0.5]), seed=42)

    def test_iv_estimation_basic(self, ar1_data):
        """AH returns valid rho in simple DGP."""
        ah = AndersonHsiao(ar1_data, "y", "entity", "time", lags=1, instrument="levels")
        result = ah.fit()

        assert "rho" in result
        assert "params" in result
        assert "resid" in result
        assert "nobs" in result
        assert np.isfinite(result["rho"])
        # AH is consistent — rho should be in a reasonable range
        assert -1.5 < result["rho"] < 1.5

    def test_instrument_levels_vs_differences(self, ar1_data):
        """Both instruments work; levels generally more efficient."""
        ah_levels = AndersonHsiao(ar1_data, "y", "entity", "time", instrument="levels")
        ah_diff = AndersonHsiao(ar1_data, "y", "entity", "time", instrument="differences")

        res_lev = ah_levels.fit()
        res_dif = ah_diff.fit()

        # Both should produce finite results
        assert np.isfinite(res_lev["rho"])
        assert np.isfinite(res_dif["rho"])

        # Both should be in reasonable range for rho_true=0.5
        # (differences instrument can be quite noisy with small samples)
        assert -2.0 < res_lev["rho"] < 2.0
        assert -5.0 < res_dif["rho"] < 5.0

    def test_exogenous_variables(self, ar1_data_with_exog):
        """AH with exogenous variables returns beta."""
        ah = AndersonHsiao(
            ar1_data_with_exog,
            "y",
            "entity",
            "time",
            exog_vars=["x1", "x2"],
            instrument="levels",
        )
        result = ah.fit()

        assert result["beta"] is not None
        assert len(result["beta"]) == 2
        assert len(result["params"]) == 3  # rho + 2 betas
        assert result["param_names"] == ["L.y", "x1", "x2"]

    def test_unbalanced_panel(self):
        """Works with unbalanced panel."""
        df = _make_ar1_panel(N=20, T=15, rho=0.5, seed=42)
        # Remove some observations to create gaps
        rng = np.random.default_rng(99)
        drop_idx = rng.choice(len(df), size=30, replace=False)
        df_unbal = df.drop(df.index[drop_idx]).reset_index(drop=True)

        ah = AndersonHsiao(df_unbal, "y", "entity", "time", instrument="levels")
        result = ah.fit()

        assert np.isfinite(result["rho"])
        assert result["nobs"] > 0
