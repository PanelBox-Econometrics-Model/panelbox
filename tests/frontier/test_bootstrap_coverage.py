"""
Tests for frontier/bootstrap.py to increase coverage from 30% to 70%+.

Tests SFABootstrap class initialization, validation, parametric and pairs
bootstrap replication methods, bias_corrected_ci, and convenience functions.
Heavy use of mocking to avoid slow SFA fitting during tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.bootstrap import (
    SFABootstrap,
    bootstrap_efficiency,
    bootstrap_sfa,
)

# ---------------------------------------------------------------------------
# Helpers: build a mock SFResult that satisfies SFABootstrap.__init__
# ---------------------------------------------------------------------------


def _make_mock_result(
    n_obs=50,
    n_params=3,
    dist="half_normal",
    frontier_type="production",
    sigma_v=0.2,
    sigma_u=0.3,
):
    """Create a lightweight mock SFResult for bootstrap tests."""
    rng = np.random.default_rng(42)

    X = np.column_stack([np.ones(n_obs), rng.normal(size=(n_obs, n_params - 1))])
    y = X @ np.arange(1, n_params + 1, dtype=float) + rng.normal(0, 0.1, n_obs)

    param_names = ["const"] + [f"x{j}" for j in range(1, n_params)]
    params = pd.Series(np.arange(1, n_params + 1, dtype=float), index=param_names)

    model = MagicMock()
    model.y = y
    model.X = X
    model.dist = dist
    model.frontier_type = frontier_type

    result = MagicMock()
    result.model = model
    result.params = params
    result.sigma_v = sigma_v
    result.sigma_u = sigma_u

    return result


# ---------------------------------------------------------------------------
# Etapa 2: SFABootstrap class
# ---------------------------------------------------------------------------


class TestSFABootstrapInit:
    """Tests for SFABootstrap initialization and validation."""

    def test_sfa_bootstrap_init(self):
        """Test SFABootstrap initialization with valid arguments."""
        result = _make_mock_result()
        boot = SFABootstrap(result, method="parametric", n_boot=100, ci_level=0.90, seed=42)

        assert boot.result is result
        assert boot.method == "parametric"
        assert boot.n_boot == 100
        assert boot.ci_level == 0.90
        assert boot.seed == 42
        assert boot.n_obs == 50
        assert boot.X.shape == (50, 3)

    def test_sfa_bootstrap_init_invalid_ci_level(self):
        """Test SFABootstrap raises for ci_level out of (0,1)."""
        result = _make_mock_result()

        with pytest.raises(ValueError, match="ci_level must be in"):
            SFABootstrap(result, ci_level=0.0)

        with pytest.raises(ValueError, match="ci_level must be in"):
            SFABootstrap(result, ci_level=1.0)

    def test_sfa_bootstrap_init_invalid_n_boot(self):
        """Test SFABootstrap raises for n_boot < 100."""
        result = _make_mock_result()

        with pytest.raises(ValueError, match="n_boot should be >= 100"):
            SFABootstrap(result, n_boot=50)

    def test_bootstrap_parameters(self):
        """Test bootstrap_parameters returns dict with expected keys."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        # Mock the replication methods so each returns a mock result
        def _mock_rep(b):
            mock_res = MagicMock()
            mock_res.params = result.params + np.random.default_rng(b).normal(
                0, 0.01, len(result.params)
            )
            mock_res.params = pd.Series(mock_res.params, index=result.params.index)
            return mock_res

        with patch.object(boot, "_bootstrap_rep_parametric", side_effect=_mock_rep):
            out = boot.bootstrap_parameters()

        assert "params_boot" in out
        assert "ci_lower" in out
        assert "ci_upper" in out
        assert "mean_boot" in out
        assert "std_boot" in out
        assert "bias" in out
        assert "results_df" in out
        assert isinstance(out["results_df"], pd.DataFrame)
        assert out["n_valid"] == 100
        assert out["n_failed"] == 0

    def test_bootstrap_parameters_with_failures(self):
        """Test bootstrap_parameters handles failed replications (None)."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        call_count = 0

        def _mock_rep(b):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:
                return None  # simulate failure
            mock_res = MagicMock()
            mock_res.params = pd.Series(
                result.params.values + np.random.default_rng(b).normal(0, 0.01, len(result.params)),
                index=result.params.index,
            )
            return mock_res

        with patch.object(boot, "_bootstrap_rep_parametric", side_effect=_mock_rep):
            out = boot.bootstrap_parameters()

        assert out["n_failed"] > 0
        assert out["n_valid"] < 100

    def test_bootstrap_parameters_pairs_method(self):
        """Test bootstrap_parameters with pairs method."""
        result = _make_mock_result()
        boot = SFABootstrap(result, method="pairs", n_boot=100, seed=0, n_jobs=1)

        def _mock_rep(b):
            mock_res = MagicMock()
            mock_res.params = pd.Series(
                result.params.values + np.random.default_rng(b).normal(0, 0.01, len(result.params)),
                index=result.params.index,
            )
            return mock_res

        with patch.object(boot, "_bootstrap_rep_pairs", side_effect=_mock_rep):
            out = boot.bootstrap_parameters()

        assert out["n_valid"] == 100


class TestBootstrapEfficiency:
    """Tests for bootstrap_efficiency method."""

    def test_bootstrap_efficiency_returns_dataframe(self):
        """Test bootstrap_efficiency returns DataFrame with CI columns."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        # Mock efficiency for original result
        eff_df = pd.DataFrame(
            {
                "te": np.random.default_rng(0).uniform(0.5, 1.0, 50),
            }
        )
        result.efficiency = MagicMock(return_value=eff_df)

        def _mock_rep(b):
            mock_res = MagicMock()
            eff_mock = pd.DataFrame(
                {
                    "te": np.random.default_rng(b).uniform(0.5, 1.0, 50),
                }
            )
            mock_res.efficiency = MagicMock(return_value=eff_mock)
            return mock_res

        with patch.object(boot, "_bootstrap_rep_parametric", side_effect=_mock_rep):
            out = boot.bootstrap_efficiency(estimator="bc")

        assert isinstance(out, pd.DataFrame)
        assert "boot_mean" in out.columns
        assert "boot_std" in out.columns
        assert "ci_lower" in out.columns
        assert "ci_upper" in out.columns
        # CI should be clipped to [0, 1]
        assert out["ci_lower"].min() >= 0.0
        assert out["ci_upper"].max() <= 1.0

    def test_bootstrap_efficiency_all_failures(self):
        """Test bootstrap_efficiency when all reps fail returns NaN CIs."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        eff_df = pd.DataFrame({"te": np.full(50, 0.8)})
        result.efficiency = MagicMock(return_value=eff_df)

        def _mock_rep(b):
            return None  # all fail

        with (
            patch.object(boot, "_bootstrap_rep_parametric", side_effect=_mock_rep),
            pytest.warns(UserWarning, match="All .* efficiency bootstrap replications failed"),
        ):
            out = boot.bootstrap_efficiency(estimator="bc")

        assert np.all(np.isnan(out["boot_mean"]))
        assert np.all(np.isnan(out["ci_lower"]))

    def test_bootstrap_efficiency_pairs(self):
        """Test bootstrap_efficiency with pairs method."""
        result = _make_mock_result()
        boot = SFABootstrap(result, method="pairs", n_boot=100, seed=0, n_jobs=1)

        eff_df = pd.DataFrame({"te": np.random.default_rng(0).uniform(0.5, 1.0, 50)})
        result.efficiency = MagicMock(return_value=eff_df)

        def _mock_rep(b):
            mock_res = MagicMock()
            eff_mock = pd.DataFrame({"te": np.random.default_rng(b).uniform(0.5, 1.0, 50)})
            mock_res.efficiency = MagicMock(return_value=eff_mock)
            return mock_res

        with patch.object(boot, "_bootstrap_rep_pairs", side_effect=_mock_rep):
            out = boot.bootstrap_efficiency(estimator="jlms")

        assert isinstance(out, pd.DataFrame)
        assert "ci_lower" in out.columns

    def test_bootstrap_efficiency_exception_in_efficiency(self):
        """Test that efficiency computation exceptions are caught as NaN."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        eff_df = pd.DataFrame({"te": np.full(50, 0.8)})
        result.efficiency = MagicMock(return_value=eff_df)

        call_count = 0

        def _mock_rep(b):
            nonlocal call_count
            call_count += 1
            mock_res = MagicMock()
            if call_count % 2 == 0:
                # Efficiency call raises exception
                mock_res.efficiency = MagicMock(side_effect=RuntimeError("boom"))
            else:
                eff_mock = pd.DataFrame({"te": np.random.default_rng(b).uniform(0.5, 1.0, 50)})
                mock_res.efficiency = MagicMock(return_value=eff_mock)
            return mock_res

        with patch.object(boot, "_bootstrap_rep_parametric", side_effect=_mock_rep):
            out = boot.bootstrap_efficiency(estimator="bc")

        assert isinstance(out, pd.DataFrame)


# ---------------------------------------------------------------------------
# Etapa 3: Replication methods
# ---------------------------------------------------------------------------


class TestBootstrapRepParametric:
    """Tests for _bootstrap_rep_parametric."""

    def test_bootstrap_rep_parametric_half_normal(self):
        """Test parametric bootstrap with half_normal distribution."""
        result = _make_mock_result(n_obs=30, dist="half_normal")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        # Mock StochasticFrontier to avoid real fitting
        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_fit_result = MagicMock()
            mock_sf_instance.fit.return_value = mock_fit_result
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_parametric(0)

        assert rep is mock_fit_result

    def test_bootstrap_rep_parametric_exponential(self):
        """Test parametric bootstrap with exponential distribution."""
        result = _make_mock_result(n_obs=30, dist="exponential")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_sf_instance.fit.return_value = MagicMock()
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_parametric(0)

        assert rep is not None

    def test_bootstrap_rep_parametric_truncated_normal(self):
        """Test parametric bootstrap with truncated_normal distribution."""
        result = _make_mock_result(n_obs=30, dist="truncated_normal")
        result.params["mu"] = 0.5  # mu parameter for truncated normal
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_sf_instance.fit.return_value = MagicMock()
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_parametric(0)

        assert rep is not None

    def test_bootstrap_rep_parametric_unknown_dist(self):
        """Test parametric bootstrap raises for unknown distribution."""
        result = _make_mock_result(n_obs=30, dist="unknown_dist")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with pytest.raises(NotImplementedError, match="Bootstrap for unknown_dist not implemented"):
            boot._bootstrap_rep_parametric(0)

    def test_bootstrap_rep_parametric_convergence_failure(self):
        """Test parametric bootstrap returns None on convergence failure."""
        result = _make_mock_result(n_obs=30, dist="half_normal")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            MockSF.side_effect = RuntimeError("convergence failed")

            rep = boot._bootstrap_rep_parametric(0)

        assert rep is None

    def test_bootstrap_rep_parametric_cost_frontier(self):
        """Test parametric bootstrap with cost frontier (sign = +1)."""
        result = _make_mock_result(n_obs=30, dist="half_normal", frontier_type="cost")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_sf_instance.fit.return_value = MagicMock()
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_parametric(0)

        assert rep is not None

    def test_bootstrap_rep_parametric_no_exog(self):
        """Test parametric bootstrap with intercept-only model (no exog)."""
        result = _make_mock_result(n_obs=30, n_params=1, dist="half_normal")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_sf_instance.fit.return_value = MagicMock()
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_parametric(0)

        assert rep is not None


class TestBootstrapRepPairs:
    """Tests for _bootstrap_rep_pairs."""

    def test_bootstrap_rep_pairs(self):
        """Test pairs bootstrap replication."""
        result = _make_mock_result(n_obs=30, dist="half_normal")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_sf_instance.fit.return_value = MagicMock()
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_pairs(0)

        assert rep is not None

    def test_bootstrap_rep_pairs_convergence_failure(self):
        """Test pairs bootstrap returns None on convergence failure."""
        result = _make_mock_result(n_obs=30, dist="half_normal")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            MockSF.side_effect = RuntimeError("convergence failed")

            rep = boot._bootstrap_rep_pairs(0)

        assert rep is None

    def test_bootstrap_rep_pairs_no_exog(self):
        """Test pairs bootstrap with intercept-only model."""
        result = _make_mock_result(n_obs=30, n_params=1, dist="half_normal")
        boot = SFABootstrap(result, n_boot=100, seed=42, n_jobs=1)

        with patch("panelbox.frontier.StochasticFrontier") as MockSF:
            mock_sf_instance = MagicMock()
            mock_sf_instance.fit.return_value = MagicMock()
            MockSF.return_value = mock_sf_instance

            rep = boot._bootstrap_rep_pairs(0)

        assert rep is not None


class TestBiasCorrectedCI:
    """Tests for bias_corrected_ci (BCa) method."""

    def test_bias_corrected_ci(self):
        """Test BCa confidence interval computation."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        # Pre-populate _boot_results to avoid running full bootstrap
        rng = np.random.default_rng(42)
        boot._boot_results = {
            "params_boot": rng.normal(
                loc=result.params.values, scale=0.1, size=(100, len(result.params))
            ),
        }

        ci = boot.bias_corrected_ci("const")

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        lower, upper = ci
        assert lower < upper

    def test_bias_corrected_ci_triggers_bootstrap(self):
        """Test BCa runs bootstrap_parameters if not already done."""
        result = _make_mock_result()
        boot = SFABootstrap(result, n_boot=100, seed=0, n_jobs=1)

        # Don't set _boot_results; it should call bootstrap_parameters
        def _mock_rep(b):
            mock_res = MagicMock()
            mock_res.params = pd.Series(
                result.params.values + np.random.default_rng(b).normal(0, 0.01, len(result.params)),
                index=result.params.index,
            )
            return mock_res

        with patch.object(boot, "_bootstrap_rep_parametric", side_effect=_mock_rep):
            ci = boot.bias_corrected_ci("const")

        assert isinstance(ci, tuple)
        assert len(ci) == 2
        assert hasattr(boot, "_boot_results")


# ---------------------------------------------------------------------------
# Etapa 4: Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for bootstrap_sfa() and bootstrap_efficiency() convenience functions."""

    def test_bootstrap_sfa_convenience(self):
        """Test bootstrap_sfa() convenience function."""
        result = _make_mock_result()

        with patch.object(SFABootstrap, "bootstrap_parameters") as mock_bp:
            mock_bp.return_value = {"params_boot": np.zeros((100, 3)), "n_valid": 100}
            out = bootstrap_sfa(result, n_boot=100, method="parametric", seed=42, n_jobs=1)

        assert "params_boot" in out
        mock_bp.assert_called_once()

    def test_bootstrap_efficiency_convenience(self):
        """Test bootstrap_efficiency() convenience function."""
        result = _make_mock_result()

        with patch.object(SFABootstrap, "bootstrap_efficiency") as mock_be:
            mock_be.return_value = pd.DataFrame({"te": [0.8], "ci_lower": [0.7], "ci_upper": [0.9]})
            out = bootstrap_efficiency(
                result, estimator="jlms", n_boot=100, method="pairs", seed=42, n_jobs=1
            )

        assert isinstance(out, pd.DataFrame)
        mock_be.assert_called_once_with(estimator="jlms")

    def test_bootstrap_sfa_passes_ci_level(self):
        """Test bootstrap_sfa passes ci_level through to SFABootstrap."""
        result = _make_mock_result()

        with (
            patch.object(SFABootstrap, "__init__", return_value=None) as mock_init,
            patch.object(SFABootstrap, "bootstrap_parameters", return_value={"ok": True}),
        ):
            bootstrap_sfa(result, n_boot=200, ci_level=0.99, seed=7, n_jobs=2)

        mock_init.assert_called_once_with(
            result=result,
            method="parametric",
            n_boot=200,
            ci_level=0.99,
            seed=7,
            n_jobs=2,
        )
