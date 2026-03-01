"""
Tests for inference/quantile/bootstrap.py.

Targets coverage gaps in QuantileBootstrap, BootstrapResult, and bootstrap_qr.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from panelbox.inference.quantile.bootstrap import (
    BootstrapResult,
    QuantileBootstrap,
    bootstrap_qr,
)


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------
class MockQuantileModel:
    """Mimics a fitted quantile panel model."""

    def __init__(self, n=100, p=3, n_entities=10):
        np.random.seed(42)
        self.X = np.random.randn(n, p)
        self.y = np.random.randn(n)
        self.nobs = n
        self.k_exog = p
        self.entity_ids = np.repeat(np.arange(n_entities), n // n_entities)
        self.params = np.random.randn(p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_frisch_newton_qr(X, y, tau, max_iter=100, tol=1e-6, verbose=False):
    """A fast fake solver that returns OLS-like coefficients."""
    # Simple least-squares as a stand-in
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return beta, {"converged": True, "n_iter": 5}


def _fake_frisch_newton_qr_diverge(X, y, tau, max_iter=100, tol=1e-6, verbose=False):
    """Fake solver that reports non-convergence."""
    beta = np.zeros(X.shape[1])
    return beta, {"converged": False, "n_iter": max_iter}


def _fake_frisch_newton_qr_raise(X, y, tau, max_iter=100, tol=1e-6, verbose=False):
    """Fake solver that raises an exception."""
    raise RuntimeError("Optimization blew up")


# ===================================================================
# QuantileBootstrap tests
# ===================================================================
class TestBootstrapVerbose:
    """Line 79: verbose=True uses tqdm progress bar."""

    @patch(
        "panelbox.inference.quantile.bootstrap.frisch_newton_qr",
        side_effect=_fake_frisch_newton_qr,
        create=True,
    )
    @patch(
        "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
        side_effect=_fake_frisch_newton_qr,
        create=True,
    )
    def test_bootstrap_verbose_true(self, mock_solver_opt, mock_solver_boot):
        """verbose=True -> line 79 (tqdm branch)."""
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        bs = QuantileBootstrap(model, tau=0.5, n_boot=5, method="pairs", random_state=42)
        with patch(
            "panelbox.inference.quantile.bootstrap.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr,
            create=True,
        ):
            result = bs.bootstrap(n_jobs=1, verbose=True)
        assert result.n_boot == 5
        assert result.boot_params.shape[0] == 5


class TestBootstrapCIMethods:
    """Lines 93-98: ci_method branches."""

    def _run_with_ci_method(self, ci_method):
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        bs = QuantileBootstrap(
            model,
            tau=0.5,
            n_boot=10,
            method="pairs",
            ci_method=ci_method,
            random_state=42,
        )
        with patch(
            "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr,
        ):
            return bs.bootstrap(n_jobs=1, verbose=False)

    def test_bootstrap_bca_ci(self):
        """ci_method='bca' -> lines 93-94."""
        result = self._run_with_ci_method("bca")
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert len(result.ci_lower) == 2

    def test_bootstrap_normal_ci(self):
        """ci_method='normal' -> lines 95-96."""
        result = self._run_with_ci_method("normal")
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert len(result.ci_lower) == 2

    def test_bootstrap_percentile_ci(self):
        """ci_method='percentile' (default)."""
        result = self._run_with_ci_method("percentile")
        assert result.ci_lower is not None
        assert result.ci_upper is not None

    def test_bootstrap_unknown_ci(self):
        """ci_method='unknown' -> lines 97-98 (ValueError)."""
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        bs = QuantileBootstrap(
            model,
            tau=0.5,
            n_boot=5,
            method="pairs",
            ci_method="unknown",
            random_state=42,
        )
        with (
            patch(
                "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
                side_effect=_fake_frisch_newton_qr,
            ),
            pytest.raises(ValueError, match="Unknown CI method"),
        ):
            bs.bootstrap(n_jobs=1, verbose=False)


class TestBootstrapMethods:
    """Lines 114-123: bootstrap method branches."""

    def _run_with_method(self, method):
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        bs = QuantileBootstrap(
            model,
            tau=0.5,
            n_boot=5,
            method=method,
            random_state=42,
        )
        with patch(
            "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr,
        ):
            return bs.bootstrap(n_jobs=1, verbose=False)

    def test_bootstrap_subsampling(self):
        """method='subsampling' -> line 120-121."""
        result = self._run_with_method("subsampling")
        assert result.method == "subsampling"
        assert result.boot_params.shape == (5, 2)

    def test_bootstrap_wild(self):
        """method='wild' -> line 118-119."""
        result = self._run_with_method("wild")
        assert result.method == "wild"
        assert result.boot_params.shape == (5, 2)

    def test_bootstrap_pairs(self):
        """method='pairs' -> line 116-117."""
        result = self._run_with_method("pairs")
        assert result.method == "pairs"
        assert result.boot_params.shape == (5, 2)

    def test_bootstrap_cluster(self):
        """method='cluster' -> line 114-115."""
        result = self._run_with_method("cluster")
        assert result.method == "cluster"
        assert result.boot_params.shape == (5, 2)


class TestBootstrapOptimizationFailure:
    """Lines 141-143: Exception handler in _single_bootstrap."""

    def test_bootstrap_optimization_failure(self):
        """Mock frisch_newton_qr to raise -> returns NaN array."""
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        bs = QuantileBootstrap(
            model,
            tau=0.5,
            n_boot=3,
            method="pairs",
            random_state=42,
        )
        with patch(
            "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr_raise,
        ):
            result = bs.bootstrap(n_jobs=1, verbose=False)
        # All bootstrap replications should be NaN
        assert np.all(np.isnan(result.boot_params))

    def test_bootstrap_non_convergence(self):
        """Non-converged solver -> returns NaN array (lines 133-137)."""
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        bs = QuantileBootstrap(
            model,
            tau=0.5,
            n_boot=3,
            method="pairs",
            random_state=42,
        )
        with patch(
            "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr_diverge,
        ):
            result = bs.bootstrap(n_jobs=1, verbose=False)
        assert np.all(np.isnan(result.boot_params))


# ===================================================================
# BootstrapResult tests
# ===================================================================
class TestBootstrapResultSummary:
    """Lines 241-250: BootstrapResult.summary()."""

    def test_bootstrap_result_summary(self, capsys):
        """summary() prints formatted table."""
        boot_params = np.random.randn(100, 3)
        ci_lower = np.array([-1.0, -2.0, -0.5])
        ci_upper = np.array([1.0, 2.0, 0.5])
        se = np.array([0.5, 1.0, 0.25])
        original_params = np.array([0.1, 0.2, 0.3])

        result = BootstrapResult(
            boot_params=boot_params,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se=se,
            original_params=original_params,
            method="cluster",
            n_boot=100,
        )

        result.summary(var_names=["alpha", "beta", "gamma"])
        captured = capsys.readouterr()
        assert "Bootstrap Results" in captured.out
        assert "cluster" in captured.out
        assert "alpha" in captured.out
        assert "beta" in captured.out
        assert "gamma" in captured.out

    def test_bootstrap_result_summary_default_names(self, capsys):
        """summary() with no var_names -> uses X0, X1, ... (line 242)."""
        boot_params = np.random.randn(50, 2)
        ci_lower = np.array([-1.0, -2.0])
        ci_upper = np.array([1.0, 2.0])
        se = np.array([0.5, 1.0])

        result = BootstrapResult(
            boot_params=boot_params,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            se=se,
            original_params=np.array([0.1, 0.2]),
            method="pairs",
            n_boot=50,
        )

        result.summary()  # No var_names
        captured = capsys.readouterr()
        assert "X0" in captured.out
        assert "X1" in captured.out


# ===================================================================
# bootstrap_qr wrapper tests
# ===================================================================
class TestBootstrapQRWrapper:
    """Lines 277-278: bootstrap_qr wrapper function."""

    def test_bootstrap_qr_wrapper(self):
        """bootstrap_qr() creates QuantileBootstrap and calls bootstrap()."""
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        with patch(
            "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr,
        ):
            result = bootstrap_qr(
                model,
                tau=0.5,
                n_boot=5,
                method="pairs",
                n_jobs=1,
                verbose=False,
            )
        assert isinstance(result, BootstrapResult)
        assert result.n_boot == 5
        assert result.method == "pairs"
        assert result.boot_params.shape == (5, 2)

    def test_bootstrap_qr_wrapper_cluster(self):
        """bootstrap_qr() with cluster method."""
        model = MockQuantileModel(n=50, p=2, n_entities=5)
        with patch(
            "panelbox.optimization.quantile.interior_point.frisch_newton_qr",
            side_effect=_fake_frisch_newton_qr,
        ):
            result = bootstrap_qr(
                model,
                tau=0.5,
                n_boot=5,
                method="cluster",
                n_jobs=1,
                verbose=False,
            )
        assert isinstance(result, BootstrapResult)
        assert result.method == "cluster"
