"""Deep coverage tests for panelbox.diagnostics.quantile.heterogeneity.

Targets uncovered lines from existing tests:
- Branch 69->73: np.isscalar(var_idx) is True in test_slope_equality
- Lines 151-152: except branch in test_joint_equality (singular V_sum)
- Lines 232-233: except branch in interquantile_range_test (singular V_iqr)
- Line 247: else branch (p_value >= 0.05) in interquantile_range_test
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from panelbox.diagnostics.quantile.heterogeneity import (
    HeterogeneityTests,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tau_result(params, cov_matrix=None):
    """Build a minimal per-tau result namespace."""
    p = len(params)
    if cov_matrix is None:
        cov_matrix = np.eye(p) * 0.01
    return SimpleNamespace(params=params, cov_matrix=cov_matrix)


def _make_result(tau_params_map, cov_matrices=None):
    """Build a mock QuantilePanelResult."""
    results = {}
    for tau, params in tau_params_map.items():
        cov = cov_matrices.get(tau) if cov_matrices else None
        results[tau] = _make_tau_result(params, cov_matrix=cov)
    return SimpleNamespace(results=results)


# ---------------------------------------------------------------------------
# Tests: test_slope_equality with scalar var_idx (line 69->73)
# ---------------------------------------------------------------------------


class TestSlopeEqualityVarIdxBranches:
    """Cover the var_idx branches in test_slope_equality."""

    def test_slope_equality_scalar_var_idx(self):
        """Cover branch 69->70: np.isscalar(var_idx) is True."""
        beta_25 = np.array([1.0, 2.0, -0.5])
        beta_75 = np.array([1.0, 3.0, -0.5])

        result = _make_result({0.25: beta_25, 0.75: beta_75})
        ht = HeterogeneityTests(result)

        # Pass a single int, not a list
        res = ht.test_slope_equality(var_idx=1)
        assert hasattr(res, "statistic")
        assert hasattr(res, "p_value")
        assert res.var_idx == [1]  # Should have been wrapped in list

    def test_slope_equality_list_var_idx(self):
        """Cover branch 69->73: var_idx is not None and not scalar (a list)."""
        beta_25 = np.array([1.0, 2.0, -0.5])
        beta_75 = np.array([1.0, 3.0, -0.5])

        result = _make_result({0.25: beta_25, 0.75: beta_75})
        ht = HeterogeneityTests(result)

        # Pass a list, skipping both 'is None' and 'isscalar' branches
        res = ht.test_slope_equality(var_idx=[0, 2])
        assert hasattr(res, "statistic")
        assert hasattr(res, "p_value")
        assert res.var_idx == [0, 2]


# ---------------------------------------------------------------------------
# Tests: test_joint_equality except branch (lines 151-152)
# ---------------------------------------------------------------------------


class TestJointEqualityExceptBranch:
    """Cover the except branch in test_joint_equality when inv fails."""

    def test_joint_equality_singular_cov(self):
        """Cover lines 151-152: np.linalg.inv raises, falls back to pinv."""
        beta_25 = np.array([1.0, 2.0])
        beta_50 = np.array([1.0, 2.5])
        beta_75 = np.array([1.0, 3.0])

        # Use singular covariance matrices
        singular_cov = np.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1

        result = _make_result(
            {0.25: beta_25, 0.50: beta_50, 0.75: beta_75},
            cov_matrices={
                0.25: singular_cov,
                0.50: singular_cov,
                0.75: singular_cov,
            },
        )
        ht = HeterogeneityTests(result)

        # Force inv to raise LinAlgError so the except uses pinv
        with patch(
            "numpy.linalg.inv",
            side_effect=np.linalg.LinAlgError("Singular matrix"),
        ):
            res = ht.test_joint_equality()

        assert hasattr(res, "statistic")
        assert hasattr(res, "p_value")
        assert res.df > 0


# ---------------------------------------------------------------------------
# Tests: interquantile_range_test except and else branches (lines 232-233, 247)
# ---------------------------------------------------------------------------


class TestInterquantileRangeTestBranches:
    """Cover except and else branches in interquantile_range_test."""

    def test_iqr_test_inv_exception(self):
        """Cover lines 232-233: np.linalg.inv raises, falls back to pinv."""
        beta_25 = np.array([1.0, 2.0, -0.5])
        beta_75 = np.array([1.0, 2.2, -0.3])

        # Singular cov matrix
        singular_cov = np.zeros((3, 3))
        singular_cov[0, 0] = 0.01

        result = _make_result(
            {0.25: beta_25, 0.75: beta_75},
            cov_matrices={0.25: singular_cov, 0.75: singular_cov},
        )
        ht = HeterogeneityTests(result)

        with patch(
            "numpy.linalg.inv",
            side_effect=np.linalg.LinAlgError("Singular matrix"),
        ):
            stat, pval = ht.interquantile_range_test()

        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_iqr_test_no_reject_else_branch(self):
        """Cover line 247: p_value >= 0.05 => 'Cannot reject H0'.

        Use identical coefficients at 0.25 and 0.75 so IQR coefs are ~0.
        """
        beta = np.array([1.0, 2.0, -0.5])

        result = _make_result(
            {0.25: beta, 0.75: beta},
            cov_matrices={
                0.25: np.eye(3) * 0.01,
                0.75: np.eye(3) * 0.01,
            },
        )
        ht = HeterogeneityTests(result)

        stat, pval = ht.interquantile_range_test()

        # Identical coefs => diff = 0 => stat = 0 => p = 1.0
        assert stat == pytest.approx(0.0, abs=1e-10)
        assert pval > 0.05
