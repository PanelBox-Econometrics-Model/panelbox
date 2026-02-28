"""Coverage tests for panelbox.frontier.panel_utils module.

Targets uncovered lines identified via --cov-report=term-missing:
- Lines 63-64: Small m3 branch in get_panel_starting_values
- Line 71->77: Constant column bias correction
- Line 91: bc95 model_type with Z
- Lines 106-107: LinAlgError fallback in bc95 delta init
- Line 122: Unknown model_type ValueError
- Line 242: lr_test_kumbhakar_constant function
- Lines 281-306: compute_panel_efficiency_bc92 function
- Lines 338-358: compute_panel_efficiency_bc95 function
- Lines 376-410: validate_panel_structure function
"""

from unittest.mock import patch

import numpy as np
import pytest

from panelbox.frontier.panel_utils import (
    compute_panel_efficiency_bc92,
    compute_panel_efficiency_bc95,
    get_panel_starting_values,
    lr_test_kumbhakar_constant,
    validate_panel_structure,
)


class TestGetPanelStartingValues:
    """Cover all branches in get_panel_starting_values."""

    def _make_data(self, N=20, T=5, with_constant=True, seed=42):
        """Create minimal panel data."""
        np.random.seed(seed)
        n = N * T
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        if with_constant:
            const = np.ones(n)
            x = np.random.normal(0, 1, n)
            X = np.column_stack([const, x])
        else:
            x = np.random.normal(0, 1, n)
            X = x.reshape(-1, 1)

        v = np.random.normal(0, 0.1, n)
        u = np.abs(np.random.normal(0, 0.2, n))
        y = X @ np.ones(X.shape[1]) + v - u

        return y, X, entity_id, time_id

    def test_pitt_lee_model(self):
        """Cover line 82: pitt_lee model_type."""
        y, X, eid, tid = self._make_data()
        theta = get_panel_starting_values("pitt_lee", y, X, eid, tid)
        # pitt_lee: [beta, ln_sv, ln_su, mu]
        assert len(theta) == X.shape[1] + 3

    def test_bc92_model(self):
        """Cover line 86: bc92 model_type."""
        y, X, eid, tid = self._make_data()
        theta = get_panel_starting_values("bc92", y, X, eid, tid)
        # bc92: [beta, ln_sv, ln_su, mu, eta]
        assert len(theta) == X.shape[1] + 4

    def test_bc95_model(self):
        """Cover lines 88-109: bc95 model_type with Z."""
        y, X, eid, tid = self._make_data()
        n = len(y)
        Z = np.random.normal(0, 1, (n, 2))
        theta = get_panel_starting_values("bc95", y, X, eid, tid, Z=Z)
        # bc95: [beta, ln_sv, ln_su, delta_0, delta_1]
        assert len(theta) == X.shape[1] + 2 + Z.shape[1]

    def test_bc95_no_z_raises(self):
        """Cover line 91: bc95 without Z raises ValueError."""
        y, X, eid, tid = self._make_data()
        with pytest.raises(ValueError, match="Z must be provided"):
            get_panel_starting_values("bc95", y, X, eid, tid, Z=None)

    def test_bc95_linalg_error_fallback(self):
        """Cover lines 106-107: LinAlgError in lstsq fallback."""
        y, X, eid, tid = self._make_data()
        n = len(y)
        Z = np.random.normal(0, 1, (n, 2))

        original_lstsq = np.linalg.lstsq
        call_count = [0]

        def mock_lstsq(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                # Second call is for delta_init = lstsq(Z, log_u_hat)
                raise np.linalg.LinAlgError("singular matrix")
            return original_lstsq(*args, **kwargs)

        with patch("numpy.linalg.lstsq", side_effect=mock_lstsq):
            theta = get_panel_starting_values("bc95", y, X, eid, tid, Z=Z)

        # Should fallback to zeros for delta
        assert len(theta) == X.shape[1] + 2 + Z.shape[1]

    def test_kumbhakar_model(self):
        """Cover lines 111-113: kumbhakar model_type."""
        y, X, eid, tid = self._make_data()
        theta = get_panel_starting_values("kumbhakar", y, X, eid, tid)
        # kumbhakar: [beta, ln_sv, ln_su, mu, b, c]
        assert len(theta) == X.shape[1] + 5

    def test_lee_schmidt_model(self):
        """Cover lines 115-119: lee_schmidt model_type."""
        y, X, eid, tid = self._make_data()
        T = 5
        theta = get_panel_starting_values("lee_schmidt", y, X, eid, tid)
        # lee_schmidt: [beta, ln_sv, ln_su, mu, delta_1..delta_{T-1}]
        assert len(theta) == X.shape[1] + 3 + (T - 1)

    def test_unknown_model_raises(self):
        """Cover line 122: unknown model_type raises ValueError."""
        y, X, eid, tid = self._make_data()
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_panel_starting_values("invalid_model", y, X, eid, tid)

    def test_small_m3_branch(self):
        """Cover lines 63-64: small m3 (abs(m3) < 1e-10)."""
        N, T = 20, 5
        n = N * T
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        # Create perfectly symmetric residuals so m3 = 0 exactly.
        # y = X @ beta exactly, so residuals = 0 -> m3 = 0
        const = np.ones(n)
        x = np.linspace(-1, 1, n)
        X = np.column_stack([const, x])

        y = X @ np.array([1.0, 0.5])  # Exact fit, residuals = 0

        theta = get_panel_starting_values("pitt_lee", y, X, entity_id, time_id)
        assert len(theta) == X.shape[1] + 3

    def test_constant_column_bias_correction(self):
        """Cover line 71->77: constant column bias correction."""
        y, X, eid, tid = self._make_data(with_constant=True)
        # With constant, first column is all 1s -> bias correction applied
        theta = get_panel_starting_values("pitt_lee", y, X, eid, tid, sign=1)
        assert len(theta) == X.shape[1] + 3

    def test_no_constant_column(self):
        """Cover branch 71->77 False: no constant column."""
        y, X, eid, tid = self._make_data(with_constant=False)
        theta = get_panel_starting_values("pitt_lee", y, X, eid, tid, sign=1)
        assert len(theta) == X.shape[1] + 3


class TestLRTestKumbhakarConstant:
    """Cover line 242: lr_test_kumbhakar_constant function."""

    def test_kumbhakar_constant_reject(self):
        """Test LR test rejecting time-invariance hypothesis."""
        result = lr_test_kumbhakar_constant(
            loglik_kumbhakar=-100.0,
            loglik_pitt_lee=-120.0,
        )
        assert result["LR_stat"] == 40.0
        assert result["df"] == 2
        assert result["reject_H0"]
        assert result["p_value"] < 0.05

    def test_kumbhakar_constant_not_reject(self):
        """Test LR test not rejecting (models very similar)."""
        result = lr_test_kumbhakar_constant(
            loglik_kumbhakar=-100.0,
            loglik_pitt_lee=-100.5,
        )
        assert result["LR_stat"] == 1.0
        assert result["df"] == 2
        assert not result["reject_H0"]


class TestComputePanelEfficiencyBC92:
    """Cover lines 281-306: compute_panel_efficiency_bc92 function."""

    def test_bc92_efficiency_basic(self):
        """Cover the full function with typical inputs."""
        np.random.seed(42)
        N, T = 10, 5
        n = N * T
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        epsilon = np.random.normal(0, 0.2, n)

        TE = compute_panel_efficiency_bc92(
            epsilon=epsilon,
            entity_id=entity_id,
            time_id=time_id,
            sigma_v=0.1,
            sigma_u=0.2,
            mu=0.0,
            eta=0.05,
            sign=1,
        )

        assert TE.shape == (n,)
        assert np.all(TE >= 0)
        assert np.all(TE <= 1)
        assert np.all(np.isfinite(TE))

    def test_bc92_efficiency_cost_frontier(self):
        """Cover with sign=-1 (cost frontier)."""
        np.random.seed(42)
        N, T = 10, 5
        n = N * T
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        epsilon = np.random.normal(0, 0.2, n)

        TE = compute_panel_efficiency_bc92(
            epsilon=epsilon,
            entity_id=entity_id,
            time_id=time_id,
            sigma_v=0.1,
            sigma_u=0.2,
            mu=0.1,
            eta=-0.05,
            sign=-1,
        )

        assert TE.shape == (n,)
        assert np.all(TE >= 0)
        assert np.all(TE <= 1)


class TestComputePanelEfficiencyBC95:
    """Cover lines 338-358: compute_panel_efficiency_bc95 function."""

    def test_bc95_efficiency_basic(self):
        """Cover the full function with typical inputs."""
        np.random.seed(42)
        n = 100
        m = 2

        epsilon = np.random.normal(0, 0.2, n)
        Z = np.random.normal(0, 1, (n, m))
        delta = np.array([0.1, 0.2])

        TE = compute_panel_efficiency_bc95(
            epsilon=epsilon,
            Z=Z,
            delta=delta,
            sigma_v=0.1,
            sigma_u=0.2,
            sign=1,
        )

        assert TE.shape == (n,)
        assert np.all(TE >= 0)
        assert np.all(TE <= 1)
        assert np.all(np.isfinite(TE))

    def test_bc95_efficiency_cost_frontier(self):
        """Cover with sign=-1 (cost frontier)."""
        np.random.seed(42)
        n = 100
        Z = np.random.normal(0, 1, (n, 1))
        delta = np.array([0.05])
        epsilon = np.random.normal(0, 0.2, n)

        TE = compute_panel_efficiency_bc95(
            epsilon=epsilon,
            Z=Z,
            delta=delta,
            sigma_v=0.1,
            sigma_u=0.2,
            sign=-1,
        )

        assert TE.shape == (n,)
        assert np.all(TE >= 0)
        assert np.all(TE <= 1)


class TestValidatePanelStructure:
    """Cover lines 376-410: validate_panel_structure function."""

    def test_balanced_panel(self):
        """Cover balanced panel path."""
        N, T = 30, 5
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        result = validate_panel_structure(entity_id, time_id)

        assert result["N"] == N
        assert result["T"] == T
        assert result["n"] == N * T
        assert result["is_balanced"]
        assert result["min_obs_per_entity"] == T
        assert result["max_obs_per_entity"] == T

    def test_unbalanced_panel(self):
        """Cover unbalanced panel warning path (line 401-405)."""
        # Create unbalanced panel: some entities have fewer obs
        entity_id = np.array([0, 0, 0, 1, 1, 2, 2, 2])
        time_id = np.array([0, 1, 2, 0, 1, 0, 1, 2])

        result = validate_panel_structure(entity_id, time_id)

        assert not result["is_balanced"]
        assert result["min_obs_per_entity"] == 2
        assert result["max_obs_per_entity"] == 3
        assert any("unbalanced" in w.lower() for w in result["warnings"])

    def test_small_n_warning(self):
        """Cover line 407-408: small N warning."""
        N, T = 10, 5
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        result = validate_panel_structure(entity_id, time_id)

        assert result["N"] == 10
        assert any("N = 10" in w for w in result["warnings"])

    def test_small_T_warning(self):
        """Cover lines 395-399: T below min_T warning."""
        N, T = 30, 2
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        result = validate_panel_structure(entity_id, time_id, min_T=3)

        assert result["T"] == 2
        assert any("T = 2" in w for w in result["warnings"])

    def test_large_panel_no_warnings(self):
        """Cover all branches - large, balanced, enough T."""
        N, T = 50, 10
        entity_id = np.repeat(np.arange(N), T)
        time_id = np.tile(np.arange(T), N)

        result = validate_panel_structure(entity_id, time_id, min_T=3)

        assert result["is_balanced"]
        assert result["warnings"] == []
