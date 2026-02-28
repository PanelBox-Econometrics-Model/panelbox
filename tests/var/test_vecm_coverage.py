"""
Coverage tests for panelbox.var.vecm targeting uncovered lines.

These tests specifically target the following uncovered lines from the
coverage report:

- Lines 152-154: max_rank >= K raises ValueError
- Line 170: _compute_residuals early return when already computed
- Line 219: _compute_residuals with p=1 (no lagged differences)
- Line 280: _compute_eigenvalues_entity not enough observations
- Lines 299-300: _compute_eigenvalues_entity lag_vars empty (R0=Y, R1=Y1)
- Lines 323-324: _compute_eigenvalues_entity LinAlgError catch
- Lines 356->349: _panel_trace_statistic T_i <= K+2 skip branch
- Line 363: _panel_trace_statistic valid_entities == 0
- Line 398: _trace_moments deterministic='nc'
- Line 404: _trace_moments deterministic='ct'
- Lines 439->432: _panel_maxeig_statistic branch conditions
- Line 446: _panel_maxeig_statistic valid_entities == 0
- Line 456: _panel_maxeig_statistic Var_LR <= 0
- Line 476: _maxeig_moments deterministic='nc'
- Line 480: _maxeig_moments deterministic='ct'
- Lines 536->520: test_rank loop branch
- Line 833->832: cointegrating_relations beta[0,j]==0 skip
- Lines 1450-1451: eigenvalue problem LinAlgError in estimation
- Lines 1478-1480: fallback beta from eigenvectors (dual LinAlgError)
- Lines 1484->1483: beta normalization skip when abs(beta[0,j]) <= 1e-10
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVARData
from panelbox.var.vecm import (
    CointegrationRankTest,
    PanelVECM,
    PanelVECMResult,
    RankSelectionResult,
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _make_panel(n_entities, n_periods, n_vars=2, seed=42, cointegrated=True):
    """Generate panel data for testing.

    Parameters
    ----------
    n_entities : int
        Number of panel entities.
    n_periods : int
        Number of time periods per entity.
    n_vars : int
        Number of endogenous variables (default 2).
    seed : int
        Random seed for reproducibility.
    cointegrated : bool
        If True, generate cointegrated series.

    Returns
    -------
    pd.DataFrame
        Long-format panel data.
    """
    np.random.seed(seed)
    records = []
    var_names = [f"y{k + 1}" for k in range(n_vars)]

    for i in range(n_entities):
        y1 = np.cumsum(np.random.randn(n_periods))
        if cointegrated:
            # All other variables cointegrated with y1
            ys = [y1] + [y1 + np.random.randn(n_periods) * 0.5 for _ in range(n_vars - 1)]
        else:
            ys = [np.cumsum(np.random.randn(n_periods)) for _ in range(n_vars)]

        for t in range(n_periods):
            row = {"entity": i, "time": t}
            for k, vname in enumerate(var_names):
                row[vname] = ys[k][t]
            records.append(row)

    return pd.DataFrame(records)


def _make_var_data(df, var_names=None, lags=2):
    """Wrap a DataFrame in PanelVARData."""
    if var_names is None:
        var_names = [c for c in df.columns if c not in ("entity", "time")]
    return PanelVARData(
        data=df,
        endog_vars=var_names,
        entity_col="entity",
        time_col="time",
        lags=lags,
    )


# ---------------------------------------------------------------------------
# Tests for CointegrationRankTest.__init__ -- lines 152-154
# ---------------------------------------------------------------------------


class TestMaxRankValidation:
    """CointegrationRankTest raises ValueError when max_rank >= K."""

    def test_max_rank_equal_to_K(self):
        """max_rank == K should raise ValueError."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        with pytest.raises(ValueError, match="max_rank must be < K"):
            CointegrationRankTest(var_data, max_rank=2)

    def test_max_rank_greater_than_K(self):
        """max_rank > K should raise ValueError."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        with pytest.raises(ValueError, match="max_rank must be < K"):
            CointegrationRankTest(var_data, max_rank=5)

    def test_max_rank_valid(self):
        """max_rank < K should succeed."""
        df = _make_panel(5, 20, n_vars=3, seed=42)
        var_data = _make_var_data(df, var_names=["y1", "y2", "y3"], lags=2)

        rank_test = CointegrationRankTest(var_data, max_rank=1)
        assert rank_test.max_rank == 1


# ---------------------------------------------------------------------------
# Tests for _compute_residuals early return -- line 170
# ---------------------------------------------------------------------------


class TestComputeResidualsEarlyReturn:
    """Calling _compute_residuals twice hits the early return on the second call."""

    def test_residuals_computed_flag(self):
        """Second call to _compute_residuals returns immediately."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        assert rank_test._residuals_computed is False

        rank_test._compute_residuals()
        assert rank_test._residuals_computed is True

        # Store references to verify they are not replaced
        r0_ref = rank_test._R0
        r1_ref = rank_test._R1

        # Second call should return immediately (line 170)
        rank_test._compute_residuals()

        assert rank_test._R0 is r0_ref
        assert rank_test._R1 is r1_ref


# ---------------------------------------------------------------------------
# Tests for _compute_residuals with p=1 (no lagged differences) -- line 219
# ---------------------------------------------------------------------------


class TestComputeResidualsNoLagVars:
    """When p=1, lag_vars is empty and fitted = zeros (line 219)."""

    def test_compute_residuals_p_equals_1(self):
        """With lags=1, _compute_residuals uses fitted=zeros path."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=1)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        # Residuals should exist and have correct shape
        assert rank_test._R0 is not None
        assert rank_test._R1 is not None
        assert rank_test._R0.shape[1] == 2
        assert rank_test._R1.shape[1] == 2


# ---------------------------------------------------------------------------
# Tests for _compute_eigenvalues_entity with insufficient data -- line 280
# ---------------------------------------------------------------------------


class TestComputeEigenvaluesEntityShortTimeSeries:
    """Entity with too few observations returns zeros (line 280)."""

    def test_very_short_time_series(self):
        """T=4, p=2 yields T_eff < K+2 after differencing/lags, returns zeros."""
        df = _make_panel(3, 4, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)

        # Each entity has T=4.  After diff + 1 lag of diff + level lag + dropna,
        # only ~1 observation remains, which is < K+2=4.
        eigenvalues = rank_test._compute_eigenvalues_entity(0)
        np.testing.assert_array_equal(eigenvalues, np.zeros(2))


# ---------------------------------------------------------------------------
# Tests for _compute_eigenvalues_entity with p=1 (lag_vars empty)
# -- lines 299-300
# ---------------------------------------------------------------------------


class TestComputeEigenvaluesEntityNoLagVars:
    """When p=1, lag_vars is empty and R0=Y, R1=Y1 (lines 299-300)."""

    def test_eigenvalues_entity_p_equals_1(self):
        """With lags=1, _compute_eigenvalues_entity uses R0=Y, R1=Y1 path."""
        df = _make_panel(5, 30, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=1)

        rank_test = CointegrationRankTest(var_data)
        eigenvalues = rank_test._compute_eigenvalues_entity(0)

        # Should return valid eigenvalues (not all zeros)
        assert eigenvalues.shape == (2,)
        assert np.all(eigenvalues >= 0)
        assert np.all(eigenvalues < 1.0)


# ---------------------------------------------------------------------------
# Tests for _compute_eigenvalues_entity LinAlgError -- lines 323-324
# ---------------------------------------------------------------------------


class TestComputeEigenvaluesEntityLinAlgError:
    """LinAlgError in eigenvalue computation returns zeros (lines 323-324)."""

    def test_linalg_error_returns_zeros(self):
        """Trigger LinAlgError in _compute_eigenvalues_entity."""
        df = _make_panel(5, 30, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)

        # Patch scipy.linalg.inv inside vecm module to raise LinAlgError
        with patch(
            "panelbox.var.vecm.inv",
            side_effect=np.linalg.LinAlgError("Singular matrix"),
        ):
            eigenvalues = rank_test._compute_eigenvalues_entity(0)

        np.testing.assert_array_equal(eigenvalues, np.zeros(2))


# ---------------------------------------------------------------------------
# Tests for _panel_trace_statistic T_i <= K+2 skip -- line 356->349
# Tests for _panel_trace_statistic valid_entities == 0 -- line 363
# ---------------------------------------------------------------------------


class TestPanelTraceStatisticEdgeCases:
    """Edge cases in _panel_trace_statistic."""

    def test_all_entities_too_short(self):
        """All entities have T_i <= K+2, so valid_entities==0 (line 363)."""
        # With K=2, p=2, we need T_i - p - 1 <= K+2 = 4
        # T_i from raw is small (T=5), so T_i - 2 - 1 = 2 <= 4
        df = _make_panel(3, 5, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        LR_bar, Z_trace = rank_test._panel_trace_statistic(0)
        assert LR_bar == 0.0
        assert Z_trace == 0.0

    def test_some_entities_skipped(self):
        """Create a mixed panel where some entities are too short."""
        np.random.seed(42)
        records = []
        # First entity: very short (T=5), will be skipped
        y1 = np.cumsum(np.random.randn(5))
        y2 = y1 + np.random.randn(5) * 0.5
        for t in range(5):
            records.append({"entity": 0, "time": t, "y1": y1[t], "y2": y2[t]})
        # Remaining entities: long enough (T=30)
        for i in range(1, 6):
            y1 = np.cumsum(np.random.randn(30))
            y2 = y1 + np.random.randn(30) * 0.5
            for t in range(30):
                records.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t]})

        df = pd.DataFrame(records)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        LR_bar, Z_trace = rank_test._panel_trace_statistic(0)
        # Should still produce a valid result from the long entities
        assert isinstance(LR_bar, float)
        assert isinstance(Z_trace, float)


# ---------------------------------------------------------------------------
# Tests for _trace_moments deterministic='nc' and 'ct' -- lines 398, 404
# ---------------------------------------------------------------------------


class TestTraceMomentsDeterministic:
    """Test _trace_moments with different deterministic specifications."""

    def test_trace_moments_nc(self):
        """deterministic='nc' uses m*(m+1)/2 formula (line 398)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="nc")
        E_LR, Var_LR = rank_test._trace_moments(0)

        m = 2  # K - rank = 2 - 0
        assert m * (m + 1) / 2 == E_LR
        assert Var_LR == m * (m + 1) / 3

    def test_trace_moments_ct(self):
        """deterministic='ct' uses m*(m+1)/2 + 2*m formula (line 404)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="ct")
        E_LR, Var_LR = rank_test._trace_moments(0)

        m = 2
        assert m * (m + 1) / 2 + 2 * m == E_LR
        assert Var_LR == m * (m + 1) / 3

    def test_trace_moments_c(self):
        """deterministic='c' uses m*(m+1)/2 + m formula (baseline)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="c")
        E_LR, Var_LR = rank_test._trace_moments(0)

        m = 2
        assert m * (m + 1) / 2 + m == E_LR
        assert Var_LR == m * (m + 1) / 3


# ---------------------------------------------------------------------------
# Tests for _panel_maxeig_statistic edge cases -- lines 439->432, 446, 456
# ---------------------------------------------------------------------------


class TestPanelMaxeigStatisticEdgeCases:
    """Edge cases in _panel_maxeig_statistic."""

    def test_all_entities_too_short_maxeig(self):
        """All entities have T_i <= K+2, so valid_entities==0 (line 446)."""
        df = _make_panel(3, 5, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        LR_bar, Z_maxeig = rank_test._panel_maxeig_statistic(0)
        assert LR_bar == 0.0
        assert Z_maxeig == 0.0

    def test_var_lr_zero_branch(self):
        """When Var_LR <= 0, Z_maxeig should be 0.0 (line 456)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        # Patch _maxeig_moments to return Var_LR = 0
        with patch.object(rank_test, "_maxeig_moments", return_value=(2.0, 0.0)):
            _LR_bar, Z_maxeig = rank_test._panel_maxeig_statistic(0)
            assert Z_maxeig == 0.0


# ---------------------------------------------------------------------------
# Tests for _maxeig_moments deterministic='nc' and 'ct' -- lines 476, 480
# ---------------------------------------------------------------------------


class TestMaxeigMomentsDeterministic:
    """Test _maxeig_moments with different deterministic specifications."""

    def test_maxeig_moments_nc(self):
        """deterministic='nc' returns E_LR=1.0 (line 476)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="nc")
        E_LR, Var_LR = rank_test._maxeig_moments(0)

        assert E_LR == 1.0
        assert Var_LR == 2.0

    def test_maxeig_moments_ct(self):
        """deterministic='ct' returns E_LR=3.0 (line 480)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="ct")
        E_LR, Var_LR = rank_test._maxeig_moments(0)

        assert E_LR == 3.0
        assert Var_LR == 2.0

    def test_maxeig_moments_c(self):
        """deterministic='c' returns E_LR=2.0 (baseline)."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="c")
        E_LR, Var_LR = rank_test._maxeig_moments(0)

        assert E_LR == 2.0
        assert Var_LR == 2.0


# ---------------------------------------------------------------------------
# Tests for test_rank loop with deterministic options -- line 536->520
# ---------------------------------------------------------------------------


class TestRankTestWithDeterministicOptions:
    """Test test_rank with nc and ct to hit branch at line 536->520."""

    def test_test_rank_nc(self):
        """test_rank with deterministic='nc'."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="nc")
        results = rank_test.test_rank()

        assert isinstance(results, RankSelectionResult)
        assert len(results.trace_tests) >= 1
        assert len(results.maxeig_tests) >= 1

    def test_test_rank_ct(self):
        """test_rank with deterministic='ct'."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="ct")
        results = rank_test.test_rank()

        assert isinstance(results, RankSelectionResult)
        assert len(results.trace_tests) >= 1
        assert len(results.maxeig_tests) >= 1

    def test_test_rank_with_max_rank_less_than_K_minus_1(self):
        """test_rank with 3 variables and max_rank=1 (less than K-1=2).

        This ensures the loop in test_rank iterates over a reduced range,
        hitting the branch condition at line 536.
        """
        df = _make_panel(10, 25, n_vars=3, seed=42)
        var_data = _make_var_data(df, var_names=["y1", "y2", "y3"], lags=2)

        rank_test = CointegrationRankTest(var_data, max_rank=1)
        results = rank_test.test_rank()

        # With max_rank=1, should test ranks 0 and 1
        assert len(results.trace_tests) == 2
        # maxeig tests: r=0 and r=1 (both < K=3)
        assert len(results.maxeig_tests) == 2


# ---------------------------------------------------------------------------
# Test for cointegrating_relations beta[0,j]==0 skip -- line 833->832
# ---------------------------------------------------------------------------


class TestCointegrationRelationsZeroBeta:
    """cointegrating_relations skips normalization when beta[0,j]==0 (line 833)."""

    def test_beta_zero_first_element(self):
        """When beta[0,j]==0, normalization is skipped."""
        K = 3
        result = PanelVECMResult(
            alpha=np.array([[-0.1, 0.05], [0.05, -0.1], [0.02, 0.02]]),
            # First element of second column is 0 -> skip normalization
            beta=np.array([[1.0, 0.0], [-0.5, 1.0], [0.3, -0.7]]),
            Gamma=[np.eye(K) * 0.1],
            Sigma=np.eye(K),
            residuals=np.random.randn(100, K),
            var_names=["y1", "y2", "y3"],
            rank=2,
            method="ml",
            N=10,
            T_avg=20.0,
        )

        beta_df = result.cointegrating_relations()
        assert isinstance(beta_df, pd.DataFrame)
        assert beta_df.shape == (3, 2)
        # First column: first element should be 1 (normalized)
        assert abs(beta_df.iloc[0, 0] - 1.0) < 1e-10
        # Second column: first element is 0, so no normalization happened
        assert beta_df.iloc[0, 1] == 0.0
        # The rest of column 2 should remain as-is
        assert beta_df.iloc[1, 1] == 1.0
        assert abs(beta_df.iloc[2, 1] - (-0.7)) < 1e-10


# ---------------------------------------------------------------------------
# Tests for _fit_ml LinAlgError in eigenvalue problem -- lines 1450-1451
# ---------------------------------------------------------------------------


class TestFitMLEigenvalueLinAlgError:
    """LinAlgError in eigenvalue problem raises RuntimeError (lines 1450-1451)."""

    def test_eigenvalue_linalg_error(self):
        """Singular data causing LinAlgError in _fit_ml raises RuntimeError."""
        df = _make_panel(5, 20, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        vecm = PanelVECM(var_data, rank=1)

        # Patch np.linalg.eig to raise LinAlgError during the main
        # eigenvalue problem in _fit_ml (first call to eig)
        original_eig = np.linalg.eig

        call_count = [0]

        def failing_eig(M):
            call_count[0] += 1
            # The eigenvalue problem in _fit_ml is the first call to eig
            if call_count[0] == 1:
                raise np.linalg.LinAlgError("Eigenvalue computation failed")
            return original_eig(M)

        with (
            patch("numpy.linalg.eig", side_effect=failing_eig),
            pytest.raises(RuntimeError, match="Failed to solve eigenvalue problem"),
        ):
            vecm.fit(method="ml")


# ---------------------------------------------------------------------------
# Tests for fallback beta from eigenvectors -- lines 1478-1480
# ---------------------------------------------------------------------------


class TestFitMLFallbackBeta:
    """When dual eigenvalue problem fails, fallback to eigenvectors (lines 1478-1480)."""

    def test_dual_linalg_error_fallback(self):
        """LinAlgError in dual problem falls back to original eigenvectors."""
        df = _make_panel(10, 30, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        vecm = PanelVECM(var_data, rank=1)

        # Patch inv so that the second call (S11_inv in the dual problem)
        # raises LinAlgError. The first two inv calls (S00, S11) in the
        # main problem succeed. The third call is S11_inv for the dual.
        original_inv = __import__("scipy.linalg", fromlist=["inv"]).inv
        call_count = [0]

        def selective_inv(M):
            call_count[0] += 1
            # The inv calls in _fit_ml:
            #   1. inv(S00)
            #   2. inv(S11) for main problem
            #   3. inv(S11) for dual problem -> make this fail
            if call_count[0] == 3:
                raise np.linalg.LinAlgError("Singular S11 in dual problem")
            return original_inv(M)

        with patch("panelbox.var.vecm.inv", side_effect=selective_inv):
            result = vecm.fit(method="ml")

        # Should still produce a result using the fallback
        assert isinstance(result, PanelVECMResult)
        assert result.beta.shape == (2, 1)
        assert result.alpha.shape == (2, 1)


# ---------------------------------------------------------------------------
# Tests for beta normalization skip -- line 1484->1483
# ---------------------------------------------------------------------------


class TestBetaNormalizationSkip:
    """Beta normalization is skipped when abs(beta[0,j]) <= 1e-10 (line 1484)."""

    def test_beta_normalization_skip_near_zero(self):
        """When dual eigenvectors have near-zero first element, skip normalization."""
        df = _make_panel(10, 30, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        vecm = PanelVECM(var_data, rank=1)

        # Patch the dual eigenvalue problem to return eigenvectors with
        # near-zero first element
        original_eig = np.linalg.eig

        call_count = [0]

        def patched_eig(M):
            call_count[0] += 1
            eig_vals, eig_vecs = original_eig(M)
            # The second call to eig is the dual problem
            if call_count[0] == 2:
                # Set the first row of eigenvectors to near-zero
                eig_vecs[0, :] = 1e-15
            return eig_vals, eig_vecs

        with patch("numpy.linalg.eig", side_effect=patched_eig):
            result = vecm.fit(method="ml")

        assert isinstance(result, PanelVECMResult)
        assert result.beta.shape == (2, 1)
        # beta[0,j] should NOT be 1.0 since normalization was skipped
        assert abs(result.beta[0, 0]) < 1e-9


# ---------------------------------------------------------------------------
# Integration tests for full rank test with nc/ct deterministic
# ---------------------------------------------------------------------------


class TestFullRankTestWithDeterministic:
    """Full integration tests exercising all deterministic options in test_rank."""

    def test_full_rank_test_nc(self):
        """Full rank test with deterministic='nc' exercises both trace/maxeig moments."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="nc")
        results = rank_test.test_rank()

        assert isinstance(results, RankSelectionResult)
        # Verify p-values are valid
        for test in results.trace_tests:
            assert 0 <= test.p_value <= 1
        for test in results.maxeig_tests:
            assert 0 <= test.p_value <= 1

    def test_full_rank_test_ct(self):
        """Full rank test with deterministic='ct' exercises both trace/maxeig moments."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data, deterministic="ct")
        results = rank_test.test_rank()

        assert isinstance(results, RankSelectionResult)
        for test in results.trace_tests:
            assert 0 <= test.p_value <= 1
        for test in results.maxeig_tests:
            assert 0 <= test.p_value <= 1


# ---------------------------------------------------------------------------
# Tests for _compute_residuals with p=1 exercised via test_rank
# ---------------------------------------------------------------------------


class TestRankTestWithLagsOne:
    """test_rank with lags=1 exercises _compute_residuals no-lag-vars path (line 219)."""

    def test_test_rank_p_equals_1(self):
        """test_rank with lags=1 should complete without errors."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=1)

        rank_test = CointegrationRankTest(var_data)
        results = rank_test.test_rank()

        assert isinstance(results, RankSelectionResult)
        assert results.selected_rank >= 0


# ---------------------------------------------------------------------------
# Test for _panel_maxeig_statistic with entity skipping -- line 439->432
# ---------------------------------------------------------------------------


class TestMaxeigSkipEntity:
    """Test _panel_maxeig_statistic where some entities are skipped."""

    def test_mixed_length_entities_maxeig(self):
        """Mixed panel: short entities are skipped in maxeig computation."""
        np.random.seed(42)
        records = []
        # Short entity (T=5)
        for t in range(5):
            records.append(
                {
                    "entity": 0,
                    "time": t,
                    "y1": np.random.randn(),
                    "y2": np.random.randn(),
                }
            )
        # Long entities
        for i in range(1, 8):
            y1 = np.cumsum(np.random.randn(30))
            y2 = y1 + np.random.randn(30) * 0.5
            for t in range(30):
                records.append({"entity": i, "time": t, "y1": y1[t], "y2": y2[t]})

        df = pd.DataFrame(records)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        LR_bar, Z_maxeig = rank_test._panel_maxeig_statistic(0)
        assert isinstance(LR_bar, float)
        assert isinstance(Z_maxeig, float)


# ---------------------------------------------------------------------------
# Test for PanelVECM with deterministic='nc' and 'ct' in fit
# ---------------------------------------------------------------------------


class TestPanelVECMDeterministicOptions:
    """PanelVECM fit with different deterministic specifications."""

    def test_fit_with_nc(self):
        """PanelVECM fit with deterministic='nc'."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        vecm = PanelVECM(var_data, rank=1, deterministic="nc")
        result = vecm.fit(method="ml")

        assert isinstance(result, PanelVECMResult)
        assert result.deterministic == "nc"

    def test_fit_with_ct(self):
        """PanelVECM fit with deterministic='ct'."""
        df = _make_panel(10, 25, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        vecm = PanelVECM(var_data, rank=1, deterministic="ct")
        result = vecm.fit(method="ml")

        assert isinstance(result, PanelVECMResult)
        assert result.deterministic == "ct"


# ---------------------------------------------------------------------------
# Test for _compute_eigenvalues_entity with lag_vars empty and sufficient obs
# -- lines 299-300 (ensure R0=Y, R1=Y1 path with enough data)
# ---------------------------------------------------------------------------


class TestEigenvaluesEntityNoLagVarsSufficientData:
    """With p=1 and enough observations, lines 299-300 are hit."""

    def test_eigenvalues_no_lag_vars_long_series(self):
        """p=1 with long time series hits R0=Y, R1=Y1 path (lines 299-300)."""
        df = _make_panel(3, 50, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=1)

        rank_test = CointegrationRankTest(var_data)

        # Entity 0 should have enough observations
        eigenvalues = rank_test._compute_eigenvalues_entity(0)
        assert eigenvalues.shape == (2,)
        # Should be valid canonical correlations squared
        assert np.all(eigenvalues >= 0)
        assert np.all(eigenvalues < 1.0)


# ---------------------------------------------------------------------------
# Test for _panel_trace_statistic and _panel_maxeig_statistic
# when valid_entities == 0 from all entities returning zeros
# -- lines 363, 446
# ---------------------------------------------------------------------------


class TestZeroValidEntities:
    """When all entities return zero eigenvalues (too short), stats are (0,0)."""

    def test_trace_and_maxeig_both_zero(self):
        """Both trace and maxeig return (0,0) when all entities too short."""
        # T=4 with p=2 -> effectively 1 observation after lags
        df = _make_panel(2, 4, n_vars=2, seed=42)
        var_data = _make_var_data(df, lags=2)

        rank_test = CointegrationRankTest(var_data)
        rank_test._compute_residuals()

        trace_stat, z_trace = rank_test._panel_trace_statistic(0)
        assert trace_stat == 0.0
        assert z_trace == 0.0

        maxeig_stat, z_maxeig = rank_test._panel_maxeig_statistic(0)
        assert maxeig_stat == 0.0
        assert z_maxeig == 0.0
