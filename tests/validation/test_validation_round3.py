"""
Round 3 coverage tests for panelbox validation modules.

Targets uncovered lines/branches in:
- validation/spatial/spatial_hausman.py
- validation/spatial/local_moran.py
- validation/spatial/lm_tests.py
- validation/cross_sectional_dependence/frees.py
- validation/robustness/influence.py
- validation/robustness/checks.py
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rook_weights(n):
    """Create a simple row-standardized rook weight matrix."""
    W = np.zeros((n, n))
    for i in range(n):
        if i > 0:
            W[i, i - 1] = 1
        if i < n - 1:
            W[i, i + 1] = 1
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


def _make_mock_result(params, bse=None, cov=None, cov_params=None):
    """Create a mock result object for SpatialHausmanTest."""

    class MockResult:
        pass

    r = MockResult()
    r.params = params
    if bse is not None:
        r.bse = bse
    if cov is not None:
        r.cov = cov
    if cov_params is not None:
        r.cov_params = cov_params
    return r


def _make_ols_result(n, with_panel=False, add_spatial_lag=False):
    """Create a mock OLS result for LM spatial tests."""
    np.random.seed(42)
    y = np.random.randn(n)
    X = np.column_stack([np.ones(n), np.random.randn(n)])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat

    if add_spatial_lag:
        W = _rook_weights(n)
        resid = resid + 0.5 * W @ y

    result = SimpleNamespace(
        resid=resid,
        fittedvalues=y_hat,
        nobs=n,
        params=beta,
        bse=np.array([0.1, 0.2]),
    )

    if with_panel:
        model = SimpleNamespace(N=n, T=1)
        model.data = SimpleNamespace(y=y)
        result.model = model

    return result


# ===========================================================================
# Tests for SpatialHausmanTest
# ===========================================================================
class TestSpatialHausmanUncoveredBranches:
    """Cover uncovered lines in spatial_hausman.py."""

    def test_run_raises_when_param_lengths_differ(self):
        """Cover line 145: raise ValueError for mismatched param lengths."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        # Use Series with non-overlapping names to get positional matching
        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
            bse=pd.Series([0.1, 0.1, 0.1], index=["a", "b", "c"]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["d", "e"]),
            bse=pd.Series([0.1, 0.1], index=["d", "e"]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        # Positional matching gives min(3,2) = 2 common params.
        # Pass subset=[0,1,2] -> r2 only has 2 params -> IndexError or ValueError
        with pytest.raises((ValueError, IndexError)):
            test.run(subset=[0, 1, 2])

    def test_linalg_error_branch(self):
        """Cover lines 170-174: LinAlgError fallback to pinv."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["x1", "x2"]),
            bse=pd.Series([0.1, 0.1], index=["x1", "x2"]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["x1", "x2"]),
            bse=pd.Series([0.1, 0.1], index=["x1", "x2"]),
        )
        test = SpatialHausmanTest(r1, r2)

        # Patch np.linalg.inv to raise LinAlgError
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("singular")):
            result = test.run()
        assert result.statistic >= 0
        assert 0 <= result.pvalue <= 1

    def test_get_params_positional_series(self):
        """Cover line 225: positional indexing on pd.Series params."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        # Use array params for r1 and series for r2 so names don't match
        # -> positional matching
        r1 = _make_mock_result(
            params=np.array([1.0, 2.0]),
            bse=np.array([0.1, 0.2]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["a", "b"]),
            bse=pd.Series([0.1, 0.2], index=["a", "b"]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        result = test.run()
        assert result.statistic >= 0

    def test_get_params_positional_array(self):
        """Cover line 233: positional indexing on numpy array params."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        # r1 has array params (-> beta_0, beta_1), r2 has Series with different names
        # This forces use_positional=True
        r1 = _make_mock_result(
            params=np.array([1.0, 2.0]),
            bse=np.array([0.1, 0.2]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["x", "y"]),
            bse=pd.Series([0.1, 0.2], index=["x", "y"]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        result = test.run()
        assert result.statistic >= 0

    def test_get_covariance_bse_positional_series(self):
        """Cover line 251: bse as Series with positional indexing."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        # Non-overlapping param names -> positional matching
        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["a", "b"]),
            bse=pd.Series([0.1, 0.2], index=["a", "b"]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["c", "d"]),
            bse=pd.Series([0.15, 0.25], index=["c", "d"]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        result = test.run()
        assert result.statistic >= 0

    def test_get_covariance_bse_positional_array(self):
        """Cover line 257: bse as array with positional indexing."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        # Non-overlapping names force positional matching
        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["a", "b"]),
            bse=np.array([0.1, 0.2]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["c", "d"]),
            bse=np.array([0.15, 0.25]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        result = test.run()
        assert result.statistic >= 0

    def test_get_covariance_dataframe_positional(self):
        """Cover line 266: cov_params as DataFrame with positional indexing."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        cov1 = pd.DataFrame(
            [[0.01, 0.001], [0.001, 0.04]],
            index=["a", "b"],
            columns=["a", "b"],
        )
        cov2 = pd.DataFrame(
            [[0.02, 0.002], [0.002, 0.08]],
            index=["c", "d"],
            columns=["c", "d"],
        )
        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["a", "b"]),
            cov_params=cov1,
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["c", "d"]),
            cov_params=cov2,
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        result = test.run()
        assert result.statistic >= 0

    def test_get_covariance_array_positional(self):
        """Cover lines 273, 280: numpy cov with positional indexing."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        cov = np.array([[0.01, 0.001], [0.001, 0.04]])
        # Non-overlapping names force positional matching
        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["a", "b"]),
            cov_params=cov,
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["c", "d"]),
            cov_params=cov * 2,
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        result = test.run()
        assert result.statistic >= 0

    def test_get_covariance_no_cov_raises(self):
        """Cover line 261: ValueError when no covariance source."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        r1 = _make_mock_result(params=pd.Series([1.0], index=["x1"]))
        r2 = _make_mock_result(
            params=pd.Series([1.5], index=["x1"]),
            bse=pd.Series([0.1], index=["x1"]),
        )
        test = SpatialHausmanTest(r1, r2)
        with pytest.raises(ValueError, match="Cannot extract covariance"):
            test.run()

    def test_get_model_name_with_class(self):
        """Cover lines 288-290: model with __class__ attribute."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        r1 = _make_mock_result(
            params=pd.Series([1.0], index=["x1"]),
            bse=pd.Series([0.1], index=["x1"]),
        )
        r1.model = SimpleNamespace()  # Has __class__ -> returns "SimpleNamespace"
        r2 = _make_mock_result(
            params=pd.Series([1.5], index=["x1"]),
            bse=pd.Series([0.1], index=["x1"]),
        )
        test = SpatialHausmanTest(r1, r2)
        name = test._get_model_name(r1)
        assert name == "SimpleNamespace"

    def test_get_model_name_fallback(self):
        """Cover line 295: 'Model' fallback when no model attribute."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        class NoModel:
            pass

        r1 = _make_mock_result(
            params=pd.Series([1.0], index=["x1"]),
            bse=pd.Series([0.1], index=["x1"]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5], index=["x1"]),
            bse=pd.Series([0.1], index=["x1"]),
        )
        # Patch _get_model_name to return "Model" for objects without model attr.
        # Actually just ensure no model attr exists.
        test = SpatialHausmanTest(r1, r2)
        name = test._get_model_name(r1)
        # MockResult class name
        assert name == "MockResult"

    def test_summary_auto_runs(self):
        """Cover lines 307-311: summary() calls run() when _last_result missing."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["x1", "x2"]),
            bse=pd.Series([0.1, 0.2], index=["x1", "x2"]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["x1", "x2"]),
            bse=pd.Series([0.15, 0.25], index=["x1", "x2"]),
        )
        test = SpatialHausmanTest(r1, r2)
        # Don't call run() first - summary() should auto-run
        summary = test.summary()
        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 2

    def test_summary_positional_se_series(self):
        """Cover lines 318, 329: summary with positional + Series bse."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["a", "b"]),
            bse=pd.Series([0.1, 0.2], index=["a", "b"]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["c", "d"]),
            bse=pd.Series([0.15, 0.25], index=["c", "d"]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        summary = test.summary()
        assert isinstance(summary, pd.DataFrame)

    def test_summary_no_bse_gives_nan(self):
        """Cover lines 324, 335: summary with no bse returns NaN SE."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        cov = np.array([[0.01, 0.001], [0.001, 0.04]])
        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["x1", "x2"]),
            cov_params=cov,
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["x1", "x2"]),
            cov_params=cov * 2,
        )
        test = SpatialHausmanTest(r1, r2)
        summary = test.summary()
        # SE columns should be NaN since no bse attribute
        assert summary.iloc[0]["MockResult_se"] != summary.iloc[0]["MockResult_se"]  # NaN

    def test_summary_positional_array_bse(self):
        """Cover line 333: summary with positional + array bse."""
        from panelbox.validation.spatial.spatial_hausman import SpatialHausmanTest

        r1 = _make_mock_result(
            params=pd.Series([1.0, 2.0], index=["a", "b"]),
            bse=np.array([0.1, 0.2]),
        )
        r2 = _make_mock_result(
            params=pd.Series([1.5, 2.5], index=["c", "d"]),
            bse=np.array([0.15, 0.25]),
        )
        test = SpatialHausmanTest(r1, r2)
        assert test.use_positional
        summary = test.summary()
        assert len(summary) == 2


# ===========================================================================
# Tests for LocalMoranI
# ===========================================================================
class TestLocalMoranUncoveredBranches:
    """Cover uncovered lines in local_moran.py."""

    def test_not_enough_data(self):
        """Cover lines 133-146: n_valid < 3 returns NaN DataFrame."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        v = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
        W = _rook_weights(5)
        moran = LocalMoranI(v, W)
        result = moran.run(seed=42)
        assert result["Ii"].isna().sum() >= 4
        assert result["cluster_type"].iloc[0] == "Not significant"

    def test_constant_variable(self):
        """Cover line 154: v_std == 0 branch."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        v = np.array([5.0, 5.0, 5.0, 5.0, 5.0])
        W = _rook_weights(5)
        moran = LocalMoranI(v, W)
        result = moran.run(seed=42, n_permutations=10)
        assert isinstance(result, pd.DataFrame)

    def test_invalid_unit_in_mask(self):
        """Cover lines 166-167, 223-225: invalid values in variable."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        v = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        W = _rook_weights(5)
        moran = LocalMoranI(v, W)
        result = moran.run(seed=42, n_permutations=10)
        assert np.isnan(result.loc[2, "Ii"])

    def test_expected_value_single_valid(self):
        """Cover line 207: n_valid <= 1 returns NaN."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        v = np.array([1.0, np.nan, np.nan, np.nan, np.nan])
        W = _rook_weights(5)
        moran = LocalMoranI(v, W)
        result = moran._compute_expected_values(np.array([True, False, False, False, False]))
        # Only 1 valid -> should return NaN
        assert np.isnan(result[0])

    def test_variance_too_few_valid(self):
        """Cover line 214: n_valid <= 2 returns NaN."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        v = np.array([1.0, 2.0, np.nan, np.nan, np.nan])
        W = _rook_weights(5)
        moran = LocalMoranI(v, W)
        z = np.array([0.5, 0.5, 0.0, 0.0, 0.0])
        mask = np.array([True, True, False, False, False])
        result = moran._compute_variance(z, W, mask)
        assert np.isnan(result[0])

    def test_no_neighbors_pvalue(self):
        """Cover lines 267-269: entity with no neighbors gets pvalue=1."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        # Isolated node (no connections in W)
        W = np.zeros((5, 5))
        W[0, 1] = 1
        W[1, 0] = 1
        W[2, 3] = 1
        W[3, 2] = 1
        # Node 4 has no neighbors
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        moran = LocalMoranI(v, W)
        result = moran.run(seed=42, n_permutations=10)
        assert result.loc[4, "pvalue"] == 1.0

    def test_plot_matplotlib(self):
        """Cover lines 411-444: matplotlib plotting backend."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.validation.spatial.local_moran import LocalMoranI

        np.random.seed(42)
        v = np.random.randn(8)
        W = _rook_weights(8)
        moran = LocalMoranI(v, W)
        results = moran.run(seed=42, n_permutations=20)
        fig = moran.plot_clusters(results, backend="matplotlib")
        assert fig is not None
        plt.close("all")

    def test_plot_plotly(self):
        """Cover lines 371-409: plotly backend without gdf."""
        pytest.importorskip("plotly")
        from panelbox.validation.spatial.local_moran import LocalMoranI

        np.random.seed(42)
        v = np.random.randn(8)
        W = _rook_weights(8)
        moran = LocalMoranI(v, W)
        results = moran.run(seed=42, n_permutations=20)
        fig = moran.plot_clusters(results, backend="plotly")
        assert fig is not None

    def test_plot_plotly_with_gdf(self):
        """Cover line 374-377: plotly with gdf (pass through)."""
        pytest.importorskip("plotly")
        from panelbox.validation.spatial.local_moran import LocalMoranI

        np.random.seed(42)
        v = np.random.randn(5)
        W = _rook_weights(5)
        moran = LocalMoranI(v, W)
        results = moran.run(seed=42, n_permutations=10)
        # gdf is provided but the current code just passes (no choropleth impl)
        fig = moran.plot_clusters(results, gdf="dummy_gdf", backend="plotly")
        assert fig is None  # passes through, returns None

    def test_many_entities_xaxis(self):
        """Cover line 440: len(entities) > 30 skips xtick labels."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.validation.spatial.local_moran import LocalMoranI

        np.random.seed(42)
        n = 35
        v = np.random.randn(n)
        W = _rook_weights(n)
        moran = LocalMoranI(v, W)
        results = moran.run(seed=42, n_permutations=10)
        fig = moran.plot_clusters(results, backend="matplotlib")
        assert fig is not None
        plt.close("all")

    def test_panel_data_init(self):
        """Cover lines 64-88: panel data initialization path."""
        from panelbox.validation.spatial.local_moran import LocalMoranI

        N = 5
        T = 3
        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        values = np.random.randn(N * T)
        W = _rook_weights(N)

        moran = LocalMoranI(values, W, entity_index=entities, time_index=times)
        assert moran.is_panel
        assert moran.N == N
        result = moran.run(seed=42, n_permutations=10)
        assert len(result) == N


# ===========================================================================
# Tests for LM Tests
# ===========================================================================
class TestLMTestsUncoveredBranches:
    """Cover uncovered lines/branches in lm_tests.py."""

    def test_lm_lag_panel_setup(self):
        """Cover lines 65-70, 76-84: panel weight setup for LMLagTest."""
        from panelbox.validation.spatial.lm_tests import LMLagTest

        N = 5
        T = 3
        n = N * T
        np.random.seed(42)
        y = np.random.randn(n)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=y_hat,
            nobs=n,
            params=beta,
            bse=np.array([0.1, 0.2]),
        )
        result.model = SimpleNamespace(N=N, T=T, data=SimpleNamespace(y=y))

        W = _rook_weights(N)
        test = LMLagTest(result, W)
        run_result = test.run()
        assert run_result.pvalue >= 0

    def test_lm_lag_w_dimension_error(self):
        """Cover line 72: W incompatible with N."""
        from panelbox.validation.spatial.lm_tests import LMLagTest

        n = 15
        np.random.seed(42)
        y = np.random.randn(n)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=y_hat,
            nobs=n,
        )
        result.model = SimpleNamespace(N=5, T=3)

        # W is 3x3, but N=5 -> mismatch
        W = _rook_weights(3)
        with pytest.raises(ValueError, match="incompatible"):
            LMLagTest(result, W)

    def test_lm_lag_no_model_w_mismatch(self):
        """Cover line 74: W != n without model."""
        from panelbox.validation.spatial.lm_tests import LMLagTest

        result = _make_ols_result(10)
        W = _rook_weights(5)  # 5x5 but n=10
        with pytest.raises(ValueError, match="!="):
            LMLagTest(result, W)

    def test_lm_error_panel_setup(self):
        """Cover lines 184-201: LMErrorTest panel path."""
        from panelbox.validation.spatial.lm_tests import LMErrorTest

        N = 5
        T = 3
        n = N * T
        np.random.seed(42)
        resid = np.random.randn(n)

        result = SimpleNamespace(resid=resid, nobs=n)
        result.model = SimpleNamespace(N=N, T=T)

        W = _rook_weights(N)
        test = LMErrorTest(result, W)
        run_result = test.run()
        assert run_result.pvalue >= 0

    def test_lm_error_w_dimension_errors(self):
        """Cover lines 191, 193: LMErrorTest W dimension errors."""
        from panelbox.validation.spatial.lm_tests import LMErrorTest

        n = 15
        resid = np.random.randn(n)
        result = SimpleNamespace(resid=resid, nobs=n)

        # With model but W incompatible with N
        result.model = SimpleNamespace(N=5, T=3)
        W = _rook_weights(3)
        with pytest.raises(ValueError, match="incompatible"):
            LMErrorTest(result, W)

        # Without model
        result2 = SimpleNamespace(resid=resid, nobs=n)
        W = _rook_weights(5)
        with pytest.raises(ValueError, match="!="):
            LMErrorTest(result2, W)

    def test_robust_lm_lag_panel(self):
        """Cover lines 289-302: RobustLMLagTest panel setup."""
        from panelbox.validation.spatial.lm_tests import RobustLMLagTest

        N = 5
        T = 3
        n = N * T
        np.random.seed(42)
        y = np.random.randn(n)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=y_hat,
            nobs=n,
        )
        result.model = SimpleNamespace(N=N, T=T, data=SimpleNamespace(y=y))

        W = _rook_weights(N)
        test = RobustLMLagTest(result, W)
        run_result = test.run()
        assert run_result.pvalue >= 0

    def test_robust_lm_error_panel(self):
        """Cover lines 397-410: RobustLMErrorTest panel setup."""
        from panelbox.validation.spatial.lm_tests import RobustLMErrorTest

        N = 5
        T = 3
        n = N * T
        np.random.seed(42)
        y = np.random.randn(n)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=y_hat,
            nobs=n,
        )
        result.model = SimpleNamespace(N=N, T=T, data=SimpleNamespace(y=y))

        W = _rook_weights(N)
        test = RobustLMErrorTest(result, W)
        run_result = test.run()
        assert run_result.pvalue >= 0

    def test_run_lm_tests_sar_recommendation(self):
        """Cover lines 533-536: SAR recommendation."""
        from panelbox.validation.spatial.lm_tests import run_lm_tests

        np.random.seed(42)
        n = 20
        W = _rook_weights(n)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        e = np.random.randn(n) * 0.1
        y = 0.7 * W @ np.random.randn(n) + X @ np.array([1.0, 2.0]) + e

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=y_hat,
            nobs=n,
        )
        output = run_lm_tests(result, W, verbose=False)
        assert output["recommendation"] in ["SAR", "SEM", "SDM", "OLS"]

    def test_run_lm_tests_ols_recommendation(self):
        """Cover lines 529-532: OLS recommendation (no spatial dep)."""
        from panelbox.validation.spatial.lm_tests import run_lm_tests

        np.random.seed(123)
        n = 20
        W = _rook_weights(n)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 5.0  # Large noise, no spatial

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=y_hat,
            nobs=n,
        )
        output = run_lm_tests(result, W, verbose=False)
        assert "recommendation" in output

    def test_run_lm_tests_verbose(self):
        """Cover lines 574-582: verbose=True logging path."""
        from panelbox.validation.spatial.lm_tests import run_lm_tests

        np.random.seed(42)
        n = 10
        W = _rook_weights(n)
        result = _make_ols_result(n)

        with patch("panelbox.validation.spatial.lm_tests.logger") as mock_logger:
            output = run_lm_tests(result, W, verbose=True)
            assert mock_logger.info.called
        assert "summary" in output

    def test_run_lm_tests_both_robust_sig(self):
        """Cover lines 549-552: both robust tests significant -> SDM."""
        from panelbox.validation.spatial.lm_tests import run_lm_tests

        np.random.seed(42)
        n = 30
        W = _rook_weights(n)
        # Very strong spatial dependence in both lag and error
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        e_raw = np.random.randn(n) * 0.01
        u = np.linalg.solve(np.eye(n) - 0.8 * W, e_raw)
        y = 0.8 * W @ (X @ np.array([1.0, 2.0]) + u) + X @ np.array([1.0, 2.0]) + u

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_hat = X @ beta
        resid = y - y_hat

        result = SimpleNamespace(resid=resid, fittedvalues=y_hat, nobs=n)
        output = run_lm_tests(result, W, verbose=False)
        assert output["recommendation"] in ["SAR", "SEM", "SDM", "OLS"]


# ===========================================================================
# Tests for FreesTest
# ===========================================================================
class TestFreesTestUncoveredBranches:
    """Cover uncovered lines in frees.py."""

    def _make_panel_result(self, N=10, T=20, add_dependence=False):
        """Create mock PanelResults for FreesTest."""
        np.random.seed(42)
        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)

        if add_dependence:
            common_shock = np.random.randn(T)
            resid = np.array([common_shock + np.random.randn(T) * 0.1 for _ in range(N)]).flatten()
        else:
            resid = np.random.randn(N * T)

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=np.zeros(N * T),
            params=np.array([1.0]),
            nobs=N * T,
            n_entities=N,
            n_periods=T,
            model_type="pooled",
            entity_index=entities,
            time_index=times,
            _model=None,
        )
        return result

    def test_frees_basic(self):
        """Cover main path of FreesTest.run()."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=5, T=10)
        test = FreesTest(result)
        vr = test.run()
        assert vr.pvalue >= 0
        assert vr.test_name == "Frees Test for Cross-Sectional Dependence"

    def test_frees_with_dependence(self):
        """Cover interpretation branch: q_frees > critical value."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=5, T=10, add_dependence=True)
        test = FreesTest(result)
        vr = test.run()
        assert vr.metadata["interpretation"] in [
            "Reject H0 (cross-sectional dependence detected)",
            "Do not reject H0 (no evidence of cross-sectional dependence)",
        ]

    def test_frees_too_few_entities(self):
        """Cover line 116: N < 2 raises ValueError."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = SimpleNamespace(
            resid=np.random.randn(10),
            fittedvalues=np.zeros(10),
            params=np.array([1.0]),
            nobs=10,
            n_entities=1,
            n_periods=10,
            model_type="pooled",
            entity_index=np.zeros(10, dtype=int),
            time_index=np.arange(10),
        )
        test = FreesTest(result)
        with pytest.raises(ValueError, match="at least 2 entities"):
            test.run()

    def test_frees_too_few_periods(self):
        """Cover line 119: T < 3 raises ValueError."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        entities = np.array([0, 0, 1, 1])
        times = np.array([0, 1, 0, 1])
        result = SimpleNamespace(
            resid=np.random.randn(4),
            fittedvalues=np.zeros(4),
            params=np.array([1.0]),
            nobs=4,
            n_entities=2,
            n_periods=2,
            model_type="pooled",
            entity_index=entities,
            time_index=times,
        )
        test = FreesTest(result)
        with pytest.raises(ValueError, match="at least 3 time"):
            test.run()

    def test_frees_no_valid_correlations(self):
        """Cover lines 152-156: no valid pairwise correlations."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        # Constant residuals for each entity -> NaN rank correlation
        N, T = 3, 5
        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        # All residuals constant within entity
        resid = np.array([i * 1.0 for i in entities])

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=np.zeros(N * T),
            params=np.array([1.0]),
            nobs=N * T,
            n_entities=N,
            n_periods=T,
            model_type="pooled",
            entity_index=entities,
            time_index=times,
        )
        test = FreesTest(result)
        with pytest.raises(ValueError, match="No valid pairwise"):
            test.run()

    def test_frees_critical_values_small_T(self):
        """Cover lines 250-253: T <= 5 critical values."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=3, T=5)
        test = FreesTest(result)
        cv = test._get_critical_values(T=5, N=3)
        assert cv["alpha_0.05"] == 0.4

    def test_frees_critical_values_medium_T(self):
        """Cover lines 254-257: T <= 10 critical values."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=3, T=8)
        test = FreesTest(result)
        cv = test._get_critical_values(T=8, N=3)
        assert cv["alpha_0.05"] == 0.2

    def test_frees_critical_values_T_20(self):
        """Cover lines 258-261: T <= 20 critical values."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=3, T=15)
        test = FreesTest(result)
        cv = test._get_critical_values(T=15, N=3)
        assert cv["alpha_0.05"] == 0.11

    def test_frees_critical_values_T_30(self):
        """Cover lines 262-265: T <= 30 critical values."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=3, T=25)
        test = FreesTest(result)
        cv = test._get_critical_values(T=25, N=3)
        assert cv["alpha_0.05"] == 0.0754

    def test_frees_critical_values_large_T(self):
        """Cover lines 266-270: T > 30 asymptotic critical values."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = self._make_panel_result(N=3, T=40)
        test = FreesTest(result)
        cv = test._get_critical_values(T=40, N=3)
        assert cv["alpha_0.05"] > 0

    def test_prepare_residual_data_no_index(self):
        """Cover lines 302-305: missing entity_index/time_index."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        result = SimpleNamespace(
            resid=np.random.randn(10),
            fittedvalues=np.zeros(10),
            params=np.array([1.0]),
            nobs=10,
            n_entities=1,
            n_periods=10,
            model_type="pooled",
        )
        test = FreesTest(result)
        with pytest.raises(AttributeError, match="entity_index"):
            test._prepare_residual_data()

    def test_frees_se_zero_branch(self):
        """Cover lines 173-174: se_qf == 0 gives z_stat = 0."""
        from panelbox.validation.cross_sectional_dependence.frees import FreesTest

        # With T=3, var_qf = 2*(3-3)/((3+1)*(3-1)^2) = 0
        N, T = 3, 3
        np.random.seed(42)
        entities = np.repeat(np.arange(N), T)
        times = np.tile(np.arange(T), N)
        resid = np.random.randn(N * T)

        result = SimpleNamespace(
            resid=resid,
            fittedvalues=np.zeros(N * T),
            params=np.array([1.0]),
            nobs=N * T,
            n_entities=N,
            n_periods=T,
            model_type="pooled",
            entity_index=entities,
            time_index=times,
        )
        test = FreesTest(result)
        vr = test.run()
        # z_stat should be 0 when var_qf=0
        assert vr.metadata["z_statistic"] == 0.0


# ===========================================================================
# Tests for InfluenceDiagnostics
# ===========================================================================
class TestInfluenceUncoveredBranches:
    """Cover uncovered lines in influence.py."""

    @pytest.fixture
    def fe_result(self):
        """Create a FixedEffects result for influence tests."""
        from panelbox import FixedEffects

        np.random.seed(42)
        n_entities = 10
        n_periods = 5
        n_obs = n_entities * n_periods
        data = pd.DataFrame(
            {
                "entity": np.repeat(range(n_entities), n_periods),
                "time": np.tile(range(n_periods), n_entities),
                "y": np.random.randn(n_obs) * 10 + 100,
                "x1": np.random.randn(n_obs) * 5 + 50,
                "x2": np.random.randn(n_obs) * 3 + 30,
            }
        )
        model = FixedEffects("y ~ x1 + x2", data, "entity", "time")
        return model.fit()

    def test_influential_observations_dffits(self, fe_result):
        """Cover lines 302-316: dffits method for influential obs."""
        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        diag = InfluenceDiagnostics(fe_result, verbose=False)
        diag.compute()
        influential = diag.influential_observations(method="dffits")
        assert isinstance(influential, pd.DataFrame)
        assert "dffits" in influential.columns

    def test_influential_observations_dfbetas(self, fe_result):
        """Cover lines 318-330: dfbetas method for influential obs."""
        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        diag = InfluenceDiagnostics(fe_result, verbose=False)
        diag.compute()
        influential = diag.influential_observations(method="dfbetas")
        assert isinstance(influential, pd.DataFrame)
        assert "max_dfbetas" in influential.columns

    def test_influential_observations_unknown_method(self, fe_result):
        """Cover line 332: unknown method raises ValueError."""
        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        diag = InfluenceDiagnostics(fe_result, verbose=False)
        diag.compute()
        with pytest.raises(ValueError, match="Unknown method"):
            diag.influential_observations(method="invalid")

    def test_plot_influence_save(self, fe_result, tmp_path):
        """Cover lines 423-426: save plot to file."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        diag = InfluenceDiagnostics(fe_result, verbose=True)
        diag.compute()
        save_path = str(tmp_path / "influence.png")
        diag.plot_influence(save_path=save_path)
        import os

        assert os.path.exists(save_path)
        plt.close("all")

    def test_plot_influence_show(self, fe_result):
        """Cover lines 427-428: show plot (no save)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        diag = InfluenceDiagnostics(fe_result, verbose=False)
        diag.compute()
        with patch("matplotlib.pyplot.show"):
            diag.plot_influence()
        plt.close("all")

    def test_leverage_pinv_fallback(self, fe_result):
        """Cover lines 188-190: pseudo-inverse fallback for leverage."""
        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        diag = InfluenceDiagnostics(fe_result, verbose=False)
        with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError("singular")):
            leverage = diag._approximate_leverage()
        assert len(leverage) > 0

    def test_no_model_raises(self):
        """Cover lines 116-117: no model reference."""
        from panelbox.validation.robustness.influence import InfluenceDiagnostics

        result = SimpleNamespace(_model=None)
        with pytest.raises(RuntimeError, match="model reference"):
            InfluenceDiagnostics(result)


# ===========================================================================
# Tests for RobustnessChecker
# ===========================================================================
class TestRobustnessCheckerUncoveredBranches:
    """Cover uncovered lines in checks.py."""

    @pytest.fixture
    def fe_result(self):
        """Create a FixedEffects result for robustness tests."""
        from panelbox import FixedEffects

        np.random.seed(42)
        n_entities = 10
        n_periods = 5
        n_obs = n_entities * n_periods
        data = pd.DataFrame(
            {
                "entity": np.repeat(range(n_entities), n_periods),
                "time": np.tile(range(n_periods), n_entities),
                "y": np.random.randn(n_obs) * 10 + 100,
                "x1": np.random.randn(n_obs) * 5 + 50,
                "x2": np.random.randn(n_obs) * 3 + 30,
                "x3": np.random.randn(n_obs) * 2 + 10,
            }
        )
        model = FixedEffects("y ~ x1 + x2 + x3", data, "entity", "time")
        return model.fit()

    def test_no_model_raises(self):
        """Cover line 42: no model reference."""
        from panelbox.validation.robustness.checks import RobustnessChecker

        result = SimpleNamespace(_model=None)
        with pytest.raises(RuntimeError, match="model reference"):
            RobustnessChecker(result)

    def test_check_alternative_specs_with_model_type(self, fe_result):
        """Cover lines 70-74: model_type parameter."""
        from panelbox.validation.robustness.checks import RobustnessChecker

        checker = RobustnessChecker(fe_result, verbose=False)
        results = checker.check_alternative_specs(
            ["y ~ x1", "y ~ x1 + x2"],
            model_type="pooled",
        )
        assert len(results) == 2

    def test_check_alternative_specs_failure(self, fe_result):
        """Cover lines 84-87: failed formula estimation."""
        from panelbox.validation.robustness.checks import RobustnessChecker

        checker = RobustnessChecker(fe_result, verbose=True)
        # Use an invalid formula to trigger exception
        results = checker.check_alternative_specs(
            ["y ~ nonexistent_variable"],
        )
        assert results[0] is None

    def test_generate_robustness_table_with_none(self, fe_result):
        """Cover line 116-117: None in results_list."""
        from panelbox.validation.robustness.checks import RobustnessChecker

        checker = RobustnessChecker(fe_result, verbose=False)
        results = checker.check_alternative_specs(["y ~ x1", "y ~ nonexistent"])
        table = checker.generate_robustness_table(results)
        assert isinstance(table, pd.DataFrame)

    def test_generate_robustness_table_empty(self, fe_result):
        """Cover lines 138-139: empty DataFrame when no valid results."""
        from panelbox.validation.robustness.checks import RobustnessChecker

        checker = RobustnessChecker(fe_result, verbose=False)
        table = checker.generate_robustness_table([None, None])
        assert isinstance(table, pd.DataFrame)
        assert len(table) == 0

    def test_generate_robustness_table_with_parameters(self, fe_result):
        """Cover lines 109-112: explicit parameters list."""
        from panelbox.validation.robustness.checks import RobustnessChecker

        checker = RobustnessChecker(fe_result, verbose=False)
        results = checker.check_alternative_specs(["y ~ x1 + x2", "y ~ x1 + x2 + x3"])
        table = checker.generate_robustness_table(results, parameters=["x1"])
        assert isinstance(table, pd.DataFrame)
