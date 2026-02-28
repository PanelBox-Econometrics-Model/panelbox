"""
Deep coverage tests for panelbox.optimization.parallel_inference.

Targets uncovered lines from the coverage report:
- Lines 93->91: Branch in ParallelPermutationTest.run() when seed is None
- Lines 245->243: Branch in ParallelBootstrap.run() when seed is None
- Lines 362-365: ParallelBootstrap._worker_bootstrap estimation failure catch block
  (param_results is empty when the very first bootstrap iteration fails)
- Lines 461->456, 464, 466: ParallelSpatialHAC._compute_chunk kernel branches
  for "epanechnikov" and unknown/else kernel
- Lines 512-541: parallel_grid_search function and _evaluate_params helper
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.optimization.parallel_inference import (
    ParallelBootstrap,
    ParallelPermutationTest,
    ParallelSpatialHAC,
    parallel_grid_search,
)

# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for multiprocessing workers)
# ---------------------------------------------------------------------------


def _test_stat_mean_lag(data, W, **kwargs):
    """Compute mean of spatial lag as a test statistic."""
    return float(np.mean(W @ data))


def _simple_estimator(data, W, formula, entity_col, time_col, **kwargs):
    """Simple OLS estimator returning a dict of parameter values."""
    x = data["x"].values
    y = data["y"].values
    denom = np.sum(x**2)
    beta = np.sum(x * y) / denom if denom > 0 else 0.0
    intercept = np.mean(y) - beta * np.mean(x)
    return {"beta": beta, "intercept": intercept}


def _always_failing_estimator(data, W, formula, entity_col, time_col, **kwargs):
    """Estimator that always raises an exception."""
    raise RuntimeError("Intentional estimation failure")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_data():
    """Small cross-sectional data array (length 10)."""
    np.random.seed(42)
    return np.random.randn(10)


@pytest.fixture
def small_W():
    """Row-standardized 10x10 spatial weight matrix (contiguity)."""
    n = 10
    W = np.zeros((n, n))
    for i in range(n - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return W / row_sums


@pytest.fixture
def panel_df():
    """Panel DataFrame with entity, time, x, y columns."""
    np.random.seed(42)
    n_entities = 6
    n_time = 4
    n = n_entities * n_time
    entities = np.repeat(np.arange(n_entities), n_time)
    times = np.tile(np.arange(n_time), n_entities)
    x = np.random.randn(n)
    y = 0.5 * x + np.random.randn(n) * 0.3
    return pd.DataFrame({"entity": entities, "time": times, "x": x, "y": y})


@pytest.fixture
def panel_W():
    """6x6 identity weight matrix matching the panel entity count."""
    return np.eye(6)


# ===========================================================================
# Tests for lines 93->91: ParallelPermutationTest.run() seed-is-None branch
# ===========================================================================


class TestPermutationSeedNoneBranch:
    """Cover the branch in run() where self.seed is None (line 94),
    and the n_perms == 0 skip branch (line 93->91)."""

    def test_run_without_seed(self, small_data, small_W):
        """ParallelPermutationTest.run() with seed=None should produce valid
        results and exercise the seed_i = None branch."""
        ppt = ParallelPermutationTest(
            test_statistic=_test_stat_mean_lag,
            n_permutations=10,
            n_jobs=1,
            seed=None,
        )
        result = ppt.run(small_data, small_W)

        assert "observed" in result
        assert "permuted_stats" in result
        assert "p_value" in result
        assert len(result["permuted_stats"]) == 10
        assert 0 <= result["p_value"] <= 1

    def test_run_without_seed_computes_observed(self, small_data, small_W):
        """When observed_stat is None, run() should compute it internally."""
        ppt = ParallelPermutationTest(
            test_statistic=_test_stat_mean_lag,
            n_permutations=5,
            n_jobs=1,
            seed=None,
        )
        result = ppt.run(small_data, small_W)
        expected_observed = _test_stat_mean_lag(small_data, small_W)
        assert_allclose(result["observed"], expected_observed)

    def test_run_more_jobs_than_permutations(self, small_data, small_W):
        """When n_jobs > n_permutations, some workers get n_perms=0.
        This exercises the 93->91 branch (skip when n_perms == 0)."""
        ppt = ParallelPermutationTest(
            test_statistic=_test_stat_mean_lag,
            n_permutations=1,
            n_jobs=3,
            seed=42,
        )
        result = ppt.run(small_data, small_W)

        assert len(result["permuted_stats"]) == 1
        assert 0 <= result["p_value"] <= 1


# ===========================================================================
# Tests for lines 245->243: ParallelBootstrap.run() seed-is-None branch
# ===========================================================================


class TestBootstrapSeedNoneBranch:
    """Cover the branch in run() where self.seed is None (line 246),
    and the n_samples == 0 skip branch (line 245->243)."""

    def test_run_without_seed(self, panel_df, panel_W):
        """ParallelBootstrap.run() with seed=None should produce valid
        bootstrap results and exercise the seed_i = None branch."""
        boot = ParallelBootstrap(
            estimator=_simple_estimator,
            n_bootstrap=6,
            n_jobs=1,
            bootstrap_type="pairs",
            seed=None,
        )
        result = boot.run(
            panel_df,
            panel_W,
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert "original" in result
        assert "bootstrap" in result
        assert "beta" in result["bootstrap"]
        assert "intercept" in result["bootstrap"]
        assert result["n_bootstrap"] == 6

    def test_run_more_jobs_than_bootstrap(self, panel_df, panel_W):
        """When n_jobs > n_bootstrap, some workers get n_samples=0.
        This exercises the 245->243 branch (skip when n_samples == 0)."""
        boot = ParallelBootstrap(
            estimator=_simple_estimator,
            n_bootstrap=1,
            n_jobs=3,
            bootstrap_type="pairs",
            seed=42,
        )
        result = boot.run(
            panel_df,
            panel_W,
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert "original" in result
        assert "bootstrap" in result
        assert result["n_bootstrap"] == 1

    def test_run_without_seed_wild(self, panel_df, panel_W):
        """Wild bootstrap with seed=None should also work."""
        boot = ParallelBootstrap(
            estimator=_simple_estimator,
            n_bootstrap=4,
            n_jobs=1,
            bootstrap_type="wild",
            seed=None,
        )
        result = boot.run(
            panel_df,
            panel_W,
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert result["type"] == "wild"
        assert "beta" in result["bootstrap"]


# ===========================================================================
# Tests for lines 362-365: _worker_bootstrap estimation failure catch block
# ===========================================================================


class TestBootstrapEstimationFailure:
    """Cover the except block in _worker_bootstrap where the estimator raises
    and param_results is empty on the very first iteration (line 362-365)."""

    def test_all_iterations_fail(self, panel_df):
        """When ALL bootstrap iterations fail, param_results is always empty
        at the first failure. This exercises the `if param_results else [np.nan]`
        branch taking the else path."""
        work_package = (
            3,
            panel_df,
            np.eye(6),
            "y ~ x",
            "entity",
            "time",
            {},
            "pairs",
            42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ParallelBootstrap._worker_bootstrap(_always_failing_estimator, work_package)

        assert result.shape[0] == 3
        # All rows should be [nan] since every iteration fails
        assert np.all(np.isnan(result))

    def test_first_iteration_fails_empty_param_results(self, panel_df):
        """Directly verify that the first failure with empty param_results
        produces [np.nan] (the else branch on line 366).

        When the first iteration fails, param_results is empty so the code
        takes the `else` path and appends [np.nan].  To avoid shape
        mismatches with subsequent successful iterations, we use a
        single-parameter estimator so all rows have length 1.
        """
        call_count = {"n": 0}

        def _fail_then_succeed(data, W, formula, entity_col, time_col, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 1:
                raise RuntimeError("First call fails")
            return {"beta": 1.0}

        # Calling _worker_bootstrap directly (not via Pool) so closures work.
        work_package = (
            3,
            panel_df,
            np.eye(6),
            "y ~ x",
            "entity",
            "time",
            {},
            "pairs",
            42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = ParallelBootstrap._worker_bootstrap(_fail_then_succeed, work_package)

        assert result.shape == (3, 1)
        # First row should be nan (param_results was empty at first failure)
        assert np.isnan(result[0, 0])
        # Subsequent rows should have the real value
        assert_allclose(result[1, 0], 1.0)
        assert_allclose(result[2, 0], 1.0)


# ===========================================================================
# Tests for lines 461->456, 464, 466: _compute_chunk kernel branches
# ===========================================================================


class TestComputeChunkKernelBranches:
    """Cover ParallelSpatialHAC._compute_chunk for different kernels."""

    def test_epanechnikov_kernel(self):
        """Epanechnikov kernel: weight = 0.75 * (1 - u^2) where u = dist/cutoff."""
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
            dtype=float,
        )
        cutoff = 4.0
        chunk_info = (0, 5, coords, cutoff, "epanechnikov")
        result = ParallelSpatialHAC._compute_chunk(chunk_info)

        assert result.shape == (5, 5)

        # Verify specific weights for known distances
        # dist(0,1) = 1.0, u = 0.25, weight = 0.75 * (1 - 0.0625) = 0.703125
        expected_01 = 0.75 * (1 - (1.0 / 4.0) ** 2)
        assert_allclose(result[0, 1], expected_01, atol=1e-12)

        # dist(0,2) = 2.0, u = 0.5, weight = 0.75 * (1 - 0.25) = 0.5625
        expected_02 = 0.75 * (1 - (2.0 / 4.0) ** 2)
        assert_allclose(result[0, 2], expected_02, atol=1e-12)

        # dist(0,3) = 3.0, u = 0.75, weight = 0.75 * (1 - 0.5625) = 0.328125
        expected_03 = 0.75 * (1 - (3.0 / 4.0) ** 2)
        assert_allclose(result[0, 3], expected_03, atol=1e-12)

        # dist(0,4) = 5.0 > cutoff=4.0, weight = 0.0
        assert result[0, 4] == 0.0

    def test_unknown_kernel_defaults_to_one(self):
        """Unknown kernel should default to weight = 1.0 for all pairs within cutoff."""
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
            dtype=float,
        )
        cutoff = 3.0
        chunk_info = (0, 4, coords, cutoff, "some_nonexistent_kernel")
        result = ParallelSpatialHAC._compute_chunk(chunk_info)

        assert result.shape == (4, 4)

        # Pairs within cutoff get weight 1.0
        assert result[0, 1] == 1.0  # dist = 1
        assert result[0, 2] == 1.0  # dist = 2
        assert result[1, 2] == 1.0  # dist = 1

        # Pair beyond cutoff gets weight 0.0
        assert result[0, 3] == 0.0  # dist = 10

    def test_epanechnikov_via_compute_spatial_weights(self):
        """End-to-end: compute_spatial_weights with epanechnikov kernel."""
        coords = np.array(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [5.0, 0.0]],
            dtype=float,
        )
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=4.0, kernel="epanechnikov")

        assert weights.shape == (5, 5)
        # Should be symmetric
        assert_allclose(weights, weights.T, atol=1e-12)
        # Diagonal should be 1.0
        assert_allclose(np.diag(weights), 1.0)
        # Neighbors within cutoff should have positive weights
        assert weights[0, 1] > 0
        assert weights[0, 2] > 0

    def test_unknown_kernel_via_compute_spatial_weights(self):
        """End-to-end: compute_spatial_weights with unknown kernel."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]], dtype=float)
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=2.0, kernel="nonexistent")

        assert weights.shape == (3, 3)
        assert_allclose(np.diag(weights), 1.0)
        # dist(0,1) = 1.0 <= cutoff=2.0, default weight = 1.0
        # After symmetrization: (1.0 + 1.0) / 2 = 1.0
        assert_allclose(weights[0, 1], 1.0)
        # dist(0,2) = 5.0 > cutoff=2.0
        assert weights[0, 2] == 0.0

    def test_bartlett_kernel_chunk(self):
        """Bartlett kernel for reference: weight = 1 - dist/cutoff."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
        cutoff = 3.0
        chunk_info = (0, 3, coords, cutoff, "bartlett")
        result = ParallelSpatialHAC._compute_chunk(chunk_info)

        # dist(0,1) = 1.0, weight = 1 - 1/3 = 2/3
        assert_allclose(result[0, 1], 2.0 / 3.0, atol=1e-12)
        # dist(0,2) = 2.0, weight = 1 - 2/3 = 1/3
        assert_allclose(result[0, 2], 1.0 / 3.0, atol=1e-12)

    def test_uniform_kernel_chunk(self):
        """Uniform kernel: weight = 1.0 for all within cutoff."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=float)
        cutoff = 2.0
        chunk_info = (0, 3, coords, cutoff, "uniform")
        result = ParallelSpatialHAC._compute_chunk(chunk_info)

        assert result[0, 1] == 1.0  # dist = 1 <= cutoff
        assert result[0, 2] == 0.0  # dist = 4 > cutoff


# ===========================================================================
# Tests for lines 512-541: parallel_grid_search function
# ===========================================================================


class _MockFitResult:
    """Picklable fit result for grid search tests."""

    def __init__(self, aic, bic, log_likelihood):
        self.aic = aic
        self.bic = bic
        self.log_likelihood = log_likelihood
        self.converged = True


class _MockModel:
    """Picklable model class for grid search tests."""

    def __init__(self, data, W, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

    def fit(self):
        # AIC and BIC increase with alpha + beta, log-likelihood decreases
        score = self.alpha + self.beta
        return _MockFitResult(
            aic=score * 10.0,
            bic=score * 12.0,
            log_likelihood=-score * 5.0,
        )


class _MockModelFailing:
    """Picklable model class that always fails during fit."""

    def __init__(self, data, W, alpha=1.0):
        self.alpha = alpha

    def fit(self):
        raise ValueError("Model fit failed intentionally")


class TestParallelGridSearch:
    """Cover lines 512-541: the parallel_grid_search function."""

    def test_grid_search_aic(self):
        """parallel_grid_search with AIC scoring should find best params."""
        data = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        W = np.eye(3)
        param_grid = {"alpha": [1.0, 2.0, 3.0], "beta": [0.5, 1.0]}

        result = parallel_grid_search(
            _MockModel,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="aic",
        )

        assert "best_params" in result
        assert "best_score" in result
        assert "all_results" in result
        # Lowest AIC corresponds to alpha=1.0, beta=0.5 -> score = 15.0
        assert result["best_params"]["alpha"] == 1.0
        assert result["best_params"]["beta"] == 0.5
        assert_allclose(result["best_score"], 15.0)
        # Total combinations: 3 * 2 = 6
        assert len(result["all_results"]) == 6

    def test_grid_search_bic(self):
        """parallel_grid_search with BIC scoring should find best params."""
        data = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        W = np.eye(2)
        param_grid = {"alpha": [1.0, 2.0], "beta": [1.0]}

        result = parallel_grid_search(
            _MockModel,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="bic",
        )

        # BIC for alpha=1.0, beta=1.0: (1+1)*12 = 24.0
        # BIC for alpha=2.0, beta=1.0: (2+1)*12 = 36.0
        assert result["best_params"]["alpha"] == 1.0
        assert_allclose(result["best_score"], 24.0)

    def test_grid_search_log_likelihood(self):
        """parallel_grid_search with log_likelihood scoring should maximize."""
        data = pd.DataFrame({"x": [1], "y": [2]})
        W = np.eye(1)
        param_grid = {"alpha": [1.0, 2.0, 3.0], "beta": [1.0]}

        result = parallel_grid_search(
            _MockModel,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="log_likelihood",
        )

        # log_likelihood = -(alpha+beta)*5
        # Highest (least negative) at alpha=1.0: -10.0
        assert result["best_params"]["alpha"] == 1.0
        assert_allclose(result["best_score"], -10.0)

    def test_grid_search_single_combination(self):
        """Grid search with a single parameter combination."""
        data = pd.DataFrame({"x": [1]})
        W = np.eye(1)
        param_grid = {"alpha": [2.0], "beta": [3.0]}

        result = parallel_grid_search(
            _MockModel,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="aic",
        )

        assert result["best_params"] == {"alpha": 2.0, "beta": 3.0}
        assert len(result["all_results"]) == 1

    def test_grid_search_with_model_failure(self):
        """Grid search handles model failures gracefully."""
        data = pd.DataFrame({"x": [1, 2]})
        W = np.eye(2)
        param_grid = {"alpha": [1.0, 2.0]}

        result = parallel_grid_search(
            _MockModelFailing,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="aic",
        )

        # All fail, all scores should be inf
        assert "best_params" in result
        for r in result["all_results"]:
            assert r["score"] == np.inf
            assert r["converged"] is False
            assert "error" in r

    def test_grid_search_with_model_kwargs(self):
        """Grid search should forward model_kwargs to the model constructor."""
        data = pd.DataFrame({"x": [1, 2]})
        W = np.eye(2)
        param_grid = {"alpha": [1.0]}

        result = parallel_grid_search(
            _MockModel,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="aic",
            beta=0.5,  # passed as model_kwargs
        )

        # alpha=1.0, beta=0.5 -> AIC = (1+0.5)*10 = 15.0
        assert_allclose(result["best_score"], 15.0)

    def test_grid_search_log_likelihood_failure_returns_neg_inf(self):
        """Failed models under log_likelihood scoring should get -inf."""
        data = pd.DataFrame({"x": [1]})
        W = np.eye(1)
        param_grid = {"alpha": [1.0]}

        result = parallel_grid_search(
            _MockModelFailing,
            param_grid,
            data,
            W,
            n_jobs=1,
            scoring="log_likelihood",
        )

        assert result["best_score"] == -np.inf


# ===========================================================================
# Additional integration-style tests
# ===========================================================================


class TestPermutationWithProvidedObserved:
    """Integration test: permutation with provided observed stat and no seed."""

    def test_provided_observed_no_seed(self, small_data, small_W):
        """Providing observed_stat should skip internal computation."""
        ppt = ParallelPermutationTest(
            test_statistic=_test_stat_mean_lag,
            n_permutations=8,
            n_jobs=1,
            seed=None,
        )
        result = ppt.run(small_data, small_W, observed_stat=99.0)

        assert result["observed"] == 99.0
        assert len(result["permuted_stats"]) == 8


class TestBootstrapWorkerSeedBranch:
    """Directly test _worker_bootstrap with seed=None to confirm no crash."""

    def test_worker_no_seed(self, panel_df):
        """_worker_bootstrap with seed=None should not set np.random.seed."""
        work_package = (
            2,
            panel_df,
            np.eye(6),
            "y ~ x",
            "entity",
            "time",
            {},
            "pairs",
            None,
        )
        result = ParallelBootstrap._worker_bootstrap(_simple_estimator, work_package)
        assert result.shape[0] == 2
        assert result.shape[1] == 2  # beta, intercept
        assert np.all(np.isfinite(result))
