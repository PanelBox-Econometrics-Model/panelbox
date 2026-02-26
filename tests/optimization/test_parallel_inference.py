"""Tests for parallel inference methods for spatial panel models.

Tests panelbox/optimization/parallel_inference.py covering:
- ParallelPermutationTest: permutation test with n_jobs=1
- ParallelBootstrap: bootstrap with different types
- ParallelSpatialHAC: spatial weights computation with kernels
- parallel_grid_search and _evaluate_params
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.optimization.parallel_inference import (
    ParallelBootstrap,
    ParallelPermutationTest,
    ParallelSpatialHAC,
    _evaluate_params,
)


def _simple_ols_estimator(data, W, formula, entity_col, time_col, **kwargs):
    """Simple OLS estimator for testing (module-level for pickling)."""
    x = data["x"].values
    y = data["y"].values
    beta = np.sum(x * y) / np.sum(x**2) if np.sum(x**2) > 0 else 0.0
    return {"beta": beta}


@pytest.fixture
def simple_data():
    """Simple cross-sectional data for testing."""
    np.random.seed(42)
    return np.random.randn(20)


@pytest.fixture
def small_W():
    """Small 20x20 spatial weight matrix."""
    np.random.seed(42)
    n = 20
    W = np.zeros((n, n))
    # Simple contiguity: each observation connected to its neighbors
    for i in range(n - 1):
        W[i, i + 1] = 1.0
        W[i + 1, i] = 1.0
    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W = W / row_sums
    return W


@pytest.fixture
def panel_data():
    """Simple panel data for bootstrap testing."""
    np.random.seed(42)
    n_entities = 10
    n_time = 5
    n = n_entities * n_time

    entities = np.repeat(np.arange(n_entities), n_time)
    times = np.tile(np.arange(n_time), n_entities)
    x = np.random.randn(n)
    y = 0.5 * x + np.random.randn(n) * 0.5

    return pd.DataFrame({"entity": entities, "time": times, "x": x, "y": y})


def moran_i_stat(data, W, **kwargs):
    """Simple Moran's I test statistic for testing."""
    n = len(data)
    z = data - data.mean()
    numerator = n * z @ W @ z
    denominator = W.sum() * (z @ z)
    if denominator == 0:
        return 0.0
    return numerator / denominator


class TestParallelPermutationTest:
    """Tests for ParallelPermutationTest."""

    def test_basic_permutation(self, simple_data, small_W):
        """Basic permutation test should produce valid p-value."""
        ppt = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=50,
            n_jobs=1,
            seed=42,
        )

        result = ppt.run(simple_data, small_W)

        assert "observed" in result
        assert "permuted_stats" in result
        assert "p_value" in result
        assert "n_permutations" in result
        assert 0 <= result["p_value"] <= 1
        assert len(result["permuted_stats"]) == 50

    def test_observed_stat_provided(self, simple_data, small_W):
        """Should use provided observed statistic."""
        ppt = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=30,
            n_jobs=1,
            seed=42,
        )

        observed = 5.0  # A large value
        result = ppt.run(simple_data, small_W, observed_stat=observed)

        assert result["observed"] == 5.0

    def test_reproducibility_with_seed(self, simple_data, small_W):
        """Same seed should produce same results."""
        ppt1 = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=30,
            n_jobs=1,
            seed=42,
        )
        ppt2 = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=30,
            n_jobs=1,
            seed=42,
        )

        result1 = ppt1.run(simple_data, small_W)
        result2 = ppt2.run(simple_data, small_W)

        assert_allclose(result1["permuted_stats"], result2["permuted_stats"])

    def test_panel_permutation(self, small_W):
        """Permutation with panel data (entity_ids and time_ids)."""
        np.random.seed(42)
        n = 20
        entity_ids = np.repeat(np.arange(4), 5)
        time_ids = np.tile(np.arange(5), 4)
        data = np.random.randn(n)

        ppt = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=30,
            n_jobs=1,
            seed=42,
        )

        result = ppt.run(data, small_W, entity_ids=entity_ids, time_ids=time_ids)

        assert 0 <= result["p_value"] <= 1
        assert len(result["permuted_stats"]) == 30

    def test_no_seed(self, simple_data, small_W):
        """Should work without seed."""
        ppt = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=20,
            n_jobs=1,
        )

        result = ppt.run(simple_data, small_W)
        assert "p_value" in result

    def test_njobs_auto(self, simple_data, small_W):
        """n_jobs=-1 should use all CPUs."""
        ppt = ParallelPermutationTest(
            test_statistic=moran_i_stat,
            n_permutations=20,
            n_jobs=-1,
            seed=42,
        )

        assert ppt.n_jobs > 0


class TestParallelBootstrap:
    """Tests for ParallelBootstrap."""

    def test_basic_bootstrap(self, panel_data, small_W):
        """Basic bootstrap should produce parameter estimates."""
        bootstrap = ParallelBootstrap(
            estimator=_simple_ols_estimator,
            n_bootstrap=20,
            n_jobs=1,
            bootstrap_type="pairs",
            seed=42,
        )

        result = bootstrap.run(
            panel_data,
            small_W[:10, :10],
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert "original" in result
        assert "bootstrap" in result
        assert "n_bootstrap" in result
        assert result["type"] == "pairs"
        assert "beta" in result["bootstrap"]
        assert "mean" in result["bootstrap"]["beta"]
        assert "std" in result["bootstrap"]["beta"]
        assert "ci_lower" in result["bootstrap"]["beta"]
        assert "ci_upper" in result["bootstrap"]["beta"]

    def test_wild_bootstrap(self, panel_data, small_W):
        """Wild bootstrap should apply Rademacher weights."""
        bootstrap = ParallelBootstrap(
            estimator=_simple_ols_estimator,
            n_bootstrap=20,
            n_jobs=1,
            bootstrap_type="wild",
            seed=42,
        )

        result = bootstrap.run(
            panel_data,
            small_W[:10, :10],
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert result["type"] == "wild"
        assert "beta" in result["bootstrap"]

    def test_block_bootstrap(self, panel_data, small_W):
        """Block bootstrap should resample time periods."""
        bootstrap = ParallelBootstrap(
            estimator=_simple_ols_estimator,
            n_bootstrap=20,
            n_jobs=1,
            bootstrap_type="block",
            seed=42,
        )

        result = bootstrap.run(
            panel_data,
            small_W[:10, :10],
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert result["type"] == "block"

    def test_residual_bootstrap(self, panel_data, small_W):
        """Residual bootstrap should permute residuals."""
        bootstrap = ParallelBootstrap(
            estimator=_simple_ols_estimator,
            n_bootstrap=20,
            n_jobs=1,
            bootstrap_type="residual",
            seed=42,
        )

        result = bootstrap.run(
            panel_data,
            small_W[:10, :10],
            formula="y ~ x",
            entity_col="entity",
            time_col="time",
        )

        assert result["type"] == "residual"

    def test_worker_bootstrap_direct(self, panel_data):
        """Test _worker_bootstrap static method directly."""
        W = np.eye(10)
        work_package = (
            3,  # n_samples
            panel_data,
            W,
            "y ~ x",
            "entity",
            "time",
            {},
            "pairs",
            42,  # seed
        )
        result = ParallelBootstrap._worker_bootstrap(_simple_ols_estimator, work_package)
        assert result.shape[0] == 3
        assert result.shape[1] == 1  # one param: beta
        assert np.all(np.isfinite(result))


class TestParallelSpatialHAC:
    """Tests for ParallelSpatialHAC."""

    def test_bartlett_kernel(self):
        """Bartlett kernel weights should decrease with distance."""
        coords = np.array(
            [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]],
            dtype=float,
        )
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=3.0, kernel="bartlett")

        assert weights.shape == (5, 5)
        # Diagonal should be 1
        assert_allclose(np.diag(weights), 1.0)
        # Symmetric
        assert_allclose(weights, weights.T, atol=1e-10)
        # Neighbors within cutoff should have positive weights
        assert weights[0, 1] > 0  # distance=1, within cutoff=3
        assert weights[0, 2] > 0  # distance=2, within cutoff=3
        # Further away should have smaller weights
        assert weights[0, 1] > weights[0, 2]

    def test_uniform_kernel(self):
        """Uniform kernel should give equal weights within cutoff."""
        coords = np.array(
            [[0, 0], [1, 0], [3, 0]],
            dtype=float,
        )
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=2.0, kernel="uniform")

        # Distance 0-1: within cutoff -> weight 1
        assert weights[0, 1] > 0
        # Distance 0-2: 3 > cutoff=2 -> weight 0
        assert weights[0, 2] == 0

    def test_epanechnikov_kernel(self):
        """Epanechnikov kernel should give smooth weights."""
        coords = np.array(
            [[0, 0], [1, 0], [2, 0]],
            dtype=float,
        )
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=3.0, kernel="epanechnikov")

        assert weights.shape == (3, 3)
        assert_allclose(np.diag(weights), 1.0)
        # Neighbors should have positive weights
        assert weights[0, 1] > 0

    def test_symmetry(self):
        """Weight matrix should always be symmetric."""
        np.random.seed(42)
        coords = np.random.randn(10, 2)
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=2.0)

        assert_allclose(weights, weights.T, atol=1e-10)

    def test_beyond_cutoff(self):
        """Points beyond cutoff should have zero weight."""
        coords = np.array([[0, 0], [100, 100]], dtype=float)
        hac = ParallelSpatialHAC(n_jobs=1)
        weights = hac.compute_spatial_weights(coords, cutoff=1.0)

        assert weights[0, 1] == 0.0
        assert weights[1, 0] == 0.0


class TestEvaluateParams:
    """Tests for _evaluate_params function."""

    def test_successful_evaluation(self):
        """Should evaluate model successfully."""

        class MockResult:
            aic = 100.0
            bic = 110.0
            log_likelihood = -50.0
            converged = True

        class MockModel:
            def __init__(self, data, W, param1=1):
                self.param1 = param1

            def fit(self):
                return MockResult()

        work_package = (
            MockModel,
            {"param1": 1},
            pd.DataFrame({"x": [1, 2]}),
            np.eye(2),
            "aic",
            {},
        )

        result = _evaluate_params(work_package)

        assert result["score"] == 100.0
        assert result["converged"] is True

    def test_failed_evaluation(self):
        """Should handle failure gracefully."""

        class FailModel:
            def __init__(self, data, W):
                raise ValueError("bad params")

            def fit(self):
                pass

        work_package = (
            FailModel,
            {},
            pd.DataFrame({"x": [1]}),
            np.eye(1),
            "aic",
            {},
        )

        result = _evaluate_params(work_package)

        assert result["score"] == np.inf
        assert result["converged"] is False
        assert "error" in result

    def test_bic_scoring(self):
        """Should use BIC scoring."""

        class MockResult:
            aic = 100.0
            bic = 110.0
            log_likelihood = -50.0
            converged = True

        class MockModel:
            def __init__(self, data, W):
                pass

            def fit(self):
                return MockResult()

        work_package = (
            MockModel,
            {},
            pd.DataFrame({"x": [1]}),
            np.eye(1),
            "bic",
            {},
        )

        result = _evaluate_params(work_package)
        assert result["score"] == 110.0

    def test_loglikelihood_scoring(self):
        """Should use log-likelihood scoring."""

        class MockResult:
            aic = 100.0
            bic = 110.0
            log_likelihood = -50.0
            converged = True

        class MockModel:
            def __init__(self, data, W):
                pass

            def fit(self):
                return MockResult()

        work_package = (
            MockModel,
            {},
            pd.DataFrame({"x": [1]}),
            np.eye(1),
            "log_likelihood",
            {},
        )

        result = _evaluate_params(work_package)
        assert result["score"] == -50.0

    def test_failed_loglikelihood_scoring(self):
        """Failed evaluation with log_likelihood should return -inf."""

        class FailModel:
            def __init__(self, data, W):
                raise RuntimeError("fail")

            def fit(self):
                pass

        work_package = (
            FailModel,
            {},
            pd.DataFrame({"x": [1]}),
            np.eye(1),
            "log_likelihood",
            {},
        )

        result = _evaluate_params(work_package)
        assert result["score"] == -np.inf


class TestParallelGridSearchEvaluation:
    """Tests for parallel_grid_search evaluation logic via _evaluate_params."""

    def test_argmin_for_aic(self):
        """Grid search should pick lowest AIC."""
        # Test _evaluate_params directly for different scoring
        results = []
        for aic_val in [100, 80, 120]:

            class MockResult:
                aic = float(aic_val)
                bic = float(aic_val + 10)
                log_likelihood = -float(aic_val) / 2
                converged = True

            class MockModel:
                def __init__(self, data, W, **kwargs):
                    pass

                def fit(self):
                    return MockResult()

            work_package = (
                MockModel,
                {"param": aic_val},
                pd.DataFrame({"x": [1]}),
                np.eye(1),
                "aic",
                {},
            )
            results.append(_evaluate_params(work_package))

        scores = [r["score"] for r in results]
        best_idx = np.argmin(scores)
        assert results[best_idx]["params"]["param"] == 80
