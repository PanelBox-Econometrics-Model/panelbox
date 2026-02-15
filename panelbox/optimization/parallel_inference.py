"""
Parallel inference methods for spatial panel models.

Implements multiprocessing for permutation tests and bootstrap inference.
"""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class ParallelPermutationTest:
    """
    Parallel implementation of permutation tests for spatial models.

    Uses multiprocessing to speed up permutation inference.
    """

    def __init__(
        self,
        test_statistic: Callable,
        n_permutations: int = 999,
        n_jobs: int = -1,
        seed: Optional[int] = None,
    ):
        """
        Initialize parallel permutation test.

        Parameters
        ----------
        test_statistic : callable
            Function that computes the test statistic
        n_permutations : int, default 999
            Number of permutations
        n_jobs : int, default -1
            Number of parallel jobs (-1 uses all CPUs)
        seed : int, optional
            Random seed for reproducibility
        """
        self.test_statistic = test_statistic
        self.n_permutations = n_permutations
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.seed = seed

    def run(
        self, data: np.ndarray, W: np.ndarray, observed_stat: Optional[float] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Run parallel permutation test.

        Parameters
        ----------
        data : np.ndarray
            Data array (e.g., residuals)
        W : np.ndarray
            Spatial weight matrix
        observed_stat : float, optional
            Observed test statistic (computed if not provided)
        **kwargs
            Additional arguments for test_statistic

        Returns
        -------
        results : dict
            Dictionary with test results
        """
        # Compute observed statistic if not provided
        if observed_stat is None:
            observed_stat = self.test_statistic(data, W, **kwargs)

        # Setup for parallel computation
        n_entities = len(np.unique(kwargs.get("entity_ids", range(len(data)))))

        # Divide permutations across workers
        perms_per_worker = self.n_permutations // self.n_jobs
        extra_perms = self.n_permutations % self.n_jobs

        # Create work packages
        work_packages = []
        perm_start = 0

        for i in range(self.n_jobs):
            n_perms = perms_per_worker + (1 if i < extra_perms else 0)
            if n_perms > 0:
                seed_i = None if self.seed is None else self.seed + i
                work_packages.append((perm_start, n_perms, data, W, kwargs, seed_i))
                perm_start += n_perms

        # Run parallel permutations
        with Pool(processes=self.n_jobs) as pool:
            worker_func = partial(self._worker_permutations, self.test_statistic)
            results = pool.map(worker_func, work_packages)

        # Combine results
        all_permuted_stats = np.concatenate(results)

        # Calculate p-value
        p_value = np.mean(np.abs(all_permuted_stats) >= np.abs(observed_stat))

        return {
            "observed": observed_stat,
            "permuted_stats": all_permuted_stats,
            "p_value": p_value,
            "n_permutations": self.n_permutations,
        }

    @staticmethod
    def _worker_permutations(test_statistic: Callable, work_package: Tuple) -> np.ndarray:
        """
        Worker function for parallel permutation computation.

        Parameters
        ----------
        test_statistic : callable
            Test statistic function
        work_package : tuple
            (start_idx, n_perms, data, W, kwargs, seed)

        Returns
        -------
        permuted_stats : np.ndarray
            Array of permuted test statistics
        """
        start_idx, n_perms, data, W, kwargs, seed = work_package

        # Set random seed for this worker
        if seed is not None:
            np.random.seed(seed)

        # Get entity structure if panel data
        entity_ids = kwargs.get("entity_ids")
        time_ids = kwargs.get("time_ids")

        permuted_stats = np.zeros(n_perms)

        for i in range(n_perms):
            # Permute data
            if entity_ids is not None:
                # Panel data: permute within time periods
                permuted_data = data.copy()
                unique_times = np.unique(time_ids)

                for t in unique_times:
                    time_mask = time_ids == t
                    time_data = data[time_mask]
                    permuted_data[time_mask] = np.random.permutation(time_data)
            else:
                # Cross-sectional: simple permutation
                permuted_data = np.random.permutation(data)

            # Compute test statistic
            permuted_stats[i] = test_statistic(permuted_data, W, **kwargs)

        return permuted_stats


class ParallelBootstrap:
    """
    Parallel bootstrap inference for spatial models.

    Implements various bootstrap schemes with multiprocessing.
    """

    def __init__(
        self,
        estimator: Callable,
        n_bootstrap: int = 1000,
        n_jobs: int = -1,
        bootstrap_type: str = "pairs",
        seed: Optional[int] = None,
    ):
        """
        Initialize parallel bootstrap.

        Parameters
        ----------
        estimator : callable
            Function that estimates the model and returns parameters
        n_bootstrap : int, default 1000
            Number of bootstrap samples
        n_jobs : int, default -1
            Number of parallel jobs
        bootstrap_type : str, default 'pairs'
            Type: 'pairs', 'residual', 'wild', 'block'
        seed : int, optional
            Random seed
        """
        self.estimator = estimator
        self.n_bootstrap = n_bootstrap
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()
        self.bootstrap_type = bootstrap_type
        self.seed = seed

    def run(
        self,
        data: pd.DataFrame,
        W: np.ndarray,
        formula: str,
        entity_col: str,
        time_col: str,
        **model_kwargs,
    ) -> Dict[str, Any]:
        """
        Run parallel bootstrap.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data
        W : np.ndarray
            Spatial weight matrix
        formula : str
            Model formula
        entity_col : str
            Entity column name
        time_col : str
            Time column name
        **model_kwargs
            Additional model arguments

        Returns
        -------
        results : dict
            Bootstrap results with confidence intervals
        """
        # Estimate on original data
        original_params = self.estimator(data, W, formula, entity_col, time_col, **model_kwargs)

        # Setup work packages
        samples_per_worker = self.n_bootstrap // self.n_jobs
        extra_samples = self.n_bootstrap % self.n_jobs

        work_packages = []
        for i in range(self.n_jobs):
            n_samples = samples_per_worker + (1 if i < extra_samples else 0)
            if n_samples > 0:
                seed_i = None if self.seed is None else self.seed + i
                work_packages.append(
                    (
                        n_samples,
                        data,
                        W,
                        formula,
                        entity_col,
                        time_col,
                        model_kwargs,
                        self.bootstrap_type,
                        seed_i,
                    )
                )

        # Run parallel bootstrap
        with Pool(processes=self.n_jobs) as pool:
            worker_func = partial(self._worker_bootstrap, self.estimator)
            results = pool.map(worker_func, work_packages)

        # Combine results
        all_params = np.vstack(results)

        # Calculate statistics
        param_names = list(original_params.keys())
        bootstrap_stats = {}

        for i, name in enumerate(param_names):
            param_samples = all_params[:, i]
            bootstrap_stats[name] = {
                "mean": np.mean(param_samples),
                "std": np.std(param_samples),
                "ci_lower": np.percentile(param_samples, 2.5),
                "ci_upper": np.percentile(param_samples, 97.5),
                "samples": param_samples,
            }

        return {
            "original": original_params,
            "bootstrap": bootstrap_stats,
            "n_bootstrap": self.n_bootstrap,
            "type": self.bootstrap_type,
        }

    @staticmethod
    def _worker_bootstrap(estimator: Callable, work_package: Tuple) -> np.ndarray:
        """
        Worker function for bootstrap computation.

        Parameters
        ----------
        estimator : callable
            Model estimator function
        work_package : tuple
            Bootstrap work package

        Returns
        -------
        params : np.ndarray
            Array of bootstrap parameter estimates
        """
        (n_samples, data, W, formula, entity_col, time_col, model_kwargs, bootstrap_type, seed) = (
            work_package
        )

        if seed is not None:
            np.random.seed(seed)

        n_entities = data[entity_col].nunique()
        n_time = data[time_col].nunique()
        param_results = []

        for _ in range(n_samples):
            # Generate bootstrap sample
            if bootstrap_type == "pairs":
                # Bootstrap entity-time pairs
                sampled_entities = np.random.choice(
                    data[entity_col].unique(), n_entities, replace=True
                )
                boot_data = pd.concat(
                    [data[data[entity_col] == ent].copy() for ent in sampled_entities],
                    ignore_index=True,
                )

                # Renumber entities
                entity_map = {old: new for new, old in enumerate(sampled_entities)}
                boot_data[entity_col] = boot_data[entity_col].map(entity_map)

            elif bootstrap_type == "wild":
                # Wild bootstrap for residuals
                boot_data = data.copy()
                # Apply Rademacher weights
                weights = np.random.choice([-1, 1], len(data))
                # This would need the residuals from original estimation
                # Simplified version here
                boot_data["y"] = boot_data["y"] * weights

            elif bootstrap_type == "block":
                # Block bootstrap by time
                sampled_times = np.random.choice(data[time_col].unique(), n_time, replace=True)
                boot_data = pd.concat(
                    [data[data[time_col] == t].copy() for t in sampled_times], ignore_index=True
                )

            else:
                # Standard residual bootstrap
                boot_data = data.copy()
                # Shuffle residuals (simplified)
                boot_data["y"] = np.random.permutation(boot_data["y"].values)

            # Estimate on bootstrap sample
            try:
                params = estimator(boot_data, W, formula, entity_col, time_col, **model_kwargs)
                param_results.append(list(params.values()))
            except Exception as e:
                # Handle estimation failure
                warnings.warn(f"Bootstrap estimation failed: {e}")
                param_results.append(
                    [np.nan] * len(param_results[0]) if param_results else [np.nan]
                )

        return np.array(param_results)


class ParallelSpatialHAC:
    """
    Parallel computation of Spatial HAC standard errors.

    Speeds up distance and kernel calculations using multiprocessing.
    """

    def __init__(self, n_jobs: int = -1):
        """
        Initialize parallel Spatial HAC.

        Parameters
        ----------
        n_jobs : int, default -1
            Number of parallel jobs
        """
        self.n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    def compute_spatial_weights(
        self, coords: np.ndarray, cutoff: float, kernel: str = "bartlett"
    ) -> np.ndarray:
        """
        Compute spatial weights matrix in parallel.

        Parameters
        ----------
        coords : np.ndarray
            Coordinate array (n x 2)
        cutoff : float
            Spatial cutoff distance
        kernel : str, default 'bartlett'
            Kernel type

        Returns
        -------
        weights : np.ndarray
            Spatial weight matrix
        """
        n = len(coords)

        # Divide computation into chunks
        chunk_size = n // self.n_jobs
        chunks = []

        for i in range(self.n_jobs):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_jobs - 1 else n
            chunks.append((start, end, coords, cutoff, kernel))

        # Compute in parallel
        with Pool(processes=self.n_jobs) as pool:
            results = pool.map(self._compute_chunk, chunks)

        # Combine results
        weights = np.zeros((n, n))
        for i, (start, end, _, _, _) in enumerate(chunks):
            weights[start:end, :] = results[i]

        # Make symmetric
        weights = (weights + weights.T) / 2
        np.fill_diagonal(weights, 1.0)

        return weights

    @staticmethod
    def _compute_chunk(chunk_info: Tuple) -> np.ndarray:
        """
        Compute a chunk of the spatial weight matrix.

        Parameters
        ----------
        chunk_info : tuple
            (start, end, coords, cutoff, kernel)

        Returns
        -------
        chunk_weights : np.ndarray
            Chunk of weight matrix
        """
        start, end, coords, cutoff, kernel = chunk_info
        n = len(coords)
        chunk_weights = np.zeros((end - start, n))

        for i in range(start, end):
            for j in range(n):
                if i != j:
                    # Compute distance
                    dist = np.linalg.norm(coords[i] - coords[j])

                    if dist <= cutoff:
                        # Apply kernel
                        if kernel == "bartlett":
                            weight = 1 - dist / cutoff
                        elif kernel == "uniform":
                            weight = 1.0
                        elif kernel == "epanechnikov":
                            u = dist / cutoff
                            weight = 0.75 * (1 - u**2)
                        else:
                            weight = 1.0

                        chunk_weights[i - start, j] = weight

        return chunk_weights


def parallel_grid_search(
    model_class,
    param_grid: Dict[str, List],
    data: pd.DataFrame,
    W: np.ndarray,
    n_jobs: int = -1,
    scoring: str = "aic",
    **model_kwargs,
) -> Dict[str, Any]:
    """
    Parallel grid search for spatial model hyperparameters.

    Parameters
    ----------
    model_class : class
        Spatial model class
    param_grid : dict
        Parameter grid to search
    data : pd.DataFrame
        Panel data
    W : np.ndarray
        Spatial weight matrix
    n_jobs : int, default -1
        Number of parallel jobs
    scoring : str, default 'aic'
        Scoring metric: 'aic', 'bic', 'log_likelihood'
    **model_kwargs
        Additional model arguments

    Returns
    -------
    results : dict
        Best parameters and scores
    """
    from itertools import product

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))

    # Create work packages
    work_packages = [
        (model_class, dict(zip(param_names, params)), data, W, scoring, model_kwargs)
        for params in param_combinations
    ]

    # Run in parallel
    n_jobs = n_jobs if n_jobs > 0 else cpu_count()

    with Pool(processes=n_jobs) as pool:
        results = pool.map(_evaluate_params, work_packages)

    # Find best parameters
    scores = [r["score"] for r in results]
    best_idx = np.argmin(scores) if scoring in ["aic", "bic"] else np.argmax(scores)

    best_result = {
        "best_params": results[best_idx]["params"],
        "best_score": results[best_idx]["score"],
        "all_results": results,
    }

    return best_result


def _evaluate_params(work_package: Tuple) -> Dict[str, Any]:
    """
    Evaluate a single parameter combination.

    Parameters
    ----------
    work_package : tuple
        (model_class, params, data, W, scoring, model_kwargs)

    Returns
    -------
    result : dict
        Parameter evaluation result
    """
    model_class, params, data, W, scoring, model_kwargs = work_package

    try:
        # Create and fit model
        model = model_class(data=data, W=W, **model_kwargs, **params)
        result = model.fit()

        # Calculate score
        if scoring == "aic":
            score = result.aic
        elif scoring == "bic":
            score = result.bic
        else:  # log_likelihood
            score = result.log_likelihood

        return {"params": params, "score": score, "converged": result.converged}
    except Exception as e:
        return {
            "params": params,
            "score": np.inf if scoring in ["aic", "bic"] else -np.inf,
            "converged": False,
            "error": str(e),
        }
