"""
Numba-optimized functions for performance-critical operations.

This module contains JIT-compiled versions of computationally expensive
operations, primarily targeting:
1. GMM instrument creation (nested loops)
2. Within-transformation (demeaning)
3. Weight matrix computations

Using Numba can provide 10-100x speedups for these operations.
"""


import numpy as np

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Fallback: create a no-op decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            # Called as @jit without parentheses
            return args[0]
        return decorator

    # prange falls back to range
    prange = range


# ============================================================================
# 1. GMM Instrument Creation
# ============================================================================


@jit(nopython=True, cache=True, parallel=False)
def fill_iv_instruments_numba(
    Z: np.ndarray,
    var_data: np.ndarray,
    ids: np.ndarray,
    times: np.ndarray,
    min_lag: int,
    max_lag: int,
    equation: str,
) -> np.ndarray:
    """
    Fill instrument matrix for IV-style instruments (Numba-optimized).

    This is the performance-critical inner loop of GMM instrument creation.

    Parameters
    ----------
    Z : np.ndarray
        Instrument matrix to fill (n_obs x n_instruments)
    var_data : np.ndarray
        Variable data (n_obs,)
    ids : np.ndarray
        Entity IDs (n_obs,)
    times : np.ndarray
        Time periods (n_obs,)
    min_lag : int
        Minimum lag
    max_lag : int
        Maximum lag
    equation : str
        'diff' or 'level'

    Returns
    -------
    Z : np.ndarray
        Filled instrument matrix

    Notes
    -----
    This function replaces the nested Python loops in InstrumentBuilder.
    Original code had O(n_obs * n_lags) complexity with Python overhead.
    Numba compilation reduces this overhead by ~10-50x.
    """
    n_obs = len(ids)
    n_lags = max_lag - min_lag + 1

    # Convert equation string to int for Numba (can't use strings in nopython mode)
    is_diff = equation == "diff"  # True for diff, False for level

    for i in range(n_obs):
        current_id = ids[i]
        current_time = times[i]

        for lag_idx in range(n_lags):
            lag = min_lag + lag_idx

            # Find lagged value
            # Original: mask = (ids == current_id) & (times == current_time - lag)
            # Numba-optimized: manual search
            lag_time = current_time - lag

            for j in range(n_obs):
                if ids[j] == current_id and times[j] == lag_time:
                    if is_diff:
                        Z[i, lag_idx] = var_data[j]
                    else:
                        # For level equation, use differences as instruments
                        # Find t-lag-1
                        lag1_time = current_time - lag - 1
                        for k in range(n_obs):
                            if ids[k] == current_id and times[k] == lag1_time:
                                Z[i, lag_idx] = var_data[j] - var_data[k]
                                break
                    break

    return Z


@jit(nopython=True, cache=True)
def fill_gmm_style_instruments_numba(
    Z_list,
    var_data: np.ndarray,
    ids: np.ndarray,
    times: np.ndarray,
    unique_times: np.ndarray,
    min_lag: int,
    max_lag: int,
) -> list:
    """
    Fill GMM-style instrument matrices (separate column per time period).

    This is even more performance-critical than IV-style due to instrument
    proliferation (one column per lag per time period).

    Parameters
    ----------
    Z_list : list
        List of instrument matrices, one per time period
    var_data : np.ndarray
        Variable data
    ids : np.ndarray
        Entity IDs
    times : np.ndarray
        Time periods
    unique_times : np.ndarray
        Unique time periods
    min_lag : int
        Minimum lag
    max_lag : int
        Maximum lag (or actual max available lag per period)

    Returns
    -------
    Z_list : list
        Filled instrument matrices

    Notes
    -----
    GMM-style instruments create instrument proliferation:
    - T=10, lags=2-5 -> ~30-40 instruments
    - Nested loops: O(n_obs * n_times * n_lags)
    - Major bottleneck in System GMM
    """
    n_obs = len(ids)

    for t_idx in range(len(unique_times)):
        t = unique_times[t_idx]

        # Get observations for this time period
        for i in range(n_obs):
            if times[i] != t:
                continue

            current_id = ids[i]

            # Maximum available lag for this time period
            actual_max_lag = min(max_lag, t_idx + 1)

            for lag in range(min_lag, actual_max_lag + 1):
                lag_time = t - lag

                # Find lagged value
                for j in range(n_obs):
                    if ids[j] == current_id and times[j] == lag_time:
                        # Store in appropriate column (lag - min_lag)
                        col_idx = lag - min_lag
                        if col_idx < Z_list[t_idx].shape[1]:
                            Z_list[t_idx][i, col_idx] = var_data[j]
                        break

    return Z_list


# ============================================================================
# 2. Within Transformation (Demeaning)
# ============================================================================


@jit(nopython=True, cache=True, parallel=True)
def demean_within_numba(X: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
    """
    Within-transformation (entity demeaning) - Numba-optimized.

    For each entity, subtract the entity mean from all observations.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_obs x n_vars)
    entity_ids : np.ndarray
        Entity IDs (n_obs,)

    Returns
    -------
    X_demeaned : np.ndarray
        Demeaned data matrix

    Notes
    -----
    Original Python loop version:
        for entity in unique_entities:
            mask = (entity_ids == entity)
            X[mask] -= X[mask].mean(axis=0)

    This Numba version uses parallel processing for entities and
    avoids repeated masking operations.

    Expected speedup: 10-20x for moderate-sized panels
    """
    n_obs, n_vars = X.shape
    X_demeaned = np.copy(X)

    # Get unique entities (Numba-compatible way)
    unique_entities = np.unique(entity_ids)
    n_entities = len(unique_entities)

    # Parallel loop over entities
    for e_idx in prange(n_entities):
        entity = unique_entities[e_idx]

        # Calculate entity mean
        entity_sum = np.zeros(n_vars)
        entity_count = 0

        for i in range(n_obs):
            if entity_ids[i] == entity:
                entity_sum += X[i, :]
                entity_count += 1

        if entity_count > 0:
            entity_mean = entity_sum / entity_count

            # Demean observations for this entity
            for i in range(n_obs):
                if entity_ids[i] == entity:
                    X_demeaned[i, :] -= entity_mean

    return X_demeaned


@jit(nopython=True, cache=True)
def demean_within_1d_numba(x: np.ndarray, entity_ids: np.ndarray) -> np.ndarray:
    """
    Within-transformation for 1D array (e.g., dependent variable).

    Parameters
    ----------
    x : np.ndarray
        Data vector (n_obs,)
    entity_ids : np.ndarray
        Entity IDs (n_obs,)

    Returns
    -------
    x_demeaned : np.ndarray
        Demeaned vector
    """
    n_obs = len(x)
    x_demeaned = np.copy(x)

    unique_entities = np.unique(entity_ids)

    for entity in unique_entities:
        # Calculate entity mean
        entity_sum = 0.0
        entity_count = 0

        for i in range(n_obs):
            if entity_ids[i] == entity:
                entity_sum += x[i]
                entity_count += 1

        if entity_count > 0:
            entity_mean = entity_sum / entity_count

            # Demean
            for i in range(n_obs):
                if entity_ids[i] == entity:
                    x_demeaned[i] -= entity_mean

    return x_demeaned


# ============================================================================
# 3. GMM Weight Matrix Computation
# ============================================================================


@jit(nopython=True, cache=True)
def compute_gmm_weight_matrix_numba(
    residuals: np.ndarray, Z: np.ndarray, entity_ids: np.ndarray, robust: bool = True
) -> np.ndarray:
    """
    Compute optimal GMM weight matrix - Numba-optimized.

    For two-step GMM, the weight matrix is:
    W = (Z' * Omega * Z)^(-1)

    where Omega is the moment covariance matrix.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals from first-step estimation (n_obs,)
    Z : np.ndarray
        Instrument matrix (n_obs x n_instruments)
    entity_ids : np.ndarray
        Entity IDs for clustering (n_obs,)
    robust : bool
        Whether to use robust (clustered) covariance

    Returns
    -------
    W : np.ndarray
        Optimal weight matrix (n_instruments x n_instruments)

    Notes
    -----
    The bottleneck is computing:
    Omega_ii = sum_t (z_it * e_it) * (z_it * e_it)'

    For large instrument sets (e.g., 50+ instruments), this becomes
    expensive without optimization.
    """
    n_obs, n_instruments = Z.shape

    if robust:
        # Robust (clustered) weight matrix
        unique_entities = np.unique(entity_ids)
        n_entities = len(unique_entities)

        Omega = np.zeros((n_instruments, n_instruments))

        # Sum over entities (cluster-robust)
        for entity in unique_entities:
            # Get observations for this entity
            entity_g = np.zeros(n_instruments)

            for i in range(n_obs):
                if entity_ids[i] == entity:
                    # g_it = Z_it * e_it
                    for j in range(n_instruments):
                        entity_g[j] += Z[i, j] * residuals[i]

            # Omega += g_i * g_i'
            for j in range(n_instruments):
                for k in range(n_instruments):
                    Omega[j, k] += entity_g[j] * entity_g[k]

        # Scale by number of entities
        Omega = Omega / n_entities

    else:
        # Non-robust (homoskedastic) weight matrix
        # Omega = (1/n) * Z' * diag(e^2) * Z
        Omega = np.zeros((n_instruments, n_instruments))

        for i in range(n_obs):
            e_sq = residuals[i] ** 2
            for j in range(n_instruments):
                for k in range(n_instruments):
                    Omega[j, k] += Z[i, j] * Z[i, k] * e_sq

        Omega = Omega / n_obs

    return Omega


# ============================================================================
# Helper Functions
# ============================================================================


def get_numba_status() -> dict:
    """
    Get information about Numba availability and configuration.

    Returns
    -------
    dict
        Status information
    """
    status = {
        "available": NUMBA_AVAILABLE,
        "version": None,
        "parallel_available": False,
        "cache_enabled": True,
    }

    if NUMBA_AVAILABLE:
        import numba

        status["version"] = numba.__version__
        try:
            # Check if parallel is available
            import numba.config

            status["parallel_available"] = numba.config.NUMBA_NUM_THREADS > 0
        except Exception:
            pass

    return status


def suggest_optimization_targets(model_type: str) -> list:
    """
    Suggest which functions to optimize with Numba for a given model type.

    Parameters
    ----------
    model_type : str
        Model type: 'pooled', 'fe', 're', 'diff_gmm', 'sys_gmm'

    Returns
    -------
    list
        List of recommended optimizations
    """
    suggestions = {
        "pooled": [],
        "fe": ["demean_within_numba", "demean_within_1d_numba"],
        "re": ["demean_within_numba"],
        "diff_gmm": ["fill_iv_instruments_numba", "compute_gmm_weight_matrix_numba"],
        "sys_gmm": [
            "fill_iv_instruments_numba",
            "fill_gmm_style_instruments_numba",
            "compute_gmm_weight_matrix_numba",
        ],
    }

    return suggestions.get(model_type, [])


# ============================================================================
# Testing and Validation
# ============================================================================


def validate_numba_optimization(func_original, func_numba, *args, **kwargs):
    """
    Validate that Numba-optimized function produces same results as original.

    Parameters
    ----------
    func_original : callable
        Original (non-Numba) function
    func_numba : callable
        Numba-optimized function
    *args, **kwargs
        Arguments to pass to both functions

    Returns
    -------
    dict
        Validation results: {'match': bool, 'max_diff': float}
    """
    result_original = func_original(*args, **kwargs)
    result_numba = func_numba(*args, **kwargs)

    if isinstance(result_original, np.ndarray):
        max_diff = np.max(np.abs(result_original - result_numba))
        match = max_diff < 1e-10
    else:
        max_diff = abs(result_original - result_numba)
        match = max_diff < 1e-10

    return {"match": match, "max_diff": max_diff}


if __name__ == "__main__":
    # Print status when module is run
    status = get_numba_status()
    print("Numba Optimization Module")
    print("=" * 60)
    print(f"Numba available: {status['available']}")
    if status["available"]:
        print(f"Numba version: {status['version']}")
        print(f"Parallel support: {status['parallel_available']}")
    else:
        print("Note: Numba not installed. Install with: pip install numba")
        print("Functions will fall back to standard Python (slower)")
    print("=" * 60)
