"""
Data generation utilities for standard errors tutorials.

This module provides functions to generate synthetic panel data with various
error structures for simulation and demonstration purposes.

Functions
---------
generate_heteroskedastic_data : Generate data with heteroskedasticity
generate_autocorrelated_panel : Generate panel with AR(1) errors
generate_spatial_panel : Generate panel with spatial correlation
generate_clustered_data : Generate data with within-cluster correlation

Author: PanelBox Development Team
Date: 2026-02-16
Version: 1.0.0
"""

from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def generate_heteroskedastic_data(
    n: int = 1000,
    k: int = 3,
    hetero_type: str = "multiplicative",
    hetero_strength: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate cross-sectional data with heteroskedasticity.

    Parameters
    ----------
    n : int, optional
        Number of observations
    k : int, optional
        Number of regressors (excluding constant)
    hetero_type : str, optional
        Type of heteroskedasticity:
        - 'multiplicative': var(u) = sigma^2 * X_1^2
        - 'exponential': var(u) = exp(alpha * X_1)
        - 'grouped': var(u) differs by group
    hetero_strength : float, optional
        Strength of heteroskedasticity (higher = more severe)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    data : pd.DataFrame
        Generated data with columns ['y', 'X1', 'X2', ..., 'Xk']
    params : dict
        True parameter values used in generation
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate regressors
    X = np.random.randn(n, k)

    # True coefficients
    beta = np.random.uniform(0.5, 2.0, k)
    intercept = 1.0

    # Linear prediction
    y_mean = intercept + X @ beta

    # Generate heteroskedastic errors
    if hetero_type == "multiplicative":
        # Variance proportional to X_1^2
        sigma = 1 + hetero_strength * np.abs(X[:, 0])
        errors = np.random.randn(n) * sigma

    elif hetero_type == "exponential":
        # Variance exponential in X_1
        sigma = np.exp(hetero_strength * X[:, 0] / 2)
        errors = np.random.randn(n) * sigma

    elif hetero_type == "grouped":
        # Different variance by group (based on X_1)
        group = (X[:, 0] > 0).astype(int)
        sigma = np.where(group == 1, 1 + hetero_strength, 1.0)
        errors = np.random.randn(n) * sigma

    else:
        raise ValueError(f"Unknown hetero_type: {hetero_type}")

    # Generate outcome
    y = y_mean + errors

    # Create DataFrame
    data = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(k)])
    data.insert(0, "y", y)

    params = {
        "intercept": intercept,
        "beta": beta,
        "n": n,
        "k": k,
        "hetero_type": hetero_type,
        "hetero_strength": hetero_strength,
    }

    return data, params


def generate_autocorrelated_panel(
    n_entities: int = 50,
    n_time: int = 20,
    k: int = 2,
    rho: float = 0.7,
    include_fixed_effects: bool = True,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate panel data with AR(1) autocorrelated errors.

    Parameters
    ----------
    n_entities : int, optional
        Number of entities (cross-sectional units)
    n_time : int, optional
        Number of time periods
    k : int, optional
        Number of time-varying regressors
    rho : float, optional
        AR(1) coefficient (-1 < rho < 1)
    include_fixed_effects : bool, optional
        Include entity fixed effects
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    data : pd.DataFrame
        Generated panel data with columns ['entity', 'time', 'y', 'X1', ..., 'Xk']
    params : dict
        True parameter values
    """
    if seed is not None:
        np.random.seed(seed)

    if not -1 < rho < 1:
        raise ValueError("rho must be in (-1, 1)")

    # True coefficients
    beta = np.random.uniform(0.5, 2.0, k)
    intercept = 1.0

    data_list = []

    for entity_id in range(1, n_entities + 1):
        # Entity fixed effect
        if include_fixed_effects:
            alpha_i = np.random.normal(0, 2)
        else:
            alpha_i = 0

        # Generate time-varying regressors
        X_entity = np.random.randn(n_time, k)

        # Linear prediction
        y_mean = intercept + alpha_i + X_entity @ beta

        # Generate AR(1) errors
        errors = np.zeros(n_time)
        errors[0] = np.random.normal(0, 1)

        for t in range(1, n_time):
            errors[t] = rho * errors[t - 1] + np.random.normal(0, 1)

        # Generate outcome
        y = y_mean + errors

        # Create entity DataFrame
        entity_df = pd.DataFrame(X_entity, columns=[f"X{i+1}" for i in range(k)])
        entity_df["entity"] = entity_id
        entity_df["time"] = np.arange(1, n_time + 1)
        entity_df["y"] = y

        data_list.append(entity_df)

    # Combine all entities
    data = pd.concat(data_list, ignore_index=True)
    data = data[["entity", "time", "y"] + [f"X{i+1}" for i in range(k)]]

    params = {
        "intercept": intercept,
        "beta": beta,
        "rho": rho,
        "n_entities": n_entities,
        "n_time": n_time,
        "k": k,
        "include_fixed_effects": include_fixed_effects,
    }

    return data, params


def generate_spatial_panel(
    n_entities: int = 100,
    n_time: int = 10,
    k: int = 2,
    spatial_decay: float = 0.3,
    spatial_range: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate panel data with spatial correlation.

    Entities are located on a 2D grid and errors exhibit spatial correlation
    that decays with distance.

    Parameters
    ----------
    n_entities : int, optional
        Number of spatial entities
    n_time : int, optional
        Number of time periods
    k : int, optional
        Number of regressors
    spatial_decay : float, optional
        Rate of spatial decay (higher = faster decay)
    spatial_range : float, optional
        Maximum range for spatial correlation
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    data : pd.DataFrame
        Generated panel with spatial structure
        Columns: ['entity', 'time', 'y', 'X1', ..., 'Xk', 'latitude', 'longitude']
    params : dict
        True parameter values and spatial coordinates
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate spatial coordinates (entities on a grid)
    grid_size = int(np.ceil(np.sqrt(n_entities)))
    coords = []

    for i in range(n_entities):
        lat = (i // grid_size) + np.random.normal(0, 0.1)
        lon = (i % grid_size) + np.random.normal(0, 0.1)
        coords.append([lat, lon])

    coords = np.array(coords)

    # Compute distance matrix
    distances = cdist(coords, coords, metric="euclidean")

    # Create spatial correlation matrix (exponential decay kernel)
    spatial_corr = np.exp(-spatial_decay * distances)
    spatial_corr[distances > spatial_range] = 0  # Truncate beyond range

    # Ensure positive definiteness
    spatial_corr = (spatial_corr + spatial_corr.T) / 2
    np.fill_diagonal(spatial_corr, 1.0)

    # Cholesky decomposition for generating correlated errors
    try:
        L = np.linalg.cholesky(spatial_corr + np.eye(n_entities) * 1e-6)
    except np.linalg.LinAlgError:
        # Fallback: use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(spatial_corr)
        eigvals = np.maximum(eigvals, 1e-6)  # Ensure positive
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    # True coefficients
    beta = np.random.uniform(0.5, 2.0, k)
    intercept = 1.0

    data_list = []

    for t in range(1, n_time + 1):
        # Generate regressors (time-varying)
        X_t = np.random.randn(n_entities, k)

        # Linear prediction
        y_mean = intercept + X_t @ beta

        # Generate spatially correlated errors for this time period
        uncorr_errors = np.random.randn(n_entities)
        spatial_errors = L @ uncorr_errors

        # Generate outcome
        y = y_mean + spatial_errors

        # Create time period DataFrame
        time_df = pd.DataFrame(X_t, columns=[f"X{i+1}" for i in range(k)])
        time_df["entity"] = np.arange(1, n_entities + 1)
        time_df["time"] = t
        time_df["y"] = y
        time_df["latitude"] = coords[:, 0]
        time_df["longitude"] = coords[:, 1]

        data_list.append(time_df)

    # Combine all time periods
    data = pd.concat(data_list, ignore_index=True)
    data = data[["entity", "time", "y"] + [f"X{i+1}" for i in range(k)] + ["latitude", "longitude"]]

    params = {
        "intercept": intercept,
        "beta": beta,
        "n_entities": n_entities,
        "n_time": n_time,
        "k": k,
        "spatial_decay": spatial_decay,
        "spatial_range": spatial_range,
        "coordinates": coords,
    }

    return data, params


def generate_clustered_data(
    n_clusters: int = 30,
    cluster_size_mean: int = 20,
    cluster_size_std: int = 5,
    k: int = 2,
    within_cluster_corr: float = 0.5,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate cross-sectional data with clustered errors.

    Errors are correlated within clusters but independent across clusters.

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters
    cluster_size_mean : int, optional
        Mean cluster size
    cluster_size_std : int, optional
        Standard deviation of cluster sizes
    k : int, optional
        Number of regressors
    within_cluster_corr : float, optional
        Within-cluster correlation (0 to 1)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    data : pd.DataFrame
        Generated data with columns ['cluster_id', 'y', 'X1', ..., 'Xk']
    params : dict
        True parameter values
    """
    if seed is not None:
        np.random.seed(seed)

    if not 0 <= within_cluster_corr < 1:
        raise ValueError("within_cluster_corr must be in [0, 1)")

    # True coefficients
    beta = np.random.uniform(0.5, 2.0, k)
    intercept = 1.0

    data_list = []

    for cluster_id in range(1, n_clusters + 1):
        # Cluster size (truncated normal)
        cluster_size = max(1, int(np.random.normal(cluster_size_mean, cluster_size_std)))

        # Generate regressors
        X_cluster = np.random.randn(cluster_size, k)

        # Linear prediction
        y_mean = intercept + X_cluster @ beta

        # Generate clustered errors
        # Error structure: u_ic = v_c + e_ic
        # where v_c is cluster component and e_ic is individual component

        # Variance decomposition to achieve target correlation
        # Corr(u_ic, u_jc) = var(v_c) / [var(v_c) + var(e_ic)] = rho
        # => var(v_c) = rho, var(e_ic) = 1 - rho

        var_cluster = within_cluster_corr
        var_individual = 1 - within_cluster_corr

        cluster_component = np.random.normal(0, np.sqrt(var_cluster))
        individual_components = np.random.normal(0, np.sqrt(var_individual), cluster_size)

        errors = cluster_component + individual_components

        # Generate outcome
        y = y_mean + errors

        # Create cluster DataFrame
        cluster_df = pd.DataFrame(X_cluster, columns=[f"X{i+1}" for i in range(k)])
        cluster_df["cluster_id"] = cluster_id
        cluster_df["y"] = y

        data_list.append(cluster_df)

    # Combine all clusters
    data = pd.concat(data_list, ignore_index=True)
    data = data[["cluster_id", "y"] + [f"X{i+1}" for i in range(k)]]

    params = {
        "intercept": intercept,
        "beta": beta,
        "n_clusters": n_clusters,
        "cluster_size_mean": cluster_size_mean,
        "within_cluster_corr": within_cluster_corr,
        "k": k,
    }

    return data, params


def generate_panel_with_effects(
    n_entities: int = 100,
    n_time: int = 10,
    k: int = 2,
    entity_fe_std: float = 1.0,
    time_fe_std: float = 0.5,
    include_entity_fe: bool = True,
    include_time_fe: bool = True,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Generate panel data with entity and/or time fixed effects.

    Parameters
    ----------
    n_entities : int, optional
        Number of entities
    n_time : int, optional
        Number of time periods
    k : int, optional
        Number of time-varying regressors
    entity_fe_std : float, optional
        Standard deviation of entity fixed effects
    time_fe_std : float, optional
        Standard deviation of time fixed effects
    include_entity_fe : bool, optional
        Include entity fixed effects
    include_time_fe : bool, optional
        Include time fixed effects
    seed : int, optional
        Random seed

    Returns
    -------
    data : pd.DataFrame
        Generated panel data
    params : dict
        True parameter values including fixed effects
    """
    if seed is not None:
        np.random.seed(seed)

    # True coefficients
    beta = np.random.uniform(0.5, 2.0, k)
    intercept = 1.0

    # Generate fixed effects
    if include_entity_fe:
        entity_effects = np.random.normal(0, entity_fe_std, n_entities)
    else:
        entity_effects = np.zeros(n_entities)

    if include_time_fe:
        time_effects = np.random.normal(0, time_fe_std, n_time)
    else:
        time_effects = np.zeros(n_time)

    data_list = []

    for entity_id in range(1, n_entities + 1):
        entity_fe = entity_effects[entity_id - 1]

        for t in range(1, n_time + 1):
            time_fe = time_effects[t - 1]

            # Generate regressors
            X_it = np.random.randn(k)

            # Linear prediction
            y_mean = intercept + entity_fe + time_fe + X_it @ beta

            # Idiosyncratic error
            error = np.random.normal(0, 1)

            # Generate outcome
            y = y_mean + error

            # Create observation
            obs = {"entity": entity_id, "time": t, "y": y}
            for i in range(k):
                obs[f"X{i+1}"] = X_it[i]

            data_list.append(obs)

    # Create DataFrame
    data = pd.DataFrame(data_list)
    data = data[["entity", "time", "y"] + [f"X{i+1}" for i in range(k)]]

    params = {
        "intercept": intercept,
        "beta": beta,
        "entity_effects": entity_effects if include_entity_fe else None,
        "time_effects": time_effects if include_time_fe else None,
        "n_entities": n_entities,
        "n_time": n_time,
        "k": k,
    }

    return data, params


# Export all functions
__all__ = [
    "generate_heteroskedastic_data",
    "generate_autocorrelated_panel",
    "generate_spatial_panel",
    "generate_clustered_data",
    "generate_panel_with_effects",
]
