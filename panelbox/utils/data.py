"""
Data utilities for panel data handling.

This module provides utilities for checking and validating panel data structures,
including balanced/unbalanced panels and data transformation functions.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd


def check_panel_data(
    y: Union[np.ndarray, pd.Series, pd.DataFrame],
    X: Union[np.ndarray, pd.DataFrame],
    entity_id: Optional[Union[np.ndarray, pd.Series]] = None,
    time_id: Optional[Union[np.ndarray, pd.Series]] = None,
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Check and validate panel data inputs.

    Parameters
    ----------
    y : array-like
        Dependent variable
    X : array-like
        Independent variables
    entity_id : array-like, optional
        Entity identifiers
    time_id : array-like, optional
        Time period identifiers
    weights : array-like, optional
        Observation weights

    Returns
    -------
    y : ndarray
        Validated dependent variable
    X : ndarray
        Validated independent variables
    entity_id : ndarray
        Entity identifiers
    time_id : ndarray
        Time identifiers
    weights : ndarray or None
        Observation weights if provided

    Raises
    ------
    ValueError
        If data dimensions are inconsistent
    """
    # Convert to numpy arrays
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    if isinstance(X, pd.DataFrame):
        X = X.values
    if entity_id is not None and isinstance(entity_id, pd.Series):
        entity_id = entity_id.values
    if time_id is not None and isinstance(time_id, pd.Series):
        time_id = time_id.values
    if weights is not None and isinstance(weights, pd.Series):
        weights = weights.values

    # Ensure arrays
    y = np.asarray(y)
    X = np.asarray(X)

    # Flatten y if needed
    if y.ndim > 1:
        y = y.flatten()

    # Check dimensions
    n_obs = len(y)
    if X.shape[0] != n_obs:
        raise ValueError(f"X has {X.shape[0]} observations but y has {n_obs}")

    # Create default entity and time IDs if not provided
    if entity_id is None:
        # Assume single entity
        entity_id = np.zeros(n_obs, dtype=int)
    else:
        entity_id = np.asarray(entity_id)

    if time_id is None:
        # Assume sequential time periods
        time_id = np.arange(n_obs, dtype=int)
    else:
        time_id = np.asarray(time_id)

    # Validate lengths
    if len(entity_id) != n_obs:
        raise ValueError(f"entity_id has {len(entity_id)} elements but y has {n_obs}")
    if len(time_id) != n_obs:
        raise ValueError(f"time_id has {len(time_id)} elements but y has {n_obs}")

    if weights is not None:
        weights = np.asarray(weights)
        if len(weights) != n_obs:
            raise ValueError(f"weights has {len(weights)} elements but y has {n_obs}")
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")

    return y, X, entity_id, time_id, weights


def is_balanced_panel(entity_id: np.ndarray, time_id: np.ndarray) -> bool:
    """
    Check if panel is balanced.

    Parameters
    ----------
    entity_id : ndarray
        Entity identifiers
    time_id : ndarray
        Time identifiers

    Returns
    -------
    bool
        True if panel is balanced, False otherwise
    """
    entities = np.unique(entity_id)
    periods = np.unique(time_id)

    # Check if each entity has all time periods
    for entity in entities:
        entity_periods = time_id[entity_id == entity]
        if len(np.unique(entity_periods)) != len(periods):
            return False

    return True


def panel_to_dict(
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> dict:
    """
    Convert panel data to dictionary format indexed by entity.

    Parameters
    ----------
    y : ndarray
        Dependent variable
    X : ndarray
        Independent variables
    entity_id : ndarray
        Entity identifiers
    time_id : ndarray
        Time identifiers
    weights : ndarray, optional
        Observation weights

    Returns
    -------
    dict
        Dictionary with entity IDs as keys and data arrays as values
    """
    data_dict = {}
    entities = np.unique(entity_id)

    for entity in entities:
        mask = entity_id == entity
        entity_data = {"y": y[mask], "X": X[mask], "time": time_id[mask]}
        if weights is not None:
            entity_data["weights"] = weights[mask]

        # Sort by time
        sort_idx = np.argsort(entity_data["time"])
        for key in entity_data:
            entity_data[key] = entity_data[key][sort_idx]

        data_dict[entity] = entity_data

    return data_dict


def get_panel_info(entity_id: np.ndarray, time_id: np.ndarray) -> dict:
    """
    Get panel structure information.

    Parameters
    ----------
    entity_id : ndarray
        Entity identifiers
    time_id : ndarray
        Time identifiers

    Returns
    -------
    dict
        Dictionary with panel information
    """
    entities = np.unique(entity_id)
    periods = np.unique(time_id)

    info = {
        "n_entities": len(entities),
        "n_periods": len(periods),
        "n_obs": len(entity_id),
        "is_balanced": is_balanced_panel(entity_id, time_id),
        "entities": entities,
        "periods": periods,
    }

    # Calculate observations per entity
    obs_per_entity = []
    for entity in entities:
        obs_per_entity.append(np.sum(entity_id == entity))

    info["min_obs_per_entity"] = np.min(obs_per_entity)
    info["max_obs_per_entity"] = np.max(obs_per_entity)
    info["avg_obs_per_entity"] = np.mean(obs_per_entity)

    return info
