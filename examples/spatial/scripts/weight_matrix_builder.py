"""
Spatial Weights Matrix Builder for Spatial Econometrics Tutorials

This module provides functions to construct various types of spatial weights matrices
for use in PanelBox spatial models.

Functions:
    build_contiguity_matrix: Construct contiguity-based weights (Queen, Rook)
    build_distance_matrix: Construct distance-based weights
    build_knn_matrix: Construct k-nearest neighbors weights
    build_inverse_distance_matrix: Construct inverse distance weights
    build_economic_weights: Construct weights based on economic flows
    normalize_weights: Row-standardize weights matrix
    compute_weights_properties: Compute W matrix properties
"""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from libpysal import weights


def build_contiguity_matrix(
    shapefile: Union[str, Path],
    criterion: str = "queen",
    id_variable: Optional[str] = None,
    row_standardize: bool = True,
):
    """
    Build contiguity-based spatial weights matrix.

    Parameters
    ----------
    shapefile : str or Path
        Path to shapefile with geographic boundaries
    criterion : str, default "queen"
        Contiguity criterion:
        - 'queen': Shared edge or vertex
        - 'rook': Shared edge only
    id_variable : str, optional
        Column name to use as identifiers
    row_standardize : bool, default True
        Whether to row-standardize the weights

    Returns
    -------
    W : libpysal.weights.W
        Spatial weights object

    Examples
    --------
    >>> W = build_contiguity_matrix(
    ...     "data/us_counties/us_counties.shp", criterion="queen", id_variable="county_fips"
    ... )
    """
    if criterion.lower() == "queen":
        W = weights.Queen.from_shapefile(shapefile, idVariable=id_variable)
    elif criterion.lower() == "rook":
        W = weights.Rook.from_shapefile(shapefile, idVariable=id_variable)
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'queen' or 'rook'.")

    if row_standardize:
        W.transform = "r"

    return W


def build_distance_matrix(
    data: Union[pd.DataFrame, gpd.GeoDataFrame],
    threshold: float,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: Optional[str] = None,
    binary: bool = True,
    row_standardize: bool = True,
):
    """
    Build distance-based spatial weights matrix.

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        Data with coordinates
    threshold : float
        Distance threshold (in same units as coordinates)
    lat_col : str, default "latitude"
        Column name for latitude (if DataFrame)
    lon_col : str, default "longitude"
        Column name for longitude (if DataFrame)
    id_col : str, optional
        Column name for identifiers
    binary : bool, default True
        If True, weights are binary (0/1)
        If False, weights are inverse distance
    row_standardize : bool, default True
        Whether to row-standardize the weights

    Returns
    -------
    W : libpysal.weights.W
        Spatial weights object

    Examples
    --------
    >>> # All units within 100 km are neighbors
    >>> W = build_distance_matrix(df, threshold=100, binary=True)
    """
    # Extract coordinates
    if isinstance(data, gpd.GeoDataFrame):
        coords = np.array([(geom.x, geom.y) for geom in data.geometry.centroid])
    else:
        coords = data[[lon_col, lat_col]].values

    # Get IDs
    ids = data[id_col].values if id_col is not None else None

    # Build distance weights
    W = weights.DistanceBand.from_array(coords, threshold=threshold, binary=binary, ids=ids)

    if row_standardize:
        W.transform = "r"

    return W


def build_knn_matrix(
    data: Union[pd.DataFrame, gpd.GeoDataFrame],
    k: int = 5,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: Optional[str] = None,
    row_standardize: bool = True,
):
    """
    Build k-nearest neighbors spatial weights matrix.

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        Data with coordinates
    k : int, default 5
        Number of nearest neighbors
    lat_col : str, default "latitude"
        Column name for latitude (if DataFrame)
    lon_col : str, default "longitude"
        Column name for longitude (if DataFrame)
    id_col : str, optional
        Column name for identifiers
    row_standardize : bool, default True
        Whether to row-standardize the weights

    Returns
    -------
    W : libpysal.weights.W
        Spatial weights object

    Examples
    --------
    >>> # Each unit has 10 nearest neighbors
    >>> W = build_knn_matrix(df, k=10)
    """
    # Extract coordinates
    if isinstance(data, gpd.GeoDataFrame):
        coords = np.array([(geom.x, geom.y) for geom in data.geometry.centroid])
    else:
        coords = data[[lon_col, lat_col]].values

    # Get IDs
    ids = data[id_col].values if id_col is not None else None

    # Build k-NN weights
    W = weights.KNN.from_array(coords, k=k, ids=ids)

    if row_standardize:
        W.transform = "r"

    return W


def build_inverse_distance_matrix(
    data: Union[pd.DataFrame, gpd.GeoDataFrame],
    power: float = 1.0,
    threshold: Optional[float] = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    id_col: Optional[str] = None,
    row_standardize: bool = True,
):
    """
    Build inverse distance weighted spatial weights matrix.

    Weights are: w_ij = 1 / d_ij^power

    Parameters
    ----------
    data : pd.DataFrame or gpd.GeoDataFrame
        Data with coordinates
    power : float, default 1.0
        Power for inverse distance (1.0 = inverse distance, 2.0 = inverse squared)
    threshold : float, optional
        Distance threshold beyond which weight is zero
    lat_col : str, default "latitude"
        Column name for latitude (if DataFrame)
    lon_col : str, default "longitude"
        Column name for longitude (if DataFrame)
    id_col : str, optional
        Column name for identifiers
    row_standardize : bool, default True
        Whether to row-standardize the weights

    Returns
    -------
    W : libpysal.weights.W
        Spatial weights object

    Examples
    --------
    >>> # Inverse distance with power=2, max distance 500
    >>> W = build_inverse_distance_matrix(df, power=2.0, threshold=500)
    """
    # Extract coordinates
    if isinstance(data, gpd.GeoDataFrame):
        coords = np.array([(geom.x, geom.y) for geom in data.geometry.centroid])
    else:
        coords = data[[lon_col, lat_col]].values

    # Get IDs
    ids = data[id_col].values if id_col is not None else None

    # Build inverse distance weights
    W = weights.DistanceBand.from_array(
        coords, threshold=threshold if threshold else np.inf, binary=False, ids=ids
    )

    # Apply inverse distance transformation
    for i in W.neighbors:
        for j_idx, j in enumerate(W.neighbors[i]):
            # Get distance
            dist = np.linalg.norm(coords[W.id2i[i]] - coords[W.id2i[j]])
            if dist > 0:
                W.weights[i][j_idx] = 1.0 / (dist**power)

    if row_standardize:
        W.transform = "r"

    return W


def build_economic_weights(
    data: pd.DataFrame,
    flow_variable: str,
    origin_id: str = "origin_id",
    destination_id: str = "dest_id",
    row_standardize: bool = True,
):
    """
    Build spatial weights matrix based on economic flows.

    Examples: trade flows, migration flows, commuting patterns

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame with origin-destination flows
    flow_variable : str
        Column name for flow values (e.g., 'trade_value', 'migrants')
    origin_id : str, default "origin_id"
        Column name for origin identifier
    destination_id : str, default "dest_id"
        Column name for destination identifier
    row_standardize : bool, default True
        Whether to row-standardize the weights

    Returns
    -------
    W : libpysal.weights.W
        Spatial weights object

    Examples
    --------
    >>> # Build weights from trade flows
    >>> W = build_economic_weights(
    ...     trade_df, flow_variable="trade_value", origin_id="exporter", destination_id="importer"
    ... )
    """
    # Get unique IDs
    ids = sorted(set(data[origin_id].unique()) | set(data[destination_id].unique()))

    # Initialize neighbors and weights dictionaries
    neighbors = {id_: [] for id_ in ids}
    weights_dict = {id_: [] for id_ in ids}

    # Fill in from flow data
    for _, row in data.iterrows():
        origin = row[origin_id]
        dest = row[destination_id]
        flow = row[flow_variable]

        if origin != dest and flow > 0:
            neighbors[origin].append(dest)
            weights_dict[origin].append(flow)

    # Create weights object
    W = weights.W(neighbors, weights_dict, ids=ids)

    if row_standardize:
        W.transform = "r"

    return W


def normalize_weights(W, method: str = "r"):
    """
    Normalize spatial weights matrix.

    Parameters
    ----------
    W : libpysal.weights.W
        Spatial weights object
    method : str, default 'r'
        Normalization method:
        - 'r': Row-standardize (sum to 1 for each row)
        - 'b': Binary (0/1)
        - 'v': Variance-stabilizing
        - 'o': Original (no transformation)

    Returns
    -------
    W : libpysal.weights.W
        Normalized weights object

    Examples
    --------
    >>> W_normalized = normalize_weights(W, method="r")
    """
    W.transform = method
    return W


def compute_weights_properties(W) -> dict:
    """
    Compute properties of spatial weights matrix.

    Parameters
    ----------
    W : libpysal.weights.W
        Spatial weights object

    Returns
    -------
    dict
        Dictionary with weights properties:
        - n: Number of observations
        - mean_neighbors: Average number of neighbors
        - min_neighbors: Minimum number of neighbors
        - max_neighbors: Maximum number of neighbors
        - pct_nonzero: Percentage of nonzero weights
        - s0: Sum of all weights
        - s1: Sum of squared row and column sums
        - s2: Sum of squared weights

    Examples
    --------
    >>> props = compute_weights_properties(W)
    >>> print(f"Average neighbors: {props['mean_neighbors']:.2f}")
    """
    n = W.n
    cardinalities = W.cardinalities
    s0 = W.s0
    s1 = W.s1
    s2 = W.s2

    properties = {
        "n": n,
        "mean_neighbors": np.mean(list(cardinalities.values())),
        "min_neighbors": min(cardinalities.values()),
        "max_neighbors": max(cardinalities.values()),
        "pct_nonzero": W.pct_nonzero,
        "s0": s0,
        "s1": s1,
        "s2": s2,
        "transform": W.transform if hasattr(W, "transform") else None,
    }

    return properties


def get_neighbors_list(W, entity_id: str, k: int = 5) -> list:
    """
    Get list of neighbors for a specific entity.

    Parameters
    ----------
    W : libpysal.weights.W
        Spatial weights object
    entity_id : str or int
        Entity identifier
    k : int, default 5
        Number of neighbors to return (sorted by weight)

    Returns
    -------
    list of tuples
        List of (neighbor_id, weight) tuples

    Examples
    --------
    >>> neighbors = get_neighbors_list(W, entity_id="01001", k=10)
    >>> for neighbor, weight in neighbors[:5]:
    ...     print(f"{neighbor}: {weight:.4f}")
    """
    if entity_id not in W.neighbors:
        raise ValueError(f"Entity {entity_id} not found in weights matrix")

    # Get neighbors and weights
    neighbor_ids = W.neighbors[entity_id]
    neighbor_weights = W.weights[entity_id]

    # Combine and sort by weight (descending)
    neighbors_with_weights = list(zip(neighbor_ids, neighbor_weights))
    neighbors_with_weights.sort(key=lambda x: x[1], reverse=True)

    return neighbors_with_weights[:k]


def visualize_weights_structure(W, sample_size: int = 10):
    """
    Print summary visualization of weights structure.

    Parameters
    ----------
    W : libpysal.weights.W
        Spatial weights object
    sample_size : int, default 10
        Number of entities to display in sample

    Examples
    --------
    >>> visualize_weights_structure(W, sample_size=5)
    """
    props = compute_weights_properties(W)

    print("Spatial Weights Matrix Summary")
    print("=" * 50)
    print(f"Number of observations: {props['n']}")
    print(f"Average neighbors: {props['mean_neighbors']:.2f}")
    print(f"Min neighbors: {props['min_neighbors']}")
    print(f"Max neighbors: {props['max_neighbors']}")
    print(f"Percent nonzero: {props['pct_nonzero']:.2%}")
    print(f"Transformation: {props['transform']}")
    print()

    print(f"Sample of {sample_size} entities:")
    print("-" * 50)

    sample_ids = list(W.neighbors.keys())[:sample_size]
    for entity_id in sample_ids:
        n_neighbors = len(W.neighbors[entity_id])
        print(f"  {entity_id}: {n_neighbors} neighbors")


if __name__ == "__main__":
    # Example usage
    print("Spatial Weights Matrix Builder")
    print("=" * 50)
    print()
    print("Available functions:")
    print("  - build_contiguity_matrix() - Queen/Rook contiguity")
    print("  - build_distance_matrix() - Distance threshold")
    print("  - build_knn_matrix() - K-nearest neighbors")
    print("  - build_inverse_distance_matrix() - Inverse distance")
    print("  - build_economic_weights() - Economic flows")
    print("  - normalize_weights() - Row-standardization")
    print("  - compute_weights_properties() - W matrix properties")
    print("  - get_neighbors_list() - List neighbors of entity")
    print("  - visualize_weights_structure() - Print W summary")
    print()
    print("Import this module in notebooks to access these utilities.")
