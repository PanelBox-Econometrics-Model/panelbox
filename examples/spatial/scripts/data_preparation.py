"""
Data Preparation Utilities for Spatial Econometrics Tutorials

This module provides functions to load, clean, and prepare spatial panel datasets
for analysis in PanelBox.

Functions:
    load_spatial_dataset: Load spatial data from CSV and shapefile
    merge_data_shapefile: Merge panel data with geographic boundaries
    balance_panel: Create balanced panel from unbalanced data
    handle_missing_values: Impute or remove missing values
    add_time_effects: Add time fixed effects dummies
    compute_spatial_lags: Compute spatial lags of variables
"""

from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd


def load_spatial_dataset(
    csv_path: Union[str, Path],
    shapefile_path: Optional[Union[str, Path]] = None,
    entity_id: str = "entity_id",
    time_id: str = "year",
) -> Union[pd.DataFrame, tuple[pd.DataFrame, gpd.GeoDataFrame]]:
    """
    Load spatial panel dataset from CSV and optionally merge with shapefile.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file with panel data
    shapefile_path : str or Path, optional
        Path to shapefile with geographic boundaries
    entity_id : str, default "entity_id"
        Column name for entity identifier
    time_id : str, default "year"
        Column name for time identifier

    Returns
    -------
    pd.DataFrame or tuple
        If shapefile_path is None, returns DataFrame
        Otherwise returns (DataFrame, GeoDataFrame)

    Examples
    --------
    >>> df = load_spatial_dataset("data/us_counties/us_counties.csv")
    >>> df, gdf = load_spatial_dataset(
    ...     "data/us_counties/us_counties.csv", "data/us_counties/us_counties.shp"
    ... )
    """
    # Load CSV data
    df = pd.read_csv(csv_path)

    # Ensure identifiers are proper types
    if entity_id in df.columns:
        df[entity_id] = df[entity_id].astype(str)
    if time_id in df.columns:
        df[time_id] = pd.to_numeric(df[time_id])

    # Sort by entity and time
    df = df.sort_values([entity_id, time_id]).reset_index(drop=True)

    if shapefile_path is None:
        return df
    else:
        # Load shapefile
        gdf = gpd.read_file(shapefile_path)
        return df, gdf


def merge_data_shapefile(
    df: pd.DataFrame, gdf: gpd.GeoDataFrame, entity_id: str = "entity_id", how: str = "inner"
) -> gpd.GeoDataFrame:
    """
    Merge panel data with shapefile geometry.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    gdf : gpd.GeoDataFrame
        Geographic data with boundaries
    entity_id : str, default "entity_id"
        Column name for entity identifier (must exist in both dataframes)
    how : str, default "inner"
        Type of merge ('inner', 'left', 'right', 'outer')

    Returns
    -------
    gpd.GeoDataFrame
        Merged geodataframe with panel data and geometry

    Examples
    --------
    >>> gdf_merged = merge_data_shapefile(df, gdf, entity_id="county_fips")
    """
    # Ensure entity_id is same type in both
    df[entity_id] = df[entity_id].astype(str)
    gdf[entity_id] = gdf[entity_id].astype(str)

    # Merge
    gdf_merged = gdf.merge(df, on=entity_id, how=how)

    return gdf_merged


def balance_panel(
    df: pd.DataFrame, entity_id: str = "entity_id", time_id: str = "year", method: str = "complete"
) -> pd.DataFrame:
    """
    Create balanced panel from potentially unbalanced data.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data (may be unbalanced)
    entity_id : str, default "entity_id"
        Column name for entity identifier
    time_id : str, default "year"
        Column name for time identifier
    method : str, default "complete"
        Method for balancing:
        - 'complete': Keep only entities observed in all periods
        - 'fill': Fill missing periods with NaN

    Returns
    -------
    pd.DataFrame
        Balanced panel

    Examples
    --------
    >>> df_balanced = balance_panel(df, entity_id="county_fips", method="complete")
    """
    if method == "complete":
        # Count observations per entity
        counts = df.groupby(entity_id)[time_id].count()
        max_periods = counts.max()

        # Keep only entities with all periods
        complete_entities = counts[counts == max_periods].index
        df_balanced = df[df[entity_id].isin(complete_entities)].copy()

    elif method == "fill":
        # Create complete index
        entities = df[entity_id].unique()
        times = df[time_id].unique()

        # Create all combinations
        index = pd.MultiIndex.from_product([entities, times], names=[entity_id, time_id])

        # Reindex and fill
        df_balanced = df.set_index([entity_id, time_id]).reindex(index).reset_index()

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_balanced.sort_values([entity_id, time_id]).reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    variables: list[str],
    method: str = "drop",
    entity_id: str = "entity_id",
    time_id: str = "year",
) -> pd.DataFrame:
    """
    Handle missing values in panel data.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    variables : list of str
        Variables to check for missing values
    method : str, default "drop"
        Method for handling missing values:
        - 'drop': Drop observations with missing values
        - 'forward_fill': Forward fill within entities
        - 'interpolate': Linear interpolation within entities
        - 'mean': Fill with variable mean
    entity_id : str, default "entity_id"
        Column name for entity identifier
    time_id : str, default "year"
        Column name for time identifier

    Returns
    -------
    pd.DataFrame
        Data with missing values handled

    Examples
    --------
    >>> df_clean = handle_missing_values(
    ...     df, variables=["income", "population"], method="forward_fill"
    ... )
    """
    df_copy = df.copy()

    if method == "drop":
        df_copy = df_copy.dropna(subset=variables)

    elif method == "forward_fill":
        df_copy = df_copy.sort_values([entity_id, time_id])
        df_copy[variables] = df_copy.groupby(entity_id)[variables].ffill()

    elif method == "interpolate":
        df_copy = df_copy.sort_values([entity_id, time_id])
        df_copy[variables] = df_copy.groupby(entity_id)[variables].apply(
            lambda x: x.interpolate(method="linear")
        )

    elif method == "mean":
        for var in variables:
            mean_val = df_copy[var].mean()
            df_copy[var] = df_copy[var].fillna(mean_val)

    else:
        raise ValueError(f"Unknown method: {method}")

    return df_copy


def add_time_effects(
    df: pd.DataFrame, time_id: str = "year", drop_first: bool = True
) -> pd.DataFrame:
    """
    Add time fixed effects dummy variables.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    time_id : str, default "year"
        Column name for time identifier
    drop_first : bool, default True
        Drop first category to avoid collinearity

    Returns
    -------
    pd.DataFrame
        Data with time dummy variables added

    Examples
    --------
    >>> df_with_dummies = add_time_effects(df, time_id="year")
    """
    df_copy = df.copy()

    # Create dummies
    time_dummies = pd.get_dummies(df_copy[time_id], prefix="year", drop_first=drop_first)

    # Concatenate
    df_copy = pd.concat([df_copy, time_dummies], axis=1)

    return df_copy


def compute_spatial_lags(
    df: pd.DataFrame,
    variables: list[str],
    W,
    entity_id: str = "entity_id",
    time_id: str = "year",
    prefix: str = "W_",
) -> pd.DataFrame:
    """
    Compute spatial lags of variables.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    variables : list of str
        Variables to compute spatial lags for
    W : spatial weights matrix
        Spatial weights matrix (from libpysal)
    entity_id : str, default "entity_id"
        Column name for entity identifier
    time_id : str, default "year"
        Column name for time identifier
    prefix : str, default "W_"
        Prefix for spatial lag variable names

    Returns
    -------
    pd.DataFrame
        Data with spatial lag variables added

    Examples
    --------
    >>> from libpysal import weights
    >>> w = weights.Queen.from_shapefile("data/us_counties/us_counties.shp")
    >>> df_with_lags = compute_spatial_lags(df, variables=["income", "population"], W=w)
    """
    from libpysal.weights import lag_spatial

    df_copy = df.copy()

    # Get unique time periods
    time_periods = df_copy[time_id].unique()

    for var in variables:
        lag_values = []

        for t in time_periods:
            # Subset to time period
            df_t = df_copy[df_copy[time_id] == t].sort_values(entity_id)

            # Compute spatial lag
            values = df_t[var].values
            lag = lag_spatial(W, values)

            lag_values.extend(lag)

        # Add to dataframe
        df_copy[f"{prefix}{var}"] = lag_values

    return df_copy


def create_distance_matrix(gdf: gpd.GeoDataFrame, entity_id: str = "entity_id") -> pd.DataFrame:
    """
    Create distance matrix from GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with entity geometries
    entity_id : str, default "entity_id"
        Column name for entity identifier

    Returns
    -------
    pd.DataFrame
        Distance matrix with entity IDs as index and columns

    Examples
    --------
    >>> dist_matrix = create_distance_matrix(gdf, entity_id="county_fips")
    """
    # Get centroids
    gdf_copy = gdf.copy()
    gdf_copy["centroid"] = gdf_copy.geometry.centroid

    # Compute pairwise distances
    n = len(gdf_copy)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i, j] = gdf_copy.iloc[i]["centroid"].distance(
                    gdf_copy.iloc[j]["centroid"]
                )

    # Create DataFrame
    entity_ids = gdf_copy[entity_id].values
    dist_df = pd.DataFrame(dist_matrix, index=entity_ids, columns=entity_ids)

    return dist_df


if __name__ == "__main__":
    # Example usage
    print("Spatial Data Preparation Utilities")
    print("===================================")
    print()
    print("Available functions:")
    print("  - load_spatial_dataset()")
    print("  - merge_data_shapefile()")
    print("  - balance_panel()")
    print("  - handle_missing_values()")
    print("  - add_time_effects()")
    print("  - compute_spatial_lags()")
    print("  - create_distance_matrix()")
    print()
    print("Import this module in notebooks to access these utilities.")
