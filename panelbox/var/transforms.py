"""
Panel VAR Data Transformations for GMM Estimation

This module implements transformation methods for removing fixed effects in Panel VAR models,
specifically designed for GMM estimation following Arellano & Bover (1995) and Abrigo & Love (2016).

Transformations:
- Forward Orthogonal Deviations (FOD): Arellano & Bover (1995)
- First-Differences (FD): Anderson & Hsiao (1981)

References:
- Arellano, M., & Bover, O. (1995). Another look at the instrumental variable estimation
  of error-components models. Journal of econometrics, 68(1), 29-51.
- Abrigo, M. R., & Love, I. (2016). Estimation of panel vector autoregression in Stata.
  The Stata Journal, 16(3), 778-804.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def forward_orthogonal_deviation(
    data: pd.DataFrame,
    entity_col: str = "entity",
    time_col: str = "time",
    value_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Forward Orthogonal Deviations (FOD) transformation to panel data.

    FOD removes entity-specific fixed effects by subtracting the forward mean from each observation,
    weighted by a normalization factor that preserves the variance structure.

    The transformation for variable y at entity i, time t is:

        y*ᵢₜ = cₜ × (yᵢₜ - ȳᵢ,>t)

    where:
        ȳᵢ,>t = (1/(Tᵢ - t)) × Σₛ₌ₜ₊₁^Tᵢ yᵢₛ  (forward mean)
        cₜ = √((Tᵢ - t)/(Tᵢ - t + 1))        (normalization factor)
        Tᵢ = number of periods for entity i

    Advantages over First-Differences:
    - Preserves orthogonality of transformed errors
    - Loses fewer observations in unbalanced panels
    - Allows using all available lags as instruments

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with entity and time identifiers
    entity_col : str, default 'entity'
        Column name for entity identifier
    time_col : str, default 'time'
        Column name for time identifier
    value_cols : list, optional
        List of column names to transform. If None, transforms all numeric columns
        except entity_col and time_col.

    Returns
    -------
    transformed : pd.DataFrame
        FOD-transformed data. Last period for each entity is dropped (no future mean available).
    meta : pd.DataFrame
        Metadata with columns: entity, time, normalization_factor, periods_ahead

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'entity': [1, 1, 1, 2, 2, 2],
    ...     'time': [1, 2, 3, 1, 2, 3],
    ...     'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ... })
    >>> transformed, meta = forward_orthogonal_deviation(df)
    >>> print(transformed.shape[0])  # Lost last period per entity
    4

    Notes
    -----
    - For unbalanced panels, Tᵢ varies by entity
    - The last observation for each entity is dropped (no future observations)
    - Missing periods within an entity are handled correctly
    - Transformation preserves the variance structure of idiosyncratic errors
    """
    # Validate inputs
    if entity_col not in data.columns:
        raise ValueError(f"Entity column '{entity_col}' not found in data")
    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")

    # Identify value columns
    if value_cols is None:
        value_cols = [
            col
            for col in data.columns
            if col not in [entity_col, time_col] and pd.api.types.is_numeric_dtype(data[col])
        ]

    if not value_cols:
        raise ValueError("No numeric columns to transform")

    # Sort data by entity and time
    df = data[[entity_col, time_col] + value_cols].copy()
    df = df.sort_values([entity_col, time_col]).reset_index(drop=True)

    # Compute entity-specific time series length
    entity_periods = df.groupby(entity_col)[time_col].transform("count")

    # Compute time index within each entity (0-indexed)
    time_idx = df.groupby(entity_col).cumcount()

    # Compute periods ahead for each observation (Tᵢ - t)
    periods_ahead = entity_periods - time_idx - 1

    # Prepare containers
    transformed_data = []
    meta_data = []

    # Process each entity separately
    for entity_id, group in df.groupby(entity_col):
        T_i = len(group)

        # Skip entities with only one period
        if T_i <= 1:
            continue

        # Process each time period (skip last one - no future mean)
        for t_idx in range(T_i - 1):
            t_obs = group.iloc[t_idx]

            # Compute forward mean for each variable
            future_obs = group.iloc[t_idx + 1 :]
            forward_mean = future_obs[value_cols].mean()

            # Normalization factor: sqrt((T_i - t) / (T_i - t + 1))
            periods_remaining = T_i - t_idx - 1  # Number of future periods
            c_t = np.sqrt(periods_remaining / (periods_remaining + 1))

            # FOD transformation: c_t * (y_it - forward_mean)
            transformed_vals = c_t * (t_obs[value_cols] - forward_mean)

            # Store transformed observation
            transformed_obs = {entity_col: entity_id, time_col: t_obs[time_col]}
            transformed_obs.update(transformed_vals.to_dict())
            transformed_data.append(transformed_obs)

            # Store metadata
            meta_data.append(
                {
                    entity_col: entity_id,
                    time_col: t_obs[time_col],
                    "normalization_factor": c_t,
                    "periods_ahead": periods_remaining,
                }
            )

    # Convert to DataFrames
    transformed_df = pd.DataFrame(transformed_data)
    meta_df = pd.DataFrame(meta_data)

    # Restore original column order
    col_order = [entity_col, time_col] + value_cols
    transformed_df = transformed_df[col_order]

    return transformed_df, meta_df


def first_difference(
    data: pd.DataFrame,
    entity_col: str = "entity",
    time_col: str = "time",
    value_cols: Optional[list] = None,
) -> pd.DataFrame:
    """
    Apply First-Differences (FD) transformation to panel data.

    FD removes entity-specific fixed effects by taking first differences:

        Δyᵢₜ = yᵢₜ - yᵢ,ₜ₋₁

    This is the classic Anderson-Hsiao transformation for dynamic panels.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with entity and time identifiers
    entity_col : str, default 'entity'
        Column name for entity identifier
    time_col : str, default 'time'
        Column name for time identifier
    value_cols : list, optional
        List of column names to transform. If None, transforms all numeric columns
        except entity_col and time_col.

    Returns
    -------
    pd.DataFrame
        First-differenced data. First period for each entity is dropped.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'entity': [1, 1, 1, 2, 2, 2],
    ...     'time': [1, 2, 3, 1, 2, 3],
    ...     'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    ... })
    >>> fd = first_difference(df)
    >>> print(fd['y'].iloc[0])  # 2.0 - 1.0 = 1.0
    1.0

    Notes
    -----
    - First observation for each entity is dropped (no lagged value)
    - For unbalanced panels, handles missing periods correctly
    - Assumes data is sorted by entity and time
    """
    # Validate inputs
    if entity_col not in data.columns:
        raise ValueError(f"Entity column '{entity_col}' not found in data")
    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")

    # Identify value columns
    if value_cols is None:
        value_cols = [
            col
            for col in data.columns
            if col not in [entity_col, time_col] and pd.api.types.is_numeric_dtype(data[col])
        ]

    if not value_cols:
        raise ValueError("No numeric columns to transform")

    # Sort data by entity and time
    df = data[[entity_col, time_col] + value_cols].copy()
    df = df.sort_values([entity_col, time_col]).reset_index(drop=True)

    # Compute first differences within each entity
    transformed_data = []

    for entity_id, group in df.groupby(entity_col):
        if len(group) <= 1:
            continue

        # Compute differences for all value columns
        for t_idx in range(1, len(group)):
            current_obs = group.iloc[t_idx]
            lagged_obs = group.iloc[t_idx - 1]

            # First difference
            diff_vals = current_obs[value_cols] - lagged_obs[value_cols]

            # Store transformed observation
            transformed_obs = {entity_col: entity_id, time_col: current_obs[time_col]}
            transformed_obs.update(diff_vals.to_dict())
            transformed_data.append(transformed_obs)

    # Convert to DataFrame
    transformed_df = pd.DataFrame(transformed_data)

    # Restore original column order
    col_order = [entity_col, time_col] + value_cols
    transformed_df = transformed_df[col_order]

    return transformed_df


def get_valid_instrument_lags(
    data: pd.DataFrame, entity_col: str, time_col: str, transform: str, var_lags: int
) -> pd.DataFrame:
    """
    Determine valid instrument lags for each observation after transformation.

    For GMM estimation, instruments must be:
    - Predetermined (dated t-p-1 or earlier for VAR(p))
    - Available (entity i has observations at those periods)

    Parameters
    ----------
    data : pd.DataFrame
        Original (untransformed) panel data
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
    transform : str
        Transformation type: 'fod' or 'fd'
    var_lags : int
        Number of lags in the VAR model (p)

    Returns
    -------
    pd.DataFrame
        Metadata with columns: entity, time, min_valid_lag, max_valid_lag, n_valid_lags

    Notes
    -----
    For VAR(p) in FOD/FD:
    - Instruments must be dated t-p-1 or earlier
    - For FOD: uses original time periods (not transformed)
    - For FD: uses original time periods (not differenced)
    """
    df = data[[entity_col, time_col]].copy()
    df = df.sort_values([entity_col, time_col])

    meta_data = []

    for entity_id, group in df.groupby(entity_col):
        times = sorted(group[time_col].values)

        for t in times:
            # Minimum valid lag: t - p - 1
            min_lag = t - var_lags - 1

            # Maximum valid lag: earliest period available for this entity
            max_lag = times[0]

            # Valid lags are: [max_lag, min_lag] ∩ available_times
            valid_lags = [lag for lag in times if lag <= min_lag]

            meta_data.append(
                {
                    entity_col: entity_id,
                    time_col: t,
                    "min_valid_lag": min_lag,
                    "max_valid_lag": max_lag if valid_lags else None,
                    "n_valid_lags": len(valid_lags),
                }
            )

    return pd.DataFrame(meta_data)
