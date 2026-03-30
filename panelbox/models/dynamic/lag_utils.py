"""
Utility functions for building lag matrices and first differences in panel data.

These functions respect the panel structure, ensuring no cross-entity contamination
when computing lags or differences.
"""

from __future__ import annotations

import pandas as pd


def build_lag_matrix(
    data: pd.DataFrame,
    var: str,
    entity_col: str,
    time_col: str,
    lags: int = 1,
) -> pd.Series:
    """
    Build lagged variable by entity (no cross-entity contamination).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data sorted by entity and time.
    var : str
        Variable name to lag.
    entity_col : str
        Column identifying entities.
    time_col : str
        Column identifying time periods.
    lags : int, default=1
        Number of lags.

    Returns
    -------
    pd.Series
        Lagged variable with NaN where lag is unavailable.
    """
    df = data.sort_values([entity_col, time_col])
    return df.groupby(entity_col)[var].shift(lags)


def build_first_difference(
    data: pd.DataFrame,
    vars: list[str],
    entity_col: str,
    time_col: str,
) -> pd.DataFrame:
    """
    First difference by entity.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data sorted by entity and time.
    vars : list of str
        Variables to difference.
    entity_col : str
        Column identifying entities.
    time_col : str
        Column identifying time periods.

    Returns
    -------
    pd.DataFrame
        First-differenced variables with NaN for the first period of each entity.
    """
    df = data.sort_values([entity_col, time_col])
    return df.groupby(entity_col)[vars].diff()


def build_dynamic_design(
    data: pd.DataFrame,
    dep_var: str,
    entity_col: str,
    time_col: str,
    lags: int = 1,
    exog_vars: list[str] | None = None,
    instrument: str = "levels",
) -> dict:
    """
    Build design matrices for dynamic panel estimation via first differences.

    Constructs:
    - Dy: first-differenced dependent variable
    - Dy_lag: first-differenced lagged dependent variable
    - DX: first-differenced exogenous regressors (if any)
    - Z: instruments (y_{t-2} in levels or Dy_{t-2} in differences)

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    dep_var : str
        Dependent variable name.
    entity_col : str
        Column identifying entities.
    time_col : str
        Column identifying time periods.
    lags : int, default=1
        Number of AR lags (currently only lags=1 supported).
    exog_vars : list of str, optional
        Exogenous regressor names.
    instrument : str, default='levels'
        'levels' uses y_{t-2} as instrument, 'differences' uses Dy_{t-2}.

    Returns
    -------
    dict
        Keys: 'Dy', 'Dy_lag', 'DX', 'Z', 'entities', 'times', 'valid_mask'
        All arrays have the same number of rows (valid observations).
    """
    if lags != 1:
        raise NotImplementedError("Only lags=1 is currently supported.")

    if instrument not in ("levels", "differences"):
        raise ValueError(f"instrument must be 'levels' or 'differences', got '{instrument}'")

    df = data.sort_values([entity_col, time_col]).copy()

    # Build lags in levels
    df["_y_lag1"] = df.groupby(entity_col)[dep_var].shift(1)
    df["_y_lag2"] = df.groupby(entity_col)[dep_var].shift(2)

    # First differences
    df["_Dy"] = df.groupby(entity_col)[dep_var].diff()
    df["_Dy_lag1"] = df.groupby(entity_col)["_Dy"].shift(1)

    # Instrument
    if instrument == "levels":
        # y_{t-2} as instrument for Dy_{t-1}
        df["_Z"] = df["_y_lag2"]
    else:
        # Dy_{t-2} as instrument
        df["_Dy_lag2"] = df.groupby(entity_col)["_Dy"].shift(2)
        df["_Z"] = df["_Dy_lag2"]

    # First-difference exogenous variables
    dx_cols = []
    if exog_vars is not None:
        for v in exog_vars:
            col = f"_D{v}"
            df[col] = df.groupby(entity_col)[v].diff()
            dx_cols.append(col)

    # Valid observations: need Dy, Dy_lag1, Z, and all DX to be non-NaN
    required = ["_Dy", "_Dy_lag1", "_Z", *dx_cols]
    valid = df[required].notna().all(axis=1)

    result = {
        "Dy": df.loc[valid, "_Dy"].values,
        "Dy_lag": df.loc[valid, "_Dy_lag1"].values,
        "Z": df.loc[valid, "_Z"].values,
        "entities": df.loc[valid, entity_col].values,
        "times": df.loc[valid, time_col].values,
        "valid_mask": valid.values,
    }

    if dx_cols:
        result["DX"] = df.loc[valid, dx_cols].values
    else:
        result["DX"] = None

    return result
