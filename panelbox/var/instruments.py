"""
Instrument matrix construction for Panel VAR GMM estimation

This module implements the construction of GMM instrument matrices following
Holtz-Eakin, Newey & Rosen (1988) and Abrigo & Love (2016).

References:
- Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating vector autoregressions
  with panel data. Econometrica, 1371-1395.
- Abrigo, M. R., & Love, I. (2016). Estimation of panel vector autoregression in Stata.
  The Stata Journal, 16(3), 778-804.
- Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM
  in Stata. The Stata Journal, 9(1), 86-136.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class PanelVARInstruments:
    """
    Construct GMM instrument matrices for Panel VAR models.

    For a VAR(p) model with K variables:
    - Each equation k uses lags of ALL K variables as instruments
    - Instruments must be dated t-p-1 or earlier (predetermined)
    - Instrument matrix has block-diagonal structure (separate for each entity-time)

    Parameters
    ----------
    var_lags : int
        Number of lags in the VAR model (p)
    n_vars : int
        Number of variables in the VAR system (K)
    instrument_type : str, default 'all'
        Type of instrument construction:
        - 'all': Use all available lags (can lead to proliferation)
        - 'collapsed': Roodman (2009) collapsed instruments (reduces dimension)
    max_instruments : int, optional
        Maximum number of instrument lags to use per variable.
        If None, uses all available lags.
    """

    def __init__(
        self,
        var_lags: int,
        n_vars: int,
        instrument_type: str = "all",
        max_instruments: Optional[int] = None,
    ):
        self.var_lags = var_lags
        self.n_vars = n_vars
        self.instrument_type = instrument_type
        self.max_instruments = max_instruments

        if instrument_type not in ["all", "collapsed"]:
            raise ValueError("instrument_type must be 'all' or 'collapsed'")

    def construct_instruments(
        self,
        data: pd.DataFrame,
        entity_col: str = "entity",
        time_col: str = "time",
        value_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Construct GMM instrument matrix Z.

        For Panel VAR, the instrument matrix has the structure:
        - Rows: observations (entity-time pairs after transformation)
        - Columns: instruments (lags of all variables)
        - Block diagonal: each (entity, time) has its own instrument set

        Parameters
        ----------
        data : pd.DataFrame
            Original (untransformed) panel data
        entity_col : str
            Entity identifier column
        time_col : str
            Time identifier column
        value_cols : list, optional
            Variable names. If None, uses all numeric columns.

        Returns
        -------
        Z : np.ndarray
            Instrument matrix, shape (N*T_transformed, n_instruments)
        metadata : dict
            Information about instrument construction:
            - n_instruments_per_equation: list of instrument counts
            - total_instruments: total number of instruments
            - instrument_lag_ranges: dict mapping (entity, time) to valid lags
            - instruments_type: 'all' or 'collapsed'
        """
        # Identify value columns
        if value_cols is None:
            value_cols = [
                col
                for col in data.columns
                if col not in [entity_col, time_col] and pd.api.types.is_numeric_dtype(data[col])
            ]

        if len(value_cols) != self.n_vars:
            raise ValueError(f"Expected {self.n_vars} variables, got {len(value_cols)}")

        # Sort data
        df = data[[entity_col, time_col] + value_cols].copy()
        df = df.sort_values([entity_col, time_col]).reset_index(drop=True)

        # Build instrument matrix
        if self.instrument_type == "all":
            Z, meta = self._construct_all_instruments(df, entity_col, time_col, value_cols)
        else:  # collapsed
            Z, meta = self._construct_collapsed_instruments(df, entity_col, time_col, value_cols)

        # Check for instrument proliferation
        n_entities = df[entity_col].nunique()
        if meta["total_instruments"] > n_entities:
            warnings.warn(
                f"Number of instruments ({meta['total_instruments']}) exceeds number of entities "
                f"({n_entities}). Consider using instrument_type='collapsed' or setting max_instruments.",
                UserWarning,
            )

        return Z, meta

    def _construct_all_instruments(
        self, data: pd.DataFrame, entity_col: str, time_col: str, value_cols: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Construct instrument matrix using all available lags.

        For each observation (i, t), instruments are:
        - All lags dated t-p-1 or earlier: y_{i,t-p-1}, y_{i,t-p-2}, ..., y_{i,1}
        - For all K variables

        This can lead to O(T^2) instruments per variable, causing proliferation.

        Note: All rows are padded to the same length (max possible lags) for consistent matrix shape.
        """
        # First pass: determine maximum lag depth across all observations
        max_lag_depth_global = 0
        for entity_id, group in data.groupby(entity_col):
            times = sorted(group[time_col].values)
            for t_idx, t in enumerate(times):
                min_valid_lag = t - self.var_lags - 1
                available_lags = [lag_t for lag_t in times if lag_t <= min_valid_lag]
                if available_lags:
                    max_lag_depth_global = max(max_lag_depth_global, len(available_lags))

        # Apply max_instruments constraint
        if self.max_instruments is not None:
            max_lag_depth_global = min(max_lag_depth_global, self.max_instruments)

        if max_lag_depth_global == 0:
            raise ValueError("No valid instruments constructed. Check that T > p + 1.")

        # Total instruments per row: max_lag_depth × n_vars
        n_instruments_total = max_lag_depth_global * self.n_vars

        # Second pass: construct instrument matrix with consistent shape
        instrument_rows = []
        instrument_metadata = []

        for entity_id, group in data.groupby(entity_col):
            times = sorted(group[time_col].values)
            var_data_dict = {var: group.set_index(time_col)[var] for var in value_cols}

            for t_idx, t in enumerate(times):
                # Minimum valid lag for instruments: t - p - 1
                min_valid_lag = t - self.var_lags - 1

                # Available lags: all periods before t in this entity
                available_lags = [lag_t for lag_t in times if lag_t <= min_valid_lag]

                if not available_lags:
                    continue

                # Keep most recent lags up to max_lag_depth_global
                if len(available_lags) > max_lag_depth_global:
                    available_lags = sorted(available_lags)[-max_lag_depth_global:]

                # Construct instrument row with padding
                instrument_row = np.zeros(n_instruments_total)

                for var_idx, var_name in enumerate(value_cols):
                    var_data = var_data_dict[var_name]

                    # Fill in available lag values
                    for lag_idx, lag_t in enumerate(available_lags):
                        col_idx = var_idx * max_lag_depth_global + lag_idx
                        if lag_t in var_data.index:
                            instrument_row[col_idx] = var_data.loc[lag_t]

                instrument_rows.append(instrument_row)
                instrument_metadata.append(
                    {
                        entity_col: entity_id,
                        time_col: t,
                        "n_lags": len(available_lags),
                        "lag_range": (
                            (min(available_lags), max(available_lags)) if available_lags else None
                        ),
                    }
                )

        # Convert to numpy array
        Z = np.array(instrument_rows)

        # Compute metadata
        metadata = {
            "total_instruments": Z.shape[1],
            "n_instruments_per_variable": max_lag_depth_global,
            "n_instruments_per_equation": [Z.shape[1]]
            * self.n_vars,  # Same for all equations in VAR
            "instruments_type": "all",
            "max_instruments": self.max_instruments,
            "observation_metadata": instrument_metadata,
        }

        return Z, metadata

    def _construct_collapsed_instruments(
        self, data: pd.DataFrame, entity_col: str, time_col: str, value_cols: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Construct collapsed instrument matrix (Roodman 2009).

        Collapsed instruments reduce dimensionality by creating one column per lag depth,
        rather than one column per (entity, time, lag) combination.

        For each lag depth d (d = p+1, p+2, ..., T):
        - Create single column using all entities at lag d
        - Reduces instruments from O(NT) to O(T)

        This dramatically reduces instrument count while maintaining moment conditions.
        """
        instrument_rows = []
        instrument_metadata = []

        # Determine maximum lag depth across all entities
        max_lag_depth = 0
        for entity_id, group in data.groupby(entity_col):
            times = sorted(group[time_col].values)
            if times:
                max_lag_depth = max(max_lag_depth, len(times))

        # Determine lag depths to use
        min_lag_depth = self.var_lags + 1  # Minimum: t - (p+1)
        lag_depths = list(range(min_lag_depth, max_lag_depth + 1))

        if self.max_instruments is not None:
            lag_depths = lag_depths[: self.max_instruments]

        # For each observation, create collapsed instrument row
        for entity_id, group in data.groupby(entity_col):
            times = sorted(group[time_col].values)

            for t_idx, t in enumerate(times):
                instrument_row = []

                # For each variable
                for var_name in value_cols:
                    var_data = group.set_index(time_col)[var_name]

                    # For each lag depth
                    for lag_depth in lag_depths:
                        lag_t = t - lag_depth
                        # Use lag value if available, else 0 (missing)
                        lag_value = var_data.loc[lag_t] if lag_t in var_data.index else 0.0
                        instrument_row.append(lag_value)

                # Skip if all zeros (no valid instruments)
                if not any(instrument_row):
                    continue

                instrument_rows.append(instrument_row)
                instrument_metadata.append(
                    {
                        entity_col: entity_id,
                        time_col: t,
                        "n_lags": len(lag_depths),
                        "lag_depths": lag_depths,
                    }
                )

        if not instrument_rows:
            raise ValueError("No valid instruments constructed. Check that T > p + 1.")

        Z = np.array(instrument_rows)

        # Compute metadata
        metadata = {
            "total_instruments": Z.shape[1],
            "n_instruments_per_variable": len(lag_depths),
            "n_instruments_per_equation": [Z.shape[1]] * self.n_vars,
            "instruments_type": "collapsed",
            "max_instruments": self.max_instruments,
            "lag_depths": lag_depths,
            "observation_metadata": instrument_metadata,
        }

        return Z, metadata

    def get_instrument_count_summary(self, metadata: Dict) -> str:
        """
        Generate human-readable summary of instrument construction.

        Parameters
        ----------
        metadata : dict
            Metadata from construct_instruments()

        Returns
        -------
        str
            Formatted summary
        """
        summary = []
        summary.append("=" * 60)
        summary.append("GMM Instrument Matrix Summary")
        summary.append("=" * 60)
        summary.append(f"Instrument type:              {metadata['instruments_type']}")
        summary.append(f"Total instruments:            {metadata['total_instruments']}")
        summary.append(f"Instruments per variable:     {metadata['n_instruments_per_variable']}")
        summary.append(f"Number of variables:          {self.n_vars}")

        if metadata.get("max_instruments"):
            summary.append(f"Max instruments limit:        {metadata['max_instruments']}")

        if metadata["instruments_type"] == "collapsed":
            summary.append(f"Lag depths used:              {len(metadata.get('lag_depths', []))}")

        summary.append("=" * 60)

        return "\n".join(summary)


def build_gmm_instruments(
    data: pd.DataFrame,
    var_lags: int,
    n_vars: int,
    entity_col: str = "entity",
    time_col: str = "time",
    value_cols: Optional[List[str]] = None,
    instrument_type: str = "all",
    max_instruments: Optional[int] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Convenience function to build GMM instrument matrix.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    var_lags : int
        Number of VAR lags (p)
    n_vars : int
        Number of variables (K)
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
    value_cols : list, optional
        Variable names
    instrument_type : str
        'all' or 'collapsed'
    max_instruments : int, optional
        Maximum instrument lags per variable

    Returns
    -------
    Z : np.ndarray
        Instrument matrix
    metadata : dict
        Instrument construction metadata

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'entity': [1, 1, 1, 1, 2, 2, 2, 2],
    ...     'time': [1, 2, 3, 4, 1, 2, 3, 4],
    ...     'y1': [1, 2, 3, 4, 5, 6, 7, 8],
    ...     'y2': [10, 20, 30, 40, 50, 60, 70, 80]
    ... })
    >>> Z, meta = build_gmm_instruments(df, var_lags=1, n_vars=2)
    >>> print(meta['total_instruments'])
    """
    builder = PanelVARInstruments(
        var_lags=var_lags,
        n_vars=n_vars,
        instrument_type=instrument_type,
        max_instruments=max_instruments,
    )

    Z, metadata = builder.construct_instruments(
        data=data, entity_col=entity_col, time_col=time_col, value_cols=value_cols
    )

    return Z, metadata
