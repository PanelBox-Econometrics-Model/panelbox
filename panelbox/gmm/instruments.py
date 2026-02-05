"""
Instrument Generation for GMM
==============================

Tools for generating and managing instrument matrices for GMM estimation.

Classes
-------
InstrumentSet : Container for instrument matrices
InstrumentBuilder : Generates instrument matrices following xtabond2 rules

References
----------
.. [1] Roodman, D. (2009). "How to do xtabond2: An Introduction to Difference
       and System GMM in Stata." Stata Journal, 9(1), 86-136.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, cast

import numpy as np
import pandas as pd


class InstrumentStyle(Enum):
    """Style of instrument generation."""

    IV = "iv"  # IV-style: one column per lag
    GMM = "gmm"  # GMM-style: separate columns per time period


class EquationType(Enum):
    """Type of equation for instruments."""

    DIFF = "diff"  # Differenced equation
    LEVEL = "level"  # Level equation


@dataclass
class InstrumentSet:
    """
    Container for instrument matrices.

    Attributes
    ----------
    Z : np.ndarray
        Instrument matrix (T*N x n_instruments)
    variable_names : List[str]
        Names of instrumented variables
    instrument_names : List[str]
        Names of instrument columns
    equation : str
        Equation type ('diff' or 'level')
    style : str
        Instrument style ('iv' or 'gmm')
    collapsed : bool
        Whether instruments are collapsed
    """

    Z: np.ndarray
    variable_names: List[str] = field(default_factory=list)
    instrument_names: List[str] = field(default_factory=list)
    equation: str = "diff"
    style: str = "gmm"
    collapsed: bool = False

    @property
    def n_instruments(self) -> int:
        """Number of instruments."""
        return self.Z.shape[1] if self.Z is not None else 0

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self.Z.shape[0] if self.Z is not None else 0

    def __repr__(self) -> str:
        return (
            f"InstrumentSet(n_instruments={self.n_instruments}, "
            f"n_obs={self.n_obs}, equation='{self.equation}', "
            f"style='{self.style}', collapsed={self.collapsed})"
        )


class InstrumentBuilder:
    """
    Generates instrument matrices for GMM estimation.

    Follows Stata xtabond2 instrument generation rules.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    id_var : str
        Name of cross-sectional identifier
    time_var : str
        Name of time variable

    Attributes
    ----------
    data : pd.DataFrame
        Panel data
    id_var : str
        Cross-sectional identifier
    time_var : str
        Time variable
    n_groups : int
        Number of cross-sectional units
    time_periods : np.ndarray
        Unique time periods (sorted)

    Examples
    --------
    >>> builder = InstrumentBuilder(data, id_var='id', time_var='year')
    >>> # IV-style instruments
    >>> Z_iv = builder.create_iv_style_instruments('x', min_lag=2, max_lag=4)
    >>> # GMM-style instruments
    >>> Z_gmm = builder.create_gmm_style_instruments('y', min_lag=2, max_lag=99)
    >>> # Collapsed GMM-style
    >>> Z_collapsed = builder.create_gmm_style_instruments(
    ...     'y', min_lag=2, max_lag=99, collapse=True
    ... )
    """

    def __init__(self, data: pd.DataFrame, id_var: str, time_var: str):
        """Initialize instrument builder."""
        self.data = data.copy()
        self.id_var = id_var
        self.time_var = time_var

        # Ensure data is sorted
        self.data = self.data.sort_values([id_var, time_var])

        # Extract groups and time periods
        self.groups = self.data[id_var].unique()
        self.n_groups = len(self.groups)
        self.time_periods = np.sort(self.data[time_var].unique())
        self.n_periods = len(self.time_periods)

    def create_iv_style_instruments(
        self, var: str, min_lag: int, max_lag: int, equation: str = "diff"
    ) -> InstrumentSet:
        """
        Create IV-style instruments (one column per lag).

        IV-style instruments create one column for each lag, with observations
        placed appropriately for each time period.

        Parameters
        ----------
        var : str
            Variable to instrument
        min_lag : int
            Minimum lag to use (e.g., 2 means t-2)
        max_lag : int
            Maximum lag to use (e.g., 4 means t-4)
        equation : str
            'diff' for differenced equation, 'level' for level equation

        Returns
        -------
        InstrumentSet
            IV-style instrument set

        Examples
        --------
        >>> # gmm(x, lag(2 4)) in IV-style creates 3 columns: x_{t-2}, x_{t-3}, x_{t-4}
        >>> Z = builder.create_iv_style_instruments('x', min_lag=2, max_lag=4)

        Notes
        -----
        For equation='diff', instruments levels: x_{i,t-k}
        For equation='level', instruments differences: Δx_{i,t-k}
        """
        n_lags = max_lag - min_lag + 1
        n_obs = len(self.data)

        # Initialize instrument matrix
        Z = np.full((n_obs, n_lags), np.nan)

        # Get variable data
        var_data = self.data[var].values
        ids = self.data[self.id_var].values
        times = self.data[self.time_var].values

        # Create instrument names
        instrument_names = []
        for lag in range(min_lag, max_lag + 1):
            if equation == "diff":
                instrument_names.append(f"{var}_L{lag}")
            else:
                instrument_names.append(f"D.{var}_L{lag}")

        # Fill instrument matrix
        for i, (current_id, current_time) in enumerate(zip(ids, times)):
            for lag_idx, lag in enumerate(range(min_lag, max_lag + 1)):
                # Find lagged value
                mask = (ids == current_id) & (times == current_time - lag)
                if np.any(mask):
                    lag_idx_data = np.where(mask)[0][0]
                    if equation == "diff":
                        Z[i, lag_idx] = var_data[lag_idx_data]
                    else:
                        # For level equation, use differences as instruments
                        # Find t-lag-1 for differencing
                        mask_lag1 = (ids == current_id) & (times == current_time - lag - 1)
                        if np.any(mask_lag1):
                            lag1_idx_data = np.where(mask_lag1)[0][0]
                            Z[i, lag_idx] = var_data[lag_idx_data] - var_data[lag1_idx_data]

        return InstrumentSet(
            Z=Z,
            variable_names=[var],
            instrument_names=instrument_names,
            equation=equation,
            style="iv",
            collapsed=False,
        )

    def create_gmm_style_instruments(
        self,
        var: str,
        min_lag: int,
        max_lag: Optional[int] = None,
        equation: str = "diff",
        collapse: bool = False,
    ) -> InstrumentSet:
        """
        Create GMM-style instruments (separate column per time period).

        GMM-style instruments create a separate column for each available lag
        in each time period, leading to instrument proliferation unless collapsed.

        Parameters
        ----------
        var : str
            Variable to instrument
        min_lag : int
            Minimum lag to use
        max_lag : int, optional
            Maximum lag to use (None = all available)
        equation : str
            'diff' for differenced equation, 'level' for level equation
        collapse : bool
            Whether to collapse instruments to avoid proliferation

        Returns
        -------
        InstrumentSet
            GMM-style instrument set

        Examples
        --------
        >>> # Without collapse: Creates many columns (one per time*lag)
        >>> Z = builder.create_gmm_style_instruments('x', min_lag=2, max_lag=99)
        >>> # With collapse: Creates one column per lag
        >>> Z_collapsed = builder.create_gmm_style_instruments(
        ...     'x', min_lag=2, max_lag=99, collapse=True
        ... )

        Notes
        -----
        Collapse mode (Roodman 2009 recommendation):
        - Reduces instrument count from O(T²) to O(T)
        - Uses sum of available lags instead of separate columns
        - Helps avoid overfitting and weak instruments
        """
        if collapse:
            return self._create_gmm_collapsed(var, min_lag, max_lag, equation)
        else:
            return self._create_gmm_standard(var, min_lag, max_lag, equation)

    def _create_gmm_standard(
        self, var: str, min_lag: int, max_lag: Optional[int], equation: str
    ) -> InstrumentSet:
        """Create GMM-style instruments without collapse."""
        var_data = self.data[var].values
        ids = self.data[self.id_var].values
        times = self.data[self.time_var].values
        n_obs = len(self.data)

        # Determine actual max_lag
        if max_lag is None:
            max_lag = int(1e6)  # Effectively infinite

        # Build instruments time period by time period
        Z_list = []
        instrument_names = []

        for t_idx, t in enumerate(self.time_periods):
            # Skip early periods where no lags available
            if t_idx < min_lag:
                continue

            # Determine available lags for this period
            available_lags = []
            for lag in range(min_lag, min(max_lag + 1, t_idx + 1)):
                available_lags.append(lag)

            if not available_lags:
                continue

            # Create instrument columns for this time period
            for lag in available_lags:
                col = np.full(n_obs, np.nan)
                col_name = f"{var}_t{t}_L{lag}"

                # Fill only for observations at time t
                mask_t = times == t
                for i in np.where(mask_t)[0]:
                    current_id = ids[i]
                    # Find lagged value
                    mask_lag = (ids == current_id) & (times == t - lag)
                    if np.any(mask_lag):
                        lag_idx = np.where(mask_lag)[0][0]
                        if equation == "diff":
                            col[i] = var_data[lag_idx]
                        else:
                            # For level equation, use differences
                            mask_lag1 = (ids == current_id) & (times == t - lag - 1)
                            if np.any(mask_lag1):
                                lag1_idx = np.where(mask_lag1)[0][0]
                                col[i] = var_data[lag_idx] - var_data[lag1_idx]

                Z_list.append(col)
                instrument_names.append(col_name)

        # Stack into matrix
        Z = np.column_stack(Z_list) if Z_list else np.empty((n_obs, 0))

        return InstrumentSet(
            Z=Z,
            variable_names=[var],
            instrument_names=instrument_names,
            equation=equation,
            style="gmm",
            collapsed=False,
        )

    def _analyze_lag_availability(
        self, var: str, min_lag: int, max_lag: int, min_coverage: float = 0.10
    ) -> List[int]:
        """
        Analyze which lags have sufficient data coverage.

        For unbalanced panels, some lags may be available for very few
        observations. This method identifies lags with sufficient coverage.

        Parameters
        ----------
        var : str
            Variable to analyze
        min_lag : int
            Minimum lag to consider
        max_lag : int
            Maximum lag to consider
        min_coverage : float
            Minimum fraction of observations that must have valid lagged values
            (default: 0.10 = 10%)

        Returns
        -------
        List[int]
            Lags with sufficient coverage
        """
        var_data = self.data[var].values
        ids = self.data[self.id_var].values
        times = self.data[self.time_var].values
        n_obs = len(self.data)

        valid_lags = []

        for lag in range(min_lag, max_lag + 1):
            # Count how many observations would have valid lagged values
            n_valid = 0
            for i in range(n_obs):
                current_id = ids[i]
                current_time = times[i]

                # Check if lagged value exists
                mask_lag = (ids == current_id) & (times == current_time - lag)
                if np.any(mask_lag):
                    lag_idx = np.where(mask_lag)[0][0]
                    if not np.isnan(var_data[lag_idx]):
                        n_valid += 1

            # Include lag if coverage is sufficient
            coverage = n_valid / n_obs
            if coverage >= min_coverage:
                valid_lags.append(lag)

        return valid_lags

    def _create_gmm_collapsed(
        self, var: str, min_lag: int, max_lag: Optional[int], equation: str
    ) -> InstrumentSet:
        """
        Create collapsed GMM-style instruments.

        Collapse creates one column per lag, summing across time periods.
        This dramatically reduces instrument count while preserving information.

        For unbalanced panels, automatically filters out lags with very low
        data coverage (< 10% of observations).
        """
        var_data = self.data[var].values
        ids = self.data[self.id_var].values
        times = self.data[self.time_var].values
        n_obs = len(self.data)

        # Determine actual max_lag based on data
        # Maximum possible lag is n_periods - 1
        actual_max_lag = self.n_periods - 1

        if max_lag is None:
            max_lag = actual_max_lag
        else:
            # Limit max_lag to what's actually available
            max_lag = min(max_lag, actual_max_lag)

        # Smart selection: Filter lags with sufficient data coverage
        # This helps with unbalanced panels by excluding mostly-NaN lags
        possible_lags = self._analyze_lag_availability(var, min_lag, max_lag, min_coverage=0.10)

        # If no lags meet the coverage threshold, use at least min_lag and min_lag+1
        if not possible_lags and min_lag <= max_lag:
            import warnings

            warnings.warn(
                f"No lags for variable '{var}' meet the 10% coverage threshold. "
                f"Using lags {min_lag} and {min(min_lag+1, max_lag)} anyway.",
                UserWarning,
            )
            possible_lags = [min_lag]
            if min_lag + 1 <= max_lag:
                possible_lags.append(min_lag + 1)

        Z_list = []
        instrument_names = []

        for lag in possible_lags:
            col = np.full(n_obs, np.nan)
            col_name = f"{var}_L{lag}_collapsed"

            # For each observation, get lagged value if available
            for i in range(n_obs):
                current_id = ids[i]
                current_time = times[i]

                # Find lagged value
                mask_lag = (ids == current_id) & (times == current_time - lag)
                if np.any(mask_lag):
                    lag_idx = np.where(mask_lag)[0][0]
                    if equation == "diff":
                        col[i] = var_data[lag_idx]
                    else:
                        # For level equation, use differences
                        mask_lag1 = (ids == current_id) & (times == current_time - lag - 1)
                        if np.any(mask_lag1):
                            lag1_idx = np.where(mask_lag1)[0][0]
                            col[i] = var_data[lag_idx] - var_data[lag1_idx]

            Z_list.append(col)
            instrument_names.append(col_name)

        # Stack into matrix
        Z = np.column_stack(Z_list) if Z_list else np.empty((n_obs, 0))

        return InstrumentSet(
            Z=Z,
            variable_names=[var],
            instrument_names=instrument_names,
            equation=equation,
            style="gmm",
            collapsed=True,
        )

    def combine_instruments(self, *instrument_sets: InstrumentSet) -> InstrumentSet:
        """
        Combine multiple instrument sets.

        Parameters
        ----------
        *instrument_sets : InstrumentSet
            Instrument sets to combine

        Returns
        -------
        InstrumentSet
            Combined instrument set

        Examples
        --------
        >>> Z_gmm = builder.create_gmm_style_instruments('y', 2, 99, collapse=True)
        >>> Z_iv = builder.create_iv_style_instruments('x', 2, 4)
        >>> Z_combined = builder.combine_instruments(Z_gmm, Z_iv)
        """
        if not instrument_sets:
            raise ValueError("Must provide at least one instrument set")

        # Combine matrices
        Z_combined = np.column_stack([iset.Z for iset in instrument_sets])

        # Combine names
        var_names = []
        inst_names = []
        for iset in instrument_sets:
            var_names.extend(iset.variable_names)
            inst_names.extend(iset.instrument_names)

        return InstrumentSet(
            Z=Z_combined,
            variable_names=var_names,
            instrument_names=inst_names,
            equation=instrument_sets[0].equation,
            style="mixed",
            collapsed=False,
        )

    def instrument_count_analysis(self, Z: InstrumentSet) -> pd.DataFrame:
        """
        Analyze instrument count.

        Parameters
        ----------
        Z : InstrumentSet
            Instrument set to analyze

        Returns
        -------
        pd.DataFrame
            Analysis of instrument counts

        Examples
        --------
        >>> Z = builder.create_gmm_style_instruments('y', 2, 99)
        >>> analysis = builder.instrument_count_analysis(Z)
        >>> print(analysis)
        """
        analysis = {
            "Total instruments": Z.n_instruments,
            "Observations": Z.n_obs,
            "Groups": self.n_groups,
            "Instrument ratio": Z.n_instruments / self.n_groups,
            "Style": Z.style,
            "Collapsed": Z.collapsed,
            "Variables": ", ".join(Z.variable_names),
        }

        # Warning if too many instruments
        if Z.n_instruments > self.n_groups:
            analysis["Warning"] = (
                f"Too many instruments ({Z.n_instruments} > {self.n_groups} groups)"
            )
        else:
            analysis["Warning"] = "OK"

        return cast(pd.DataFrame, pd.DataFrame([analysis]).T)

    def get_valid_obs_mask(self, Z: InstrumentSet) -> np.ndarray:
        """
        Get mask of valid observations (non-missing instruments).

        Parameters
        ----------
        Z : InstrumentSet
            Instrument set

        Returns
        -------
        np.ndarray
            Boolean mask of valid observations
        """
        # Valid if at least one instrument is non-missing
        return np.asarray(~np.all(np.isnan(Z.Z), axis=1))
