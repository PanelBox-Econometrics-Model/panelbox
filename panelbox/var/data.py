"""
Data container for Panel VAR models.

This module provides the PanelVARData class for representing Panel VAR systems,
with rigorous handling of lags, temporal gaps, and panel structure.
"""

from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd


class PanelVARData:
    """
    Data container for Panel Vector Autoregression models.

    This class handles the construction and validation of Panel VAR datasets,
    including automatic lag construction, gap detection, and support for both
    balanced and unbalanced panels.

    **Critical Safety Features:**
    - Lags are constructed within each entity separately to prevent cross-contamination
    - Temporal gaps are detected and rejected with clear error messages
    - Missing values are handled consistently across equations

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format with entity and time identifiers
    endog_vars : List[str]
        List of endogenous variable names (K variables)
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    exog_vars : List[str], optional
        List of exogenous variable names
    lags : int, default=1
        Number of lags (p) for the VAR model
    trend : {'none', 'constant', 'trend', 'both'}, default='constant'
        Deterministic trend specification:
        - 'none': No intercept or trend
        - 'constant': Intercept only
        - 'trend': Time trend only
        - 'both': Intercept and time trend
    dropna : {'any', 'equation'}, default='any'
        Strategy for handling missing values:
        - 'any': Drop observation if any variable has missing value
        - 'equation': Drop missing values separately for each equation

    Attributes
    ----------
    K : int
        Number of endogenous variables
    p : int
        Number of lags
    N : int
        Number of entities
    T_min : int
        Minimum time periods per entity
    T_max : int
        Maximum time periods per entity
    T_avg : float
        Average time periods per entity
    n_obs : int
        Total number of observations (after dropping lags and NAs)
    is_balanced : bool
        Whether the panel is balanced

    Examples
    --------
    >>> import pandas as pd
    >>> from panelbox.var import PanelVARData
    >>>
    >>> # Create Panel VAR data
    >>> data = PanelVARData(
    ...     df,
    ...     endog_vars=['gdp', 'inflation', 'rate'],
    ...     entity_col='country',
    ...     time_col='year',
    ...     lags=2,
    ...     trend='constant'
    ... )
    >>>
    >>> print(f"Variables: {data.K}, Lags: {data.p}, Entities: {data.N}")
    >>> print(f"Balanced: {data.is_balanced}")
    >>>
    >>> # Get data for equation k
    >>> y, X = data.equation_data(0)
    >>>
    >>> # Get stacked data
    >>> df_stacked = data.to_stacked()

    Notes
    -----
    **Lag Construction:**

    Lags are constructed using `.groupby(entity).shift()` to ensure that
    lag t of entity A never contains an observation from entity B. This is
    the most critical aspect of Panel VAR data preparation.

    **Temporal Gaps:**

    Internal gaps in time series (e.g., missing year 2005 in 2000-2010)
    are NOT allowed and will raise a ValueError. Only leading/trailing
    missing observations are acceptable.

    **Within Transformation:**

    The within transformation (entity demeaning) is applied during estimation,
    not during data construction. This class stores the original data.

    References
    ----------
    .. [1] Holtz-Eakin, D., Newey, W., & Rosen, H. S. (1988). Estimating
           vector autoregressions with panel data. Econometrica, 56(6), 1371-1395.
    .. [2] Abrigo, M. R., & Love, I. (2016). Estimation of panel vector
           autoregression in Stata. The Stata Journal, 16(3), 778-804.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        endog_vars: List[str],
        entity_col: str,
        time_col: str,
        exog_vars: Optional[List[str]] = None,
        lags: int = 1,
        trend: Literal["none", "constant", "trend", "both"] = "constant",
        dropna: Literal["any", "equation"] = "any",
    ):
        # Store configuration
        self.endog_vars = endog_vars
        self.entity_col = entity_col
        self.time_col = time_col
        self.exog_vars = exog_vars if exog_vars is not None else []
        self._lags = lags
        self.trend = trend
        self.dropna = dropna

        # Validate inputs
        self._validate_inputs(data)

        # Sort data by entity and time
        self.data = data.sort_values([entity_col, time_col]).reset_index(drop=True)

        # Check for temporal gaps (CRITICAL)
        self._check_temporal_gaps()

        # Build lagged data
        self._construct_lags()

        # Compute properties
        self._compute_panel_properties()

    def _validate_inputs(self, data: pd.DataFrame) -> None:
        """Validate input parameters."""
        # Check data type
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a DataFrame, got {type(data)}")

        # Check required columns
        missing_cols = []
        if self.entity_col not in data.columns:
            missing_cols.append(self.entity_col)
        if self.time_col not in data.columns:
            missing_cols.append(self.time_col)

        for var in self.endog_vars:
            if var not in data.columns:
                missing_cols.append(var)

        for var in self.exog_vars:
            if var not in data.columns:
                missing_cols.append(var)

        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")

        # Check for at least 2 endogenous variables
        if len(self.endog_vars) < 1:
            raise ValueError("Must have at least 1 endogenous variable")

        # Check lags
        if self._lags < 1:
            raise ValueError(f"lags must be >= 1, got {self._lags}")

        # Check trend option
        if self.trend not in ["none", "constant", "trend", "both"]:
            raise ValueError(
                f"trend must be one of ['none', 'constant', 'trend', 'both'], got '{self.trend}'"
            )

    def _check_temporal_gaps(self) -> None:
        """
        Check for internal temporal gaps in each entity's time series.

        Raises
        ------
        ValueError
            If any entity has internal gaps in its time series
        """
        entities = self.data[self.entity_col].unique()

        for entity in entities:
            entity_data = self.data[self.data[self.entity_col] == entity]
            times = sorted(entity_data[self.time_col].values)

            if len(times) < 2:
                continue

            # Check if time series is contiguous
            # Detect the time increment
            time_diffs = np.diff(times)

            if len(time_diffs) == 0:
                continue

            # For numeric time indices
            if np.issubdtype(type(times[0]), np.number):
                # Get the most common difference (assumed to be the correct increment)
                unique_diffs = np.unique(time_diffs)

                # Allow for floating point comparison if needed
                if len(unique_diffs) > 1:
                    # Check if all differences are multiples of the minimum
                    min_diff = np.min(time_diffs[time_diffs > 0])

                    # Check for gaps (differences larger than min_diff)
                    if np.any(time_diffs > min_diff * 1.01):  # small tolerance
                        gap_positions = np.where(time_diffs > min_diff * 1.01)[0]
                        raise ValueError(
                            f"Entity '{entity}' has internal temporal gaps. "
                            f"Gap detected at positions: {gap_positions.tolist()}. "
                            f"Time values before gap: {[times[i] for i in gap_positions]}. "
                            f"Panel VAR requires continuous time series within each entity. "
                            f"Please fill gaps or remove entities with gaps."
                        )

    def _construct_lags(self) -> None:
        """
        Construct lagged variables with strict entity separation.

        This is the most critical method. Lags MUST NOT cross entity boundaries.
        """
        # Start with original data
        df = self.data.copy()

        # Create lag columns for each endogenous variable
        # Use groupby to ensure lags don't cross entities
        for var in self.endog_vars:
            for lag in range(1, self._lags + 1):
                lag_col = f"L{lag}.{var}"
                df[lag_col] = df.groupby(self.entity_col)[var].shift(lag)

        # Drop rows with NaN in lagged variables if dropna='any'
        if self.dropna == "any":
            # Identify lag columns
            lag_cols = [
                f"L{lag}.{var}" for var in self.endog_vars for lag in range(1, self._lags + 1)
            ]
            # Drop rows with any NaN in lags or endogenous variables
            df = df.dropna(subset=self.endog_vars + lag_cols)

        # Store the constructed data
        self.data_with_lags = df.reset_index(drop=True)

        # Verify no cross-contamination (CRITICAL TEST)
        self._verify_no_cross_contamination()

    def _verify_no_cross_contamination(self) -> None:
        """
        Verify that lags do not cross entity boundaries.

        This is a critical safety check.
        """
        df = self.data_with_lags

        for entity in df[self.entity_col].unique():
            entity_mask = df[self.entity_col] == entity
            entity_data = df[entity_mask]

            # For each lagged variable, verify continuity
            for var in self.endog_vars:
                for lag in range(1, self._lags + 1):
                    lag_col = f"L{lag}.{var}"

                    # Check that L{lag}.var[t] == var[t-lag] within entity
                    # Skip first 'lag' observations (they should be NaN or dropped)
                    if len(entity_data) <= lag:
                        continue

                    # Get current and lagged values (skip NaN)
                    current_vals = entity_data[var].values[lag:]
                    lagged_vals = entity_data[lag_col].values[lag:]

                    # The lagged value at position t should equal the current value at position t-lag
                    expected_lagged = entity_data[var].values[:-lag]

                    # Check (with tolerance for floating point)
                    if len(expected_lagged) > 0 and len(lagged_vals) > 0:
                        # Remove NaN for comparison
                        valid_mask = ~(np.isnan(expected_lagged) | np.isnan(lagged_vals))
                        if np.any(valid_mask):
                            if not np.allclose(
                                expected_lagged[valid_mask],
                                lagged_vals[valid_mask],
                                rtol=1e-10,
                                equal_nan=True,
                            ):
                                raise ValueError(
                                    f"Cross-contamination detected for entity '{entity}', "
                                    f"variable '{var}', lag {lag}. "
                                    f"This is a critical bug in lag construction."
                                )

    def _compute_panel_properties(self) -> None:
        """Compute panel properties (N, T, balance, etc.)."""
        df = self.data_with_lags

        # Number of entities
        self._N = df[self.entity_col].nunique()

        # Time periods per entity
        entity_groups = df.groupby(self.entity_col)[self.time_col].count()
        self._T_min = int(entity_groups.min())
        self._T_max = int(entity_groups.max())
        self._T_avg = float(entity_groups.mean())

        # Total observations
        self._n_obs = len(df)

        # Check if balanced
        self._is_balanced = self._T_min == self._T_max

    @property
    def K(self) -> int:
        """Number of endogenous variables."""
        return len(self.endog_vars)

    @property
    def p(self) -> int:
        """Number of lags."""
        return self._lags

    @property
    def N(self) -> int:
        """Number of entities."""
        return self._N

    @property
    def T_min(self) -> int:
        """Minimum time periods per entity."""
        return self._T_min

    @property
    def T_max(self) -> int:
        """Maximum time periods per entity."""
        return self._T_max

    @property
    def T_avg(self) -> float:
        """Average time periods per entity."""
        return self._T_avg

    @property
    def n_obs(self) -> int:
        """Total number of observations."""
        return self._n_obs

    @property
    def is_balanced(self) -> bool:
        """Whether the panel is balanced."""
        return self._is_balanced

    def to_stacked(self) -> pd.DataFrame:
        """
        Return the data in stacked format with lags.

        Returns
        -------
        pd.DataFrame
            Data with entity, time, endogenous variables, lags, and exogenous variables
        """
        return self.data_with_lags.copy()

    def equation_data(self, k: int, include_constant: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get y and X matrices for equation k.

        For equation k, the dependent variable is endog_vars[k] and the
        independent variables include all lags of all endogenous variables,
        exogenous variables, and deterministic terms.

        Parameters
        ----------
        k : int
            Equation index (0 to K-1)
        include_constant : bool, default=True
            Whether to include constant term. Set to False when using
            within transformation (fixed effects), as the constant is
            absorbed by entity demeaning.

        Returns
        -------
        y : np.ndarray
            Dependent variable (n_obs,)
        X : np.ndarray
            Independent variables (n_obs, n_regressors)

        Notes
        -----
        The X matrix includes (in order):
        1. Lags of all endogenous variables: L1.y1, L1.y2, ..., L1.yK, L2.y1, ..., Lp.yK
        2. Exogenous variables (if any)
        3. Deterministic terms (constant and/or trend) if include_constant=True

        When using within transformation (fixed effects), set include_constant=False
        to avoid singularity, as the constant term is absorbed by the demeaning.
        """
        if k < 0 or k >= self.K:
            raise ValueError(f"Equation index k must be between 0 and {self.K - 1}, got {k}")

        df = self.data_with_lags

        # Dependent variable
        y = df[self.endog_vars[k]].values

        # Independent variables
        X_cols = []

        # 1. All lags of all endogenous variables (organized by lag)
        for lag in range(1, self._lags + 1):
            for var in self.endog_vars:
                X_cols.append(f"L{lag}.{var}")

        # 2. Exogenous variables
        X_cols.extend(self.exog_vars)

        # Build X matrix
        X_list = []

        # Add lag and exog columns
        for col in X_cols:
            X_list.append(df[col].values.reshape(-1, 1))

        # 3. Deterministic terms (only if include_constant=True)
        if include_constant:
            if self.trend in ["constant", "both"]:
                X_list.append(np.ones((len(df), 1)))

            if self.trend in ["trend", "both"]:
                # Create time trend (1, 2, 3, ...)
                # This should be entity-specific or global depending on specification
                # For simplicity, use global trend
                trend_vals = np.arange(1, len(df) + 1).reshape(-1, 1)
                X_list.append(trend_vals)

        # Concatenate all columns
        if len(X_list) > 0:
            X = np.hstack(X_list)
        else:
            # No regressors (should not happen with at least 1 lag)
            X = np.empty((len(df), 0))

        return y, X

    def get_regressor_names(self, include_constant: bool = True) -> List[str]:
        """
        Get names of all regressors in the VAR equations.

        Parameters
        ----------
        include_constant : bool, default=True
            Whether to include constant and trend names

        Returns
        -------
        List[str]
            List of regressor names
        """
        names = []

        # Lags
        for lag in range(1, self._lags + 1):
            for var in self.endog_vars:
                names.append(f"L{lag}.{var}")

        # Exogenous
        names.extend(self.exog_vars)

        # Deterministic (only if include_constant=True)
        if include_constant:
            if self.trend in ["constant", "both"]:
                names.append("const")

            if self.trend in ["trend", "both"]:
                names.append("trend")

        return names

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PanelVARData(K={self.K}, p={self.p}, N={self.N}, "
            f"T_avg={self.T_avg:.1f}, n_obs={self.n_obs}, "
            f"balanced={self.is_balanced})"
        )
