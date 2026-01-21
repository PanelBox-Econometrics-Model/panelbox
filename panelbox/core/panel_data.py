"""
PanelData - Container for panel data with validation and transformations.

This module provides the core PanelData class for handling panel datasets.
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd


class PanelData:
    """
    Container for panel data with validation and transformations.

    This class provides a structured way to work with panel data, ensuring
    proper validation of the panel structure and offering common transformations
    used in panel econometrics.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format (one row per entity-time observation)
    entity_col : str
        Name of the column identifying entities (e.g., 'firm', 'country', 'id')
    time_col : str
        Name of the column identifying time periods (e.g., 'year', 'quarter', 'time')

    Attributes
    ----------
    data : pd.DataFrame
        Original data in long format
    entity_col : str
        Entity identifier column name
    time_col : str
        Time identifier column name
    is_balanced : bool
        Whether the panel is balanced (all entities have same number of periods)
    n_entities : int
        Number of unique entities
    n_periods : int
        Number of time periods (max if unbalanced)
    n_obs : int
        Total number of observations
    entities : np.ndarray
        Array of unique entity identifiers
    time_periods : np.ndarray
        Array of unique time periods

    Examples
    --------
    >>> import pandas as pd
    >>> import panelbox as pb
    >>>
    >>> # Create sample panel data
    >>> data = pd.DataFrame({
    ...     'firm': [1, 1, 1, 2, 2, 2],
    ...     'year': [2020, 2021, 2022, 2020, 2021, 2022],
    ...     'invest': [100, 110, 115, 200, 210, 220],
    ...     'value': [1000, 1100, 1200, 2000, 2100, 2200]
    ... })
    >>>
    >>> # Create PanelData object
    >>> panel = pb.PanelData(data, entity_col='firm', time_col='year')
    >>> print(panel.summary())
    """

    def __init__(
        self,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str
    ):
        # Validate inputs
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if entity_col not in data.columns:
            raise ValueError(f"entity_col '{entity_col}' not found in data columns")

        if time_col not in data.columns:
            raise ValueError(f"time_col '{time_col}' not found in data columns")

        # Store data and identifiers
        self.data = data.copy()
        self.entity_col = entity_col
        self.time_col = time_col

        # Sort by entity and time for consistency
        self.data = self.data.sort_values([entity_col, time_col]).reset_index(drop=True)

        # Compute panel structure
        self.entities = self.data[entity_col].unique()
        self.time_periods = self.data[time_col].unique()
        self.n_entities = len(self.entities)
        self.n_obs = len(self.data)

        # Check if balanced
        obs_per_entity = self.data.groupby(entity_col).size()
        self.n_periods = int(obs_per_entity.max())
        self.is_balanced = (obs_per_entity == self.n_periods).all()

        if not self.is_balanced:
            self.min_periods = int(obs_per_entity.min())
            self.avg_periods = float(obs_per_entity.mean())

    def demeaning(
        self,
        variables: Optional[Union[str, List[str]]] = None,
        method: str = 'entity'
    ) -> pd.DataFrame:
        """
        Remove means from variables (within transformation).

        This is the core transformation for fixed effects estimation.

        Parameters
        ----------
        variables : str or list of str, optional
            Variables to demean. If None, demeans all numeric columns
            except entity and time identifiers.
        method : str, default='entity'
            Type of demeaning:
            - 'entity': Remove entity-specific means (within transformation)
            - 'time': Remove time-specific means
            - 'both': Remove both entity and time means (two-way demeaning)

        Returns
        -------
        pd.DataFrame
            Demeaned data

        Examples
        --------
        >>> demeaned = panel.demeaning(['invest', 'value'], method='entity')
        """
        if variables is None:
            # Demean all numeric columns except identifiers
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            variables = [col for col in numeric_cols
                        if col not in [self.entity_col, self.time_col]]
        elif isinstance(variables, str):
            variables = [variables]

        # Validate variables
        for var in variables:
            if var not in self.data.columns:
                raise ValueError(f"Variable '{var}' not found in data")

        result = self.data.copy()

        if method == 'entity':
            # Remove entity means
            group_means = result.groupby(self.entity_col)[variables].transform('mean')
            result[variables] = result[variables] - group_means

        elif method == 'time':
            # Remove time means
            group_means = result.groupby(self.time_col)[variables].transform('mean')
            result[variables] = result[variables] - group_means

        elif method == 'both':
            # Two-way demeaning (entity and time)
            # First remove entity means
            entity_means = result.groupby(self.entity_col)[variables].transform('mean')
            result[variables] = result[variables] - entity_means

            # Then remove time means from demeaned data
            time_means = result.groupby(self.time_col)[variables].transform('mean')
            result[variables] = result[variables] - time_means
        else:
            raise ValueError("method must be 'entity', 'time', or 'both'")

        return result

    def first_difference(
        self,
        variables: Optional[Union[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Compute first differences (Δy_it = y_it - y_i,t-1).

        This transformation eliminates time-invariant fixed effects.

        Parameters
        ----------
        variables : str or list of str, optional
            Variables to difference. If None, differences all numeric columns
            except entity and time identifiers.

        Returns
        -------
        pd.DataFrame
            First-differenced data (observations for t=1 are dropped)

        Examples
        --------
        >>> diff_data = panel.first_difference(['invest', 'value'])
        """
        if variables is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            variables = [col for col in numeric_cols
                        if col not in [self.entity_col, self.time_col]]
        elif isinstance(variables, str):
            variables = [variables]

        # Validate variables
        for var in variables:
            if var not in self.data.columns:
                raise ValueError(f"Variable '{var}' not found in data")

        result = self.data.copy()

        # Compute differences within each entity
        for var in variables:
            result[var] = result.groupby(self.entity_col)[var].diff()

        # Drop first observation for each entity (NaN from diff)
        result = result.dropna(subset=variables)

        return result

    def lag(
        self,
        variable: str,
        lags: Union[int, List[int]] = 1
    ) -> pd.DataFrame:
        """
        Create lagged variables.

        Parameters
        ----------
        variable : str
            Variable to lag
        lags : int or list of int, default=1
            Lag order(s). Can be a single integer or list of integers.

        Returns
        -------
        pd.DataFrame
            Data with lagged variable(s) added.
            Column names will be 'L{lag}.{variable}'

        Examples
        --------
        >>> # Create single lag
        >>> data_lag1 = panel.lag('invest', lags=1)
        >>>
        >>> # Create multiple lags
        >>> data_lags = panel.lag('invest', lags=[1, 2, 3])
        """
        if variable not in self.data.columns:
            raise ValueError(f"Variable '{variable}' not found in data")

        if isinstance(lags, int):
            lags = [lags]

        result = self.data.copy()

        for lag in lags:
            if lag < 1:
                raise ValueError("Lag order must be >= 1")

            lag_name = f'L{lag}.{variable}'
            result[lag_name] = result.groupby(self.entity_col)[variable].shift(lag)

        return result

    def lead(
        self,
        variable: str,
        leads: Union[int, List[int]] = 1
    ) -> pd.DataFrame:
        """
        Create lead variables (forward lags).

        Parameters
        ----------
        variable : str
            Variable to lead
        leads : int or list of int, default=1
            Lead order(s). Can be a single integer or list of integers.

        Returns
        -------
        pd.DataFrame
            Data with lead variable(s) added.
            Column names will be 'F{lead}.{variable}'

        Examples
        --------
        >>> data_lead = panel.lead('invest', leads=1)
        """
        if variable not in self.data.columns:
            raise ValueError(f"Variable '{variable}' not found in data")

        if isinstance(leads, int):
            leads = [leads]

        result = self.data.copy()

        for lead in leads:
            if lead < 1:
                raise ValueError("Lead order must be >= 1")

            lead_name = f'F{lead}.{variable}'
            result[lead_name] = result.groupby(self.entity_col)[variable].shift(-lead)

        return result

    def balance(self) -> 'PanelData':
        """
        Balance the panel by keeping only entities with complete time series.

        This removes any entities that don't have observations for all time periods.

        Returns
        -------
        PanelData
            New PanelData object with balanced panel

        Examples
        --------
        >>> balanced_panel = panel.balance()
        >>> print(f"Original: {panel.n_entities} entities")
        >>> print(f"Balanced: {balanced_panel.n_entities} entities")
        """
        if self.is_balanced:
            return self

        # Count observations per entity
        obs_counts = self.data.groupby(self.entity_col).size()

        # Keep only entities with max number of periods
        complete_entities = obs_counts[obs_counts == self.n_periods].index

        # Filter data
        balanced_data = self.data[self.data[self.entity_col].isin(complete_entities)]

        return PanelData(balanced_data, self.entity_col, self.time_col)

    def summary(self) -> str:
        """
        Generate a summary of the panel structure.

        Returns
        -------
        str
            Formatted summary of panel characteristics

        Examples
        --------
        >>> print(panel.summary())
        """
        lines = []
        lines.append("=" * 60)
        lines.append("PANEL DATA SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Entity identifier: {self.entity_col}")
        lines.append(f"Time identifier:   {self.time_col}")
        lines.append("-" * 60)
        lines.append(f"Number of entities:     {self.n_entities:,}")
        lines.append(f"Number of time periods: {len(self.time_periods):,}")
        lines.append(f"Total observations:     {self.n_obs:,}")
        lines.append("-" * 60)
        lines.append(f"Balanced:               {'Yes' if self.is_balanced else 'No'}")

        if not self.is_balanced:
            lines.append(f"Min periods per entity: {self.min_periods}")
            lines.append(f"Max periods per entity: {self.n_periods}")
            lines.append(f"Avg periods per entity: {self.avg_periods:.1f}")
        else:
            lines.append(f"Periods per entity:     {self.n_periods}")

        lines.append("-" * 60)
        lines.append(f"Time period range:      {self.time_periods.min()} to {self.time_periods.max()}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of PanelData."""
        balanced_str = "Balanced" if self.is_balanced else "Unbalanced"
        return (f"PanelData({balanced_str}, "
                f"n_entities={self.n_entities}, "
                f"n_periods={self.n_periods}, "
                f"n_obs={self.n_obs})")
