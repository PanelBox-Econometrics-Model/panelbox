"""
First Difference estimator for panel data.

This module provides the First Difference estimator which eliminates
entity fixed effects through first-differencing.
"""

from typing import Optional

import numpy as np
import pandas as pd

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults
from panelbox.standard_errors import (
    cluster_by_entity,
    driscoll_kraay,
    newey_west,
    pcse,
    robust_covariance,
    twoway_cluster,
)
from panelbox.utils.matrix_ops import compute_ols, compute_panel_rsquared, compute_vcov_nonrobust


class FirstDifferenceEstimator(PanelModel):
    """
    First Difference estimator for panel data.

    This estimator eliminates unobserved entity-specific fixed effects
    through first-differencing. Instead of demeaning (like Fixed Effects),
    it takes differences:
        Δy_it = y_it - y_{i,t-1} = β Δx_it + Δε_it

    where Δ denotes the first difference operator.

    The entity fixed effect (α_i) cancels out because it's time-invariant:
        Δα_i = α_i - α_i = 0

    Advantages over Fixed Effects (within estimator):
    - More robust when T is small (few time periods)
    - Better suited for models with serially correlated errors
    - Handles unbalanced panels naturally
    - No dummy variable trap issues

    Disadvantages:
    - Loses one time period per entity (first period dropped)
    - Amplifies measurement error (differences magnify noise)
    - Less efficient than FE under homoskedastic errors
    - Loses time-invariant variables (like FE)

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format (must be sorted by entity and time)
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    weights : np.ndarray, optional
        Observation weights (applied to differenced data)

    Attributes
    ----------
    n_obs_original : int
        Number of observations before differencing
    n_obs_differenced : int
        Number of observations after differencing (loses first period per entity)

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Load data
    >>> data = pb.load_grunfeld()
    >>>
    >>> # First Difference estimator
    >>> fd = pb.FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year")
    >>> results = fd.fit(cov_type='robust')
    >>> print(results.summary())
    >>>
    >>> # Compare with Fixed Effects
    >>> fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    >>> results_fe = fe.fit(cov_type='robust')
    >>>
    >>> print(f"First Diff coefs: {results.params.values}")
    >>> print(f"Fixed Effects coefs: {results_fe.params.values}")
    >>>
    >>> # Under homoskedasticity, should be similar
    >>> # Under serial correlation, FD may be more consistent

    Notes
    -----
    **Data Requirements:**
    - Data must be sorted by entity and time before estimation
    - Missing periods will be handled by taking differences only within consecutive observations
    - At least 2 time periods per entity required

    **First Differencing:**
    - For each entity i, compute: Δy_it = y_it - y_{i,t-1}
    - Drops the first observation for each entity
    - If N entities and T periods (balanced): N*T → N*(T-1) observations

    **Inference:**
    - Standard errors account for structure of differenced data
    - Cluster-robust SE recommended (clustering by entity)
    - Driscoll-Kraay useful for serial correlation and heteroskedasticity

    **Comparison with Fixed Effects:**
    - FE uses within transformation (demeaning): y_it - ȳ_i
    - FD uses first difference: y_it - y_{i,t-1}
    - Under random walk: y_it = y_{i,t-1} + ε_it → FD removes unit root
    - Under classical RE/FE assumptions: FE is more efficient

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
       and Panel Data. MIT Press. Section 10.5.
    .. [2] Baltagi, B. H. (2013). Econometric Analysis of Panel Data.
       Wiley. Chapter 3.
    .. [3] Hsiao, C. (2014). Analysis of Panel Data. Cambridge University Press.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

        # Store original observation count
        self.n_obs_original = len(data)
        self.n_obs_differenced: Optional[int] = None

    def fit(self, cov_type: str = "nonrobust", **cov_kwds) -> PanelResults:
        """
        Fit the First Difference estimator.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Type of covariance estimator:
            - 'nonrobust': Classical standard errors
            - 'robust' or 'hc1': Heteroskedasticity-robust (HC1)
            - 'hc0', 'hc2', 'hc3': Other HC variants
            - 'clustered': Cluster-robust (by entity, recommended for FD)
            - 'twoway': Two-way clustered (entity and time)
            - 'driscoll_kraay': Driscoll-Kraay (for serial correlation)
            - 'newey_west': Newey-West HAC
            - 'pcse': Panel-Corrected Standard Errors
        **cov_kwds
            Additional arguments for covariance estimation:
            - cluster_col: For custom clustering (default: entity)
            - max_lags: For Driscoll-Kraay and Newey-West
            - kernel: For HAC estimators ('bartlett', 'parzen', 'quadratic_spectral')

        Returns
        -------
        PanelResults
            Fitted model results

        Examples
        --------
        >>> # Classical standard errors
        >>> results = model.fit(cov_type='nonrobust')

        >>> # Heteroskedasticity-robust (recommended)
        >>> results = model.fit(cov_type='robust')

        >>> # Cluster-robust by entity (recommended for FD)
        >>> results = model.fit(cov_type='clustered')

        >>> # Driscoll-Kraay (for serial correlation + heteroskedasticity)
        >>> results = model.fit(cov_type='driscoll_kraay', max_lags=2)

        Notes
        -----
        For First Difference models, clustered or Driscoll-Kraay standard errors
        are typically recommended because:
        - Differencing can induce serial correlation (MA(1) structure)
        - Cluster-robust SE account for within-entity correlation
        - Driscoll-Kraay handles both serial correlation and heteroskedasticity
        """
        # Build design matrices from original data
        y_orig, X_orig = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )

        # Get variable names
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Remove intercept from variable names (differencing eliminates it)
        # First differences remove constant terms
        if "Intercept" in var_names:
            var_names = [v for v in var_names if v != "Intercept"]
            # Remove intercept column from X
            X_orig = X_orig[:, 1:]

        # Get entity and time identifiers
        entities = self.data.data[self.data.entity_col].values
        times = self.data.data[self.data.time_col].values

        # Apply first difference transformation
        y_diff, X_diff, entities_diff, times_diff, valid_idx = self._first_difference(
            y_orig, X_orig, entities, times
        )

        # Store differenced observation count
        self.n_obs_differenced = len(y_diff)

        # Check that we have enough observations
        if self.n_obs_differenced < X_diff.shape[1]:
            raise ValueError(
                f"Insufficient observations after differencing: {self.n_obs_differenced} obs, "
                f"{X_diff.shape[1]} parameters. Need at least 2 time periods per entity."
            )

        # Estimate coefficients on differenced data (no intercept)
        beta, resid_diff, fitted_diff = compute_ols(y_diff, X_diff, self.weights)

        # Degrees of freedom
        n = len(y_diff)
        k = X_diff.shape[1]
        df_model = k  # No intercept in first difference
        df_resid = n - k

        # Ensure df_resid is positive
        if df_resid <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom: df_resid = {df_resid}. " f"n={n}, k={k}"
            )

        # Compute fitted values and residuals in original scale (levels)
        # This requires integrating back from differences (not unique, use cumsum)
        # For reporting purposes, we'll use the differenced residuals
        fitted_orig = np.full(len(y_orig), np.nan)
        resid_orig = np.full(len(y_orig), np.nan)
        fitted_orig[valid_idx] = fitted_diff
        resid_orig[valid_idx] = resid_diff

        # Compute covariance matrix (on differenced data)
        cov_type_lower = cov_type.lower()

        if cov_type_lower == "nonrobust":
            vcov = compute_vcov_nonrobust(X_diff, resid_diff, df_resid)

        elif cov_type_lower in ["robust", "hc0", "hc1", "hc2", "hc3"]:
            # Map 'robust' to 'hc1'
            method = "HC1" if cov_type_lower == "robust" else cov_type_lower.upper()
            result = robust_covariance(X_diff, resid_diff, method=method)
            vcov = result.cov_matrix

        elif cov_type_lower == "clustered":
            # Cluster by entity (recommended for FD)
            result = cluster_by_entity(X_diff, resid_diff, entities_diff, df_correction=True)
            vcov = result.cov_matrix

        elif cov_type_lower == "twoway":
            # Two-way clustering: entity and time
            result = twoway_cluster(
                X_diff, resid_diff, entities_diff, times_diff, df_correction=True
            )
            vcov = result.cov_matrix

        elif cov_type_lower == "driscoll_kraay":
            # Driscoll-Kraay for serial correlation (recommended for FD)
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = driscoll_kraay(
                X_diff, resid_diff, times_diff, max_lags=max_lags, kernel=kernel
            )
            vcov = result.cov_matrix

        elif cov_type_lower == "newey_west":
            # Newey-West HAC
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = newey_west(X_diff, resid_diff, max_lags=max_lags, kernel=kernel)
            vcov = result.cov_matrix

        elif cov_type_lower == "pcse":
            # Panel-Corrected Standard Errors
            result = pcse(X_diff, resid_diff, entities_diff, times_diff)
            vcov = result.cov_matrix

        else:
            raise ValueError(
                f"cov_type must be one of: 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3', "
                f"'clustered', 'twoway', 'driscoll_kraay', 'newey_west', 'pcse', got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Compute R-squared measures on differenced data
        # For FD, R² measures fit of differenced model
        tss_diff = np.sum((y_diff - y_diff.mean()) ** 2)
        ess_diff = np.sum(resid_diff**2)
        rsquared = 1 - ess_diff / tss_diff if tss_diff > 0 else 0.0

        # Adjusted R-squared
        rsquared_adj = 1 - (1 - rsquared) * (n - 1) / df_resid

        # For FD, within/between/overall R² are less meaningful
        # We report the R² of the differenced model as the primary measure
        rsquared_within = rsquared  # Differencing is similar to within transformation
        rsquared_between = np.nan  # Not applicable for FD
        rsquared_overall = np.nan  # Not applicable for FD

        # Create Series/DataFrame with variable names
        params = pd.Series(beta.ravel(), index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Model information
        model_info = {
            "model_type": "First Difference",
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "entity_effects": True,  # FD eliminates entity FE
            "time_effects": False,
        }

        # Data information
        data_info = {
            "nobs": n,  # Number of differenced observations
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "n_obs_original": self.n_obs_original,
            "n_obs_dropped": self.n_obs_original - n,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": entities_diff,
            "time_index": times_diff,
        }

        # R-squared dictionary
        rsquared_dict = {
            "rsquared": rsquared,  # R² of differenced model
            "rsquared_adj": rsquared_adj,
            "rsquared_within": rsquared_within,
            "rsquared_between": rsquared_between,
            "rsquared_overall": rsquared_overall,
        }

        # Create results object
        results = PanelResults(
            params=params,
            std_errors=std_errors_series,
            cov_params=cov_params,
            resid=resid_orig,  # Residuals in original indexing
            fittedvalues=fitted_orig,  # Fitted values in original indexing
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store results and update state
        self._results = results
        self._fitted = True

        return results

    def _first_difference(
        self, y: np.ndarray, X: np.ndarray, entities: np.ndarray, times: np.ndarray
    ) -> tuple:
        """
        Apply first difference transformation.

        Computes Δy_it = y_it - y_{i,t-1} and Δx_it = x_it - x_{i,t-1}
        for each entity i.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Independent variables
        entities : np.ndarray
            Entity identifiers
        times : np.ndarray
            Time identifiers

        Returns
        -------
        y_diff : np.ndarray
            Differenced dependent variable
        X_diff : np.ndarray
            Differenced independent variables
        entities_diff : np.ndarray
            Entity identifiers for differenced observations
        times_diff : np.ndarray
            Time identifiers for differenced observations
        valid_idx : np.ndarray
            Indices of valid differenced observations in original data
        """
        # Get unique entities
        unique_entities = np.unique(entities)

        # Initialize lists for differenced data
        y_diff_list = []
        X_diff_list = []
        entities_diff_list = []
        times_diff_list = []
        valid_idx_list = []

        # For each entity, compute first differences
        for entity in unique_entities:
            # Get observations for this entity
            mask = entities == entity
            indices = np.where(mask)[0]

            # Get entity-specific data
            y_entity = y[mask]
            X_entity = X[mask]
            times_entity = times[mask]

            # Sort by time (should already be sorted, but ensure)
            sort_idx = np.argsort(times_entity)
            y_entity = y_entity[sort_idx]
            X_entity = X_entity[sort_idx]
            times_entity = times_entity[sort_idx]
            indices_sorted = indices[sort_idx]

            # Compute first differences (drop first observation)
            if len(y_entity) >= 2:
                y_diff_entity = y_entity[1:] - y_entity[:-1]
                X_diff_entity = X_entity[1:] - X_entity[:-1]
                times_diff_entity = times_entity[1:]  # Use time of current period
                entities_diff_entity = np.full(len(y_diff_entity), entity)
                valid_idx_entity = indices_sorted[1:]  # Indices in original data

                # Append to lists
                y_diff_list.append(y_diff_entity)
                X_diff_list.append(X_diff_entity)
                entities_diff_list.append(entities_diff_entity)
                times_diff_list.append(times_diff_entity)
                valid_idx_list.append(valid_idx_entity)

        # Concatenate all entities
        y_diff = np.concatenate(y_diff_list)
        X_diff = np.vstack(X_diff_list)
        entities_diff = np.concatenate(entities_diff_list)
        times_diff = np.concatenate(times_diff_list)
        valid_idx = np.concatenate(valid_idx_list)

        return y_diff, X_diff, entities_diff, times_diff, valid_idx

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate coefficients (implementation of abstract method).

        Returns
        -------
        np.ndarray
            Estimated coefficients
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

        # Remove intercept
        if self.formula_parser.has_intercept:
            X = X[:, 1:]

        # Get identifiers
        entities = self.data.data[self.data.entity_col].values
        times = self.data.data[self.data.time_col].values

        # Apply first difference
        y_diff, X_diff, _, _, _ = self._first_difference(y, X, entities, times)

        # OLS on differenced data
        beta, _, _ = compute_ols(y_diff, X_diff, self.weights)
        return beta
