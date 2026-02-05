"""
Fixed Effects (Within) estimator for panel data.

This module provides the Fixed Effects estimator which removes entity-specific
(and optionally time-specific) fixed effects through demeaning.
"""

from typing import Dict, Optional

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
from panelbox.utils.matrix_ops import (
    compute_ols,
    compute_panel_rsquared,
    compute_vcov_nonrobust,
    demean_matrix,
)


class FixedEffects(PanelModel):
    """
    Fixed Effects (Within) estimator for panel data.

    This estimator removes unobserved entity-specific (and optionally time-specific)
    fixed effects through demeaning (within transformation). This is equivalent to
    including entity (and time) dummy variables, but more efficient computationally.

    The within transformation removes time-invariant variables from the model.

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    entity_effects : bool, default=True
        Include entity fixed effects
    time_effects : bool, default=False
        Include time fixed effects
    weights : np.ndarray, optional
        Observation weights

    Attributes
    ----------
    entity_effects : bool
        Whether entity fixed effects are included
    time_effects : bool
        Whether time fixed effects are included
    entity_fe : pd.Series, optional
        Estimated entity fixed effects (after fitting)
    time_fe : pd.Series, optional
        Estimated time fixed effects (after fitting)

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Load data
    >>> data = pd.read_csv('panel_data.csv')
    >>>
    >>> # Entity fixed effects only
    >>> model = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year")
    >>> results = model.fit(cov_type='clustered')
    >>> print(results.summary())
    >>>
    >>> # Two-way fixed effects (entity + time)
    >>> model_twoway = pb.FixedEffects(
    ...     "y ~ x1 + x2", data, "firm", "year",
    ...     entity_effects=True,
    ...     time_effects=True
    ... )
    >>> results_twoway = model_twoway.fit()
    >>>
    >>> # Access estimated fixed effects
    >>> entity_fe = model.entity_fe
    >>> time_fe = model_twoway.time_fe
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        entity_effects: bool = True,
        time_effects: bool = False,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

        self.entity_effects = entity_effects
        self.time_effects = time_effects

        if not entity_effects and not time_effects:
            raise ValueError(
                "At least one of entity_effects or time_effects must be True. "
                "Use PooledOLS if you don't want fixed effects."
            )

        # Fixed effects (computed after fitting)
        self.entity_fe: Optional[pd.Series] = None
        self.time_fe: Optional[pd.Series] = None

    def fit(self, cov_type: str = "nonrobust", **cov_kwds) -> PanelResults:
        """
        Fit the Fixed Effects model.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Type of covariance estimator:
            - 'nonrobust': Classical standard errors
            - 'robust' or 'hc1': Heteroskedasticity-robust (HC1)
            - 'hc0', 'hc2', 'hc3': Other HC variants
            - 'clustered': Cluster-robust (by entity by default)
            - 'twoway': Two-way clustered (entity and time)
            - 'driscoll_kraay': Driscoll-Kraay (spatial/temporal dependence)
            - 'newey_west': Newey-West HAC
            - 'pcse': Panel-Corrected Standard Errors (requires T > N)
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

        >>> # Heteroskedasticity-robust
        >>> results = model.fit(cov_type='robust')
        >>> results = model.fit(cov_type='hc3')

        >>> # Cluster-robust by entity
        >>> results = model.fit(cov_type='clustered')

        >>> # Two-way clustering
        >>> results = model.fit(cov_type='twoway')

        >>> # Driscoll-Kraay (for spatial/temporal dependence)
        >>> results = model.fit(cov_type='driscoll_kraay', max_lags=3)

        >>> # Newey-West HAC
        >>> results = model.fit(cov_type='newey_west', max_lags=4)

        >>> # Panel-Corrected SE (requires T > N)
        >>> results = model.fit(cov_type='pcse')
        """
        # Build design matrices
        y_orig, X_orig = self.formula_parser.build_design_matrices(
            self.data.data, return_type="array"
        )

        # Get variable names before demeaning
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Remove intercept from variable names (FE absorbs it)
        if "Intercept" in var_names:
            var_names = [v for v in var_names if v != "Intercept"]
            # Remove intercept column from X
            X_orig = X_orig[:, 1:]

        # Get entity and time identifiers as arrays
        entities = self.data.data[self.data.entity_col].values
        times = self.data.data[self.data.time_col].values

        # Store original data for fixed effects computation
        self._y_orig = y_orig
        self._X_orig = X_orig
        self._entities = entities
        self._times = times

        # Apply within transformation (demeaning)
        if self.entity_effects and self.time_effects:
            # Two-way demeaning
            y = self._demean_both(y_orig.reshape(-1, 1), entities, times).ravel()
            X = self._demean_both(X_orig, entities, times)
        elif self.entity_effects:
            # Entity demeaning only
            y = demean_matrix(y_orig.reshape(-1, 1), entities).ravel()
            X = demean_matrix(X_orig, entities)
        else:  # time_effects only
            # Time demeaning only
            y = demean_matrix(y_orig.reshape(-1, 1), times).ravel()
            X = demean_matrix(X_orig, times)

        # Estimate coefficients on demeaned data
        beta, resid_demeaned, fitted_demeaned = compute_ols(y, X, self.weights)

        # Compute residuals and fitted values in original scale
        fitted = (X_orig @ beta).ravel()
        resid = (y_orig - fitted).ravel()

        # Degrees of freedom
        n = len(y_orig)
        k = X.shape[1]

        # Account for absorbed fixed effects
        if self.entity_effects:
            n_fe_entity = self.data.n_entities
        else:
            n_fe_entity = 0

        if self.time_effects:
            n_fe_time = len(np.unique(times))
        else:
            n_fe_time = 0

        # df_model: number of slopes (excludes fixed effects and intercept)
        df_model = k

        # df_resid: n - k - n_fixed_effects
        df_resid = n - k - n_fe_entity - n_fe_time

        # Ensure df_resid is positive
        if df_resid <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom: df_resid = {df_resid}. "
                f"n={n}, k={k}, entity FE={n_fe_entity}, time FE={n_fe_time}"
            )

        # Compute covariance matrix (on demeaned data)
        cov_type_lower = cov_type.lower()

        if cov_type_lower == "nonrobust":
            vcov = compute_vcov_nonrobust(X, resid_demeaned, df_resid)

        elif cov_type_lower in ["robust", "hc0", "hc1", "hc2", "hc3"]:
            # Map 'robust' to 'hc1' (default robust method)
            method = "HC1" if cov_type_lower == "robust" else cov_type_lower.upper()
            result = robust_covariance(X, resid_demeaned, method=method)
            vcov = result.cov_matrix

        elif cov_type_lower == "clustered":
            # Default: cluster by entity
            result = cluster_by_entity(X, resid_demeaned, entities, df_correction=True)
            vcov = result.cov_matrix

        elif cov_type_lower == "twoway":
            # Two-way clustering: entity and time
            result = twoway_cluster(X, resid_demeaned, entities, times, df_correction=True)
            vcov = result.cov_matrix

        elif cov_type_lower == "driscoll_kraay":
            # Driscoll-Kraay for spatial/temporal dependence
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = driscoll_kraay(X, resid_demeaned, times, max_lags=max_lags, kernel=kernel)
            vcov = result.cov_matrix

        elif cov_type_lower == "newey_west":
            # Newey-West HAC
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = newey_west(X, resid_demeaned, max_lags=max_lags, kernel=kernel)
            vcov = result.cov_matrix

        elif cov_type_lower == "pcse":
            # Panel-Corrected Standard Errors
            result = pcse(X, resid_demeaned, entities, times)
            vcov = result.cov_matrix

        else:
            raise ValueError(
                f"cov_type must be one of: 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3', "
                f"'clustered', 'twoway', 'driscoll_kraay', 'newey_west', 'pcse', got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Compute panel R-squared measures
        rsquared_within, rsquared_between, rsquared_overall = compute_panel_rsquared(
            y_orig, fitted, resid, entities
        )

        # Adjusted R-squared (within)
        rsquared_adj = 1 - (1 - rsquared_within) * (n - 1) / df_resid

        # Create Series/DataFrame with variable names
        params = pd.Series(beta.ravel(), index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Compute fixed effects
        self._compute_fixed_effects(beta)

        # Model information
        model_type = "Fixed Effects"
        if self.entity_effects and self.time_effects:
            model_type = "Fixed Effects (Two-Way)"
        elif self.time_effects:
            model_type = "Fixed Effects (Time)"

        model_info = {
            "model_type": model_type,
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "entity_effects": self.entity_effects,
            "time_effects": self.time_effects,
        }

        # Data information
        data_info = {
            "nobs": n,
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "n_fe_entity": n_fe_entity if self.entity_effects else 0,
            "n_fe_time": n_fe_time if self.time_effects else 0,
            "entity_index": entities.ravel() if hasattr(entities, "ravel") else entities,
            "time_index": times.ravel() if hasattr(times, "ravel") else times,
        }

        # R-squared dictionary
        rsquared_dict = {
            "rsquared": rsquared_within,  # For FE, R² = within R²
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
            resid=resid,
            fittedvalues=fitted,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self,
        )

        # Store results and update state
        self._results = results
        self._fitted = True

        return results

    def _demean_both(self, X: np.ndarray, entities: np.ndarray, times: np.ndarray) -> np.ndarray:
        """
        Apply two-way demeaning (entity and time).

        Parameters
        ----------
        X : np.ndarray
            Data to demean
        entities : np.ndarray
            Entity identifiers
        times : np.ndarray
            Time identifiers

        Returns
        -------
        np.ndarray
            Two-way demeaned data
        """
        # First demean by entity
        X_entity_demeaned = demean_matrix(X, entities)

        # Then demean by time
        X_both_demeaned = demean_matrix(X_entity_demeaned, times)

        return X_both_demeaned

    def _compute_fixed_effects(self, beta: np.ndarray) -> None:
        """
        Compute estimated fixed effects.

        Parameters
        ----------
        beta : np.ndarray
            Estimated coefficients
        """
        # Fitted values from slope coefficients
        fitted_from_slopes = self._X_orig @ beta

        # Overall residual: y - X*beta
        overall_resid = self._y_orig - fitted_from_slopes

        if self.entity_effects:
            # Entity fixed effects: mean residual by entity
            unique_entities = np.unique(self._entities)
            entity_fe_values = []

            for entity in unique_entities:
                mask = self._entities == entity
                entity_mean_resid = overall_resid[mask].mean()
                entity_fe_values.append(entity_mean_resid)

            self.entity_fe = pd.Series(entity_fe_values, index=unique_entities, name="entity_fe")

        if self.time_effects:
            # Time fixed effects: mean residual by time (after removing entity FE if present)
            if self.entity_effects:
                # Remove entity FE first
                resid_after_entity = overall_resid.copy()
                for i, entity in enumerate(self._entities):
                    resid_after_entity[i] -= self.entity_fe[entity]
                base_resid = resid_after_entity
            else:
                base_resid = overall_resid

            unique_times = np.unique(self._times)
            time_fe_values = []

            for time in unique_times:
                mask = self._times == time
                time_mean_resid = base_resid[mask].mean()
                time_fe_values.append(time_mean_resid)

            self.time_fe = pd.Series(time_fe_values, index=unique_times, name="time_fe")

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

        # Demean
        if self.entity_effects and self.time_effects:
            y_dm = self._demean_both(y.reshape(-1, 1), entities, times).ravel()
            X_dm = self._demean_both(X, entities, times)
        elif self.entity_effects:
            y_dm = demean_matrix(y.reshape(-1, 1), entities).ravel()
            X_dm = demean_matrix(X, entities)
        else:
            y_dm = demean_matrix(y.reshape(-1, 1), times).ravel()
            X_dm = demean_matrix(X, times)

        beta, _, _ = compute_ols(y_dm, X_dm, self.weights)
        return beta

    def _compute_vcov_robust(self, X: np.ndarray, resid: np.ndarray, df_resid: int) -> np.ndarray:
        """
        Compute heteroskedasticity-robust covariance matrix (HC1).

        Parameters
        ----------
        X : np.ndarray
            Design matrix (demeaned)
        resid : np.ndarray
            Residuals (demeaned)
        df_resid : int
            Degrees of freedom

        Returns
        -------
        np.ndarray
            Robust covariance matrix
        """
        n = len(resid)
        k = X.shape[1]

        # HC1: adjustment factor n/(n-k)
        adjustment = n / df_resid

        # Bread: (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)

        # Meat: X' diag(resid^2) X
        meat = X.T @ (resid[:, np.newaxis] ** 2 * X)

        # Sandwich
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return vcov

    def _compute_vcov_clustered(
        self, X: np.ndarray, resid: np.ndarray, entities: np.ndarray, df_resid: int
    ) -> np.ndarray:
        """
        Compute cluster-robust covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (demeaned)
        resid : np.ndarray
            Residuals (demeaned)
        entities : np.ndarray
            Entity identifiers
        df_resid : int
            Degrees of freedom

        Returns
        -------
        np.ndarray
            Cluster-robust covariance matrix
        """
        n = len(resid)
        k = X.shape[1]

        unique_entities = np.unique(entities)
        n_clusters = len(unique_entities)

        # Bread: (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)

        # Meat: sum over clusters
        meat = np.zeros((k, k))
        for entity in unique_entities:
            mask = entities == entity
            X_c = X[mask]
            resid_c = resid[mask]
            score = X_c.T @ resid_c
            meat += np.outer(score, score)

        # Small sample adjustment
        adjustment = (n_clusters / (n_clusters - 1)) * (df_resid / (df_resid - k))

        # Sandwich
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return vcov
