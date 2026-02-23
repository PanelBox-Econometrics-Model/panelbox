"""
Fixed Effects (Within) estimator for panel data.

This module provides the Fixed Effects estimator which removes entity-specific
(and optionally time-specific) fixed effects through demeaning.
"""

from __future__ import annotations

import logging

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

logger = logging.getLogger(__name__)


class FixedEffects(PanelModel):
    """
    Fixed Effects (Within) estimator for panel data.

    Removes unobserved entity-specific (and optionally time-specific) fixed
    effects through demeaning (within transformation). This is equivalent to
    including entity (and time) dummy variables, but more computationally
    efficient.

    The model estimated is:

        y_it = α_i + γ_t + X_it β + ε_it

    where α_i are entity fixed effects and γ_t are time fixed effects
    (if time_effects=True). The within transformation removes these effects
    by demeaning:

        (y_it - ȳ_i) = (X_it - X̄_i) β + (ε_it - ε̄_i)

    **Important:** Time-invariant variables are automatically dropped from
    the model as they are absorbed by the fixed effects.

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format (one row per entity-time observation)
    entity_col : str
        Name of the column identifying entities (e.g., 'firm', 'country')
    time_col : str
        Name of the column identifying time periods (e.g., 'year', 'quarter')
    entity_effects : bool, default=True
        Include entity fixed effects (one-way FE if time_effects=False)
    time_effects : bool, default=False
        Include time fixed effects (two-way FE if entity_effects=True)
    weights : np.ndarray, optional
        Observation weights for WLS estimation

    Attributes
    ----------
    entity_effects : bool
        Whether entity fixed effects are included
    time_effects : bool
        Whether time fixed effects are included
    entity_fe : pd.Series, optional
        Estimated entity fixed effects (populated after fit())
    time_fe : pd.Series, optional
        Estimated time fixed effects (populated after fit())
    formula_parser : FormulaParser
        Parsed formula object
    data : PanelData
        Panel data container

    Examples
    --------
    >>> import panelbox as pb
    >>> from panelbox.datasets import load_grunfeld
    >>>
    >>> # Load example data
    >>> data = load_grunfeld()
    >>>
    >>> # One-way fixed effects (entity only)
    >>> model = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    >>> results = model.fit(cov_type="clustered")
    >>> print(results.summary())
    >>>
    >>> # Two-way fixed effects (entity + time)
    >>> model_twoway = pb.FixedEffects(
    ...     "invest ~ value + capital", data, "firm", "year", entity_effects=True, time_effects=True
    ... )
    >>> results_twoway = model_twoway.fit()
    >>>
    >>> # Access estimated fixed effects
    >>> print(f"Entity FE: {model.entity_fe.head()}")

    Notes
    -----
    **When to Use:**

    Fixed Effects is appropriate when:

    - Unobserved entity heterogeneity exists and is correlated with regressors
    - You want to control for time-invariant confounders
    - Strict exogeneity holds: E[ε_it | X_i, α_i] = 0

    **Advantages:**

    - Consistent under correlation between α_i and X_it
    - Does not require assumptions about distribution of α_i
    - Controls for all time-invariant unobserved factors

    **Limitations:**

    - Cannot estimate coefficients on time-invariant variables
    - Inefficient if Random Effects assumptions hold
    - May amplify measurement error in differenced data
    - Requires T ≥ 2 observations per entity

    **R-squared Interpretation:**

    - `rsquared_within`: R² for demeaned (within) model
    - `rsquared_between`: R² for entity means
    - `rsquared_overall`: Overall R² including fixed effects

    The within R² is the most relevant for FE models.

    **Standard Error Options:**

    Supports the same 9 types as PooledOLS. Clustered standard errors
    (`cov_type='clustered'`) are recommended to account for within-entity
    correlation remaining after fixed effects.

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Chapter 10.
    .. [2] Baltagi, B. H. (2021). Econometric Analysis of Panel Data
           (6th ed.). Springer. Chapter 2.
    .. [3] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics:
           Methods and Applications. Cambridge University Press. Chapter 21.

    See Also
    --------
    RandomEffects : Random Effects (GLS) estimator
    PooledOLS : Pooled OLS without fixed effects
    FirstDifferences : First differences estimator (alternative to FE)
    HausmanTest : Test for choosing between FE and RE
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        entity_effects: bool = True,
        time_effects: bool = False,
        weights: np.ndarray | None = None,
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
        self.entity_fe: pd.Series | None = None
        self.time_fe: pd.Series | None = None

    def _apply_demeaning(
        self,
        y_orig: np.ndarray,
        X_orig: np.ndarray,
        entities: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply the within transformation (demeaning) based on effect type."""
        if self.entity_effects and self.time_effects:
            y = self._demean_both(y_orig.reshape(-1, 1), entities, times).ravel()
            X = self._demean_both(X_orig, entities, times)
        elif self.entity_effects:
            y = demean_matrix(y_orig.reshape(-1, 1), entities).ravel()
            X = demean_matrix(X_orig, entities)
        else:
            y = demean_matrix(y_orig.reshape(-1, 1), times).ravel()
            X = demean_matrix(X_orig, times)
        return y, X

    def _extract_group_fe(
        self,
        overall_resid: np.ndarray,
        group_ids: np.ndarray,
        prior_fe_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Extract centered fixed effects for one grouping dimension.

        Returns (fe_array, centered_values, fe_mean, unique_groups).
        """
        unique_groups = np.unique(group_ids)
        residual = overall_resid - prior_fe_array
        fe_values = np.array([residual[group_ids == g].mean() for g in unique_groups])
        fe_mean = fe_values.mean()
        fe_centered = fe_values - fe_mean
        fe_array = np.zeros(len(overall_resid))
        for i, g in enumerate(unique_groups):
            fe_array[group_ids == g] = fe_centered[i]
        return fe_array, fe_centered, fe_mean, unique_groups

    def _compute_fe_vcov(
        self,
        cov_type: str,
        X: np.ndarray,
        resid_demeaned: np.ndarray,
        df_resid: int,
        entities: np.ndarray,
        times: np.ndarray,
        **cov_kwds,
    ) -> np.ndarray:
        """Compute covariance matrix for the fixed effects estimator."""
        cov_type_lower = cov_type.lower()

        if cov_type_lower == "nonrobust":
            return compute_vcov_nonrobust(X, resid_demeaned, df_resid)

        if cov_type_lower in ["robust", "hc0", "hc1", "hc2", "hc3"]:
            method = "HC1" if cov_type_lower == "robust" else cov_type_lower.upper()
            return robust_covariance(X, resid_demeaned, method=method).cov_matrix

        if cov_type_lower == "clustered":
            return cluster_by_entity(X, resid_demeaned, entities, df_correction=True).cov_matrix

        if cov_type_lower == "twoway":
            return twoway_cluster(X, resid_demeaned, entities, times, df_correction=True).cov_matrix

        if cov_type_lower == "driscoll_kraay":
            max_lags = cov_kwds.get("max_lags")
            kernel = cov_kwds.get("kernel", "bartlett")
            return driscoll_kraay(
                X, resid_demeaned, times, max_lags=max_lags, kernel=kernel
            ).cov_matrix

        if cov_type_lower == "newey_west":
            max_lags = cov_kwds.get("max_lags")
            kernel = cov_kwds.get("kernel", "bartlett")
            return newey_west(X, resid_demeaned, max_lags=max_lags, kernel=kernel).cov_matrix

        if cov_type_lower == "pcse":
            return pcse(X, resid_demeaned, entities, times).cov_matrix

        raise ValueError(
            f"cov_type must be one of: 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3', "
            f"'clustered', 'twoway', 'driscoll_kraay', 'newey_west', 'pcse', got '{cov_type}'"
        )

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
        >>> results = model.fit(cov_type="nonrobust")

        >>> # Heteroskedasticity-robust
        >>> results = model.fit(cov_type="robust")
        >>> results = model.fit(cov_type="hc3")

        >>> # Cluster-robust by entity
        >>> results = model.fit(cov_type="clustered")

        >>> # Two-way clustering
        >>> results = model.fit(cov_type="twoway")

        >>> # Driscoll-Kraay (for spatial/temporal dependence)
        >>> results = model.fit(cov_type="driscoll_kraay", max_lags=3)

        >>> # Newey-West HAC
        >>> results = model.fit(cov_type="newey_west", max_lags=4)

        >>> # Panel-Corrected SE (requires T > N)
        >>> results = model.fit(cov_type="pcse")
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
        y, X = self._apply_demeaning(y_orig, X_orig, entities, times)

        # Estimate coefficients on demeaned data
        beta, resid_demeaned, _fitted_demeaned = compute_ols(y, X, self.weights)

        # Compute fitted values from slopes only
        fitted_from_slopes = (X_orig @ beta).ravel()
        overall_resid = y_orig - fitted_from_slopes

        # Extract fixed effects
        entity_fe_array = np.zeros(len(y_orig))
        time_fe_array = np.zeros(len(y_orig))
        entity_fe_mean = 0.0
        time_fe_mean = 0.0
        entity_fe_values_centered = None
        unique_entities = np.unique(entities)
        time_fe_values_centered = None
        unique_times = np.unique(times)

        if self.entity_effects:
            entity_fe_array, entity_fe_values_centered, entity_fe_mean, unique_entities = (
                self._extract_group_fe(overall_resid, entities, np.zeros(len(y_orig)))
            )

        if self.time_effects:
            time_fe_array, time_fe_values_centered, time_fe_mean, unique_times = (
                self._extract_group_fe(overall_resid, times, entity_fe_array)
            )

        # Intercept term (accumulated from centered FE means)
        intercept = entity_fe_mean + time_fe_mean

        # Full fitted values include slopes + centered FE + intercept
        fitted = fitted_from_slopes + entity_fe_array + time_fe_array + intercept
        resid = y_orig - fitted

        # Degrees of freedom
        n = len(y_orig)
        k = X.shape[1]
        n_fe_entity = self.data.n_entities if self.entity_effects else 0
        n_fe_time = len(unique_times) if self.time_effects else 0
        df_model = k
        df_resid = n - k - n_fe_entity - n_fe_time

        if df_resid <= 0:
            raise ValueError(
                f"Insufficient degrees of freedom: df_resid = {df_resid}. "
                f"n={n}, k={k}, entity FE={n_fe_entity}, time FE={n_fe_time}"
            )

        # Compute covariance matrix (on demeaned data)
        vcov = self._compute_fe_vcov(
            cov_type, X, resid_demeaned, df_resid, entities, times, **cov_kwds
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

        rsquared_dict = {
            "rsquared": rsquared_within,
            "rsquared_adj": rsquared_adj,
            "rsquared_within": rsquared_within,
            "rsquared_between": rsquared_between,
            "rsquared_overall": rsquared_overall,
        }

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
            formula_parser=self.formula_parser,
        )

        # Store fixed effects info on results for predict(newdata)
        results._entity_fe = (
            pd.Series(entity_fe_values_centered, index=unique_entities, name="entity_fe")
            if self.entity_effects
            else None
        )
        results._time_fe = (
            pd.Series(time_fe_values_centered, index=unique_times, name="time_fe")
            if self.time_effects
            else None
        )
        results._entity_col = self.data.entity_col
        results._time_col = self.data.time_col
        results._entity_effects = self.entity_effects
        results._time_effects = self.time_effects
        results._intercept = intercept

        # Compute F-statistic for testing Fixed Effects vs Pooled OLS
        if self.entity_effects:
            f_stat, f_pval = self._compute_f_test_pooled_vs_fe(
                y_orig, X_orig, resid, n_fe_entity, k, n
            )
            results.f_statistic = f_stat
            results.f_pvalue = f_pval

        # Store results and update state
        self._results = results
        self._fitted = True

        return results

    def _compute_f_test_pooled_vs_fe(
        self,
        y: np.ndarray,
        X: np.ndarray,
        resid_fe: np.ndarray,
        n_fe_entity: int,
        k: int,
        n: int,
    ) -> tuple[float, float]:
        """
        Compute F-statistic for testing Fixed Effects vs Pooled OLS.

        Tests the null hypothesis that all entity fixed effects are equal
        (i.e., pooled OLS is adequate) against the alternative that at least
        one entity effect differs (i.e., Fixed Effects is needed).

        H₀: α₁ = α₂ = ... = αₙ (all entity effects are equal)
        H₁: At least one αᵢ differs

        Parameters
        ----------
        y : np.ndarray
            Dependent variable (original, not demeaned)
        X : np.ndarray
            Independent variables (original, not demeaned)
        resid_fe : np.ndarray
            Residuals from Fixed Effects model
        n_fe_entity : int
            Number of entity fixed effects
        k : int
            Number of slope parameters
        n : int
            Number of observations

        Returns
        -------
        tuple[float, float]
            F-statistic and p-value

        Notes
        -----
        The F-statistic is computed as:

        F = (SSR_pooled - SSR_fe) / (N - 1) / (SSR_fe / (NT - N - K))

        where:
        - SSR_pooled: Sum of squared residuals from Pooled OLS
        - SSR_fe: Sum of squared residuals from Fixed Effects
        - N: Number of entities
        - T: Number of time periods
        - K: Number of slope parameters

        The test statistic follows an F-distribution with df1 = N-1 and
        df2 = NT - N - K degrees of freedom under the null hypothesis.

        References
        ----------
        Wooldridge (2010), Econometric Analysis of Cross Section and Panel Data,
        Section 10.4.3
        """
        # Compute SSR for Fixed Effects model
        ssr_fe = np.sum(resid_fe**2)

        # Fit Pooled OLS to get SSR
        # Add intercept if not present
        X_pooled = np.column_stack([np.ones(n), X]) if not np.allclose(X[:, 0], 1) else X

        # OLS: β = (X'X)^(-1) X'y
        beta_pooled = np.linalg.lstsq(X_pooled, y, rcond=None)[0]
        fitted_pooled = X_pooled @ beta_pooled
        resid_pooled = y - fitted_pooled
        ssr_pooled = np.sum(resid_pooled**2)

        # Degrees of freedom
        df1 = n_fe_entity - 1  # Restrictions: N-1 (one entity effect is normalized)
        df2 = n - n_fe_entity - k  # Residual df for FE model

        # F-statistic
        if df1 <= 0 or df2 <= 0:
            return np.nan, np.nan

        f_stat = ((ssr_pooled - ssr_fe) / df1) / (ssr_fe / df2)

        # P-value from F-distribution
        from scipy import stats

        f_pval = 1 - stats.f.cdf(f_stat, df1, df2)

        return float(f_stat), float(f_pval)

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

            # Center fixed effects (identification constraint: mean = 0)
            entity_fe_values = np.array(entity_fe_values)
            entity_fe_values_centered = entity_fe_values - entity_fe_values.mean()

            self.entity_fe = pd.Series(
                entity_fe_values_centered, index=unique_entities, name="entity_fe"
            )

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

            # Center fixed effects (identification constraint: mean = 0)
            time_fe_values = np.array(time_fe_values)
            time_fe_values_centered = time_fe_values - time_fe_values.mean()

            self.time_fe = pd.Series(time_fe_values_centered, index=unique_times, name="time_fe")

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
        X.shape[1]

        # HC1: adjustment factor n/(n-k)
        adjustment = n / df_resid

        # Bread: (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)

        # Meat: X' diag(resid^2) X
        meat = X.T @ (resid[:, np.newaxis] ** 2 * X)

        # Sandwich
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return np.asarray(vcov)

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
        len(resid)
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
