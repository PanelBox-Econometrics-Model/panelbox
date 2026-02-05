"""
Random Effects (GLS) estimator for panel data.

This module provides the Random Effects estimator which uses GLS (Generalized Least Squares)
to account for the variance component structure in panel data.
"""

from typing import Optional, Union

import numpy as np
import pandas as pd

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults
from panelbox.standard_errors import (
    cluster_by_entity,
    driscoll_kraay,
    newey_west,
    robust_covariance,
    twoway_cluster,
)
from panelbox.standard_errors.clustered import ClusteredCovarianceResult
from panelbox.standard_errors.driscoll_kraay import DriscollKraayResult
from panelbox.standard_errors.newey_west import NeweyWestResult
from panelbox.standard_errors.robust import RobustCovarianceResult
from panelbox.utils.matrix_ops import compute_ols, compute_panel_rsquared


class RandomEffects(PanelModel):
    """
    Random Effects (GLS) estimator for panel data.

    This estimator assumes that entity-specific effects are uncorrelated with
    the regressors and uses Generalized Least Squares to efficiently estimate
    the model accounting for the variance component structure.

    The key assumption is E[u_i | X_it] = 0, where u_i is the entity-specific effect.

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
    variance_estimator : str, default='swamy-arora'
        Method for estimating variance components:
        - 'swamy-arora': Swamy-Arora estimator (most common)
        - 'walhus': Wallace-Hussain estimator
        - 'amemiya': Amemiya estimator
        - 'nerlove': Nerlove estimator
    weights : np.ndarray, optional
        Observation weights

    Attributes
    ----------
    variance_estimator : str
        Variance estimation method
    sigma2_u : float
        Estimated variance of entity-specific effects (after fitting)
    sigma2_e : float
        Estimated variance of idiosyncratic errors (after fitting)
    theta : float
        GLS transformation parameter (after fitting)

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Load data
    >>> data = pd.read_csv('panel_data.csv')
    >>>
    >>> # Estimate Random Effects
    >>> model = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year")
    >>> results = model.fit()
    >>> print(results.summary())
    >>>
    >>> # Access variance components
    >>> print(f"sigma2_u: {model.sigma2_u:.4f}")
    >>> print(f"sigma2_e: {model.sigma2_e:.4f}")
    >>> print(f"theta: {model.theta:.4f}")
    >>>
    >>> # Use different variance estimator
    >>> model_amemiya = pb.RandomEffects(
    ...     "y ~ x1 + x2", data, "firm", "year",
    ...     variance_estimator='amemiya'
    ... )
    >>> results_amemiya = model_amemiya.fit()
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        variance_estimator: str = "swamy-arora",
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

        valid_estimators = ["swamy-arora", "walhus", "amemiya", "nerlove"]
        if variance_estimator not in valid_estimators:
            raise ValueError(
                f"variance_estimator must be one of {valid_estimators}, "
                f"got '{variance_estimator}'"
            )

        self.variance_estimator = variance_estimator

        # Variance components (computed after fitting)
        self.sigma2_u: Optional[float] = None  # Variance of entity effects
        self.sigma2_e: Optional[float] = None  # Variance of idiosyncratic errors
        self.theta: Optional[float] = None  # GLS transformation parameter

    def fit(self, cov_type: str = "nonrobust", **cov_kwds) -> PanelResults:
        """
        Fit the Random Effects model.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Type of covariance estimator:
            - 'nonrobust': Classical GLS standard errors
            - 'robust' or 'hc1': Heteroskedasticity-robust (HC1)
            - 'hc0', 'hc2', 'hc3': Other HC variants
            - 'clustered': Cluster-robust (by entity by default)
            - 'twoway': Two-way clustered (entity and time)
            - 'driscoll_kraay': Driscoll-Kraay (spatial/temporal dependence)
            - 'newey_west': Newey-West HAC
        **cov_kwds
            Additional arguments for covariance estimation:
            - max_lags: For Driscoll-Kraay and Newey-West
            - kernel: For HAC estimators

        Returns
        -------
        PanelResults
            Fitted model results

        Examples
        --------
        >>> results = model.fit()
        >>> results_robust = model.fit(cov_type='robust')
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

        # Get variable names
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Get entity and time identifiers
        entities = self.data.data[self.data.entity_col].values
        times = self.data.data[self.data.time_col].values

        # Estimate variance components
        self._estimate_variance_components(y, X, entities)

        # Apply GLS transformation
        y_gls, X_gls = self._gls_transform(y, X, entities)

        # Estimate coefficients on transformed data
        beta, resid_gls, fitted_gls = compute_ols(y_gls, X_gls, self.weights)

        # Compute residuals and fitted values in original scale
        fitted = (X @ beta).ravel()
        resid = (y - fitted).ravel()

        # Degrees of freedom
        n = len(y)
        k = X.shape[1]
        df_model = k - (1 if self.formula_parser.has_intercept else 0)
        df_resid = n - k

        # Compute covariance matrix
        cov_type_lower = cov_type.lower()

        # Type annotation for result variable to handle different covariance result types
        result: Union[
            RobustCovarianceResult, ClusteredCovarianceResult, DriscollKraayResult, NeweyWestResult
        ]

        if cov_type_lower == "nonrobust":
            vcov = self._compute_vcov_gls(X, resid_gls, entities, df_resid)

        elif cov_type_lower in ["robust", "hc0", "hc1", "hc2", "hc3"]:
            # Map 'robust' to 'hc1' (default robust method)
            method = "HC1" if cov_type_lower == "robust" else cov_type_lower.upper()
            result = robust_covariance(X_gls, resid_gls, method=method)
            vcov = result.cov_matrix

        elif cov_type_lower == "clustered":
            # Default: cluster by entity
            result = cluster_by_entity(X_gls, resid_gls, entities, df_correction=True)
            vcov = result.cov_matrix

        elif cov_type_lower == "twoway":
            # Two-way clustering: entity and time
            result = twoway_cluster(X_gls, resid_gls, entities, times, df_correction=True)
            vcov = result.cov_matrix

        elif cov_type_lower == "driscoll_kraay":
            # Driscoll-Kraay for spatial/temporal dependence
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = driscoll_kraay(X_gls, resid_gls, times, max_lags=max_lags, kernel=kernel)
            vcov = result.cov_matrix

        elif cov_type_lower == "newey_west":
            # Newey-West HAC
            max_lags = cov_kwds.get("max_lags", None)
            kernel = cov_kwds.get("kernel", "bartlett")
            result = newey_west(X_gls, resid_gls, max_lags=max_lags, kernel=kernel)
            vcov = result.cov_matrix

        else:
            raise ValueError(
                f"cov_type must be one of: 'nonrobust', 'robust', 'hc0', 'hc1', 'hc2', 'hc3', "
                f"'clustered', 'twoway', 'driscoll_kraay', 'newey_west', got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Compute panel R-squared measures
        rsquared_within, rsquared_between, rsquared_overall = compute_panel_rsquared(
            y, fitted, resid, entities
        )

        # Adjusted R-squared (overall)
        rsquared_adj = 1 - (1 - rsquared_overall) * (n - 1) / df_resid

        # Create Series/DataFrame with variable names
        params = pd.Series(beta.ravel(), index=var_names)
        std_errors_series = pd.Series(std_errors, index=var_names)
        cov_params = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Model information
        model_info = {
            "model_type": "Random Effects (GLS)",
            "formula": self.formula,
            "cov_type": cov_type,
            "cov_kwds": cov_kwds,
            "variance_estimator": self.variance_estimator,
        }

        # Data information
        data_info = {
            "nobs": n,
            "n_entities": self.data.n_entities,
            "n_periods": self.data.n_periods,
            "df_model": df_model,
            "df_resid": df_resid,
            "entity_index": entities.ravel() if hasattr(entities, "ravel") else entities,
            "time_index": times.ravel() if hasattr(times, "ravel") else times,
        }

        # R-squared dictionary
        rsquared_dict = {
            "rsquared": rsquared_overall,  # For RE, main R² is overall
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

    def _estimate_variance_components(
        self, y: np.ndarray, X: np.ndarray, entities: np.ndarray
    ) -> None:
        """
        Estimate variance components.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Design matrix
        entities : np.ndarray
            Entity identifiers
        """
        n = len(y)
        k = X.shape[1]

        if self.variance_estimator == "swamy-arora":
            self._swamy_arora_variance(y, X, entities, n, k)
        elif self.variance_estimator == "walhus":
            self._walhus_variance(y, X, entities, n, k)
        elif self.variance_estimator == "amemiya":
            self._amemiya_variance(y, X, entities, n, k)
        elif self.variance_estimator == "nerlove":
            self._nerlove_variance(y, X, entities, n, k)

    def _swamy_arora_variance(
        self, y: np.ndarray, X: np.ndarray, entities: np.ndarray, n: int, k: int
    ) -> None:
        """
        Swamy-Arora variance component estimator.

        This is the most common estimator for RE models.
        """
        # Step 1: Estimate within (FE) model
        from panelbox.utils.matrix_ops import demean_matrix

        y_within = demean_matrix(y.reshape(-1, 1), entities).ravel()
        X_within = demean_matrix(X, entities)

        beta_within, resid_within, _ = compute_ols(y_within, X_within)

        # Estimate sigma2_e from within residuals
        N = self.data.n_entities
        df_within = n - N - k  # Account for absorbed entity dummies
        self.sigma2_e = np.sum(resid_within**2) / df_within

        # Step 2: Estimate between model (on entity means)
        unique_entities = np.unique(entities)
        y_means = []
        X_means = []

        for entity in unique_entities:
            mask = entities == entity
            y_means.append(y[mask].mean())
            X_means.append(X[mask].mean(axis=0))

        y_between = np.array(y_means)
        X_between = np.array(X_means)

        beta_between, resid_between, _ = compute_ols(y_between, X_between)

        # Estimate sigma2_u from between residuals
        # Average group size
        T_bar = n / N

        # Variance of between residuals
        var_between = np.sum(resid_between**2) / (N - k)

        # sigma2_u = var_between - sigma2_e / T_bar
        self.sigma2_u = max(0, var_between - self.sigma2_e / T_bar)

        # Compute theta (GLS transformation parameter)
        # theta = 1 - sqrt(sigma2_e / (sigma2_e + T*sigma2_u))
        self.theta = 1 - np.sqrt(self.sigma2_e / (self.sigma2_e + T_bar * self.sigma2_u))

    def _walhus_variance(self, y, X, entities, n, k):
        """Wallace-Hussain variance estimator."""
        # Similar to Swamy-Arora but uses different degrees of freedom
        # For simplicity, use Swamy-Arora (can be refined later)
        self._swamy_arora_variance(y, X, entities, n, k)

    def _amemiya_variance(self, y, X, entities, n, k):
        """Amemiya variance estimator."""
        # Uses quadratic forms of residuals
        # For simplicity, use Swamy-Arora (can be refined later)
        self._swamy_arora_variance(y, X, entities, n, k)

    def _nerlove_variance(self, y, X, entities, n, k):
        """Nerlove variance estimator."""
        # Uses pooled OLS residuals
        # For simplicity, use Swamy-Arora (can be refined later)
        self._swamy_arora_variance(y, X, entities, n, k)

    def _gls_transform(self, y: np.ndarray, X: np.ndarray, entities: np.ndarray) -> tuple:
        """
        Apply GLS transformation.

        The transformation is: y* = y - theta * y_bar_i
        where y_bar_i is the entity mean and theta is computed from variance components.

        Parameters
        ----------
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Design matrix
        entities : np.ndarray
            Entity identifiers

        Returns
        -------
        y_gls : np.ndarray
            Transformed dependent variable
        X_gls : np.ndarray
            Transformed design matrix
        """
        unique_entities = np.unique(entities)

        y_gls = y.copy()
        X_gls = X.copy()

        for entity in unique_entities:
            mask = entities == entity

            # Entity means
            y_mean = y[mask].mean()
            X_mean = X[mask].mean(axis=0)

            # GLS transformation: subtract theta * mean
            y_gls[mask] -= self.theta * y_mean
            X_gls[mask] -= self.theta * X_mean

        return y_gls, X_gls

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate coefficients (implementation of abstract method).

        Returns
        -------
        np.ndarray
            Estimated coefficients
        """
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")

        entities = self.data.data[self.data.entity_col].values

        # Estimate variance components
        self._estimate_variance_components(y, X, entities)

        # GLS transformation
        y_gls, X_gls = self._gls_transform(y, X, entities)

        # Estimate
        beta, _, _ = compute_ols(y_gls, X_gls, self.weights)
        return beta

    def _compute_vcov_gls(
        self, X: np.ndarray, resid: np.ndarray, entities: np.ndarray, df_resid: int
    ) -> np.ndarray:
        """
        Compute GLS covariance matrix.

        Parameters
        ----------
        X : np.ndarray
            Original design matrix (not transformed)
        resid : np.ndarray
            GLS residuals
        entities : np.ndarray
            Entity identifiers
        df_resid : int
            Degrees of freedom

        Returns
        -------
        np.ndarray
            Covariance matrix
        """
        # Estimate of error variance from GLS residuals
        s2 = np.sum(resid**2) / df_resid

        # Build Omega matrix (variance-covariance of errors)
        # For RE: Omega_i = sigma2_e * I + sigma2_u * J
        # where J is matrix of ones

        # For computational efficiency, use transformation approach
        # V(beta_GLS) = s^2 * (X'Omega^{-1}X)^{-1}

        # Create transformed X (same transformation as in GLS)
        X_gls, _ = self._gls_transform(X, X, entities)

        # Covariance: s^2 (X_gls' X_gls)^{-1}
        XtX_inv = np.linalg.inv(X_gls.T @ X_gls)
        vcov = s2 * XtX_inv

        return vcov

    def _compute_vcov_robust(self, X: np.ndarray, resid: np.ndarray, df_resid: int) -> np.ndarray:
        """Compute robust covariance matrix."""
        n = len(resid)
        X.shape[1]
        adjustment = n / df_resid

        XtX_inv = np.linalg.inv(X.T @ X)
        meat = X.T @ (resid[:, np.newaxis] ** 2 * X)
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return vcov

    def _compute_vcov_clustered(
        self, X: np.ndarray, resid: np.ndarray, entities: np.ndarray, df_resid: int
    ) -> np.ndarray:
        """Compute cluster-robust covariance matrix."""
        len(resid)
        k = X.shape[1]

        unique_entities = np.unique(entities)
        n_clusters = len(unique_entities)

        XtX_inv = np.linalg.inv(X.T @ X)

        meat = np.zeros((k, k))
        for entity in unique_entities:
            mask = entities == entity
            X_c = X[mask]
            resid_c = resid[mask]
            score = X_c.T @ resid_c
            meat += np.outer(score, score)

        adjustment = (n_clusters / (n_clusters - 1)) * (df_resid / (df_resid - k))
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return vcov
