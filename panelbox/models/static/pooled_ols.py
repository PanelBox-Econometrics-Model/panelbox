"""
Pooled OLS estimator for panel data.

This module provides the Pooled OLS estimator which ignores the panel structure
and estimates a standard OLS regression.
"""

from typing import Optional
import numpy as np
import pandas as pd

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults
from panelbox.utils.matrix_ops import (
    compute_ols,
    compute_vcov_nonrobust,
    compute_rsquared
)


class PooledOLS(PanelModel):
    """
    Pooled OLS estimator for panel data.

    This estimator ignores the panel structure and pools all observations
    together, estimating a standard OLS regression. This is often used as
    a baseline comparison for panel-specific estimators.

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
    weights : np.ndarray, optional
        Observation weights for WLS estimation

    Attributes
    ----------
    All attributes from PanelModel plus model-specific attributes
    after fitting.

    Examples
    --------
    >>> import panelbox as pb
    >>> import pandas as pd
    >>>
    >>> # Load data
    >>> data = pd.read_csv('panel_data.csv')
    >>>
    >>> # Estimate Pooled OLS
    >>> model = pb.PooledOLS("y ~ x1 + x2", data, "firm", "year")
    >>> results = model.fit(cov_type='robust')
    >>> print(results.summary())
    >>>
    >>> # With clustered standard errors
    >>> results_cluster = model.fit(cov_type='clustered')
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: Optional[np.ndarray] = None
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

    def fit(
        self,
        cov_type: str = 'nonrobust',
        **cov_kwds
    ) -> PanelResults:
        """
        Fit the Pooled OLS model.

        Parameters
        ----------
        cov_type : str, default='nonrobust'
            Type of covariance estimator:
            - 'nonrobust': Classical OLS standard errors
            - 'robust': Heteroskedasticity-robust (HC1)
            - 'clustered': Cluster-robust (clustered by entity by default)
        **cov_kwds
            Additional arguments for covariance estimation

        Returns
        -------
        PanelResults
            Fitted model results

        Examples
        --------
        >>> results = model.fit(cov_type='robust')
        >>> results_cluster = model.fit(cov_type='clustered')
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(
            self.data.data,
            return_type='array'
        )

        # Get variable names
        var_names = self.formula_parser.get_variable_names(self.data.data)

        # Estimate coefficients
        beta, resid, fitted = compute_ols(y, X, self.weights)

        # Degrees of freedom
        n = len(y)
        k = X.shape[1]
        df_model = k - (1 if self.formula_parser.has_intercept else 0)
        df_resid = n - k

        # Compute covariance matrix
        if cov_type == 'nonrobust':
            vcov = compute_vcov_nonrobust(X, resid, df_resid)
        elif cov_type == 'robust':
            vcov = self._compute_vcov_robust(X, resid)
        elif cov_type == 'clustered':
            vcov = self._compute_vcov_clustered(X, resid)
        else:
            raise ValueError(
                f"cov_type must be 'nonrobust', 'robust', or 'clustered', "
                f"got '{cov_type}'"
            )

        # Standard errors
        std_errors = np.sqrt(np.diag(vcov))

        # Compute R-squared
        rsquared = compute_rsquared(
            y, fitted, resid,
            has_intercept=self.formula_parser.has_intercept
        )

        # Adjusted R-squared
        rsquared_adj = 1 - (1 - rsquared) * (n - 1) / df_resid

        # Create Series/DataFrame with variable names
        params = pd.Series(beta.ravel(), index=var_names)
        std_errors = pd.Series(std_errors, index=var_names)
        cov_params = pd.DataFrame(vcov, index=var_names, columns=var_names)

        # Model information
        model_info = {
            'model_type': 'Pooled OLS',
            'formula': self.formula,
            'cov_type': cov_type,
            'cov_kwds': cov_kwds
        }

        # Data information
        data_info = {
            'nobs': n,
            'n_entities': self.data.n_entities,
            'n_periods': self.data.n_periods,
            'df_model': df_model,
            'df_resid': df_resid,
            'entity_index': self.data.data[self.data.entity_col].values.ravel(),
            'time_index': self.data.data[self.data.time_col].values.ravel()
        }

        # R-squared dictionary
        rsquared_dict = {
            'rsquared': rsquared,
            'rsquared_adj': rsquared_adj,
            'rsquared_within': np.nan,
            'rsquared_between': np.nan,
            'rsquared_overall': rsquared
        }

        # Create results object
        results = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=resid,
            fittedvalues=fitted,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=rsquared_dict,
            model=self
        )

        # Store results and update state
        self._results = results
        self._fitted = True

        return results

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate coefficients (implementation of abstract method).

        Returns
        -------
        np.ndarray
            Estimated coefficients
        """
        y, X = self.formula_parser.build_design_matrices(
            self.data.data,
            return_type='array'
        )
        beta, _, _ = compute_ols(y, X, self.weights)
        return beta

    def _compute_vcov_robust(
        self,
        X: np.ndarray,
        resid: np.ndarray
    ) -> np.ndarray:
        """
        Compute heteroskedasticity-robust covariance matrix (HC1).

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        resid : np.ndarray
            Residuals

        Returns
        -------
        np.ndarray
            Robust covariance matrix
        """
        n, k = X.shape
        df_resid = n - k

        # HC1: adjustment factor n/(n-k)
        adjustment = n / df_resid

        # Bread: (X'X)^{-1}
        XtX_inv = np.linalg.inv(X.T @ X)

        # Meat: X' diag(resid^2) X
        meat = X.T @ (resid[:, np.newaxis]**2 * X)

        # Sandwich: (X'X)^{-1} * X'ΩX * (X'X)^{-1}
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return vcov

    def _compute_vcov_clustered(
        self,
        X: np.ndarray,
        resid: np.ndarray
    ) -> np.ndarray:
        """
        Compute cluster-robust covariance matrix.

        Clusters by entity by default.

        Parameters
        ----------
        X : np.ndarray
            Design matrix
        resid : np.ndarray
            Residuals

        Returns
        -------
        np.ndarray
            Cluster-robust covariance matrix
        """
        n, k = X.shape

        # Get entity identifiers
        entities = self.data.data[self.data.entity_col].values
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
            # Sum of (X_i * resid_i) for cluster
            score = X_c.T @ resid_c
            meat += np.outer(score, score)

        # Small sample adjustment: G/(G-1) * (N-1)/(N-K)
        adjustment = (n_clusters / (n_clusters - 1)) * ((n - 1) / (n - k))

        # Sandwich
        vcov = adjustment * (XtX_inv @ meat @ XtX_inv)

        return vcov
