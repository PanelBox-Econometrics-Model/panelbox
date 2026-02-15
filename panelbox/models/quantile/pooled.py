"""
Pooled quantile regression model for panel data.

This module implements pooled quantile regression, which pools all observations
and estimates quantile coefficients, with optional cluster-robust standard errors.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd

from panelbox.models.quantile.base import (
    ConvergenceWarning,
    QuantileRegressionModel,
    QuantileRegressionResults,
)
from panelbox.optimization.quantile import interior_point_qr
from panelbox.standard_errors import cluster_by_entity

if TYPE_CHECKING:
    pass


class PooledQuantile(QuantileRegressionModel):
    """
    Pooled Quantile Regression model for panel data.

    This model pools all observations across entities and time periods,
    estimating quantile regression coefficients while allowing for
    cluster-robust standard errors by entity.

    The model estimated is:

        Q_τ(y_it | X_it) = X_it' β_τ

    where Q_τ is the conditional τ-th quantile and β_τ are the quantile
    coefficients specific to quantile level τ.

    Parameters
    ----------
    endog : array-like
        Dependent variable (n_obs,)
    exog : array-like
        Independent variables (n_obs, n_vars)
    entity_id : array-like, optional
        Entity identifiers for clustering standard errors
    time_id : array-like, optional
        Time identifiers
    quantiles : float or array-like, default 0.5
        Quantile level(s) to estimate, must be in (0, 1)
    weights : array-like, optional
        Observation weights

    Attributes
    ----------
    params : ndarray
        Estimated quantile coefficients
    vcov : ndarray
        Variance-covariance matrix
    std_errors : ndarray
        Standard errors
    tvalues : ndarray
        t-statistics
    pvalues : ndarray
        p-values
    converged : bool
        Whether optimization converged

    Examples
    --------
    >>> import panelbox as pb
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> n_obs = 500
    >>> y = np.random.randn(n_obs)
    >>> X = np.column_stack([np.ones(n_obs), np.random.randn(n_obs, 3)])
    >>> entity_id = np.repeat(np.arange(50), 10)
    >>>
    >>> # Single quantile
    >>> model = pb.PooledQuantile(y, X, entity_id=entity_id, quantiles=0.5)
    >>> results = model.fit()
    >>> print(results.summary())
    >>>
    >>> # Multiple quantiles
    >>> model_multi = pb.PooledQuantile(y, X, entity_id=entity_id,
    ...                                  quantiles=[0.25, 0.5, 0.75])
    >>> results_multi = model_multi.fit()

    Notes
    -----
    **Computation:**

    The interior point method is used to minimize the check loss function.
    For n observations and p parameters, typical convergence occurs in <2
    seconds for n=1000, p=10.

    **Standard Errors:**

    Default uses cluster-robust standard errors by entity to account for
    within-entity correlation. Set `se_type='nonrobust'` for classical SEs.

    **References**

    Koenker, R. (2005). Quantile Regression. Cambridge University Press.

    Angrist, J. D., & Pischke, J. S. (2009). Mostly Harmless Econometrics:
    An Empiricist's Companion. Princeton University Press.
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog: Union[np.ndarray, pd.DataFrame],
        entity_id: Optional[Union[np.ndarray, pd.Series]] = None,
        time_id: Optional[Union[np.ndarray, pd.Series]] = None,
        quantiles: Union[float, np.ndarray] = 0.5,
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        """Initialize pooled quantile regression model."""
        super().__init__(endog, exog, entity_id, time_id, quantiles, weights)

        # Store parameter names if available
        if isinstance(exog, pd.DataFrame):
            self.param_names = exog.columns.tolist()
        else:
            self.param_names = None

    def fit(
        self,
        method: str = "interior_point",
        maxiter: int = 1000,
        tol: float = 1e-6,
        se_type: str = "cluster",
        alpha: float = 0.05,
        **kwargs,
    ) -> PooledQuantileResults:
        """
        Fit the pooled quantile regression model.

        Parameters
        ----------
        method : str, default 'interior_point'
            Optimization method. Options: 'interior_point', 'gradient_descent'
        maxiter : int, default 1000
            Maximum number of iterations
        tol : float, default 1e-6
            Convergence tolerance
        se_type : str, default 'cluster'
            Standard error type. Options:
            - 'cluster': Cluster-robust by entity
            - 'robust': Heteroskedasticity-robust
            - 'nonrobust': Classical standard errors
        alpha : float, default 0.05
            Significance level for confidence intervals
        **kwargs
            Additional arguments passed to optimizer

        Returns
        -------
        PooledQuantileResults
            Results object with fitted parameters and inference

        Raises
        ------
        ConvergenceWarning
            If optimization does not converge
        """
        X = self.exog
        y = self.endog
        n_obs, n_vars = X.shape

        # Initialize parameters
        params_list = []
        vcov_list = []

        # Fit for each quantile
        for tau in self.quantiles:
            # Get initial parameters (OLS)
            try:
                params_init = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                params_init = np.zeros(n_vars)

            # Fit using interior point method
            params, success = interior_point_qr(
                y, X, tau=tau, params_init=params_init, maxiter=maxiter, tol=tol, **kwargs
            )

            if not success:
                warnings.warn(f"Optimization did not converge for τ={tau}", ConvergenceWarning)
                self.converged = False
            else:
                self.converged = True

            # Compute standard errors
            if se_type == "cluster" and self.entity_id is not None:
                vcov = self._compute_cluster_vcov(params, tau, X, y, n_obs, n_vars)
            elif se_type == "robust":
                vcov = self._compute_robust_vcov(params, tau, X, y, n_obs, n_vars)
            else:
                vcov = self._compute_nonrobust_vcov(params, tau, X, y, n_obs, n_vars)

            params_list.append(params)
            vcov_list.append(vcov)

        # Stack results
        params_arr = np.column_stack(params_list)
        if len(vcov_list) == 1:
            vcov_arr = vcov_list[0]
        else:
            vcov_arr = np.dstack(vcov_list)

        # Create results object
        return PooledQuantileResults(self, params_arr, vcov_arr, self.quantiles)

    def _compute_nonrobust_vcov(
        self, params: np.ndarray, tau: float, X: np.ndarray, y: np.ndarray, n_obs: int, n_vars: int
    ) -> np.ndarray:
        """
        Compute classical (non-robust) variance-covariance matrix.

        Parameters
        ----------
        params : ndarray
            Parameter estimates
        tau : float
            Quantile level
        X : ndarray
            Design matrix
        y : ndarray
            Dependent variable
        n_obs : int
            Number of observations
        n_vars : int
            Number of variables

        Returns
        -------
        ndarray
            Variance-covariance matrix
        """
        # Compute residuals
        residuals = y - X @ params

        # Estimate sparsity parameter (bandwidth)
        # Using method from Koenker (2005): fit local linear regression
        h = self._estimate_sparsity(residuals, tau)

        # Compute meat of sandwich
        X_residuals = X * (residuals[:, np.newaxis] ** 2)
        meat = X_residuals.T @ X

        # Compute bread
        bread_inv = np.linalg.pinv(X.T @ X)

        # Assemble sandwich
        vcov = (1 / (n_obs * h**2)) * (bread_inv @ meat @ bread_inv)

        return vcov

    def _compute_robust_vcov(
        self, params: np.ndarray, tau: float, X: np.ndarray, y: np.ndarray, n_obs: int, n_vars: int
    ) -> np.ndarray:
        """
        Compute heteroskedasticity-robust variance-covariance matrix.

        Uses a heteroskedasticity-consistent covariance matrix estimator.

        Parameters
        ----------
        params : ndarray
            Parameter estimates
        tau : float
            Quantile level
        X : ndarray
            Design matrix
        y : ndarray
            Dependent variable
        n_obs : int
            Number of observations
        n_vars : int
            Number of variables

        Returns
        -------
        ndarray
            Variance-covariance matrix
        """
        # Compute residuals and indicator
        residuals = y - X @ params
        indicator = (residuals < 0).astype(float)

        # Heteroskedasticity in quantile regression
        h = self._estimate_sparsity(residuals, tau)

        # Standardized residuals
        standardized = (indicator - tau) / h

        # Compute variance
        X_stand = X * standardized[:, np.newaxis]
        meat = X_stand.T @ X_stand

        # Compute bread
        bread_inv = np.linalg.pinv(X.T @ X)

        # Assemble sandwich
        vcov = (1 / n_obs) * (bread_inv @ meat @ bread_inv)

        return vcov

    def _compute_cluster_vcov(
        self, params: np.ndarray, tau: float, X: np.ndarray, y: np.ndarray, n_obs: int, n_vars: int
    ) -> np.ndarray:
        """
        Compute cluster-robust variance-covariance matrix.

        Uses cluster-robust covariance matrix estimator with clustering
        by entity.

        Parameters
        ----------
        params : ndarray
            Parameter estimates
        tau : float
            Quantile level
        X : ndarray
            Design matrix
        y : ndarray
            Dependent variable
        n_obs : int
            Number of observations
        n_vars : int
            Number of variables

        Returns
        -------
        ndarray
            Variance-covariance matrix
        """
        # Compute residuals
        residuals = y - X @ params

        # Estimate sparsity
        h = self._estimate_sparsity(residuals, tau)

        # Compute influence function
        indicator = (residuals < 0).astype(float)
        influence = (tau - indicator) / h

        # Cluster-robust variance
        # Group by entity
        unique_entities = np.unique(self.entity_id)
        n_clusters = len(unique_entities)

        # Initialize meat matrix
        meat = np.zeros((n_vars, n_vars))

        # Sum over clusters
        for entity in unique_entities:
            idx = self.entity_id == entity
            X_cluster = X[idx]
            influence_cluster = influence[idx]

            # Cluster contribution
            grad_cluster = X_cluster.T @ influence_cluster
            meat += np.outer(grad_cluster, grad_cluster)

        # Compute bread
        bread_inv = np.linalg.pinv(X.T @ X)

        # Assemble sandwich with cluster adjustment
        # Degree of freedom adjustment
        # Degree of freedom adjustment (with protection for small samples)
        denom = n_clusters - n_vars
        if denom <= 0:
            dof_adj = 1.0
        else:
            dof_adj = (n_clusters - 1) / denom
        vcov = dof_adj * (bread_inv @ meat @ bread_inv)

        return vcov

    @staticmethod
    def _estimate_sparsity(residuals: np.ndarray, tau: float, h: Optional[float] = None) -> float:
        """
        Estimate sparsity parameter (density at the quantile).

        Parameters
        ----------
        residuals : ndarray
            Model residuals
        tau : float
            Quantile level
        h : float, optional
            Bandwidth for kernel density estimation. If None, uses
            rule of thumb: h = z_τ / (2 * φ(z_τ)) * n^{-1/5}

        Returns
        -------
        float
            Estimated sparsity parameter
        """
        from scipy import stats

        n = len(residuals)

        if h is None:
            # Default bandwidth using rule of thumb
            z_tau = stats.norm.ppf(tau)
            phi_z = stats.norm.pdf(z_tau)
            h = z_tau / (2 * phi_z) * n ** (-0.2)

        # Local linear density estimation at 0
        # Count residuals within ±h
        count = np.sum(np.abs(residuals) <= h)

        # Estimate density
        if count > 0:
            f_hat = count / (n * 2 * h)
        else:
            # Fallback: use normal approximation
            f_hat = stats.norm.pdf(0) / np.std(residuals)

        return 1.0 / (f_hat + 1e-6)

    def predict(
        self,
        params: Optional[np.ndarray] = None,
        exog: Optional[np.ndarray] = None,
        quantile_idx: int = 0,
    ) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        params : ndarray, optional
            Parameter vector. If None, raises error (call fit first).
        exog : ndarray, optional
            Design matrix for prediction. If None, uses training data.
        quantile_idx : int, default 0
            Index of quantile for prediction (if multiple quantiles)

        Returns
        -------
        ndarray
            Predicted values

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if params is None:
            raise ValueError("Model must be fitted first")

        if exog is None:
            exog = self.exog

        if params.ndim == 1:
            pred = exog @ params
        else:
            pred = exog @ params[:, quantile_idx]

        return pred


class PooledQuantileResults(QuantileRegressionResults):
    """
    Results object for pooled quantile regression.

    Extends base QuantileRegressionResults with additional methods
    specific to pooled quantile models.

    Parameters
    ----------
    model : PooledQuantile
        The fitted pooled quantile model
    params : ndarray
        Estimated parameters
    vcov : ndarray
        Variance-covariance matrix
    quantiles : array-like
        Quantile levels
    """

    def __init__(
        self, model: PooledQuantile, params: np.ndarray, vcov: np.ndarray, quantiles: np.ndarray
    ):
        """Initialize results object."""
        super().__init__(model, params, vcov, quantiles)
        self.converged = model.converged

    def predict(self, exog: Optional[np.ndarray] = None, quantile_idx: int = 0) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        exog : ndarray, optional
            Design matrix for prediction
        quantile_idx : int, default 0
            Index of quantile for prediction

        Returns
        -------
        ndarray
            Predictions
        """
        if exog is None:
            exog = self.model.exog

        if self.params.ndim == 1:
            pred = exog @ self.params
        else:
            pred = exog @ self.params[:, quantile_idx]

        return pred

    def summary(self) -> str:
        """
        Generate formatted summary table.

        Returns
        -------
        str
            Summary string
        """
        return super().summary()
