"""
Base classes for panel data models.

This module provides the base classes that all panel models inherit from,
including linear and nonlinear specifications.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from ..utils.data import check_panel_data, get_panel_info
from ..utils.statistics import (
    compute_confidence_intervals,
    compute_p_values,
    compute_standard_errors,
    compute_t_statistics,
)


class PanelModel(ABC):
    """
    Abstract base class for panel data models.

    This class provides the common interface and functionality
    that all panel models share.

    Parameters
    ----------
    endog : array-like
        Dependent variable
    exog : array-like
        Independent variables
    entity_id : array-like, optional
        Entity identifiers
    time_id : array-like, optional
        Time identifiers
    weights : array-like, optional
        Observation weights
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series, pd.DataFrame],
        exog: Union[np.ndarray, pd.DataFrame],
        entity_id: Optional[Union[np.ndarray, pd.Series]] = None,
        time_id: Optional[Union[np.ndarray, pd.Series]] = None,
        weights: Optional[Union[np.ndarray, pd.Series]] = None,
    ):
        """Initialize panel model."""
        # Validate and process data
        self.endog, self.exog, self.entity_id, self.time_id, self.weights = check_panel_data(
            endog, exog, entity_id, time_id, weights
        )

        # Get panel info
        self.panel_info = get_panel_info(self.entity_id, self.time_id)

        # Model attributes
        self.n_obs = len(self.endog)
        self.n_params = self.exog.shape[1]
        self.fitted = False
        self.results = None

    @abstractmethod
    def fit(self, **kwargs):
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, params=None, exog=None):
        """Generate predictions."""
        pass


class NonlinearPanelModel(PanelModel):
    """
    Base class for nonlinear panel data models.

    This class provides common functionality for models
    estimated via maximum likelihood or nonlinear methods.
    """

    def __init__(self, *args, **kwargs):
        """Initialize nonlinear panel model."""
        super().__init__(*args, **kwargs)
        self.method = "MLE"

    @abstractmethod
    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood.

        Parameters
        ----------
        params : ndarray
            Model parameters

        Returns
        -------
        float
            Log-likelihood value
        """
        pass

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood.

        Default implementation uses numerical differentiation.

        Parameters
        ----------
        params : ndarray
            Model parameters

        Returns
        -------
        ndarray
            Gradient vector
        """
        # Numerical gradient
        eps = 1e-8
        grad = np.zeros_like(params)

        for i in range(len(params)):
            params_plus = params.copy()
            params_plus[i] += eps

            params_minus = params.copy()
            params_minus[i] -= eps

            grad[i] = (self._log_likelihood(params_plus) - self._log_likelihood(params_minus)) / (
                2 * eps
            )

        return grad

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix of log-likelihood.

        Default implementation uses numerical differentiation.

        Parameters
        ----------
        params : ndarray
            Model parameters

        Returns
        -------
        ndarray
            Hessian matrix
        """
        # Numerical Hessian
        eps = 1e-8
        k = len(params)
        hess = np.zeros((k, k))

        for i in range(k):
            for j in range(k):
                params_pp = params.copy()
                params_pp[i] += eps
                params_pp[j] += eps

                params_pm = params.copy()
                params_pm[i] += eps
                params_pm[j] -= eps

                params_mp = params.copy()
                params_mp[i] -= eps
                params_mp[j] += eps

                params_mm = params.copy()
                params_mm[i] -= eps
                params_mm[j] -= eps

                hess[i, j] = (
                    self._log_likelihood(params_pp)
                    - self._log_likelihood(params_pm)
                    - self._log_likelihood(params_mp)
                    + self._log_likelihood(params_mm)
                ) / (4 * eps * eps)

        return hess

    def fit(self, start_params=None, method="BFGS", maxiter=1000, **kwargs):
        """
        Fit the model using maximum likelihood.

        Parameters
        ----------
        start_params : array-like, optional
            Starting values for parameters
        method : str
            Optimization method
        maxiter : int
            Maximum number of iterations
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        PanelModelResults
            Fitted model results
        """
        from scipy import optimize

        # Get starting values
        if start_params is None:
            # Try to get starting values from subclass
            if hasattr(self, "_get_start_params"):
                start_params = self._get_start_params()
            else:
                # Default: zeros
                start_params = np.zeros(self.n_params)

        # Optimize
        result = optimize.minimize(
            lambda p: -self._log_likelihood(p),
            start_params,
            method=method,
            jac=lambda p: -self._gradient(p) if hasattr(self, "_gradient") else None,
            options={"maxiter": maxiter, "disp": False},
            **kwargs,
        )

        # Store results
        self.params = result.x
        self.llf = -result.fun

        # Compute covariance matrix
        hessian = self._hessian(self.params)
        try:
            self.vcov = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            self.vcov = -np.linalg.pinv(hessian)

        # Create and return results object
        return PanelModelResults(self, self.params, self.vcov)


class PanelModelResults:
    """
    Results container for panel model estimation.

    This class stores and provides access to model results,
    including parameters, standard errors, and test statistics.

    Parameters
    ----------
    model : PanelModel
        The fitted model instance
    params : ndarray
        Estimated parameters
    vcov : ndarray
        Variance-covariance matrix
    """

    def __init__(self, model: PanelModel, params: np.ndarray, vcov: np.ndarray):
        """Initialize results object."""
        self.model = model
        self.params = params
        self.vcov = vcov

        # Compute standard errors and test statistics
        self.se = compute_standard_errors(vcov)
        self.tvalues = compute_t_statistics(params, self.se)
        self.pvalues = compute_p_values(self.tvalues)

        # Confidence intervals
        self.conf_int_lower, self.conf_int_upper = compute_confidence_intervals(params, self.se)

    def summary(self) -> str:
        """
        Generate summary of model results.

        Returns
        -------
        str
            Formatted summary string
        """
        summary = f"""
Model Results
=============
Number of Obs: {self.model.n_obs}
Number of Parameters: {len(self.params)}

Parameter Estimates:
-------------------
"""
        for i, param in enumerate(self.params):
            summary += f"  Param {i}: {param:.4f} "
            summary += f"(SE: {self.se[i]:.4f}, "
            summary += f"t: {self.tvalues[i]:.2f}, "
            summary += f"p: {self.pvalues[i]:.4f})\n"

        return summary

    def predict(self, exog: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """
        Generate predictions using fitted parameters.

        Parameters
        ----------
        exog : ndarray, optional
            Exogenous variables for prediction

        Returns
        -------
        ndarray
            Predictions
        """
        return self.model.predict(self.params, exog, **kwargs)
