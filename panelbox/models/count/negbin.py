"""
Negative Binomial models for panel count data.

This module implements Negative Binomial regression models for panel data,
including pooled, fixed effects, and random effects specifications.
The NB2 parameterization is used where Var(y) = mu + alpha * mu^2.
"""

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import digamma, gammaln, polygamma

from ...utils.data import check_panel_data
from ...utils.statistics import (
    compute_sandwich_covariance,
    compute_standard_errors,
    likelihood_ratio_test,
)
from ..base import NonlinearPanelModel, PanelModelResults
from ..count.poisson import PooledPoisson


class NegativeBinomial(NonlinearPanelModel):
    """
    Negative Binomial regression for panel data.

    Implements the NB2 parameterization where:
    Var(y) = mu + alpha * mu^2

    If alpha = 0, the model reduces to Poisson.

    Parameters
    ----------
    endog : array-like
        Dependent variable (count data)
    exog : array-like
        Independent variables
    entity_id : array-like, optional
        Entity identifiers for panel structure
    time_id : array-like, optional
        Time identifiers for panel structure
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
        """Initialize Negative Binomial model."""
        super().__init__(endog, exog, entity_id, time_id, weights)

        # Model-specific attributes
        self.alpha = None  # Overdispersion parameter
        self.link = "log"  # Log link function

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute NB2 log-likelihood.

        Parameters
        ----------
        params : ndarray
            Parameters [beta, log_alpha]

        Returns
        -------
        float
            Log-likelihood value
        """
        # Split parameters
        beta = params[:-1]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)  # Ensure positivity

        # Linear predictor
        eta = self.exog @ beta
        mu = np.exp(eta)

        # NB2 log-likelihood
        y = self.endog
        r = 1 / alpha  # Shape parameter

        # Use gammaln for numerical stability
        llf = np.sum(
            gammaln(y + r)
            - gammaln(y + 1)
            - gammaln(r)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        )

        # Add weights if provided
        if self.weights is not None:
            llf = np.sum(self.weights * llf)

        return llf

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient of log-likelihood.

        Parameters
        ----------
        params : ndarray
            Parameters [beta, log_alpha]

        Returns
        -------
        ndarray
            Gradient vector
        """
        # Split parameters
        beta = params[:-1]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)

        # Predictions
        eta = self.exog @ beta
        mu = np.exp(eta)
        y = self.endog
        r = 1 / alpha

        # Gradient w.r.t. beta
        residual = (y - mu) / (1 + alpha * mu)
        grad_beta = self.exog.T @ (residual * mu)

        # Gradient w.r.t. log(alpha)
        # Using chain rule: d/d(log_alpha) = alpha * d/d(alpha)
        term1 = digamma(y + r) - digamma(r)
        term2 = -np.log(1 + alpha * mu)
        term3 = (r - y) * alpha * mu / (1 + alpha * mu)

        grad_log_alpha = alpha * np.sum(-(r**2) * (term1 + term2) + r * term3)

        # Combine gradients
        gradient = np.concatenate([grad_beta, [grad_log_alpha]])

        # Weight if needed
        if self.weights is not None:
            gradient *= np.sum(self.weights)

        return gradient

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix of log-likelihood.

        Parameters
        ----------
        params : ndarray
            Parameters [beta, log_alpha]

        Returns
        -------
        ndarray
            Hessian matrix
        """
        k = len(params)
        hessian = np.zeros((k, k))

        # Split parameters
        beta = params[:-1]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)

        # Predictions
        eta = self.exog @ beta
        mu = np.exp(eta)
        y = self.endog
        r = 1 / alpha

        # Hessian w.r.t. beta (upper-left block)
        w = mu * (r + y) / ((1 + alpha * mu) ** 2 * r)
        W = np.diag(w)
        hessian[:-1, :-1] = -self.exog.T @ W @ self.exog

        # Mixed derivatives (off-diagonal)
        # These are more complex for NB, using numerical approximation
        eps = 1e-8
        for i in range(len(beta)):
            params_plus = params.copy()
            params_plus[i] += eps
            grad_plus = self._gradient(params_plus)

            params_minus = params.copy()
            params_minus[i] -= eps
            grad_minus = self._gradient(params_minus)

            hessian[i, -1] = (grad_plus[-1] - grad_minus[-1]) / (2 * eps)
            hessian[-1, i] = hessian[i, -1]

        # Hessian w.r.t. log_alpha (bottom-right)
        # Using numerical derivative
        params_plus = params.copy()
        params_plus[-1] += eps
        grad_plus = self._gradient(params_plus)

        params_minus = params.copy()
        params_minus[-1] -= eps
        grad_minus = self._gradient(params_minus)

        hessian[-1, -1] = (grad_plus[-1] - grad_minus[-1]) / (2 * eps)

        return hessian

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "NegativeBinomialResults":
        """
        Fit the Negative Binomial model.

        Parameters
        ----------
        start_params : ndarray, optional
            Starting values for parameters
        method : str
            Optimization method
        maxiter : int
            Maximum iterations
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        NegativeBinomialResults
            Fitted model results
        """
        # Get starting values
        if start_params is None:
            # Start with Poisson estimates
            poisson = PooledPoisson(
                self.endog, self.exog, self.entity_id, self.time_id, self.weights
            )
            poisson_result = poisson.fit()

            # Start alpha at 0.1
            start_params = np.append(poisson_result.params, np.log(0.1))

        # Optimize
        result = optimize.minimize(
            lambda p: -self._log_likelihood(p),
            start_params,
            method=method,
            jac=lambda p: -self._gradient(p),
            options={"maxiter": maxiter},
        )

        # Store alpha
        self.alpha = np.exp(result.x[-1])

        # Compute covariance matrix
        hessian = self._hessian(result.x)
        vcov = -np.linalg.inv(hessian)

        # Create results object
        return NegativeBinomialResults(
            self, result.x, vcov, llf=-result.fun, converged=result.success
        )

    def predict(
        self,
        params: Optional[np.ndarray] = None,
        exog: Optional[np.ndarray] = None,
        which: str = "mean",
    ) -> np.ndarray:
        """
        Predict using the model.

        Parameters
        ----------
        params : ndarray, optional
            Parameters to use for prediction
        exog : ndarray, optional
            Exogenous variables for prediction
        which : str
            Type of prediction ('mean' or 'linear')

        Returns
        -------
        ndarray
            Predictions
        """
        if exog is None:
            exog = self.exog

        if params is None:
            raise ValueError("Parameters required for prediction")

        # Extract beta (excluding alpha)
        beta = params[:-1] if len(params) > self.exog.shape[1] else params

        # Linear predictor
        eta = exog @ beta

        if which == "linear":
            return eta
        else:  # 'mean'
            return np.exp(eta)


class NegativeBinomialResults(PanelModelResults):
    """Results class for Negative Binomial models."""

    def __init__(self, model, params, vcov, llf=None, converged=True):
        """Initialize NB results."""
        super().__init__(model, params, vcov)
        self.llf = llf
        self.converged = converged
        self.alpha = model.alpha

    @property
    def params_exog(self):
        """Return coefficients excluding alpha."""
        return self.params[:-1]

    def lr_test_poisson(self) -> Dict[str, Any]:
        """
        Likelihood ratio test: Poisson vs Negative Binomial.

        Tests H0: alpha = 0 (Poisson is adequate)
        vs H1: alpha > 0 (NB is needed)

        Returns
        -------
        dict
            Test results with statistic, p-value, conclusion
        """
        # Fit restricted (Poisson) model
        poisson = PooledPoisson(
            self.model.endog,
            self.model.exog,
            self.model.entity_id,
            self.model.time_id,
            self.model.weights,
        )
        poisson_result = poisson.fit()

        # LR test
        return likelihood_ratio_test(
            llf_unrestricted=self.llf, llf_restricted=poisson_result.llf, df=1
        )

    def summary(self):
        """Generate summary of results."""
        summary_str = f"""
Negative Binomial Regression Results
=====================================
Dependent Variable: y
Number of Obs: {len(self.model.endog)}
Log-Likelihood: {self.llf:.4f}
Converged: {self.converged}
Overdispersion (alpha): {self.alpha:.4f}

Parameters:
-----------
"""
        # Add coefficient table
        se = compute_standard_errors(self.vcov)
        for i in range(len(self.params_exog)):
            summary_str += f"beta[{i}]: {self.params_exog[i]:.4f} (SE: {se[i]:.4f})\n"

        # LR test vs Poisson
        lr_test = self.lr_test_poisson()
        summary_str += f"\nLR Test vs Poisson:\n"
        summary_str += f"Statistic: {lr_test['statistic']:.4f}\n"
        summary_str += f"P-value: {lr_test['pvalue']:.4f}\n"
        summary_str += f"Conclusion: {lr_test['conclusion']}\n"

        return summary_str


class FixedEffectsNegativeBinomial(NegativeBinomial):
    """
    Fixed Effects Negative Binomial model.

    Implementation following Allison & Waterman (2002).
    Note: This is not a true fixed effects estimator
    and may not fully control for unobserved heterogeneity.
    """

    def __init__(self, *args, **kwargs):
        """Initialize FE NB model."""
        super().__init__(*args, **kwargs)
        self.model_type = "Fixed Effects Negative Binomial"

    def fit(self, *args, **kwargs):
        """
        Fit FE NB model.

        Note: Adds entity dummies to design matrix.
        """
        # Create entity dummies
        entities = np.unique(self.entity_id)
        n_entities = len(entities)

        if n_entities > 100:
            import warnings

            warnings.warn(f"Large number of entities ({n_entities}). " "Estimation may be slow.")

        # Add entity dummies to exog (excluding one for identification)
        entity_dummies = np.zeros((len(self.endog), n_entities - 1))
        for i, entity in enumerate(entities[:-1]):
            entity_dummies[:, i] = (self.entity_id == entity).astype(float)

        # Augment design matrix
        self.exog_with_fe = np.column_stack([self.exog, entity_dummies])

        # Store original exog and replace temporarily
        original_exog = self.exog
        self.exog = self.exog_with_fe

        # Fit model
        result = super().fit(*args, **kwargs)

        # Restore original exog
        self.exog = original_exog

        # Extract structural parameters only
        k_exog = original_exog.shape[1]
        result.params_exog = result.params[:k_exog]

        return result
