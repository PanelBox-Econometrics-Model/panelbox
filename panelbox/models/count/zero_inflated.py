"""
Zero-inflated count models for panel data.

This module implements zero-inflated models for count data with excess zeros:
- Zero-Inflated Poisson (ZIP)
- Zero-Inflated Negative Binomial (ZINB)

These models are two-part models that combine:
1. A binary model for structural zeros (logit/probit)
2. A count model for the count process (Poisson/NB)

The probability structure is:
P(y=0) = π + (1-π) × P_count(0)
P(y=k) = (1-π) × P_count(k) for k > 0

where π is the probability of a structural zero.

References
----------
Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application
    to Defects in Manufacturing." Technometrics, 34(1), 1-14.
Hall, D. B. (2000). "Zero-inflated Poisson and binomial regression with random
    effects: a case study." Biometrics, 56(4), 1030-1039.
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

from panelbox.models.base import NonlinearPanelModel, PanelModelResults


class ZeroInflatedPoisson(NonlinearPanelModel):
    """
    Zero-Inflated Poisson model for count data with excess zeros.

    This is a two-part model:
    1. Binary model for structural zeros (logit)
    2. Poisson model for the count process

    Parameters
    ----------
    endog : array-like or pd.Series
        Dependent variable (count data)
    exog_count : array-like or pd.DataFrame
        Regressors for the count model
    exog_inflate : array-like or pd.DataFrame, optional
        Regressors for the inflation model. If None, uses exog_count
    entity_col : str, optional
        Name of entity/individual identifier column
    time_col : str, optional
        Name of time identifier column

    Attributes
    ----------
    n_count_params : int
        Number of parameters in count model
    n_inflate_params : int
        Number of parameters in inflation model
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog_count: Union[np.ndarray, pd.DataFrame],
        exog_inflate: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        entity_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """Initialize ZIP model."""
        super().__init__(endog, exog_count, entity_col, time_col)

        # Store count model regressors
        self.exog = self._prepare_data(exog_count)

        # Inflation model regressors
        if exog_inflate is None:
            self.exog_inflate = self.exog.copy()
        else:
            self.exog_inflate = self._prepare_data(exog_inflate)

        # Parameter counts
        self.n_count_params = self.exog.shape[1]
        self.n_inflate_params = self.exog_inflate.shape[1]
        self.n_params = self.n_count_params + self.n_inflate_params

        # Check for non-negative integers
        if np.any(self.endog < 0):
            raise ValueError("Count data must be non-negative")
        if not np.allclose(self.endog, self.endog.astype(int)):
            warnings.warn("Count data contains non-integer values")

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for ZIP model.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [beta_count, gamma_inflate]

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params :]

        # Linear predictors
        xb_count = self.exog @ beta
        xb_inflate = self.exog_inflate @ gamma

        # Probabilities
        pi = 1 / (1 + np.exp(-xb_inflate))  # Inflation probability
        lambda_ = np.exp(xb_count)  # Poisson parameter

        # Log-likelihood
        y = self.endog
        zero_mask = y == 0

        # For zeros: log(π + (1-π)exp(-λ))
        ll_zero = np.log(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-lambda_[zero_mask]))

        # For non-zeros: log((1-π)) + log(Poisson(y|λ))
        ll_nonzero = (
            np.log(1 - pi[~zero_mask])
            + y[~zero_mask] * np.log(lambda_[~zero_mask])
            - lambda_[~zero_mask]
            - gammaln(y[~zero_mask] + 1)
        )

        # Total log-likelihood
        ll = np.sum(ll_zero) + np.sum(ll_nonzero)

        return -ll  # Return negative for minimization

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute analytical gradient of log-likelihood.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector

        Returns
        -------
        np.ndarray
            Gradient vector
        """
        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params :]

        # Linear predictors
        xb_count = self.exog @ beta
        xb_inflate = self.exog_inflate @ gamma

        # Probabilities
        pi = 1 / (1 + np.exp(-xb_inflate))
        lambda_ = np.exp(xb_count)

        y = self.endog
        zero_mask = y == 0

        # Gradient for count parameters (beta)
        # For zeros
        p0 = np.exp(-lambda_)
        denom_zero = pi + (1 - pi) * p0
        w_zero = (1 - pi) * p0 / denom_zero

        grad_beta_zero = (
            -w_zero[zero_mask][:, np.newaxis]
            * lambda_[zero_mask][:, np.newaxis]
            * self.exog[zero_mask]
        )

        # For non-zeros
        grad_beta_nonzero = (
            (y[~zero_mask] / lambda_[~zero_mask] - 1)[:, np.newaxis]
            * lambda_[~zero_mask][:, np.newaxis]
            * self.exog[~zero_mask]
        )

        grad_beta = np.sum(grad_beta_zero, axis=0) + np.sum(grad_beta_nonzero, axis=0)

        # Gradient for inflation parameters (gamma)
        # For zeros
        v_zero = (1 - p0[zero_mask]) / denom_zero[zero_mask]
        grad_gamma_zero = (
            v_zero[:, np.newaxis]
            * pi[zero_mask][:, np.newaxis]
            * (1 - pi[zero_mask])[:, np.newaxis]
            * self.exog_inflate[zero_mask]
        )

        # For non-zeros
        grad_gamma_nonzero = -pi[~zero_mask][:, np.newaxis] * self.exog_inflate[~zero_mask]

        grad_gamma = np.sum(grad_gamma_zero, axis=0) + np.sum(grad_gamma_nonzero, axis=0)

        # Combine gradients (negative for minimization)
        return -np.concatenate([grad_beta, grad_gamma])

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "ZeroInflatedPoissonResult":
        """
        Estimate ZIP model parameters.

        Parameters
        ----------
        start_params : np.ndarray, optional
            Starting values for optimization
        method : str, default='BFGS'
            Optimization method
        maxiter : int, default=1000
            Maximum iterations
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        ZeroInflatedPoissonResult
            Fitted model results
        """
        # Starting values if not provided
        if start_params is None:
            # Use Poisson regression for count part
            from panelbox.models.count import PooledPoisson

            poisson_model = PooledPoisson(self.endog, self.exog)
            poisson_result = poisson_model.fit()
            beta_start = poisson_result.params

            # Use constant-only logit for inflation part
            gamma_start = np.zeros(self.n_inflate_params)
            prop_zeros = np.mean(self.endog == 0)
            if prop_zeros > 0.01 and prop_zeros < 0.99:
                # Initial value for constant in logit
                gamma_start[0] = np.log(prop_zeros / (1 - prop_zeros))

            start_params = np.concatenate([beta_start, gamma_start])

        # Optimize
        result = minimize(
            fun=self.log_likelihood,
            x0=start_params,
            jac=self.gradient,
            method=method,
            options={"maxiter": maxiter},
            **kwargs,
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Create result object
        return ZeroInflatedPoissonResult(
            model=self,
            params=result.params,
            llf=-result.fun,
            converged=result.success,
            iterations=result.nit,
        )

    def predict(
        self,
        params: np.ndarray,
        exog: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        which: str = "mean",
    ) -> np.ndarray:
        """
        Predict using fitted model.

        Parameters
        ----------
        params : np.ndarray
            Model parameters
        exog : array-like, optional
            New data for prediction
        which : str, default='mean'
            What to predict:
            - 'mean': Expected value E[y]
            - 'prob-zero': Probability of zero
            - 'prob-zero-structural': Probability of structural zero

        Returns
        -------
        np.ndarray
            Predictions
        """
        if exog is None:
            exog_count = self.exog
            exog_inflate = self.exog_inflate
        else:
            # Handle new data
            exog_count = self._prepare_data(exog)
            exog_inflate = exog_count  # Simplified for now

        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params :]

        # Predictions
        pi = 1 / (1 + np.exp(-exog_inflate @ gamma))
        lambda_ = np.exp(exog_count @ beta)

        if which == "mean":
            # E[y] = (1-π) × λ
            return (1 - pi) * lambda_
        elif which == "prob-zero":
            # P(y=0) = π + (1-π)exp(-λ)
            return pi + (1 - pi) * np.exp(-lambda_)
        elif which == "prob-zero-structural":
            # P(structural zero) = π
            return pi
        else:
            raise ValueError(f"Unknown prediction type: {which}")


class ZeroInflatedPoissonResult(PanelModelResults):
    """Results class for Zero-Inflated Poisson model."""

    def __init__(self, model, params, llf, converged, iterations):
        """Initialize ZIP results."""
        super().__init__(model, params, llf)
        self.converged = converged
        self.iterations = iterations

        # Split parameters
        self.params_count = params[: model.n_count_params]
        self.params_inflate = params[model.n_count_params :]

        # Compute standard errors
        self._compute_standard_errors()

        # Compute fit statistics
        self._compute_fit_statistics()

    def _compute_standard_errors(self):
        """Compute standard errors via Hessian."""
        try:
            from scipy.optimize import approx_fprime

            # Numerical Hessian
            eps = 1e-5
            hess = np.zeros((self.model.n_params, self.model.n_params))

            for i in range(self.model.n_params):

                def grad_i(params):
                    return self.model.gradient(params)[i]

                hess[i, :] = approx_fprime(self.params, grad_i, eps)

            # Invert Hessian for covariance matrix
            self.cov_params = np.linalg.inv(hess)
            self.bse = np.sqrt(np.diag(self.cov_params))

            # Split standard errors
            self.bse_count = self.bse[: self.model.n_count_params]
            self.bse_inflate = self.bse[self.model.n_count_params :]

        except Exception as e:
            warnings.warn(f"Could not compute standard errors: {e}")
            self.bse = np.full(len(self.params), np.nan)
            self.cov_params = np.full((len(self.params), len(self.params)), np.nan)

    def _compute_fit_statistics(self):
        """Compute goodness-of-fit statistics."""
        # Predictions
        y_pred = self.model.predict(self.params, which="mean")
        y = self.model.endog

        # Proportion of zeros
        self.actual_zeros = np.mean(y == 0)
        self.predicted_zeros = np.mean(self.model.predict(self.params, which="prob-zero"))

        # AIC/BIC
        n = len(y)
        k = len(self.params)
        self.aic = 2 * k - 2 * self.llf
        self.bic = np.log(n) * k - 2 * self.llf

        # Vuong test statistic (ZIP vs Poisson)
        self._compute_vuong_test()

    def _compute_vuong_test(self):
        """
        Compute Vuong test for ZIP vs standard Poisson.

        Tests whether ZIP significantly improves over Poisson.
        """
        try:
            from panelbox.models.count import PooledPoisson

            # Fit standard Poisson
            poisson_model = PooledPoisson(self.model.endog, self.model.exog)
            poisson_result = poisson_model.fit()

            # Log-likelihood contributions
            y = self.model.endog

            # ZIP log-likelihood per observation
            pi = 1 / (1 + np.exp(-self.model.exog_inflate @ self.params_inflate))
            lambda_zip = np.exp(self.model.exog @ self.params_count)

            ll_zip = np.where(
                y == 0,
                np.log(pi + (1 - pi) * np.exp(-lambda_zip)),
                np.log(1 - pi) + y * np.log(lambda_zip) - lambda_zip - gammaln(y + 1),
            )

            # Poisson log-likelihood per observation
            lambda_pois = np.exp(self.model.exog @ poisson_result.params)
            ll_pois = y * np.log(lambda_pois) - lambda_pois - gammaln(y + 1)

            # Vuong statistic
            m = ll_zip - ll_pois
            v_stat = np.sqrt(len(y)) * np.mean(m) / np.std(m, ddof=1)

            self.vuong_stat = v_stat
            self.vuong_pvalue = 2 * (1 - stats.norm.cdf(abs(v_stat)))

        except Exception as e:
            warnings.warn(f"Could not compute Vuong test: {e}")
            self.vuong_stat = np.nan
            self.vuong_pvalue = np.nan

    def summary(self) -> str:
        """
        Generate summary of results.

        Returns
        -------
        str
            Formatted summary
        """
        summary = []
        summary.append("=" * 70)
        summary.append("Zero-Inflated Poisson Model Results")
        summary.append("=" * 70)

        # Model info
        summary.append(f"Number of observations: {len(self.model.endog)}")
        summary.append(f"Log-likelihood: {self.llf:.4f}")
        summary.append(f"AIC: {self.aic:.4f}")
        summary.append(f"BIC: {self.bic:.4f}")
        summary.append(f"Converged: {self.converged}")
        summary.append("")

        # Zero proportions
        summary.append(f"Actual proportion of zeros: {self.actual_zeros:.4f}")
        summary.append(f"Predicted proportion of zeros: {self.predicted_zeros:.4f}")
        summary.append("")

        # Vuong test
        if not np.isnan(self.vuong_stat):
            summary.append("Vuong test (ZIP vs Poisson):")
            summary.append(f"  Statistic: {self.vuong_stat:.4f}")
            summary.append(f"  p-value: {self.vuong_pvalue:.4f}")
            if self.vuong_pvalue < 0.05:
                summary.append("  ZIP is preferred over standard Poisson")
            summary.append("")

        # Count model parameters
        summary.append("-" * 70)
        summary.append("Count Model (Poisson)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<8} {'P>|z|':<8}")

        for i in range(self.model.n_count_params):
            if not np.isnan(self.bse_count[i]):
                z_stat = self.params_count[i] / self.bse_count[i]
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                summary.append(
                    f"{'X' + str(i):<20} {self.params_count[i]:<12.4f} "
                    f"{self.bse_count[i]:<12.4f} {z_stat:<8.3f} {p_val:<8.3f}"
                )
            else:
                summary.append(
                    f"{'X' + str(i):<20} {self.params_count[i]:<12.4f} "
                    f"{'NA':<12} {'NA':<8} {'NA':<8}"
                )

        summary.append("")

        # Inflation model parameters
        summary.append("-" * 70)
        summary.append("Zero-Inflation Model (Logit)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<8} {'P>|z|':<8}")

        for i in range(self.model.n_inflate_params):
            if not np.isnan(self.bse_inflate[i]):
                z_stat = self.params_inflate[i] / self.bse_inflate[i]
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                summary.append(
                    f"{'Z' + str(i):<20} {self.params_inflate[i]:<12.4f} "
                    f"{self.bse_inflate[i]:<12.4f} {z_stat:<8.3f} {p_val:<8.3f}"
                )
            else:
                summary.append(
                    f"{'Z' + str(i):<20} {self.params_inflate[i]:<12.4f} "
                    f"{'NA':<12} {'NA':<8} {'NA':<8}"
                )

        summary.append("=" * 70)

        return "\n".join(summary)


class ZeroInflatedNegativeBinomial(NonlinearPanelModel):
    """
    Zero-Inflated Negative Binomial model for overdispersed count data with excess zeros.

    This is a two-part model:
    1. Binary model for structural zeros (logit)
    2. Negative Binomial model for the count process

    Parameters
    ----------
    endog : array-like or pd.Series
        Dependent variable (count data)
    exog_count : array-like or pd.DataFrame
        Regressors for the count model
    exog_inflate : array-like or pd.DataFrame, optional
        Regressors for the inflation model. If None, uses exog_count
    entity_col : str, optional
        Name of entity/individual identifier column
    time_col : str, optional
        Name of time identifier column
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog_count: Union[np.ndarray, pd.DataFrame],
        exog_inflate: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        entity_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """Initialize ZINB model."""
        super().__init__(endog, exog_count, entity_col, time_col)

        # Store count model regressors
        self.exog = self._prepare_data(exog_count)

        # Inflation model regressors
        if exog_inflate is None:
            self.exog_inflate = self.exog.copy()
        else:
            self.exog_inflate = self._prepare_data(exog_inflate)

        # Parameter counts (include alpha for NB)
        self.n_count_params = self.exog.shape[1]
        self.n_inflate_params = self.exog_inflate.shape[1]
        self.n_params = self.n_count_params + self.n_inflate_params + 1  # +1 for alpha

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for ZINB model.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector [beta_count, gamma_inflate, log_alpha]

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params : self.n_count_params + self.n_inflate_params]
        log_alpha = params[-1]
        alpha = np.exp(log_alpha)  # Ensure alpha > 0

        # Linear predictors
        xb_count = self.exog @ beta
        xb_inflate = self.exog_inflate @ gamma

        # Probabilities
        pi = 1 / (1 + np.exp(-xb_inflate))  # Inflation probability
        mu = np.exp(xb_count)  # NB mean parameter

        # NB parameters
        size = 1 / alpha  # NB size parameter
        prob = size / (size + mu)  # NB prob parameter

        # Log-likelihood
        y = self.endog
        zero_mask = y == 0

        # For zeros: log(π + (1-π) × NB(0|μ,α))
        nb_zero_prob = (size / (size + mu[zero_mask])) ** size
        ll_zero = np.log(pi[zero_mask] + (1 - pi[zero_mask]) * nb_zero_prob)

        # For non-zeros: log((1-π)) + log(NB(y|μ,α))
        ll_nonzero = (
            np.log(1 - pi[~zero_mask])
            + gammaln(y[~zero_mask] + size)
            - gammaln(y[~zero_mask] + 1)
            - gammaln(size)
            + size * np.log(prob[~zero_mask])
            + y[~zero_mask] * np.log(1 - prob[~zero_mask])
        )

        # Total log-likelihood
        ll = np.sum(ll_zero) + np.sum(ll_nonzero)

        return -ll  # Return negative for minimization

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        maxiter: int = 1000,
        **kwargs,
    ) -> "ZeroInflatedNegativeBinomialResult":
        """
        Estimate ZINB model parameters.

        Parameters
        ----------
        start_params : np.ndarray, optional
            Starting values for optimization
        method : str, default='L-BFGS-B'
            Optimization method
        maxiter : int, default=1000
            Maximum iterations
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        ZeroInflatedNegativeBinomialResult
            Fitted model results
        """
        # Starting values if not provided
        if start_params is None:
            # Use ZIP for starting values
            zip_model = ZeroInflatedPoisson(self.endog, self.exog, self.exog_inflate)
            zip_result = zip_model.fit()

            # Add starting value for log(alpha)
            start_params = np.concatenate([zip_result.params, [0.0]])  # log(alpha) = 0 => alpha = 1

        # Optimize with bounds for alpha
        bounds = [(None, None)] * (self.n_params - 1) + [(-10, 10)]  # Bounds for log(alpha)

        result = minimize(
            fun=self.log_likelihood,
            x0=start_params,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter},
            **kwargs,
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Create result object
        return ZeroInflatedNegativeBinomialResult(
            model=self,
            params=result.params,
            llf=-result.fun,
            converged=result.success,
            iterations=result.nit,
        )


class ZeroInflatedNegativeBinomialResult(PanelModelResults):
    """Results class for Zero-Inflated Negative Binomial model."""

    def __init__(self, model, params, llf, converged, iterations):
        """Initialize ZINB results."""
        super().__init__(model, params, llf)
        self.converged = converged
        self.iterations = iterations

        # Split parameters
        self.params_count = params[: model.n_count_params]
        self.params_inflate = params[
            model.n_count_params : model.n_count_params + model.n_inflate_params
        ]
        self.log_alpha = params[-1]
        self.alpha = np.exp(self.log_alpha)

        # Compute fit statistics
        self._compute_fit_statistics()

    def _compute_fit_statistics(self):
        """Compute goodness-of-fit statistics."""
        y = self.model.endog

        # AIC/BIC
        n = len(y)
        k = len(self.params)
        self.aic = 2 * k - 2 * self.llf
        self.bic = np.log(n) * k - 2 * self.llf

        # Proportion of zeros
        self.actual_zeros = np.mean(y == 0)

    def summary(self) -> str:
        """
        Generate summary of results.

        Returns
        -------
        str
            Formatted summary
        """
        summary = []
        summary.append("=" * 70)
        summary.append("Zero-Inflated Negative Binomial Model Results")
        summary.append("=" * 70)

        # Model info
        summary.append(f"Number of observations: {len(self.model.endog)}")
        summary.append(f"Log-likelihood: {self.llf:.4f}")
        summary.append(f"AIC: {self.aic:.4f}")
        summary.append(f"BIC: {self.bic:.4f}")
        summary.append(f"Alpha (overdispersion): {self.alpha:.4f}")
        summary.append(f"Converged: {self.converged}")
        summary.append("")

        # Zero proportions
        summary.append(f"Actual proportion of zeros: {self.actual_zeros:.4f}")
        summary.append("")

        # Count model parameters
        summary.append("-" * 70)
        summary.append("Count Model (Negative Binomial)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12}")

        for i in range(self.model.n_count_params):
            summary.append(f"{'X' + str(i):<20} {self.params_count[i]:<12.4f}")

        summary.append("")

        # Inflation model parameters
        summary.append("-" * 70)
        summary.append("Zero-Inflation Model (Logit)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12}")

        for i in range(self.model.n_inflate_params):
            summary.append(f"{'Z' + str(i):<20} {self.params_inflate[i]:<12.4f}")

        summary.append("=" * 70)

        return "\n".join(summary)
