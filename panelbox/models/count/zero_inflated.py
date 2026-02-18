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


def _to_array(data):
    """Convert input data to numpy array."""
    if isinstance(data, pd.DataFrame):
        return data.values.astype(float)
    elif isinstance(data, pd.Series):
        return data.values.astype(float)
    elif isinstance(data, np.ndarray):
        return data.astype(float)
    else:
        return np.asarray(data, dtype=float)


class ZeroInflatedPoisson:
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
        Regressors for the count model (should include constant)
    exog_inflate : array-like or pd.DataFrame, optional
        Regressors for the inflation model (should include constant).
        If None, uses exog_count.
    exog_count_names : list of str, optional
        Names for count model variables
    exog_inflate_names : list of str, optional
        Names for inflation model variables
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog_count: Union[np.ndarray, pd.DataFrame],
        exog_inflate: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        exog_count_names: Optional[list] = None,
        exog_inflate_names: Optional[list] = None,
    ):
        """Initialize ZIP model."""
        # Convert to arrays
        self.endog = _to_array(endog).ravel()
        self.exog = _to_array(exog_count)
        if self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)

        # Inflation model regressors
        if exog_inflate is None:
            self.exog_inflate = self.exog.copy()
        else:
            self.exog_inflate = _to_array(exog_inflate)
            if self.exog_inflate.ndim == 1:
                self.exog_inflate = self.exog_inflate.reshape(-1, 1)

        # Store variable names
        if exog_count_names is not None:
            self.exog_count_names = list(exog_count_names)
        elif isinstance(exog_count, pd.DataFrame):
            self.exog_count_names = list(exog_count.columns)
        else:
            self.exog_count_names = [f"X{i}" for i in range(self.exog.shape[1])]

        if exog_inflate_names is not None:
            self.exog_inflate_names = list(exog_inflate_names)
        elif isinstance(exog_inflate, pd.DataFrame):
            self.exog_inflate_names = list(exog_inflate.columns)
        else:
            self.exog_inflate_names = [f"Z{i}" for i in range(self.exog_inflate.shape[1])]

        # Parameter counts
        self.n_count_params = self.exog.shape[1]
        self.n_inflate_params = self.exog_inflate.shape[1]
        self.n_params = self.n_count_params + self.n_inflate_params
        self.n_obs = len(self.endog)

        # Check for non-negative integers
        if np.any(self.endog < 0):
            raise ValueError("Count data must be non-negative")
        if not np.allclose(self.endog, self.endog.astype(int)):
            warnings.warn("Count data contains non-integer values")

    def _neg_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for ZIP model.

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

        # Probabilities with numerical safeguards
        xb_inflate = np.clip(xb_inflate, -30, 30)
        pi = 1 / (1 + np.exp(-xb_inflate))  # Inflation probability
        xb_count = np.clip(xb_count, -30, 30)
        lambda_ = np.exp(xb_count)  # Poisson parameter

        # Log-likelihood
        y = self.endog
        zero_mask = y == 0

        # For zeros: log(π + (1-π)exp(-λ))
        ll_zero = np.log(
            np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * np.exp(-lambda_[zero_mask]), 1e-300)
        )

        # For non-zeros: log((1-π)) + log(Poisson(y|λ))
        ll_nonzero = (
            np.log(np.maximum(1 - pi[~zero_mask], 1e-300))
            + y[~zero_mask] * np.log(np.maximum(lambda_[~zero_mask], 1e-300))
            - lambda_[~zero_mask]
            - gammaln(y[~zero_mask] + 1)
        )

        # Total log-likelihood
        ll = np.sum(ll_zero) + np.sum(ll_nonzero)

        return -ll  # Return negative for minimization

    def _gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute analytical gradient of negative log-likelihood.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector

        Returns
        -------
        np.ndarray
            Gradient vector (of negative log-likelihood)
        """
        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params :]

        # Linear predictors
        xb_count = np.clip(self.exog @ beta, -30, 30)
        xb_inflate = np.clip(self.exog_inflate @ gamma, -30, 30)

        # Probabilities
        pi = 1 / (1 + np.exp(-xb_inflate))
        lambda_ = np.exp(xb_count)

        y = self.endog
        zero_mask = y == 0

        # Gradient for count parameters (beta)
        p0 = np.exp(-lambda_)
        denom_zero = np.maximum(pi + (1 - pi) * p0, 1e-300)
        w_zero = (1 - pi) * p0 / denom_zero

        grad_beta_zero = (
            -w_zero[zero_mask][:, np.newaxis]
            * lambda_[zero_mask][:, np.newaxis]
            * self.exog[zero_mask]
        )

        grad_beta_nonzero = (y[~zero_mask] - lambda_[~zero_mask])[:, np.newaxis] * self.exog[
            ~zero_mask
        ]

        grad_beta = np.sum(grad_beta_zero, axis=0) + np.sum(grad_beta_nonzero, axis=0)

        # Gradient for inflation parameters (gamma)
        v_zero = (1 - p0[zero_mask]) / denom_zero[zero_mask]
        grad_gamma_zero = (
            v_zero[:, np.newaxis]
            * pi[zero_mask][:, np.newaxis]
            * (1 - pi[zero_mask])[:, np.newaxis]
            * self.exog_inflate[zero_mask]
        )

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
            from panelbox.models.count import PooledPoisson

            poisson_model = PooledPoisson(self.endog, self.exog)
            poisson_result = poisson_model.fit(se_type="robust")
            beta_start = poisson_result.params

            # Use constant-only logit for inflation part
            gamma_start = np.zeros(self.n_inflate_params)
            prop_zeros = np.mean(self.endog == 0)
            if 0.01 < prop_zeros < 0.99:
                gamma_start[0] = np.log(prop_zeros / (1 - prop_zeros))

            start_params = np.concatenate([beta_start, gamma_start])

        # Optimize
        result = minimize(
            fun=self._neg_log_likelihood,
            x0=start_params,
            jac=self._gradient,
            method=method,
            options={"maxiter": maxiter},
            **kwargs,
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        # Store log-likelihood for the fitted model
        self.llf = -result.fun

        return ZeroInflatedPoissonResult(
            model=self,
            params=result.x,
            llf=-result.fun,
            converged=result.success,
            iterations=result.nit,
        )

    def predict(
        self,
        params: np.ndarray,
        exog_count: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        exog_inflate: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        which: str = "mean",
    ) -> np.ndarray:
        """
        Predict using fitted model.

        Parameters
        ----------
        params : np.ndarray
            Model parameters
        exog_count : array-like, optional
            New count model data. If None, uses training data.
        exog_inflate : array-like, optional
            New inflation model data. If None, uses training data.
        which : str, default='mean'
            What to predict:
            - 'mean': Expected value E[y] = (1-pi)*lambda
            - 'prob-zero': Total probability of zero P(y=0)
            - 'prob-zero-structural': Probability of structural zero (pi)
            - 'prob-zero-sampling': Probability of sampling zero (1-pi)*exp(-lambda)
            - 'count-mean': Expected count among potential users (lambda)

        Returns
        -------
        np.ndarray
            Predictions
        """
        if exog_count is None:
            exog_c = self.exog
        else:
            exog_c = _to_array(exog_count)

        if exog_inflate is None:
            exog_i = self.exog_inflate
        else:
            exog_i = _to_array(exog_inflate)

        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params :]

        # Predictions
        xb_inflate = np.clip(exog_i @ gamma, -30, 30)
        pi = 1 / (1 + np.exp(-xb_inflate))
        xb_count = np.clip(exog_c @ beta, -30, 30)
        lambda_ = np.exp(xb_count)

        if which == "mean":
            return (1 - pi) * lambda_
        elif which == "prob-zero":
            return pi + (1 - pi) * np.exp(-lambda_)
        elif which == "prob-zero-structural":
            return pi
        elif which == "prob-zero-sampling":
            return (1 - pi) * np.exp(-lambda_)
        elif which == "count-mean":
            return lambda_
        else:
            raise ValueError(f"Unknown prediction type: {which}")


class ZeroInflatedPoissonResult:
    """Results class for Zero-Inflated Poisson model."""

    def __init__(self, model, params, llf, converged, iterations):
        """Initialize ZIP results."""
        self.model = model
        self.params = params
        self.llf = llf
        self.converged = converged
        self.iterations = iterations
        self.n_obs = model.n_obs

        # Split parameters
        self.params_count = params[: model.n_count_params]
        self.params_inflate = params[model.n_count_params :]

        # Compute standard errors
        self._compute_standard_errors()

        # Compute fit statistics
        self._compute_fit_statistics()

    def _compute_standard_errors(self):
        """Compute standard errors via numerical Hessian."""
        try:
            from scipy.optimize import approx_fprime

            eps = 1e-5
            n_params = self.model.n_params
            hess = np.zeros((n_params, n_params))

            for i in range(n_params):

                def grad_i(params, idx=i):
                    return self.model._gradient(params)[idx]

                hess[i, :] = approx_fprime(self.params, grad_i, eps)

            # Symmetrize
            hess = (hess + hess.T) / 2

            # Invert Hessian for covariance matrix
            self.cov_params = np.linalg.inv(hess)
            diag = np.diag(self.cov_params)
            # Handle potential negative diagonal elements
            self.bse = np.sqrt(np.maximum(diag, 0))

            # Split standard errors
            self.bse_count = self.bse[: self.model.n_count_params]
            self.bse_inflate = self.bse[self.model.n_count_params :]

        except Exception as e:
            warnings.warn(f"Could not compute standard errors: {e}")
            self.bse = np.full(len(self.params), np.nan)
            self.bse_count = np.full(self.model.n_count_params, np.nan)
            self.bse_inflate = np.full(self.model.n_inflate_params, np.nan)
            self.cov_params = np.full((len(self.params), len(self.params)), np.nan)

    def _compute_fit_statistics(self):
        """Compute goodness-of-fit statistics."""
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
        """Compute Vuong test for ZIP vs standard Poisson."""
        try:
            from panelbox.models.count import PooledPoisson

            # Fit standard Poisson
            poisson_model = PooledPoisson(self.model.endog, self.model.exog)
            poisson_result = poisson_model.fit(se_type="robust")

            y = self.model.endog

            # ZIP log-likelihood per observation
            pi = 1 / (1 + np.exp(-np.clip(self.model.exog_inflate @ self.params_inflate, -30, 30)))
            lambda_zip = np.exp(np.clip(self.model.exog @ self.params_count, -30, 30))

            ll_zip = np.where(
                y == 0,
                np.log(np.maximum(pi + (1 - pi) * np.exp(-lambda_zip), 1e-300)),
                np.log(np.maximum(1 - pi, 1e-300))
                + y * np.log(np.maximum(lambda_zip, 1e-300))
                - lambda_zip
                - gammaln(y + 1),
            )

            # Poisson log-likelihood per observation
            lambda_pois = np.exp(np.clip(self.model.exog @ poisson_result.params, -30, 30))
            ll_pois = y * np.log(np.maximum(lambda_pois, 1e-300)) - lambda_pois - gammaln(y + 1)

            # Vuong statistic
            m = ll_zip - ll_pois
            sd_m = np.std(m, ddof=1)
            if sd_m > 0:
                v_stat = np.sqrt(len(y)) * np.mean(m) / sd_m
                self.vuong_stat = v_stat
                self.vuong_pvalue = 2 * (1 - stats.norm.cdf(abs(v_stat)))
            else:
                self.vuong_stat = np.nan
                self.vuong_pvalue = np.nan

            # Store the Poisson log-likelihood for comparisons
            self._poisson_llf = poisson_model.llf

        except Exception as e:
            warnings.warn(f"Could not compute Vuong test: {e}")
            self.vuong_stat = np.nan
            self.vuong_pvalue = np.nan

    def summary(self, count_names=None, inflate_names=None) -> str:
        """
        Generate summary of results.

        Parameters
        ----------
        count_names : list of str, optional
            Variable names for count model
        inflate_names : list of str, optional
            Variable names for inflation model

        Returns
        -------
        str
            Formatted summary
        """
        if count_names is None:
            count_names = self.model.exog_count_names
        if inflate_names is None:
            inflate_names = self.model.exog_inflate_names

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
                summary.append("  --> ZIP is preferred over standard Poisson")
            else:
                summary.append("  --> Standard Poisson is adequate")
            summary.append("")

        # Count model parameters
        summary.append("-" * 70)
        summary.append("Count Model (Poisson)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<10} {'P>|z|':<10}")

        for i in range(self.model.n_count_params):
            name = count_names[i] if i < len(count_names) else f"X{i}"
            if not np.isnan(self.bse_count[i]) and self.bse_count[i] > 0:
                z_stat = self.params_count[i] / self.bse_count[i]
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                summary.append(
                    f"{name:<20} {self.params_count[i]:<12.4f} "
                    f"{self.bse_count[i]:<12.4f} {z_stat:<10.3f} {p_val:<10.4f}"
                )
            else:
                summary.append(
                    f"{name:<20} {self.params_count[i]:<12.4f} " f"{'NA':<12} {'NA':<10} {'NA':<10}"
                )

        summary.append("")

        # Inflation model parameters
        summary.append("-" * 70)
        summary.append("Zero-Inflation Model (Logit)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<10} {'P>|z|':<10}")

        for i in range(self.model.n_inflate_params):
            name = inflate_names[i] if i < len(inflate_names) else f"Z{i}"
            if not np.isnan(self.bse_inflate[i]) and self.bse_inflate[i] > 0:
                z_stat = self.params_inflate[i] / self.bse_inflate[i]
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                summary.append(
                    f"{name:<20} {self.params_inflate[i]:<12.4f} "
                    f"{self.bse_inflate[i]:<12.4f} {z_stat:<10.3f} {p_val:<10.4f}"
                )
            else:
                summary.append(
                    f"{name:<20} {self.params_inflate[i]:<12.4f} "
                    f"{'NA':<12} {'NA':<10} {'NA':<10}"
                )

        summary.append("=" * 70)

        return "\n".join(summary)


class ZeroInflatedNegativeBinomial:
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
        Regressors for the count model (should include constant)
    exog_inflate : array-like or pd.DataFrame, optional
        Regressors for the inflation model (should include constant).
        If None, uses exog_count.
    exog_count_names : list of str, optional
        Names for count model variables
    exog_inflate_names : list of str, optional
        Names for inflation model variables
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog_count: Union[np.ndarray, pd.DataFrame],
        exog_inflate: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        exog_count_names: Optional[list] = None,
        exog_inflate_names: Optional[list] = None,
    ):
        """Initialize ZINB model."""
        # Convert to arrays
        self.endog = _to_array(endog).ravel()
        self.exog = _to_array(exog_count)
        if self.exog.ndim == 1:
            self.exog = self.exog.reshape(-1, 1)

        # Inflation model regressors
        if exog_inflate is None:
            self.exog_inflate = self.exog.copy()
        else:
            self.exog_inflate = _to_array(exog_inflate)
            if self.exog_inflate.ndim == 1:
                self.exog_inflate = self.exog_inflate.reshape(-1, 1)

        # Store variable names
        if exog_count_names is not None:
            self.exog_count_names = list(exog_count_names)
        elif isinstance(exog_count, pd.DataFrame):
            self.exog_count_names = list(exog_count.columns)
        else:
            self.exog_count_names = [f"X{i}" for i in range(self.exog.shape[1])]

        if exog_inflate_names is not None:
            self.exog_inflate_names = list(exog_inflate_names)
        elif isinstance(exog_inflate, pd.DataFrame):
            self.exog_inflate_names = list(exog_inflate.columns)
        else:
            self.exog_inflate_names = [f"Z{i}" for i in range(self.exog_inflate.shape[1])]

        # Parameter counts (include alpha for NB)
        self.n_count_params = self.exog.shape[1]
        self.n_inflate_params = self.exog_inflate.shape[1]
        self.n_params = self.n_count_params + self.n_inflate_params + 1  # +1 for alpha
        self.n_obs = len(self.endog)

        # Validate
        if np.any(self.endog < 0):
            raise ValueError("Count data must be non-negative")

    def _neg_log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute negative log-likelihood for ZINB model.

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
        alpha = np.exp(np.clip(log_alpha, -10, 10))

        # Linear predictors with safeguards
        xb_count = np.clip(self.exog @ beta, -30, 30)
        xb_inflate = np.clip(self.exog_inflate @ gamma, -30, 30)

        # Probabilities
        pi = 1 / (1 + np.exp(-xb_inflate))
        mu = np.exp(xb_count)

        # NB parameters
        size = 1 / alpha
        prob = size / (size + mu)

        y = self.endog
        zero_mask = y == 0

        # For zeros: log(π + (1-π) × NB(0|μ,α))
        nb_zero_prob = np.power(prob[zero_mask], size)
        ll_zero = np.log(np.maximum(pi[zero_mask] + (1 - pi[zero_mask]) * nb_zero_prob, 1e-300))

        # For non-zeros: log((1-π)) + log(NB(y|μ,α))
        ll_nonzero = (
            np.log(np.maximum(1 - pi[~zero_mask], 1e-300))
            + gammaln(y[~zero_mask] + size)
            - gammaln(y[~zero_mask] + 1)
            - gammaln(size)
            + size * np.log(np.maximum(prob[~zero_mask], 1e-300))
            + y[~zero_mask] * np.log(np.maximum(1 - prob[~zero_mask], 1e-300))
        )

        ll = np.sum(ll_zero) + np.sum(ll_nonzero)
        return -ll

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

        Returns
        -------
        ZeroInflatedNegativeBinomialResult
            Fitted model results
        """
        if start_params is None:
            # Use ZIP for starting values
            zip_model = ZeroInflatedPoisson(self.endog, self.exog, self.exog_inflate)
            zip_result = zip_model.fit()
            start_params = np.concatenate([zip_result.params, [0.0]])

        # Optimize with bounds for log(alpha)
        bounds = [(None, None)] * (self.n_params - 1) + [(-10, 10)]

        result = minimize(
            fun=self._neg_log_likelihood,
            x0=start_params,
            method=method,
            bounds=bounds,
            options={"maxiter": maxiter},
            **kwargs,
        )

        if not result.success:
            warnings.warn(f"Optimization did not converge: {result.message}")

        self.llf = -result.fun

        return ZeroInflatedNegativeBinomialResult(
            model=self,
            params=result.x,
            llf=-result.fun,
            converged=result.success,
            iterations=result.nit,
        )

    def predict(
        self,
        params: np.ndarray,
        exog_count: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        exog_inflate: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        which: str = "mean",
    ) -> np.ndarray:
        """
        Predict using fitted model.

        Parameters
        ----------
        params : np.ndarray
            Model parameters [beta_count, gamma_inflate, log_alpha]
        exog_count : array-like, optional
            New count model data
        exog_inflate : array-like, optional
            New inflation model data
        which : str, default='mean'
            What to predict:
            - 'mean': Expected value E[y] = (1-pi)*mu
            - 'prob-zero': Total probability of zero
            - 'prob-zero-structural': Probability of structural zero (pi)
            - 'prob-zero-sampling': Probability of sampling zero
            - 'count-mean': Expected count among potential users (mu)

        Returns
        -------
        np.ndarray
            Predictions
        """
        if exog_count is None:
            exog_c = self.exog
        else:
            exog_c = _to_array(exog_count)

        if exog_inflate is None:
            exog_i = self.exog_inflate
        else:
            exog_i = _to_array(exog_inflate)

        # Split parameters
        beta = params[: self.n_count_params]
        gamma = params[self.n_count_params : self.n_count_params + self.n_inflate_params]
        log_alpha = params[-1]
        alpha = np.exp(np.clip(log_alpha, -10, 10))

        # Predictions
        xb_inflate = np.clip(exog_i @ gamma, -30, 30)
        pi = 1 / (1 + np.exp(-xb_inflate))
        xb_count = np.clip(exog_c @ beta, -30, 30)
        mu = np.exp(xb_count)

        if which == "mean":
            return (1 - pi) * mu
        elif which == "prob-zero":
            size = 1 / alpha
            nb_zero = np.power(size / (size + mu), size)
            return pi + (1 - pi) * nb_zero
        elif which == "prob-zero-structural":
            return pi
        elif which == "prob-zero-sampling":
            size = 1 / alpha
            nb_zero = np.power(size / (size + mu), size)
            return (1 - pi) * nb_zero
        elif which == "count-mean":
            return mu
        else:
            raise ValueError(f"Unknown prediction type: {which}")


class ZeroInflatedNegativeBinomialResult:
    """Results class for Zero-Inflated Negative Binomial model."""

    def __init__(self, model, params, llf, converged, iterations):
        """Initialize ZINB results."""
        self.model = model
        self.params = params
        self.llf = llf
        self.converged = converged
        self.iterations = iterations
        self.n_obs = model.n_obs

        # Split parameters
        self.params_count = params[: model.n_count_params]
        self.params_inflate = params[
            model.n_count_params : model.n_count_params + model.n_inflate_params
        ]
        self.log_alpha = params[-1]
        self.alpha = np.exp(np.clip(self.log_alpha, -10, 10))

        # Compute standard errors
        self._compute_standard_errors()

        # Compute fit statistics
        self._compute_fit_statistics()

    def _compute_standard_errors(self):
        """Compute standard errors via numerical Hessian."""
        try:
            from scipy.optimize import approx_fprime

            eps = 1e-5
            n_params = self.model.n_params
            hess = np.zeros((n_params, n_params))

            for i in range(n_params):

                def grad_func(params, idx=i):
                    # Numerical gradient of neg log-likelihood w.r.t. all params
                    g = approx_fprime(params, self.model._neg_log_likelihood, eps)
                    return g[idx]

                hess[i, :] = approx_fprime(self.params, grad_func, eps)

            # Symmetrize
            hess = (hess + hess.T) / 2

            self.cov_params = np.linalg.inv(hess)
            diag = np.diag(self.cov_params)
            self.bse = np.sqrt(np.maximum(diag, 0))

            # Split standard errors
            self.bse_count = self.bse[: self.model.n_count_params]
            self.bse_inflate = self.bse[
                self.model.n_count_params : self.model.n_count_params + self.model.n_inflate_params
            ]
            self.bse_alpha = self.bse[-1]

        except Exception as e:
            warnings.warn(f"Could not compute standard errors: {e}")
            self.bse = np.full(len(self.params), np.nan)
            self.bse_count = np.full(self.model.n_count_params, np.nan)
            self.bse_inflate = np.full(self.model.n_inflate_params, np.nan)
            self.bse_alpha = np.nan
            self.cov_params = np.full((len(self.params), len(self.params)), np.nan)

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
        self.predicted_zeros = np.mean(self.model.predict(self.params, which="prob-zero"))

    def summary(self, count_names=None, inflate_names=None) -> str:
        """
        Generate summary of results.

        Parameters
        ----------
        count_names : list of str, optional
            Variable names for count model
        inflate_names : list of str, optional
            Variable names for inflation model

        Returns
        -------
        str
            Formatted summary
        """
        if count_names is None:
            count_names = self.model.exog_count_names
        if inflate_names is None:
            inflate_names = self.model.exog_inflate_names

        summary = []
        summary.append("=" * 70)
        summary.append("Zero-Inflated Negative Binomial Model Results")
        summary.append("=" * 70)

        summary.append(f"Number of observations: {len(self.model.endog)}")
        summary.append(f"Log-likelihood: {self.llf:.4f}")
        summary.append(f"AIC: {self.aic:.4f}")
        summary.append(f"BIC: {self.bic:.4f}")
        summary.append(f"Alpha (overdispersion): {self.alpha:.4f}")
        summary.append(f"Converged: {self.converged}")
        summary.append("")

        summary.append(f"Actual proportion of zeros: {self.actual_zeros:.4f}")
        summary.append(f"Predicted proportion of zeros: {self.predicted_zeros:.4f}")
        summary.append("")

        # Count model parameters
        summary.append("-" * 70)
        summary.append("Count Model (Negative Binomial)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<10} {'P>|z|':<10}")

        for i in range(self.model.n_count_params):
            name = count_names[i] if i < len(count_names) else f"X{i}"
            if not np.isnan(self.bse_count[i]) and self.bse_count[i] > 0:
                z_stat = self.params_count[i] / self.bse_count[i]
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                summary.append(
                    f"{name:<20} {self.params_count[i]:<12.4f} "
                    f"{self.bse_count[i]:<12.4f} {z_stat:<10.3f} {p_val:<10.4f}"
                )
            else:
                summary.append(
                    f"{name:<20} {self.params_count[i]:<12.4f} " f"{'NA':<12} {'NA':<10} {'NA':<10}"
                )

        summary.append("")

        # Inflation model parameters
        summary.append("-" * 70)
        summary.append("Zero-Inflation Model (Logit)")
        summary.append("-" * 70)
        summary.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<10} {'P>|z|':<10}")

        for i in range(self.model.n_inflate_params):
            name = inflate_names[i] if i < len(inflate_names) else f"Z{i}"
            if not np.isnan(self.bse_inflate[i]) and self.bse_inflate[i] > 0:
                z_stat = self.params_inflate[i] / self.bse_inflate[i]
                p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                summary.append(
                    f"{name:<20} {self.params_inflate[i]:<12.4f} "
                    f"{self.bse_inflate[i]:<12.4f} {z_stat:<10.3f} {p_val:<10.4f}"
                )
            else:
                summary.append(
                    f"{name:<20} {self.params_inflate[i]:<12.4f} "
                    f"{'NA':<12} {'NA':<10} {'NA':<10}"
                )

        summary.append("=" * 70)

        return "\n".join(summary)
