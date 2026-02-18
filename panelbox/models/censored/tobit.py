"""
Tobit models for censored panel data.

This module implements Tobit models for handling censored dependent variables,
including Random Effects and Pooled specifications.

Author: PanelBox Developers
License: MIT
"""

import warnings
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

from panelbox.optimization.quadrature import gauss_hermite_quadrature


class NonlinearPanelModel:
    """Simple base class for nonlinear panel models."""

    def __init__(self, endog, exog, groups, time=None):
        self.endog = np.asarray(endog).flatten()
        self.exog = np.asarray(exog)
        self.groups = np.asarray(groups).flatten()
        self.time = time if time is not None else np.arange(len(endog))

        self.n_obs = len(self.endog)
        self.n_features = self.exog.shape[1] if len(self.exog.shape) > 1 else 1

        # Get unique entities
        self.entities = np.unique(self.groups)
        self.n_entities = len(self.entities)


class RandomEffectsTobit(NonlinearPanelModel):
    """
    Random Effects Tobit model for censored panel data.

    The model is:
        y*_it = X_it'β + α_i + ε_it
        y_it = max(c, y*_it) for left censoring
        y_it = min(c, y*_it) for right censoring

    where:
        α_i ~ N(0, σ²_α) is the individual random effect
        ε_it ~ N(0, σ²_ε) is the idiosyncratic error
        c is the censoring point

    Parameters
    ----------
    endog : array-like
        The dependent variable (N*T, 1)
    exog : array-like
        The independent variables (N*T, K)
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    censoring_point : float, default=0
        The censoring threshold
    censoring_type : str, default='left'
        Type of censoring: 'left', 'right', or 'both'
    lower_limit : float, optional
        Lower censoring point for 'both' type censoring
    upper_limit : float, optional
        Upper censoring point for 'both' type censoring
    quadrature_points : int, default=12
        Number of Gauss-Hermite quadrature points for integration
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        groups: np.ndarray,
        time: Optional[np.ndarray] = None,
        censoring_point: float = 0.0,
        censoring_type: Literal["left", "right", "both"] = "left",
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
        quadrature_points: int = 12,
    ):
        super().__init__(endog, exog, groups, time)

        self.censoring_point = censoring_point
        self.censoring_type = censoring_type
        self.lower_limit = lower_limit if lower_limit is not None else censoring_point
        self.upper_limit = upper_limit if upper_limit is not None else censoring_point
        self.quadrature_points = quadrature_points

        # Get quadrature nodes and weights
        self.nodes, self.weights = gauss_hermite_quadrature(quadrature_points)

        # Parameter names for output
        self.param_names = [f"beta_{i}" for i in range(self.n_features)]
        self.param_names.extend(["sigma_eps", "sigma_alpha"])

        # Cache for entity-specific data
        self._prepare_panel_data()

    def _prepare_panel_data(self):
        """Prepare panel data structure for efficient computation."""
        self.entity_data = {}
        for entity_id in self.entities:
            mask = self.groups == entity_id
            self.entity_data[entity_id] = {
                "y": self.endog[mask],
                "X": self.exog[mask],
                "n_obs": mask.sum(),
            }

    def _is_censored(self, y: float) -> bool:
        """Check if observation is censored."""
        if self.censoring_type == "left":
            return np.abs(y - self.censoring_point) < 1e-10
        elif self.censoring_type == "right":
            return np.abs(y - self.censoring_point) < 1e-10
        elif self.censoring_type == "both":
            return np.abs(y - self.lower_limit) < 1e-10 or np.abs(y - self.upper_limit) < 1e-10
        return False

    def _log_likelihood_i(
        self,
        y_i: np.ndarray,
        X_i: np.ndarray,
        beta: np.ndarray,
        sigma_eps: float,
        sigma_alpha: float,
    ) -> float:
        """
        Compute log-likelihood contribution for entity i.

        Uses Gauss-Hermite quadrature to integrate over random effect α_i.
        """
        n_obs = len(y_i)

        # Quadrature integration
        entity_contributions = []

        for node, weight in zip(self.nodes, self.weights):
            # Transform quadrature node to α_i ~ N(0, σ²_α)
            alpha_i = np.sqrt(2) * sigma_alpha * node

            # Likelihood contribution for this α_i value
            log_contrib = 0.0

            for t in range(n_obs):
                y_it = y_i[t]
                X_it = X_i[t]
                mean_it = X_it @ beta + alpha_i

                if self._is_censored(y_it):
                    # Censored observation
                    if self.censoring_type == "left":
                        z = (self.censoring_point - mean_it) / sigma_eps
                        log_contrib += stats.norm.logcdf(z)
                    elif self.censoring_type == "right":
                        z = (self.censoring_point - mean_it) / sigma_eps
                        log_contrib += stats.norm.logsf(z)  # log(1 - CDF)
                    elif self.censoring_type == "both":
                        if np.abs(y_it - self.lower_limit) < 1e-10:
                            z = (self.lower_limit - mean_it) / sigma_eps
                            log_contrib += stats.norm.logcdf(z)
                        else:  # upper limit
                            z = (self.upper_limit - mean_it) / sigma_eps
                            log_contrib += stats.norm.logsf(z)
                else:
                    # Uncensored observation
                    z = (y_it - mean_it) / sigma_eps
                    log_contrib += stats.norm.logpdf(z) - np.log(sigma_eps)

            # Weight by quadrature weight (in log space)
            entity_contributions.append(np.log(weight) + log_contrib)

        # Sum contributions (in log space to avoid underflow)
        return logsumexp(entity_contributions)

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute marginal log-likelihood via Gauss-Hermite quadrature.

        Parameters
        ----------
        params : array-like
            [beta, log(sigma_eps), log(sigma_alpha)]
        """
        K = self.n_features
        beta = params[:K]
        sigma_eps = np.exp(params[K])  # Ensure positive
        sigma_alpha = np.exp(params[K + 1])  # Ensure positive

        llf = 0.0
        for entity_id in self.entities:
            data = self.entity_data[entity_id]
            llf += self._log_likelihood_i(data["y"], data["X"], beta, sigma_eps, sigma_alpha)

        return llf

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """Negative log-likelihood for minimization."""
        return -self._log_likelihood(params)

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the score (gradient) of the log-likelihood.

        Uses numerical differentiation for now.
        """
        from scipy.optimize import approx_fprime

        eps = 1e-8
        return -approx_fprime(params, self._negative_log_likelihood, eps)

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "RandomEffectsTobit":
        """
        Fit the Random Effects Tobit model.

        Parameters
        ----------
        start_params : array-like, optional
            Starting values for optimization
        method : str, default='BFGS'
            Optimization method
        maxiter : int, default=1000
            Maximum iterations
        **kwargs : dict
            Additional arguments for optimizer

        Returns
        -------
        self : RandomEffectsTobit
            Fitted model
        """
        if start_params is None:
            # Get starting values from pooled OLS using numpy
            # Use only uncensored observations for initial values
            uncensored = ~np.array([self._is_censored(y) for y in self.endog])
            if uncensored.sum() > self.n_features:
                X_unc = self.exog[uncensored]
                y_unc = self.endog[uncensored]
                # Simple OLS: beta = (X'X)^-1 X'y
                beta_init = np.linalg.lstsq(X_unc, y_unc, rcond=None)[0]
            else:
                # If too few uncensored, use all observations
                beta_init = np.linalg.lstsq(self.exog, self.endog, rcond=None)[0]

            # Initial variance estimates
            residuals = self.endog - self.exog @ beta_init
            sigma_eps_init = np.log(np.std(residuals))
            sigma_alpha_init = np.log(0.5 * np.std(residuals))  # Start with smaller RE variance

            start_params = np.concatenate([beta_init, [sigma_eps_init, sigma_alpha_init]])

        # Optimize
        options = kwargs.pop("options", {})
        options.setdefault("maxiter", maxiter)
        options.setdefault("disp", True)

        result = optimize.minimize(
            self._negative_log_likelihood,
            start_params,
            method=method,
            jac=self._score if method in ["BFGS", "L-BFGS-B"] else None,
            options=options,
            **kwargs,
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}", RuntimeWarning)

        # Store results
        self.params = result.x
        self.llf = -result.fun
        self.converged = result.success
        self.n_iter = result.nit if hasattr(result, "nit") else None

        # Transform variance parameters back
        K = self.n_features
        self.beta = self.params[:K]
        self.sigma_eps = np.exp(self.params[K])
        self.sigma_alpha = np.exp(self.params[K + 1])

        # Compute standard errors (using numerical Hessian)
        self._compute_standard_errors()

        return self

    def _compute_standard_errors(self):
        """Compute standard errors from the Hessian."""
        from scipy.optimize import approx_fprime

        # Numerical Hessian
        eps = 1e-5
        n_params = len(self.params)
        hessian = np.zeros((n_params, n_params))

        for i in range(n_params):

            def grad_i(params):
                return approx_fprime(params, self._negative_log_likelihood, eps)[i]

            hessian[i, :] = approx_fprime(self.params, grad_i, eps)

        # Make symmetric
        hessian = 0.5 * (hessian + hessian.T)

        # Compute covariance matrix
        try:
            self.cov_params = np.linalg.inv(hessian)
            self.bse = np.sqrt(np.diag(self.cov_params))
        except np.linalg.LinAlgError:
            warnings.warn("Singular Hessian matrix, cannot compute standard errors", RuntimeWarning)
            self.cov_params = np.full((n_params, n_params), np.nan)
            self.bse = np.full(n_params, np.nan)

    def predict(
        self,
        exog: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        pred_type: Literal["latent", "censored"] = "censored",
    ) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        exog : array-like, optional
            Explanatory variables for prediction (uses training data if None)
        groups : array-like, optional
            Group identifiers for random effects
        pred_type : str, default='censored'
            Type of prediction:
            - 'latent': E[y*|X] = X'β (latent variable)
            - 'censored': E[y|X] accounting for censoring

        Returns
        -------
        predictions : array-like
            Predicted values
        """
        if not hasattr(self, "params"):
            raise ValueError("Model must be fitted before prediction")

        if exog is None:
            exog = self.exog
            groups = self.groups
        elif groups is None:
            # If no groups specified, use population-average prediction
            groups = np.zeros(len(exog))

        linear_pred = exog @ self.beta

        if pred_type == "latent":
            return linear_pred

        elif pred_type == "censored":
            # E[y|X] accounting for censoring
            predictions = np.zeros(len(exog))

            for i in range(len(exog)):
                mu = linear_pred[i]

                if self.censoring_type == "left":
                    # E[y|X] = mu*Φ((mu-c)/σ) + c*Φ((c-mu)/σ) + σ*φ((mu-c)/σ)
                    c = self.censoring_point
                    z = (mu - c) / self.sigma_eps
                    prob_uncensored = stats.norm.cdf(z)
                    lambda_z = stats.norm.pdf(z)  # Not inverse Mills here

                    predictions[i] = (
                        mu * prob_uncensored + c * (1 - prob_uncensored) + self.sigma_eps * lambda_z
                    )

                elif self.censoring_type == "right":
                    c = self.censoring_point
                    z = (c - mu) / self.sigma_eps
                    prob_uncensored = stats.norm.cdf(z)
                    lambda_z = stats.norm.pdf(z)

                    predictions[i] = (
                        mu * prob_uncensored + c * (1 - prob_uncensored) - self.sigma_eps * lambda_z
                    )

                elif self.censoring_type == "both":
                    # Doubly censored case
                    c_low = self.lower_limit
                    c_high = self.upper_limit
                    z_low = (c_low - mu) / self.sigma_eps
                    z_high = (c_high - mu) / self.sigma_eps

                    prob_low = stats.norm.cdf(z_low)
                    prob_high = 1 - stats.norm.cdf(z_high)
                    prob_middle = 1 - prob_low - prob_high

                    # Expected value components
                    exp_low = c_low * prob_low
                    exp_high = c_high * prob_high
                    exp_middle = mu * prob_middle + self.sigma_eps * (
                        stats.norm.pdf(z_low) - stats.norm.pdf(z_high)
                    )

                    predictions[i] = exp_low + exp_middle + exp_high

            return predictions

        else:
            raise ValueError(f"Unknown prediction type: {pred_type}")

    def marginal_effects(
        self,
        at: str = "overall",
        which: str = "conditional",
        varlist: Optional[List[str]] = None,
    ):
        """
        Compute marginal effects for Tobit model.

        Parameters
        ----------
        at : str, default='overall'
            Where to evaluate marginal effects:
            - 'overall': AME (Average Marginal Effects) - average across sample
            - 'mean': MEM (Marginal Effects at Means) - evaluate at mean of X
        which : str, default='conditional'
            Type of prediction for marginal effects:
            - 'conditional': E[y|y>c, X] - conditional on non-censoring
            - 'unconditional': E[y|X] - unconditional expectation
            - 'probability': P(y>c|X) - probability of non-censoring
        varlist : list of str, optional
            Variables to compute marginal effects for. If None, compute for all.

        Returns
        -------
        MarginalEffectsResult
            Container with marginal effects, standard errors, and summary methods

        Examples
        --------
        >>> # Fit model
        >>> model = PooledTobit(y, X, censoring_point=0)
        >>> result = model.fit()
        >>>
        >>> # Average marginal effects on conditional mean
        >>> ame = result.marginal_effects(at='overall', which='conditional')
        >>> print(ame.summary())
        >>>
        >>> # Marginal effects at means on probability of non-censoring
        >>> mem = result.marginal_effects(at='mean', which='probability')
        >>> print(mem.summary())

        Notes
        -----
        The three types of marginal effects have different interpretations:

        1. **Conditional** (which='conditional'):
           Effect on E[y|y>c, X] - "Among non-censored observations,
           how does a change in x affect y?"

        2. **Unconditional** (which='unconditional'):
           Effect on E[y|X] - "How does a change in x affect y,
           accounting for the possibility of censoring?"

        3. **Probability** (which='probability'):
           Effect on P(y>c|X) - "How does a change in x affect
           the probability of being non-censored?"

        See Also
        --------
        panelbox.marginal_effects.censored_me.compute_tobit_ame
        panelbox.marginal_effects.censored_me.compute_tobit_mem
        """
        from panelbox.marginal_effects.censored_me import compute_tobit_ame, compute_tobit_mem

        if at == "overall":
            return compute_tobit_ame(self, which=which, varlist=varlist)
        elif at == "mean":
            return compute_tobit_mem(self, which=which, varlist=varlist)
        else:
            raise ValueError(f"Unknown 'at' value: {at}. Must be 'overall' or 'mean'")

    def summary(self) -> str:
        """Generate model summary."""
        if not hasattr(self, "params"):
            return "Model has not been fitted yet"

        summary_lines = [
            "=" * 60,
            "Random Effects Tobit Results",
            "=" * 60,
            f"Number of obs:        {self.n_obs:>8d}",
            f"Number of groups:     {self.n_entities:>8d}",
            f"Censoring type:       {self.censoring_type:>8s}",
            f"Censoring point:      {self.censoring_point:>8.3f}",
            f"Log-likelihood:       {self.llf:>8.3f}",
            f"Converged:            {self.converged}",
            "-" * 60,
            f"{'Variable':<20} {'Coef.':<12} {'Std.Err.':<12} {'t':<8} {'P>|t|':<8}",
            "-" * 60,
        ]

        # Coefficients
        for i, name in enumerate(self.param_names):
            coef = self.params[i]
            se = self.bse[i] if hasattr(self, "bse") else np.nan
            t_stat = coef / se if not np.isnan(se) and se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan

            summary_lines.append(
                f"{name:<20} {coef:>11.4f} {se:>11.4f} {t_stat:>7.2f} {p_value:>7.3f}"
            )

        summary_lines.extend(
            [
                "-" * 60,
                f"sigma_eps:            {self.sigma_eps:>8.4f}",
                f"sigma_alpha:          {self.sigma_alpha:>8.4f}",
                "=" * 60,
            ]
        )

        return "\n".join(summary_lines)


class PooledTobit(NonlinearPanelModel):
    """
    Pooled Tobit model for censored data.

    This model treats the data as cross-sectional or pooled,
    ignoring the panel structure but allowing for cluster-robust
    standard errors.

    Parameters
    ----------
    endog : array-like
        Dependent variable (possibly censored)
    exog : array-like
        Independent variables
    groups : array-like, optional
        Group identifiers for clustered standard errors
    censoring_point : float, default=0
        Censoring threshold
    censoring_type : {'left', 'right', 'both'}, default='left'
        Type of censoring
    lower_limit : float, optional
        Lower limit for 'both' censoring
    upper_limit : float, optional
        Upper limit for 'both' censoring
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        groups: Optional[np.ndarray] = None,
        censoring_point: float = 0.0,
        censoring_type: Literal["left", "right", "both"] = "left",
        lower_limit: Optional[float] = None,
        upper_limit: Optional[float] = None,
    ):
        """Initialize Pooled Tobit model."""
        if groups is None:
            groups = np.arange(len(endog))  # Each obs is its own group

        super().__init__(endog, exog, groups)

        self.censoring_point = censoring_point
        self.censoring_type = censoring_type
        self.lower_limit = lower_limit if lower_limit is not None else censoring_point
        self.upper_limit = upper_limit if upper_limit is not None else censoring_point

        # Validate censoring type
        if censoring_type not in ["left", "right", "both"]:
            raise ValueError("censoring_type must be 'left', 'right', or 'both'")

        if censoring_type == "both" and (lower_limit is None or upper_limit is None):
            raise ValueError("For 'both' censoring, provide lower_limit and upper_limit")

        # Parameter names
        self.param_names = [f"beta_{i}" for i in range(self.n_features)]
        self.param_names.append("sigma")

    def _is_censored(self, y: float) -> bool:
        """Check if observation is censored."""
        if self.censoring_type == "left":
            return np.abs(y - self.censoring_point) < 1e-10
        elif self.censoring_type == "right":
            return np.abs(y - self.censoring_point) < 1e-10
        elif self.censoring_type == "both":
            return np.abs(y - self.lower_limit) < 1e-10 or np.abs(y - self.upper_limit) < 1e-10
        return False

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for pooled Tobit.

        Parameters
        ----------
        params : ndarray
            [beta, log_sigma]

        Returns
        -------
        float
            Log-likelihood
        """
        k = self.n_features
        beta = params[:k]
        sigma = np.exp(params[k])  # Ensure positive

        # Linear prediction
        linear_pred = self.exog @ beta

        llf = 0
        for i in range(len(self.endog)):
            y_i = self.endog[i]
            mu_i = linear_pred[i]

            if self._is_censored(y_i):
                # Censored observation
                if self.censoring_type == "left":
                    z = (self.censoring_point - mu_i) / sigma
                    llf += stats.norm.logcdf(z)
                elif self.censoring_type == "right":
                    z = (mu_i - self.censoring_point) / sigma
                    llf += stats.norm.logsf(z)  # log(1 - CDF)
                elif self.censoring_type == "both":
                    if np.abs(y_i - self.lower_limit) < 1e-10:
                        z = (self.lower_limit - mu_i) / sigma
                        llf += stats.norm.logcdf(z)
                    else:
                        z = (self.upper_limit - mu_i) / sigma
                        llf += stats.norm.logsf(z)
            else:
                # Uncensored observation
                z = (y_i - mu_i) / sigma
                llf += stats.norm.logpdf(z) - np.log(sigma)

        return llf

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """Negative log-likelihood for minimization."""
        return -self._log_likelihood(params)

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "PooledTobit":
        """
        Fit the Pooled Tobit model.

        Parameters
        ----------
        start_params : array-like, optional
            Starting values for optimization
        method : str, default='BFGS'
            Optimization method
        maxiter : int, default=1000
            Maximum iterations

        Returns
        -------
        self : PooledTobit
            Fitted model
        """
        if start_params is None:
            # Get starting values from OLS on uncensored observations
            if self.censoring_type == "left":
                uncensored = self.endog > self.censoring_point
            elif self.censoring_type == "right":
                uncensored = self.endog < self.censoring_point
            else:
                uncensored = (self.endog > self.lower_limit) & (self.endog < self.upper_limit)

            if np.sum(uncensored) > self.n_features:
                X_uncensored = self.exog[uncensored]
                y_uncensored = self.endog[uncensored]
                beta_start = np.linalg.lstsq(X_uncensored, y_uncensored, rcond=None)[0]
                resid = y_uncensored - X_uncensored @ beta_start
                sigma_start = np.log(np.std(resid))
            else:
                beta_start = np.zeros(self.n_features)
                sigma_start = np.log(1.0)

            start_params = np.concatenate([beta_start, [sigma_start]])

        # Optimize
        options = kwargs.pop("options", {})
        options.setdefault("maxiter", maxiter)
        options.setdefault("disp", False)

        result = optimize.minimize(
            self._negative_log_likelihood, start_params, method=method, options=options, **kwargs
        )

        if not result.success:
            warnings.warn(f"Optimization failed: {result.message}", RuntimeWarning)

        # Store results
        self.params = result.x
        self.llf = -result.fun
        self.converged = result.success
        self.n_iter = result.nit if hasattr(result, "nit") else None

        # Extract parameters
        k = self.n_features
        self.beta = self.params[:k]
        self.sigma = np.exp(self.params[k])

        # Compute standard errors
        self._compute_standard_errors()

        return self

    def _compute_standard_errors(self):
        """Compute standard errors from the Hessian."""
        from scipy.optimize import approx_fprime

        # Numerical Hessian
        eps = 1e-5
        n_params = len(self.params)
        hessian = np.zeros((n_params, n_params))

        for i in range(n_params):

            def grad_i(params):
                return approx_fprime(params, self._negative_log_likelihood, eps)[i]

            hessian[i, :] = approx_fprime(self.params, grad_i, eps)

        # Make symmetric
        hessian = 0.5 * (hessian + hessian.T)

        # Compute covariance matrix
        try:
            self.cov_params = np.linalg.inv(hessian)
            self.bse = np.sqrt(np.diag(self.cov_params))
        except np.linalg.LinAlgError:
            warnings.warn("Singular Hessian matrix, cannot compute standard errors", RuntimeWarning)
            self.cov_params = np.full((n_params, n_params), np.nan)
            self.bse = np.full(n_params, np.nan)

    def predict(
        self,
        exog: Optional[np.ndarray] = None,
        pred_type: Literal["latent", "censored", "probability"] = "censored",
    ) -> np.ndarray:
        """
        Generate predictions from fitted model.

        Parameters
        ----------
        exog : ndarray, optional
            Exogenous variables for prediction
        pred_type : {'latent', 'censored', 'probability'}
            Type of prediction

        Returns
        -------
        ndarray
            Predictions
        """
        if not hasattr(self, "params"):
            raise ValueError("Model must be fitted before prediction")

        if exog is None:
            exog = self.exog

        linear_pred = exog @ self.beta

        if pred_type == "latent":
            return linear_pred

        elif pred_type == "censored":
            predictions = np.zeros(len(exog))

            for i in range(len(exog)):
                mu = linear_pred[i]

                if self.censoring_type == "left":
                    c = self.censoring_point
                    z = (mu - c) / self.sigma
                    prob_uncensored = stats.norm.cdf(z)
                    lambda_z = stats.norm.pdf(z)

                    predictions[i] = (
                        mu * prob_uncensored + c * (1 - prob_uncensored) + self.sigma * lambda_z
                    )

                elif self.censoring_type == "right":
                    c = self.censoring_point
                    z = (c - mu) / self.sigma
                    prob_uncensored = stats.norm.cdf(z)
                    lambda_z = stats.norm.pdf(z)

                    predictions[i] = (
                        mu * prob_uncensored + c * (1 - prob_uncensored) - self.sigma * lambda_z
                    )

                elif self.censoring_type == "both":
                    c_low = self.lower_limit
                    c_high = self.upper_limit
                    z_low = (c_low - mu) / self.sigma
                    z_high = (c_high - mu) / self.sigma

                    prob_low = stats.norm.cdf(z_low)
                    prob_high = 1 - stats.norm.cdf(z_high)
                    prob_middle = 1 - prob_low - prob_high

                    exp_low = c_low * prob_low
                    exp_high = c_high * prob_high
                    exp_middle = mu * prob_middle + self.sigma * (
                        stats.norm.pdf(z_low) - stats.norm.pdf(z_high)
                    )

                    predictions[i] = exp_low + exp_middle + exp_high

            return predictions

        elif pred_type == "probability":
            # Probability of censoring
            if self.censoring_type == "left":
                z = (self.censoring_point - linear_pred) / self.sigma
                return stats.norm.cdf(z)
            elif self.censoring_type == "right":
                z = (linear_pred - self.censoring_point) / self.sigma
                return stats.norm.cdf(z)
            else:
                z_lower = (self.lower_limit - linear_pred) / self.sigma
                z_upper = (self.upper_limit - linear_pred) / self.sigma
                return stats.norm.cdf(z_lower) + (1 - stats.norm.cdf(z_upper))

        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

    def marginal_effects(
        self,
        at: str = "overall",
        which: str = "conditional",
        varlist: Optional[List[str]] = None,
    ):
        """
        Compute marginal effects for Tobit model.

        Parameters
        ----------
        at : str, default='overall'
            Where to evaluate marginal effects:
            - 'overall': AME (Average Marginal Effects) - average across sample
            - 'mean': MEM (Marginal Effects at Means) - evaluate at mean of X
        which : str, default='conditional'
            Type of prediction for marginal effects:
            - 'conditional': E[y|y>c, X] - conditional on non-censoring
            - 'unconditional': E[y|X] - unconditional expectation
            - 'probability': P(y>c|X) - probability of non-censoring
        varlist : list of str, optional
            Variables to compute marginal effects for. If None, compute for all.

        Returns
        -------
        MarginalEffectsResult
            Container with marginal effects, standard errors, and summary methods

        Examples
        --------
        >>> # Fit model
        >>> model = PooledTobit(y, X, censoring_point=0)
        >>> result = model.fit()
        >>>
        >>> # Average marginal effects on conditional mean
        >>> ame = result.marginal_effects(at='overall', which='conditional')
        >>> print(ame.summary())
        >>>
        >>> # Marginal effects at means on probability of non-censoring
        >>> mem = result.marginal_effects(at='mean', which='probability')
        >>> print(mem.summary())

        Notes
        -----
        The three types of marginal effects have different interpretations:

        1. **Conditional** (which='conditional'):
           Effect on E[y|y>c, X] - "Among non-censored observations,
           how does a change in x affect y?"

        2. **Unconditional** (which='unconditional'):
           Effect on E[y|X] - "How does a change in x affect y,
           accounting for the possibility of censoring?"

        3. **Probability** (which='probability'):
           Effect on P(y>c|X) - "How does a change in x affect
           the probability of being non-censored?"

        See Also
        --------
        panelbox.marginal_effects.censored_me.compute_tobit_ame
        panelbox.marginal_effects.censored_me.compute_tobit_mem
        """
        from panelbox.marginal_effects.censored_me import compute_tobit_ame, compute_tobit_mem

        if at == "overall":
            return compute_tobit_ame(self, which=which, varlist=varlist)
        elif at == "mean":
            return compute_tobit_mem(self, which=which, varlist=varlist)
        else:
            raise ValueError(f"Unknown 'at' value: {at}. Must be 'overall' or 'mean'")

    def summary(self) -> str:
        """Generate model summary."""
        if not hasattr(self, "params"):
            return "Model has not been fitted yet"

        summary_lines = [
            "=" * 60,
            "Pooled Tobit Results",
            "=" * 60,
            f"Number of obs:        {self.n_obs:>8d}",
            f"Censoring type:       {self.censoring_type:>8s}",
            f"Censoring point:      {self.censoring_point:>8.3f}",
            f"Log-likelihood:       {self.llf:>8.3f}",
            f"Converged:            {self.converged}",
            "-" * 60,
            f"{'Variable':<20} {'Coef.':<12} {'Std.Err.':<12} {'t':<8} {'P>|t|':<8}",
            "-" * 60,
        ]

        # Coefficients
        for i, name in enumerate(self.param_names):
            coef = self.params[i]
            se = self.bse[i] if hasattr(self, "bse") else np.nan
            t_stat = coef / se if not np.isnan(se) and se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(abs(t_stat))) if not np.isnan(t_stat) else np.nan

            summary_lines.append(
                f"{name:<20} {coef:>11.4f} {se:>11.4f} {t_stat:>7.2f} {p_value:>7.3f}"
            )

        summary_lines.extend(["-" * 60, f"sigma:                {self.sigma:>8.4f}", "=" * 60])

        return "\n".join(summary_lines)
