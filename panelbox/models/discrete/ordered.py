"""
Ordered Choice Models for Panel Data.

This module implements ordered logit and probit models for ordinal
dependent variables in panel data settings.

Author: PanelBox Developers
License: MIT
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
from scipy import optimize, stats
from scipy.special import expit, logsumexp

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


class OrderedChoiceModel(NonlinearPanelModel, ABC):
    """
    Base class for ordered choice models.

    The model is:
        y*_it = X_it'β + ε_it
        y_it = j if κ_{j-1} < y*_it ≤ κ_j

    where κ are the cutpoints/thresholds to be estimated.
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        groups: np.ndarray,
        time: Optional[np.ndarray] = None,
        n_categories: Optional[int] = None,
    ):
        super().__init__(endog, exog, groups, time)

        # Ensure endog contains integer categories starting from 0
        self.endog = np.asarray(self.endog, dtype=int).flatten()
        unique_vals = np.unique(self.endog)

        if n_categories is None:
            self.n_categories = len(unique_vals)
        else:
            self.n_categories = n_categories

        # Check that categories are 0, 1, 2, ..., J-1
        expected_cats = np.arange(self.n_categories)
        if not np.array_equal(np.sort(unique_vals), expected_cats):
            warnings.warn(
                f"Categories should be 0, 1, ..., {self.n_categories-1}. " "Remapping categories.",
                RuntimeWarning,
            )
            # Remap to 0-based consecutive integers
            self.category_map = {val: i for i, val in enumerate(unique_vals)}
            self.endog = np.array([self.category_map[val] for val in self.endog])

        self.n_cutpoints = self.n_categories - 1

        # Parameter names
        self.param_names = [f"beta_{i}" for i in range(self.n_features)]
        self.param_names.extend([f"cutpoint_{j}" for j in range(self.n_cutpoints)])

    @abstractmethod
    def _cdf(self, z: np.ndarray) -> np.ndarray:
        """CDF function for the error distribution."""
        pass

    @abstractmethod
    def _pdf(self, z: np.ndarray) -> np.ndarray:
        """PDF function for the error distribution."""
        pass

    def _transform_cutpoints(self, cutpoint_params: np.ndarray) -> np.ndarray:
        """
        Transform unconstrained parameters to ordered cutpoints.

        Uses the parameterization:
        κ_0 = γ_0
        κ_j = κ_{j-1} + exp(γ_j) for j > 0

        This ensures κ_0 < κ_1 < ... < κ_{J-2}
        """
        cutpoints = np.zeros(len(cutpoint_params))
        cutpoints[0] = cutpoint_params[0]

        for j in range(1, len(cutpoint_params)):
            cutpoints[j] = cutpoints[j - 1] + np.exp(cutpoint_params[j])

        return cutpoints

    def _inverse_transform_cutpoints(self, cutpoints: np.ndarray) -> np.ndarray:
        """
        Inverse transform from ordered cutpoints to unconstrained parameters.
        """
        cutpoint_params = np.zeros(len(cutpoints))
        cutpoint_params[0] = cutpoints[0]

        for j in range(1, len(cutpoints)):
            diff = cutpoints[j] - cutpoints[j - 1]
            cutpoint_params[j] = np.log(np.maximum(diff, 1e-10))

        return cutpoint_params

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute the log-likelihood for ordered choice model.

        Parameters
        ----------
        params : array-like
            [beta, cutpoint_params]
        """
        K = self.n_features
        beta = params[:K]
        cutpoint_params = params[K:]

        # Transform to ensure ordered cutpoints
        cutpoints = self._transform_cutpoints(cutpoint_params)

        # Add boundary cutpoints
        cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

        llf = 0.0
        for i in range(self.n_obs):
            y_i = self.endog[i]
            X_i = self.exog[i]
            linear_pred = X_i @ beta

            # P(y = j) = F(κ_j - X'β) - F(κ_{j-1} - X'β)
            upper_prob = self._cdf(cutpoints_extended[y_i + 1] - linear_pred)
            lower_prob = self._cdf(cutpoints_extended[y_i] - linear_pred)
            prob = upper_prob - lower_prob

            # Avoid log(0)
            llf += np.log(np.maximum(prob, 1e-10))

        return llf

    def _negative_log_likelihood(self, params: np.ndarray) -> float:
        """Negative log-likelihood for minimization."""
        return -self._log_likelihood(params)

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Compute the score (gradient) of the log-likelihood.

        Uses analytical gradient for efficiency.
        """
        K = self.n_features
        beta = params[:K]
        cutpoint_params = params[K:]

        cutpoints = self._transform_cutpoints(cutpoint_params)
        cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

        # Initialize gradient
        grad_beta = np.zeros(K)
        grad_cutpoint_raw = np.zeros(self.n_cutpoints)

        for i in range(self.n_obs):
            y_i = self.endog[i]
            X_i = self.exog[i]
            linear_pred = X_i @ beta

            # Compute probabilities
            z_upper = cutpoints_extended[y_i + 1] - linear_pred
            z_lower = cutpoints_extended[y_i] - linear_pred

            F_upper = self._cdf(z_upper)
            F_lower = self._cdf(z_lower)
            prob = F_upper - F_lower

            f_upper = self._pdf(z_upper) if np.isfinite(z_upper) else 0
            f_lower = self._pdf(z_lower) if np.isfinite(z_lower) else 0

            # Gradient w.r.t. beta
            grad_beta += X_i * (f_lower - f_upper) / np.maximum(prob, 1e-10)

            # Gradient w.r.t. cutpoints
            if y_i < self.n_categories - 1:  # Not the highest category
                grad_cutpoint_raw[y_i] += f_upper / np.maximum(prob, 1e-10)
            if y_i > 0:  # Not the lowest category
                grad_cutpoint_raw[y_i - 1] -= f_lower / np.maximum(prob, 1e-10)

        # Transform gradient for cutpoint parameters
        # Account for the exponential transformation
        grad_cutpoint_params = np.zeros(self.n_cutpoints)
        grad_cutpoint_params[0] = grad_cutpoint_raw[0]

        for j in range(1, self.n_cutpoints):
            # d/dγ_j = d/dκ_j * dκ_j/dγ_j
            # Since κ_j = κ_{j-1} + exp(γ_j), we have dκ_j/dγ_j = exp(γ_j)
            # Also, κ_{j+1}, κ_{j+2}, ... all depend on κ_j
            exp_gamma_j = np.exp(cutpoint_params[j])
            for k in range(j, self.n_cutpoints):
                grad_cutpoint_params[j] += grad_cutpoint_raw[k] * exp_gamma_j

        return -np.concatenate([grad_beta, grad_cutpoint_params])

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "OrderedChoiceModel":
        """
        Fit the ordered choice model.

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
        self : OrderedChoiceModel
            Fitted model
        """
        if start_params is None:
            # Initialize beta with zeros
            beta_init = np.zeros(self.n_features)

            # Initialize cutpoints based on empirical distribution
            cutpoints_init = np.zeros(self.n_cutpoints)
            for j in range(self.n_cutpoints):
                # Proportion in categories 0 to j
                prop_j = np.mean(self.endog <= j)
                # Use inverse CDF to get initial cutpoint
                if 0 < prop_j < 1:
                    if isinstance(self, OrderedLogit):
                        # Inverse logistic CDF
                        cutpoints_init[j] = np.log(prop_j / (1 - prop_j))
                    else:  # OrderedProbit
                        # Inverse normal CDF
                        cutpoints_init[j] = stats.norm.ppf(prop_j)
                else:
                    cutpoints_init[j] = j - (self.n_cutpoints - 1) / 2

            # Transform cutpoints to unconstrained parameters
            cutpoint_params_init = self._inverse_transform_cutpoints(cutpoints_init)

            start_params = np.concatenate([beta_init, cutpoint_params_init])

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

        # Extract components
        K = self.n_features
        self.beta = self.params[:K]
        self.cutpoint_params = self.params[K:]
        self.cutpoints = self._transform_cutpoints(self.cutpoint_params)

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
                return self._score(params)[i]

            hessian[i, :] = approx_fprime(self.params, grad_i, eps)

        # Make symmetric
        hessian = 0.5 * (hessian + hessian.T)

        # Compute covariance matrix
        try:
            # For MLE, cov = inv(-H) where H is Hessian of log-likelihood
            # We computed Hessian of negative log-likelihood, so no need for extra negative
            self.cov_params = np.linalg.inv(hessian)
            self.bse = np.sqrt(np.diag(self.cov_params))
        except np.linalg.LinAlgError:
            warnings.warn("Singular Hessian matrix, cannot compute standard errors", RuntimeWarning)
            self.cov_params = np.full((n_params, n_params), np.nan)
            self.bse = np.full(n_params, np.nan)

    def predict_proba(self, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict probabilities for each category.

        Parameters
        ----------
        exog : array-like, optional
            Explanatory variables for prediction (uses training data if None)

        Returns
        -------
        probabilities : array-like
            Predicted probabilities for each category (N x J)
        """
        if not hasattr(self, "params"):
            raise ValueError("Model must be fitted before prediction")

        if exog is None:
            exog = self.exog

        exog = np.asarray(exog)
        if exog.ndim == 1:
            exog = exog.reshape(1, -1)

        n_obs = len(exog)
        linear_pred = exog @ self.beta

        # Extended cutpoints with boundaries
        cutpoints_extended = np.concatenate([[-np.inf], self.cutpoints, [np.inf]])

        # Compute probabilities for each category
        probs = np.zeros((n_obs, self.n_categories))

        for j in range(self.n_categories):
            upper = cutpoints_extended[j + 1] - linear_pred
            lower = cutpoints_extended[j] - linear_pred

            probs[:, j] = self._cdf(upper) - self._cdf(lower)

        # Ensure probabilities sum to 1 (numerical stability)
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(
        self, exog: Optional[np.ndarray] = None, type: Literal["category", "prob"] = "category"
    ) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        exog : array-like, optional
            Explanatory variables for prediction
        type : str, default='category'
            Type of prediction:
            - 'category': Most probable category
            - 'prob': Probabilities for each category

        Returns
        -------
        predictions : array-like
            Predicted categories or probabilities
        """
        if type == "prob":
            return self.predict_proba(exog)
        elif type == "category":
            probs = self.predict_proba(exog)
            return np.argmax(probs, axis=1)
        else:
            raise ValueError(f"Unknown prediction type: {type}")

    def summary(self) -> str:
        """Generate model summary."""
        if not hasattr(self, "params"):
            return "Model has not been fitted yet"

        model_name = self.__class__.__name__

        summary_lines = [
            "=" * 60,
            f"{model_name} Results",
            "=" * 60,
            f"Number of obs:        {self.n_obs:>8d}",
            f"Number of categories: {self.n_categories:>8d}",
            f"Log-likelihood:       {self.llf:>8.3f}",
            f"Converged:            {self.converged}",
            "-" * 60,
            f"{'Variable':<20} {'Coef.':<12} {'Std.Err.':<12} {'z':<8} {'P>|z|':<8}",
            "-" * 60,
        ]

        # Coefficients
        for i, name in enumerate(self.param_names):
            coef = self.params[i]
            se = self.bse[i] if hasattr(self, "bse") else np.nan
            z_stat = coef / se if not np.isnan(se) and se > 0 else np.nan
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if not np.isnan(z_stat) else np.nan

            summary_lines.append(
                f"{name:<20} {coef:>11.4f} {se:>11.4f} {z_stat:>7.2f} {p_value:>7.3f}"
            )

        summary_lines.extend(["-" * 60, "Cutpoints (ordered):", "-" * 60])

        for j, cutpoint in enumerate(self.cutpoints):
            summary_lines.append(f"κ_{j:<3d}                  {cutpoint:>11.4f}")

        summary_lines.append("=" * 60)

        return "\n".join(summary_lines)


class OrderedLogit(OrderedChoiceModel):
    """
    Ordered Logit model for ordinal panel data.

    The model uses the logistic distribution for the error term.

    Parameters
    ----------
    endog : array-like
        The dependent variable with ordinal categories (0, 1, 2, ...)
    exog : array-like
        The independent variables
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    n_categories : int, optional
        Number of categories (inferred from data if not provided)
    """

    def _cdf(self, z: np.ndarray) -> np.ndarray:
        """Logistic CDF."""
        return expit(z)

    def _pdf(self, z: np.ndarray) -> np.ndarray:
        """Logistic PDF."""
        # PDF = F(z) * (1 - F(z))
        F_z = expit(z)
        return F_z * (1 - F_z)


class OrderedProbit(OrderedChoiceModel):
    """
    Ordered Probit model for ordinal panel data.

    The model uses the standard normal distribution for the error term.

    Parameters
    ----------
    endog : array-like
        The dependent variable with ordinal categories (0, 1, 2, ...)
    exog : array-like
        The independent variables
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    n_categories : int, optional
        Number of categories (inferred from data if not provided)
    """

    def _cdf(self, z: np.ndarray) -> np.ndarray:
        """Standard normal CDF."""
        return stats.norm.cdf(z)

    def _pdf(self, z: np.ndarray) -> np.ndarray:
        """Standard normal PDF."""
        return stats.norm.pdf(z)


class RandomEffectsOrderedLogit(OrderedChoiceModel):
    """
    Random Effects Ordered Logit model.

    The model is:
        y*_it = X_it'β + α_i + ε_it

    where α_i ~ N(0, σ²_α) and ε_it follows a logistic distribution.

    Parameters
    ----------
    endog : array-like
        The dependent variable with ordinal categories
    exog : array-like
        The independent variables
    groups : array-like
        Group identifiers for panel structure
    time : array-like, optional
        Time identifiers for panel structure
    n_categories : int, optional
        Number of categories
    quadrature_points : int, default=12
        Number of Gauss-Hermite quadrature points
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        groups: np.ndarray,
        time: Optional[np.ndarray] = None,
        n_categories: Optional[int] = None,
        quadrature_points: int = 12,
    ):
        super().__init__(endog, exog, groups, time, n_categories)

        self.quadrature_points = quadrature_points
        self.nodes, self.weights = gauss_hermite_quadrature(quadrature_points)

        # Add sigma_alpha to parameter names
        self.param_names.append("sigma_alpha")

        # Prepare entity data
        self._prepare_entity_data()

    def _prepare_entity_data(self):
        """Prepare data by entity for efficient computation."""
        self.entity_data = {}
        for entity_id in self.entities:
            mask = self.groups == entity_id
            self.entity_data[entity_id] = {
                "y": self.endog[mask],
                "X": self.exog[mask],
                "n_obs": mask.sum(),
            }

    def _cdf(self, z: np.ndarray) -> np.ndarray:
        """Logistic CDF."""
        return expit(z)

    def _pdf(self, z: np.ndarray) -> np.ndarray:
        """Logistic PDF."""
        F_z = expit(z)
        return F_z * (1 - F_z)

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute marginal log-likelihood with random effects.

        Integrates out the random effect using Gauss-Hermite quadrature.
        """
        K = self.n_features
        beta = params[:K]
        cutpoint_params = params[K : K + self.n_cutpoints]
        sigma_alpha = np.exp(params[-1])  # Ensure positive

        cutpoints = self._transform_cutpoints(cutpoint_params)
        cutpoints_extended = np.concatenate([[-np.inf], cutpoints, [np.inf]])

        llf = 0.0

        for entity_id in self.entities:
            data = self.entity_data[entity_id]
            y_i = data["y"]
            X_i = data["X"]
            n_obs_i = data["n_obs"]

            # Quadrature integration over α_i
            entity_contributions = []

            for node, weight in zip(self.nodes, self.weights):
                # Transform to α_i ~ N(0, σ²_α)
                alpha_i = np.sqrt(2) * sigma_alpha * node

                # Likelihood for this α_i
                log_lik_alpha = 0.0

                for t in range(n_obs_i):
                    y_it = y_i[t]
                    X_it = X_i[t]
                    linear_pred = X_it @ beta + alpha_i

                    upper_prob = self._cdf(cutpoints_extended[y_it + 1] - linear_pred)
                    lower_prob = self._cdf(cutpoints_extended[y_it] - linear_pred)
                    prob = upper_prob - lower_prob

                    log_lik_alpha += np.log(np.maximum(prob, 1e-10))

                # Weight by quadrature weight
                entity_contributions.append(np.log(weight) + log_lik_alpha)

            # Sum over quadrature points (in log space)
            llf += logsumexp(entity_contributions)

        return llf

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "RandomEffectsOrderedLogit":
        """
        Fit the Random Effects Ordered Logit model.

        Parameters include β, cutpoints, and σ_α.
        """
        if start_params is None:
            # Fit pooled model for initial values
            pooled = OrderedLogit(self.endog, self.exog, self.groups, self.time, self.n_categories)
            pooled_result = pooled.fit(maxiter=100)

            # Add initial sigma_alpha (log scale)
            sigma_alpha_init = np.log(0.5)  # Start with moderate RE variance

            start_params = np.concatenate([pooled_result.params, [sigma_alpha_init]])

        # Optimize
        options = kwargs.pop("options", {})
        options.setdefault("maxiter", maxiter)
        options.setdefault("disp", True)

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

        # Extract components
        K = self.n_features
        self.beta = self.params[:K]
        self.cutpoint_params = self.params[K : K + self.n_cutpoints]
        self.cutpoints = self._transform_cutpoints(self.cutpoint_params)
        self.sigma_alpha = np.exp(self.params[-1])

        return self
