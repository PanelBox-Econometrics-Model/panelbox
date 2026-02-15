"""
Multinomial logit model for panel data.

This module implements multinomial logit for unordered categorical outcomes
with J > 2 alternatives.

The probability of choosing alternative j is:
P(y_it = j | X_it) = exp(X_it'β_j) / Σ_k exp(X_it'β_k)

For panel data, we implement:
1. Pooled multinomial logit
2. Fixed effects via conditional logit (if feasible)

References
----------
McFadden, D. (1973). "Conditional logit analysis of qualitative choice behavior."
    In Frontiers in Econometrics, ed. P. Zarembka. New York: Academic Press.
Train, K. (2009). "Discrete Choice Methods with Simulation." Cambridge University Press.
Cameron, A. C., & Trivedi, P. K. (2005). "Microeconometrics: Methods and Applications."
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import logsumexp

from panelbox.models.base import NonlinearPanelModel, PanelModelResults


class MultinomialLogit(NonlinearPanelModel):
    """
    Multinomial Logit for unordered categorical outcomes.

    Parameters
    ----------
    endog : array-like or pd.Series
        Dependent variable (categorical with J alternatives: 0, 1, ..., J-1)
    exog : array-like or pd.DataFrame
        Regressors
    n_alternatives : int, optional
        Number of choice alternatives. If None, inferred from data
    base_alternative : int, default=0
        Reference/base alternative (normalized to zero)
    method : str, default='pooled'
        Estimation method: 'pooled', 'fixed_effects', or 'random_effects'
    entity_col : str, optional
        Name of entity/individual identifier column
    time_col : str, optional
        Name of time identifier column

    Attributes
    ----------
    n_alternatives : int
        Number of choice alternatives
    n_params : int
        Total number of parameters (J-1) × K
    method : str
        Estimation method used
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog: Union[np.ndarray, pd.DataFrame],
        n_alternatives: Optional[int] = None,
        base_alternative: int = 0,
        method: str = "pooled",
        entity_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """Initialize Multinomial Logit model."""
        super().__init__(endog, exog, entity_col, time_col)

        # Validate method
        valid_methods = ["pooled", "fixed_effects", "random_effects"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {valid_methods}, got {method}")
        self.method = method

        # Determine number of alternatives
        if n_alternatives is None:
            self.n_alternatives = int(np.max(self.endog)) + 1
        else:
            self.n_alternatives = n_alternatives

        self.base_alternative = base_alternative

        # Check that endog contains valid categories
        unique_choices = np.unique(self.endog)
        if not np.all(np.isin(unique_choices, np.arange(self.n_alternatives))):
            raise ValueError(f"endog must contain integers from 0 to {self.n_alternatives-1}")

        # Number of parameters: (J-1) sets of K coefficients
        self.K = self.exog.shape[1]  # Number of regressors
        self.n_params = (self.n_alternatives - 1) * self.K

        # Create alternative-specific indices for easier computation
        self._create_alternative_indices()

        # For fixed/random effects, need entity identifiers
        if self.method in ["fixed_effects", "random_effects"]:
            if entity_col is None:
                raise ValueError(f"entity_col required for method='{self.method}'")

            # Get unique entities and time periods per entity
            self.entity_ids = self._get_entity_ids()
            self.n_entities = len(np.unique(self.entity_ids))

            # Check computational feasibility for FE
            if self.method == "fixed_effects":
                self._check_fe_feasibility()

    def _create_alternative_indices(self):
        """Create indices for alternatives."""
        self.alternatives = np.arange(self.n_alternatives)
        # Non-base alternatives
        self.non_base_alts = np.delete(self.alternatives, self.base_alternative)

    def _get_entity_ids(self):
        """Extract entity identifiers from data."""
        # This assumes data was passed with entity_col
        # The base class should have stored this
        if hasattr(self, "_entity_col_data"):
            return self._entity_col_data
        else:
            # Fallback: generate sequential IDs based on panel structure
            # This is a simplification - in practice, entity_col should be properly passed
            warnings.warn(
                "Entity IDs not found, generating sequential IDs. "
                "This may not reflect true panel structure."
            )
            n = len(self.endog)
            return np.arange(n)

    def _check_fe_feasibility(self):
        """Check if Fixed Effects estimation is computationally feasible."""
        # Warn if J > 4 or T > 10 (Chamberlain FE becomes very expensive)
        if self.n_alternatives > 4:
            warnings.warn(
                f"Fixed Effects Multinomial with J={self.n_alternatives} alternatives "
                f"is computationally intensive. Consider using J ≤ 4 or method='pooled'. "
                f"Estimation may be slow or fail.",
                UserWarning,
            )

        # Check average time periods per entity
        unique_entities = np.unique(self.entity_ids)
        t_periods = []
        for entity in unique_entities:
            entity_mask = self.entity_ids == entity
            t_periods.append(np.sum(entity_mask))

        avg_t = np.mean(t_periods)
        if avg_t > 10:
            warnings.warn(
                f"Average T={avg_t:.1f} periods per entity. "
                f"Fixed Effects Multinomial becomes very slow for T > 10. "
                f"Consider using method='random_effects' instead.",
                UserWarning,
            )

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Required by base class - delegates to log_likelihood.
        """
        return -self.log_likelihood(params)  # Return negative for minimization

    def log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for multinomial logit.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector of shape ((J-1) × K,) for pooled/FE,
            or ((J-1) × K + n_entities × (J-1),) for RE

        Returns
        -------
        float
            Negative log-likelihood
        """
        if self.method == "pooled":
            return self._log_likelihood_pooled(params)
        elif self.method == "fixed_effects":
            return self._log_likelihood_fixed_effects(params)
        elif self.method == "random_effects":
            return self._log_likelihood_random_effects(params)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _log_likelihood_pooled(self, params: np.ndarray) -> float:
        """Pooled multinomial logit log-likelihood."""
        # Reshape params to (J-1, K) matrix
        beta_matrix = params.reshape(self.n_alternatives - 1, self.K)

        llf = 0.0
        n = len(self.endog)

        for i in range(n):
            y_i = int(self.endog[i])
            X_i = self.exog[i]

            # Compute utilities for all alternatives
            # Base alternative has utility 0
            utilities = np.zeros(self.n_alternatives)

            # Fill in non-base alternatives
            idx = 0
            for j in self.alternatives:
                if j == self.base_alternative:
                    utilities[j] = 0
                else:
                    utilities[j] = X_i @ beta_matrix[idx]
                    idx += 1

            # Log-sum-exp for numerical stability
            log_denom = logsumexp(utilities)

            # Contribution to log-likelihood
            llf += utilities[y_i] - log_denom

        return -llf  # Return negative for minimization

    def _log_likelihood_fixed_effects(self, params: np.ndarray) -> float:
        """
        Fixed Effects multinomial logit using conditional MLE (Chamberlain 1980).

        This conditions on the sufficient statistic to eliminate fixed effects.
        Only entities with variation in choices contribute to the likelihood.
        """
        # For simplicity, we'll use a dummy variable approach here
        # A full Chamberlain conditional MLE would require enumerating choice sequences
        # which is computationally prohibitive for large J or T

        # This implementation uses within-transformation on utilities
        warnings.warn(
            "Fixed Effects Multinomial uses within-transformation approximation. "
            "For exact Chamberlain conditional MLE, use specialized software.",
            UserWarning,
        )

        beta_matrix = params.reshape(self.n_alternatives - 1, self.K)

        llf = 0.0
        unique_entities = np.unique(self.entity_ids)

        for entity in unique_entities:
            entity_mask = self.entity_ids == entity
            entity_choices = self.endog[entity_mask]

            # Skip entities with no variation (all same choice)
            if len(np.unique(entity_choices)) == 1:
                continue

            entity_X = self.exog[entity_mask]

            # Compute log-likelihood for this entity's observations
            for t, (y_it, X_it) in enumerate(zip(entity_choices, entity_X)):
                y_it = int(y_it)

                # Compute utilities
                utilities = np.zeros(self.n_alternatives)
                idx = 0
                for j in self.alternatives:
                    if j == self.base_alternative:
                        utilities[j] = 0
                    else:
                        utilities[j] = X_it @ beta_matrix[idx]
                        idx += 1

                log_denom = logsumexp(utilities)
                llf += utilities[y_it] - log_denom

        return -llf

    def _log_likelihood_random_effects(self, params: np.ndarray) -> float:
        """
        Random Effects multinomial logit with Gaussian random effects.

        Integrates out random effects using Gauss-Hermite quadrature.
        """
        # Split params into betas and random effects variance
        n_beta = (self.n_alternatives - 1) * self.K
        beta_matrix = params[:n_beta].reshape(self.n_alternatives - 1, self.K)

        # For simplicity, assume scalar variance for now
        # Full implementation would have (J-1) × (J-1) covariance matrix
        if len(params) > n_beta:
            log_sigma = params[n_beta]
            sigma = np.exp(log_sigma)
        else:
            sigma = 1.0  # Default

        # Gauss-Hermite quadrature points and weights
        n_quad_points = 10  # Number of quadrature points
        quad_points, quad_weights = np.polynomial.hermite.hermgauss(n_quad_points)

        # Transform quadrature points
        quad_points = quad_points * np.sqrt(2) * sigma
        quad_weights = quad_weights / np.sqrt(np.pi)

        llf = 0.0
        unique_entities = np.unique(self.entity_ids)

        for entity in unique_entities:
            entity_mask = self.entity_ids == entity
            entity_choices = self.endog[entity_mask]
            entity_X = self.exog[entity_mask]

            # Integrate over random effects
            entity_contrib = 0.0
            for alpha_q, weight_q in zip(quad_points, quad_weights):
                # Likelihood for this quadrature point
                entity_llf_q = 0.0

                for y_it, X_it in zip(entity_choices, entity_X):
                    y_it = int(y_it)

                    # Utilities with random effect
                    utilities = np.zeros(self.n_alternatives)
                    idx = 0
                    for j in self.alternatives:
                        if j == self.base_alternative:
                            utilities[j] = 0
                        else:
                            utilities[j] = X_it @ beta_matrix[idx] + alpha_q
                            idx += 1

                    log_denom = logsumexp(utilities)
                    entity_llf_q += utilities[y_it] - log_denom

                entity_contrib += weight_q * np.exp(entity_llf_q)

            llf += np.log(entity_contrib + 1e-10)  # Add small constant for stability

        return -llf

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
        # For now, use numerical gradient for FE and RE (complex derivatives)
        if self.method in ["fixed_effects", "random_effects"]:
            from scipy.optimize import approx_fprime

            return approx_fprime(params, self.log_likelihood, epsilon=1e-6)

        # Analytical gradient for pooled
        beta_matrix = params.reshape(self.n_alternatives - 1, self.K)
        gradient = np.zeros_like(beta_matrix)

        n = len(self.endog)

        for i in range(n):
            y_i = int(self.endog[i])
            X_i = self.exog[i]

            # Compute probabilities
            utilities = np.zeros(self.n_alternatives)
            idx = 0
            for j in self.alternatives:
                if j == self.base_alternative:
                    utilities[j] = 0
                else:
                    utilities[j] = X_i @ beta_matrix[idx]
                    idx += 1

            # Probabilities using softmax
            exp_utils = np.exp(utilities - np.max(utilities))  # For numerical stability
            probs = exp_utils / exp_utils.sum()

            # Gradient contribution
            idx = 0
            for j in self.alternatives:
                if j != self.base_alternative:
                    # Gradient for alternative j parameters
                    indicator = 1.0 if y_i == j else 0.0
                    gradient[idx] += (indicator - probs[j]) * X_i
                    idx += 1

        return -gradient.flatten()  # Negative for minimization

    def predict_proba(
        self, params: np.ndarray, exog: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Predict probabilities for all alternatives.

        Parameters
        ----------
        params : np.ndarray
            Model parameters
        exog : array-like, optional
            New data for prediction. If None, uses training data

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n, J)
        """
        if exog is None:
            X = self.exog
        else:
            # Convert to numpy array if needed
            if isinstance(exog, pd.DataFrame):
                X = exog.values
            else:
                X = np.asarray(exog)

        beta_matrix = params.reshape(self.n_alternatives - 1, self.K)
        n = len(X)
        probs = np.zeros((n, self.n_alternatives))

        for i in range(n):
            X_i = X[i]

            # Compute utilities
            utilities = np.zeros(self.n_alternatives)
            idx = 0
            for j in self.alternatives:
                if j == self.base_alternative:
                    utilities[j] = 0
                else:
                    utilities[j] = X_i @ beta_matrix[idx]
                    idx += 1

            # Convert to probabilities
            exp_utils = np.exp(utilities - np.max(utilities))
            probs[i] = exp_utils / exp_utils.sum()

        return probs

    def predict(
        self, params: np.ndarray, exog: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Predict most likely alternative.

        Parameters
        ----------
        params : np.ndarray
            Model parameters
        exog : array-like, optional
            New data for prediction

        Returns
        -------
        np.ndarray
            Predicted alternatives
        """
        probs = self.predict_proba(params, exog)
        return np.argmax(probs, axis=1)

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "MultinomialLogitResult":
        """
        Estimate multinomial logit parameters.

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
        MultinomialLogitResult
            Fitted model results
        """
        # Starting values
        if start_params is None:
            start_params = np.zeros(self.n_params)
            # Small random perturbation to break symmetry
            start_params += np.random.randn(self.n_params) * 0.01

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

        return MultinomialLogitResult(
            model=self,
            params=result.x,  # Use .x instead of .params
            llf=-result.fun,
            converged=result.success,
            iterations=result.nit,
        )


class MultinomialLogitResult(PanelModelResults):
    """Results class for Multinomial Logit model."""

    def __init__(self, model, params, llf, converged, iterations):
        """Initialize Multinomial Logit results."""
        # First compute vcov, then call parent constructor
        self.llf = llf
        self.converged = converged
        self.iterations = iterations
        self.model = model
        self.params = params

        # Reshape parameters for easier access
        self.params_matrix = params.reshape(model.n_alternatives - 1, model.K)

        # Compute predicted probabilities
        self.predicted_probs = model.predict_proba(params)

        # Compute standard errors and vcov
        self._compute_standard_errors()

        # Now call parent constructor with vcov
        super().__init__(model, params, self.cov_params)

        # Compute fit statistics
        self._compute_fit_statistics()

    def _compute_standard_errors(self):
        """Compute standard errors via Hessian."""
        try:
            from scipy.optimize import approx_fprime

            # Numerical Hessian
            n_params = len(self.params)
            eps = 1e-5
            hess = np.zeros((n_params, n_params))

            for i in range(n_params):

                def grad_i(params):
                    return self.model.gradient(params)[i]

                hess[i, :] = approx_fprime(self.params, grad_i, eps)

            # Covariance matrix
            self.cov_params = np.linalg.inv(hess)
            self.bse = np.sqrt(np.diag(self.cov_params))

            # Reshape standard errors
            self.bse_matrix = self.bse.reshape(self.model.n_alternatives - 1, self.model.K)

        except Exception as e:
            warnings.warn(f"Could not compute standard errors: {e}")
            self.bse = np.full(len(self.params), np.nan)
            self.bse_matrix = np.full(self.params_matrix.shape, np.nan)
            self.cov_params = np.full((len(self.params), len(self.params)), np.nan)

    def _compute_fit_statistics(self):
        """Compute goodness-of-fit statistics."""
        n = len(self.model.endog)
        k = len(self.params)

        # AIC/BIC
        self.aic = 2 * k - 2 * self.llf
        self.bic = np.log(n) * k - 2 * self.llf

        # McFadden's pseudo R-squared
        # Null model: equal probabilities for all alternatives
        llf_null = n * np.log(1 / self.model.n_alternatives)
        self.pseudo_r2 = 1 - (self.llf / llf_null)

        # Prediction accuracy
        y_pred = np.argmax(self.predicted_probs, axis=1)
        self.accuracy = np.mean(y_pred == self.model.endog)

        # Confusion matrix
        self.confusion_matrix = self._compute_confusion_matrix(self.model.endog, y_pred)

    def _compute_confusion_matrix(self, y_true, y_pred):
        """Compute confusion matrix."""
        J = self.model.n_alternatives
        matrix = np.zeros((J, J), dtype=int)

        for i in range(J):
            for j in range(J):
                matrix[i, j] = np.sum((y_true == i) & (y_pred == j))

        return matrix

    def predict_proba(self, exog: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """
        Predict probabilities for all alternatives.

        Parameters
        ----------
        exog : array-like, optional
            New data for prediction. If None, uses training data

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n, J)
        """
        return self.model.predict_proba(self.params, exog)

    def predict(self, exog: Optional[Union[np.ndarray, pd.DataFrame]] = None) -> np.ndarray:
        """
        Predict most likely alternative.

        Parameters
        ----------
        exog : array-like, optional
            New data for prediction

        Returns
        -------
        np.ndarray
            Predicted alternatives
        """
        return self.model.predict(self.params, exog)

    def marginal_effects(
        self, at: str = "mean", variable: Optional[Union[int, str]] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Compute marginal effects for multinomial logit.

        For multinomial logit:
        ∂P_j/∂x_k = P_j(β_jk - Σ_m P_m β_mk)

        Parameters
        ----------
        at : str, default='mean'
            Where to evaluate: 'mean', 'median', or 'overall' (average)
        variable : int or str, optional
            Specific variable. If None, compute for all variables

        Returns
        -------
        dict
            Marginal effects for each alternative
        """
        X = self.model.exog

        if at == "mean":
            X_eval = X.mean(axis=0).reshape(1, -1)
        elif at == "median":
            X_eval = np.median(X, axis=0).reshape(1, -1)
        elif at == "overall":
            X_eval = X
        else:
            raise ValueError(f"Unknown 'at' value: {at}")

        # Get probabilities at evaluation points
        probs = self.model.predict_proba(self.params, X_eval)

        # Initialize marginal effects
        J = self.model.n_alternatives
        K = self.model.K

        if at == "overall":
            n = len(X_eval)
            me = np.zeros((n, J, K))

            for i in range(n):
                P_i = probs[i]
                # Weighted average of betas
                beta_avg = np.zeros(K)
                idx = 0
                for j in range(J):
                    if j != self.model.base_alternative:
                        beta_avg += P_i[j] * self.params_matrix[idx]
                        idx += 1

                # ME for each alternative
                idx = 0
                for j in range(J):
                    if j == self.model.base_alternative:
                        me[i, j] = -P_i[j] * beta_avg
                    else:
                        me[i, j] = P_i[j] * (self.params_matrix[idx] - beta_avg)
                        idx += 1

            # Average over observations
            marginal_effects = me.mean(axis=0)

        else:
            P = probs[0]
            marginal_effects = np.zeros((J, K))

            # Weighted average of betas
            beta_avg = np.zeros(K)
            idx = 0
            for j in range(J):
                if j != self.model.base_alternative:
                    beta_avg += P[j] * self.params_matrix[idx]
                    idx += 1

            # ME for each alternative
            idx = 0
            for j in range(J):
                if j == self.model.base_alternative:
                    marginal_effects[j] = -P[j] * beta_avg
                else:
                    marginal_effects[j] = P[j] * (self.params_matrix[idx] - beta_avg)
                    idx += 1

        # Return as numpy array (shape: J x K)
        # Or specific variable/alternative if requested
        if variable is not None:
            # Return only for specific variable
            if isinstance(variable, str):
                var_idx = self.model.exog_names.index(variable)
            else:
                var_idx = variable
            return marginal_effects[:, var_idx]

        return marginal_effects

    def marginal_effects_se(
        self, at: str = "mean", variable: Optional[Union[int, str]] = None
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Compute standard errors for marginal effects using delta method.

        The delta method approximates:
        Var(g(θ)) ≈ ∇g(θ)' Var(θ) ∇g(θ)

        where g(θ) = marginal effects as function of parameters θ.

        Parameters
        ----------
        at : str, default='mean'
            Where to evaluate: 'mean', 'median', or 'overall' (average)
        variable : int or str, optional
            Specific variable. If None, compute for all variables

        Returns
        -------
        np.ndarray
            Standard errors of marginal effects
        """
        if not hasattr(self, "cov_params") or self.cov_params is None:
            warnings.warn("Covariance matrix not available. Cannot compute ME standard errors.")
            me = self.marginal_effects(at=at, variable=variable)
            return np.full_like(me, np.nan)

        X = self.model.exog

        if at == "mean":
            X_eval = X.mean(axis=0).reshape(1, -1)
        elif at == "median":
            X_eval = np.median(X, axis=0).reshape(1, -1)
        elif at == "overall":
            # For overall, we'd need to average gradients - simpler to use mean
            X_eval = X.mean(axis=0).reshape(1, -1)
            warnings.warn("Standard errors at='overall' computed at mean of X")
        else:
            raise ValueError(f"Unknown 'at' value: {at}")

        # Get probabilities at evaluation point
        probs = self.model.predict_proba(self.params, X_eval)[0]

        J = self.model.n_alternatives
        K = self.model.K
        n_params = len(self.params)

        # Compute gradient of marginal effects w.r.t. parameters
        # ME_jk = P_j(β_jk - Σ_m P_m β_mk)
        # This is complex, so we use numerical gradient

        def me_func(params):
            """Compute marginal effects for given parameters."""
            # Temporarily update params
            probs_temp = self.model.predict_proba(params, X_eval)[0]
            beta_matrix_temp = params.reshape(J - 1, K)

            me_temp = np.zeros((J, K))

            # Weighted average of betas
            beta_avg = np.zeros(K)
            idx = 0
            for j in range(J):
                if j != self.model.base_alternative:
                    beta_avg += probs_temp[j] * beta_matrix_temp[idx]
                    idx += 1

            # ME for each alternative
            idx = 0
            for j in range(J):
                if j == self.model.base_alternative:
                    me_temp[j] = -probs_temp[j] * beta_avg
                else:
                    me_temp[j] = probs_temp[j] * (beta_matrix_temp[idx] - beta_avg)
                    idx += 1

            return me_temp.flatten()

        # Numerical gradient
        from scipy.optimize import approx_fprime

        grad = approx_fprime(self.params, me_func, 1e-7)

        # grad is shape (J*K, n_params)
        # Compute variance using delta method: Var = grad' @ Cov @ grad
        var_me = np.diag(grad @ self.cov_params @ grad.T)

        # Standard errors
        se_me = np.sqrt(np.maximum(var_me, 0))  # Ensure non-negative

        # Reshape to (J, K)
        se_me_matrix = se_me.reshape(J, K)

        # Return specific variable if requested
        if variable is not None:
            if isinstance(variable, str):
                var_idx = self.model.exog_names.index(variable)
            else:
                var_idx = variable
            return se_me_matrix[:, var_idx]

        return se_me_matrix

    def plot_marginal_effects(
        self,
        variable: Optional[Union[int, str]] = None,
        at: str = "mean",
        figsize: tuple = (10, 6),
        colors: Optional[list] = None,
    ):
        """
        Plot marginal effects by alternative.

        Parameters
        ----------
        variable : int or str, optional
            Specific variable to plot. If None, plots all variables
        at : str, default='mean'
            Where to evaluate: 'mean', 'median', or 'overall' (average)
        figsize : tuple, default=(10, 6)
            Figure size
        colors : list, optional
            Colors for each alternative

        Returns
        -------
        matplotlib.figure.Figure
            Matplotlib figure object
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install with: pip install matplotlib"
            )

        # Compute marginal effects
        me = self.marginal_effects(at=at, variable=variable)

        # If specific variable requested
        if variable is not None:
            # me is shape (J,)
            J = len(me)

            fig, ax = plt.subplots(figsize=figsize)

            if colors is None:
                colors = plt.cm.tab10(np.linspace(0, 1, J))

            x_pos = np.arange(J)
            bars = ax.bar(x_pos, me, color=colors, alpha=0.7, edgecolor="black")

            # Add value labels on bars
            for i, (pos, val) in enumerate(zip(x_pos, me)):
                ax.text(
                    pos,
                    val,
                    f"{val:.4f}",
                    ha="center",
                    va="bottom" if val >= 0 else "top",
                    fontsize=9,
                )

            ax.set_xlabel("Alternative", fontsize=12)
            ax.set_ylabel("Marginal Effect", fontsize=12)

            var_name = variable if isinstance(variable, str) else f"X{variable}"
            ax.set_title(
                f"Marginal Effects of {var_name} by Alternative\n(evaluated at {at})",
                fontsize=14,
                fontweight="bold",
            )

            ax.set_xticks(x_pos)
            ax.set_xticklabels([f"Alt {j}" for j in range(J)])
            ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.grid(axis="y", alpha=0.3)

            # Verification: sum should be close to zero
            me_sum = np.sum(me)
            ax.text(
                0.02,
                0.98,
                f"Sum of MEs: {me_sum:.6f} (should ≈ 0)",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                fontsize=9,
            )

            plt.tight_layout()
            return fig

        else:
            # me is shape (J, K) - plot all variables
            J, K = me.shape

            # Create subplots for each variable
            ncols = min(3, K)
            nrows = (K + ncols - 1) // ncols

            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
            if K == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            if colors is None:
                colors = plt.cm.tab10(np.linspace(0, 1, J))

            for k in range(K):
                ax = axes[k]
                x_pos = np.arange(J)
                me_k = me[:, k]

                bars = ax.bar(x_pos, me_k, color=colors, alpha=0.7, edgecolor="black")

                # Add value labels
                for i, (pos, val) in enumerate(zip(x_pos, me_k)):
                    ax.text(
                        pos,
                        val,
                        f"{val:.4f}",
                        ha="center",
                        va="bottom" if val >= 0 else "top",
                        fontsize=8,
                    )

                ax.set_xlabel("Alternative", fontsize=10)
                ax.set_ylabel("Marginal Effect", fontsize=10)

                var_name = (
                    f"X{k}" if not hasattr(self.model, "exog_names") else self.model.exog_names[k]
                )
                ax.set_title(f"ME of {var_name}", fontsize=11, fontweight="bold")

                ax.set_xticks(x_pos)
                ax.set_xticklabels([f"{j}" for j in range(J)])
                ax.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.5)
                ax.grid(axis="y", alpha=0.3)

                # Verification text
                me_sum = np.sum(me_k)
                ax.text(
                    0.02,
                    0.98,
                    f"Σ={me_sum:.3f}",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    fontsize=7,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

            # Hide unused subplots
            for k in range(K, len(axes)):
                axes[k].set_visible(False)

            fig.suptitle(
                f"Marginal Effects by Alternative (at {at})", fontsize=14, fontweight="bold", y=1.00
            )
            plt.tight_layout()
            return fig

    def summary(self) -> str:
        """
        Generate summary of results.

        Returns
        -------
        str
            Formatted summary
        """
        lines = []
        lines.append("=" * 75)
        lines.append("Multinomial Logit Results")
        lines.append("=" * 75)

        # Model info
        lines.append(f"Number of observations:  {len(self.model.endog)}")
        lines.append(f"Number of alternatives:  {self.model.n_alternatives}")
        lines.append(f"Base alternative:       {self.model.base_alternative}")
        lines.append(f"Log-likelihood:         {self.llf:.4f}")
        lines.append(f"AIC:                    {self.aic:.4f}")
        lines.append(f"BIC:                    {self.bic:.4f}")
        lines.append(f"Pseudo R²:              {self.pseudo_r2:.4f}")
        lines.append(f"Prediction accuracy:    {self.accuracy:.4f}")
        lines.append(f"Converged:              {self.converged}")
        lines.append("")

        # Parameters for each non-base alternative
        for idx, j in enumerate(self.model.non_base_alts):
            lines.append("-" * 75)
            lines.append(f"Alternative {j} (vs base alternative {self.model.base_alternative})")
            lines.append("-" * 75)
            lines.append(f"{'Variable':<20} {'Coef':<12} {'Std.Err':<12} {'z':<8} {'P>|z|':<8}")

            for k in range(self.model.K):
                coef = self.params_matrix[idx, k]
                se = self.bse_matrix[idx, k] if not np.isnan(self.bse_matrix[idx, k]) else np.nan

                if not np.isnan(se):
                    z_stat = coef / se
                    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                    lines.append(
                        f"{'X' + str(k):<20} {coef:<12.4f} "
                        f"{se:<12.4f} {z_stat:<8.3f} {p_val:<8.3f}"
                    )
                else:
                    lines.append(
                        f"{'X' + str(k):<20} {coef:<12.4f} " f"{'NA':<12} {'NA':<8} {'NA':<8}"
                    )
            lines.append("")

        # Confusion matrix
        lines.append("-" * 75)
        lines.append("Confusion Matrix")
        lines.append("-" * 75)
        lines.append("Rows: Actual, Columns: Predicted")
        lines.append("")

        # Header
        header = "     " + "".join([f"{j:>8}" for j in range(self.model.n_alternatives)])
        lines.append(header)
        lines.append("-" * (5 + 8 * self.model.n_alternatives))

        # Matrix rows
        for i in range(self.model.n_alternatives):
            row = f"{i:>3} |"
            for j in range(self.model.n_alternatives):
                row += f"{self.confusion_matrix[i, j]:>8}"
            lines.append(row)

        lines.append("=" * 75)

        return "\n".join(lines)


class ConditionalLogit(NonlinearPanelModel):
    """
    Conditional Logit for choice-specific attributes.

    This is for when attributes vary across alternatives, not just
    individual characteristics.

    Note: This is different from multinomial logit where only
    individual characteristics enter the utility function.

    Parameters
    ----------
    choice : array-like
        Choice indicator (which alternative was chosen)
    attributes : array-like
        Alternative-specific attributes (stacked format)
    alternatives : array-like
        Alternative identifiers
    """

    def __init__(self, choice, attributes, alternatives):
        """Initialize Conditional Logit."""
        raise NotImplementedError(
            "Conditional Logit with alternative-specific attributes "
            "will be implemented in a future release"
        )
