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
    """

    def __init__(
        self,
        endog: Union[np.ndarray, pd.Series],
        exog: Union[np.ndarray, pd.DataFrame],
        n_alternatives: Optional[int] = None,
        base_alternative: int = 0,
        entity_col: Optional[str] = None,
        time_col: Optional[str] = None,
    ):
        """Initialize Multinomial Logit model."""
        super().__init__(endog, exog, entity_col, time_col)

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

    def _create_alternative_indices(self):
        """Create indices for alternatives."""
        self.alternatives = np.arange(self.n_alternatives)
        # Non-base alternatives
        self.non_base_alts = np.delete(self.alternatives, self.base_alternative)

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
            Parameter vector of shape ((J-1) × K,)

        Returns
        -------
        float
            Negative log-likelihood
        """
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
            X = self._prepare_data(exog)

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
        super().__init__(model, params, llf)
        self.converged = converged
        self.iterations = iterations

        # Reshape parameters for easier access
        self.params_matrix = params.reshape(model.n_alternatives - 1, model.K)

        # Compute predicted probabilities
        self.predicted_probs = model.predict_proba(params)

        # Compute fit statistics
        self._compute_fit_statistics()
        self._compute_standard_errors()

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

    def marginal_effects(
        self, at: str = "mean", variable: Optional[Union[int, str]] = None
    ) -> Dict[str, np.ndarray]:
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

        # Return as dictionary
        result = {}
        for j in range(J):
            result[f"alternative_{j}"] = marginal_effects[j]

        if variable is not None:
            # Return only for specific variable
            if isinstance(variable, str):
                var_idx = self.model.exog_names.index(variable)
            else:
                var_idx = variable

            for key in result:
                result[key] = result[key][var_idx]

        return result

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
        lines.append("Multinomial Logit Model Results")
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
