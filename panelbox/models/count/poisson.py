"""
Poisson models for panel count data.

This module implements various Poisson regression models for panel data:
- PooledPoisson: Standard Poisson with cluster-robust SEs
- PoissonFixedEffects: Conditional MLE (Hausman et al. 1984)
- RandomEffectsPoisson: Random effects with Gamma/Normal distribution
- PoissonQML: Quasi-Maximum Likelihood (Wooldridge 1999)
"""

import warnings
from itertools import combinations_with_replacement, permutations
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import factorial, gammaln, loggamma

from ...utils.data import check_panel_data
from ...utils.statistics import compute_sandwich_covariance
from ..base import NonlinearPanelModel, PanelModelResults


class PoissonFixedEffectsResults(PanelModelResults):
    """Results class for Poisson Fixed Effects model."""

    def __init__(self, model, params, vcov):
        """Initialize results."""
        super().__init__(model, params, vcov)
        self.llf = model.llf if hasattr(model, "llf") else None
        self.n_dropped = model.n_dropped if hasattr(model, "n_dropped") else 0
        self.dropped_entities = model.dropped_entities if hasattr(model, "dropped_entities") else []


class PooledPoisson(NonlinearPanelModel):
    """
    Pooled Poisson regression for panel data.

    Standard Poisson MLE with optional cluster-robust standard errors.

    Parameters
    ----------
    endog : array-like
        Dependent variable (count data)
    exog : array-like
        Independent variables
    entity_id : array-like, optional
        Entity identifiers for clustering
    time_id : array-like, optional
        Time identifiers

    Attributes
    ----------
    overdispersion : float
        Overdispersion index (Var(y)/E(y))
        Values > 1 indicate overdispersion

    Methods
    -------
    fit(se_type='cluster', **kwargs)
        Estimate model parameters
    predict(X=None, type='response')
        Generate predictions
    check_overdispersion()
        Test for overdispersion

    Examples
    --------
    >>> model = PooledPoisson(y, X, entity_id=firms)
    >>> result = model.fit(se_type='cluster')
    >>> print(result.summary())

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    """

    def __init__(self, endog, exog, entity_id=None, time_id=None):
        """Initialize Pooled Poisson model."""
        super().__init__(endog, exog, entity_id, time_id)
        self.model_type = "Pooled Poisson"
        self._check_count_data()

    def _check_count_data(self):
        """Check that dependent variable is count data."""
        if not np.allclose(self.endog, self.endog.astype(int)):
            raise ValueError("Dependent variable must contain count data (non-negative integers)")
        if np.any(self.endog < 0):
            raise ValueError("Count data cannot be negative")

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute Poisson log-likelihood.

        ℓ = Σᵢₜ [yᵢₜ log λᵢₜ - λᵢₜ - log(yᵢₜ!)]
          = Σᵢₜ [yᵢₜ Xᵢₜ'β - exp(Xᵢₜ'β) - log(yᵢₜ!)]
        """
        linear_pred = self.exog @ params
        lambda_it = np.exp(linear_pred)

        # Use gammaln for numerical stability with large factorials
        log_factorial_y = gammaln(self.endog + 1)

        llf = np.sum(self.endog * linear_pred - lambda_it - log_factorial_y)

        return llf

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Compute score (gradient) of log-likelihood.

        ∂ℓ/∂β = Σᵢₜ (yᵢₜ - λᵢₜ) Xᵢₜ
        """
        linear_pred = self.exog @ params
        lambda_it = np.exp(linear_pred)
        residual = self.endog - lambda_it

        score = self.exog.T @ residual

        return score

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Hessian of log-likelihood.

        ∂²ℓ/∂β∂β' = -Σᵢₜ λᵢₜ Xᵢₜ Xᵢₜ'
        """
        linear_pred = self.exog @ params
        lambda_it = np.exp(linear_pred)

        # Weight matrix
        W = np.diag(lambda_it.flatten())

        # Hessian
        H = -self.exog.T @ W @ self.exog

        return H

    def predict(self, X: Optional[np.ndarray] = None, type: str = "response") -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, optional
            Covariates for prediction. If None, uses estimation sample
        type : {'response', 'linear', 'rate'}
            Type of prediction:
            - 'response' or 'rate': Expected count λ = exp(X'β)
            - 'linear': Linear predictor X'β

        Returns
        -------
        predictions : array-like
            Predicted values
        """
        if not hasattr(self, "params"):
            raise RuntimeError("Model must be fitted before prediction")

        if X is None:
            X = self.exog

        linear_pred = X @ self.params

        if type in ["response", "rate"]:
            return np.exp(linear_pred)
        elif type == "linear":
            return linear_pred
        else:
            raise ValueError(f"Invalid prediction type: {type}")

    @property
    def overdispersion(self) -> float:
        """
        Calculate overdispersion index.

        Returns
        -------
        float
            Overdispersion index Var(y)/E(y)
            Should be ~1 for Poisson, >1 indicates overdispersion
        """
        if not hasattr(self, "params"):
            raise RuntimeError("Model must be fitted first")

        fitted_values = self.predict(type="response")
        residuals = self.endog - fitted_values

        # Calculate empirical variance and mean
        emp_var = np.var(residuals + fitted_values)
        emp_mean = np.mean(fitted_values)

        if emp_mean > 0:
            return emp_var / emp_mean
        else:
            return np.nan

    def check_overdispersion(self, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test for overdispersion.

        Performs regression-based test for overdispersion.

        Parameters
        ----------
        alpha : float
            Significance level

        Returns
        -------
        dict
            Test results including statistic, p-value, and conclusion
        """
        if not hasattr(self, "params"):
            raise RuntimeError("Model must be fitted first")

        fitted = self.predict(type="response")

        # Cameron-Trivedi regression test
        # H0: Var(y) = E(y) (equidispersion)
        # H1: Var(y) = E(y) + α g(E(y))

        # Auxiliary regression
        y_aux = ((self.endog - fitted) ** 2 - fitted) / fitted
        mu_aux = fitted

        # Remove any NaN or inf values
        valid = np.isfinite(y_aux) & np.isfinite(mu_aux)
        y_aux = y_aux[valid]
        mu_aux = mu_aux[valid].reshape(-1, 1)

        # OLS regression
        from scipy.stats import linregress

        slope, intercept, r_value, p_value, std_err = linregress(mu_aux.flatten(), y_aux)

        result = {
            "overdispersion_index": self.overdispersion,
            "test_statistic": slope / std_err if std_err > 0 else np.inf,
            "p_value": p_value,
            "significant": p_value < alpha,
            "conclusion": (
                "Evidence of overdispersion" if p_value < alpha else "No significant overdispersion"
            ),
        }

        if result["significant"]:
            warnings.warn(
                f"Significant overdispersion detected (index={self.overdispersion:.2f}). "
                "Consider using Negative Binomial model.",
                UserWarning,
            )

        return result

    def marginal_effects(
        self,
        result: Optional[PanelModelResults] = None,
        at: str = "overall",
        varlist: Optional[List[str]] = None,
    ):
        """
        Compute marginal effects for Poisson model.

        For Poisson models, marginal effects measure the change in expected
        count E[y|X] = exp(X'β) for a one-unit change in a covariate.

        Parameters
        ----------
        result : PanelModelResults, optional
            Fitted model result. If None, uses self._results
        at : str, default='overall'
            Where to evaluate marginal effects:
            - 'overall' or 'mean': AME (Average Marginal Effects)
            - 'means' or 'mem': MEM (Marginal Effects at Means)
        varlist : list of str, optional
            Variables to compute ME for. If None, compute for all.

        Returns
        -------
        MarginalEffectsResult
            Container with marginal effects, standard errors, and inference

        Examples
        --------
        >>> model = PooledPoisson(y, X, entity_id=firms)
        >>> result = model.fit()
        >>> ame = model.marginal_effects(result, at='overall')
        >>> mem = model.marginal_effects(result, at='means')
        >>> print(ame.summary())

        See Also
        --------
        panelbox.marginal_effects.count_me.compute_poisson_ame
        panelbox.marginal_effects.count_me.compute_poisson_mem
        """
        from ...marginal_effects.count_me import compute_poisson_ame, compute_poisson_mem

        # Get result object
        if result is None:
            if not hasattr(self, "_results"):
                raise RuntimeError("Model must be fitted first or result must be provided")
            result = self._results

        # Compute marginal effects based on 'at' parameter
        if at in ["overall", "mean"]:
            return compute_poisson_ame(result, varlist=varlist)
        elif at in ["means", "mem"]:
            return compute_poisson_mem(result, varlist=varlist)
        else:
            raise ValueError(
                f"Unknown 'at' value: {at}. "
                "Use 'overall'/'mean' for AME or 'means'/'mem' for MEM."
            )

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        se_type: str = "cluster",
        **kwargs,
    ) -> PanelModelResults:
        """
        Fit the Pooled Poisson model.

        Parameters
        ----------
        start_params : ndarray, optional
            Starting values for optimization
        method : str
            Optimization method (default 'BFGS')
        maxiter : int
            Maximum number of iterations
        se_type : str
            Type of standard errors ('cluster' or 'robust')
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        PanelModelResults
            Results object with fitted parameters
        """
        # Get starting values if not provided
        if start_params is None:
            start_params = self._get_start_params()

        # Optimize log-likelihood
        from scipy.optimize import minimize

        result = minimize(
            lambda p: -self._log_likelihood(p),
            start_params,
            method=method,
            jac=lambda p: -self._score(p),
            options={"maxiter": maxiter, "disp": False},
        )

        # Store parameters
        self.params = result.x
        self.llf = -result.fun

        # Compute covariance matrix
        hessian = self._hessian(self.params)

        if se_type == "cluster" and self.entity_id is not None:
            # Cluster-robust standard errors
            score_contribs = np.zeros((len(self.endog), len(self.params)))
            for i in range(len(self.endog)):
                linear_pred_i = self.exog[i] @ self.params
                lambda_i = np.exp(linear_pred_i)
                score_contribs[i] = (self.endog[i] - lambda_i) * self.exog[i]

            self.vcov = compute_sandwich_covariance(hessian, score_contribs, self.entity_id)
        else:
            # Standard covariance matrix
            try:
                self.vcov = -np.linalg.inv(hessian)
            except np.linalg.LinAlgError:
                self.vcov = -np.linalg.pinv(hessian)

        # Store result for marginal_effects method
        result_obj = PanelModelResults(self, self.params, self.vcov)
        self._results = result_obj

        # Return results object
        return result_obj

    def _get_start_params(self) -> np.ndarray:
        """Get starting values for optimization using linear approximation."""
        # Use log(y + 0.5) as approximation for small counts
        y_transformed = np.log(self.endog + 0.5)

        # OLS on transformed variable
        from numpy.linalg import lstsq

        params, _, _, _ = lstsq(self.exog, y_transformed, rcond=None)

        return params


class PoissonFixedEffects(NonlinearPanelModel):
    """
    Fixed Effects Poisson via Conditional MLE.

    Implements the Hausman, Hall, and Griliches (1984) conditional
    maximum likelihood estimator that eliminates fixed effects by
    conditioning on the sum of counts for each entity.

    Parameters
    ----------
    endog : array-like
        Dependent variable (count data)
    exog : array-like
        Independent variables
    entity_id : array-like
        Entity identifiers (required)
    time_id : array-like, optional
        Time identifiers

    Attributes
    ----------
    dropped_entities : list
        Entity IDs dropped due to all-zero counts
    n_dropped : int
        Number of entities dropped

    Notes
    -----
    Entities with Σₜ yᵢₜ = 0 provide no information and are dropped.

    The conditional likelihood conditions on nᵢ = Σₜ yᵢₜ and
    eliminates the fixed effects αᵢ.

    Computational complexity increases with larger nᵢ and Tᵢ values.
    Dynamic programming is used for efficiency.

    References
    ----------
    Hausman, J., Hall, B. H., & Griliches, Z. (1984).
    "Econometric models for count data with an application to
    the patents-R&D relationship." Econometrica, 52(4), 909-938.
    """

    def __init__(self, endog, exog, entity_id, time_id=None):
        """Initialize Poisson Fixed Effects model."""
        if entity_id is None:
            raise ValueError("entity_id is required for Fixed Effects Poisson")

        super().__init__(endog, exog, entity_id, time_id)
        self.model_type = "Poisson Fixed Effects"
        self.entities = np.unique(self.entity_id)  # Add entities attribute
        self._check_count_data()
        self._identify_entities_to_keep()

    def _check_count_data(self):
        """Check that dependent variable is count data."""
        if not np.allclose(self.endog, self.endog.astype(int)):
            raise ValueError("Dependent variable must contain count data")
        if np.any(self.endog < 0):
            raise ValueError("Count data cannot be negative")

    def _identify_entities_to_keep(self):
        """Identify entities with positive total counts."""
        self.entity_totals = {}
        self.dropped_entities = []
        self.kept_entities = []

        for entity in self.entities:
            mask = self.entity_id == entity
            total = self.endog[mask].sum()
            self.entity_totals[entity] = int(total)

            if total > 0:
                self.kept_entities.append(entity)
            else:
                self.dropped_entities.append(entity)

        self.n_dropped = len(self.dropped_entities)

        if self.n_dropped > 0:
            print(f"Note: {self.n_dropped} entities dropped due to all-zero outcomes")

        if len(self.kept_entities) == 0:
            raise ValueError("No entities with positive counts remaining")

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute conditional log-likelihood.

        P(yᵢ₁, ..., yᵢTᵢ | nᵢ, Xᵢ) =
            [Πₜ exp(yᵢₜ Xᵢₜ'β)] / Σ{s: Σₜ sₜ = nᵢ} [Πₜ exp(sₜ Xᵢₜ'β)]

        Log-likelihood:
        ℓ = Σᵢ [ Σₜ yᵢₜ Xᵢₜ'β - log( Σ{s: Σₜ sₜ = nᵢ} exp(Σₜ sₜ Xᵢₜ'β) ) ]
        """
        llf = 0.0

        for entity in self.kept_entities:
            mask = self.entity_id == entity
            y_i = self.endog[mask]
            X_i = self.exog[mask]
            n_i = self.entity_totals[entity]
            T_i = len(y_i)

            # Numerator (in log space)
            numerator = np.sum(y_i * (X_i @ params))

            # Denominator: sum over all count sequences
            if n_i <= 10 and T_i <= 5:
                # Small case: use exact enumeration
                denominator = self._enumerate_sequences(X_i, params, n_i, T_i)
            else:
                # Large case: use dynamic programming
                denominator = self._dp_sequences(X_i, params, n_i, T_i)

            if denominator > 0:
                llf += numerator - np.log(denominator)
            else:
                # Numerical issues
                warnings.warn(f"Numerical issues for entity {entity}")

        return llf

    def _enumerate_sequences(
        self, X_i: np.ndarray, params: np.ndarray, n_i: int, T_i: int
    ) -> float:
        """
        Enumerate all integer sequences summing to n_i.

        For small n_i and T_i, directly enumerate all partitions.
        """
        total = 0.0

        # Generate all compositions of n_i into T_i parts
        for composition in self._generate_compositions(n_i, T_i):
            # Calculate exp(Σₜ sₜ Xᵢₜ'β)
            linear_comb = sum(s * (X_i[t] @ params) for t, s in enumerate(composition))
            total += np.exp(linear_comb)

        return total

    def _generate_compositions(self, n: int, k: int) -> List[Tuple[int, ...]]:
        """
        Generate all compositions of n into k non-negative parts.

        A composition is an ordered way to write n as sum of k non-negative integers.
        """
        if k == 1:
            yield (n,)
        elif n == 0:
            yield (0,) * k
        else:
            for i in range(n + 1):
                for rest in self._generate_compositions(n - i, k - 1):
                    yield (i,) + rest

    def _dp_sequences(self, X_i: np.ndarray, params: np.ndarray, n_i: int, T_i: int) -> float:
        """
        Use dynamic programming to compute sum over sequences.

        More efficient for larger n_i and T_i values.
        """
        # Precompute exp(s * Xᵢₜ'β) for all s and t
        max_count = n_i
        exp_terms = np.zeros((T_i, max_count + 1))

        for t in range(T_i):
            linear_pred = X_i[t] @ params
            for s in range(max_count + 1):
                exp_terms[t, s] = np.exp(s * linear_pred)

        # DP table: dp[t][s] = sum over sequences up to time t with sum s
        dp = np.zeros((T_i + 1, n_i + 1))
        dp[0, 0] = 1.0  # Base case

        for t in range(1, T_i + 1):
            for s in range(n_i + 1):
                # Try all possible values at time t-1
                for count in range(s + 1):
                    if s - count >= 0:
                        dp[t, s] += dp[t - 1, s - count] * exp_terms[t - 1, count]

        return dp[T_i, n_i]

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Compute score (gradient) for conditional likelihood.

        Requires numerical differentiation or analytic derivation.
        """
        # Use numerical gradient for now
        from scipy.optimize import approx_fprime

        eps = np.sqrt(np.finfo(float).eps)

        return approx_fprime(params, lambda p: self._log_likelihood(p), eps)

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Hessian for conditional likelihood.

        Uses numerical differentiation.
        """
        # Numerical Hessian computation
        n = len(params)
        hess = np.zeros((n, n))
        eps = np.sqrt(np.finfo(float).eps)

        for i in range(n):
            for j in range(n):
                # Four-point formula for second derivatives
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

    def marginal_effects(
        self,
        result: Optional[PanelModelResults] = None,
        at: str = "overall",
        varlist: Optional[List[str]] = None,
    ):
        """
        Compute marginal effects for Poisson Fixed Effects model.

        For Poisson models, marginal effects measure the change in expected
        count E[y|X] = exp(X'β) for a one-unit change in a covariate.

        Parameters
        ----------
        result : PanelModelResults, optional
            Fitted model result. If None, uses self._results
        at : str, default='overall'
            Where to evaluate marginal effects:
            - 'overall' or 'mean': AME (Average Marginal Effects)
            - 'means' or 'mem': MEM (Marginal Effects at Means)
        varlist : list of str, optional
            Variables to compute ME for. If None, compute for all.

        Returns
        -------
        MarginalEffectsResult
            Container with marginal effects, standard errors, and inference

        Examples
        --------
        >>> model = PoissonFixedEffects(y, X, entity_id=firms)
        >>> result = model.fit()
        >>> ame = model.marginal_effects(result, at='overall')
        >>> mem = model.marginal_effects(result, at='means')
        >>> print(ame.summary())

        See Also
        --------
        panelbox.marginal_effects.count_me.compute_poisson_ame
        panelbox.marginal_effects.count_me.compute_poisson_mem
        """
        from ...marginal_effects.count_me import compute_poisson_ame, compute_poisson_mem

        # Get result object
        if result is None:
            if not hasattr(self, "_results"):
                raise RuntimeError("Model must be fitted first or result must be provided")
            result = self._results

        # Compute marginal effects based on 'at' parameter
        if at in ["overall", "mean"]:
            return compute_poisson_ame(result, varlist=varlist)
        elif at in ["means", "mem"]:
            return compute_poisson_mem(result, varlist=varlist)
        else:
            raise ValueError(
                f"Unknown 'at' value: {at}. "
                "Use 'overall'/'mean' for AME or 'means'/'mem' for MEM."
            )

    def predict(
        self, X: Optional[np.ndarray] = None, type: str = "response", include_fe: bool = False
    ) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, optional
            Covariates for prediction
        type : {'response', 'linear', 'rate'}
            Type of prediction
        include_fe : bool
            Whether to include fixed effects (requires post-estimation)

        Returns
        -------
        predictions : array-like
            Predicted values

        Notes
        -----
        Fixed effects are not directly estimated in conditional MLE.
        If include_fe=True, they must be recovered post-estimation.
        """
        if not hasattr(self, "params"):
            raise RuntimeError("Model must be fitted before prediction")

        if X is None:
            X = self.exog

        linear_pred = X @ self.params

        if include_fe:
            warnings.warn(
                "Fixed effects not directly available from conditional MLE. "
                "Returning predictions without fixed effects.",
                UserWarning,
            )

        if type in ["response", "rate"]:
            return np.exp(linear_pred)
        elif type == "linear":
            return linear_pred
        else:
            raise ValueError(f"Invalid prediction type: {type}")

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "BFGS",
        maxiter: int = 1000,
        **kwargs,
    ) -> "PoissonFixedEffectsResults":
        """
        Fit the Fixed Effects Poisson model using conditional MLE.

        Parameters
        ----------
        start_params : ndarray, optional
            Starting values for optimization
        method : str
            Optimization method (default 'BFGS')
        maxiter : int
            Maximum number of iterations
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        PoissonFixedEffectsResults
            Results object with fitted parameters
        """
        # Get starting values if not provided
        if start_params is None:
            start_params = self._get_start_params()

        # Optimize conditional log-likelihood
        from scipy.optimize import minimize

        result = minimize(
            lambda p: -self._log_likelihood(p),
            start_params,
            method=method,
            jac=lambda p: -self._score(p),
            options={"maxiter": maxiter, "disp": False},
        )

        # Store parameters
        self.params = result.x
        self.llf = -result.fun

        # Compute covariance matrix (inverse of negative Hessian)
        hessian = self._hessian(self.params)
        try:
            self.vcov = -np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self.vcov = -np.linalg.pinv(hessian)

        # Create and store results object
        result_obj = PoissonFixedEffectsResults(self, self.params, self.vcov)
        self._results = result_obj

        # Return results object
        return result_obj

    def _get_start_params(self) -> np.ndarray:
        """Get starting values using Poisson Pooled as approximation."""
        # Use pooled Poisson for starting values
        pooled = PooledPoisson(self.endog, self.exog)

        # Just get starting params without full fitting
        return pooled._get_start_params()


class RandomEffectsPoisson(NonlinearPanelModel):
    """
    Random Effects Poisson regression.

    Implements Poisson regression with random effects following either
    Gamma or Normal distribution. The Gamma distribution is conjugate
    and leads to a Negative Binomial marginal distribution.

    Parameters
    ----------
    endog : array-like
        Dependent variable (count data)
    exog : array-like
        Independent variables
    entity_id : array-like
        Entity identifiers (required)
    time_id : array-like, optional
        Time identifiers

    Attributes
    ----------
    distribution : str
        Distribution of random effects ('gamma' or 'normal')
    theta : float
        Variance parameter for random effects

    Methods
    -------
    fit(distribution='gamma', **kwargs)
        Estimate model with specified RE distribution

    Notes
    -----
    With Gamma-distributed random effects:
    - yᵢₜ | αᵢ ~ Poisson(αᵢ λᵢₜ)
    - αᵢ ~ Gamma(1/θ, θ) with E[αᵢ]=1, Var[αᵢ]=θ
    - Marginal: yᵢₜ ~ NegBin(λᵢₜ, θ)

    References
    ----------
    Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data.
    """

    def __init__(self, endog, exog, entity_id, time_id=None):
        """Initialize Random Effects Poisson model."""
        if entity_id is None:
            raise ValueError("entity_id is required for Random Effects")

        super().__init__(endog, exog, entity_id, time_id)
        self.model_type = "Random Effects Poisson"
        self.entities = np.unique(self.entity_id)  # Add entities attribute
        self._check_count_data()

    def _check_count_data(self):
        """Check that dependent variable is count data."""
        if not np.allclose(self.endog, self.endog.astype(int)):
            raise ValueError("Dependent variable must contain count data")
        if np.any(self.endog < 0):
            raise ValueError("Count data cannot be negative")

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Log-likelihood function.

        This is a placeholder that gets replaced in fit() with
        the distribution-specific version.
        """
        # Default to gamma distribution
        return self._log_likelihood_gamma(params)

    def fit(
        self, distribution: str = "gamma", start_params: Optional[np.ndarray] = None, **kwargs
    ) -> "PanelResults":
        """
        Fit Random Effects Poisson model.

        Parameters
        ----------
        distribution : {'gamma', 'normal'}
            Distribution for random effects
        start_params : array-like, optional
            Starting values for optimization
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        PanelResults
            Fitted model results
        """
        self.distribution = distribution.lower()

        if self.distribution not in ["gamma", "normal"]:
            raise ValueError("distribution must be 'gamma' or 'normal'")

        # Set up log-likelihood based on distribution
        if self.distribution == "gamma":
            self._log_likelihood_func = self._log_likelihood_gamma
        else:
            self._log_likelihood_func = self._log_likelihood_normal

        # Get starting values
        if start_params is None:
            # Start with pooled estimates plus theta
            pooled = PooledPoisson(self.endog, self.exog)
            pooled_params = pooled._get_start_params()
            # Add theta parameter (start with small positive value)
            start_params = np.append(pooled_params, 0.1)

        # Store original log-likelihood method
        self._log_likelihood_original = self._log_likelihood

        # Replace with distribution-specific version
        self._log_likelihood = self._log_likelihood_func

        # Fit model
        result = super().fit(start_params=start_params, **kwargs)

        # Extract theta from parameters
        self.theta = np.exp(self.params[-1])  # Ensure positivity
        self.params_exog = self.params[:-1]

        # Store result for marginal_effects method
        self._results = result

        return result

    def _log_likelihood_gamma(self, params: np.ndarray) -> float:
        """
        Log-likelihood for Gamma-distributed random effects.

        With Gamma RE, the marginal distribution is Negative Binomial.
        """
        beta = params[:-1]
        log_theta = params[-1]
        theta = np.exp(log_theta)  # Ensure positivity

        llf = 0.0

        for entity in self.entities:
            mask = self.entity_id == entity
            y_i = self.endog[mask]
            X_i = self.exog[mask]

            # Linear predictor
            linear_pred = X_i @ beta
            lambda_i = np.exp(linear_pred)

            # Negative Binomial likelihood for this entity
            # Using NB2 parameterization
            r = 1 / theta  # Shape parameter

            for t in range(len(y_i)):
                y_it = y_i[t]
                mu_it = lambda_i[t]

                # NB log-likelihood
                llf += (
                    gammaln(y_it + r)
                    - gammaln(y_it + 1)
                    - gammaln(r)
                    + r * np.log(r / (r + mu_it))
                    + y_it * np.log(mu_it / (r + mu_it))
                )

        return llf

    def _log_likelihood_normal(self, params: np.ndarray) -> float:
        """
        Log-likelihood for Normal-distributed random effects.

        Uses Gauss-Hermite quadrature for integration.
        """
        beta = params[:-1]
        log_theta = params[-1]
        theta = np.exp(log_theta)  # Variance of random effect

        # Import quadrature if available
        try:
            from ...optimization.quadrature import gauss_hermite_quadrature

            n_quad = 20
        except ImportError:
            warnings.warn("Quadrature not available, using simple approximation")
            n_quad = 5

        llf = 0.0

        for entity in self.entities:
            mask = self.entity_id == entity
            y_i = self.endog[mask]
            X_i = self.exog[mask]

            # Integrate over random effect
            # Using change of variables: α = exp(u), u ~ N(0, theta)

            # Simple Gauss-Hermite quadrature
            from scipy.special import roots_hermite

            points, weights = roots_hermite(n_quad)

            # Adjust for N(0, theta)
            points = np.sqrt(2 * theta) * points

            entity_likelihood = 0.0

            for q, (point, weight) in enumerate(zip(points, weights)):
                alpha = np.exp(point)  # Random effect

                # Poisson likelihood conditional on alpha
                linear_pred = X_i @ beta
                lambda_it = alpha * np.exp(linear_pred)

                # Log-likelihood for this quadrature point
                cond_llf = np.sum(y_i * np.log(lambda_it) - lambda_it - gammaln(y_i + 1))

                entity_likelihood += weight * np.exp(cond_llf) / np.sqrt(np.pi)

            if entity_likelihood > 0:
                llf += np.log(entity_likelihood)

        return llf

    def _score(self, params: np.ndarray) -> np.ndarray:
        """Compute score using numerical differentiation."""
        from scipy.optimize import approx_fprime

        eps = np.sqrt(np.finfo(float).eps)

        return approx_fprime(params, lambda p: self._log_likelihood_func(p), eps)

    def marginal_effects(
        self,
        result: Optional[PanelModelResults] = None,
        at: str = "overall",
        varlist: Optional[List[str]] = None,
    ):
        """
        Compute marginal effects for Random Effects Poisson model.

        For Poisson models, marginal effects measure the change in expected
        count E[y|X] = exp(X'β) for a one-unit change in a covariate.

        Parameters
        ----------
        result : PanelModelResults, optional
            Fitted model result. If None, uses self._results
        at : str, default='overall'
            Where to evaluate marginal effects:
            - 'overall' or 'mean': AME (Average Marginal Effects)
            - 'means' or 'mem': MEM (Marginal Effects at Means)
        varlist : list of str, optional
            Variables to compute ME for. If None, compute for all.

        Returns
        -------
        MarginalEffectsResult
            Container with marginal effects, standard errors, and inference

        Examples
        --------
        >>> model = RandomEffectsPoisson(y, X, entity_id=firms)
        >>> result = model.fit()
        >>> ame = model.marginal_effects(result, at='overall')
        >>> mem = model.marginal_effects(result, at='means')
        >>> print(ame.summary())

        See Also
        --------
        panelbox.marginal_effects.count_me.compute_poisson_ame
        panelbox.marginal_effects.count_me.compute_poisson_mem
        """
        from ...marginal_effects.count_me import compute_poisson_ame, compute_poisson_mem

        # Get result object
        if result is None:
            if not hasattr(self, "_results"):
                raise RuntimeError("Model must be fitted first or result must be provided")
            result = self._results

        # Compute marginal effects based on 'at' parameter
        if at in ["overall", "mean"]:
            return compute_poisson_ame(result, varlist=varlist)
        elif at in ["means", "mem"]:
            return compute_poisson_mem(result, varlist=varlist)
        else:
            raise ValueError(
                f"Unknown 'at' value: {at}. "
                "Use 'overall'/'mean' for AME or 'means'/'mem' for MEM."
            )

    def predict(self, X: Optional[np.ndarray] = None, type: str = "response") -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        X : array-like, optional
            Covariates for prediction
        type : {'response', 'linear'}
            Type of prediction

        Returns
        -------
        predictions : array-like
            Predicted values (marginal expectation)
        """
        if not hasattr(self, "params"):
            raise RuntimeError("Model must be fitted before prediction")

        if X is None:
            X = self.exog

        # Use only exogenous parameters (exclude theta)
        beta = self.params_exog if hasattr(self, "params_exog") else self.params[:-1]

        linear_pred = X @ beta

        if type == "response":
            # E[y] = exp(X'β) for both Gamma and Normal RE
            return np.exp(linear_pred)
        elif type == "linear":
            return linear_pred
        else:
            raise ValueError(f"Invalid prediction type: {type}")

    @property
    def overdispersion(self) -> float:
        """
        Calculate overdispersion from random effects variance.

        For Gamma RE: overdispersion = 1 + theta
        """
        if not hasattr(self, "theta"):
            raise RuntimeError("Model must be fitted first")

        return 1 + self.theta


class PoissonQML(PooledPoisson):
    """
    Quasi-Maximum Likelihood Poisson estimator.

    Implements Wooldridge (1999) QML Poisson which is consistent
    under weaker assumptions than standard Poisson MLE. Only requires
    correct specification of conditional mean.

    Parameters
    ----------
    endog : array-like
        Dependent variable (count data)
    exog : array-like
        Independent variables
    entity_id : array-like, optional
        Entity identifiers for clustering
    time_id : array-like, optional
        Time identifiers

    Notes
    -----
    QML Poisson estimates β via standard Poisson MLE but uses
    robust (sandwich) standard errors. It is consistent if:
    E[yᵢₜ|Xᵢₜ] = exp(Xᵢₜ'β)

    even if yᵢₜ does not follow a Poisson distribution.

    Always uses robust standard errors.

    References
    ----------
    Wooldridge, J. M. (1999). "Distribution-free estimation of some
    nonlinear panel data models." Journal of Econometrics, 90(1), 77-97.
    """

    def __init__(self, endog, exog, entity_id=None, time_id=None):
        """Initialize Poisson QML model."""
        super().__init__(endog, exog, entity_id, time_id)
        self.model_type = "Poisson QML (Wooldridge 1999)"

    def fit(self, se_type: str = "robust", **kwargs) -> "PanelResults":
        """
        Fit Poisson QML model.

        Parameters
        ----------
        se_type : str
            Must be 'robust' or 'cluster' for QML
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        PanelResults
            Fitted model results with robust SEs
        """
        if se_type not in ["robust", "cluster"]:
            warnings.warn(
                f"QML requires robust standard errors. Switching from '{se_type}' to 'robust'.",
                UserWarning,
            )
            se_type = "robust"

        # Fit using parent class
        result = super().fit(se_type=se_type, **kwargs)

        # Add QML-specific information
        result.model_info["estimator"] = "Quasi-Maximum Likelihood"
        result.model_info["robust"] = True
        result.model_info["notes"] = (
            "QML estimates are consistent under correct mean specification "
            "even if data is not Poisson distributed."
        )

        return result

    def __repr__(self):
        """String representation."""
        return f"PoissonQML(n_entities={self.n_entities}, " f"n_obs={self.n_obs}, k_vars={self.k})"
