"""
Base class for nonlinear panel models estimated via Maximum Likelihood.

This module provides the abstract base class for all MLE-based panel models
including discrete choice and count data models.
"""

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import optimize

from panelbox.core.base_model import PanelModel
from panelbox.core.results import PanelResults
from panelbox.optimization.numerical_grad import approx_gradient, approx_hessian

if TYPE_CHECKING:
    pass


class ConvergenceWarning(UserWarning):
    """Warning for convergence issues in MLE optimization."""

    pass


class NonlinearPanelModel(PanelModel):
    """
    Abstract base class for nonlinear panel models estimated via MLE.

    This class extends PanelModel with Maximum Likelihood Estimation
    functionality, providing:
    - Multiple optimization algorithms (BFGS, Newton, Trust-Region)
    - Multiple starting values to avoid local minima
    - Convergence diagnostics
    - Numerical gradient/Hessian computation
    - Support for constrained optimization

    All discrete choice and count data models inherit from this class.

    Parameters
    ----------
    formula : str
        Model formula in R-style syntax (e.g., "y ~ x1 + x2")
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of the column identifying entities
    time_col : str
        Name of the column identifying time periods
    weights : np.ndarray, optional
        Observation weights

    Attributes
    ----------
    formula : str
        Model formula
    data : PanelData
        Panel data container
    weights : np.ndarray, optional
        Observation weights
    formula_parser : FormulaParser
        Parsed formula object
    _fitted : bool
        Whether model has been fitted
    _results : PanelResults, optional
        Fitted model results
    _optimization_result : scipy.optimize.OptimizeResult, optional
        Raw optimization result
    _optimization_history : list, optional
        History of function evaluations during optimization

    Notes
    -----
    **Subclass Requirements:**

    Subclasses must implement:
    - `_log_likelihood(params)`: Compute log-likelihood at parameter values
    - Optionally `_score(params)`: Analytical gradient (if available)
    - Optionally `_hessian(params)`: Analytical Hessian (if available)

    **Optimization Algorithms:**

    - BFGS: Quasi-Newton method, good default choice
    - Newton: Newton-Raphson, requires Hessian (analytical or numerical)
    - Trust-Constr: Trust-region, for constrained problems

    **Convergence Diagnostics:**

    The optimization result includes:
    - Gradient norm at solution (should be close to 0)
    - Hessian eigenvalues (should be negative for maximum)
    - Condition number of Hessian
    - Number of iterations and function evaluations

    Examples
    --------
    This is an abstract class. See concrete implementations like
    PooledLogit, FixedEffectsLogit for usage examples.

    References
    ----------
    .. [1] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data (2nd ed.). MIT Press. Chapter 15.
    .. [2] Cameron, A. C., & Trivedi, P. K. (2005). Microeconometrics:
           Methods and Applications. Cambridge University Press.
    """

    def __init__(
        self,
        formula: str,
        data: pd.DataFrame,
        entity_col: str,
        time_col: str,
        weights: Optional[np.ndarray] = None,
    ):
        super().__init__(formula, data, entity_col, time_col, weights)

        # Optimization results
        self._optimization_result: Optional[optimize.OptimizeResult] = None
        self._optimization_history: list = []

    @abstractmethod
    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood at parameter values.

        This is the core method that subclasses must implement.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector

        Returns
        -------
        float
            Log-likelihood value (scalar)

        Notes
        -----
        IMPORTANT: This method must return a scalar float, not an array.
        """
        pass

    def _score(self, params: np.ndarray) -> np.ndarray:
        """
        Compute score (gradient of log-likelihood).

        Default implementation uses numerical differentiation.
        Subclasses can override with analytical gradient for speed.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector

        Returns
        -------
        np.ndarray
            Score vector (gradient)
        """
        return approx_gradient(self._log_likelihood, params, method="central")

    def _hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix of log-likelihood.

        Default implementation uses numerical differentiation.
        Subclasses can override with analytical Hessian for speed.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector

        Returns
        -------
        np.ndarray
            Hessian matrix
        """
        return approx_hessian(self._log_likelihood, params, method="central")

    def _get_starting_values(self, n_params: int, method: str = "zeros") -> np.ndarray:
        """
        Generate starting values for optimization.

        Parameters
        ----------
        n_params : int
            Number of parameters
        method : str, default='zeros'
            Method for generating starting values:
            - 'zeros': All zeros
            - 'random': Small random values
            - 'ols': OLS estimates (for linear models)

        Returns
        -------
        np.ndarray
            Starting parameter vector
        """
        if method == "zeros":
            return np.zeros(n_params)
        elif method == "random":
            np.random.seed(42)  # For reproducibility
            return 0.1 * np.random.randn(n_params)
        else:
            return np.zeros(n_params)

    def _check_convergence(
        self, result: optimize.OptimizeResult, params: np.ndarray, verbose: bool = False
    ) -> None:
        """
        Check convergence and issue warnings if needed.

        Parameters
        ----------
        result : scipy.optimize.OptimizeResult
            Optimization result
        params : np.ndarray
            Final parameter estimates
        verbose : bool, default=False
            Print detailed diagnostic information
        """
        # Check if optimization succeeded
        if not result.success:
            warnings.warn(
                f"Optimization may not have converged: {result.message}",
                ConvergenceWarning,
                stacklevel=2,
            )

        # Check gradient norm
        try:
            score = self._score(params)
            grad_norm = np.linalg.norm(score)

            if grad_norm > 1e-3:
                warnings.warn(
                    f"Large gradient norm at solution: {grad_norm:.6f}. "
                    f"Optimization may not have converged.",
                    ConvergenceWarning,
                    stacklevel=2,
                )

            if verbose:
                print(f"Gradient norm: {grad_norm:.6e}")

        except Exception as e:
            if verbose:
                print(f"Could not compute gradient norm: {e}")

        # Check Hessian
        try:
            H = self._hessian(params)
            eigenvalues = np.linalg.eigvalsh(H)

            # For maximum, Hessian should be negative definite
            if np.any(eigenvalues > 1e-10):
                warnings.warn(
                    f"Hessian is not negative definite (has positive eigenvalues). "
                    f"Solution may be a saddle point or minimum instead of maximum.",
                    ConvergenceWarning,
                    stacklevel=2,
                )

            # Check condition number
            cond = np.linalg.cond(H)
            if cond > 1e10:
                warnings.warn(
                    f"Hessian is poorly conditioned (cond={cond:.2e}). "
                    f"Standard errors may be unreliable.",
                    ConvergenceWarning,
                    stacklevel=2,
                )

            if verbose:
                print(
                    f"Hessian eigenvalues (min, max): {eigenvalues.min():.6e}, {eigenvalues.max():.6e}"
                )
                print(f"Hessian condition number: {cond:.6e}")

        except Exception as e:
            if verbose:
                print(f"Could not compute Hessian diagnostics: {e}")

    def fit(
        self,
        method: Literal["bfgs", "newton", "trust-constr"] = "bfgs",
        start_params: Optional[np.ndarray] = None,
        n_starts: int = 1,
        bounds: Optional[Any] = None,
        constraints: Optional[Any] = None,
        maxiter: int = 1000,
        verbose: bool = False,
        **kwargs,
    ) -> PanelResults:
        """
        Fit model via Maximum Likelihood Estimation.

        Parameters
        ----------
        method : {'bfgs', 'newton', 'trust-constr'}, default='bfgs'
            Optimization algorithm:
            - 'bfgs': Quasi-Newton (BFGS), good general-purpose choice
            - 'newton': Newton-Raphson, requires Hessian
            - 'trust-constr': Trust-region, for constrained optimization
        start_params : np.ndarray, optional
            Starting parameter values. If None, uses zeros.
        n_starts : int, default=1
            Number of different starting values to try (to avoid local minima).
            Best result is returned.
        bounds : sequence, optional
            Parameter bounds for constrained optimization
        constraints : dict or sequence, optional
            Constraints for trust-constr method
        maxiter : int, default=1000
            Maximum number of iterations
        verbose : bool, default=False
            Print optimization progress and diagnostics
        **kwargs
            Additional arguments passed to scipy.optimize.minimize

        Returns
        -------
        PanelResults
            Fitted model results

        Notes
        -----
        **Multiple Starting Values:**

        When n_starts > 1, the optimization is run from multiple random
        starting values. The solution with highest log-likelihood is returned.
        This helps avoid local maxima.

        **Optimization Methods:**

        - BFGS is recommended as default (robust and fast)
        - Newton is faster when analytical Hessian is available
        - Trust-constr is needed for constrained problems

        Examples
        --------
        >>> # Basic estimation (BFGS)
        >>> results = model.fit()
        >>>
        >>> # Try multiple starting values
        >>> results = model.fit(n_starts=5)
        >>>
        >>> # Use Newton method with custom starting values
        >>> results = model.fit(method='newton', start_params=beta0)
        >>>
        >>> # Constrained optimization
        >>> results = model.fit(method='trust-constr', bounds=[(0, None), ...])

        See Also
        --------
        scipy.optimize.minimize : Underlying optimization function
        """
        # Build design matrices
        y, X = self.formula_parser.build_design_matrices(self.data.data, return_type="array")
        var_names = self.formula_parser.get_variable_names(self.data.data)

        n_params = X.shape[1]

        # Objective function (negative log-likelihood for minimization)
        def neg_ll(params):
            return -self._log_likelihood(params)

        # Negative score (for minimization)
        def neg_score(params):
            return -self._score(params)

        # Negative Hessian (for minimization)
        def neg_hess(params):
            return -self._hessian(params)

        # Choose scipy method name
        if method == "bfgs":
            scipy_method = "BFGS"
        elif method == "newton":
            scipy_method = "Newton-CG"
        elif method == "trust-constr":
            scipy_method = "trust-constr"
        else:
            raise ValueError(f"method must be 'bfgs', 'newton', or 'trust-constr', got '{method}'")

        # Try multiple starting values
        best_result = None
        best_ll = -np.inf

        for i in range(n_starts):
            # Generate starting values
            if start_params is not None and i == 0:
                x0 = start_params
            elif i == 0:
                x0 = self._get_starting_values(n_params, method="zeros")
            else:
                x0 = self._get_starting_values(n_params, method="random")

            try:
                # Optimize
                if scipy_method == "Newton-CG":
                    result = optimize.minimize(
                        neg_ll,
                        x0,
                        method=scipy_method,
                        jac=neg_score,
                        hess=neg_hess,
                        options={"maxiter": maxiter, "disp": verbose},
                        **kwargs,
                    )
                elif scipy_method == "trust-constr":
                    result = optimize.minimize(
                        neg_ll,
                        x0,
                        method=scipy_method,
                        jac=neg_score,
                        hess=neg_hess,
                        bounds=bounds,
                        constraints=constraints,
                        options={"maxiter": maxiter, "verbose": 2 if verbose else 0},
                        **kwargs,
                    )
                else:  # BFGS
                    result = optimize.minimize(
                        neg_ll,
                        x0,
                        method=scipy_method,
                        jac=neg_score,
                        options={"maxiter": maxiter, "disp": verbose},
                        **kwargs,
                    )

                # Check if this is the best result so far
                ll = -result.fun
                if ll > best_ll:
                    best_ll = ll
                    best_result = result

                if verbose and n_starts > 1:
                    print(f"Start {i+1}/{n_starts}: LL = {ll:.4f}")

            except Exception as e:
                if verbose:
                    print(f"Start {i+1}/{n_starts} failed: {e}")
                continue

        if best_result is None:
            raise RuntimeError("Optimization failed for all starting values")

        # Store optimization result
        self._optimization_result = best_result

        # Get final parameters
        params = best_result.x

        # Check convergence
        self._check_convergence(best_result, params, verbose=verbose)

        # This method will be implemented by subclasses to create appropriate results object
        return self._create_results(params, var_names, y, X)

    def _create_results(
        self, params: np.ndarray, var_names: list, y: np.ndarray, X: np.ndarray
    ) -> PanelResults:
        """
        Create PanelResults object from estimated parameters.

        This is a placeholder that will be overridden by subclasses
        to create appropriate results objects with model-specific
        inference and diagnostics.

        Parameters
        ----------
        params : np.ndarray
            Estimated parameters
        var_names : list
            Parameter names
        y : np.ndarray
            Dependent variable
        X : np.ndarray
            Design matrix

        Returns
        -------
        PanelResults
            Results object
        """
        raise NotImplementedError("Subclasses must implement _create_results")

    def _estimate_coefficients(self) -> np.ndarray:
        """
        Estimate coefficients (implementation of abstract method from PanelModel).

        Returns
        -------
        np.ndarray
            Estimated coefficients
        """
        # This calls fit() and extracts parameters
        if not self._fitted:
            self.fit()

        return self._optimization_result.x if self._optimization_result else np.array([])
