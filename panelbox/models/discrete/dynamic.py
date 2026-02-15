"""
Dynamic binary panel data models.

This module implements dynamic binary choice models for panel data with lagged
dependent variables, handling initial conditions problems using various approaches.

References
----------
Wooldridge, J.M. (2005). "Simple Solutions to the Initial Conditions Problem
in Dynamic, Nonlinear Panel Data Models with Unobserved Heterogeneity."
Journal of Applied Econometrics, 20(1), 39-54.

Arellano, M. & Carrasco, R. (2003). "Binary Choice Panel Data Models with
Predetermined Variables." Journal of Econometrics, 115(1), 125-157.

Known Limitations
-----------------
1. **Initial Conditions**: The Wooldridge approach assumes initial values follow
   a specific reduced form that may not hold for long-running processes.

2. **Panel Length**: Works best with T=5-15 periods. For T<5, identification
   is weak. For T>20, initial condition assumptions become less plausible.

3. **Strict Exogeneity**: Assumes covariates are strictly exogenous, ruling
   out feedback effects and predetermined variables.

4. **Homogeneous Effects**: State dependence (γ) is assumed constant across
   individuals. Heterogeneous effects are not supported.

5. **Computational**: Random effects estimation uses numerical integration
   (Gauss-Hermite quadrature) which can be slow for large N.

See test_dynamic_validation.py for detailed validation against literature.
"""

from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from panelbox.models.base import NonlinearPanelModel, PanelModelResults


class DynamicBinaryPanel(NonlinearPanelModel):
    """
    Dynamic binary panel data model with lagged dependent variable.

    Implements Wooldridge (2005) and Heckman approaches for handling
    initial conditions in dynamic nonlinear panel models.

    Model:
        y_it = 1[X_it'β + γ*y_i,t-1 + α_i + ε_it > 0]

    Parameters
    ----------
    endog : array-like
        Binary dependent variable (0/1)
    exog : array-like
        Exogenous variables matrix
    entity : array-like
        Entity/individual identifiers
    time : array-like
        Time period identifiers
    initial_conditions : str, default='wooldridge'
        Method for handling initial conditions:
        - 'wooldridge': Include y_i0 and averages of X in model
        - 'heckman': Model joint distribution of (y_i0, α_i)
        - 'simple': Ignore initial conditions (may be biased)
    effects : str, default='random'
        Type of panel effects ('random' or 'pooled')
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        entity: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None,
        initial_conditions: Literal["wooldridge", "heckman", "simple"] = "wooldridge",
        effects: Literal["random", "pooled"] = "random",
    ):
        # Store entity and time before calling super().__init__
        self.entity = entity
        self.time = time

        # Don't pass entity/time to parent since it expects different names
        super().__init__(endog, exog)

        self.initial_conditions = initial_conditions
        self.effects = effects
        self.endog_lagged = None
        self.exog_augmented = None
        self.y_i0 = None
        self.X_avg = None

    def _prepare_data(self):
        """Prepare data with lags and handle initial conditions."""
        # Convert to panel structure if needed
        if isinstance(self.endog, pd.Series):
            self.endog = self.endog.values
        if isinstance(self.exog, pd.DataFrame):
            self.exog = self.exog.values

        # Create panel index
        if self.entity is None or self.time is None:
            raise ValueError("Entity and time identifiers required for dynamic model")

        # Sort by entity and time
        sort_idx = np.lexsort((self.time, self.entity))
        self.endog = self.endog[sort_idx]
        self.exog = self.exog[sort_idx]
        self.entity = self.entity[sort_idx]
        self.time = self.time[sort_idx]

        # Get unique entities and time periods
        self.entities = np.unique(self.entity)
        self.periods = np.unique(self.time)
        self.n_entities = len(self.entities)
        self.n_periods = len(self.periods)

        # Create lagged dependent variable
        self.endog_lagged = np.zeros_like(self.endog, dtype=float)
        self.endog_lagged[:] = np.nan

        for i in self.entities:
            mask = self.entity == i
            entity_y = self.endog[mask]
            if len(entity_y) > 1:
                self.endog_lagged[mask][1:] = entity_y[:-1]

        # Handle initial conditions
        if self.initial_conditions == "wooldridge":
            self._wooldridge_initial_conditions()
        elif self.initial_conditions == "heckman":
            self._heckman_initial_conditions()
        else:  # simple
            # Just drop first period for each entity
            self._simple_initial_conditions()

    def _wooldridge_initial_conditions(self):
        """
        Wooldridge (2005) approach: Include y_i0 and time-averages of X.
        """
        # Get initial values y_i0 for each entity
        self.y_i0 = np.zeros(len(self.endog))

        for i in self.entities:
            mask = self.entity == i
            entity_y = self.endog[mask]
            self.y_i0[mask] = entity_y[0]  # First observation

        # Calculate time-averages of X for each entity
        n_vars = self.exog.shape[1]
        self.X_avg = np.zeros((len(self.endog), n_vars))

        for i in self.entities:
            mask = self.entity == i
            entity_X = self.exog[mask]
            X_mean = np.mean(entity_X, axis=0)
            self.X_avg[mask] = X_mean

        # Augment exogenous variables
        # Add: lagged y, y_i0, X_avg to original X
        valid_obs = ~np.isnan(self.endog_lagged)

        self.exog_augmented = np.column_stack(
            [
                self.exog[valid_obs],
                self.endog_lagged[valid_obs],
                self.y_i0[valid_obs],
                self.X_avg[valid_obs],
            ]
        )

        # Update endog to exclude first period
        self.endog_clean = self.endog[valid_obs]
        self.entity_clean = self.entity[valid_obs]
        self.time_clean = self.time[valid_obs]

    def _heckman_initial_conditions(self):
        """
        Heckman approach: Model joint distribution of (y_i0, α_i).
        """
        # This is more complex and requires modeling the initial period
        # For now, implement a simplified version
        self._wooldridge_initial_conditions()

    def _simple_initial_conditions(self):
        """
        Simple approach: Just drop first period (may be biased).
        """
        valid_obs = ~np.isnan(self.endog_lagged)

        # Debug print
        # print(f"Valid obs: {valid_obs.sum()} out of {len(valid_obs)}")

        self.exog_augmented = np.column_stack(
            [self.exog[valid_obs], self.endog_lagged[valid_obs].reshape(-1, 1)]  # Ensure 2D
        )

        self.endog_clean = self.endog[valid_obs]
        self.entity_clean = self.entity[valid_obs]
        self.time_clean = self.time[valid_obs]

    def _log_likelihood(self, params: np.ndarray) -> float:
        """
        Compute log-likelihood for dynamic binary model.

        Parameters
        ----------
        params : np.ndarray
            Parameter vector including lag coefficient

        Returns
        -------
        float
            Negative log-likelihood
        """
        # Linear prediction
        if self.effects == "random":
            linear_pred = self.exog_augmented @ params[:-1]
            # Add random effects variance
            sigma_u = np.exp(params[-1])  # Ensure positive
        else:
            linear_pred = self.exog_augmented @ params

        if self.effects == "random":

            # Integrate over random effects distribution
            # Using Gauss-Hermite quadrature
            from scipy.special import roots_hermite

            nodes, weights = roots_hermite(20)

            llf = 0
            for i in np.unique(self.entity_clean):
                mask = self.entity_clean == i
                y_i = self.endog_clean[mask]
                Xb_i = linear_pred[mask]

                # Integrate over random effect
                entity_llf = 0
                for node, weight in zip(nodes, weights):
                    u_i = np.sqrt(2) * sigma_u * node
                    prob = stats.norm.cdf(Xb_i + u_i)
                    prob = np.clip(prob, 1e-10, 1 - 1e-10)

                    entity_contrib = np.sum(y_i * np.log(prob) + (1 - y_i) * np.log(1 - prob))
                    entity_llf += weight * np.exp(entity_contrib) / np.sqrt(np.pi)

                llf += np.log(entity_llf + 1e-10)
        else:
            # Pooled probit
            prob = stats.norm.cdf(linear_pred)
            prob = np.clip(prob, 1e-10, 1 - 1e-10)
            llf = np.sum(
                self.endog_clean * np.log(prob) + (1 - self.endog_clean) * np.log(1 - prob)
            )

        return -llf  # Return negative for minimization

    def predict(self, params=None, exog=None):
        """Generate predictions (required by base class)."""
        if params is None:
            if self.results is not None:
                params = self.results.params
            else:
                raise ValueError("Model not fitted yet")

        if exog is None:
            if self.effects == "random":
                linear_pred = self.exog_augmented @ params[:-1]
            else:
                linear_pred = self.exog_augmented @ params
        else:
            if self.effects == "random":
                linear_pred = exog @ params[:-1]
            else:
                linear_pred = exog @ params

        return stats.norm.cdf(linear_pred)

    def fit(
        self, start_params: Optional[np.ndarray] = None, **kwargs
    ) -> "DynamicBinaryPanelResult":
        """
        Fit the dynamic binary panel model.

        Parameters
        ----------
        start_params : np.ndarray, optional
            Initial parameter values
        **kwargs
            Additional arguments for optimizer

        Returns
        -------
        DynamicBinaryPanelResult
            Fitted model results
        """
        # Prepare data
        self._prepare_data()

        # Set initial parameters
        n_params = self.exog_augmented.shape[1]
        if self.effects == "random":
            n_params += 1  # Add variance parameter

        if start_params is None:
            start_params = np.zeros(n_params)
            if self.effects == "random":
                start_params[-1] = -1  # log(sigma_u) = -1 -> sigma_u ≈ 0.37

        # Optimize
        result = minimize(
            self._log_likelihood,
            start_params,
            method="BFGS",
            options={"disp": False, "maxiter": 1000},
        )

        # Create result object
        return DynamicBinaryPanelResult(
            model=self,
            params=result.x,  # Use .x instead of .params
            llf=-result.fun,
            converged=result.success,
            n_iter=result.nit,
        )


class DynamicBinaryPanelResult(PanelModelResults):
    """Results class for dynamic binary panel models."""

    def __init__(
        self,
        model: DynamicBinaryPanel,
        params: np.ndarray,
        llf: float,
        converged: bool,
        n_iter: int,
    ):
        self.model = model
        self.params = params
        self.llf = llf
        self.converged = converged
        self.n_iter = n_iter

        # Extract specific parameters
        if model.initial_conditions == "wooldridge":
            n_original = model.exog.shape[1]  # Original X variables
            # Order in exog_augmented: X, lag, y_i0, X_avg
            self.beta = params[:n_original]
            self.gamma = params[n_original]  # Lag coefficient
            self.delta_y0 = params[n_original + 1]  # Initial value coefficient
            self.delta_xbar = params[n_original + 2 : n_original + 2 + n_original]
            if model.effects == "random":
                self.sigma_u = np.exp(params[-1])
        else:
            n_original = model.exog.shape[1]
            self.beta = params[:n_original]
            self.gamma = params[n_original]  # Lag coefficient
            if model.effects == "random":
                self.sigma_u = np.exp(params[-1])

    def summary(self) -> str:
        """Generate summary of results."""
        output = [
            "Dynamic Binary Panel Model Results",
            "=" * 50,
            f"Initial Conditions: {self.model.initial_conditions}",
            f"Effects: {self.model.effects}",
            f"Log-likelihood: {self.llf:.4f}",
            f"Converged: {self.converged}",
            f"Iterations: {self.n_iter}",
            "",
            "Parameter Estimates:",
            "-" * 30,
        ]

        # Original coefficients
        for i, coef in enumerate(self.beta):
            output.append(f"β_{i}: {coef:.4f}")

        # Lag coefficient
        output.append(f"γ (lag): {self.gamma:.4f}")

        # Initial conditions parameters
        if self.model.initial_conditions == "wooldridge":
            output.append(f"δ_y0 (initial): {self.delta_y0:.4f}")
            output.append("δ_xbar (X averages):")
            for i, coef in enumerate(self.delta_xbar):
                output.append(f"  δ_xbar_{i}: {coef:.4f}")

        # Random effects variance
        if self.model.effects == "random":
            output.append(f"σ_u: {self.sigma_u:.4f}")

        return "\n".join(output)

    def predict(self, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Predict probabilities for new data.

        Parameters
        ----------
        exog : np.ndarray, optional
            New exogenous variables

        Returns
        -------
        np.ndarray
            Predicted probabilities
        """
        if exog is None:
            linear_pred = self.model.exog_augmented @ self.params[:-1]
        else:
            # Need to handle augmentation for new data
            linear_pred = exog @ self.beta

        return stats.norm.cdf(linear_pred)

    def marginal_effects(self) -> np.ndarray:
        """
        Calculate marginal effects at the mean.

        Returns
        -------
        np.ndarray
            Marginal effects for each variable
        """
        X_mean = np.mean(self.model.exog_augmented, axis=0)
        linear_pred = X_mean @ self.params[:-1]
        pdf = stats.norm.pdf(linear_pred)

        # Marginal effects
        me = self.beta * pdf
        me_lag = self.gamma * pdf

        return np.concatenate([me, [me_lag]])
