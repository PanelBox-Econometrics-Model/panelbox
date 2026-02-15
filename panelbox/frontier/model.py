"""
Main stochastic frontier model class.

This module implements the core StochasticFrontier class which provides
a unified interface for estimating production and cost frontiers with
various distributional assumptions.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .data import (
    DistributionType,
    FrontierType,
    ModelType,
    check_distribution_compatibility,
    prepare_panel_index,
    validate_frontier_data,
)


class StochasticFrontier:
    """Stochastic Frontier Analysis model.

    Estimates production or cost frontiers using maximum likelihood estimation
    with various distributional assumptions for the inefficiency term.

    The basic model structure is:
        Production: y = f(x) * exp(v - u)  =>  ln(y) = β'x + v - u
        Cost:       y = f(x) * exp(v + u)  =>  ln(y) = β'x + v + u

    where:
        v ~ N(0, σ²_v) is random noise
        u ≥ 0 is inefficiency with specified distribution

    Parameters:
        data: DataFrame containing all variables
        depvar: Name of dependent variable (usually in logs)
        exog: List of exogenous variable names
        entity: Entity identifier for panel data (optional)
        time: Time identifier for panel data (optional)
        frontier: Type of frontier ('production' or 'cost')
        dist: Distribution for inefficiency term
        inefficiency_vars: Variables for E[u] in BC95 models (optional)
        het_vars: Variables for heteroskedasticity (optional)
        model_type: Type of panel model (auto-detected from entity/time if None)

    Attributes:
        n_obs: Number of observations
        n_entities: Number of entities (panel data only)
        n_periods: Number of time periods (panel data only)
        n_exog: Number of exogenous variables
        is_balanced: Whether panel is balanced (panel data only)

    Example:
        >>> import pandas as pd
        >>> from panelbox.frontier import StochasticFrontier
        >>>
        >>> # Cross-sectional production frontier
        >>> sf = StochasticFrontier(
        ...     data=df,
        ...     depvar='log_output',
        ...     exog=['log_labor', 'log_capital'],
        ...     frontier='production',
        ...     dist='half_normal'
        ... )
        >>> result = sf.fit(method='mle')
        >>> print(result.summary())
        >>>
        >>> # Get efficiency estimates
        >>> eff = result.efficiency(estimator='bc')
        >>> print(eff.describe())

    References:
        Aigner, D., Lovell, C. K., & Schmidt, P. (1977).
            Formulation and estimation of stochastic frontier production
            function models. Journal of Econometrics, 6(1), 21-37.

        Meeusen, W., & van Den Broeck, J. (1977).
            Efficiency estimation from Cobb-Douglas production functions with
            composed error. International Economic Review, 435-444.

        Battese, G. E., & Coelli, T. J. (1995).
            A model for technical inefficiency effects in a stochastic frontier
            production function for panel data. Empirical Economics, 20(2), 325-332.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        depvar: str,
        exog: List[str],
        entity: Optional[str] = None,
        time: Optional[str] = None,
        frontier: Union[str, FrontierType] = "production",
        dist: Union[str, DistributionType] = "half_normal",
        inefficiency_vars: Optional[List[str]] = None,
        het_vars: Optional[List[str]] = None,
        model_type: Optional[Union[str, ModelType]] = None,
    ):
        """Initialize StochasticFrontier model."""
        # Store basic configuration
        self.depvar = depvar
        self.exog = exog
        self.entity = entity
        self.time = time

        # Convert string enums to proper types
        if isinstance(frontier, str):
            frontier = FrontierType(frontier.lower())
        if isinstance(dist, str):
            dist = DistributionType(dist.lower())

        self.frontier_type = frontier
        self.dist = dist
        self.inefficiency_vars = inefficiency_vars or []
        self.het_vars = het_vars or []

        # Auto-detect model type if not specified
        if model_type is None:
            if entity is None and time is None:
                model_type = ModelType.CROSS_SECTION
            elif entity and not time:
                model_type = ModelType.POOLED
            elif entity and time:
                # Default to Pitt-Lee for simple panel
                model_type = (
                    ModelType.PITT_LEE if not inefficiency_vars else ModelType.BATTESE_COELLI_95
                )
        elif isinstance(model_type, str):
            model_type = ModelType(model_type.lower())

        self.model_type = model_type

        # Check distribution compatibility
        check_distribution_compatibility(self.dist, self.model_type, self.inefficiency_vars)

        # Validate data
        validation_result = validate_frontier_data(
            data=data,
            depvar=depvar,
            exog=exog,
            entity=entity,
            time=time,
            inefficiency_vars=inefficiency_vars,
            het_vars=het_vars,
        )

        # Store validation results
        self._n_obs = validation_result["n_obs"]
        self._n_entities = validation_result["n_entities"]
        self._n_periods = validation_result["n_periods"]
        self._is_balanced = validation_result["is_balanced"]
        self._n_exog = validation_result["n_exog"]

        # Prepare and store data
        self.data = prepare_panel_index(data, entity, time)

        # Extract arrays for estimation
        self._prepare_estimation_arrays()

        # Storage for results
        self._result = None

    def _prepare_estimation_arrays(self) -> None:
        """Prepare numpy arrays for estimation."""
        # Dependent variable
        self.y = self.data[self.depvar].values.astype(float)

        # Exogenous variables (add constant if not present)
        X = self.data[self.exog].values.astype(float)

        # Check if constant is present
        X_std = X.std(axis=0)
        has_constant = np.any(X_std < 1e-10)

        if not has_constant:
            # Add constant
            X = np.column_stack([np.ones(len(X)), X])
            self.exog_names = ["const"] + self.exog
        else:
            self.exog_names = self.exog

        self.X = X

        # Inefficiency variables (for BC95)
        if self.inefficiency_vars:
            Z = self.data[self.inefficiency_vars].values.astype(float)
            # Check for constant
            Z_std = Z.std(axis=0)
            has_constant_z = np.any(Z_std < 1e-10)

            if not has_constant_z:
                Z = np.column_stack([np.ones(len(Z)), Z])
                self.ineff_var_names = ["const"] + self.inefficiency_vars
            else:
                self.ineff_var_names = self.inefficiency_vars

            self.Z = Z
        else:
            self.Z = None
            self.ineff_var_names = []

        # Heteroskedasticity variables (for future use)
        if self.het_vars:
            H = self.data[self.het_vars].values.astype(float)
            self.H = H
            self.het_var_names = self.het_vars
        else:
            self.H = None
            self.het_var_names = []

    @property
    def n_obs(self) -> int:
        """Number of observations."""
        return self._n_obs

    @property
    def n_entities(self) -> Optional[int]:
        """Number of entities (panel data only)."""
        return self._n_entities

    @property
    def n_periods(self) -> Optional[int]:
        """Number of time periods (panel data only)."""
        return self._n_periods

    @property
    def n_exog(self) -> int:
        """Number of exogenous variables (including constant)."""
        return len(self.exog_names)

    @property
    def is_balanced(self) -> Optional[bool]:
        """Whether panel is balanced (panel data only)."""
        return self._is_balanced

    @property
    def is_panel(self) -> bool:
        """Whether model uses panel structure."""
        return self.model_type != ModelType.CROSS_SECTION

    def __repr__(self) -> str:
        """String representation of the model."""
        parts = [
            f"StochasticFrontier(",
            f"  type={self.frontier_type.value}",
            f"  dist={self.dist.value}",
            f"  model={self.model_type.value}",
            f"  n_obs={self.n_obs}",
        ]

        if self.is_panel:
            parts.append(f"  n_entities={self.n_entities}")
            parts.append(f"  n_periods={self.n_periods}")
            parts.append(f"  balanced={self.is_balanced}")

        parts.append(f"  n_exog={self.n_exog}")
        parts.append(")")

        return "\n".join(parts)

    def fit(
        self,
        method: str = "mle",
        start_params: Optional[np.ndarray] = None,
        optimizer: str = "L-BFGS-B",
        maxiter: int = 1000,
        tol: float = 1e-8,
        grid_search: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """Fit the stochastic frontier model.

        Parameters:
            method: Estimation method ('mle' is currently supported)
            start_params: Initial parameter values (auto-computed if None)
            optimizer: Optimization algorithm ('L-BFGS-B', 'Newton-CG', 'BFGS')
            maxiter: Maximum number of iterations
            tol: Convergence tolerance
            grid_search: Whether to use grid search for starting values
            verbose: Whether to print optimization progress
            **kwargs: Additional arguments passed to optimizer

        Returns:
            SFResult object with estimation results

        Raises:
            ValueError: If method is not supported
            RuntimeError: If optimization fails to converge
        """
        if method.lower() != "mle":
            raise ValueError(f"Method '{method}' not supported. Only 'mle' is currently available.")

        # Import here to avoid circular dependency
        from .estimation import estimate_mle

        # Estimate model
        result = estimate_mle(
            model=self,
            start_params=start_params,
            optimizer=optimizer,
            maxiter=maxiter,
            tol=tol,
            grid_search=grid_search,
            verbose=verbose,
            **kwargs,
        )

        # Store result
        self._result = result

        return result

    @property
    def result(self):
        """Access the most recent estimation result.

        Returns:
            SFResult object from last fit() call

        Raises:
            RuntimeError: If fit() has not been called
        """
        if self._result is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._result


def _sign_convention(epsilon: np.ndarray, frontier_type: FrontierType) -> np.ndarray:
    """Apply sign convention for frontier type.

    Production frontier: y = f(x) * exp(v - u) => ε = v - u
    Cost frontier:       y = f(x) * exp(v + u) => ε = v + u

    For estimation, we need the composed error to have the right sign
    relative to the inefficiency term.

    Parameters:
        epsilon: Composed error term (y - X'β)
        frontier_type: Type of frontier

    Returns:
        Signed epsilon for likelihood calculation
    """
    if frontier_type == FrontierType.PRODUCTION:
        # For production: u reduces output, so ε = v - u
        # In likelihood, we want positive ε to indicate low u
        return epsilon
    else:
        # For cost: u increases cost, so ε = v + u
        # Flip sign so likelihood formulas work
        return -epsilon
