"""
Honoré (1992) Trimmed LAD Estimator for Fixed Effects Tobit.

This module implements the semiparametric trimmed LAD estimator
proposed by Honoré (1992) for fixed effects Tobit models.

Author: PanelBox Developers
License: MIT
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import optimize


class ExperimentalWarning(UserWarning):
    """Warning for experimental features."""

    pass


@dataclass
class HonoreResults:
    """Results container for Honoré estimator."""

    params: np.ndarray
    converged: bool
    n_iter: int
    n_obs: int
    n_entities: int
    n_trimmed: int


class HonoreTrimmedEstimator:
    """
    Honoré (1992) Trimmed LAD Estimator for Fixed Effects Tobit.

    WARNING: This estimator is computationally intensive and experimental.
    Use with caution, especially for large datasets.

    This is a semiparametric estimator that does not require distributional
    assumptions on the individual effects α_i or the error terms ε_it.

    The estimator uses pairwise differences to eliminate fixed effects and
    applies trimming to handle the censoring problem.

    Parameters
    ----------
    endog : array-like
        The dependent variable (N*T, 1)
    exog : array-like
        The independent variables (N*T, K)
    groups : array-like
        Group identifiers for panel structure
    time : array-like
        Time identifiers for panel structure
    censoring_point : float, default=0
        The censoring threshold for left censoring

    References
    ----------
    Honoré, B. E. (1992). Trimmed LAD and least squares estimation of
    truncated and censored regression models with fixed effects.
    Econometrica, 60(3), 533-565.
    """

    def __init__(
        self,
        endog: np.ndarray,
        exog: np.ndarray,
        groups: np.ndarray,
        time: np.ndarray,
        censoring_point: float = 0.0,
    ):
        warnings.warn(
            "The Honoré Trimmed Estimator is computationally intensive "
            "and experimental. It may take a long time to converge for "
            "moderate to large datasets. Use with caution.",
            ExperimentalWarning,
        )

        self.endog = np.asarray(endog).flatten()
        self.exog = np.asarray(exog)
        self.groups = np.asarray(groups).flatten()
        self.time = np.asarray(time).flatten()
        self.censoring_point = censoring_point

        # Check dimensions
        n_obs = len(self.endog)
        if self.exog.shape[0] != n_obs:
            raise ValueError("Dimension mismatch between endog and exog")
        if len(self.groups) != n_obs or len(self.time) != n_obs:
            raise ValueError("groups and time must have same length as endog")

        self.n_obs = n_obs
        self.n_features = self.exog.shape[1]

        # Get unique entities
        self.entities = np.unique(self.groups)
        self.n_entities = len(self.entities)

        # Prepare panel structure
        self._prepare_panel_data()

    def _prepare_panel_data(self):
        """Organize data by entity for efficient pairwise differencing."""
        self.entity_data = {}

        for entity_id in self.entities:
            mask = self.groups == entity_id
            entity_times = self.time[mask]
            sorted_idx = np.argsort(entity_times)

            self.entity_data[entity_id] = {
                "y": self.endog[mask][sorted_idx],
                "X": self.exog[mask][sorted_idx],
                "time": entity_times[sorted_idx],
                "n_obs": mask.sum(),
            }

    def _create_pairwise_differences(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create pairwise differences for all entities and time periods.

        Returns
        -------
        delta_y : array-like
            Pairwise differences of dependent variable
        delta_X : array-like
            Pairwise differences of independent variables
        trim_indicator : array-like
            Indicator for trimming (1 = keep, 0 = trim)
        """
        delta_y_list = []
        delta_X_list = []
        trim_list = []

        for entity_id in self.entities:
            data = self.entity_data[entity_id]
            n_periods = data["n_obs"]

            if n_periods < 2:
                continue  # Need at least 2 periods for differencing

            y_i = data["y"]
            X_i = data["X"]

            # Create all pairwise differences for this entity
            for t in range(n_periods):
                for s in range(t + 1, n_periods):
                    # Difference between periods t and s
                    delta_y_ts = y_i[t] - y_i[s]
                    delta_X_ts = X_i[t] - X_i[s]

                    # Trimming rule from Honoré (1992)
                    # Keep observation if at least one period is uncensored
                    # and the difference is informative
                    y_t_censored = np.abs(y_i[t] - self.censoring_point) < 1e-10
                    y_s_censored = np.abs(y_i[s] - self.censoring_point) < 1e-10

                    # Trimming: keep if not both censored at same point
                    trim_indicator = not (y_t_censored and y_s_censored)

                    delta_y_list.append(delta_y_ts)
                    delta_X_list.append(delta_X_ts)
                    trim_list.append(float(trim_indicator))

        if len(delta_y_list) == 0:
            raise ValueError("No valid pairwise differences found. Check your data.")

        return (np.array(delta_y_list), np.array(delta_X_list), np.array(trim_list))

    def _trimmed_lad_objective(
        self, beta: np.ndarray, delta_y: np.ndarray, delta_X: np.ndarray, trim_indicator: np.ndarray
    ) -> float:
        """
        Compute the trimmed LAD objective function.

        Parameters
        ----------
        beta : array-like
            Parameter vector
        delta_y : array-like
            Pairwise differences of y
        delta_X : array-like
            Pairwise differences of X
        trim_indicator : array-like
            Trimming indicators

        Returns
        -------
        objective : float
            Trimmed LAD objective value
        """
        residuals = delta_y - delta_X @ beta
        trimmed_residuals = residuals * trim_indicator

        # LAD objective: sum of absolute values
        return np.sum(np.abs(trimmed_residuals))

    def _trimmed_lad_gradient(
        self, beta: np.ndarray, delta_y: np.ndarray, delta_X: np.ndarray, trim_indicator: np.ndarray
    ) -> np.ndarray:
        """
        Compute the subgradient of the trimmed LAD objective.

        Note: LAD is not differentiable everywhere, so we use a subgradient.
        """
        residuals = delta_y - delta_X @ beta
        trimmed_residuals = residuals * trim_indicator

        # Subgradient of |r| is sign(r)
        # Use smooth approximation near zero
        epsilon = 1e-8
        signs = trimmed_residuals / np.maximum(np.abs(trimmed_residuals), epsilon)

        # Gradient w.r.t. beta
        gradient = -delta_X.T @ (signs * trim_indicator)

        return gradient

    def fit(
        self,
        start_params: Optional[np.ndarray] = None,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        tol: float = 1e-6,
        verbose: bool = True,
        **kwargs,
    ) -> HonoreResults:
        """
        Fit the Honoré Trimmed LAD estimator.

        Parameters
        ----------
        start_params : array-like, optional
            Starting values for optimization
        method : str, default='L-BFGS-B'
            Optimization method (should handle non-smooth objectives)
        maxiter : int, default=500
            Maximum iterations
        tol : float, default=1e-6
            Convergence tolerance
        verbose : bool, default=True
            Print progress information
        **kwargs : dict
            Additional arguments for optimizer

        Returns
        -------
        results : HonoreResults
            Fitted model results
        """
        if verbose:
            print("Creating pairwise differences...")

        # Create pairwise differences
        delta_y, delta_X, trim_indicator = self._create_pairwise_differences()

        n_pairs = len(delta_y)
        n_trimmed = int(n_pairs - trim_indicator.sum())

        if verbose:
            print(f"Total pairwise differences: {n_pairs}")
            print(f"Trimmed observations: {n_trimmed}")
            print(f"Retained observations: {int(trim_indicator.sum())}")

        # Starting values
        if start_params is None:
            # Use simple OLS on untrimmed differences as starting values
            kept = trim_indicator > 0.5
            if kept.sum() > self.n_features:
                # Least squares on retained observations
                start_params = np.linalg.lstsq(delta_X[kept], delta_y[kept], rcond=None)[0]
            else:
                start_params = np.zeros(self.n_features)

        if verbose:
            print(f"Starting optimization with method: {method}")

        # Optimize
        options = kwargs.pop("options", {})
        options.setdefault("maxiter", maxiter)
        options.setdefault("disp", verbose)
        options.setdefault("ftol", tol)
        options.setdefault("gtol", tol)

        # For non-smooth optimization, might need special handling
        if method == "L-BFGS-B":
            # L-BFGS-B can handle non-smooth objectives reasonably well
            result = optimize.minimize(
                lambda b: self._trimmed_lad_objective(b, delta_y, delta_X, trim_indicator),
                start_params,
                method=method,
                jac=lambda b: self._trimmed_lad_gradient(b, delta_y, delta_X, trim_indicator),
                options=options,
                **kwargs,
            )
        else:
            # For other methods, might not use gradient
            result = optimize.minimize(
                lambda b: self._trimmed_lad_objective(b, delta_y, delta_X, trim_indicator),
                start_params,
                method=method,
                options=options,
                **kwargs,
            )

        if verbose:
            if result.success:
                print(f"Optimization converged in {result.nit} iterations")
            else:
                print(f"Optimization failed: {result.message}")

        # Store results
        self.params = result.x
        self.converged = result.success
        self.n_iter = result.nit if hasattr(result, "nit") else 0

        results = HonoreResults(
            params=self.params,
            converged=self.converged,
            n_iter=self.n_iter,
            n_obs=self.n_obs,
            n_entities=self.n_entities,
            n_trimmed=n_trimmed,
        )

        return results

    def predict(self, exog: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate predictions from the fitted model.

        Note: This returns linear predictions X'β without accounting
        for fixed effects (which are differenced out) or censoring.

        Parameters
        ----------
        exog : array-like, optional
            Explanatory variables for prediction (uses training data if None)

        Returns
        -------
        predictions : array-like
            Linear predictions X'β
        """
        if not hasattr(self, "params"):
            raise ValueError("Model must be fitted before prediction")

        if exog is None:
            exog = self.exog

        return exog @ self.params

    def summary(self) -> str:
        """Generate model summary."""
        if not hasattr(self, "params"):
            return "Model has not been fitted yet"

        summary_lines = [
            "=" * 60,
            "Honoré Trimmed LAD Estimator Results",
            "=" * 60,
            f"Number of obs:        {self.n_obs:>8d}",
            f"Number of groups:     {self.n_entities:>8d}",
            f"Converged:            {self.converged}",
            f"Iterations:           {self.n_iter:>8d}",
            "-" * 60,
            "Coefficients:",
            "-" * 60,
        ]

        for i in range(self.n_features):
            summary_lines.append(f"beta_{i:<3d}              {self.params[i]:>11.4f}")

        summary_lines.extend(
            [
                "-" * 60,
                "Note: This is a semiparametric estimator.",
                "Standard errors are not computed.",
                "Fixed effects are eliminated by differencing.",
                "=" * 60,
            ]
        )

        return "\n".join(summary_lines)
