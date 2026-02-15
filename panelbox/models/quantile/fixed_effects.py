"""
Fixed Effects Quantile Regression using Koenker (2004) penalty method.

This module implements fixed effects quantile regression for panel data using
L1 penalization on the fixed effects to handle the incidental parameters problem
when T is small.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panelbox.core.panel_data import PanelData

from .base import QuantilePanelModel


class FixedEffectsQuantile(QuantilePanelModel):
    """
    Fixed Effects Quantile Regression using Koenker (2004) penalty method.

    Adds L1 penalty to fixed effects to handle the incidental
    parameters problem in QR with fixed T.

    Parameters
    ----------
    data : PanelData
        Panel data object
    formula : str, optional
        Model formula
    tau : float or array-like
        Quantile(s) to estimate
    lambda_fe : float or 'auto'
        Penalty parameter for fixed effects.
        If 'auto', select via cross-validation.
    """

    def __init__(
        self,
        data: PanelData,
        formula: Optional[str] = None,
        tau: Union[float, List[float]] = 0.5,
        lambda_fe: Union[float, str] = "auto",
    ):
        super().__init__(data, formula, tau)
        self.lambda_fe = lambda_fe

        # Setup data matrices
        self._setup_data()

        # Setup entity structure
        self.n_entities = len(np.unique(self.entity_ids))
        self._setup_entity_indicators()

    def _setup_data(self):
        """Setup data matrices from panel data."""
        # Get dependent and independent variables
        if self.formula:
            # Parse formula to get variables
            self._parse_formula()
            self.y = self.data.df[self.dependent_var].values
            self.X = self.data.df[self.independent_vars].values
        else:
            # Use all variables except the first as X
            self.y = self.data.df.iloc[:, 0].values
            self.X = self.data.df.iloc[:, 1:].values

        # Add constant if not present
        if not np.any(np.all(self.X == self.X[0], axis=0)):
            self.X = np.column_stack([np.ones(len(self.y)), self.X])

        self.nobs, self.k_exog = self.X.shape
        self.entity_ids = self.data.entity_ids.values

    def _setup_entity_indicators(self):
        """Create mapping from observations to entities."""
        self.entity_map = {}
        for i, entity_id in enumerate(np.unique(self.entity_ids)):
            self.entity_map[entity_id] = i

        self.obs_to_entity = np.array([self.entity_map[eid] for eid in self.entity_ids])

    def _objective(self, params: np.ndarray, tau: float) -> float:
        """
        Compute objective function (without penalty).

        Used for compatibility with base class methods.
        """
        # Split parameters
        beta = params[: self.k_exog]

        # Compute residuals (no fixed effects for base objective)
        residuals = self.y - self.X @ beta

        # Check loss
        return np.sum(self.check_loss(residuals, tau))

    def _objective_penalized(self, params: np.ndarray, tau: float, lambda_val: float) -> float:
        """
        Penalized objective function.

        L(β,α) = Σᵢₜ ρτ(yᵢₜ - xᵢₜ'β - αᵢ) + λ Σᵢ |αᵢ|
        """
        # Split parameters
        beta = params[: self.k_exog]
        alpha = params[self.k_exog :]

        # Expand alpha to match observations
        alpha_expanded = alpha[self.obs_to_entity]

        # Compute residuals
        residuals = self.y - self.X @ beta - alpha_expanded

        # Check loss
        check_loss = np.sum(self.check_loss(residuals, tau))

        # L1 penalty on fixed effects
        penalty = lambda_val * np.sum(np.abs(alpha))

        return check_loss + penalty

    def _gradient_penalized(self, params: np.ndarray, tau: float, lambda_val: float) -> np.ndarray:
        """Subgradient of penalized objective."""
        beta = params[: self.k_exog]
        alpha = params[self.k_exog :]

        # Expand alpha
        alpha_expanded = alpha[self.obs_to_entity]

        # Residuals
        residuals = self.y - self.X @ beta - alpha_expanded

        # Subgradient of check loss
        psi = self.check_loss_gradient(residuals, tau)

        # Gradient w.r.t. beta
        grad_beta = -self.X.T @ psi

        # Gradient w.r.t. alpha (aggregate by entity)
        grad_alpha = np.zeros(self.n_entities)
        for i in range(self.n_entities):
            mask = self.obs_to_entity == i
            grad_alpha[i] = -np.sum(psi[mask])

        # Add subgradient of penalty
        grad_alpha += lambda_val * np.sign(alpha)

        return np.concatenate([grad_beta, grad_alpha])

    def _select_lambda_cv(
        self,
        tau: float,
        lambda_grid: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        verbose: bool = False,
    ) -> float:
        """
        Select penalty parameter via K-fold cross-validation.

        Uses entity-based splitting to preserve panel structure.
        """
        if lambda_grid is None:
            # Automatic grid based on data scale
            lambda_max = self._compute_lambda_max(tau)
            lambda_grid = np.logspace(np.log10(lambda_max * 0.001), np.log10(lambda_max), 20)

        if verbose:
            print(f"Cross-validation for λ selection (τ={tau})")
            print(
                f"Testing {len(lambda_grid)} values from {lambda_grid[0]:.4f} to {lambda_grid[-1]:.4f}"
            )

        # Entity-based CV splits
        entities = np.unique(self.entity_ids)
        np.random.shuffle(entities)
        fold_size = len(entities) // cv_folds

        cv_scores = []

        for lambda_val in lambda_grid:
            fold_losses = []

            for fold in range(cv_folds):
                # Split entities
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < cv_folds - 1 else len(entities)
                val_entities = entities[start_idx:end_idx]
                train_entities = np.setdiff1d(entities, val_entities)

                # Create masks
                train_mask = np.isin(self.entity_ids, train_entities)
                val_mask = np.isin(self.entity_ids, val_entities)

                # Fit on training data
                result_train = self._fit_with_lambda(
                    tau,
                    lambda_val,
                    X_train=self.X[train_mask],
                    y_train=self.y[train_mask],
                    entity_train=self.obs_to_entity[train_mask],
                )

                # Evaluate on validation data
                # For validation, use average of training alphas as prediction for new entities
                alpha_mean = np.mean(result_train["alpha"])
                pred_val = self.X[val_mask] @ result_train["beta"] + alpha_mean
                residuals_val = self.y[val_mask] - pred_val
                val_loss = np.mean(self.check_loss(residuals_val, tau))

                fold_losses.append(val_loss)

            cv_scores.append(np.mean(fold_losses))

            if verbose:
                print(f"  λ = {lambda_val:.4f}: CV score = {cv_scores[-1]:.6f}")

        # Select lambda with minimum CV score
        best_idx = np.argmin(cv_scores)
        best_lambda = lambda_grid[best_idx]

        if verbose:
            print(f"Selected λ = {best_lambda:.4f} (CV score = {cv_scores[best_idx]:.6f})")

        self.cv_results_ = {
            "lambda_grid": lambda_grid,
            "cv_scores": cv_scores,
            "best_lambda": best_lambda,
        }

        return best_lambda

    def _compute_lambda_max(self, tau: float) -> float:
        """
        Compute maximum reasonable lambda value.

        This is the smallest λ that sets all αᵢ = 0.
        """
        # Gradient at α = 0
        residuals = self.y - self.X @ np.zeros(self.k_exog)
        psi = self.check_loss_gradient(residuals, tau)

        # Maximum gradient per entity
        max_grads = []
        for i in range(self.n_entities):
            mask = self.obs_to_entity == i
            max_grads.append(np.abs(np.sum(psi[mask])))

        return np.max(max_grads)

    def _fit_with_lambda(
        self,
        tau: float,
        lambda_val: float,
        X_train: Optional[np.ndarray] = None,
        y_train: Optional[np.ndarray] = None,
        entity_train: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Fit model with specific lambda value."""
        if X_train is None:
            X_train = self.X
            y_train = self.y
            entity_train = self.obs_to_entity
            n_entities_train = self.n_entities
        else:
            # Count unique entities in training data
            n_entities_train = len(np.unique(entity_train))
            # Remap entity indices to be consecutive
            unique_entities = np.unique(entity_train)
            entity_map = {e: i for i, e in enumerate(unique_entities)}
            entity_train = np.array([entity_map[e] for e in entity_train])

        # Initial values
        x0 = np.zeros(X_train.shape[1] + n_entities_train)

        # Create temporary model for optimization
        temp_self = type("temp", (), {})()
        temp_self.X = X_train
        temp_self.y = y_train
        temp_self.k_exog = X_train.shape[1]
        temp_self.n_entities = n_entities_train
        temp_self.obs_to_entity = entity_train
        temp_self.check_loss = self.check_loss
        temp_self.check_loss_gradient = self.check_loss_gradient

        # Define objective for this specific data
        def obj(p):
            beta = p[: X_train.shape[1]]
            alpha = p[X_train.shape[1] :]
            alpha_expanded = alpha[entity_train]
            residuals = y_train - X_train @ beta - alpha_expanded
            check_loss = np.sum(self.check_loss(residuals, tau))
            penalty = lambda_val * np.sum(np.abs(alpha))
            return check_loss + penalty

        def grad(p):
            beta = p[: X_train.shape[1]]
            alpha = p[X_train.shape[1] :]
            alpha_expanded = alpha[entity_train]
            residuals = y_train - X_train @ beta - alpha_expanded
            psi = self.check_loss_gradient(residuals, tau)

            grad_beta = -X_train.T @ psi
            grad_alpha = np.zeros(n_entities_train)
            for i in range(n_entities_train):
                mask = entity_train == i
                grad_alpha[i] = -np.sum(psi[mask])
            grad_alpha += lambda_val * np.sign(alpha)

            return np.concatenate([grad_beta, grad_alpha])

        # Optimize
        result = minimize(
            fun=obj, x0=x0, method="L-BFGS-B", jac=grad, options={"maxiter": 1000, "ftol": 1e-8}
        )

        # Extract parameters
        beta = result.x[: X_train.shape[1]]
        alpha = result.x[X_train.shape[1] :]

        return {"beta": beta, "alpha": alpha, "converged": result.success, "objective": result.fun}

    def fit(
        self, method: str = "L-BFGS-B", cv_folds: int = 5, verbose: bool = False, **kwargs
    ) -> "FixedEffectsQuantilePanelResult":
        """
        Fit Fixed Effects Quantile Regression.

        Parameters
        ----------
        method : str
            Optimization method
        cv_folds : int
            Number of CV folds for lambda selection
        verbose : bool
            Print progress
        **kwargs
            Additional optimizer arguments

        Returns
        -------
        FixedEffectsQuantilePanelResult
        """
        results = {}

        for tau in self.tau:
            if verbose:
                print(f"\nEstimating Fixed Effects QR for τ = {tau}")

            # Select or use provided lambda
            if self.lambda_fe == "auto":
                lambda_optimal = self._select_lambda_cv(tau, cv_folds=cv_folds, verbose=verbose)
            else:
                lambda_optimal = self.lambda_fe

            # Fit with optimal lambda
            fit_result = self._fit_with_lambda(tau, lambda_optimal)

            # Compute standard errors
            cov_matrix = self._compute_covariance_fe(
                fit_result["beta"], fit_result["alpha"], tau, lambda_optimal
            )

            results[tau] = FixedEffectsQuantileResult(
                params=fit_result["beta"],
                fixed_effects=fit_result["alpha"],
                cov_matrix=cov_matrix,
                tau=tau,
                lambda_fe=lambda_optimal,
                converged=fit_result["converged"],
                model=self,
            )

        return FixedEffectsQuantilePanelResult(self, results)

    def _compute_covariance_fe(
        self, beta: np.ndarray, alpha: np.ndarray, tau: float, lambda_val: float
    ) -> np.ndarray:
        """
        Compute covariance matrix for Fixed Effects QR.

        Accounts for the penalty and fixed effects.
        """
        # Expand alpha
        alpha_expanded = alpha[self.obs_to_entity]

        # Residuals
        residuals = self.y - self.X @ beta - alpha_expanded

        # Sparsity estimation using kernel density
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(residuals)
        f_hat = kde(0)[0]  # Density at zero

        # Hessian of unpenalized objective
        H_beta = (self.X.T @ self.X) / (self.nobs * f_hat)

        # Adjustment for fixed effects and penalty
        # This is a simplified version - full theory is complex
        adjustment_factor = 1 + lambda_val / (self.nobs * f_hat)

        # Inverse Hessian
        try:
            H_inv = np.linalg.inv(H_beta) * adjustment_factor
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H_beta) * adjustment_factor

        # Score outer product (clustered by entity)
        psi = (tau - (residuals < 0).astype(float))[:, np.newaxis] * self.X

        # Aggregate by entity for clustering
        cluster_scores = []
        for entity_id in np.unique(self.entity_ids):
            mask = self.entity_ids == entity_id
            cluster_scores.append(np.sum(psi[mask], axis=0))
        cluster_scores = np.array(cluster_scores)

        B = cluster_scores.T @ cluster_scores / self.nobs

        # Sandwich
        V = H_inv @ B @ H_inv / self.nobs

        return V

    def plot_shrinkage_path(
        self,
        tau: float = 0.5,
        lambda_grid: Optional[np.ndarray] = None,
        var_names: Optional[List[str]] = None,
    ):
        """
        Plot coefficient paths as function of penalty parameter.

        Shows how coefficients shrink as λ increases.
        """
        import matplotlib.pyplot as plt

        if lambda_grid is None:
            lambda_max = self._compute_lambda_max(tau)
            lambda_grid = np.logspace(np.log10(lambda_max * 0.001), np.log10(lambda_max), 50)

        # Estimate for each lambda
        coef_paths = []
        fe_paths = []

        for lambda_val in lambda_grid:
            result = self._fit_with_lambda(tau, lambda_val)
            coef_paths.append(result["beta"])
            fe_paths.append(result["alpha"])

        coef_paths = np.array(coef_paths)
        fe_paths = np.array(fe_paths)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Coefficient paths
        if var_names is None:
            var_names = [f"β{i+1}" for i in range(self.k_exog)]

        for i, var in enumerate(var_names):
            ax1.plot(np.log10(lambda_grid), coef_paths[:, i], label=var)

        ax1.set_xlabel("log₁₀(λ)")
        ax1.set_ylabel("Coefficient Value")
        ax1.set_title(f"Coefficient Shrinkage Paths (τ={tau})")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fixed effects paths (show subset)
        n_show = min(10, self.n_entities)
        for i in range(n_show):
            ax2.plot(np.log10(lambda_grid), fe_paths[:, i], alpha=0.7)

        ax2.set_xlabel("log₁₀(λ)")
        ax2.set_ylabel("Fixed Effect Value")
        ax2.set_title(f"Fixed Effects Shrinkage (first {n_show} entities)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color="red", linestyle="--", alpha=0.5)

        plt.tight_layout()
        return fig


class FixedEffectsQuantileResult:
    """Results for Fixed Effects Quantile Regression."""

    def __init__(
        self,
        params: np.ndarray,
        fixed_effects: np.ndarray,
        cov_matrix: np.ndarray,
        tau: float,
        lambda_fe: float,
        converged: bool,
        model: FixedEffectsQuantile,
    ):
        self.params = params
        self.fixed_effects = fixed_effects
        self.cov_matrix = cov_matrix
        self.tau = tau
        self.lambda_fe = lambda_fe
        self.converged = converged
        self.model = model

    @property
    def bse(self) -> np.ndarray:
        """Standard errors of coefficients."""
        return np.sqrt(np.diag(self.cov_matrix))

    def summary(self):
        """Generate summary of results."""
        print(f"\nFixed Effects Quantile Regression (τ={self.tau})")
        print("=" * 60)
        print(f"Penalty parameter λ: {self.lambda_fe:.6f}")
        print(f"Number of entities: {len(self.fixed_effects)}")
        print(f"Converged: {self.converged}")

        print("\nCoefficients:")
        print("-" * 40)
        for i in range(len(self.params)):
            print(f"  β{i+1}: {self.params[i]:8.4f} ({self.bse[i]:.4f})")

        print("\nFixed Effects Distribution:")
        print("-" * 40)
        print(f"  Mean:     {np.mean(self.fixed_effects):8.4f}")
        print(f"  Std Dev:  {np.std(self.fixed_effects):8.4f}")
        print(f"  Min:      {np.min(self.fixed_effects):8.4f}")
        print(f"  Max:      {np.max(self.fixed_effects):8.4f}")
        print(f"  # Zero:   {np.sum(np.abs(self.fixed_effects) < 1e-6)}")

    def plot_fixed_effects(self):
        """Plot distribution of estimated fixed effects."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram
        ax1.hist(self.fixed_effects, bins=30, edgecolor="black", alpha=0.7)
        ax1.axvline(0, color="red", linestyle="--")
        ax1.set_xlabel("Fixed Effect Value")
        ax1.set_ylabel("Frequency")
        ax1.set_title(f"Distribution of Fixed Effects (λ={self.lambda_fe:.4f})")

        # Sorted plot
        sorted_fe = np.sort(self.fixed_effects)
        ax2.plot(range(len(sorted_fe)), sorted_fe)
        ax2.axhline(0, color="red", linestyle="--")
        ax2.set_xlabel("Entity Rank")
        ax2.set_ylabel("Fixed Effect Value")
        ax2.set_title("Ranked Fixed Effects")

        plt.tight_layout()
        return fig


class FixedEffectsQuantilePanelResult:
    """Container for Fixed Effects QR results across multiple quantiles."""

    def __init__(
        self, model: FixedEffectsQuantile, results: Dict[float, FixedEffectsQuantileResult]
    ):
        self.model = model
        self.results = results

    def summary(self, tau: Optional[float] = None):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("FIXED EFFECTS QUANTILE REGRESSION RESULTS")
        print("=" * 60)

        if tau is None:
            tau_list = sorted(self.results.keys())
        else:
            tau_list = [tau] if np.isscalar(tau) else tau

        for tau in tau_list:
            self.results[tau].summary()

    def plot_coefficients(self, var_idx: Optional[int] = None):
        """Plot coefficients across quantiles."""
        import matplotlib.pyplot as plt

        tau_list = sorted(self.results.keys())

        if var_idx is None:
            # Plot all coefficients
            n_coef = len(self.results[tau_list[0]].params)

            fig, axes = plt.subplots((n_coef + 1) // 2, 2, figsize=(12, 4 * ((n_coef + 1) // 2)))
            if n_coef == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            for i in range(n_coef):
                ax = axes[i]
                coefs = [self.results[tau].params[i] for tau in tau_list]
                se = [self.results[tau].bse[i] for tau in tau_list]

                ax.plot(tau_list, coefs, "o-", label=f"β{i+1}")
                ax.fill_between(
                    tau_list,
                    np.array(coefs) - 1.96 * np.array(se),
                    np.array(coefs) + 1.96 * np.array(se),
                    alpha=0.3,
                )
                ax.set_xlabel("Quantile (τ)")
                ax.set_ylabel(f"β{i+1}")
                ax.set_title(f"Coefficient {i+1} across Quantiles")
                ax.grid(True, alpha=0.3)
                ax.legend()

            # Hide unused subplots
            for j in range(n_coef, len(axes)):
                axes[j].set_visible(False)

            plt.suptitle("Fixed Effects QR Coefficients", fontsize=14)
        else:
            # Plot single coefficient
            fig, ax = plt.subplots(figsize=(8, 6))

            coefs = [self.results[tau].params[var_idx] for tau in tau_list]
            se = [self.results[tau].bse[var_idx] for tau in tau_list]

            ax.plot(tau_list, coefs, "o-", linewidth=2, markersize=8)
            ax.fill_between(
                tau_list,
                np.array(coefs) - 1.96 * np.array(se),
                np.array(coefs) + 1.96 * np.array(se),
                alpha=0.3,
            )
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel(f"β{var_idx+1}")
            ax.set_title(f"Coefficient {var_idx+1} across Quantiles (Fixed Effects QR)")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
