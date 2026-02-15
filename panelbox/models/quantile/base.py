"""
Base class for quantile regression models for panel data.

This module provides the abstract base class for quantile regression models,
implementing the check loss function and common functionality for panel quantile
regression estimation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np


class QuantilePanelModel(ABC):
    """
    Abstract base class for quantile regression panel models.

    Parameters
    ----------
    data : PanelData
        Panel data object
    formula : str, optional
        Model formula
    tau : float or array-like
        Quantile(s) to estimate. Default 0.5 (median).
    """

    def __init__(self, data, formula=None, tau=0.5, **kwargs):
        self.data = data
        self.formula = formula
        self.tau = np.atleast_1d(tau)
        self.n_quantiles = len(self.tau)

        # Validate tau values
        if np.any(self.tau <= 0) or np.any(self.tau >= 1):
            raise ValueError("Quantile levels tau must be in (0, 1)")

        # Parse formula if provided
        if formula:
            self._parse_formula()

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _parse_formula(self):
        """Parse formula to extract variables."""
        # This is a simplified version - full implementation would use formula parser
        # For now, assume formula is "y ~ x1 + x2 + ..."
        if "~" in self.formula:
            lhs, rhs = self.formula.split("~")
            self.dependent_var = lhs.strip()
            self.independent_vars = [v.strip() for v in rhs.split("+")]
        else:
            raise ValueError("Invalid formula format")

    @staticmethod
    def check_loss(u, tau):
        """
        Compute check (pinball) loss function.

        ρτ(u) = u(τ - 1{u<0})
        """
        return u * (tau - (u < 0).astype(float))

    @staticmethod
    def check_loss_gradient(u, tau):
        """
        Subgradient of check loss.

        ∂ρτ(u)/∂u = τ - 1{u<0}
        """
        return tau - (u < 0).astype(float)

    @abstractmethod
    def _objective(self, params, tau):
        """Compute objective function for optimization at quantile tau."""
        pass

    def fit(self, method="interior-point", bootstrap=False, n_boot=999, n_jobs=1, **kwargs):
        """
        Fit quantile regression model.

        Parameters
        ----------
        method : str
            Optimization method: 'interior-point', 'simplex', 'smooth'
        bootstrap : bool
            Whether to compute bootstrap standard errors
        n_boot : int
            Number of bootstrap replications
        n_jobs : int
            Number of parallel jobs for bootstrap
        **kwargs : dict
            Additional arguments to optimizer

        Returns
        -------
        QuantilePanelResult
            Results object containing estimates for all quantiles
        """
        from panelbox.optimization.quantile import optimize_quantile

        results = {}

        # Estimate for each quantile
        for tau in self.tau:
            self.current_tau = tau  # Store for use in methods

            # Optimize
            result_tau = optimize_quantile(
                objective=lambda p: self._objective(p, tau),
                gradient=lambda p: self._gradient(p, tau) if hasattr(self, "_gradient") else None,
                n_params=self.k_exog if hasattr(self, "k_exog") else self.X.shape[1],
                tau=tau,
                method=method,
                **kwargs,
            )

            # Bootstrap inference if requested
            if bootstrap:
                result_tau = self._bootstrap_inference(result_tau, tau, n_boot, n_jobs)

            results[tau] = result_tau

        return QuantilePanelResult(self, results)

    def _bootstrap_inference(self, result, tau, n_boot, n_jobs):
        """Compute bootstrap standard errors and confidence intervals."""
        from panelbox.inference.quantile.bootstrap import bootstrap_qr

        boot_result = bootstrap_qr(
            model=self,
            tau=tau,
            n_boot=n_boot,
            method="cluster",  # Default to cluster bootstrap for panels
            n_jobs=n_jobs,
        )

        result.boot_params = boot_result.boot_params
        result.se_boot = boot_result.se
        result.ci_boot_lower = boot_result.ci_lower
        result.ci_boot_upper = boot_result.ci_upper

        return result


class QuantilePanelResult:
    """Results container for quantile panel regression."""

    def __init__(self, model, results):
        self.model = model
        self.results = results

    def summary(self):
        """Print summary of results."""
        print("\n" + "=" * 70)
        print("Quantile Regression Results")
        print("=" * 70)

        for tau in sorted(self.results.keys()):
            result = self.results[tau]
            print(f"\nQuantile τ = {tau:.3f}")
            print("-" * 40)

            if "params" in result:
                print("Parameters:", result["params"])
            if "converged" in result:
                print("Converged:", result["converged"])
            if "iterations" in result:
                print("Iterations:", result["iterations"])

        print("=" * 70)
