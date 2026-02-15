"""
Comparison tools for Fixed Effects Quantile Regression estimators.

This module provides utilities for comparing different FE QR approaches,
including performance benchmarks, diagnostic tests, and visualization tools.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from panelbox.utils.data import PanelData

from .canay import CanayTwoStep
from .fixed_effects import FixedEffectsQuantile
from .pooled import PooledQuantile


class FEQuantileComparison:
    """
    Tools for comparing Fixed Effects Quantile Regression estimators.
    """

    def __init__(
        self, data: PanelData, formula: Optional[str] = None, tau: Union[float, List[float]] = 0.5
    ):
        self.data = data
        self.formula = formula
        self.tau = tau if isinstance(tau, list) else [tau]
        self.results = {}

    def compare_all(
        self, methods: List[str] = ["canay", "penalty", "pooled"], verbose: bool = True, **kwargs
    ) -> "ComparisonResults":
        """
        Compare multiple FE QR estimators.

        Parameters
        ----------
        methods : list
            Methods to compare:
            - 'canay': Canay two-step
            - 'penalty': Koenker penalty method
            - 'pooled': Pooled QR (no FE)
        verbose : bool
            Print progress
        **kwargs
            Additional arguments for estimators

        Returns
        -------
        ComparisonResults
        """
        if verbose:
            print("\nCOMPARING FIXED EFFECTS QR ESTIMATORS")
            print("=" * 60)

        timing = {}
        estimates = {}
        diagnostics = {}

        # Pooled QR (baseline)
        if "pooled" in methods:
            if verbose:
                print("\n1. Pooled Quantile Regression")
                print("-" * 40)

            model = PooledQuantile(self.data, self.formula, self.tau)

            start = time.time()
            result = model.fit()
            timing["pooled"] = time.time() - start

            estimates["pooled"] = result
            diagnostics["pooled"] = self._compute_diagnostics(model, result)

            if verbose:
                print(f"  Time: {timing['pooled']:.2f} seconds")
                print(f"  Pseudo-R²: {diagnostics['pooled']['pseudo_r2']:.4f}")

        # Canay two-step
        if "canay" in methods:
            if verbose:
                print("\n2. Canay Two-Step Estimator")
                print("-" * 40)

            model = CanayTwoStep(self.data, self.formula, self.tau)

            start = time.time()
            result = model.fit(verbose=False)
            timing["canay"] = time.time() - start

            estimates["canay"] = result

            # Test location shift
            if len(self.tau) == 1:
                # Need multiple quantiles for location shift test
                temp_model = CanayTwoStep(self.data, self.formula, [0.25, 0.5, 0.75])
                loc_test = temp_model.test_location_shift()
                location_shift_pval = loc_test.p_value
            else:
                loc_test = model.test_location_shift(tau_grid=self.tau)
                location_shift_pval = loc_test.p_value

            diagnostics["canay"] = {
                "pseudo_r2": self._compute_pseudo_r2(model, result),
                "location_shift_pval": location_shift_pval,
            }

            if verbose:
                print(f"  Time: {timing['canay']:.2f} seconds")
                print(f"  Location shift p-value: {location_shift_pval:.4f}")

        # Penalty method
        if "penalty" in methods:
            if verbose:
                print("\n3. Koenker Penalty Method")
                print("-" * 40)

            model = FixedEffectsQuantile(
                self.data, self.formula, self.tau, lambda_fe=kwargs.get("lambda_fe", "auto")
            )

            start = time.time()
            result = model.fit(verbose=False)
            timing["penalty"] = time.time() - start

            estimates["penalty"] = result

            # Get results for first tau
            first_tau = self.tau[0]
            diagnostics["penalty"] = {
                "pseudo_r2": self._compute_pseudo_r2(model, result),
                "lambda_optimal": result.results[first_tau].lambda_fe,
                "n_zero_fe": np.sum(np.abs(result.results[first_tau].fixed_effects) < 1e-6),
            }

            if verbose:
                print(f"  Time: {timing['penalty']:.2f} seconds")
                print(f"  Optimal λ: {diagnostics['penalty']['lambda_optimal']:.4f}")
                print(f"  Zero FE: {diagnostics['penalty']['n_zero_fe']}")

        # Store results
        self.results = ComparisonResults(
            estimates=estimates, timing=timing, diagnostics=diagnostics, tau=self.tau
        )

        if verbose:
            print("\n" + "=" * 60)
            self.results.print_summary()

        return self.results

    def _compute_diagnostics(self, model, result) -> Dict[str, Any]:
        """Compute diagnostic measures for a result."""
        pseudo_r2 = self._compute_pseudo_r2(model, result)
        return {"pseudo_r2": pseudo_r2}

    def _compute_pseudo_r2(self, model, result) -> float:
        """Compute pseudo-R² for any result type."""
        # Get first tau for comparison
        tau = self.tau[0]

        if hasattr(result, "results"):
            # Multi-tau result
            res = result.results[tau]
            params = res.params
        else:
            params = result.params if hasattr(result, "params") else result.get("params")

        # Get y and X from model
        y = model.y if hasattr(model, "y") else model.data.df.iloc[:, 0].values
        X = model.X if hasattr(model, "X") else model.data.df.iloc[:, 1:].values

        # Add constant if needed
        if X.shape[1] < len(params):
            X = np.column_stack([np.ones(len(y)), X])

        # Check if we have transformed y for Canay
        if hasattr(model, "y_transformed_"):
            y = model.y_transformed_

        fitted = X @ params
        residuals = y - fitted

        # Compute check loss
        rho_full = np.sum(model.check_loss(residuals, tau))
        q_tau = np.quantile(y, tau)
        rho_null = np.sum(model.check_loss(y - q_tau, tau))

        return 1 - rho_full / rho_null if rho_null > 0 else 0

    def bootstrap_comparison(
        self, n_boot: int = 100, methods: List[str] = ["canay", "penalty"]
    ) -> Dict[str, Dict]:
        """
        Bootstrap comparison of methods to assess stability.
        """
        boot_results = {method: [] for method in methods}

        for b in range(n_boot):
            # Resample entities with replacement
            entities = np.unique(self.data.entity_ids)
            boot_entities = np.random.choice(entities, len(entities), replace=True)

            # Create bootstrap sample - need to handle duplicates properly
            boot_indices = []
            for entity in boot_entities:
                entity_mask = self.data.entity_ids == entity
                boot_indices.extend(np.where(entity_mask)[0])

            # Create bootstrap data
            boot_df = self.data.df.iloc[boot_indices].reset_index(drop=True)
            boot_entity_ids = self.data.entity_ids.iloc[boot_indices].reset_index(drop=True)
            boot_time_ids = self.data.time_ids.iloc[boot_indices].reset_index(drop=True)

            boot_data = PanelData(
                df=boot_df,
                entity_col=boot_entity_ids.name if hasattr(boot_entity_ids, "name") else "entity",
                time_col=boot_time_ids.name if hasattr(boot_time_ids, "name") else "time",
            )
            boot_data.entity_ids = boot_entity_ids
            boot_data.time_ids = boot_time_ids

            # Estimate with each method
            for method in methods:
                try:
                    if method == "canay":
                        model = CanayTwoStep(boot_data, self.formula, self.tau[0])
                    elif method == "penalty":
                        model = FixedEffectsQuantile(boot_data, self.formula, self.tau[0])
                    else:
                        continue

                    result = model.fit(verbose=False)

                    # Extract params
                    if hasattr(result, "results"):
                        params = result.results[self.tau[0]].params
                    else:
                        params = result.params

                    boot_results[method].append(params)
                except Exception as e:
                    # Skip if estimation fails
                    warnings.warn(f"Bootstrap iteration {b} failed for {method}: {e}")

        # Compute statistics
        stats = {}
        for method in methods:
            if boot_results[method]:
                params = np.array(boot_results[method])
                stats[method] = {
                    "mean": np.mean(params, axis=0),
                    "std": np.std(params, axis=0),
                    "coverage": len(boot_results[method]) / n_boot,
                    "ci_lower": np.percentile(params, 2.5, axis=0),
                    "ci_upper": np.percentile(params, 97.5, axis=0),
                }

        return stats


class ComparisonResults:
    """Container for comparison results."""

    def __init__(self, estimates: Dict, timing: Dict, diagnostics: Dict, tau: List[float]):
        self.estimates = estimates
        self.timing = timing
        self.diagnostics = diagnostics
        self.tau = tau

    def print_summary(self):
        """Print comparison summary."""
        methods = list(self.estimates.keys())

        # Use first tau for comparison
        tau_compare = self.tau[0]

        # Coefficient comparison
        print("\nCOEFFICIENT ESTIMATES")
        print("-" * 60)

        # Get coefficient names and values
        coef_dict = {}
        for method in methods:
            result = self.estimates[method]
            if hasattr(result, "results"):
                params = result.results[tau_compare].params
            else:
                params = result.params if hasattr(result, "params") else result.get("params")
            coef_dict[method] = params

        # Determine number of coefficients
        n_coef = len(coef_dict[methods[0]])

        # Create table
        header = "Coef    " + "  ".join(f"{m:>10}" for m in methods)
        print(header)
        print("-" * len(header))

        for i in range(n_coef):
            row = f"β{i+1:<6}"
            for method in methods:
                coef = coef_dict[method][i]
                row += f"  {coef:10.4f}"
            print(row)

        # Timing comparison
        print("\nCOMPUTATIONAL TIME (seconds)")
        print("-" * 40)
        for method in methods:
            print(f"{method:<15} {self.timing[method]:10.2f}")

        # Find fastest
        fastest = min(self.timing, key=self.timing.get)
        print(f"\nFastest: {fastest}")

        # Diagnostics
        print("\nDIAGNOSTICS")
        print("-" * 40)
        for method in methods:
            print(f"\n{method}:")
            for key, value in self.diagnostics[method].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

    def plot_comparison(self):
        """Visualize comparison results."""
        methods = list(self.estimates.keys())
        tau_compare = self.tau[0]

        # Extract coefficients
        coef_dict = {}
        se_dict = {}
        for method in methods:
            result = self.estimates[method]
            if hasattr(result, "results"):
                params = result.results[tau_compare].params
                if hasattr(result.results[tau_compare], "bse"):
                    se = result.results[tau_compare].bse
                else:
                    se = np.sqrt(np.diag(result.results[tau_compare].cov_matrix))
            else:
                params = result.params if hasattr(result, "params") else result.get("params")
                if hasattr(result, "bse"):
                    se = result.bse
                elif hasattr(result, "cov_matrix"):
                    se = np.sqrt(np.diag(result.cov_matrix))
                else:
                    se = np.zeros_like(params)
            coef_dict[method] = params
            se_dict[method] = se

        n_coef = len(coef_dict[methods[0]])

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Coefficient comparison
        ax = axes[0, 0]
        x = np.arange(n_coef)
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            coefs = coef_dict[method]
            ax.bar(x + i * width, coefs, width, label=method)

        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Value")
        ax.set_title("Coefficient Estimates Comparison")
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([f"β{i+1}" for i in range(n_coef)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Timing comparison
        ax = axes[0, 1]
        ax.bar(methods, [self.timing[m] for m in methods])
        ax.set_ylabel("Time (seconds)")
        ax.set_title("Computational Time")
        ax.grid(True, alpha=0.3)

        # 3. Standard errors comparison
        ax = axes[1, 0]
        for i, method in enumerate(methods):
            se = se_dict[method]
            ax.plot(range(len(se)), se, "o-", label=method)

        ax.set_xlabel("Coefficient")
        ax.set_ylabel("Standard Error")
        ax.set_title("Standard Errors Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Diagnostics
        ax = axes[1, 1]
        ax.axis("off")

        # Text summary
        text = "Diagnostic Summary\n" + "=" * 30 + "\n\n"
        for method in methods:
            text += f"{method}:\n"
            for key, value in self.diagnostics[method].items():
                if isinstance(value, float):
                    text += f"  {key}: {value:.4f}\n"
                else:
                    text += f"  {key}: {value}\n"
            text += "\n"

        ax.text(
            0.1,
            0.9,
            text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )

        plt.suptitle(f"FE QR Methods Comparison (τ={tau_compare})", fontsize=14)
        plt.tight_layout()
        return fig

    def coefficient_correlation_matrix(self):
        """Compute and plot correlation matrix of coefficients across methods."""
        methods = list(self.estimates.keys())
        tau_compare = self.tau[0]

        # Extract coefficients
        coef_matrix = []
        for method in methods:
            result = self.estimates[method]
            if hasattr(result, "results"):
                params = result.results[tau_compare].params
            else:
                params = result.params if hasattr(result, "params") else result.get("params")
            coef_matrix.append(params)

        coef_matrix = np.array(coef_matrix).T  # Variables x Methods

        # Compute correlation matrix
        corr_matrix = np.corrcoef(coef_matrix, rowvar=False)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(len(methods)))
        ax.set_yticks(np.arange(len(methods)))
        ax.set_xticklabels(methods)
        ax.set_yticklabels(methods)

        # Rotate the tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation")

        # Add correlation values
        for i in range(len(methods)):
            for j in range(len(methods)):
                text = ax.text(
                    j, i, f"{corr_matrix[i, j]:.3f}", ha="center", va="center", color="black"
                )

        ax.set_title(f"Coefficient Correlation Matrix (τ={tau_compare})")
        plt.tight_layout()
        return fig, corr_matrix
