"""
Quantile Monotonicity and Non-Crossing Constraints.

Tools for ensuring monotonic (non-crossing) quantile curves across different
quantile levels. Implements various methods including detection, rearrangement,
isotonic regression, and constrained optimization.

References:
    Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010).
    Quantile and probability curves without crossing. Econometrica, 78(3), 1093-1125.

    Bondell, H. D., Reich, B. J., & Wang, H. (2010).
    Noncrossing quantile regression curve estimation. Biometrika, 97(4), 825-838.
"""

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


class QuantileMonotonicity:
    """
    Tools for ensuring monotonic (non-crossing) quantile curves.

    Implements various methods:
    - Detection of crossing
    - Rearrangement (Chernozhukov et al.)
    - Isotonic regression
    - Constrained optimization
    """

    @staticmethod
    def detect_crossing(
        results: Dict, X_test: Optional[np.ndarray] = None, n_test: int = 100
    ) -> "CrossingReport":
        """
        Detect if quantile curves cross.

        Parameters
        ----------
        results : dict
            {tau: QuantileResult} from QR estimation
        X_test : array, optional
            Test points. If None, use sample
        n_test : int
            Number of test points if X_test not provided

        Returns
        -------
        CrossingReport
            Details about crossing locations and severity
        """
        tau_list = sorted(results.keys())

        # Get test points
        if X_test is None:
            # Use a sample of observations
            model = list(results.values())[0].model
            n_sample = min(n_test, model.nobs)
            idx = np.random.choice(model.nobs, n_sample, replace=False)
            X_test = model.X[idx]

        # Check for crossings
        crossings = []
        total_inversions = 0

        for i in range(len(tau_list) - 1):
            tau1, tau2 = tau_list[i], tau_list[i + 1]

            # Predictions
            pred1 = X_test @ results[tau1].params
            pred2 = X_test @ results[tau2].params

            # Check monotonicity
            inversions = pred1 > pred2 + 1e-10  # Small tolerance for numerical errors
            n_inversions = np.sum(inversions)

            if n_inversions > 0:
                max_violation = (
                    np.max(pred1[inversions] - pred2[inversions]) if n_inversions > 0 else 0
                )

                crossings.append(
                    {
                        "tau_pair": (tau1, tau2),
                        "n_inversions": n_inversions,
                        "pct_inversions": 100 * n_inversions / len(X_test),
                        "max_violation": max_violation,
                        "mean_violation": (
                            np.mean(pred1[inversions] - pred2[inversions])
                            if n_inversions > 0
                            else 0
                        ),
                    }
                )

                total_inversions += n_inversions

        return CrossingReport(
            has_crossing=len(crossings) > 0,
            crossings=crossings,
            total_inversions=total_inversions,
            pct_affected=100 * total_inversions / (len(X_test) * (len(tau_list) - 1)),
        )

    @staticmethod
    def rearrangement(results: Dict, X: Optional[np.ndarray] = None) -> Dict:
        """
        Rearrangement method (Chernozhukov, Fernández-Val, Galichon 2010).

        Sorts predictions to ensure monotonicity.

        Parameters
        ----------
        results : dict
            {tau: QuantileResult}
        X : array, optional
            Covariates for rearrangement

        Returns
        -------
        dict
            Rearranged results
        """
        tau_list = sorted(results.keys())

        if X is None:
            X = list(results.values())[0].model.X

        # Get predictions for all quantiles
        predictions = np.column_stack([X @ results[tau].params for tau in tau_list])

        # Rearrange each observation
        rearranged = np.zeros_like(predictions)
        for i in range(len(X)):
            # Sort predictions for this observation
            rearranged[i] = np.sort(predictions[i])

        # Create new results with rearranged coefficients
        # This is approximate - assumes linear model
        rearranged_results = {}

        for j, tau in enumerate(tau_list):
            # Solve for coefficients that give rearranged predictions
            # min ||Xβ - y_rearranged||²
            beta_rearranged = np.linalg.lstsq(X, rearranged[:, j], rcond=None)[0]

            # Copy result and update params
            new_result = copy.deepcopy(results[tau])
            new_result.params = beta_rearranged
            new_result.rearranged = True

            rearranged_results[tau] = new_result

        return rearranged_results

    @staticmethod
    def isotonic_regression(coef_matrix: np.ndarray, tau_list: np.ndarray) -> np.ndarray:
        """
        Apply isotonic regression to coefficient paths.

        Ensures each coefficient is monotonic in τ.

        Parameters
        ----------
        coef_matrix : array (n_tau, n_coef)
            Coefficient matrix
        tau_list : array
            Quantile levels

        Returns
        -------
        array
            Monotonized coefficient matrix
        """
        n_tau, n_coef = coef_matrix.shape
        monotonized = np.zeros_like(coef_matrix)

        for j in range(n_coef):
            # Fit isotonic regression for this coefficient
            ir = IsotonicRegression(increasing="auto")
            monotonized[:, j] = ir.fit_transform(tau_list, coef_matrix[:, j])

        return monotonized

    @staticmethod
    def constrained_qr(
        X: np.ndarray,
        y: np.ndarray,
        tau_list: np.ndarray,
        method: str = "trust-constr",
        max_iter: int = 1000,
        verbose: bool = False,
    ) -> Dict[float, np.ndarray]:
        """
        Estimate QR with non-crossing constraints.

        Solves the constrained optimization problem:
        min Σ_τ Σ_i ρ_τ(y_i - X_i'β(τ))
        s.t. X_i'β(τ₁) ≤ X_i'β(τ₂) for all i and τ₁ < τ₂

        Parameters
        ----------
        X : array (n, p)
            Design matrix
        y : array (n,)
            Response
        tau_list : array
            Quantiles to estimate
        method : str
            Optimization method
        max_iter : int
            Maximum iterations
        verbose : bool
            Print progress

        Returns
        -------
        dict
            {tau: beta} with non-crossing guaranteed
        """
        n, p = X.shape
        n_tau = len(tau_list)

        # Stack all parameters
        n_params = n_tau * p

        def check_loss(u: np.ndarray, tau: float) -> np.ndarray:
            """Check loss function for quantile regression."""
            return u * (tau - (u < 0).astype(float))

        def objective(params_stacked: np.ndarray) -> float:
            """Combined objective for all quantiles."""
            params_matrix = params_stacked.reshape(n_tau, p)
            total_loss = 0

            for i, tau in enumerate(tau_list):
                beta = params_matrix[i]
                residuals = y - X @ beta
                total_loss += np.sum(check_loss(residuals, tau))

            return total_loss

        def constraint_noncrossing(params_stacked: np.ndarray) -> np.ndarray:
            """
            Non-crossing constraints.

            Returns vector that should be >= 0.
            """
            params_matrix = params_stacked.reshape(n_tau, p)
            constraints = []

            for i in range(n_tau - 1):
                # X @ β(τᵢ) <= X @ β(τᵢ₊₁)
                diff = X @ (params_matrix[i + 1] - params_matrix[i])
                constraints.extend(diff)

            return np.array(constraints)

        # Initial values (unconstrained estimates)
        if verbose:
            print("Computing initial unconstrained estimates...")

        init_params = []
        for tau in tau_list:
            # Quick quantile regression using weighted least squares approximation
            q_tau = np.quantile(y, tau)
            beta_init = np.zeros(p)
            beta_init[0] = q_tau  # Intercept as quantile

            # Simple gradient descent for better initialization
            for _ in range(20):
                residuals = y - X @ beta_init
                gradient = -X.T @ (tau - (residuals < 0).astype(float)) / n
                beta_init -= 0.1 * gradient

            init_params.extend(beta_init)

        init_params = np.array(init_params)

        # Constraints
        constraints = {"type": "ineq", "fun": constraint_noncrossing}

        # Optimize
        if verbose:
            print("Optimizing with non-crossing constraints...")

        result = minimize(
            fun=objective,
            x0=init_params,
            method=method,
            constraints=constraints,
            options={"maxiter": max_iter, "disp": verbose},
        )

        if not result.success:
            warnings.warn(f"Constrained optimization did not converge: {result.message}")

        # Extract results
        params_matrix = result.x.reshape(n_tau, p)
        results = {tau: params_matrix[i] for i, tau in enumerate(tau_list)}

        return results

    @staticmethod
    def simultaneous_qr(
        X: np.ndarray,
        y: np.ndarray,
        tau_list: np.ndarray,
        lambda_nc: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False,
    ) -> Dict[float, np.ndarray]:
        """
        Simultaneous quantile regression with soft non-crossing penalty.

        Minimizes:
        Σ_τ Σ_i ρ_τ(y_i - X_i'β(τ)) + λ Σ_τ Σ_i max(0, X_i'β(τ) - X_i'β(τ+1))²

        Parameters
        ----------
        X : array (n, p)
            Design matrix
        y : array (n,)
            Response
        tau_list : array
            Quantiles to estimate
        lambda_nc : float
            Non-crossing penalty parameter
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        verbose : bool
            Print progress

        Returns
        -------
        dict
            {tau: beta} estimated simultaneously
        """
        n, p = X.shape
        n_tau = len(tau_list)

        # Initialize with independent QR estimates
        beta_matrix = np.zeros((n_tau, p))
        for i, tau in enumerate(tau_list):
            # Simple initialization
            q_tau = np.quantile(y, tau)
            beta_matrix[i, 0] = q_tau

        # Iterative optimization
        for iteration in range(max_iter):
            beta_old = beta_matrix.copy()

            # Update each tau's coefficients
            for i, tau in enumerate(tau_list):
                # Compute gradient
                residuals = y - X @ beta_matrix[i]
                grad_loss = -X.T @ (tau - (residuals < 0).astype(float)) / n

                # Add non-crossing penalty gradient
                grad_penalty = np.zeros(p)
                if i > 0:
                    # Penalty from previous quantile
                    diff = X @ (beta_matrix[i - 1] - beta_matrix[i])
                    violations = np.maximum(0, diff)
                    grad_penalty -= 2 * lambda_nc * X.T @ violations / n

                if i < n_tau - 1:
                    # Penalty from next quantile
                    diff = X @ (beta_matrix[i] - beta_matrix[i + 1])
                    violations = np.maximum(0, diff)
                    grad_penalty += 2 * lambda_nc * X.T @ violations / n

                # Update with step size
                step_size = 0.1 / (1 + iteration * 0.01)
                beta_matrix[i] -= step_size * (grad_loss + grad_penalty)

            # Check convergence
            if np.max(np.abs(beta_matrix - beta_old)) < tol:
                if verbose:
                    print(f"Converged after {iteration+1} iterations")
                break

        results = {tau: beta_matrix[i] for i, tau in enumerate(tau_list)}

        return results

    @staticmethod
    def project_to_monotone(predictions: np.ndarray, method: str = "averaging") -> np.ndarray:
        """
        Project predictions to monotone space.

        Parameters
        ----------
        predictions : array (n_obs, n_tau)
            Predictions for each observation and quantile
        method : str
            'averaging': Average adjacent crossing quantiles
            'isotonic': Use isotonic regression

        Returns
        -------
        array
            Monotone predictions
        """
        n_obs, n_tau = predictions.shape
        projected = predictions.copy()

        if method == "averaging":
            # Simple averaging of crossing quantiles
            for i in range(n_obs):
                for j in range(n_tau - 1):
                    if projected[i, j] > projected[i, j + 1]:
                        # Average the crossing values
                        avg = (projected[i, j] + projected[i, j + 1]) / 2
                        projected[i, j] = avg
                        projected[i, j + 1] = avg

        elif method == "isotonic":
            # Isotonic regression for each observation
            for i in range(n_obs):
                ir = IsotonicRegression(increasing=True)
                projected[i] = ir.fit_transform(range(n_tau), predictions[i])

        return projected


class CrossingReport:
    """Report on quantile crossing detection."""

    def __init__(
        self, has_crossing: bool, crossings: List[Dict], total_inversions: int, pct_affected: float
    ):
        self.has_crossing = has_crossing
        self.crossings = crossings
        self.total_inversions = total_inversions
        self.pct_affected = pct_affected

    def summary(self) -> "CrossingReport":
        """Print crossing summary."""
        print("\nQuantile Crossing Detection Report")
        print("=" * 50)

        if not self.has_crossing:
            print("✓ No crossing detected - quantile curves are monotonic")
        else:
            print("✗ CROSSING DETECTED")
            print(f"\nTotal inversions: {self.total_inversions}")
            print(f"Percentage affected: {self.pct_affected:.2f}%")

            print("\nDetails by quantile pair:")
            print("-" * 40)
            for crossing in self.crossings:
                tau1, tau2 = crossing["tau_pair"]
                print(f"τ={tau1:.2f} vs τ={tau2:.2f}:")
                print(
                    f"  Inversions: {crossing['n_inversions']} "
                    f"({crossing['pct_inversions']:.1f}%)"
                )
                print(f"  Max violation: {crossing['max_violation']:.4f}")
                print(f"  Mean violation: {crossing['mean_violation']:.4f}")

        return self

    def plot_violations(self, X: np.ndarray, results: Dict):
        """Visualize crossing violations."""
        import matplotlib.pyplot as plt

        tau_list = sorted(results.keys())

        # Compute predictions
        predictions = np.column_stack([X @ results[tau].params for tau in tau_list])

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Sample of curves
        n_show = min(20, len(X))
        idx_show = np.random.choice(len(X), n_show, replace=False)

        for i in idx_show:
            ax1.plot(tau_list, predictions[i], alpha=0.5)

        ax1.set_xlabel("Quantile (τ)")
        ax1.set_ylabel("Predicted Value")
        ax1.set_title("Sample of Quantile Curves")
        ax1.grid(True, alpha=0.3)

        # Violation heatmap
        violations = np.zeros((len(tau_list) - 1, len(X)))
        for i in range(len(tau_list) - 1):
            violations[i] = np.maximum(0, predictions[:, i] - predictions[:, i + 1])

        im = ax2.imshow(violations, aspect="auto", cmap="Reds")
        ax2.set_xlabel("Observation")
        ax2.set_ylabel("Quantile Pair Index")
        ax2.set_title("Crossing Violations (red = crossing)")
        plt.colorbar(im, ax=ax2, label="Violation Magnitude")

        plt.tight_layout()
        return fig

    def to_dataframe(self) -> pd.DataFrame:
        """Convert crossing report to DataFrame."""
        if not self.crossings:
            return pd.DataFrame()

        df = pd.DataFrame(self.crossings)
        df[["tau1", "tau2"]] = pd.DataFrame(df["tau_pair"].tolist(), index=df.index)
        df = df.drop("tau_pair", axis=1)
        df = df[
            ["tau1", "tau2", "n_inversions", "pct_inversions", "max_violation", "mean_violation"]
        ]

        return df


class MonotonicityComparison:
    """Compare different monotonicity enforcement methods."""

    def __init__(self, X: np.ndarray, y: np.ndarray, tau_list: np.ndarray):
        self.X = X
        self.y = y
        self.tau_list = tau_list
        self.results = {}

    def compare_methods(
        self,
        methods: List[str] = ["unconstrained", "rearrangement", "isotonic", "constrained"],
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Compare different monotonicity methods.

        Parameters
        ----------
        methods : list
            Methods to compare
        verbose : bool
            Print progress

        Returns
        -------
        DataFrame
            Comparison results
        """
        from ..pooled import PooledQuantile

        comparison = []

        for method in methods:
            if verbose:
                print(f"\nEstimating with method: {method}")

            if method == "unconstrained":
                # Standard QR without constraints
                results = {}
                for tau in self.tau_list:
                    model = PooledQuantile(data=None, endog=self.y, exog=self.X, tau=tau)
                    result = model.fit()
                    results[tau] = result

            elif method == "rearrangement":
                # First estimate unconstrained, then rearrange
                unconstrained = {}
                for tau in self.tau_list:
                    model = PooledQuantile(data=None, endog=self.y, exog=self.X, tau=tau)
                    result = model.fit()
                    unconstrained[tau] = result

                results = QuantileMonotonicity.rearrangement(unconstrained, self.X)

            elif method == "isotonic":
                # Isotonic regression on coefficients
                coef_matrix = []
                for tau in self.tau_list:
                    model = PooledQuantile(data=None, endog=self.y, exog=self.X, tau=tau)
                    result = model.fit()
                    coef_matrix.append(result.params)

                coef_matrix = np.array(coef_matrix)
                monotone_coefs = QuantileMonotonicity.isotonic_regression(
                    coef_matrix, self.tau_list
                )

                # Create results dict
                results = {}
                for i, tau in enumerate(self.tau_list):
                    from copy import deepcopy

                    results[tau] = type(
                        "Result",
                        (),
                        {"params": monotone_coefs[i], "model": type("Model", (), {"X": self.X})()},
                    )()

            elif method == "constrained":
                # Constrained optimization
                beta_dict = QuantileMonotonicity.constrained_qr(
                    self.X, self.y, self.tau_list, verbose=verbose
                )

                # Create results dict
                results = {}
                for tau in self.tau_list:
                    results[tau] = type(
                        "Result",
                        (),
                        {"params": beta_dict[tau], "model": type("Model", (), {"X": self.X})()},
                    )()

            # Check crossing
            crossing_report = QuantileMonotonicity.detect_crossing(results, self.X)

            # Compute fit
            total_loss = 0
            for tau in self.tau_list:
                predictions = self.X @ results[tau].params
                residuals = self.y - predictions
                check_loss = residuals * (tau - (residuals < 0).astype(float))
                total_loss += np.sum(check_loss)

            comparison.append(
                {
                    "method": method,
                    "has_crossing": crossing_report.has_crossing,
                    "total_inversions": crossing_report.total_inversions,
                    "pct_affected": crossing_report.pct_affected,
                    "total_loss": total_loss,
                    "avg_loss": total_loss / (len(self.y) * len(self.tau_list)),
                }
            )

            self.results[method] = results

        return pd.DataFrame(comparison)

    def plot_comparison(self, var_idx: int = 0):
        """
        Plot coefficient paths for different methods.

        Parameters
        ----------
        var_idx : int
            Index of variable to plot
        """
        import matplotlib.pyplot as plt

        n_methods = len(self.results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5))

        if n_methods == 1:
            axes = [axes]

        for ax, (method, results) in zip(axes, self.results.items()):
            coefs = [results[tau].params[var_idx] for tau in self.tau_list]
            ax.plot(self.tau_list, coefs, "o-", linewidth=2)
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel(f"Coefficient {var_idx+1}")
            ax.set_title(f"{method.capitalize()} Method")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
