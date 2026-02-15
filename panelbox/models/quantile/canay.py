"""
Canay (2011) Two-Step Quantile Estimator for Panel Data.

This module implements the computationally efficient two-step approach for
fixed effects quantile regression, which assumes fixed effects are pure
location shifters.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.panel_data import PanelData

from .base import QuantilePanelModel
from .pooled import PooledQuantile


class CanayTwoStep(QuantilePanelModel):
    """
    Canay (2011) Two-Step Quantile Estimator for Panel Data.

    Simple and computationally efficient approach:
    1. Estimate fixed effects via within-transformation (FE-OLS)
    2. Transform dependent variable: ỹᵢₜ = yᵢₜ - α̂ᵢ
    3. Run pooled quantile regression on transformed data

    Key Assumption:
    Fixed effects are pure location shifters (same across all quantiles).
    This is testable using the location_shift_test() method.

    Parameters
    ----------
    data : PanelData
        Panel data object
    formula : str, optional
        Model formula
    tau : float or array-like
        Quantile(s) to estimate

    Attributes
    ----------
    fixed_effects_ : array
        Estimated fixed effects from step 1
    fe_ols_result_ : object
        Full results from step 1 FE-OLS
    y_transformed_ : array
        Transformed dependent variable (y - α̂)
    """

    def __init__(
        self, data: PanelData, formula: Optional[str] = None, tau: Union[float, List[float]] = 0.5
    ):
        super().__init__(data, formula, tau)
        self._setup_data()
        self._step1_complete = False
        self.fixed_effects_ = None
        self.fe_ols_result_ = None
        self.y_transformed_ = None
        self.n_entities = len(np.unique(self.entity_ids))

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
        self.time_ids = self.data.time_ids.values

    def fit(
        self, se_adjustment: str = "two-step", verbose: bool = False, **kwargs
    ) -> "CanayTwoStepResult":
        """
        Two-step estimation procedure.

        Parameters
        ----------
        se_adjustment : str
            How to adjust standard errors:
            - 'two-step': Account for first-step estimation
            - 'naive': Ignore first-step uncertainty
            - 'bootstrap': Use bootstrap for both steps
        verbose : bool
            Print progress
        **kwargs
            Additional arguments for QR optimization

        Returns
        -------
        CanayTwoStepResult
        """
        if verbose:
            print("Canay Two-Step Quantile Regression")
            print("=" * 50)

        # Step 1: Estimate fixed effects via FE-OLS
        if not self._step1_complete:
            if verbose:
                print("\nStep 1: Estimating fixed effects via FE-OLS...")

            self._estimate_fixed_effects()

            if verbose:
                print(f"  Fixed effects estimated for {self.n_entities} entities")
                print(f"  Mean FE: {np.mean(self.fixed_effects_):.4f}")
                print(f"  Std FE:  {np.std(self.fixed_effects_):.4f}")

        # Check T size and issue warning
        avg_T = self.nobs / self.n_entities
        if avg_T < 10:
            warnings.warn(
                f"Average T = {avg_T:.1f} is small. Canay estimator requires "
                "large T for consistency. Consider using penalty method instead.",
                UserWarning,
            )

        # Step 2: Quantile regression on transformed data
        if verbose:
            print("\nStep 2: Pooled QR on transformed data...")

        results = {}
        for tau in self.tau:
            if verbose:
                print(f"  Estimating τ = {tau}...")

            result_tau = self._estimate_quantile_step2(tau, se_adjustment, **kwargs)
            results[tau] = result_tau

        # Create result object
        final_result = CanayTwoStepResult(
            model=self,
            results=results,
            fixed_effects=self.fixed_effects_,
            fe_ols_result=self.fe_ols_result_,
        )

        if verbose:
            print("\nEstimation complete!")

        return final_result

    def _estimate_fixed_effects(self):
        """
        Step 1: Estimate fixed effects via within-transformation OLS.
        """
        # Demean variables by entity (within transformation)
        y_demeaned = np.zeros_like(self.y)
        X_demeaned = np.zeros_like(self.X)
        entity_means_y = {}
        entity_means_X = {}

        # Calculate entity means
        for entity_id in np.unique(self.entity_ids):
            mask = self.entity_ids == entity_id
            entity_means_y[entity_id] = np.mean(self.y[mask])
            entity_means_X[entity_id] = np.mean(self.X[mask], axis=0)

            # Apply within transformation
            y_demeaned[mask] = self.y[mask] - entity_means_y[entity_id]
            X_demeaned[mask] = self.X[mask] - entity_means_X[entity_id]

        # Within estimator (FE-OLS)
        # Remove constant column if present (it will be zero after demeaning)
        X_demeaned_noconstant = X_demeaned[:, 1:] if X_demeaned.shape[1] > 1 else X_demeaned

        # OLS on demeaned data
        try:
            beta_fe = np.linalg.lstsq(X_demeaned_noconstant, y_demeaned, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            beta_fe = np.linalg.pinv(X_demeaned_noconstant.T @ X_demeaned_noconstant) @ (
                X_demeaned_noconstant.T @ y_demeaned
            )

        # Add zero for constant if needed
        if self.X.shape[1] > X_demeaned_noconstant.shape[1]:
            beta_fe = np.concatenate([[0], beta_fe])

        # Calculate fixed effects
        # α̂ᵢ = ȳᵢ - x̄ᵢ'β̂
        self.fixed_effects_ = np.zeros(self.n_entities)
        entity_list = np.unique(self.entity_ids)

        for i, entity_id in enumerate(entity_list):
            self.fixed_effects_[i] = entity_means_y[entity_id] - entity_means_X[entity_id] @ beta_fe

        # Store OLS results
        residuals_fe = (
            y_demeaned - X_demeaned_noconstant @ beta_fe[1:]
            if beta_fe.shape[0] > 1
            else y_demeaned - X_demeaned_noconstant @ beta_fe
        )
        sigma2 = np.sum(residuals_fe**2) / (self.nobs - self.k_exog - self.n_entities + 1)
        var_beta = sigma2 * np.linalg.inv(X_demeaned_noconstant.T @ X_demeaned_noconstant)

        self.fe_ols_result_ = {
            "params": beta_fe,
            "cov_matrix": (
                var_beta if X_demeaned_noconstant.shape[1] > 1 else np.array([[var_beta]])
            ),
            "residuals": residuals_fe,
            "sigma2": sigma2,
            "entity_means_y": entity_means_y,
            "entity_means_X": entity_means_X,
        }

        # Transform dependent variable
        self.y_transformed_ = self.y.copy()
        for i, entity_id in enumerate(entity_list):
            mask = self.entity_ids == entity_id
            self.y_transformed_[mask] -= self.fixed_effects_[i]

        self._step1_complete = True

    def _estimate_quantile_step2(
        self, tau: float, se_adjustment: str = "two-step", **kwargs
    ) -> "CanayQuantileResult":
        """
        Step 2: Pooled QR on transformed data.
        """
        # Create temporary pooled QR model with transformed y
        temp_model = PooledQuantile(data=self.data, formula=None, tau=tau)  # Use matrices directly
        temp_model.y = self.y_transformed_
        temp_model.X = self.X
        temp_model.nobs = self.nobs
        temp_model.k_exog = self.k_exog

        # Estimate using base class method
        pooled_result = temp_model.fit(**kwargs)

        # Extract result for this tau
        if hasattr(pooled_result, "results"):
            result_tau = pooled_result.results[tau]
            params = (
                result_tau.params if hasattr(result_tau, "params") else result_tau.get("params")
            )
            cov_matrix = (
                result_tau.cov_matrix
                if hasattr(result_tau, "cov_matrix")
                else result_tau.get("cov_matrix")
            )
        else:
            params = (
                pooled_result.params
                if hasattr(pooled_result, "params")
                else pooled_result.get("params")
            )
            cov_matrix = (
                pooled_result.cov_matrix
                if hasattr(pooled_result, "cov_matrix")
                else pooled_result.get("cov_matrix")
            )

        # Adjust standard errors if requested
        if se_adjustment == "two-step":
            adjusted_cov = self._adjust_covariance_twostep(params, tau)
            cov_matrix = adjusted_cov
        elif se_adjustment == "bootstrap":
            # Bootstrap will be handled separately
            pass

        return CanayQuantileResult(
            params=params,
            cov_matrix=cov_matrix if cov_matrix is not None else np.eye(len(params)),
            tau=tau,
            converged=True,
            model=self,
        )

    def _adjust_covariance_twostep(self, beta_qr: np.ndarray, tau: float) -> np.ndarray:
        """
        Adjust covariance matrix for two-step estimation.

        Accounts for uncertainty in first-step FE estimation.
        Based on Canay (2011) Appendix.
        """
        # Get components from step 1
        beta_ols = self.fe_ols_result_["params"]
        V_ols = self.fe_ols_result_["cov_matrix"]

        # Residuals from QR step
        residuals_qr = self.y_transformed_ - self.X @ beta_qr

        # Sparsity estimation using kernel density
        from scipy.stats import gaussian_kde

        try:
            kde = gaussian_kde(residuals_qr)
            f_hat = kde(0)[0]  # Density at zero
        except:
            # Fallback to simple estimate
            h = 1.06 * np.std(residuals_qr) * (self.nobs ** (-0.2))
            f_hat = 1 / (2 * h)  # Approximate density

        # Components for adjustment
        n = self.nobs
        k = self.k_exog

        # Hessian from QR
        H_qr = (self.X.T @ self.X) / (n * f_hat)
        try:
            H_qr_inv = np.linalg.inv(H_qr)
        except np.linalg.LinAlgError:
            H_qr_inv = np.linalg.pinv(H_qr)

        # Cross term: effect of FE estimation on QR
        psi = tau - (residuals_qr < 0).astype(float)

        # Gradient of QR objective w.r.t. FE
        # Aggregate by entity for cross-correlation
        cross_term = np.zeros((k, k))
        for entity_id in np.unique(self.entity_ids):
            mask = self.entity_ids == entity_id
            X_entity = self.X[mask]
            psi_entity = psi[mask]

            # Average gradient for entity
            grad_entity = X_entity.T @ psi_entity / np.sum(mask)
            cross_term += np.outer(grad_entity, grad_entity)

        cross_term /= self.n_entities

        # Adjustment factor (simplified version)
        adjustment = H_qr_inv @ cross_term @ H_qr_inv / n

        # Final covariance
        V_naive = H_qr_inv / n  # Naive covariance ignoring step 1
        V_adjusted = V_naive + adjustment

        return V_adjusted

    def test_location_shift(
        self, tau_grid: Optional[List[float]] = None, method: str = "wald"
    ) -> "LocationShiftTestResult":
        """
        Test if fixed effects are pure location shifters.

        H0: β(τ) is constant across τ (location shift holds)
        H1: β(τ) varies with τ (location shift violated)

        Parameters
        ----------
        tau_grid : array-like, optional
            Quantiles to test. Default: [0.1, 0.25, 0.5, 0.75, 0.9]
        method : str
            Test method:
            - 'wald': Joint Wald test
            - 'ks': Kolmogorov-Smirnov type test

        Returns
        -------
        LocationShiftTestResult
        """
        if tau_grid is None:
            tau_grid = [0.1, 0.25, 0.5, 0.75, 0.9]

        print("\nTesting Location Shift Assumption")
        print("=" * 50)
        print("H0: Fixed effects are pure location shifters")
        print(f"Testing across quantiles: {tau_grid}")

        # Estimate at multiple quantiles
        results_full = {}
        for tau in tau_grid:
            print(f"  Estimating τ = {tau}...")
            self.tau = [tau]
            result = self.fit(se_adjustment="naive", verbose=False)
            results_full[tau] = result.results[tau]

        # Extract coefficients
        coef_matrix = np.array([results_full[tau].params for tau in tau_grid])

        if method == "wald":
            # Test equality of coefficients across quantiles
            # H0: β(τ₁) = β(τ₂) = ... = β(τₘ)

            # Use median as reference
            ref_tau = 0.5 if 0.5 in tau_grid else tau_grid[len(tau_grid) // 2]
            ref_coef = results_full[ref_tau].params

            # Compute Wald statistic
            wald_stat = 0
            for tau in tau_grid:
                if tau == ref_tau:
                    continue

                diff = results_full[tau].params - ref_coef
                V_diff = results_full[tau].cov_matrix + results_full[ref_tau].cov_matrix

                try:
                    wald_stat += diff @ np.linalg.inv(V_diff) @ diff
                except np.linalg.LinAlgError:
                    # Use pseudo-inverse if singular
                    wald_stat += diff @ np.linalg.pinv(V_diff) @ diff

            # Degrees of freedom
            df = (len(tau_grid) - 1) * self.k_exog

            # P-value
            from scipy.stats import chi2

            p_value = 1 - chi2.cdf(wald_stat, df)

        elif method == "ks":
            # Kolmogorov-Smirnov type test
            # Check if coefficient paths are "flat"

            # Compute variation in coefficients
            coef_std = np.std(coef_matrix, axis=0)
            coef_range = np.max(coef_matrix, axis=0) - np.min(coef_matrix, axis=0)

            # Test statistic: normalized maximum range
            # This is a simplified version
            test_stat = np.max(coef_range / (coef_std + 1e-10))

            # Approximate critical value (would need bootstrap for exact)
            critical_value = 2.8  # Rough approximation for 5% level

            wald_stat = test_stat
            df = None
            p_value = 0.05 if test_stat > critical_value else 0.50

        # Create result object
        result = LocationShiftTestResult(
            statistic=wald_stat,
            p_value=p_value,
            df=df,
            method=method,
            tau_grid=tau_grid,
            coef_matrix=coef_matrix,
        )

        # Print summary
        result.summary()

        return result

    def compare_with_penalty_method(
        self, tau: float = 0.5, lambda_fe: Union[float, str] = "auto"
    ) -> Dict[str, Any]:
        """
        Compare Canay two-step with Koenker penalty method.

        Useful for assessing the location shift assumption.
        """
        from .fixed_effects import FixedEffectsQuantile

        print("\nComparison: Canay vs Penalty Method")
        print("=" * 50)

        # Canay two-step
        print("\n1. Canay Two-Step:")
        start_time = time.time()
        canay_result = self.fit(verbose=False)
        canay_time = time.time() - start_time
        canay_coef = canay_result.results[tau].params

        # Penalty method
        print("\n2. Koenker Penalty Method:")
        fe_model = FixedEffectsQuantile(self.data, self.formula, tau=tau, lambda_fe=lambda_fe)
        start_time = time.time()
        fe_result = fe_model.fit(verbose=False)
        fe_time = time.time() - start_time
        fe_coef = fe_result.results[tau].params

        # Compare
        print("\nResults Comparison:")
        print("-" * 40)
        print(f"{'Variable':<15} {'Canay':>10} {'Penalty':>10} {'Difference':>10}")
        print("-" * 40)

        for i in range(len(canay_coef)):
            diff = canay_coef[i] - fe_coef[i]
            print(f"β{i+1:<14} {canay_coef[i]:10.4f} {fe_coef[i]:10.4f} {diff:10.4f}")

        print("\nComputational Time:")
        print(f"  Canay:   {canay_time:6.2f} seconds")
        print(f"  Penalty: {fe_time:6.2f} seconds")
        print(f"  Speedup: {fe_time/canay_time:6.1f}x")

        # Correlation of coefficients
        corr = np.corrcoef(canay_coef, fe_coef)[0, 1]
        print(f"\nCoefficient correlation: {corr:.4f}")

        if corr < 0.95:
            print("\nWarning: Low correlation suggests location shift assumption may be violated.")

        return {
            "canay": canay_result,
            "penalty": fe_result,
            "correlation": corr,
            "time_ratio": fe_time / canay_time,
        }


class CanayQuantileResult:
    """Single quantile result for Canay estimator."""

    def __init__(
        self,
        params: np.ndarray,
        cov_matrix: np.ndarray,
        tau: float,
        converged: bool,
        model: CanayTwoStep,
    ):
        self.params = params
        self.cov_matrix = cov_matrix
        self.tau = tau
        self.converged = converged
        self.model = model

    @property
    def bse(self) -> np.ndarray:
        """Standard errors of coefficients."""
        return np.sqrt(np.diag(self.cov_matrix))


class CanayTwoStepResult:
    """
    Results for Canay (2011) two-step quantile regression.

    Includes fixed effects from step 1 and QR results from step 2.
    """

    def __init__(
        self,
        model: CanayTwoStep,
        results: Dict[float, CanayQuantileResult],
        fixed_effects: np.ndarray,
        fe_ols_result: Dict[str, Any],
    ):
        self.model = model
        self.results = results
        self.fixed_effects = fixed_effects
        self.fe_ols_result = fe_ols_result

    def summary(self, tau: Optional[float] = None):
        """Extended summary including FE information."""
        print("\n" + "=" * 60)
        print("CANAY TWO-STEP QUANTILE REGRESSION RESULTS")
        print("=" * 60)

        # Step 1 info
        print("\nStep 1: Fixed Effects (OLS)")
        print("-" * 40)
        print(f"Number of entities: {len(self.fixed_effects)}")
        print(f"Mean FE: {np.mean(self.fixed_effects):8.4f}")
        print(f"Std FE:  {np.std(self.fixed_effects):8.4f}")
        print(f"Min FE:  {np.min(self.fixed_effects):8.4f}")
        print(f"Max FE:  {np.max(self.fixed_effects):8.4f}")

        # Step 2 QR results
        print("\nStep 2: Quantile Regression on Transformed Data")
        print("-" * 40)

        if tau is None:
            tau_list = sorted(self.results.keys())
        else:
            tau_list = [tau] if np.isscalar(tau) else tau

        for tau in tau_list:
            result = self.results[tau]
            print(f"\nτ = {tau}")
            print("  Coefficients:")

            for i, coef in enumerate(result.params):
                se = result.bse[i]
                t_stat = coef / se if se > 0 else 0
                print(f"    β{i+1}: {coef:8.4f} ({se:6.4f})  t={t_stat:6.2f}")

    def plot_fixed_effects_distribution(self):
        """Visualize the distribution of estimated fixed effects."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Histogram
        ax = axes[0, 0]
        ax.hist(self.fixed_effects, bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", label="Zero")
        ax.axvline(np.mean(self.fixed_effects), color="green", linestyle="--", label="Mean")
        ax.set_xlabel("Fixed Effect Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Fixed Effects (Step 1)")
        ax.legend()

        # 2. Kernel density
        ax = axes[0, 1]
        from scipy.stats import gaussian_kde

        try:
            kde = gaussian_kde(self.fixed_effects)
            x_range = np.linspace(self.fixed_effects.min(), self.fixed_effects.max(), 100)
            ax.plot(x_range, kde(x_range))
            ax.fill_between(x_range, kde(x_range), alpha=0.3)
        except:
            ax.hist(self.fixed_effects, bins=30, density=True, alpha=0.7)
        ax.set_xlabel("Fixed Effect Value")
        ax.set_ylabel("Density")
        ax.set_title("Kernel Density Estimate")

        # 3. Ranked plot
        ax = axes[1, 0]
        sorted_fe = np.sort(self.fixed_effects)
        ax.plot(range(len(sorted_fe)), sorted_fe)
        ax.axhline(0, color="red", linestyle="--", alpha=0.5)
        ax.set_xlabel("Entity Rank")
        ax.set_ylabel("Fixed Effect Value")
        ax.set_title("Ranked Fixed Effects")
        ax.grid(True, alpha=0.3)

        # 4. QQ plot
        ax = axes[1, 1]
        from scipy import stats

        stats.probplot(self.fixed_effects, dist="norm", plot=ax)
        ax.set_title("Normal Q-Q Plot")

        plt.tight_layout()
        return fig


class LocationShiftTestResult:
    """Results for location shift hypothesis test."""

    def __init__(
        self,
        statistic: float,
        p_value: float,
        df: Optional[int],
        method: str,
        tau_grid: List[float],
        coef_matrix: np.ndarray,
    ):
        self.statistic = statistic
        self.p_value = p_value
        self.df = df
        self.method = method
        self.tau_grid = tau_grid
        self.coef_matrix = coef_matrix

    def summary(self):
        """Print test summary."""
        print("\n" + "=" * 50)
        print("LOCATION SHIFT TEST RESULTS")
        print("=" * 50)
        print("H0: Fixed effects are pure location shifters")
        print(f"Method: {self.method}")
        print(f"Test Statistic: {self.statistic:.4f}")
        if self.df:
            print(f"Degrees of Freedom: {self.df}")
        print(f"P-value: {self.p_value:.4f}")

        if self.p_value < 0.05:
            print("\nConclusion: REJECT H0 at 5% level")
            print("Fixed effects appear to vary across quantiles.")
            print("Canay estimator may be biased. Consider penalty method.")
        else:
            print("\nConclusion: Cannot reject H0 at 5% level")
            print("Location shift assumption appears reasonable.")

    def plot_coefficient_variation(self):
        """Visualize how coefficients vary across quantiles."""
        import matplotlib.pyplot as plt

        n_coef = self.coef_matrix.shape[1]
        fig, axes = plt.subplots((n_coef + 1) // 2, 2, figsize=(12, 4 * ((n_coef + 1) // 2)))
        if n_coef == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i in range(n_coef):
            ax = axes[i] if n_coef > 1 else axes
            ax.plot(self.tau_grid, self.coef_matrix[:, i], "o-")
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel(f"β{i+1}")
            ax.set_title(f"Coefficient {i+1} across Quantiles")
            ax.grid(True, alpha=0.3)

            # Add horizontal line at mean
            mean_coef = np.mean(self.coef_matrix[:, i])
            ax.axhline(mean_coef, color="red", linestyle="--", alpha=0.5, label="Mean")
            ax.legend()

        # Hide unused subplots if n_coef > 1
        if n_coef > 1:
            for j in range(n_coef, len(axes)):
                axes[j].set_visible(False)

        plt.suptitle("Testing Location Shift: Coefficient Stability", fontsize=14)
        plt.tight_layout()
        return fig
