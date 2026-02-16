"""
Location-Scale Quantile Regression Model.

Implements Machado-Santos Silva (2019) "Quantiles via Moments" approach
for panel data quantile regression.

Key features:
- Guarantees non-crossing quantile curves by construction
- Computationally efficient (method of moments)
- Natural handling of fixed effects
- Allows extrapolation beyond observed quantiles

References:
    Machado, J. A., & Santos Silva, J. M. C. (2019).
    Quantiles via moments. Journal of Econometrics, 213(1), 145-173.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import digamma

from .base import QuantilePanelModel, QuantilePanelResult


class LocationScale(QuantilePanelModel):
    """
    Machado-Santos Silva (2019) Quantiles via Moments.

    Estimates conditional quantiles through location-scale model:
    Q_y(τ|X) = X'α + (X'γ)^(1/2) × q(τ)

    where:
    - α: location parameters (conditional mean)
    - γ: scale parameters (conditional variance)
    - q(τ): quantile function of reference distribution

    Key advantages:
    - Guarantees non-crossing quantile curves
    - Computationally efficient (method of moments)
    - Natural handling of panel fixed effects
    - Allows extrapolation beyond observed quantiles

    Parameters
    ----------
    data : PanelData
        Panel data object
    formula : str, optional
        Model formula
    tau : float or array-like
        Quantile(s) to estimate
    distribution : str or callable
        Reference distribution: 'normal', 'logistic', 't', 'laplace'
        or callable returning quantile function
    fixed_effects : bool
        Include entity fixed effects
    df_t : float
        Degrees of freedom if distribution='t'
    """

    def __init__(
        self,
        data,
        formula: Optional[str] = None,
        tau: Union[float, np.ndarray] = 0.5,
        distribution: Union[str, callable] = "normal",
        fixed_effects: bool = False,
        df_t: float = 5,
    ):
        # Store data and parameters
        self.data = data
        self.formula = formula
        self.tau = np.atleast_1d(tau)
        self.distribution = distribution
        self.fixed_effects = fixed_effects
        self.df_t = df_t

        # Store estimated parameters
        self.location_params_ = None
        self.scale_params_ = None
        self.location_result_ = None
        self.scale_result_ = None
        self.location_residuals_ = None
        self.location_fitted_ = None

    def _objective(self, params: np.ndarray, tau: float) -> float:
        """
        Compute objective function (not used for location-scale, but required by abstract base class).

        The location-scale model doesn't directly optimize the check loss; instead, it uses
        method of moments (OLS for location, log-OLS for scale). This method is provided
        for interface compatibility.
        """
        # This model uses method of moments, not direct quantile optimization
        # Return 0.0 as placeholder
        return 0.0

    def fit(
        self, robust_scale: bool = True, verbose: bool = False, **kwargs
    ) -> "LocationScaleResult":
        """
        Estimate location and scale parameters via method of moments.

        Parameters
        ----------
        robust_scale : bool
            Use robust scale estimation (log transformation)
        verbose : bool
            Print progress
        **kwargs
            Additional arguments

        Returns
        -------
        LocationScaleResult
            Fitted model results
        """
        if verbose:
            print("Location-Scale Quantile Regression (MSS 2019)")
            print("=" * 50)

        # Step 1: Estimate location (conditional mean)
        if verbose:
            print("\nStep 1: Estimating location parameters...")

        self._estimate_location()

        if verbose:
            print(f"  Location R²: {self.location_result_.r_squared:.4f}")

        # Step 2: Estimate scale (conditional variance)
        if verbose:
            print("\nStep 2: Estimating scale parameters...")

        self._estimate_scale(robust=robust_scale)

        if verbose:
            print(f"  Scale R²: {self.scale_result_.r_squared:.4f}")

        # Step 3: Compute quantile coefficients for requested τ
        if verbose:
            print(f"\nStep 3: Computing quantile coefficients for τ = {self.tau}")

        results = {}
        for tau in self.tau:
            beta_tau = self._compute_quantile_coefficients(tau)
            cov_tau = self._compute_covariance_delta_method(tau)

            results[tau] = LocationScaleQuantileResult(
                params=beta_tau,
                cov_matrix=cov_tau,
                tau=tau,
                location_params=self.location_params_,
                scale_params=self.scale_params_,
                distribution=self.distribution,
                model=self,
            )

        if verbose:
            print("\nEstimation complete!")

        return LocationScaleResult(
            model=self,
            results=results,
            location_result=self.location_result_,
            scale_result=self.scale_result_,
        )

    def _estimate_location(self):
        """
        Step 1: Estimate location parameters (conditional mean).

        Uses OLS or Fixed Effects OLS depending on specification.
        """
        if self.fixed_effects:
            from ..linear.fixed_effects import FixedEffectsOLS

            location_model = FixedEffectsOLS(self.data, self.formula)
        else:
            from ..linear.pooled import PooledOLS

            location_model = PooledOLS(self.data, self.formula)

        self.location_result_ = location_model.fit()
        self.location_params_ = self.location_result_.params
        self.location_residuals_ = self.location_result_.resids

        # Store fitted values for later
        self.location_fitted_ = self.y - self.location_residuals_

    def _estimate_scale(self, robust: bool = True):
        """
        Step 2: Estimate scale parameters (conditional variance).

        Uses regression on squared residuals or log transformation.

        Parameters
        ----------
        robust : bool
            Use log transformation for robust estimation
        """
        if robust:
            # Log transformation approach (more robust)
            # log(|ε_i|) = X'γ/2 + v_i

            # Avoid log(0) issues
            abs_resids = np.abs(self.location_residuals_)
            abs_resids = np.maximum(abs_resids, 1e-10)

            y_scale = np.log(abs_resids)

            # Adjustment for E[log|ε|] under different distributions
            adjustment = self._get_log_residual_adjustment()
            y_scale = y_scale - adjustment

        else:
            # Direct squared residuals approach
            # ε_i² = exp(X'γ) + v_i
            y_scale = self.location_residuals_**2

        # Create temporary data for scale regression
        import pandas as pd

        from ...utils.data import PanelData

        # Create a temporary PanelData object for scale regression
        scale_data = pd.DataFrame(self.X, columns=[f"X{i}" for i in range(self.X.shape[1])])
        scale_data["y_scale"] = y_scale
        scale_data["entity"] = self.data.entity_id
        scale_data["time"] = self.data.time_id

        scale_panel_data = PanelData(scale_data, entity="entity", time="time")

        # Estimate scale model
        if self.fixed_effects:
            from ..linear.fixed_effects import FixedEffectsOLS

            scale_model = FixedEffectsOLS(
                data=scale_panel_data,
                formula="y_scale ~ "
                + " + ".join([f"X{i}" for i in range(self.X.shape[1])])
                + " - 1",
            )
        else:
            from ..linear.pooled import PooledOLS

            scale_model = PooledOLS(
                data=scale_panel_data,
                formula="y_scale ~ "
                + " + ".join([f"X{i}" for i in range(self.X.shape[1])])
                + " - 1",
            )

        self.scale_result_ = scale_model.fit()

        if robust:
            # γ parameters (need to multiply by 2 for variance scale)
            self.scale_params_ = self.scale_result_.params * 2
        else:
            # Transform to log scale for consistency
            self.scale_params_ = np.log(np.maximum(self.scale_result_.params, 1e-10))

    def _get_log_residual_adjustment(self) -> float:
        """
        Get E[log|ε|] adjustment for different distributions.

        This ensures unbiased scale estimation.
        """
        if self.distribution == "normal":
            # E[log|Z|] for Z ~ N(0,1)
            return -0.5 * (np.log(2) + np.log(np.pi)) - np.euler_gamma / 2
        elif self.distribution == "logistic":
            # E[log|Z|] for Z ~ Logistic(0,1)
            return -np.log(2)
        elif self.distribution == "t":
            # Approximation for t-distribution
            return digamma(self.df_t / 2) - digamma((self.df_t - 1) / 2)
        elif self.distribution == "laplace":
            # E[log|Z|] for Z ~ Laplace(0,1)
            return -np.euler_gamma
        else:
            # Default: no adjustment
            return 0

    def _compute_quantile_coefficients(self, tau: float) -> np.ndarray:
        """
        Step 3: Compute quantile coefficients.

        β(τ) = α + σ × q(τ)

        where σ = exp(γ/2) and q(τ) is quantile function.
        """
        q_tau = self._get_quantile_function(tau)

        # Note: scale_params_ contains γ (log-variance scale)
        # We need σ = exp(γ/2) for standard deviation scale
        scale_multiplier = np.exp(self.scale_params_ / 2)

        # Quantile coefficients
        beta_tau = self.location_params_ + scale_multiplier * q_tau

        return beta_tau

    def _get_quantile_function(self, tau: float) -> float:
        """
        Get q(τ) for specified reference distribution.

        Returns
        -------
        float
            Quantile value for standard distribution
        """
        if callable(self.distribution):
            # User-provided quantile function
            return self.distribution(tau)

        elif self.distribution == "normal":
            return stats.norm.ppf(tau)

        elif self.distribution == "logistic":
            return stats.logistic.ppf(tau)

        elif self.distribution == "t":
            return stats.t.ppf(tau, df=self.df_t)

        elif self.distribution == "laplace":
            return stats.laplace.ppf(tau)

        elif self.distribution == "uniform":
            # Uniform on [-√3, √3] for unit variance
            return 2 * np.sqrt(3) * (tau - 0.5)

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _compute_covariance_delta_method(self, tau: float) -> np.ndarray:
        """
        Compute covariance matrix for β(τ) via delta method.

        β(τ) = α + exp(γ/2) × q(τ)

        Var[β(τ)] = Var[α] + q²(τ) × J × Var[γ] × J' + cross-terms

        where J is Jacobian of exp(γ/2) w.r.t. γ
        """
        q_tau = self._get_quantile_function(tau)

        # Get covariance matrices from two-step estimation
        var_alpha = self.location_result_.cov_matrix
        var_gamma = self.scale_result_.cov_matrix * 4  # Adjust for γ = 2×(scale params)

        # For simplicity, assume independence between steps
        # (More complex: account for correlation)

        # Jacobian of σ = exp(γ/2) w.r.t. γ
        scale_multiplier = np.exp(self.scale_params_ / 2)
        jacobian = np.diag(scale_multiplier / 2)

        # Variance of scale term
        var_scale_term = jacobian @ var_gamma @ jacobian.T

        # Total variance (assuming independence)
        var_beta_tau = var_alpha + q_tau**2 * var_scale_term

        return var_beta_tau

    def predict_quantiles(
        self,
        X: Optional[np.ndarray] = None,
        tau: Optional[Union[float, np.ndarray]] = None,
        ci: bool = True,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """
        Predict quantiles for given covariates.

        Guarantees monotonicity across τ by construction.

        Parameters
        ----------
        X : array-like, optional
            Covariates. If None, use estimation sample
        tau : float or array, optional
            Quantile(s) to predict
        ci : bool
            Include confidence intervals
        alpha : float
            Significance level for CI

        Returns
        -------
        DataFrame
            Predicted quantiles (and CI if requested)
        """
        if X is None:
            X = self.X
        if tau is None:
            tau = self.tau

        tau = np.atleast_1d(tau)

        # Compute location and scale for X
        location = X @ self.location_params_
        log_scale = X @ self.scale_params_
        scale = np.exp(log_scale / 2)

        predictions = pd.DataFrame(index=range(len(X)))

        for t in tau:
            q_t = self._get_quantile_function(t)

            # Predicted quantile
            pred = location + scale * q_t
            predictions[f"q{int(t*100)}"] = pred

            if ci:
                # Approximate CI via delta method
                # This is simplified - full version would be more complex
                se_location = np.sqrt(np.diag(X @ self.location_result_.cov_matrix @ X.T))
                se_scale = np.sqrt(np.diag(X @ self.scale_result_.cov_matrix @ X.T))

                # Approximate combined SE
                se_combined = np.sqrt(se_location**2 + (q_t * scale * se_scale) ** 2)

                z_alpha = stats.norm.ppf(1 - alpha / 2)
                predictions[f"q{int(t*100)}_lower"] = pred - z_alpha * se_combined
                predictions[f"q{int(t*100)}_upper"] = pred + z_alpha * se_combined

        return predictions

    def predict_density(
        self,
        X: Optional[np.ndarray] = None,
        y_grid: Optional[np.ndarray] = None,
        n_points: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict complete conditional density.

        Uses the location-scale structure to compute f(y|X).

        Parameters
        ----------
        X : array-like, optional
            Covariates (single observation or mean)
        y_grid : array-like, optional
            Points to evaluate density
        n_points : int
            Number of grid points if y_grid not provided

        Returns
        -------
        y_grid : array
            Evaluation points
        density : array
            Density values
        """
        if X is None:
            X = np.mean(self.X, axis=0, keepdims=True)

        # Get location and scale
        location = X @ self.location_params_
        scale = np.exp(X @ self.scale_params_ / 2)

        if y_grid is None:
            # Create grid around predicted range
            y_min = location - 4 * scale
            y_max = location + 4 * scale
            y_grid = np.linspace(y_min, y_max, n_points).flatten()

        # Standardize
        z = (y_grid - location.flatten()[:, np.newaxis]) / scale.flatten()[:, np.newaxis]

        # Get density of reference distribution
        if self.distribution == "normal":
            density = stats.norm.pdf(z) / scale.flatten()[:, np.newaxis]
        elif self.distribution == "logistic":
            density = stats.logistic.pdf(z) / scale.flatten()[:, np.newaxis]
        elif self.distribution == "t":
            density = stats.t.pdf(z, df=self.df_t) / scale.flatten()[:, np.newaxis]
        elif self.distribution == "laplace":
            density = stats.laplace.pdf(z) / scale.flatten()[:, np.newaxis]
        else:
            raise ValueError(f"Density not available for {self.distribution}")

        return y_grid, density.squeeze()

    def test_normality(self, tau_grid: Optional[np.ndarray] = None) -> "NormalityTestResult":
        """
        Test if normal distribution is appropriate.

        Compares empirical quantiles with theoretical under normality.

        Parameters
        ----------
        tau_grid : array-like, optional
            Quantiles to test

        Returns
        -------
        NormalityTestResult
            Test results
        """
        if tau_grid is None:
            tau_grid = np.arange(0.05, 1.0, 0.05)

        # Standardized residuals
        location = self.location_fitted_
        scale = np.exp(self.X @ self.scale_params_ / 2)
        z_resids = (self.y - location) / scale

        # Empirical vs theoretical quantiles
        empirical_q = np.quantile(z_resids, tau_grid)
        theoretical_q = stats.norm.ppf(tau_grid)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(z_resids, "norm")

        # Jarque-Bera test
        jb_stat, jb_pval = stats.jarque_bera(z_resids)

        return NormalityTestResult(
            ks_stat=ks_stat,
            ks_pval=ks_pval,
            jb_stat=jb_stat,
            jb_pval=jb_pval,
            empirical_quantiles=empirical_q,
            theoretical_quantiles=theoretical_q,
            tau_grid=tau_grid,
        )


class LocationScaleQuantileResult:
    """Results for a single quantile from Location-Scale model."""

    def __init__(
        self,
        params: np.ndarray,
        cov_matrix: np.ndarray,
        tau: float,
        location_params: np.ndarray,
        scale_params: np.ndarray,
        distribution: str,
        model: LocationScale,
    ):
        self.params = params
        self.cov_matrix = cov_matrix
        self.tau = tau
        self.location_params = location_params
        self.scale_params = scale_params
        self.distribution = distribution
        self.model = model

    @property
    def std_errors(self) -> np.ndarray:
        """Standard errors of coefficients."""
        return np.sqrt(np.diag(self.cov_matrix))

    @property
    def t_stats(self) -> np.ndarray:
        """T-statistics."""
        return self.params / self.std_errors

    @property
    def p_values(self) -> np.ndarray:
        """P-values (two-tailed)."""
        return 2 * (1 - stats.norm.cdf(np.abs(self.t_stats)))


class LocationScaleResult(QuantilePanelResult):
    """Results for Location-Scale quantile regression."""

    def __init__(self, model, results, location_result, scale_result):
        super().__init__(model, results)
        self.location_result = location_result
        self.scale_result = scale_result

    def summary(self, tau: Optional[float] = None):
        """Extended summary for location-scale model."""
        print("\n" + "=" * 60)
        print("LOCATION-SCALE QUANTILE REGRESSION (MSS 2019)")
        print("=" * 60)

        print(f"\nDistribution: {self.model.distribution}")
        print(f"Fixed Effects: {self.model.fixed_effects}")

        print("\nLocation Model (Conditional Mean):")
        print("-" * 40)
        print(f"R²: {self.location_result.r_squared:.4f}")
        for i, param in enumerate(self.location_result.params):
            se = np.sqrt(self.location_result.cov_matrix[i, i])
            print(f"  α{i+1}: {param:8.4f} ({se:.4f})")

        print("\nScale Model (Log Conditional Variance):")
        print("-" * 40)
        print(f"R²: {self.scale_result.r_squared:.4f}")
        for i, param in enumerate(self.model.scale_params_):
            se = np.sqrt(self.scale_result.cov_matrix[i, i] * 4)
            print(f"  γ{i+1}: {param:8.4f} ({se:.4f})")

        print("\nImplied Quantile Coefficients:")
        print("-" * 40)

        if tau is None:
            tau_list = sorted(self.results.keys())
        else:
            tau_list = [tau] if np.isscalar(tau) else tau

        # Table header
        print(f"{'Coef':<8}", end="")
        for t in tau_list:
            print(f"τ={t:4.2f}  ", end="")
        print()
        print("-" * (8 + 8 * len(tau_list)))

        # Coefficients for each tau
        for i in range(len(self.location_result.params)):
            print(f"β{i+1:<7}", end="")
            for t in tau_list:
                coef = self.results[t].params[i]
                print(f"{coef:7.3f} ", end="")
            print()

    def plot_location_scale_effects(self):
        """Visualize location and scale contributions."""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Location effects
        params = self.location_result.params
        se = np.sqrt(np.diag(self.location_result.cov_matrix))

        x = range(len(params))
        ax1.bar(x, params, yerr=1.96 * se, capsize=5, alpha=0.7)
        ax1.set_xlabel("Variable")
        ax1.set_ylabel("Location Effect (α)")
        ax1.set_title("Location Parameters (Conditional Mean)")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"X{i+1}" for i in x])
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.grid(True, alpha=0.3)

        # Scale effects
        params = self.model.scale_params_
        se = np.sqrt(np.diag(self.scale_result.cov_matrix * 4))

        ax2.bar(x, params, yerr=1.96 * se, capsize=5, alpha=0.7, color="orange")
        ax2.set_xlabel("Variable")
        ax2.set_ylabel("Scale Effect (γ)")
        ax2.set_title("Scale Parameters (Log Conditional Variance)")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"X{i+1}" for i in x])
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_quantile_decomposition(self, var_idx: int = 0):
        """
        Decompose quantile effect into location and scale components.

        Shows how β(τ) = α + σ×q(τ) varies with τ.
        """
        import matplotlib.pyplot as plt

        tau_grid = np.arange(0.05, 1.0, 0.05)

        # Get components
        alpha = self.location_result.params[var_idx]
        gamma = self.model.scale_params_[var_idx]
        sigma = np.exp(gamma / 2)

        # Compute quantile coefficients
        beta_tau = []
        scale_contribution = []

        for tau in tau_grid:
            q_tau = self.model._get_quantile_function(tau)
            beta = alpha + sigma * q_tau
            beta_tau.append(beta)
            scale_contribution.append(sigma * q_tau)

        beta_tau = np.array(beta_tau)
        scale_contribution = np.array(scale_contribution)

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Total effect
        ax1.plot(tau_grid, beta_tau, "b-", linewidth=2, label="Total β(τ)")
        ax1.axhline(alpha, color="red", linestyle="--", label="Location (α)")
        ax1.fill_between(tau_grid, alpha, beta_tau, alpha=0.3)
        ax1.set_xlabel("Quantile (τ)")
        ax1.set_ylabel(f"Coefficient for Variable {var_idx+1}")
        ax1.set_title("Quantile Coefficient Decomposition")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Scale contribution
        ax2.plot(tau_grid, scale_contribution, "g-", linewidth=2)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_xlabel("Quantile (τ)")
        ax2.set_ylabel("Scale Contribution (σ×q(τ))")
        ax2.set_title("Scale Effect Across Quantiles")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


class NormalityTestResult:
    """Results from normality test for location-scale model."""

    def __init__(
        self,
        ks_stat: float,
        ks_pval: float,
        jb_stat: float,
        jb_pval: float,
        empirical_quantiles: np.ndarray,
        theoretical_quantiles: np.ndarray,
        tau_grid: np.ndarray,
    ):
        self.ks_stat = ks_stat
        self.ks_pval = ks_pval
        self.jb_stat = jb_stat
        self.jb_pval = jb_pval
        self.empirical_quantiles = empirical_quantiles
        self.theoretical_quantiles = theoretical_quantiles
        self.tau_grid = tau_grid

    def summary(self):
        """Print test summary."""
        print("\nNormality Test Results")
        print("=" * 40)
        print(f"Kolmogorov-Smirnov test:")
        print(f"  Statistic: {self.ks_stat:.4f}")
        print(f"  P-value:   {self.ks_pval:.4f}")
        print(f"\nJarque-Bera test:")
        print(f"  Statistic: {self.jb_stat:.4f}")
        print(f"  P-value:   {self.jb_pval:.4f}")

        if self.ks_pval < 0.05 or self.jb_pval < 0.05:
            print("\n⚠ Warning: Evidence against normality assumption")
            print("Consider using alternative distributions (logistic, t, laplace)")
        else:
            print("\n✓ Normal distribution appears reasonable")

    def plot_qq(self):
        """Q-Q plot comparing empirical and theoretical quantiles."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))

        ax.scatter(self.theoretical_quantiles, self.empirical_quantiles, alpha=0.6)

        # Add diagonal reference line
        lims = [
            np.min([self.theoretical_quantiles.min(), self.empirical_quantiles.min()]),
            np.max([self.theoretical_quantiles.max(), self.empirical_quantiles.max()]),
        ]
        ax.plot(lims, lims, "r--", alpha=0.5, label="Perfect fit")

        ax.set_xlabel("Theoretical Quantiles (Normal)")
        ax.set_ylabel("Empirical Quantiles")
        ax.set_title("Q-Q Plot for Normality Assessment")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
