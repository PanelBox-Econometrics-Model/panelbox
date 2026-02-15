"""
Results class for stochastic frontier estimation.

This module defines the SFResult class which stores and presents
estimation results from SFA models.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .data import DistributionType, FrontierType, ModelType


class SFResult:
    """Results from stochastic frontier estimation.

    Attributes:
        params: Estimated parameters as Series
        se: Standard errors
        tvalues: t-statistics
        pvalues: p-values
        loglik: Log-likelihood value
        aic: Akaike Information Criterion
        bic: Bayesian Information Criterion
        sigma_v: Standard deviation of noise (v)
        sigma_u: Standard deviation of inefficiency (u)
        sigma: Composite standard deviation √(σ²_v + σ²_u)
        lambda_param: λ = σ_u / σ_v
        gamma: γ = σ²_u / (σ²_v + σ²_u)
        converged: Whether optimization converged
        nobs: Number of observations
        nparams: Number of parameters
        model: Reference to original model
    """

    def __init__(
        self,
        params: np.ndarray,
        param_names: list,
        hessian: Optional[np.ndarray],
        loglik: float,
        converged: bool,
        model: Any,
        optimization_result: Any = None,
    ):
        """Initialize SFResult.

        Parameters:
            params: Parameter estimates (transformed to natural scale)
            param_names: Parameter names
            hessian: Hessian matrix at optimum (for standard errors)
            loglik: Log-likelihood value
            converged: Convergence flag
            model: StochasticFrontier model object
            optimization_result: Full scipy optimization result
        """
        self.model = model
        self.converged = converged
        self.loglik = loglik
        self.optimization_result = optimization_result

        # Store parameters as Series
        self.params = pd.Series(params, index=param_names, name="coefficient")

        # Compute variance-covariance matrix from Hessian
        self._compute_inference(hessian, params, param_names)

        # Extract variance components
        self._extract_variance_components()

        # Compute information criteria
        self.nobs = model.n_obs
        self.nparams = len(params)
        self.aic = -2 * loglik + 2 * self.nparams
        self.bic = -2 * loglik + np.log(self.nobs) * self.nparams

        # Storage for efficiency estimates
        self._efficiency_cache = {}

    def _compute_inference(
        self, hessian: Optional[np.ndarray], params: np.ndarray, param_names: list
    ) -> None:
        """Compute standard errors and inference statistics."""
        if hessian is not None:
            try:
                # Variance-covariance matrix = inverse of negative Hessian
                self.cov = np.linalg.inv(-hessian)
                se = np.sqrt(np.diag(self.cov))

                # Handle delta method for variance parameters
                # σ²_v and σ²_u are estimated as ln(σ²), need delta method
                se_adjusted = self._delta_method_variance(se, params, param_names)

                self.se = pd.Series(se_adjusted, index=param_names, name="std_err")
                self.tvalues = pd.Series(self.params / self.se, index=param_names, name="t_stat")
                pvalues = 2 * (1 - stats.norm.cdf(np.abs(self.tvalues)))
                self.pvalues = pd.Series(pvalues, index=param_names, name="p_value")

            except np.linalg.LinAlgError:
                print("Warning: Could not invert Hessian. " "Standard errors not available.")
                self.cov = None
                self.se = pd.Series(np.nan, index=param_names, name="std_err")
                self.tvalues = pd.Series(np.nan, index=param_names, name="t_stat")
                self.pvalues = pd.Series(np.nan, index=param_names, name="p_value")
        else:
            self.cov = None
            self.se = pd.Series(np.nan, index=param_names, name="std_err")
            self.tvalues = pd.Series(np.nan, index=param_names, name="t_stat")
            self.pvalues = pd.Series(np.nan, index=param_names, name="p_value")

    def _delta_method_variance(
        self, se: np.ndarray, params: np.ndarray, param_names: list
    ) -> np.ndarray:
        """Apply delta method for transformed parameters.

        Variance parameters are estimated as ln(σ²), so we need
        delta method to get SE for σ² and σ.

        For θ = ln(σ²): Var(σ²) = σ⁴ * Var(θ)
                        SE(σ²) = σ² * SE(θ)
        """
        se_adjusted = se.copy()

        # Find variance parameter indices
        for i, name in enumerate(param_names):
            if "ln_sigma" in name or "log_sigma" in name:
                # This is ln(σ²), delta method gives SE for σ²
                # SE(σ²) = exp(ln(σ²)) * SE(ln(σ²)) = σ² * SE(θ)
                sigma_sq = np.exp(params[i])
                se_adjusted[i] = sigma_sq * se[i]

        return se_adjusted

    def _extract_variance_components(self) -> None:
        """Extract and compute variance components."""
        # Find variance parameters in params
        param_names = self.params.index.tolist()

        # Look for sigma_v and sigma_u parameters
        sigma_v_sq = None
        sigma_u_sq = None

        for name in param_names:
            if "sigma_v" in name.lower():
                # This is σ²_v (already transformed from ln scale)
                sigma_v_sq = self.params[name]
            elif "sigma_u" in name.lower():
                # This is σ²_u (already transformed from ln scale)
                sigma_u_sq = self.params[name]

        if sigma_v_sq is not None and sigma_u_sq is not None:
            self.sigma_v_sq = sigma_v_sq
            self.sigma_u_sq = sigma_u_sq
            self.sigma_v = np.sqrt(sigma_v_sq)
            self.sigma_u = np.sqrt(sigma_u_sq)
            self.sigma_sq = sigma_v_sq + sigma_u_sq
            self.sigma = np.sqrt(self.sigma_sq)

            # Derived parameters
            self.lambda_param = self.sigma_u / self.sigma_v if self.sigma_v > 0 else np.inf
            self.gamma = self.sigma_u_sq / self.sigma_sq if self.sigma_sq > 0 else 0
        else:
            # Could not extract variance components
            self.sigma_v_sq = np.nan
            self.sigma_u_sq = np.nan
            self.sigma_v = np.nan
            self.sigma_u = np.nan
            self.sigma_sq = np.nan
            self.sigma = np.nan
            self.lambda_param = np.nan
            self.gamma = np.nan

    def summary(self, alpha: float = 0.05, title: Optional[str] = None) -> str:
        """Generate summary table of results.

        Parameters:
            alpha: Significance level for confidence intervals
            title: Optional title for summary table

        Returns:
            Formatted summary string
        """
        if title is None:
            title = f"Stochastic Frontier Analysis Results"

        # Critical value for CI
        z_crit = stats.norm.ppf(1 - alpha / 2)

        # Build summary components
        lines = []
        lines.append("=" * 78)
        lines.append(title.center(78))
        lines.append("=" * 78)

        # Model information
        lines.append(f"Model:                  {self.model.model_type.value}")
        lines.append(f"Frontier:               {self.model.frontier_type.value}")
        lines.append(f"Distribution:           {self.model.dist.value}")
        lines.append(f"No. Observations:       {self.nobs}")

        if self.model.is_panel:
            lines.append(f"No. Entities:           {self.model.n_entities}")
            lines.append(f"No. Time Periods:       {self.model.n_periods}")
            lines.append(f"Balanced Panel:         {self.model.is_balanced}")

        lines.append(f"Log-Likelihood:         {self.loglik:.4f}")
        lines.append(f"AIC:                    {self.aic:.4f}")
        lines.append(f"BIC:                    {self.bic:.4f}")
        lines.append(f"Converged:              {self.converged}")
        lines.append("-" * 78)

        # Variance components
        lines.append("Variance Components:")
        lines.append(f"  σ_v (noise):          {self.sigma_v:.6f}")
        lines.append(f"  σ_u (inefficiency):   {self.sigma_u:.6f}")
        lines.append(f"  σ (composite):        {self.sigma:.6f}")
        lines.append(f"  λ = σ_u/σ_v:          {self.lambda_param:.6f}")
        lines.append(f"  γ = σ²_u/σ²:          {self.gamma:.6f}")
        lines.append("-" * 78)

        # Parameter estimates table
        lines.append("Parameter Estimates:")
        lines.append(
            f"{'Variable':<20} {'Coef.':<12} {'Std.Err.':<12} "
            f"{'t':<10} {'P>|t|':<10} {'[' + f'{alpha/2:.3f}':<10} "
            f"{1-alpha/2:.3f}]"
        )
        lines.append("-" * 78)

        # Only show frontier parameters (exclude variance params from table)
        for var in self.params.index:
            if "sigma" not in var.lower() and "ln_" not in var.lower():
                coef = self.params[var]
                se = self.se[var]
                t = self.tvalues[var]
                p = self.pvalues[var]
                ci_lower = coef - z_crit * se
                ci_upper = coef + z_crit * se

                lines.append(
                    f"{var:<20} {coef:>11.6f} {se:>11.6f} "
                    f"{t:>9.4f} {p:>9.4f} {ci_lower:>9.4f} {ci_upper:>9.4f}"
                )

        lines.append("=" * 78)

        return "\n".join(lines)

    def efficiency(self, estimator: str = "bc", ci_level: float = 0.95) -> pd.DataFrame:
        """Estimate technical/cost efficiency.

        Parameters:
            estimator: Efficiency estimator type
                      'jlms' - Jondrow et al. (1982) E[u|ε]
                      'bc' - Battese & Coelli (1988) E[exp(-u)|ε]
                      'mode' - Modal estimator M[u|ε]
            ci_level: Confidence level for intervals (0-1)

        Returns:
            DataFrame with efficiency estimates and confidence intervals

        Raises:
            ValueError: If estimator not recognized
        """
        # Check cache
        cache_key = (estimator, ci_level)
        if cache_key in self._efficiency_cache:
            return self._efficiency_cache[cache_key]

        # Import efficiency module
        from .efficiency import estimate_efficiency

        # Compute efficiency
        eff_df = estimate_efficiency(result=self, estimator=estimator, ci_level=ci_level)

        # Cache result
        self._efficiency_cache[cache_key] = eff_df

        return eff_df

    @property
    def mean_efficiency(self) -> float:
        """Mean technical/cost efficiency (BC estimator)."""
        eff_df = self.efficiency(estimator="bc")
        return eff_df["efficiency"].mean()

    @property
    def residuals(self) -> np.ndarray:
        """Residuals from frontier equation (ε = y - X'β)."""
        # Extract beta parameters
        beta_names = [
            name
            for name in self.params.index
            if "sigma" not in name.lower() and "ln_" not in name.lower()
        ]
        beta = self.params[beta_names].values

        # Compute residuals
        epsilon = self.model.y - self.model.X @ beta

        return epsilon

    def compare_distributions(self, other_results: list) -> pd.DataFrame:
        """Compare this model with other distributional specifications.

        Parameters:
            other_results: List of other SFResult objects to compare

        Returns:
            DataFrame with comparison statistics
        """
        models = [self] + other_results

        comparison = {
            "Distribution": [m.model.dist.value for m in models],
            "Log-Likelihood": [m.loglik for m in models],
            "AIC": [m.aic for m in models],
            "BIC": [m.bic for m in models],
            "σ_v": [m.sigma_v for m in models],
            "σ_u": [m.sigma_u for m in models],
            "λ": [m.lambda_param for m in models],
            "Mean Efficiency": [m.mean_efficiency for m in models],
            "Converged": [m.converged for m in models],
        }

        df = pd.DataFrame(comparison)

        # Add indicators for best model
        df["Best AIC"] = df["AIC"] == df["AIC"].min()
        df["Best BIC"] = df["BIC"] == df["BIC"].min()

        return df

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SFResult(dist={self.model.dist.value}, "
            f"loglik={self.loglik:.2f}, converged={self.converged})"
        )


class PanelSFResult(SFResult):
    """Results from panel stochastic frontier estimation.

    Extends SFResult to handle panel-specific features like:
    - Time-varying efficiency (Battese-Coelli 1992)
    - Inefficiency determinants (Battese-Coelli 1995)
    - Temporal parameters (η, b, c, δ_t)

    Additional Attributes:
        panel_type: Type of panel model ('pitt_lee', 'bc92', 'bc95', etc.)
        temporal_params: Dictionary of temporal parameters (η, b, c, δ_t)
        has_time_varying: Whether inefficiency varies over time
        has_determinants: Whether inefficiency has determinants (BC95)
    """

    def __init__(
        self,
        params: np.ndarray,
        param_names: list,
        hessian: Optional[np.ndarray],
        loglik: float,
        converged: bool,
        model: Any,
        panel_type: str = "pitt_lee",
        temporal_params: Optional[Dict[str, float]] = None,
        optimization_result: Any = None,
    ):
        """Initialize PanelSFResult.

        Parameters:
            params: Parameter estimates
            param_names: Parameter names
            hessian: Hessian matrix
            loglik: Log-likelihood value
            converged: Convergence flag
            model: Panel stochastic frontier model object
            panel_type: Type of panel model
            temporal_params: Dictionary of temporal parameters
            optimization_result: Full scipy optimization result
        """
        super().__init__(
            params=params,
            param_names=param_names,
            hessian=hessian,
            loglik=loglik,
            converged=converged,
            model=model,
            optimization_result=optimization_result,
        )

        self.panel_type = panel_type
        self.temporal_params = temporal_params or {}

        # Determine model characteristics
        self.has_time_varying = panel_type in ["bc92", "kumbhakar", "lee_schmidt"]
        self.has_determinants = panel_type == "bc95"

    def efficiency(
        self, estimator: str = "bc", ci_level: float = 0.95, by_period: bool = False
    ) -> pd.DataFrame:
        """Estimate technical/cost efficiency for panel data.

        Parameters:
            estimator: Efficiency estimator type
                      'jlms' - Jondrow et al. (1982) E[u|ε]
                      'bc' - Battese & Coelli (1988) E[exp(-u)|ε]
                      'mode' - Modal estimator M[u|ε]
            ci_level: Confidence level for intervals (0-1)
            by_period: If True, return efficiency by (entity, period)
                      If False, return time-averaged efficiency by entity

        Returns:
            DataFrame with efficiency estimates

        Notes:
            - For Pitt-Lee: efficiency is constant over time (one per entity)
            - For BC92/Kumbhakar/Lee-Schmidt: efficiency varies by (entity, period)
            - For BC95: efficiency depends on determinants Z_it
        """
        # Check cache
        cache_key = (estimator, ci_level, by_period)
        if cache_key in self._efficiency_cache:
            return self._efficiency_cache[cache_key]

        # Import efficiency module
        from .efficiency import estimate_panel_efficiency

        # Compute efficiency
        eff_df = estimate_panel_efficiency(
            result=self, estimator=estimator, ci_level=ci_level, by_period=by_period
        )

        # Cache result
        self._efficiency_cache[cache_key] = eff_df

        return eff_df

    def summary(self, alpha: float = 0.05, title: Optional[str] = None) -> str:
        """Generate summary table of panel SFA results.

        Parameters:
            alpha: Significance level for confidence intervals
            title: Optional title for summary table

        Returns:
            Formatted summary string with panel-specific information
        """
        if title is None:
            title = f"Panel Stochastic Frontier Analysis Results ({self.panel_type.upper()})"

        # Get base summary
        base_summary = super().summary(alpha=alpha, title=title)

        # Add panel-specific information
        lines = base_summary.split("\n")

        # Insert temporal parameters after variance components
        if self.temporal_params:
            insert_idx = None
            for i, line in enumerate(lines):
                if "Variance Components:" in line:
                    # Find end of variance section
                    for j in range(i + 1, len(lines)):
                        if "-" * 78 in lines[j]:
                            insert_idx = j
                            break
                    break

            if insert_idx:
                temporal_lines = []
                temporal_lines.append("Temporal Parameters:")

                if "eta" in self.temporal_params:
                    eta = self.temporal_params["eta"]
                    temporal_lines.append(f"  η (decay parameter):  {eta:.6f}")
                    if eta > 0:
                        temporal_lines.append("    → Efficiency improves over time")
                    elif eta < 0:
                        temporal_lines.append("    → Efficiency worsens over time")
                    else:
                        temporal_lines.append("    → Efficiency constant over time")

                if "b" in self.temporal_params and "c" in self.temporal_params:
                    b = self.temporal_params["b"]
                    c = self.temporal_params["c"]
                    temporal_lines.append(f"  b (linear term):      {b:.6f}")
                    temporal_lines.append(f"  c (quadratic term):   {c:.6f}")
                    temporal_lines.append("    → Kumbhakar (1990) flexible time pattern")

                if "delta_t" in self.temporal_params:
                    delta_t = self.temporal_params["delta_t"]
                    temporal_lines.append("  δ_t (time loadings):  See detailed output")
                    temporal_lines.append(f"    → {len(delta_t)} time periods")

                temporal_lines.append("-" * 78)

                # Insert temporal section
                lines[insert_idx:insert_idx] = temporal_lines

        return "\n".join(lines)

    def test_temporal_constancy(self) -> Dict[str, Any]:
        """Test whether efficiency is constant over time.

        Returns:
            Dictionary with test results:
            - test_statistic: LR or Wald statistic
            - p_value: p-value for the test
            - df: degrees of freedom
            - conclusion: 'constant' or 'time_varying'

        Notes:
            - For BC92: Tests H0: η = 0
            - For Kumbhakar: Tests H0: b = c = 0
            - For Lee-Schmidt: Tests H0: δ_1 = ... = δ_{T-1} = 1
        """
        if not self.has_time_varying:
            return {
                "test_statistic": np.nan,
                "p_value": np.nan,
                "df": 0,
                "conclusion": "Model does not allow time variation",
            }

        # This requires fitting a restricted model (Pitt-Lee)
        # For now, return placeholder
        return {
            "test_statistic": np.nan,
            "p_value": np.nan,
            "df": np.nan,
            "conclusion": "Test not yet implemented",
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PanelSFResult(type={self.panel_type}, "
            f"loglik={self.loglik:.2f}, N={self.model.n_entities}, "
            f"T={self.model.n_periods}, converged={self.converged})"
        )
