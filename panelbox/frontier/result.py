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

    def summary(
        self, alpha: float = 0.05, title: Optional[str] = None, include_diagnostics: bool = True
    ) -> str:
        """Generate summary table of results.

        Parameters:
            alpha: Significance level for confidence intervals
            title: Optional title for summary table
            include_diagnostics: If True, include diagnostic tests in summary

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

        # Diagnostic test for presence of inefficiency
        if include_diagnostics and hasattr(self.model, "ols_loglik"):
            from .tests import inefficiency_presence_test

            ineff_test = inefficiency_presence_test(
                loglik_sfa=self.loglik,
                loglik_ols=self.model.ols_loglik,
                residuals_ols=self.model.ols_residuals,
                frontier_type=self.model.frontier_type.value,
                distribution=self.model.dist.value,
            )

            lines.append("Inefficiency Test:")
            lines.append(f"  LR statistic:         {ineff_test['lr_statistic']:.4f}")
            lines.append(f"  P-value (mixed χ²):   {ineff_test['pvalue']:.4f}")
            lines.append(f"  Conclusion:           {ineff_test['conclusion']}")
            lines.append(f"  Skewness:             {ineff_test['skewness']:.4f}")

            if ineff_test["skewness_warning"]:
                lines.append(f"  ⚠ {ineff_test['skewness_warning']}")

            lines.append("-" * 78)

        # Variance components
        lines.append("Variance Components:")
        lines.append(f"  σ_v (noise):          {self.sigma_v:.6f}")
        lines.append(f"  σ_u (inefficiency):   {self.sigma_u:.6f}")
        lines.append(f"  σ (composite):        {self.sigma:.6f}")
        lines.append(f"  λ = σ_u/σ_v:          {self.lambda_param:.6f}")
        lines.append(f"  γ = σ²_u/σ²:          {self.gamma:.6f}")
        lines.append("-" * 78)

        # Variance decomposition with confidence intervals
        if include_diagnostics:
            var_decomp = self.variance_decomposition(ci_level=1 - alpha, method="delta")
            lines.append("Variance Decomposition:")
            lines.append(f"  γ (inefficiency share):  {var_decomp['gamma']:.4f}")
            lines.append(
                f"    95% CI:                [{var_decomp['gamma_ci'][0]:.4f}, {var_decomp['gamma_ci'][1]:.4f}]"
            )
            lines.append(f"  λ (ratio σ_u/σ_v):       {var_decomp['lambda_param']:.4f}")
            lines.append(
                f"    95% CI:                [{var_decomp['lambda_ci'][0]:.4f}, {var_decomp['lambda_ci'][1]:.4f}]"
            )
            lines.append(f"  Interpretation: {var_decomp['interpretation']}")
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

    def compare_distributions(
        self, other_results: Optional[list] = None, distributions: Optional[list] = None
    ) -> pd.DataFrame:
        """Compare this model with other distributional specifications.

        Parameters:
            other_results: List of other SFResult objects to compare
                          If None and distributions is provided, will estimate models automatically
            distributions: List of distribution names to compare (e.g., ['half_normal', 'exponential', 'truncated_normal'])
                          Only used if other_results is None

        Returns:
            DataFrame with comparison statistics and recommendation

        Example:
            # Compare with pre-estimated models
            >>> result.compare_distributions(other_results=[result2, result3])

            # Auto-estimate and compare distributions
            >>> result.compare_distributions(distributions=['half_normal', 'exponential', 'truncated_normal'])
        """
        if other_results is None and distributions is not None:
            # Auto-estimate models with different distributions
            from .data import DistributionType

            other_results = []
            for dist_name in distributions:
                if dist_name == self.model.dist.value:
                    continue  # Skip current distribution

                try:
                    # Create new model with same data but different distribution
                    from .model import StochasticFrontier

                    dist_enum = DistributionType(dist_name)
                    new_model = StochasticFrontier(
                        data=self.model.data,
                        depvar=self.model.depvar,
                        exog=self.model.exog,
                        frontier=self.model.frontier_type,
                        dist=dist_enum,
                        is_panel=self.model.is_panel,
                    )
                    new_result = new_model.fit()
                    other_results.append(new_result)
                except Exception as e:
                    print(f"Warning: Failed to estimate {dist_name} distribution: {e}")
                    continue

        if other_results is None:
            other_results = []

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

        # Add rank columns
        df["Rank AIC"] = df["AIC"].rank().astype(int)
        df["Rank BIC"] = df["BIC"].rank().astype(int)

        # Sort by AIC
        df = df.sort_values("AIC").reset_index(drop=True)

        return df

    def variance_decomposition(
        self, ci_level: float = 0.95, method: str = "delta"
    ) -> Dict[str, Any]:
        """Decompose variance into noise and inefficiency components.

        Computes variance decomposition measures with confidence intervals:
            γ = σ²_u / (σ²_v + σ²_u) - proportion due to inefficiency
            λ = σ_u / σ_v - ratio of standard deviations
            σ² = σ²_v + σ²_u - total variance

        Parameters:
            ci_level: Confidence level for intervals (default 0.95)
            method: Method for computing CIs ('delta' or 'bootstrap')

        Returns:
            Dictionary with decomposition results:
                - gamma: Proportion of variance due to inefficiency
                - gamma_ci: Confidence interval for gamma
                - lambda_param: Ratio σ_u / σ_v
                - lambda_ci: Confidence interval for lambda
                - sigma_sq: Total variance
                - sigma_sq_u: Inefficiency variance
                - sigma_sq_v: Noise variance
                - interpretation: Interpretation of results

        Notes:
            γ → 0: Variation is primarily noise (OLS may be adequate)
            γ → 1: Variation is primarily inefficiency (deterministic frontier)
            γ ∈ [0.3, 0.7]: Both components are important
        """
        # Extract variance components
        sigma_v_sq = self.sigma_v_sq
        sigma_u_sq = self.sigma_u_sq
        sigma_sq = sigma_v_sq + sigma_u_sq

        # Compute gamma and lambda
        gamma = sigma_u_sq / sigma_sq if sigma_sq > 0 else 0
        lambda_param = np.sqrt(sigma_u_sq / sigma_v_sq) if sigma_v_sq > 0 else np.inf

        # Compute confidence intervals using delta method
        if method == "delta":
            # Find parameter indices for variance parameters
            param_names = self.params.index.tolist()
            idx_sigma_v = None
            idx_sigma_u = None

            for i, name in enumerate(param_names):
                if "sigma_v" in name.lower():
                    idx_sigma_v = i
                elif "sigma_u" in name.lower():
                    idx_sigma_u = i

            if idx_sigma_v is not None and idx_sigma_u is not None and self.cov is not None:
                # Delta method for gamma
                # γ = σ²_u / (σ²_v + σ²_u)
                # ∂γ/∂(σ²_v) = -σ²_u / (σ²_v + σ²_u)²
                # ∂γ/∂(σ²_u) = σ²_v / (σ²_v + σ²_u)²

                d_gamma_d_sigma_v_sq = -sigma_u_sq / (sigma_sq**2) if sigma_sq > 0 else 0
                d_gamma_d_sigma_u_sq = sigma_v_sq / (sigma_sq**2) if sigma_sq > 0 else 0

                # Gradient vector (only for variance parameters)
                grad_gamma = np.zeros(len(param_names))
                grad_gamma[idx_sigma_v] = d_gamma_d_sigma_v_sq
                grad_gamma[idx_sigma_u] = d_gamma_d_sigma_u_sq

                # Variance of gamma
                var_gamma = grad_gamma @ self.cov @ grad_gamma

                # Confidence interval for gamma
                z_crit = stats.norm.ppf((1 + ci_level) / 2)
                se_gamma = np.sqrt(var_gamma) if var_gamma > 0 else 0
                gamma_ci = (gamma - z_crit * se_gamma, gamma + z_crit * se_gamma)

                # Delta method for lambda
                # λ = σ_u / σ_v = √(σ²_u / σ²_v)
                # ∂λ/∂(σ²_v) = -0.5 * λ / σ²_v
                # ∂λ/∂(σ²_u) = 0.5 * λ / σ²_u

                if sigma_v_sq > 0 and sigma_u_sq > 0:
                    d_lambda_d_sigma_v_sq = -0.5 * lambda_param / sigma_v_sq
                    d_lambda_d_sigma_u_sq = 0.5 * lambda_param / sigma_u_sq

                    grad_lambda = np.zeros(len(param_names))
                    grad_lambda[idx_sigma_v] = d_lambda_d_sigma_v_sq
                    grad_lambda[idx_sigma_u] = d_lambda_d_sigma_u_sq

                    var_lambda = grad_lambda @ self.cov @ grad_lambda
                    se_lambda = np.sqrt(var_lambda) if var_lambda > 0 else 0
                    lambda_ci = (
                        lambda_param - z_crit * se_lambda,
                        lambda_param + z_crit * se_lambda,
                    )
                else:
                    lambda_ci = (np.nan, np.nan)

            else:
                # Could not compute CIs
                gamma_ci = (np.nan, np.nan)
                lambda_ci = (np.nan, np.nan)

        elif method == "bootstrap":
            # Bootstrap confidence intervals
            try:
                n_bootstrap = 1000
                gamma_boot = np.zeros(n_bootstrap)
                lambda_boot = np.zeros(n_bootstrap)

                # Get original parameter estimates
                theta_hat = self.params.values

                # Bootstrap resampling
                for b in range(n_bootstrap):
                    # Resample parameters from multivariate normal
                    if self.cov is not None:
                        theta_b = np.random.multivariate_normal(theta_hat, self.cov)

                        # Extract variance parameters
                        sigma_v_sq_b = (
                            theta_b[idx_sigma_v] if idx_sigma_v is not None else sigma_v_sq
                        )
                        sigma_u_sq_b = (
                            theta_b[idx_sigma_u] if idx_sigma_u is not None else sigma_u_sq
                        )

                        # Ensure non-negative
                        sigma_v_sq_b = max(sigma_v_sq_b, 1e-8)
                        sigma_u_sq_b = max(sigma_u_sq_b, 1e-8)

                        sigma_sq_b = sigma_v_sq_b + sigma_u_sq_b
                        gamma_boot[b] = sigma_u_sq_b / sigma_sq_b if sigma_sq_b > 0 else 0
                        lambda_boot[b] = (
                            np.sqrt(sigma_u_sq_b / sigma_v_sq_b) if sigma_v_sq_b > 0 else np.inf
                        )
                    else:
                        # Cannot bootstrap without covariance
                        gamma_boot[b] = gamma
                        lambda_boot[b] = lambda_param

                # Compute percentile confidence intervals
                alpha = 1 - ci_level
                gamma_ci = (
                    np.percentile(gamma_boot, 100 * alpha / 2),
                    np.percentile(gamma_boot, 100 * (1 - alpha / 2)),
                )
                lambda_ci = (
                    np.percentile(lambda_boot, 100 * alpha / 2),
                    np.percentile(lambda_boot, 100 * (1 - alpha / 2)),
                )

            except Exception as e:
                print(f"Warning: Bootstrap failed ({e}). Using delta method instead.")
                # Fallback to delta method
                return self.variance_decomposition(ci_level=ci_level, method="delta")

        else:
            raise ValueError(f"Unknown method: {method}. Use 'delta' or 'bootstrap'.")

        # Interpretation
        if gamma < 0.1:
            interpretation = (
                f"γ = {gamma:.4f} indicates that inefficiency accounts for only {100*gamma:.1f}% "
                "of total variance. OLS regression may be adequate."
            )
        elif gamma > 0.9:
            interpretation = (
                f"γ = {gamma:.4f} indicates that inefficiency accounts for {100*gamma:.1f}% "
                "of total variance. Frontier is nearly deterministic. "
                "Consider checking model specification."
            )
        else:
            interpretation = (
                f"γ = {gamma:.4f} indicates that inefficiency accounts for {100*gamma:.1f}% "
                "of total variance, while noise accounts for {100*(1-gamma):.1f}%. "
                "Both components are important."
            )

        return {
            "gamma": gamma,
            "gamma_ci": gamma_ci,
            "lambda_param": lambda_param,
            "lambda_ci": lambda_ci,
            "sigma_sq": sigma_sq,
            "sigma_sq_u": sigma_u_sq,
            "sigma_sq_v": sigma_v_sq,
            "interpretation": interpretation,
            "ci_level": ci_level,
            "method": method,
        }

    def returns_to_scale_test(
        self, input_vars: Optional[list] = None, alpha: float = 0.05
    ) -> Dict[str, Any]:
        """Test for constant returns to scale (CRS).

        For Cobb-Douglas production function in logs:
            ln(y) = β₀ + β₁·ln(x₁) + β₂·ln(x₂) + ... + v - u

        RTS = Σⱼ βⱼ (sum of input elasticities)

        H0: RTS = 1 (constant returns to scale)
        H1: RTS ≠ 1

        Parameters:
            input_vars: List of input variable names (exclude intercept and time)
                       If None, uses all frontier parameters except intercept
            alpha: Significance level for test

        Returns:
            Dictionary with test results:
                - rts: Returns to scale estimate
                - rts_se: Standard error of RTS
                - test_statistic: Wald test statistic
                - pvalue: P-value for H0: RTS = 1
                - conclusion: 'CRS', 'IRS', or 'DRS'
                - ci: Confidence interval for RTS

        Notes:
            - RTS > 1: Increasing returns to scale (IRS)
            - RTS = 1: Constant returns to scale (CRS)
            - RTS < 1: Decreasing returns to scale (DRS)
        """
        # Extract frontier parameters (exclude variance parameters and intercept)
        param_names = self.params.index.tolist()

        if input_vars is None:
            # Use all parameters except intercept and variance parameters
            input_vars = [
                name
                for name in param_names
                if "sigma" not in name.lower()
                and "ln_" not in name.lower()
                and "const" not in name.lower()
                and "intercept" not in name.lower()
            ]

        # Check that input_vars are in params
        missing_vars = set(input_vars) - set(param_names)
        if missing_vars:
            raise ValueError(f"Variables not found in parameters: {missing_vars}")

        # Compute RTS = sum of input elasticities
        beta_inputs = self.params[input_vars].values
        rts = np.sum(beta_inputs)

        # Compute standard error using delta method
        # RTS = 1'β where 1 is vector of ones
        # Var(RTS) = 1' Var(β) 1
        if self.cov is not None:
            input_indices = [param_names.index(var) for var in input_vars]
            vcov_inputs = self.cov[np.ix_(input_indices, input_indices)]

            var_rts = np.sum(vcov_inputs)  # Sum of all elements of covariance matrix
            se_rts = np.sqrt(var_rts) if var_rts > 0 else 0

            # Wald test for H0: RTS = 1
            if se_rts > 0:
                test_stat = (rts - 1) ** 2 / var_rts
                pvalue = 1 - stats.chi2.cdf(test_stat, df=1)
            else:
                test_stat = np.nan
                pvalue = np.nan

            # Confidence interval
            z_crit = stats.norm.ppf(1 - alpha / 2)
            ci = (rts - z_crit * se_rts, rts + z_crit * se_rts)

        else:
            se_rts = np.nan
            test_stat = np.nan
            pvalue = np.nan
            ci = (np.nan, np.nan)

        # Conclusion
        if not np.isnan(pvalue):
            if pvalue < alpha:
                # Reject H0: RTS ≠ 1
                if rts > 1:
                    conclusion = "IRS"
                    interpretation = (
                        f"Reject H0 (p = {pvalue:.4f}). "
                        f"RTS = {rts:.4f} > 1 indicates Increasing Returns to Scale."
                    )
                else:
                    conclusion = "DRS"
                    interpretation = (
                        f"Reject H0 (p = {pvalue:.4f}). "
                        f"RTS = {rts:.4f} < 1 indicates Decreasing Returns to Scale."
                    )
            else:
                conclusion = "CRS"
                interpretation = (
                    f"Do not reject H0 (p = {pvalue:.4f}). "
                    f"RTS = {rts:.4f} ≈ 1 indicates Constant Returns to Scale."
                )
        else:
            conclusion = "unknown"
            interpretation = "Could not perform test (covariance matrix not available)."

        return {
            "rts": rts,
            "rts_se": se_rts,
            "test_statistic": test_stat,
            "pvalue": pvalue,
            "conclusion": conclusion,
            "interpretation": interpretation,
            "ci": ci,
            "alpha": alpha,
            "input_vars": input_vars,
        }

    def elasticities(
        self,
        input_vars: Optional[list] = None,
        translog: bool = False,
        translog_vars: Optional[list] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calculate production elasticities.

        For Cobb-Douglas (log-linear): elasticities are constant (= β_j)
        For Translog: elasticities vary by observation

        Parameters:
            input_vars: List of input variable names
                       If None, uses all frontier parameters except intercept
            translog: Whether this is a Translog specification
            translog_vars: For Translog, list of base variable names (e.g., ['ln_K', 'ln_L'])
                          Squared and interaction terms will be identified automatically

        Returns:
            For Cobb-Douglas: Series with constant elasticities
            For Translog: DataFrame with elasticities for each observation

        Example (Cobb-Douglas):
            >>> result.elasticities(input_vars=['ln_K', 'ln_L'])
            # Returns: Series([ε_K, ε_L])

        Example (Translog):
            >>> result.elasticities(translog=True, translog_vars=['ln_K', 'ln_L'])
            # Returns: DataFrame with columns ['ε_K', 'ε_L'] and rows for each obs

        Notes:
            For Translog: ε_j = β_j + Σ_k β_{jk} × ln(x_k)
            Elasticities depend on input levels (vary across observations)
        """
        param_names = self.params.index.tolist()

        if not translog:
            # Cobb-Douglas: elasticities = coefficients (constant)
            if input_vars is None:
                # Use all parameters except intercept and variance parameters
                input_vars = [
                    name
                    for name in param_names
                    if "sigma" not in name.lower()
                    and "ln_" not in name.lower()
                    and "const" not in name.lower()
                    and "intercept" not in name.lower()
                ]

            # Return constant elasticities
            return self.params[input_vars].copy()

        else:
            # Translog: elasticities vary by observation
            if translog_vars is None:
                raise ValueError("For Translog, must specify translog_vars (base variable names)")

            # Check that variables exist in data
            missing_vars = set(translog_vars) - set(self.model.X_df.columns)
            if missing_vars:
                raise ValueError(f"Variables not found in data: {missing_vars}")

            n_obs = len(self.model.X_df)
            elasticities = np.zeros((n_obs, len(translog_vars)))

            # For each input variable, calculate elasticity
            for j, var_j in enumerate(translog_vars):
                # Base coefficient β_j
                if var_j in param_names:
                    beta_j = self.params[var_j]
                else:
                    beta_j = 0

                # Start with linear term
                elas_j = np.full(n_obs, beta_j)

                # Add squared term: β_{jj} × ln(x_j)
                sq_term = f"{var_j}_sq"
                if sq_term in param_names:
                    beta_jj = self.params[sq_term]
                    elas_j += beta_jj * self.model.X_df[var_j].values

                # Add interaction terms: β_{jk} × ln(x_k) for all k
                for k, var_k in enumerate(translog_vars):
                    if j == k:
                        continue  # Already handled squared term

                    # Look for interaction term (could be var_j_var_k or var_k_var_j)
                    interaction_name = f"{var_j}_{var_k}" if j < k else f"{var_k}_{var_j}"

                    if interaction_name in param_names:
                        beta_jk = self.params[interaction_name]
                        elas_j += beta_jk * self.model.X_df[var_k].values

                elasticities[:, j] = elas_j

            # Create DataFrame with elasticities
            elas_df = pd.DataFrame(
                elasticities,
                columns=[f"ε_{var.replace('ln_', '')}" for var in translog_vars],
                index=self.model.X_df.index,
            )

            return elas_df

    def efficient_scale(
        self,
        translog_vars: Optional[list] = None,
        initial_scale: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Calculate efficient scale where RTS = 1.

        For Translog production functions, finds the input levels where
        returns to scale equal 1 (constant returns).

        Parameters:
            translog_vars: List of base variable names for Translog
                          (e.g., ['ln_K', 'ln_L'])
            initial_scale: Initial guess for optimization (optional)
                          If None, uses mean input levels

        Returns:
            Dictionary with:
                - efficient_scale: Input levels where RTS = 1
                - rts_at_efficient: RTS at efficient scale (should be ≈ 1)
                - elasticities: Elasticities at efficient scale
                - converged: Whether optimization converged

        Notes:
            - Only applicable for Translog specifications
            - For Cobb-Douglas, RTS is constant, so no unique efficient scale
            - Uses numerical optimization to find scale where Σ ε_j = 1

        Example:
            >>> # Estimate Translog model
            >>> data_tl = add_translog(data, ['ln_K', 'ln_L'])
            >>> result = fit_sfa(data_tl, ...)
            >>> eff_scale = result.efficient_scale(translog_vars=['ln_K', 'ln_L'])
            >>> print(eff_scale['efficient_scale'])
        """
        if translog_vars is None:
            raise ValueError("Must specify translog_vars for efficient scale calculation")

        from scipy.optimize import minimize

        # Get parameter names
        param_names = self.params.index.tolist()

        # Define objective: minimize (RTS - 1)²
        def objective(ln_x):
            """Objective function: squared deviation from RTS = 1."""
            # Calculate RTS at this input level
            rts = 0
            for j, var_j in enumerate(translog_vars):
                # Base coefficient
                if var_j in param_names:
                    beta_j = self.params[var_j]
                else:
                    beta_j = 0

                # Elasticity for input j
                elas_j = beta_j

                # Add squared term
                sq_term = f"{var_j}_sq"
                if sq_term in param_names:
                    beta_jj = self.params[sq_term]
                    elas_j += beta_jj * ln_x[j]

                # Add interaction terms
                for k, var_k in enumerate(translog_vars):
                    if j == k:
                        continue
                    interaction_name = f"{var_j}_{var_k}" if j < k else f"{var_k}_{var_j}"
                    if interaction_name in param_names:
                        beta_jk = self.params[interaction_name]
                        elas_j += beta_jk * ln_x[k]

                rts += elas_j

            return (rts - 1) ** 2

        # Initial guess
        if initial_scale is None:
            # Use mean of data
            initial_scale = np.array([self.model.X_df[var].mean() for var in translog_vars])

        # Optimize
        result = minimize(objective, initial_scale, method="BFGS")

        # Extract results
        efficient_ln_x = result.x
        converged = result.success

        # Calculate RTS and elasticities at efficient scale
        rts_at_efficient = 0
        elasticities_at_efficient = {}

        for j, var_j in enumerate(translog_vars):
            # Base coefficient
            if var_j in param_names:
                beta_j = self.params[var_j]
            else:
                beta_j = 0

            # Elasticity for input j
            elas_j = beta_j

            # Add squared term
            sq_term = f"{var_j}_sq"
            if sq_term in param_names:
                beta_jj = self.params[sq_term]
                elas_j += beta_jj * efficient_ln_x[j]

            # Add interaction terms
            for k, var_k in enumerate(translog_vars):
                if j == k:
                    continue
                interaction_name = f"{var_j}_{var_k}" if j < k else f"{var_k}_{var_j}"
                if interaction_name in param_names:
                    beta_jk = self.params[interaction_name]
                    elas_j += beta_jk * efficient_ln_x[k]

            elasticities_at_efficient[var_j] = elas_j
            rts_at_efficient += elas_j

        return {
            "efficient_scale": efficient_ln_x,
            "rts_at_efficient": rts_at_efficient,
            "elasticities": elasticities_at_efficient,
            "converged": converged,
            "objective_value": result.fun,
        }

    def compare_functional_form(
        self,
        translog_result: "SFResult",
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """Compare Cobb-Douglas vs Translog functional forms.

        Performs likelihood ratio test to determine if the additional
        flexibility of Translog specification is statistically justified.

        H0: Cobb-Douglas is adequate (all quadratic and interaction terms = 0)
        H1: Translog provides better fit

        Parameters:
            translog_result: SFResult from Translog specification
            alpha: Significance level for test

        Returns:
            Dictionary with test results:
                - lr_statistic: LR test statistic
                - df: Degrees of freedom (number of additional Translog terms)
                - pvalue: P-value for LR test
                - aic_cd: AIC for Cobb-Douglas
                - aic_tl: AIC for Translog
                - bic_cd: BIC for Cobb-Douglas
                - bic_tl: BIC for Translog
                - conclusion: 'cobb_douglas' or 'translog'
                - interpretation: Detailed interpretation

        Example:
            >>> # Estimate both models
            >>> cd_result = fit_sfa(data, depvar='ln_y', exog=['ln_K', 'ln_L'])
            >>> data_tl = add_translog(data, ['ln_K', 'ln_L'])
            >>> tl_result = fit_sfa(data_tl, depvar='ln_y',
            ...                     exog=['ln_K', 'ln_L', 'ln_K_sq', ...])
            >>> # Compare
            >>> cd_result.compare_functional_form(tl_result)

        Notes:
            - This test assumes Cobb-Douglas is nested in Translog
            - Self should be the restricted (Cobb-Douglas) model
            - translog_result should be the unrestricted (Translog) model
        """
        from .tests import lr_test

        # This model is Cobb-Douglas (restricted), other is Translog (unrestricted)
        cd_loglik = self.loglik
        tl_loglik = translog_result.loglik

        # Degrees of freedom = difference in number of parameters
        df = translog_result.nparams - self.nparams

        if df <= 0:
            raise ValueError(
                "Translog model should have more parameters than Cobb-Douglas. "
                f"Got: CD={self.nparams}, TL={translog_result.nparams}"
            )

        # Perform LR test
        lr_result = lr_test(
            loglik_restricted=cd_loglik,
            loglik_unrestricted=tl_loglik,
            df_diff=df,
        )

        # Compare information criteria
        aic_cd = self.aic
        aic_tl = translog_result.aic
        bic_cd = self.bic
        bic_tl = translog_result.bic

        # Decision based on LR test
        if lr_result["pvalue"] < alpha:
            conclusion = "translog"
            interpretation = (
                f"Reject H0 at {100*alpha}% level (p = {lr_result['pvalue']:.4f}). "
                f"Translog specification provides significantly better fit than Cobb-Douglas. "
                f"The additional flexibility (df={df}) is statistically justified."
            )
        else:
            conclusion = "cobb_douglas"
            interpretation = (
                f"Do not reject H0 (p = {lr_result['pvalue']:.4f}). "
                f"Cobb-Douglas specification is adequate. "
                f"The additional {df} parameters in Translog are not statistically justified. "
                f"Prefer simpler Cobb-Douglas model."
            )

        # Add information criteria comparison
        if aic_tl < aic_cd and bic_tl < bic_cd:
            ic_comment = "\nBoth AIC and BIC favor Translog."
        elif aic_cd < aic_tl and bic_cd < bic_tl:
            ic_comment = "\nBoth AIC and BIC favor Cobb-Douglas."
        else:
            ic_comment = (
                f"\nAIC favors {'Translog' if aic_tl < aic_cd else 'Cobb-Douglas'}, "
                f"BIC favors {'Translog' if bic_tl < bic_cd else 'Cobb-Douglas'}."
            )

        interpretation += ic_comment

        return {
            "lr_statistic": lr_result["statistic"],
            "df": df,
            "pvalue": lr_result["pvalue"],
            "aic_cd": aic_cd,
            "aic_tl": aic_tl,
            "bic_cd": bic_cd,
            "bic_tl": bic_tl,
            "delta_aic": aic_tl - aic_cd,
            "delta_bic": bic_tl - bic_cd,
            "conclusion": conclusion,
            "interpretation": interpretation,
            "alpha": alpha,
        }

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

    def variance_decomposition(
        self, ci_level: float = 0.95, method: str = "delta"
    ) -> Dict[str, Any]:
        """Decompose variance into noise, inefficiency, and heterogeneity components.

        For True RE models, decomposes total variance into three components:
            γ_v = σ²_v / (σ²_v + σ²_u + σ²_w) - proportion due to noise
            γ_u = σ²_u / (σ²_v + σ²_u + σ²_w) - proportion due to inefficiency
            γ_w = σ²_w / (σ²_v + σ²_u + σ²_w) - proportion due to heterogeneity

        Parameters:
            ci_level: Confidence level for intervals (default 0.95)
            method: Method for computing CIs ('delta' or 'bootstrap')

        Returns:
            Dictionary with decomposition results:
                - gamma_v: Proportion of variance due to noise
                - gamma_u: Proportion of variance due to inefficiency
                - gamma_w: Proportion of variance due to heterogeneity (if TRE)
                - gamma_ci_v, gamma_ci_u, gamma_ci_w: Confidence intervals
                - lambda_param: Ratio σ_u / σ_v
                - sigma_sq: Total variance
                - interpretation: Interpretation of results

        Notes:
            For True RE: γ_v + γ_u + γ_w = 1
            For other panel models: falls back to base class method
        """
        # Check if this is a TRE model with sigma_w
        param_names = self.params.index.tolist()
        has_sigma_w = any("sigma_w" in name.lower() for name in param_names)

        if not has_sigma_w:
            # Fall back to base class (two-component decomposition)
            return super().variance_decomposition(ci_level=ci_level, method=method)

        # Extract variance components for TRE model
        sigma_v_sq = self.sigma_v_sq
        sigma_u_sq = self.sigma_u_sq

        # Find sigma_w parameter
        sigma_w_sq = None
        for name in param_names:
            if "sigma_w" in name.lower():
                sigma_w_sq = self.params[name]
                break

        if sigma_w_sq is None:
            # Fallback
            return super().variance_decomposition(ci_level=ci_level, method=method)

        # Total variance
        sigma_sq = sigma_v_sq + sigma_u_sq + sigma_w_sq

        # Compute gamma components
        gamma_v = sigma_v_sq / sigma_sq if sigma_sq > 0 else 0
        gamma_u = sigma_u_sq / sigma_sq if sigma_sq > 0 else 0
        gamma_w = sigma_w_sq / sigma_sq if sigma_sq > 0 else 0

        # Also compute two-component gammas for compatibility
        gamma_2comp = sigma_u_sq / (sigma_v_sq + sigma_u_sq) if (sigma_v_sq + sigma_u_sq) > 0 else 0
        lambda_param = np.sqrt(sigma_u_sq / sigma_v_sq) if sigma_v_sq > 0 else np.inf

        # Compute confidence intervals using delta method
        if method == "delta" and self.cov is not None:
            # Find parameter indices
            idx_sigma_v = None
            idx_sigma_u = None
            idx_sigma_w = None

            for i, name in enumerate(param_names):
                if "sigma_v" in name.lower():
                    idx_sigma_v = i
                elif "sigma_u" in name.lower():
                    idx_sigma_u = i
                elif "sigma_w" in name.lower():
                    idx_sigma_w = i

            if idx_sigma_v is not None and idx_sigma_u is not None and idx_sigma_w is not None:
                # Delta method for gamma_v, gamma_u, gamma_w
                # γ_v = σ²_v / σ²_total
                # ∂γ_v/∂(σ²_v) = (σ²_total - σ²_v) / σ²_total²
                # ∂γ_v/∂(σ²_u) = -σ²_v / σ²_total²
                # ∂γ_v/∂(σ²_w) = -σ²_v / σ²_total²

                # For gamma_v
                d_gamma_v_d_sigma_v_sq = (
                    (sigma_sq - sigma_v_sq) / (sigma_sq**2) if sigma_sq > 0 else 0
                )
                d_gamma_v_d_sigma_u_sq = -sigma_v_sq / (sigma_sq**2) if sigma_sq > 0 else 0
                d_gamma_v_d_sigma_w_sq = -sigma_v_sq / (sigma_sq**2) if sigma_sq > 0 else 0

                grad_gamma_v = np.zeros(len(param_names))
                grad_gamma_v[idx_sigma_v] = d_gamma_v_d_sigma_v_sq
                grad_gamma_v[idx_sigma_u] = d_gamma_v_d_sigma_u_sq
                grad_gamma_v[idx_sigma_w] = d_gamma_v_d_sigma_w_sq

                var_gamma_v = grad_gamma_v @ self.cov @ grad_gamma_v
                z_crit = stats.norm.ppf((1 + ci_level) / 2)
                se_gamma_v = np.sqrt(var_gamma_v) if var_gamma_v > 0 else 0
                gamma_ci_v = (gamma_v - z_crit * se_gamma_v, gamma_v + z_crit * se_gamma_v)

                # For gamma_u
                d_gamma_u_d_sigma_v_sq = -sigma_u_sq / (sigma_sq**2) if sigma_sq > 0 else 0
                d_gamma_u_d_sigma_u_sq = (
                    (sigma_sq - sigma_u_sq) / (sigma_sq**2) if sigma_sq > 0 else 0
                )
                d_gamma_u_d_sigma_w_sq = -sigma_u_sq / (sigma_sq**2) if sigma_sq > 0 else 0

                grad_gamma_u = np.zeros(len(param_names))
                grad_gamma_u[idx_sigma_v] = d_gamma_u_d_sigma_v_sq
                grad_gamma_u[idx_sigma_u] = d_gamma_u_d_sigma_u_sq
                grad_gamma_u[idx_sigma_w] = d_gamma_u_d_sigma_w_sq

                var_gamma_u = grad_gamma_u @ self.cov @ grad_gamma_u
                se_gamma_u = np.sqrt(var_gamma_u) if var_gamma_u > 0 else 0
                gamma_ci_u = (gamma_u - z_crit * se_gamma_u, gamma_u + z_crit * se_gamma_u)

                # For gamma_w
                d_gamma_w_d_sigma_v_sq = -sigma_w_sq / (sigma_sq**2) if sigma_sq > 0 else 0
                d_gamma_w_d_sigma_u_sq = -sigma_w_sq / (sigma_sq**2) if sigma_sq > 0 else 0
                d_gamma_w_d_sigma_w_sq = (
                    (sigma_sq - sigma_w_sq) / (sigma_sq**2) if sigma_sq > 0 else 0
                )

                grad_gamma_w = np.zeros(len(param_names))
                grad_gamma_w[idx_sigma_v] = d_gamma_w_d_sigma_v_sq
                grad_gamma_w[idx_sigma_u] = d_gamma_w_d_sigma_u_sq
                grad_gamma_w[idx_sigma_w] = d_gamma_w_d_sigma_w_sq

                var_gamma_w = grad_gamma_w @ self.cov @ grad_gamma_w
                se_gamma_w = np.sqrt(var_gamma_w) if var_gamma_w > 0 else 0
                gamma_ci_w = (gamma_w - z_crit * se_gamma_w, gamma_w + z_crit * se_gamma_w)

                # Lambda CI (same as base class)
                if sigma_v_sq > 0 and sigma_u_sq > 0:
                    d_lambda_d_sigma_v_sq = -0.5 * lambda_param / sigma_v_sq
                    d_lambda_d_sigma_u_sq = 0.5 * lambda_param / sigma_u_sq

                    grad_lambda = np.zeros(len(param_names))
                    grad_lambda[idx_sigma_v] = d_lambda_d_sigma_v_sq
                    grad_lambda[idx_sigma_u] = d_lambda_d_sigma_u_sq

                    var_lambda = grad_lambda @ self.cov @ grad_lambda
                    se_lambda = np.sqrt(var_lambda) if var_lambda > 0 else 0
                    lambda_ci = (
                        lambda_param - z_crit * se_lambda,
                        lambda_param + z_crit * se_lambda,
                    )
                else:
                    lambda_ci = (np.nan, np.nan)

            else:
                # Could not compute CIs
                gamma_ci_v = (np.nan, np.nan)
                gamma_ci_u = (np.nan, np.nan)
                gamma_ci_w = (np.nan, np.nan)
                lambda_ci = (np.nan, np.nan)

        elif method == "bootstrap":
            raise NotImplementedError(
                "Bootstrap confidence intervals not yet implemented. Use method='delta'."
            )
        else:
            gamma_ci_v = (np.nan, np.nan)
            gamma_ci_u = (np.nan, np.nan)
            gamma_ci_w = (np.nan, np.nan)
            lambda_ci = (np.nan, np.nan)

        # Interpretation
        interpretation = (
            f"Three-component variance decomposition (True RE):\n"
            f"  - Noise (v): {100*gamma_v:.1f}% of total variance\n"
            f"  - Inefficiency (u): {100*gamma_u:.1f}% of total variance\n"
            f"  - Heterogeneity (w): {100*gamma_w:.1f}% of total variance\n"
        )

        if gamma_u < 0.1:
            interpretation += (
                f"Inefficiency accounts for only {100*gamma_u:.1f}% of variance. "
                "Consider whether SFA is necessary."
            )
        elif gamma_w > 0.5:
            interpretation += (
                f"Heterogeneity is dominant ({100*gamma_w:.1f}%). "
                "Entity-specific effects are important."
            )

        return {
            "gamma_v": gamma_v,
            "gamma_u": gamma_u,
            "gamma_w": gamma_w,
            "gamma": gamma_2comp,  # For compatibility
            "gamma_ci_v": gamma_ci_v,
            "gamma_ci_u": gamma_ci_u,
            "gamma_ci_w": gamma_ci_w,
            "lambda_param": lambda_param,
            "lambda_ci": lambda_ci,
            "sigma_sq": sigma_sq,
            "sigma_sq_v": sigma_v_sq,
            "sigma_sq_u": sigma_u_sq,
            "sigma_sq_w": sigma_w_sq,
            "interpretation": interpretation,
            "ci_level": ci_level,
            "method": method,
            "is_three_component": True,
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PanelSFResult(type={self.panel_type}, "
            f"loglik={self.loglik:.2f}, N={self.model.n_entities}, "
            f"T={self.model.n_periods}, converged={self.converged})"
        )
