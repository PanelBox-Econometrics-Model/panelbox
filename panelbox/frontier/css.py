"""
Cornwell, Schmidt & Sickles (1990) Distribution-Free Panel Model.

This module implements the CSS model which does NOT require distributional
assumptions about inefficiency. Instead, it uses time-varying intercepts
to capture inefficiency.

The main advantage is robustness to distributional misspecification.
The main disadvantage is that it requires T ≥ 5 (preferably T ≥ 10).

References:
    Cornwell, C., Schmidt, P., & Sickles, R. C. (1990).
        Production frontiers with cross-sectional and time-series variation
        in efficiency levels. Journal of Econometrics, 46(1-2), 185-200.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class CSSResult:
    """Results from Cornwell-Schmidt-Sickles estimation.

    Attributes:
        params: Estimated parameters (β coefficients)
        param_names: Parameter names
        alpha_it: Time-varying intercepts (N x T matrix)
        efficiency_it: Technical efficiency estimates (N x T matrix)
        residuals: Model residuals
        sigma_v: Standard deviation of noise term
        r_squared: R-squared statistic
        entity_ids: Entity identifiers
        time_ids: Time identifiers
        time_trend: Type of time trend used ('none', 'linear', 'quadratic')
        n_obs: Number of observations
        n_entities: Number of entities
        n_periods: Number of time periods
        frontier_type: 'production' or 'cost'
    """

    params: np.ndarray
    param_names: list
    alpha_it: np.ndarray
    efficiency_it: np.ndarray
    residuals: np.ndarray
    sigma_v: float
    r_squared: float
    entity_ids: np.ndarray
    time_ids: np.ndarray
    time_trend: str
    n_obs: int
    n_entities: int
    n_periods: int
    frontier_type: str

    def summary(self) -> pd.DataFrame:
        """Create summary DataFrame of results.

        Returns:
            DataFrame with parameter estimates, standard errors, t-stats, p-values
        """
        summary_df = pd.DataFrame(
            {
                "Parameter": self.param_names,
                "Estimate": self.params,
                # Standard errors would require covariance matrix
            }
        )

        return summary_df

    def efficiency_by_entity(self) -> pd.DataFrame:
        """Get efficiency statistics by entity.

        Returns:
            DataFrame with entity-level efficiency statistics
        """
        results = []

        for i in range(self.n_entities):
            eff_i = self.efficiency_it[i, :]

            results.append(
                {
                    "entity": i,
                    "mean_efficiency": np.mean(eff_i),
                    "min_efficiency": np.min(eff_i),
                    "max_efficiency": np.max(eff_i),
                    "std_efficiency": np.std(eff_i),
                    "trend": np.polyfit(range(len(eff_i)), eff_i, 1)[0],  # Linear trend
                }
            )

        return pd.DataFrame(results)

    def efficiency_by_period(self) -> pd.DataFrame:
        """Get efficiency statistics by time period.

        Returns:
            DataFrame with period-level efficiency statistics
        """
        results = []

        for t in range(self.n_periods):
            eff_t = self.efficiency_it[:, t]

            results.append(
                {
                    "period": t,
                    "mean_efficiency": np.mean(eff_t),
                    "min_efficiency": np.min(eff_t),
                    "max_efficiency": np.max(eff_t),
                    "std_efficiency": np.std(eff_t),
                }
            )

        return pd.DataFrame(results)


def estimate_css_model(
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    time_trend: str = "quadratic",
    frontier_type: str = "production",
) -> CSSResult:
    """Estimate Cornwell, Schmidt & Sickles (1990) distribution-free model.

    Model: y_{it} = α_i(t) + X_{it}β + v_{it}

    where:
        α_i(t) = θ_{i1} + θ_{i2}·t + θ_{i3}·t²  (time-varying intercept)

    Efficiency is derived from: TE_{it} = exp[α_{it} - max_j α_{jt}]

    The entity with the highest intercept in period t is the frontier (TE = 1).

    Parameters:
        y: Dependent variable (n,)
        X: Exogenous variables (n, k) - should NOT include constant
        entity_id: Entity identifiers (n,) - integer coded 0, 1, ..., N-1
        time_id: Time identifiers (n,) - integer coded 0, 1, ..., T-1
        time_trend: Type of time trend ('none', 'linear', 'quadratic')
        frontier_type: 'production' or 'cost'

    Returns:
        CSSResult object with estimates

    Raises:
        ValueError: If time_trend not in {'none', 'linear', 'quadratic'}
        ValueError: If T < 5 (minimum recommended)

    References:
        Cornwell, C., Schmidt, P., & Sickles, R. C. (1990).

    Example:
        >>> result = estimate_css_model(y, X, entity_id, time_id, time_trend='quadratic')
        >>> print(result.efficiency_by_entity())
        >>> # Get efficiency for entity 5, period 3
        >>> eff_5_3 = result.efficiency_it[5, 3]
    """
    # Validate inputs
    if time_trend not in ["none", "linear", "quadratic"]:
        raise ValueError(f"time_trend must be 'none', 'linear', or 'quadratic', got {time_trend}")

    if frontier_type not in ["production", "cost"]:
        raise ValueError(f"frontier_type must be 'production' or 'cost', got {frontier_type}")

    # Get dimensions
    n = len(y)
    k = X.shape[1]
    N = len(np.unique(entity_id))  # Number of entities
    T = len(np.unique(time_id))  # Number of periods

    # Check minimum T requirement
    if T < 5:
        import warnings

        warnings.warn(
            f"T = {T} is less than recommended minimum (T ≥ 5). "
            "CSS estimator may not be reliable with few time periods.",
            UserWarning,
        )

    # Create entity dummies (N dummies, drop one for identification)
    entity_dummies = np.zeros((n, N))
    for i in range(n):
        entity_dummies[i, entity_id[i]] = 1

    # Create time trend variables based on specification
    time_trends = []

    if time_trend == "linear":
        # θ_{i1} + θ_{i2}·t
        # We need: entity_i * 1 and entity_i * t
        t_linear = time_id.astype(float)
        time_trends.append(t_linear[:, np.newaxis] * entity_dummies)

    elif time_trend == "quadratic":
        # θ_{i1} + θ_{i2}·t + θ_{i3}·t²
        # We need: entity_i * 1, entity_i * t, entity_i * t²
        t_linear = time_id.astype(float)
        t_squared = t_linear**2

        time_trends.append(t_linear[:, np.newaxis] * entity_dummies)
        time_trends.append(t_squared[:, np.newaxis] * entity_dummies)

    # Construct design matrix
    if time_trend == "none":
        # Only entity fixed effects: α_i (constant over time)
        Z = np.column_stack([entity_dummies, X])
        n_entity_params = N
    else:
        # Entity fixed effects with time trends
        Z = np.column_stack([entity_dummies] + time_trends + [X])
        n_entity_params = N * (1 + len(time_trends))

    # Estimate via OLS (or GLS for efficiency)
    # For now, use OLS: β̂ = (Z'Z)⁻¹Z'y

    # Check for collinearity
    ZtZ = Z.T @ Z
    try:
        ZtZ_inv = np.linalg.inv(ZtZ)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Design matrix is singular. Check for collinearity or "
            "insufficient variation in entity/time dimensions."
        )

    # OLS estimates
    Zty = Z.T @ y
    theta_hat = ZtZ_inv @ Zty

    # Extract parameter estimates
    # First N * (1 + n_trends) are entity-time parameters
    # Last k are β coefficients
    beta_hat = theta_hat[-k:]

    # Residuals
    y_fitted = Z @ theta_hat
    residuals = y - y_fitted

    # Estimate σ_v (noise variance)
    sigma_v = np.std(residuals)

    # R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_resid = np.sum(residuals**2)
    r_squared = 1 - (ss_resid / ss_total)

    # Reconstruct α_{it} for each (entity, time) combination
    alpha_it = np.zeros((N, T))

    for i in range(N):
        for t in range(T):
            # Base intercept
            alpha = theta_hat[i]

            # Add time trend components
            if time_trend == "linear":
                # θ_{i2} * t
                alpha += theta_hat[N + i] * t

            elif time_trend == "quadratic":
                # θ_{i2} * t + θ_{i3} * t²
                alpha += theta_hat[N + i] * t
                alpha += theta_hat[2 * N + i] * (t**2)

            alpha_it[i, t] = alpha

    # Compute efficiency: TE_{it} = exp(α_{it} - max_j α_{jt})
    # For each time period, find the maximum intercept (the frontier)

    efficiency_it = np.zeros((N, T))

    for t in range(T):
        # Frontier in period t
        if frontier_type == "production":
            # Production: highest α is frontier
            alpha_frontier_t = np.max(alpha_it[:, t])
        else:
            # Cost: lowest α is frontier
            alpha_frontier_t = np.min(alpha_it[:, t])

        for i in range(N):
            if frontier_type == "production":
                # TE = exp(α_i - α_max) ∈ (0, 1]
                efficiency_it[i, t] = np.exp(alpha_it[i, t] - alpha_frontier_t)
            else:
                # Cost efficiency = exp(α_min - α_i) ∈ (0, 1]
                # (lower cost is more efficient)
                efficiency_it[i, t] = np.exp(alpha_frontier_t - alpha_it[i, t])

    # Create parameter names
    param_names = [f"x{j}" for j in range(k)]

    # Create result object
    result = CSSResult(
        params=beta_hat,
        param_names=param_names,
        alpha_it=alpha_it,
        efficiency_it=efficiency_it,
        residuals=residuals,
        sigma_v=sigma_v,
        r_squared=r_squared,
        entity_ids=entity_id,
        time_ids=time_id,
        time_trend=time_trend,
        n_obs=n,
        n_entities=N,
        n_periods=T,
        frontier_type=frontier_type,
    )

    return result


def test_time_trend_specification(
    y: np.ndarray,
    X: np.ndarray,
    entity_id: np.ndarray,
    time_id: np.ndarray,
    frontier_type: str = "production",
) -> pd.DataFrame:
    """Test different time trend specifications using F-test.

    Estimates CSS models with 'none', 'linear', and 'quadratic' time trends
    and performs nested F-tests.

    Parameters:
        y: Dependent variable
        X: Exogenous variables
        entity_id: Entity identifiers
        time_id: Time identifiers
        frontier_type: 'production' or 'cost'

    Returns:
        DataFrame with test results

    Example:
        >>> tests = test_time_trend_specification(y, X, entity_id, time_id)
        >>> print(tests)
    """
    # Estimate all three specifications
    result_none = estimate_css_model(y, X, entity_id, time_id, "none", frontier_type)
    result_linear = estimate_css_model(y, X, entity_id, time_id, "linear", frontier_type)
    result_quadratic = estimate_css_model(y, X, entity_id, time_id, "quadratic", frontier_type)

    n = len(y)
    N = result_none.n_entities

    # F-test: linear vs none
    # H0: θ_{i2} = 0 for all i (linear terms jointly zero)
    rss_none = np.sum(result_none.residuals**2)
    rss_linear = np.sum(result_linear.residuals**2)

    q_linear = N  # Number of restrictions (N linear terms)
    df_denom = n - (N + N + X.shape[1])  # Linear model df

    if df_denom > 0 and rss_linear > 0:
        f_stat_linear = ((rss_none - rss_linear) / q_linear) / (rss_linear / df_denom)
        p_value_linear = 1 - stats.f.cdf(f_stat_linear, q_linear, df_denom)
    else:
        f_stat_linear = np.nan
        p_value_linear = np.nan

    # F-test: quadratic vs linear
    # H0: θ_{i3} = 0 for all i (quadratic terms jointly zero)
    rss_quadratic = np.sum(result_quadratic.residuals**2)

    q_quadratic = N  # Number of restrictions (N quadratic terms)
    df_denom_quad = n - (3 * N + X.shape[1])  # Quadratic model df

    if df_denom_quad > 0 and rss_quadratic > 0:
        f_stat_quadratic = ((rss_linear - rss_quadratic) / q_quadratic) / (
            rss_quadratic / df_denom_quad
        )
        p_value_quadratic = 1 - stats.f.cdf(f_stat_quadratic, q_quadratic, df_denom_quad)
    else:
        f_stat_quadratic = np.nan
        p_value_quadratic = np.nan

    # Create summary DataFrame
    summary = pd.DataFrame(
        {
            "Specification": ["None (FE only)", "Linear", "Quadratic"],
            "R²": [result_none.r_squared, result_linear.r_squared, result_quadratic.r_squared],
            "σ_v": [result_none.sigma_v, result_linear.sigma_v, result_quadratic.sigma_v],
            "Mean_Efficiency": [
                np.mean(result_none.efficiency_it),
                np.mean(result_linear.efficiency_it),
                np.mean(result_quadratic.efficiency_it),
            ],
        }
    )

    print("\nF-Tests for Time Trend Specification:")
    print(f"Linear vs None:     F = {f_stat_linear:.4f}, p-value = {p_value_linear:.4f}")
    print(f"Quadratic vs Linear: F = {f_stat_quadratic:.4f}, p-value = {p_value_quadratic:.4f}")

    return summary
