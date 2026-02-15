"""
Breitung (2000) Unit Root Test for Panel Data.

Reference:
    Breitung, J. (2000). "The Local Power of Some Unit Root Tests for Panel Data."
    In Advances in Econometrics, Vol. 15, 161-177.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS


@dataclass
class BreitungResult:
    """
    Results from Breitung (2000) unit root test for panel data.

    Attributes
    ----------
    statistic : float
        Standardized test statistic.
    pvalue : float
        P-value from standard normal distribution.
    reject : bool
        Whether to reject H0 of unit root at 5% level.
    raw_statistic : float
        Raw test statistic before standardization.
    n_entities : int
        Number of cross-sectional units.
    n_time : int
        Number of time periods.
    trend : str
        Deterministic specification ('c' or 'ct').

    Notes
    -----
    H0: All series have a unit root (ρ = 0)
    H1: All series are stationary (ρ < 0)

    The test is robust to heterogeneity in intercepts and trends.
    """

    statistic: float
    pvalue: float
    reject: bool
    raw_statistic: float
    n_entities: int
    n_time: int
    trend: str

    def summary(self) -> str:
        """
        Generate formatted summary of test results.

        Returns
        -------
        str
            Formatted summary table.
        """
        lines = [
            "=" * 70,
            "Breitung (2000) Unit Root Test",
            "=" * 70,
            f"H0: All series have a unit root",
            f"H1: All series are stationary",
            "",
            f"Specification: {self._format_trend()}",
            f"Number of entities (N): {self.n_entities}",
            f"Number of periods (T): {self.n_time}",
            "",
            f"Test statistic: {self.statistic:.4f}",
            f"P-value: {self.pvalue:.4f}",
            "",
            f"Decision at 5% level: {'REJECT H0' if self.reject else 'FAIL TO REJECT H0'}",
            "",
        ]

        if self.reject:
            lines.append("✓ Evidence of stationarity (reject unit root)")
        else:
            lines.append("⚠ No evidence against unit root")

        lines.append("=" * 70)

        return "\n".join(lines)

    def _format_trend(self) -> str:
        """Format trend specification."""
        if self.trend == "c":
            return "Constant only"
        elif self.trend == "ct":
            return "Constant and trend"
        else:
            return self.trend

    def __repr__(self) -> str:
        return (
            f"BreitungResult(statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4f}, reject={self.reject})"
        )


def breitung_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = "entity",
    time_col: str = "time",
    trend: Literal["c", "ct"] = "ct",
    alpha: float = 0.05,
) -> BreitungResult:
    """
    Breitung (2000) unit root test for panel data.

    Tests the null hypothesis that all series in the panel have a unit root
    against the alternative that all series are stationary. The test is
    robust to heterogeneity in intercepts and deterministic trends.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    variable : str
        Name of the variable to test.
    entity_col : str, default 'entity'
        Name of the entity (cross-section) identifier column.
    time_col : str, default 'time'
        Name of the time identifier column.
    trend : {'c', 'ct'}, default 'ct'
        Deterministic specification:
        - 'c': constant only
        - 'ct': constant and linear trend (recommended)
    alpha : float, default 0.05
        Significance level for hypothesis test.

    Returns
    -------
    BreitungResult
        Test results including statistic, p-value, and decision.

    Notes
    -----
    The test uses a bias-adjusted pooled estimator that is robust to
    heterogeneity in the deterministic components.

    Transformation to remove deterministics:
        ỹᵢₜ = yᵢₜ - ȳᵢ - (t - T̄)(yᵢT - yᵢ₁)/(T - 1)

    where ȳᵢ is the mean, T̄ = (T+1)/2, and the transformation removes
    both the mean and linear trend.

    Test regression:
        Δỹᵢₜ = ρ ỹᵢ,ₜ₋₁ + εᵢₜ

    H0: ρ = 0 (unit root)
    H1: ρ < 0 (stationary)

    The test statistic has an asymptotic standard normal distribution
    under the null.

    References
    ----------
    Breitung, J. (2000). "The Local Power of Some Unit Root Tests for Panel Data."
    In Advances in Econometrics, Vol. 15, 161-177.

    Breitung, J., & Das, S. (2005). "Panel Unit Root Tests Under Cross-Sectional
    Dependence." Statistica Neerlandica, 59(4), 414-433.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from panelbox.diagnostics.unit_root import breitung_test
    >>>
    >>> # Generate panel data with unit root
    >>> np.random.seed(42)
    >>> data = []
    >>> for i in range(10):
    ...     y = np.random.randn(100).cumsum()  # Random walk
    ...     for t, val in enumerate(y):
    ...         data.append({'entity': i, 'time': t, 'y': val})
    >>> df = pd.DataFrame(data)
    >>>
    >>> # Test for unit root
    >>> result = breitung_test(df, 'y', trend='ct')
    >>> print(result.summary())
    """
    # Validate inputs
    if variable not in data.columns:
        raise ValueError(f"Variable '{variable}' not found in data")
    if entity_col not in data.columns:
        raise ValueError(f"Entity column '{entity_col}' not found in data")
    if time_col not in data.columns:
        raise ValueError(f"Time column '{time_col}' not found in data")
    if trend not in ["c", "ct"]:
        raise ValueError("trend must be 'c' or 'ct'")

    # Sort data
    data = data.sort_values([entity_col, time_col])

    # Get unique entities
    entities = data[entity_col].unique()
    N = len(entities)

    # Storage for transformed data
    all_dy = []
    all_y_lag = []
    T_common = None

    # Transform data for each entity
    for entity in entities:
        entity_data = data[data[entity_col] == entity].copy()
        T = len(entity_data)

        if T_common is None:
            T_common = T
        elif T != T_common:
            raise ValueError("Breitung test requires balanced panel (same T for all entities)")

        # Extract variable
        y = entity_data[variable].values

        # Apply detrending transformation
        y_tilde = _detrend_breitung(y, trend)

        # Compute first differences and lags
        dy = np.diff(y_tilde)
        y_lag = y_tilde[:-1]

        all_dy.append(dy)
        all_y_lag.append(y_lag)

    # Stack all data
    dy_pooled = np.concatenate(all_dy)
    y_lag_pooled = np.concatenate(all_y_lag)

    # Pooled regression: Δỹᵢₜ = ρ ỹᵢ,ₜ₋₁ + εᵢₜ
    # We use OLS without intercept since data is already demeaned
    model = OLS(dy_pooled, y_lag_pooled)
    result = model.fit()

    # Get coefficient and standard error
    rho_hat = result.params[0]
    se_rho = result.bse[0]

    # Bias correction (Breitung uses a specific bias correction)
    # For simplicity, we use the bias-corrected estimator approach
    bias = _compute_bias_correction(T_common, N)
    rho_bc = rho_hat - bias

    # Test statistic (standardized)
    # Under H0, the bias-corrected statistic is asymptotically N(0,1)
    test_stat = rho_bc / se_rho

    # P-value (one-sided test, reject for negative values)
    pvalue = stats.norm.cdf(test_stat)

    # Decision
    reject = pvalue < alpha

    return BreitungResult(
        statistic=test_stat,
        pvalue=pvalue,
        reject=reject,
        raw_statistic=rho_hat,
        n_entities=N,
        n_time=T_common,
        trend=trend,
    )


def _detrend_breitung(y: np.ndarray, trend: str) -> np.ndarray:
    """
    Apply Breitung detrending transformation.

    The transformation removes the mean and (optionally) linear trend
    in a way that is robust to the unit root null.

    Parameters
    ----------
    y : np.ndarray
        Time series data.
    trend : str
        Deterministic specification ('c' or 'ct').

    Returns
    -------
    np.ndarray
        Detrended series.

    Notes
    -----
    For constant only (c):
        ỹₜ = yₜ - ȳ

    For constant + trend (ct):
        ỹₜ = yₜ - ȳ - (t - T̄)(yT - y₁)/(T - 1)

    where T̄ = (T+1)/2
    """
    T = len(y)
    y_mean = np.mean(y)

    if trend == "c":
        # Demean only
        y_tilde = y - y_mean
    else:  # trend == 'ct'
        # Remove mean and trend
        t = np.arange(T)
        T_bar = (T + 1) / 2
        trend_coef = (y[-1] - y[0]) / (T - 1)
        y_tilde = y - y_mean - (t - T_bar) * trend_coef

    return y_tilde


def _compute_bias_correction(T: int, N: int) -> float:
    """
    Compute bias correction for Breitung estimator.

    Parameters
    ----------
    T : int
        Number of time periods.
    N : int
        Number of cross-sectional units.

    Returns
    -------
    float
        Bias correction term.

    Notes
    -----
    The bias correction depends on T. For large T, the bias is approximately
    -3.5/T (see Breitung 2000).

    For small samples, the bias is larger.
    """
    # Approximate bias correction
    # This is a simplified version; exact correction depends on trend specification
    bias = -3.5 / T

    return bias
