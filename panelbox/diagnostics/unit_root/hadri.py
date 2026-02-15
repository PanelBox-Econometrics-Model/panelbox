"""
Hadri (2000) LM Test for Panel Data Stationarity.

Reference:
    Hadri, K. (2000). "Testing for Stationarity in Heterogeneous Panel Data."
    Econometrics Journal, 3(2), 148-161.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS


@dataclass
class HadriResult:
    """
    Results from Hadri (2000) LM test for panel data stationarity.

    Attributes
    ----------
    statistic : float
        Z-statistic (standardized LM statistic).
    pvalue : float
        P-value from standard normal distribution.
    reject : bool
        Whether to reject H0 of stationarity at 5% level.
    lm_statistic : float
        Raw LM statistic before standardization.
    individual_lm : np.ndarray
        LM statistic for each cross-sectional unit.
    n_entities : int
        Number of cross-sectional units.
    n_time : int
        Number of time periods.
    trend : str
        Deterministic specification ('c' or 'ct').
    robust : bool
        Whether heteroskedasticity-robust version was used.

    Notes
    -----
    H0: All series are stationary (σ²ᵤᵢ = 0 for all i)
    H1: At least one series has a unit root (σ²ᵤᵢ > 0 for some i)

    This is the opposite of traditional unit root tests like IPS/LLC.
    """

    statistic: float
    pvalue: float
    reject: bool
    lm_statistic: float
    individual_lm: np.ndarray
    n_entities: int
    n_time: int
    trend: str
    robust: bool

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
            "Hadri (2000) LM Test for Stationarity",
            "=" * 70,
            f"H0: All series are stationary",
            f"H1: At least one series has a unit root",
            "",
            f"Specification: {self._format_trend()}",
            f"Robust version: {'Yes' if self.robust else 'No'}",
            f"Number of entities (N): {self.n_entities}",
            f"Number of periods (T): {self.n_time}",
            "",
            f"LM statistic: {self.lm_statistic:.4f}",
            f"Z-statistic: {self.statistic:.4f}",
            f"P-value: {self.pvalue:.4f}",
            "",
            f"Decision at 5% level: {'REJECT H0' if self.reject else 'FAIL TO REJECT H0'}",
            "",
        ]

        if self.reject:
            lines.append("⚠ Evidence against stationarity (at least one series has unit root)")
        else:
            lines.append("✓ No evidence against stationarity (series appear stationary)")

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
            f"HadriResult(statistic={self.statistic:.4f}, "
            f"pvalue={self.pvalue:.4f}, reject={self.reject})"
        )


def hadri_test(
    data: pd.DataFrame,
    variable: str,
    entity_col: str = "entity",
    time_col: str = "time",
    trend: Literal["c", "ct"] = "c",
    robust: bool = True,
    alpha: float = 0.05,
) -> HadriResult:
    """
    Hadri (2000) LM test for stationarity in panel data.

    Tests the null hypothesis that all series in the panel are stationary
    against the alternative that at least one series has a unit root.

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
    trend : {'c', 'ct'}, default 'c'
        Deterministic specification:
        - 'c': constant only
        - 'ct': constant and linear trend
    robust : bool, default True
        If True, use heteroskedasticity-robust version.
    alpha : float, default 0.05
        Significance level for hypothesis test.

    Returns
    -------
    HadriResult
        Test results including statistic, p-value, and decision.

    Notes
    -----
    The test is based on the KPSS approach applied to panel data.

    Model decomposition:
        yᵢₜ = rᵢₜ + βᵢ t + εᵢₜ
        rᵢₜ = rᵢ,ₜ₋₁ + uᵢₜ

    H0: σ²ᵤᵢ = 0 for all i (no random walk → stationary)

    LM statistic:
        LM = (1/N) Σᵢ LMᵢ
        LMᵢ = (1/T²) Σₜ S²ᵢₜ / σ̂²εᵢ
        Sᵢₜ = Σₛ₌₁ᵗ ε̂ᵢₛ (partial sum of residuals)

    Asymptotic distribution:
        √N (LM - μ) / σ →ᵈ N(0, 1)

    where μ, σ depend on T and deterministic specification.

    References
    ----------
    Hadri, K. (2000). "Testing for Stationarity in Heterogeneous Panel Data."
    Econometrics Journal, 3(2), 148-161.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from panelbox.diagnostics.unit_root import hadri_test
    >>>
    >>> # Generate stationary panel data
    >>> np.random.seed(42)
    >>> data = []
    >>> for i in range(10):
    ...     y = np.random.randn(100).cumsum() * 0.1  # Near stationary
    ...     for t, val in enumerate(y):
    ...         data.append({'entity': i, 'time': t, 'y': val})
    >>> df = pd.DataFrame(data)
    >>>
    >>> # Test for stationarity
    >>> result = hadri_test(df, 'y', trend='c')
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

    # Storage for individual LM statistics
    individual_lm = np.zeros(N)
    T_common = None

    # Compute LM statistic for each entity
    for idx, entity in enumerate(entities):
        entity_data = data[data[entity_col] == entity].copy()
        T = len(entity_data)

        if T_common is None:
            T_common = T
        elif T != T_common:
            raise ValueError("Hadri test requires balanced panel (same T for all entities)")

        # Extract variable
        y = entity_data[variable].values

        # Create regressors
        if trend == "c":
            X = np.ones((T, 1))
        else:  # trend == 'ct'
            t = np.arange(T)
            X = np.column_stack([np.ones(T), t])

        # Regression to get residuals
        model = OLS(y, X)
        result = model.fit()
        residuals = result.resid

        # Compute partial sums
        partial_sums = np.cumsum(residuals)

        # Compute variance of residuals
        if robust:
            # Heteroskedasticity-robust variance
            sigma2_e = _compute_robust_variance(residuals)
        else:
            # Homoskedastic variance
            sigma2_e = np.var(residuals, ddof=len(X[0]))

        # Compute LM statistic for this entity
        # LMᵢ = (1/T²) Σₜ S²ᵢₜ / σ̂²εᵢ
        individual_lm[idx] = np.sum(partial_sums**2) / (T**2 * sigma2_e)

    # Average LM statistic
    lm_stat = np.mean(individual_lm)

    # Get asymptotic moments
    mu, sigma = _get_asymptotic_moments(T_common, trend)

    # Standardized statistic
    z_stat = np.sqrt(N) * (lm_stat - mu) / sigma

    # P-value (one-sided test, reject for large values)
    pvalue = 1 - stats.norm.cdf(z_stat)

    # Decision
    reject = pvalue < alpha

    return HadriResult(
        statistic=z_stat,
        pvalue=pvalue,
        reject=reject,
        lm_statistic=lm_stat,
        individual_lm=individual_lm,
        n_entities=N,
        n_time=T_common,
        trend=trend,
        robust=robust,
    )


def _compute_robust_variance(residuals: np.ndarray) -> float:
    """
    Compute heteroskedasticity-robust variance estimate.

    Uses Newey-West type estimator as in Hadri (2000).

    Parameters
    ----------
    residuals : np.ndarray
        Residuals from regression.

    Returns
    -------
    float
        Robust variance estimate.
    """
    T = len(residuals)

    # Automatic bandwidth selection (Newey-West)
    bandwidth = int(np.floor(4 * (T / 100) ** (2 / 9)))

    # Compute variance with Bartlett kernel
    gamma0 = np.var(residuals, ddof=0)

    variance = gamma0
    for lag in range(1, bandwidth + 1):
        weight = 1 - lag / (bandwidth + 1)  # Bartlett kernel
        gamma_lag = np.mean(residuals[lag:] * residuals[:-lag])
        variance += 2 * weight * gamma_lag

    return variance


def _get_asymptotic_moments(T: int, trend: str) -> tuple[float, float]:
    """
    Get asymptotic mean and standard deviation for LM statistic.

    Based on Hadri (2000) Table 1.

    Parameters
    ----------
    T : int
        Number of time periods.
    trend : str
        Deterministic specification ('c' or 'ct').

    Returns
    -------
    tuple[float, float]
        Mean (μ) and standard deviation (σ) for standardization.

    Notes
    -----
    These are asymptotic moments. For finite T, they are approximations.

    The exact formulas are:
    - For constant only (c):
        μ = 1/6
        σ² = 1/45
    - For constant + trend (ct):
        μ = 1/15
        σ² = 1/6300
    """
    if trend == "c":
        # Constant only
        mu = 1.0 / 6.0
        sigma = np.sqrt(1.0 / 45.0)
    else:  # trend == 'ct'
        # Constant and trend
        mu = 1.0 / 15.0
        sigma = np.sqrt(1.0 / 6300.0)

    return mu, sigma
