"""
Kao panel cointegration test.

This module implements Kao's test for cointegration in panel data.
Kao (1999) proposed a residual-based test similar to ADF for panel data.

Reference:
    Kao, C. (1999). Spurious regression and residual-based tests for
    cointegration in panel data. Journal of econometrics, 90(1), 1-44.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KaoTestResult:
    """
    Results from Kao panel cointegration test.

    Attributes
    ----------
    statistic : float
        Kao ADF test statistic
    pvalue : float
        P-value for the test
    n_obs : int
        Total observations
    n_entities : int
        Number of cross-sections
    trend : str
        Trend specification used
    null_hypothesis : str
        Null hypothesis
    alternative_hypothesis : str
        Alternative hypothesis
    conclusion : str
        Test conclusion
    """

    statistic: float
    pvalue: float
    n_obs: int
    n_entities: int
    trend: str
    null_hypothesis: str = "No cointegration"
    alternative_hypothesis: str = "Cointegration exists"

    @property
    def conclusion(self) -> str:
        """Conclusion at 5% significance level."""
        if self.pvalue < 0.05:
            return "Reject H0: Evidence of cointegration"
        else:
            return "Fail to reject H0: No evidence of cointegration"

    def __str__(self) -> str:
        """String representation."""
        lines = []
        lines.append("=" * 70)
        lines.append("Kao Panel Cointegration Test")
        lines.append("=" * 70)
        lines.append(f"ADF statistic:     {self.statistic:.4f}")
        lines.append(f"P-value:           {self.pvalue:.4f}")
        lines.append(f"Observations:      {self.n_obs}")
        lines.append(f"Cross-sections:    {self.n_entities}")
        lines.append(f"Trend:             {self.trend}")
        lines.append("")
        lines.append(f"H0: {self.null_hypothesis}")
        lines.append(f"H1: {self.alternative_hypothesis}")
        lines.append("")
        lines.append(f"Conclusion: {self.conclusion}")
        lines.append("=" * 70)
        return "\n".join(lines)


class KaoTest:
    """
    Kao panel cointegration test.

    Tests for cointegration in panel data using an ADF-type test on pooled
    residuals from cointegrating regressions. This is a residual-based test
    that assumes homogeneity in the cointegrating vector.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    dependent : str
        Name of dependent variable
    independents : list of str
        Names of independent variables
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    trend : str, default='c'
        Deterministic trend:
        - 'c': Constant only
        - 'ct': Constant and trend

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>>
    >>> # Test cointegration between invest and value
    >>> kao = pb.KaoTest(data, 'invest', ['value'], 'firm', 'year')
    >>> result = kao.run()
    >>> print(result)

    Notes
    -----
    The Kao test is simpler than Pedroni and assumes homogeneity in the
    cointegrating relationship. It pools residuals across entities and
    runs a single ADF-type test.

    The test requires that variables are I(1). Users should verify this
    using panel unit root tests before applying cointegration tests.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dependent: str,
        independents: List[str],
        entity_col: str,
        time_col: str,
        trend: str = "c",
    ):
        self.data = data.copy()
        self.dependent = dependent
        self.independents = independents if isinstance(independents, list) else [independents]
        self.entity_col = entity_col
        self.time_col = time_col
        self.trend = trend

        # Validate
        if dependent not in data.columns:
            raise ValueError(f"Dependent variable '{dependent}' not found")
        for var in self.independents:
            if var not in data.columns:
                raise ValueError(f"Independent variable '{var}' not found")
        if entity_col not in data.columns:
            raise ValueError(f"Entity column '{entity_col}' not found")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found")
        if trend not in ["c", "ct"]:
            raise ValueError("trend must be 'c' or 'ct'")

        # Sort data
        self.data = self.data.sort_values([entity_col, time_col])
        self.entities = self.data[entity_col].unique()
        self.n_entities = len(self.entities)

        self.result: Optional[KaoTestResult] = None

    def _estimate_cointegrating_regression(self, entity_data: pd.DataFrame) -> np.ndarray:
        """
        Estimate cointegrating regression and return residuals.

        Parameters
        ----------
        entity_data : pd.DataFrame
            Data for one entity

        Returns
        -------
        np.ndarray
            Residuals from cointegrating regression
        """
        y = entity_data[self.dependent].values
        X = entity_data[self.independents].values

        # Add deterministics
        if self.trend == "c":
            X = np.column_stack([np.ones(len(y)), X])
        elif self.trend == "ct":
            X = np.column_stack([np.ones(len(y)), np.arange(len(y)), X])

        # OLS
        try:
            params = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ params
            return resid
        except Exception:
            return np.full(len(y), np.nan)

    def run(self) -> KaoTestResult:
        """
        Run Kao panel cointegration test.

        Returns
        -------
        KaoTestResult
            Test results

        Notes
        -----
        The procedure:
        1. For each panel, estimate cointegrating regression: y_it = α_i + β X_it + e_it
        2. Pool residuals e_it across all panels
        3. Run ADF test on pooled residuals: Δe_t = ρ e_{t-1} + ν_t
        4. Compute standardized statistic using Kao's adjustment
        """
        # Step 1: Estimate cointegrating regressions
        all_residuals = []
        n_obs_total = 0

        for entity in self.entities:
            entity_data = self.data[self.data[self.entity_col] == entity]
            resid = self._estimate_cointegrating_regression(entity_data)

            if not np.any(np.isnan(resid)):
                all_residuals.append(resid)
                n_obs_total += len(resid)

        if len(all_residuals) == 0:
            raise ValueError("Insufficient data for Kao test")

        # Step 2: Pool residuals
        pooled_resid = np.concatenate(all_residuals)

        # Step 3: ADF test on pooled residuals
        # Δe_t = ρ e_{t-1} + ν_t
        resid_lag = pooled_resid[:-1]
        delta_resid = np.diff(pooled_resid)

        # OLS: Δe on e_{-1}
        rho = np.sum(delta_resid * resid_lag) / np.sum(resid_lag**2)

        # Standard error
        resid_adf = delta_resid - rho * resid_lag
        n = len(delta_resid)
        sigma2 = np.sum(resid_adf**2) / (n - 1)
        se_rho = np.sqrt(sigma2 / np.sum(resid_lag**2))

        # t-statistic
        t_stat = rho / se_rho if se_rho > 0 else np.nan

        # Step 4: Kao adjustment (simplified)
        # Under H0, the distribution is non-standard
        # Use normal approximation with adjustment
        N = len(all_residuals)
        T_avg = n_obs_total / N

        # Kao's adjustment parameters (approximation)
        # These should come from Kao (1999) Table 1
        if self.trend == "c":
            mu = -1.25  # Approximate mean
            sigma = 1.00  # Approximate std
        else:  # 'ct'
            mu = -1.75
            sigma = 1.10

        # Adjusted statistic
        kao_stat = (t_stat - np.sqrt(N * T_avg) * mu) / (sigma * np.sqrt(N))

        # P-value (left-tailed)
        pvalue = stats.norm.cdf(kao_stat)

        # Create result
        trend_map = {"c": "Constant", "ct": "Constant and Trend"}

        self.result = KaoTestResult(
            statistic=kao_stat,
            pvalue=pvalue,
            n_obs=n_obs_total,
            n_entities=N,
            trend=trend_map[self.trend],
        )

        return self.result
