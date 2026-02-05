"""
Pedroni panel cointegration tests.

This module implements Pedroni's tests for cointegration in panel data.
Pedroni (1999, 2004) proposed seven test statistics: four within-dimension
(panel tests) and three between-dimension (group tests).

Reference:
    Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous
    panels with multiple regressors. Oxford Bulletin of Economics and statistics,
    61(S1), 653-670.

    Pedroni, P. (2004). Panel cointegration: asymptotic and finite sample
    properties of pooled time series tests with an application to the PPP
    hypothesis. Econometric theory, 20(3), 597-625.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PedroniTestResult:
    """
    Results from Pedroni panel cointegration tests.

    Attributes
    ----------
    panel_v : float
        Panel v-statistic (variance ratio)
    panel_rho : float
        Panel rho-statistic (Phillips-Perron type)
    panel_pp : float
        Panel PP-statistic (Phillips-Perron)
    panel_adf : float
        Panel ADF-statistic
    group_rho : float
        Group rho-statistic
    group_pp : float
        Group PP-statistic
    group_adf : float
        Group ADF-statistic
    pvalues : Dict[str, float]
        P-values for each statistic
    n_obs : int
        Total observations
    n_entities : int
        Number of cross-sections
    trend : str
        Trend specification used
    """

    panel_v: float
    panel_rho: float
    panel_pp: float
    panel_adf: float
    group_rho: float
    group_pp: float
    group_adf: float
    pvalues: Dict[str, float]
    n_obs: int
    n_entities: int
    trend: str

    @property
    def summary_conclusion(self) -> str:
        """Overall conclusion based on majority of tests."""
        reject_count = sum(1 for p in self.pvalues.values() if p < 0.05)
        total = len(self.pvalues)

        if reject_count >= total / 2:
            return f"Reject H0 ({reject_count}/{total} tests): Evidence of cointegration"
        else:
            return f"Fail to reject H0 ({total - reject_count}/{total} tests): No evidence of cointegration"

    def __str__(self) -> str:
        """String representation."""
        lines = []
        lines.append("=" * 70)
        lines.append("Pedroni Panel Cointegration Tests")
        lines.append("=" * 70)
        lines.append("")
        lines.append("Within-dimension (Panel statistics):")
        lines.append(
            f"  Panel v-statistic:      {self.panel_v:10.4f}  (p = {self.pvalues.get('panel_v', np.nan):.4f})"
        )
        lines.append(
            f"  Panel rho-statistic:    {self.panel_rho:10.4f}  (p = {self.pvalues['panel_rho']:.4f})"
        )
        lines.append(
            f"  Panel PP-statistic:     {self.panel_pp:10.4f}  (p = {self.pvalues['panel_pp']:.4f})"
        )
        lines.append(
            f"  Panel ADF-statistic:    {self.panel_adf:10.4f}  (p = {self.pvalues['panel_adf']:.4f})"
        )
        lines.append("")
        lines.append("Between-dimension (Group statistics):")
        lines.append(
            f"  Group rho-statistic:    {self.group_rho:10.4f}  (p = {self.pvalues['group_rho']:.4f})"
        )
        lines.append(
            f"  Group PP-statistic:     {self.group_pp:10.4f}  (p = {self.pvalues['group_pp']:.4f})"
        )
        lines.append(
            f"  Group ADF-statistic:    {self.group_adf:10.4f}  (p = {self.pvalues['group_adf']:.4f})"
        )
        lines.append("")
        lines.append(f"Observations:      {self.n_obs}")
        lines.append(f"Cross-sections:    {self.n_entities}")
        lines.append(f"Trend:             {self.trend}")
        lines.append("")
        lines.append("H0: No cointegration")
        lines.append("H1: Cointegration exists")
        lines.append("")
        lines.append(f"Conclusion: {self.summary_conclusion}")
        lines.append("=" * 70)
        return "\n".join(lines)


class PedroniTest:
    """
    Pedroni panel cointegration test.

    Tests for cointegration in panel data using seven statistics proposed
    by Pedroni (1999, 2004). Tests the null hypothesis of no cointegration
    against the alternative of cointegration.

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
    lags : int, optional
        Number of lags for ADF test. If None, uses automatic selection.

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>>
    >>> # Test cointegration between invest and value
    >>> ped = pb.PedroniTest(data, 'invest', ['value'], 'firm', 'year')
    >>> result = ped.run()
    >>> print(result)

    Notes
    -----
    This is a residual-based cointegration test. It first estimates the
    cointegrating regression for each panel, then tests for unit roots in
    the residuals using various panel unit root statistics.

    The test requires that variables are I(1). Users should verify this
    using panel unit root tests (LLC, IPS) before applying cointegration tests.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        dependent: str,
        independents: List[str],
        entity_col: str,
        time_col: str,
        trend: str = "c",
        lags: Optional[int] = None,
    ):
        self.data = data.copy()
        self.dependent = dependent
        self.independents = independents if isinstance(independents, list) else [independents]
        self.entity_col = entity_col
        self.time_col = time_col
        self.trend = trend
        self.lags = lags

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

        self.result: Optional[PedroniTestResult] = None

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
            return np.asarray(resid)
        except Exception:
            return np.full(len(y), np.nan)

    def _compute_panel_statistics(self, residuals_dict: Dict) -> Dict[str, float]:
        """
        Compute panel (within-dimension) statistics.

        Parameters
        ----------
        residuals_dict : dict
            Dictionary mapping entity to residuals

        Returns
        -------
        dict
            Panel statistics
        """
        # Collect residuals and compute statistics
        all_resid: list[float] = []
        all_resid_lag: list[float] = []
        all_delta_resid: list[float] = []

        for entity, resid in residuals_dict.items():
            if len(resid) < 3:
                continue

            resid_lag = resid[:-1]
            delta_resid = np.diff(resid)

            all_resid.extend(resid[1:])
            all_resid_lag.extend(resid_lag)
            all_delta_resid.extend(delta_resid)

        all_resid = np.array(all_resid)
        all_resid_lag = np.array(all_resid_lag)
        all_delta_resid = np.array(all_delta_resid)

        # Panel v-statistic (variance ratio)
        T = len(all_resid) // self.n_entities
        var_resid = np.var(all_resid, ddof=1)
        var_delta = np.var(all_delta_resid, ddof=1)
        panel_v = T**2 * self.n_entities * var_delta / var_resid if var_resid > 0 else np.nan

        # Panel rho-statistic (PP-type)
        numerator = np.sum(all_resid_lag * all_delta_resid)
        denominator = np.sum(all_resid_lag**2)
        panel_rho = numerator / denominator if denominator > 0 else np.nan

        # Panel PP-statistic
        sigma2 = np.var(all_delta_resid, ddof=1)
        panel_pp = (
            (numerator / np.sqrt(sigma2 * denominator))
            if denominator > 0 and sigma2 > 0
            else np.nan
        )

        # Panel ADF-statistic (simplified)
        # Δe_t = ρ e_{t-1} + error
        if len(all_resid_lag) > 0 and len(all_delta_resid) > 0:
            rho = np.sum(all_delta_resid * all_resid_lag) / np.sum(all_resid_lag**2)
            resid_adf = all_delta_resid - rho * all_resid_lag
            sigma2_adf = np.var(resid_adf, ddof=1)
            se_rho = np.sqrt(sigma2_adf / np.sum(all_resid_lag**2))
            panel_adf = rho / se_rho if se_rho > 0 else np.nan
        else:
            panel_adf = np.nan

        return {
            "panel_v": panel_v,
            "panel_rho": panel_rho,
            "panel_pp": panel_pp,
            "panel_adf": panel_adf,
        }

    def _compute_group_statistics(self, residuals_dict: Dict) -> Dict[str, float]:
        """
        Compute group (between-dimension) statistics.

        Parameters
        ----------
        residuals_dict : dict
            Dictionary mapping entity to residuals

        Returns
        -------
        dict
            Group statistics
        """
        rho_list = []
        pp_list = []
        adf_list = []

        for entity, resid in residuals_dict.items():
            if len(resid) < 3:
                continue

            resid_lag = resid[:-1]
            delta_resid = np.diff(resid)

            # Group rho
            num = np.sum(resid_lag * delta_resid)
            denom = np.sum(resid_lag**2)
            if denom > 0:
                rho_list.append(num / denom)

            # Group PP
            sigma2 = np.var(delta_resid, ddof=1)
            if denom > 0 and sigma2 > 0:
                pp_list.append(num / np.sqrt(sigma2 * denom))

            # Group ADF
            if len(resid_lag) > 0:
                rho_est = np.sum(delta_resid * resid_lag) / np.sum(resid_lag**2)
                resid_adf = delta_resid - rho_est * resid_lag
                sigma2_adf = np.var(resid_adf, ddof=1)
                se_rho = np.sqrt(sigma2_adf / np.sum(resid_lag**2))
                if se_rho > 0:
                    adf_list.append(rho_est / se_rho)

        return {
            "group_rho": np.mean(rho_list) if rho_list else np.nan,
            "group_pp": np.mean(pp_list) if pp_list else np.nan,
            "group_adf": np.mean(adf_list) if adf_list else np.nan,
        }

    def run(self) -> PedroniTestResult:
        """
        Run Pedroni panel cointegration tests.

        Returns
        -------
        PedroniTestResult
            Test results with all seven statistics

        Notes
        -----
        The procedure:
        1. For each panel, estimate cointegrating regression: y_it = α_i + β_i X_it + e_it
        2. Collect residuals e_it for all panels
        3. Compute panel statistics (pooled residuals)
        4. Compute group statistics (average of individual statistics)
        5. Standardize statistics and compute p-values
        """
        # Step 1: Estimate cointegrating regressions
        residuals_dict = {}
        n_obs_total = 0

        for entity in self.entities:
            entity_data = self.data[self.data[self.entity_col] == entity]
            resid = self._estimate_cointegrating_regression(entity_data)

            if not np.any(np.isnan(resid)):
                residuals_dict[entity] = resid
                n_obs_total += len(resid)

        if len(residuals_dict) == 0:
            raise ValueError("Insufficient data for Pedroni test")

        # Step 2: Compute panel statistics
        panel_stats = self._compute_panel_statistics(residuals_dict)

        # Step 3: Compute group statistics
        group_stats = self._compute_group_statistics(residuals_dict)

        # Step 4: Compute p-values (using standard normal approximation)
        # Note: These should be standardized using Pedroni's tables,
        # but for simplicity we use normal approximation
        pvalues = {}

        # Panel v is right-tailed
        if not np.isnan(panel_stats["panel_v"]):
            pvalues["panel_v"] = 1 - stats.norm.cdf(panel_stats["panel_v"])

        # All others are left-tailed
        for key in ["panel_rho", "panel_pp", "panel_adf", "group_rho", "group_pp", "group_adf"]:
            key.replace("panel_", "").replace("group_", "")
            if key.startswith("panel"):
                stat = panel_stats.get(key.replace("panel_", "panel_"))
            else:
                stat = group_stats.get(key.replace("group_", "group_"))

            if key in panel_stats:
                stat = panel_stats[key]
            elif key in group_stats:
                stat = group_stats[key]
            else:
                stat = np.nan

            if not np.isnan(stat):
                pvalues[key] = stats.norm.cdf(stat)

        # Create result
        trend_map = {"c": "Constant", "ct": "Constant and Trend"}

        self.result = PedroniTestResult(
            panel_v=panel_stats["panel_v"],
            panel_rho=panel_stats["panel_rho"],
            panel_pp=panel_stats["panel_pp"],
            panel_adf=panel_stats["panel_adf"],
            group_rho=group_stats["group_rho"],
            group_pp=group_stats["group_pp"],
            group_adf=group_stats["group_adf"],
            pvalues=pvalues,
            n_obs=n_obs_total,
            n_entities=len(residuals_dict),
            trend=trend_map[self.trend],
        )

        return self.result
