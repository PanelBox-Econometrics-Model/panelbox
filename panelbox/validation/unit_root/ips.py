"""
Im-Pesaran-Shin (IPS) panel unit root test.

This module implements the IPS test for unit roots in panel data.
Unlike LLC, the IPS test allows for heterogeneity in the autoregressive
coefficients across panels.

Reference:
    Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in
    heterogeneous panels. Journal of econometrics, 115(1), 53-74.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class IPSTestResult:
    """
    Results from IPS panel unit root test.

    Attributes
    ----------
    statistic : float
        IPS W-statistic (standardized)
    t_bar : float
        Average of individual ADF t-statistics
    pvalue : float
        P-value for the test
    lags : int or list
        Number of lags used (int if same for all, list if different)
    n_obs : int
        Total number of observations used
    n_entities : int
        Number of cross-sections
    individual_stats : dict
        Individual ADF t-statistics for each entity
    test_type : str
        Type of test ('IPS')
    deterministics : str
        Deterministic terms included
    null_hypothesis : str
        Description of null hypothesis
    alternative_hypothesis : str
        Description of alternative hypothesis
    conclusion : str
        Test conclusion at 5% significance level
    """

    statistic: float
    t_bar: float
    pvalue: float
    lags: Any  # int or list
    n_obs: int
    n_entities: int
    individual_stats: Dict[Any, float]
    test_type: str
    deterministics: str
    null_hypothesis: str = "All panels contain unit roots"
    alternative_hypothesis: str = "Some panels are stationary"

    @property
    def conclusion(self) -> str:
        """Conclusion at 5% significance level."""
        if self.pvalue < 0.05:
            return "Reject H0: Evidence that some panels are stationary"
        else:
            return "Fail to reject H0: Evidence of unit root"

    def __str__(self) -> str:
        """String representation."""
        lines = []
        lines.append("=" * 70)
        lines.append("Im-Pesaran-Shin Panel Unit Root Test")
        lines.append("=" * 70)
        lines.append(f"W-statistic:       {self.statistic:.4f}")
        lines.append(f"t-bar statistic:   {self.t_bar:.4f}")
        lines.append(f"P-value:           {self.pvalue:.4f}")
        if isinstance(self.lags, int):
            lines.append(f"Lags:              {self.lags}")
        else:
            lines.append(f"Lags:              Variable (mean={np.mean(self.lags):.1f})")
        lines.append(f"Observations:      {self.n_obs}")
        lines.append(f"Cross-sections:    {self.n_entities}")
        lines.append(f"Deterministics:    {self.deterministics}")
        lines.append("")
        lines.append(f"H0: {self.null_hypothesis}")
        lines.append(f"H1: {self.alternative_hypothesis}")
        lines.append("")
        lines.append(f"Conclusion: {self.conclusion}")
        lines.append("=" * 70)
        return "\n".join(lines)


class IPSTest:
    """
    Im-Pesaran-Shin panel unit root test.

    The IPS test allows for heterogeneity in the autoregressive coefficients:
    Δy_it = ρ_i y_{i,t-1} + Σ_j θ_ij Δy_{i,t-j} + α_i + δ_i t + ε_it

    H0: ρ_i = 0 for all i (all panels have unit root)
    H1: ρ_i < 0 for some i (some panels are stationary)

    The test averages individual ADF t-statistics and standardizes them.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    variable : str
        Name of variable to test for unit root
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    lags : int or dict, optional
        Number of lags for augmented term. If int, same for all entities.
        If dict, maps entity to lags. If None, selected automatically per entity.
    trend : str, default='c'
        Deterministic terms to include:
        - 'n': No deterministic terms
        - 'c': Constant only
        - 'ct': Constant and trend

    Attributes
    ----------
    result : IPSTestResult
        Test results after running

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>>
    >>> # Test for unit root in 'invest'
    >>> ips = pb.IPSTest(data, 'invest', 'firm', 'year')
    >>> result = ips.run()
    >>> print(result)
    >>>
    >>> # With trend
    >>> ips_trend = pb.IPSTest(data, 'invest', 'firm', 'year', trend='ct')
    >>> result = ips_trend.run()

    Notes
    -----
    The IPS test allows for heterogeneity across panels, making it more general
    than the LLC test. It can handle unbalanced panels, though observations are
    dropped for entities with insufficient data.

    The test uses simulated critical values from IPS (2003) for standardization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variable: str,
        entity_col: str,
        time_col: str,
        lags: Optional[Any] = None,
        trend: str = "c",
    ):
        self.data = data.copy()
        self.variable = variable
        self.entity_col = entity_col
        self.time_col = time_col
        self.lags = lags
        self.trend = trend

        # Validate inputs
        if variable not in data.columns:
            raise ValueError(f"Variable '{variable}' not found in data")
        if entity_col not in data.columns:
            raise ValueError(f"Entity column '{entity_col}' not found in data")
        if time_col not in data.columns:
            raise ValueError(f"Time column '{time_col}' not found in data")
        if trend not in ["n", "c", "ct"]:
            raise ValueError("trend must be 'n', 'c', or 'ct'")

        # Sort data
        self.data = self.data.sort_values([entity_col, time_col])

        # Get entities
        self.entities = self.data[entity_col].unique()
        self.n_entities = len(self.entities)

        self.result = None

    def _select_lags_for_entity(self, entity_data: np.ndarray, max_lags: int = 12) -> int:
        """
        Select optimal lags for a single entity using AIC.

        Parameters
        ----------
        entity_data : np.ndarray
            Time series data for one entity
        max_lags : int
            Maximum number of lags to consider

        Returns
        -------
        int
            Optimal number of lags
        """
        T = len(entity_data)
        max_lags = min(max_lags, T // 4)

        if max_lags < 1:
            return 0

        best_lag = 0
        best_aic = np.inf

        for p in range(max_lags + 1):
            try:
                aic = self._compute_aic_entity(entity_data, p)
                if aic < best_aic:
                    best_aic = aic
                    best_lag = p
            except Exception:
                continue

        return best_lag

    def _compute_aic_entity(self, entity_data: np.ndarray, lags: int) -> float:
        """Compute AIC for given entity and lag order."""
        if len(entity_data) < lags + 3:
            return np.inf

        dy = np.diff(entity_data)
        y_lag = entity_data[lags:-1]
        dy_dep = dy[lags:]

        # Build regressors
        X = [y_lag]

        # Add lagged differences
        for j in range(1, lags + 1):
            if lags - j >= 0 and len(dy) > lags:
                dy_lag = dy[lags - j : -j] if j > 0 else dy[lags:]
                if len(dy_lag) == len(dy_dep):
                    X.append(dy_lag)

        # Add deterministics
        if self.trend == "c":
            X.append(np.ones(len(dy_dep)))
        elif self.trend == "ct":
            X.append(np.ones(len(dy_dep)))
            X.append(np.arange(len(dy_dep)))

        try:
            X_mat = np.column_stack(X)
            params = np.linalg.lstsq(X_mat, dy_dep, rcond=None)[0]
            resid = dy_dep - X_mat @ params
            n = len(resid)
            k = len(X)
            sigma2 = np.sum(resid**2) / n
            aic = n * np.log(sigma2) + 2 * k
            return aic
        except Exception:
            return np.inf

    def _adf_test_entity(self, entity_data: np.ndarray, lags: int) -> Tuple[float, int]:
        """
        Run ADF test for a single entity.

        Parameters
        ----------
        entity_data : np.ndarray
            Time series for one entity
        lags : int
            Number of lags

        Returns
        -------
        tuple
            (t-statistic, effective observations)
        """
        if len(entity_data) < lags + 3:
            return np.nan, 0

        dy = np.diff(entity_data)
        y_lag = entity_data[lags:-1]
        dy_dep = dy[lags:]

        # Build regressors for ADF regression
        X = []

        # Add lagged differences
        for j in range(1, lags + 1):
            if lags - j >= 0 and len(dy) > lags:
                dy_lag = dy[lags - j : -j] if j > 0 else dy[lags:]
                if len(dy_lag) == len(dy_dep):
                    X.append(dy_lag)

        # Add deterministics
        if self.trend == "c":
            X.append(np.ones(len(dy_dep)))
        elif self.trend == "ct":
            X.append(np.ones(len(dy_dep)))
            X.append(np.arange(len(dy_dep)))

        # Full regression: Δy_t = ρ y_{t-1} + Σ θ_j Δy_{t-j} + deterministics
        if len(X) > 0:
            X.append(y_lag)  # Add lagged level last
            try:
                X_mat = np.column_stack(X)
                params = np.linalg.lstsq(X_mat, dy_dep, rcond=None)[0]
                resid = dy_dep - X_mat @ params

                # t-statistic for ρ (coefficient on y_{t-1})
                rho_idx = -1  # Last coefficient
                rho = params[rho_idx]

                # Standard error of ρ
                n = len(resid)
                sigma2 = np.sum(resid**2) / (n - len(params))
                X_cov = np.linalg.inv(X_mat.T @ X_mat) * sigma2
                se_rho = np.sqrt(X_cov[rho_idx, rho_idx])

                t_stat = rho / se_rho

                return t_stat, n
            except Exception:
                return np.nan, 0
        else:
            # No lags, simple ADF
            try:
                X_mat = y_lag.reshape(-1, 1)
                params = np.linalg.lstsq(X_mat, dy_dep, rcond=None)[0]
                resid = dy_dep - X_mat.flatten() * params[0]

                n = len(resid)
                sigma2 = np.sum(resid**2) / (n - 1)
                se_rho = np.sqrt(sigma2 / np.sum(y_lag**2))
                t_stat = params[0] / se_rho

                return t_stat, n
            except Exception:
                return np.nan, 0

    def _get_critical_values(self, T: int) -> Dict[str, float]:
        """
        Get simulated mean and variance for standardization.

        Based on IPS (2003) Table 2.

        Parameters
        ----------
        T : int
            Average time dimension

        Returns
        -------
        dict
            Dictionary with 'mean' and 'std' for the t-bar distribution under H0
        """
        # Simulated values from IPS (2003) Table 2
        # These are approximations for large T

        if self.trend == "n":
            # No deterministic terms
            if T <= 25:
                mean = -1.00
                std = 0.80
            elif T <= 50:
                mean = -1.01
                std = 0.81
            else:
                mean = -1.02
                std = 0.82
        elif self.trend == "c":
            # Constant
            if T <= 25:
                mean = -1.53
                std = 0.90
            elif T <= 50:
                mean = -1.66
                std = 0.96
            else:
                mean = -1.73
                std = 1.00
        else:  # 'ct'
            # Constant and trend
            if T <= 25:
                mean = -2.17
                std = 0.93
            elif T <= 50:
                mean = -2.33
                std = 0.99
            else:
                mean = -2.51
                std = 1.04

        return {"mean": mean, "std": std}

    def run(self) -> IPSTestResult:
        """
        Run the IPS panel unit root test.

        Returns
        -------
        IPSTestResult
            Test results

        Notes
        -----
        The test procedure:
        1. For each panel i, run ADF test to get t_i statistic
        2. Compute average: t-bar = (1/N) Σ t_i
        3. Standardize: W = √N (t-bar - E[t_i]) / √Var[t_i]
        4. W follows N(0,1) under H0
        """
        # Step 1: Determine lags for each entity
        lags_dict = {}

        if self.lags is None:
            # Auto-select for each entity
            for entity in self.entities:
                entity_data = self.data[self.data[self.entity_col] == entity][self.variable].values
                T = len(entity_data)
                max_lags = min(12, int(np.floor((T / 100) ** (1 / 3) * 12)))
                lags_dict[entity] = self._select_lags_for_entity(entity_data, max_lags)
        elif isinstance(self.lags, dict):
            lags_dict = self.lags
        else:
            # Same lags for all
            for entity in self.entities:
                lags_dict[entity] = self.lags

        # Step 2: Run ADF test for each entity
        t_stats = []
        t_stats_dict = {}
        n_obs_total = 0
        T_list = []

        for entity in self.entities:
            entity_data = self.data[self.data[self.entity_col] == entity][self.variable].values
            lags = lags_dict[entity]

            t_stat, n_obs = self._adf_test_entity(entity_data, lags)

            if not np.isnan(t_stat) and n_obs > 0:
                t_stats.append(t_stat)
                t_stats_dict[entity] = t_stat
                n_obs_total += n_obs
                T_list.append(n_obs)

        if len(t_stats) == 0:
            raise ValueError("Insufficient data for IPS test")

        # Step 3: Compute t-bar
        N = len(t_stats)
        t_bar = np.mean(t_stats)

        # Step 4: Standardize using critical values
        T_avg = np.mean(T_list)
        crit_vals = self._get_critical_values(int(T_avg))

        E_t = crit_vals["mean"]
        Var_t = crit_vals["std"] ** 2

        # W-statistic (IPS standardized statistic)
        W_stat = np.sqrt(N) * (t_bar - E_t) / np.sqrt(Var_t)

        # P-value from standard normal
        pvalue = stats.norm.cdf(W_stat)

        # Determine lags for output
        lags_list = [lags_dict[entity] for entity in self.entities if entity in t_stats_dict]
        if len(set(lags_list)) == 1:
            lags_output = lags_list[0]
        else:
            lags_output = lags_list

        # Deterministics string
        det_map = {"n": "None", "c": "Constant", "ct": "Constant and Trend"}

        # Create result
        self.result = IPSTestResult(
            statistic=W_stat,
            t_bar=t_bar,
            pvalue=pvalue,
            lags=lags_output,
            n_obs=n_obs_total,
            n_entities=N,
            individual_stats=t_stats_dict,
            test_type="IPS",
            deterministics=det_map[self.trend],
        )

        return self.result
