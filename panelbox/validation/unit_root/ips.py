"""
Im-Pesaran-Shin (IPS) panel unit root test.

This module implements the IPS test for unit roots in panel data.
Unlike LLC, the IPS test allows for heterogeneity in the autoregressive
coefficients across panels.

Reference:
    Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in
    heterogeneous panels. Journal of econometrics, 115(1), 53-74.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Moments of the individual ADF t-statistic under the null (Im, Pesaran and
# Shin 2003, Table 3), used to standardize t-bar into the W_tbar statistic.
# Rows: ADF lags 0..8; columns: effective T in _IPS_T. NaN = not tabulated.
_IPS_T = np.array([10, 15, 20, 25, 30, 40, 50, 60, 70, 100], dtype=float)
_NA = np.nan
_IPS_MOMENTS = {
    "c": {
        "mean": np.array(
            [
                [-1.504, -1.514, -1.522, -1.520, -1.526, -1.523, -1.527, -1.519, -1.524, -1.532],
                [-1.488, -1.503, -1.516, -1.514, -1.519, -1.520, -1.524, -1.519, -1.522, -1.530],
                [-1.319, -1.387, -1.428, -1.443, -1.460, -1.476, -1.493, -1.490, -1.498, -1.514],
                [-1.306, -1.366, -1.413, -1.433, -1.453, -1.471, -1.489, -1.486, -1.495, -1.512],
                [-1.171, -1.260, -1.329, -1.363, -1.394, -1.428, -1.454, -1.458, -1.470, -1.495],
                [_NA, _NA, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494],
                [_NA, _NA, _NA, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476],
                [_NA, _NA, _NA, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474],
                [_NA, _NA, _NA, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456],
            ]
        ),
        "var": np.array(
            [
                [1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735],
                [1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745],
                [1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754],
                [1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761],
                [2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771],
                [_NA, _NA, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781],
                [_NA, _NA, _NA, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795],
                [_NA, _NA, _NA, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806],
                [_NA, _NA, _NA, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818],
            ]
        ),
    },
    "ct": {
        "mean": np.array(
            [
                [-2.166, -2.167, -2.168, -2.167, -2.172, -2.173, -2.176, -2.174, -2.174, -2.177],
                [-2.173, -2.169, -2.172, -2.172, -2.173, -2.177, -2.180, -2.178, -2.176, -2.179],
                [-1.914, -1.999, -2.047, -2.074, -2.095, -2.120, -2.137, -2.143, -2.146, -2.158],
                [-1.922, -1.977, -2.032, -2.065, -2.091, -2.117, -2.137, -2.142, -2.146, -2.158],
                [-1.750, -1.823, -1.911, -1.968, -2.009, -2.057, -2.091, -2.103, -2.114, -2.135],
                [_NA, _NA, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135],
                [_NA, _NA, _NA, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113],
                [_NA, _NA, _NA, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112],
                [_NA, _NA, _NA, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088],
            ]
        ),
        "var": np.array(
            [
                [1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597],
                [1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605],
                [1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613],
                [2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625],
                [3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629],
                [_NA, _NA, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638],
                [_NA, _NA, _NA, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650],
                [_NA, _NA, _NA, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661],
                [_NA, _NA, _NA, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670],
            ]
        ),
    },
}


def _ips_moments(t_eff: float, lags: int, trend: str) -> tuple[float, float]:
    """Interpolated (E[t], Var[t]) for one entity's effective length and lags.

    The "no deterministics" case is not tabulated by IPS; the intercept
    moments are used as the closest available approximation.
    """
    key = "ct" if trend == "ct" else "c"
    row = min(max(int(lags), 0), 8)
    mean_row = _IPS_MOMENTS[key]["mean"][row]
    var_row = _IPS_MOMENTS[key]["var"][row]
    ok = ~np.isnan(mean_row)
    t = float(np.clip(t_eff, _IPS_T[ok][0], _IPS_T[ok][-1]))
    return float(np.interp(t, _IPS_T[ok], mean_row[ok])), float(
        np.interp(t, _IPS_T[ok], var_row[ok])
    )


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
    individual_stats: dict[Any, float]
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
    >>> ips = pb.IPSTest(data, "invest", "firm", "year")
    >>> result = ips.run()
    >>> print(result)
    >>>
    >>> # With trend
    >>> ips_trend = pb.IPSTest(data, "invest", "firm", "year", trend="ct")
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
        lags: Any | None = None,
        trend: str = "c",
        df_correction: bool = False,
    ):
        self.data = data.copy()
        self.variable = variable
        self.entity_col = entity_col
        self.time_col = time_col
        self.lags = lags
        self.trend = trend
        # plm::purtest computes the individual ADF t-statistics without a
        # degrees-of-freedom correction (dfcor = FALSE); keep that default
        # so results are comparable to the reference implementation
        self.df_correction = df_correction

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

        self.result: IPSTestResult | None = None

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
                logger.debug("AIC computation failed for lag %d", p)
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
            return float(aic)
        except Exception:
            return np.inf

    def _adf_test_entity(self, entity_data: np.ndarray, lags: int) -> tuple[float, int]:
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
                dof = n - len(params) if self.df_correction else n
                sigma2 = np.sum(resid**2) / dof
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
                sigma2 = np.sum(resid**2) / (n - 1 if self.df_correction else n)
                se_rho = np.sqrt(sigma2 / np.sum(y_lag**2))
                t_stat = params[0] / se_rho

                return t_stat, n
            except Exception:
                return np.nan, 0

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
        moments = []

        for entity in self.entities:
            entity_data = self.data[self.data[self.entity_col] == entity][self.variable].values
            lags = lags_dict[entity]

            t_stat, n_obs = self._adf_test_entity(entity_data, lags)

            if not np.isnan(t_stat) and n_obs > 0:
                t_stats.append(t_stat)
                t_stats_dict[entity] = t_stat
                n_obs_total += n_obs
                T_list.append(n_obs)
                moments.append(_ips_moments(n_obs, lags, self.trend))

        if len(t_stats) == 0:
            raise ValueError("Insufficient data for IPS test")

        # Step 3: Compute t-bar
        N = len(t_stats)
        t_bar = np.mean(t_stats)

        # Step 4: Standardize with the IPS (2003, Table 3) moments, averaged
        # over entities so that unbalanced panels and entity-specific lags
        # are handled as in plm::purtest
        E_t = float(np.mean([m[0] for m in moments]))
        Var_t = float(np.mean([m[1] for m in moments]))

        # W-statistic (IPS standardized statistic)
        W_stat = np.sqrt(N) * (t_bar - E_t) / np.sqrt(Var_t)

        # P-value from standard normal
        pvalue = stats.norm.cdf(W_stat)

        # Determine lags for output
        lags_list = [lags_dict[entity] for entity in self.entities if entity in t_stats_dict]
        lags_output: int | list[int]
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
