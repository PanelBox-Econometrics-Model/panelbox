"""
Levin-Lin-Chu (LLC) panel unit root test.

This module implements the Levin-Lin-Chu test for unit roots in panel data.
The test assumes a common unit root process across all cross-sections.

Reference:
    Levin, A., Lin, C. F., & Chu, C. S. J. (2002). Unit root tests in panel
    data: asymptotic and finite-sample properties. Journal of econometrics,
    108(1), 1-24.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Mean and standard deviation adjustments for the LLC t-statistic
# (Levin, Lin and Chu 2002, Table 2), indexed by the effective time
# dimension T~ = T - lags - 1. Values are linearly interpolated in between.
_LLC_ADJ_T = np.array([25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 250, 500])
_LLC_ADJ = {
    "n": (
        np.array([0.004, 0.003, 0.002, 0.002, 0.001, 0.001, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array(
            [
                1.049,
                1.035,
                1.027,
                1.021,
                1.017,
                1.014,
                1.011,
                1.008,
                1.007,
                1.006,
                1.005,
                1.001,
                1.0,
            ]
        ),
    ),
    "c": (
        np.array(
            [
                -0.554,
                -0.546,
                -0.541,
                -0.537,
                -0.533,
                -0.531,
                -0.527,
                -0.524,
                -0.521,
                -0.520,
                -0.518,
                -0.509,
                -0.500,
            ]
        ),
        np.array(
            [
                0.919,
                0.889,
                0.867,
                0.850,
                0.837,
                0.826,
                0.810,
                0.798,
                0.789,
                0.782,
                0.776,
                0.742,
                0.707,
            ]
        ),
    ),
    "ct": (
        np.array(
            [
                -0.703,
                -0.674,
                -0.653,
                -0.637,
                -0.624,
                -0.614,
                -0.598,
                -0.587,
                -0.578,
                -0.571,
                -0.566,
                -0.533,
                -0.500,
            ]
        ),
        np.array(
            [
                1.003,
                0.949,
                0.906,
                0.871,
                0.842,
                0.818,
                0.780,
                0.751,
                0.728,
                0.710,
                0.695,
                0.603,
                0.500,
            ]
        ),
    ),
}


def _llc_adjustment(t_tilde: float, trend: str) -> tuple[float, float]:
    """Interpolated (mu*, sigma*) for the effective time dimension."""
    mu, sig = _LLC_ADJ[trend]
    t = float(np.clip(t_tilde, _LLC_ADJ_T[0], _LLC_ADJ_T[-1]))
    return float(np.interp(t, _LLC_ADJ_T, mu)), float(np.interp(t, _LLC_ADJ_T, sig))


def _long_run_variance(y: np.ndarray, trend: str, q: int | None = None) -> float:
    """Bartlett-kernel long-run variance of the first difference of ``y``.

    Truncation lag defaults to round(3.21 * T ** (1/3)) as in LLC (2002).
    """
    T = len(y)
    if q is None:
        q = round(3.21 * T ** (1.0 / 3.0))
    dx = np.diff(y)
    if trend == "c":
        dx = dx - dx.mean()
    elif trend == "ct":
        X = np.column_stack([np.ones(len(dx)), np.arange(1, len(dx) + 1)])
        dx = dx - X @ np.linalg.lstsq(X, dx, rcond=None)[0]
    n = len(dx)
    lrv = np.sum(dx**2) / n
    for lag in range(1, min(q, n - 1) + 1):
        lrv += 2.0 * (1.0 - lag / (q + 1.0)) * np.sum(dx[:-lag] * dx[lag:]) / n
    return float(lrv)


@dataclass
class LLCTestResult:
    """
    Results from LLC panel unit root test.

    Attributes
    ----------
    statistic : float
        LLC test statistic (adjusted t-statistic)
    pvalue : float
        P-value for the test
    lags : int
        Number of lags used
    n_obs : int
        Number of observations used
    n_entities : int
        Number of cross-sections
    test_type : str
        Type of test specification
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
    pvalue: float
    lags: int
    n_obs: int
    n_entities: int
    test_type: str
    deterministics: str
    null_hypothesis: str = "All panels contain unit roots"
    alternative_hypothesis: str = "All panels are stationary"

    @property
    def conclusion(self) -> str:
        """Conclusion at 5% significance level."""
        if self.pvalue < 0.05:
            return "Reject H0: Evidence against unit root (panels are stationary)"
        else:
            return "Fail to reject H0: Evidence of unit root"

    def __str__(self) -> str:
        """String representation."""
        lines = []
        lines.append("=" * 70)
        lines.append("Levin-Lin-Chu Panel Unit Root Test")
        lines.append("=" * 70)
        lines.append(f"Test statistic:    {self.statistic:.4f}")
        lines.append(f"P-value:           {self.pvalue:.4f}")
        lines.append(f"Lags:              {self.lags}")
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


class LLCTest:
    """
    Levin-Lin-Chu panel unit root test.

    The LLC test assumes a common unit root process across all panels:
    Δy_it = ρy_{i,t-1} + Σ_j θ_ij Δy_{i,t-j} + α_i + δ_i t + ε_it

    H0: ρ = 0 (all panels have unit root)
    H1: ρ < 0 (all panels are stationary)

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
    lags : int, optional
        Number of lags for augmented term. If None, selected automatically
    trend : str, default='c'
        Deterministic terms to include:
        - 'n': No deterministic terms
        - 'c': Constant only
        - 'ct': Constant and trend

    Attributes
    ----------
    result : LLCTestResult
        Test results after running

    Examples
    --------
    >>> import panelbox as pb
    >>> data = pb.load_grunfeld()
    >>>
    >>> # Test for unit root in 'invest'
    >>> llc = pb.LLCTest(data, "invest", "firm", "year")
    >>> result = llc.run()
    >>> print(result)
    >>>
    >>> # With trend
    >>> llc_trend = pb.LLCTest(data, "invest", "firm", "year", trend="ct")
    >>> result = llc_trend.run()

    Notes
    -----
    The LLC test requires a balanced panel. If the panel is unbalanced,
    observations will be dropped to balance it.

    The test assumes homogeneity in the autoregressive coefficient ρ
    across all panels, which may be restrictive. Consider using the IPS
    test if heterogeneity is suspected.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variable: str,
        entity_col: str,
        time_col: str,
        lags: int | None = None,
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

        # Check balance
        obs_per_entity = self.data.groupby(entity_col).size()
        self.is_balanced = (obs_per_entity == obs_per_entity.iloc[0]).all()

        if not self.is_balanced:
            import warnings

            warnings.warn(
                "Panel is unbalanced. LLC test requires balanced panel. "
                "Consider using IPS test for unbalanced panels.",
                UserWarning,
                stacklevel=2,
            )

        self.result: LLCTestResult | None = None

    def _select_lags(self) -> int:
        """
        Automatic lag selection using AIC.

        Returns
        -------
        int
            Optimal number of lags
        """
        # Use simple rule: T^(1/3) as maximum lag
        T = len(self.data) // self.n_entities
        max_lags = int(np.floor((T / 100) ** (1 / 3) * 12))
        max_lags = min(max_lags, T // 4)  # At most T/4 lags

        if max_lags < 1:
            return 0

        # Use AIC to select optimal lag
        best_lag = 0
        best_aic = np.inf

        for p in range(max_lags + 1):
            try:
                aic = self._compute_aic(p)
                if aic < best_aic:
                    best_aic = aic
                    best_lag = p
            except Exception:
                logger.debug("AIC computation failed for lag %d", p)
                continue

        return best_lag

    def _compute_aic(self, lags: int) -> float:
        """Compute AIC for given lag order."""
        # Simplified AIC computation
        residuals: list[float] = []

        for entity in self.entities:
            entity_data = self.data[self.data[self.entity_col] == entity][self.variable].values

            if len(entity_data) < lags + 2:
                continue

            # Construct regression for this entity
            y = np.diff(entity_data[lags:])
            X_parts = [entity_data[lags:-1]]  # Lagged level

            # Add lagged differences
            for j in range(1, lags + 1):
                X_parts.append(
                    np.diff(
                        entity_data[lags - j : -j - 1]
                        if j < len(entity_data) - 1
                        else entity_data[lags - j :]
                    )
                )

            # Add intercept if needed
            if self.trend in ["c", "ct"]:
                X_parts.append(np.ones(len(y)))

            # Add trend if needed
            if self.trend == "ct":
                X_parts.append(np.arange(len(y)))

            try:
                X = np.column_stack(X_parts)
                params = np.linalg.lstsq(X, y, rcond=None)[0]
                resid = y - X @ params
                residuals.extend(resid)
            except Exception:
                logger.debug("Regression failed for entity in AIC computation")
                continue

        if len(residuals) == 0:
            return np.inf

        residuals_arr: np.ndarray = np.array(residuals)
        n = len(residuals_arr)
        k = lags + 2  # parameters

        sigma2 = np.sum(residuals_arr**2) / n
        aic = n * np.log(sigma2) + 2 * k

        return float(aic)

    def _demean_data(self, X: np.ndarray, method: str = "within") -> np.ndarray:
        """
        Demean data by entity.

        Parameters
        ----------
        X : np.ndarray
            Data to demean
        method : str
            Method to use ('within' for entity means)

        Returns
        -------
        np.ndarray
            Demeaned data
        """
        X_demeaned = np.zeros_like(X)
        start_idx = 0

        for entity in self.entities:
            entity_size = len(self.data[self.data[self.entity_col] == entity])
            end_idx = start_idx + entity_size

            entity_data = X[start_idx:end_idx]
            entity_mean = entity_data.mean(axis=0)
            X_demeaned[start_idx:end_idx] = entity_data - entity_mean

            start_idx = end_idx

        return X_demeaned

    def _process_entity(
        self,
        entity,
    ) -> tuple[np.ndarray, np.ndarray, float, int, float] | None:
        """Process a single entity for the LLC test.

        Returns (e_tilde, v_tilde, sigma_i, T_i, s_i) or None if entity is skipped.
        """
        entity_data = self.data[self.data[self.entity_col] == entity][self.variable].values

        if len(entity_data) < self.lags + 3:
            return None

        dy = np.diff(entity_data)
        T_i = len(dy) - self.lags
        y_lag = entity_data[self.lags : -1]
        dy_dep = dy[self.lags :]

        Z = self._build_regressors(dy, T_i)

        if len(y_lag) != T_i or len(dy_dep) != T_i:
            return None
        out = self._orthogonalize(Z, dy_dep, y_lag, T_i)
        if out is None:
            return None
        e_tilde, v_tilde, sigma_i, T_i = out
        s_i = np.sqrt(_long_run_variance(entity_data, self.trend)) / sigma_i
        return (e_tilde, v_tilde, sigma_i, T_i, s_i)

    def _build_regressors(self, dy: np.ndarray, T_i: int) -> list[np.ndarray]:
        """Build regressor matrix (lagged differences + deterministics)."""
        Z = []
        for j in range(1, self.lags + 1):
            if self.lags - j >= 0 and len(dy) > self.lags:
                lag_idx_start = self.lags - j
                lag_idx_end = len(dy) - j if j > 0 else len(dy)
                if lag_idx_end > lag_idx_start:
                    dy_lag = dy[lag_idx_start:lag_idx_end]
                    if len(dy_lag) == T_i:
                        Z.append(dy_lag)

        if self.trend == "c":
            Z.append(np.ones(T_i))
        elif self.trend == "ct":
            Z.append(np.ones(T_i))
            Z.append(np.arange(T_i))

        return Z

    def _orthogonalize(
        self,
        Z: list[np.ndarray],
        dy_dep: np.ndarray,
        y_lag: np.ndarray,
        T_i: int,
    ) -> tuple[np.ndarray, float, float, int] | None:
        """Orthogonalize Δy and y_{t-1} w.r.t. regressors Z.

        ``sigma_i`` is the residual standard deviation of the full ADF
        regression Δy on (y_{t-1}, Z), without degrees-of-freedom correction.
        """
        try:
            if len(Z) > 0:
                Z_mat = np.column_stack(Z)
                e_tilde = dy_dep - Z_mat @ np.linalg.lstsq(Z_mat, dy_dep, rcond=None)[0]
                v_tilde = y_lag - Z_mat @ np.linalg.lstsq(Z_mat, y_lag, rcond=None)[0]
                X_full = np.column_stack([y_lag, Z_mat])
            else:
                e_tilde, v_tilde = dy_dep, y_lag
                X_full = y_lag.reshape(-1, 1)
            resid = dy_dep - X_full @ np.linalg.lstsq(X_full, dy_dep, rcond=None)[0]
            sigma_i = float(np.sqrt(np.sum(resid**2) / len(resid)))
            # A (numerically) perfect fit means the entity carries no
            # information for the pooled regression; skip it
            scale = max(1.0, float(np.max(np.abs(dy_dep))))
            if sigma_i > 1e-10 * scale and len(e_tilde) > 0:
                return (e_tilde, v_tilde, sigma_i, T_i)
        except Exception:
            logger.debug("Orthogonalization failed for entity")
        return None

    def _pooled_regression(
        self,
        e_tilde_list: list[np.ndarray],
        v_tilde_list: list[np.ndarray],
        sigma_list: list[float],
        T_list: list[int],
        s_list: list[float],
    ) -> LLCTestResult:
        """Pool the normalized series and compute the adjusted LLC statistic.

        Follows Levin, Lin and Chu (2002), eq. (12):

            t* = (t_rho - N T~ S_N sigma_eps^-2 se(rho) mu*_T~) / sigma*_T~

        where S_N is the average ratio of long-run to short-run standard
        deviations and (mu*, sigma*) come from Table 2. For unbalanced
        panels T~ is the average effective length.
        """
        e_pooled = np.concatenate([e / s for e, s in zip(e_tilde_list, sigma_list)])
        v_pooled = np.concatenate([v / s for v, s in zip(v_tilde_list, sigma_list)])

        rho = np.sum(e_pooled * v_pooled) / np.sum(v_pooled**2)
        resid = e_pooled - rho * v_pooled
        n_total = len(e_pooled)
        rss = float(np.sum(resid**2))
        se_rho = np.sqrt((rss / n_total) / np.sum(v_pooled**2))
        t_rho = rho / se_rho

        n_entities = len(e_tilde_list)
        t_tilde = float(np.mean(T_list))
        sigma_eps2 = rss / (n_entities * t_tilde)
        s_bar = float(np.mean(s_list))
        mu_star, sigma_star = _llc_adjustment(t_tilde, self.trend)

        t_adj = (t_rho - n_entities * t_tilde * s_bar / sigma_eps2 * se_rho * mu_star) / sigma_star
        pvalue = stats.norm.cdf(t_adj)

        det_map = {"n": "None", "c": "Constant", "ct": "Constant and Trend"}

        self.result = LLCTestResult(
            statistic=float(t_adj),
            pvalue=float(pvalue),
            lags=self.lags,
            n_obs=n_total,
            n_entities=n_entities,
            test_type="LLC",
            deterministics=det_map[self.trend],
        )
        return self.result

    def run(self) -> LLCTestResult:
        """
        Run the LLC panel unit root test.

        Returns
        -------
        LLCTestResult
            Test results

        Notes
        -----
        The test procedure follows LLC (2002):
        1. For each panel, run regression: Δy_it = α_i + ρy_{i,t-1} + Σθ_ijΔy_{i,t-j} + ε_it
        2. Orthogonalize Δy and y_{t-1} with respect to lags and deterministics
        3. Normalize the orthogonalized series by long-run standard deviation
        4. Pool and run regression to get t-statistic
        5. Adjust using mean and std from asymptotic distribution
        """
        if self.lags is None:
            self.lags = self._select_lags()

        e_tilde_list = []
        v_tilde_list = []
        sigma_list = []
        T_list = []
        s_list = []

        for entity in self.entities:
            result = self._process_entity(entity)
            if result is not None:
                e_tilde, v_tilde, sigma_i, T_i, s_i = result
                e_tilde_list.append(e_tilde)
                v_tilde_list.append(v_tilde)
                sigma_list.append(sigma_i)
                T_list.append(T_i)
                s_list.append(s_i)

        if len(e_tilde_list) == 0:
            raise ValueError("Insufficient data for LLC test")

        return self._pooled_regression(e_tilde_list, v_tilde_list, sigma_list, T_list, s_list)
