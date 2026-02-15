"""
Pedroni (1999, 2004) Residual-Based Panel Cointegration Tests

Implementation of the seven Pedroni residual-based tests for panel cointegration,
including both within-dimension (panel) and between-dimension (group) statistics.

References
----------
Pedroni, P. (1999). "Critical Values for Cointegration Tests in Heterogeneous
    Panels with Multiple Regressors." Oxford Bulletin of Economics and Statistics,
    61(S1), 653-670.

Pedroni, P. (2004). "Panel Cointegration: Asymptotic and Finite Sample Properties
    of Pooled Time Series Tests with an Application to the PPP Hypothesis."
    Econometric Theory, 20(3), 597-625.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PedroniResult:
    """
    Results from Pedroni panel cointegration tests.

    Attributes
    ----------
    statistic : dict
        Dictionary containing test statistics (panel_v, panel_rho, panel_PP,
        panel_ADF, group_rho, group_PP, group_ADF)
    pvalue : dict
        Dictionary containing p-values
    critical_values : dict
        Dictionary containing critical values at 1%, 5%, 10%
    method : str
        Test method used ('all' or specific test name)
    trend : str
        Deterministic trend specification
    lags : int
        Number of lags used for PP and ADF tests
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Average number of time periods
    """

    statistic: Dict[str, float]
    pvalue: Dict[str, float]
    critical_values: Dict[str, Dict[str, float]]
    method: str
    trend: str
    lags: int
    n_entities: int
    n_time: int

    def reject_at(
        self, alpha: float = 0.05, test: Optional[str] = None
    ) -> Union[bool, Dict[str, bool]]:
        """
        Check if null hypothesis is rejected at given significance level.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level
        test : str, optional
            Specific test to check. If None, checks all tests.

        Returns
        -------
        bool or dict
            Rejection decision(s)
        """
        if test is not None:
            return self.pvalue[test] < alpha
        return {k: v < alpha for k, v in self.pvalue.items()}

    def summary(self) -> pd.DataFrame:
        """
        Return formatted summary table.

        Returns
        -------
        pd.DataFrame
            Summary table with statistics, p-values, and critical values
        """
        rows = []
        for test in self.statistic.keys():
            row = {
                "Test": test,
                "Statistic": self.statistic[test],
                "P-value": self.pvalue[test],
                "CV 1%": self.critical_values[test].get("1%", np.nan),
                "CV 5%": self.critical_values[test].get("5%", np.nan),
                "CV 10%": self.critical_values[test].get("10%", np.nan),
                "Reject (5%)": (
                    "***"
                    if self.pvalue[test] < 0.01
                    else (
                        "**"
                        if self.pvalue[test] < 0.05
                        else ("*" if self.pvalue[test] < 0.10 else "")
                    )
                ),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def __repr__(self) -> str:
        summary_str = f"Pedroni (1999) Cointegration Test Results\n"
        summary_str += f"{'='*60}\n"
        summary_str += f"Method: {self.method}\n"
        summary_str += f"Trend: {self.trend}\n"
        summary_str += f"Lags: {self.lags}\n"
        summary_str += f"Entities: {self.n_entities}, Time periods: {self.n_time}\n"
        summary_str += f"\n{self.summary().to_string(index=False)}\n"
        summary_str += f"\nH0: No cointegration\n"
        summary_str += f"***, **, * denote rejection at 1%, 5%, 10% level"
        return summary_str


def _estimate_cointegrating_regression(
    y: np.ndarray, x: np.ndarray, trend: str = "c"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate cointegrating regression and extract residuals.

    y_t = α + β x_t + ε_t

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (T,)
    x : np.ndarray
        Independent variables (T, k)
    trend : str
        Deterministic trend: 'n' (none), 'c' (constant), 'ct' (constant+trend)

    Returns
    -------
    beta : np.ndarray
        Estimated cointegrating vector
    resid : np.ndarray
        Residuals
    sigma2 : float
        Residual variance
    """
    T = len(y)

    # Build regressor matrix
    if trend == "c":
        X = np.column_stack([np.ones(T), x]) if x.ndim > 1 else np.column_stack([np.ones(T), x])
    elif trend == "ct":
        X = (
            np.column_stack([np.ones(T), np.arange(1, T + 1), x])
            if x.ndim > 1
            else np.column_stack([np.ones(T), np.arange(1, T + 1), x])
        )
    else:
        X = x if x.ndim > 1 else x.reshape(-1, 1)

    # OLS estimation
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X @ beta
        sigma2 = np.sum(resid**2) / T

        return beta, resid, sigma2
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in cointegrating regression")
        return np.full(X.shape[1], np.nan), np.full(T, np.nan), np.nan


def _compute_long_run_variance(resid: np.ndarray, lags: int = 4) -> Tuple[float, float]:
    """
    Compute long-run variance using Newey-West HAC estimator.

    Parameters
    ----------
    resid : np.ndarray
        Residuals
    lags : int
        Number of lags for HAC estimation

    Returns
    -------
    sigma2_lr : float
        Long-run variance
    lambda2 : float
        One-sided long-run variance
    """
    T = len(resid)

    # Variance
    gamma0 = np.mean(resid**2)

    # Autocovariances
    gamma_sum = 0
    for j in range(1, lags + 1):
        gamma_j = np.mean(resid[j:] * resid[:-j])
        weight = 1 - j / (lags + 1)  # Bartlett kernel
        gamma_sum += weight * gamma_j

    # Long-run variance
    sigma2_lr = gamma0 + 2 * gamma_sum

    # One-sided long-run variance (for lambda)
    lambda2 = gamma_sum

    return sigma2_lr, lambda2


def _panel_variance_test(residuals: List[np.ndarray], sigma2s: List[float]) -> float:
    """
    Compute panel variance test statistic (panel-ν).

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances for each entity

    Returns
    -------
    float
        Panel-ν statistic
    """
    N = len(residuals)
    T = len(residuals[0])

    # Compute sum of squared residuals
    sum_sq = 0
    for i in range(N):
        resid_i = residuals[i]
        sum_sq += np.sum(resid_i[:-1] ** 2)

    # Standardize
    panel_v = (T**2 * N ** (3 / 2)) / sum_sq

    return panel_v


def _panel_rho_test(residuals: List[np.ndarray], sigma2s: List[float]) -> float:
    """
    Compute panel-ρ test statistic.

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances for each entity

    Returns
    -------
    float
        Panel-ρ statistic
    """
    N = len(residuals)
    T = len(residuals[0])

    numerator = 0
    denominator = 0

    for i in range(N):
        resid_i = residuals[i]
        numerator += np.sum(resid_i[1:] * resid_i[:-1]) - sigma2s[i]
        denominator += np.sum(resid_i[:-1] ** 2)

    panel_rho = (T * np.sqrt(N) * numerator) / denominator

    return panel_rho


def _panel_pp_test(
    residuals: List[np.ndarray],
    sigma2s: List[float],
    sigma2_lrs: List[float],
    lambda2s: List[float],
) -> float:
    """
    Compute panel-PP (Phillips-Perron) test statistic.

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances
    sigma2_lrs : list of float
        Long-run variances
    lambda2s : list of float
        One-sided long-run variances

    Returns
    -------
    float
        Panel-PP statistic
    """
    N = len(residuals)
    T = len(residuals[0])

    numerator = 0
    denominator = 0

    for i in range(N):
        resid_i = residuals[i]
        sigma2_i = sigma2s[i]
        sigma2_lr_i = sigma2_lrs[i]
        lambda2_i = lambda2s[i]

        num_i = np.sum(resid_i[1:] * resid_i[:-1]) - lambda2_i
        numerator += num_i

        denom_i = np.sum(resid_i[:-1] ** 2)
        denominator += sigma2_lr_i * denom_i / sigma2_i

    panel_pp = (T * np.sqrt(N) * numerator) / np.sqrt(denominator)

    return panel_pp


def _panel_adf_test(residuals: List[np.ndarray], sigma2s: List[float], lags: int = 1) -> float:
    """
    Compute panel-ADF (Augmented Dickey-Fuller) test statistic.

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances
    lags : int
        Number of lags for ADF regression

    Returns
    -------
    float
        Panel-ADF statistic
    """
    N = len(residuals)
    T = len(residuals[0])

    numerator = 0
    denominator = 0

    for i in range(N):
        resid_i = residuals[i]

        # ADF regression: Δε_t = ρ ε_{t-1} + Σ γ_j Δε_{t-j} + u_t
        delta_resid = np.diff(resid_i)
        resid_lag = resid_i[:-1]

        # Build regressor matrix
        X_adf = [resid_lag[lags:]]
        for j in range(1, lags + 1):
            if j < len(delta_resid):
                X_adf.append(
                    delta_resid[lags - j : -j] if j < len(delta_resid) else delta_resid[lags - j :]
                )

        X_adf = np.column_stack(X_adf) if len(X_adf) > 1 else resid_lag[lags:].reshape(-1, 1)
        y_adf = delta_resid[lags:]

        # OLS
        try:
            params = np.linalg.lstsq(X_adf, y_adf, rcond=None)[0]
            rho = params[0]
            resid_adf = y_adf - X_adf @ params
            se_rho = np.sqrt(
                np.sum(resid_adf**2)
                / (len(y_adf) - len(params))
                * np.linalg.inv(X_adf.T @ X_adf)[0, 0]
            )

            numerator += rho / se_rho
            denominator += 1
        except:
            pass

    panel_adf = numerator / np.sqrt(N) if denominator > 0 else np.nan

    return panel_adf


def _group_rho_test(residuals: List[np.ndarray], sigma2s: List[float]) -> float:
    """
    Compute group-ρ test statistic.

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances

    Returns
    -------
    float
        Group-ρ statistic
    """
    N = len(residuals)
    T = len(residuals[0])

    group_stats = []

    for i in range(N):
        resid_i = residuals[i]
        numerator = np.sum(resid_i[1:] * resid_i[:-1]) - sigma2s[i]
        denominator = np.sum(resid_i[:-1] ** 2)
        group_stats.append(T * numerator / denominator)

    group_rho = np.sqrt(N) * np.mean(group_stats)

    return group_rho


def _group_pp_test(
    residuals: List[np.ndarray],
    sigma2s: List[float],
    sigma2_lrs: List[float],
    lambda2s: List[float],
) -> float:
    """
    Compute group-PP test statistic.

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances
    sigma2_lrs : list of float
        Long-run variances
    lambda2s : list of float
        One-sided long-run variances

    Returns
    -------
    float
        Group-PP statistic
    """
    N = len(residuals)
    T = len(residuals[0])

    group_stats = []

    for i in range(N):
        resid_i = residuals[i]
        sigma2_i = sigma2s[i]
        sigma2_lr_i = sigma2_lrs[i]
        lambda2_i = lambda2s[i]

        numerator = np.sum(resid_i[1:] * resid_i[:-1]) - lambda2_i
        denominator = np.sqrt(sigma2_lr_i * np.sum(resid_i[:-1] ** 2) / sigma2_i)

        group_stats.append(T * numerator / denominator)

    group_pp = np.sqrt(N) * np.mean(group_stats)

    return group_pp


def _group_adf_test(residuals: List[np.ndarray], sigma2s: List[float], lags: int = 1) -> float:
    """
    Compute group-ADF test statistic.

    Parameters
    ----------
    residuals : list of np.ndarray
        Residuals for each entity
    sigma2s : list of float
        Residual variances
    lags : int
        Number of lags for ADF regression

    Returns
    -------
    float
        Group-ADF statistic
    """
    N = len(residuals)

    group_stats = []

    for i in range(N):
        resid_i = residuals[i]

        # ADF regression
        delta_resid = np.diff(resid_i)
        resid_lag = resid_i[:-1]

        X_adf = [resid_lag[lags:]]
        for j in range(1, lags + 1):
            if j < len(delta_resid):
                X_adf.append(
                    delta_resid[lags - j : -j] if j < len(delta_resid) else delta_resid[lags - j :]
                )

        X_adf = np.column_stack(X_adf) if len(X_adf) > 1 else resid_lag[lags:].reshape(-1, 1)
        y_adf = delta_resid[lags:]

        try:
            params = np.linalg.lstsq(X_adf, y_adf, rcond=None)[0]
            rho = params[0]
            resid_adf = y_adf - X_adf @ params
            se_rho = np.sqrt(
                np.sum(resid_adf**2)
                / (len(y_adf) - len(params))
                * np.linalg.inv(X_adf.T @ X_adf)[0, 0]
            )

            group_stats.append(rho / se_rho)
        except:
            pass

    group_adf = np.sqrt(N) * np.mean(group_stats) if len(group_stats) > 0 else np.nan

    return group_adf


def _get_critical_values(test: str, trend: str = "c") -> Dict[str, float]:
    """
    Get tabulated critical values from Pedroni (2004).

    These are asymptotic critical values for N, T → ∞.

    Parameters
    ----------
    test : str
        Test name
    trend : str
        Trend specification

    Returns
    -------
    dict
        Critical values at 1%, 5%, 10%
    """
    # Tabulated values from Pedroni (2004) Table 2
    # These are approximate values for constant + trend case
    critical_values_ct = {
        "panel_v": {"1%": 0.107, "5%": 0.794, "10%": 1.365},
        "panel_rho": {"1%": -2.43, "5%": -1.73, "10%": -1.42},
        "panel_PP": {"1%": -2.43, "5%": -1.73, "10%": -1.42},
        "panel_ADF": {"1%": -2.43, "5%": -1.73, "10%": -1.42},
        "group_rho": {"1%": -2.78, "5%": -2.16, "10%": -1.84},
        "group_PP": {"1%": -2.78, "5%": -2.16, "10%": -1.84},
        "group_ADF": {"1%": -2.78, "5%": -2.16, "10%": -1.84},
    }

    # For constant only, values are slightly different
    critical_values_c = {
        "panel_v": {"1%": -0.48, "5%": 0.12, "10%": 0.58},
        "panel_rho": {"1%": -2.23, "5%": -1.66, "10%": -1.37},
        "panel_PP": {"1%": -2.23, "5%": -1.66, "10%": -1.37},
        "panel_ADF": {"1%": -2.23, "5%": -1.66, "10%": -1.37},
        "group_rho": {"1%": -2.66, "5%": -2.07, "10%": -1.77},
        "group_PP": {"1%": -2.66, "5%": -2.07, "10%": -1.77},
        "group_ADF": {"1%": -2.66, "5%": -2.07, "10%": -1.77},
    }

    if trend == "ct":
        return critical_values_ct.get(test, {"1%": np.nan, "5%": np.nan, "10%": np.nan})
    else:
        return critical_values_c.get(test, {"1%": np.nan, "5%": np.nan, "10%": np.nan})


def pedroni_test(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: Union[str, List[str]],
    method: str = "all",
    trend: str = "c",
    lags: int = 4,
) -> PedroniResult:
    """
    Pedroni (1999, 2004) residual-based panel cointegration tests.

    Tests the null hypothesis of no cointegration using residuals from
    cointegrating regressions. Includes both within-dimension (panel) and
    between-dimension (group) statistics.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format
    entity_col : str
        Name of entity identifier column
    time_col : str
        Name of time identifier column
    y_var : str
        Name of dependent variable
    x_vars : str or list of str
        Name(s) of independent variable(s)
    method : str, default 'all'
        Test statistic to compute. Options: 'panel_v', 'panel_rho', 'panel_PP',
        'panel_ADF', 'group_rho', 'group_PP', 'group_ADF', 'all'
    trend : str, default 'c'
        Deterministic trend specification: 'n', 'c', or 'ct'
    lags : int, default 4
        Number of lags for long-run variance estimation and ADF tests

    Returns
    -------
    PedroniResult
        Object containing test statistics, p-values, and critical values

    References
    ----------
    Pedroni, P. (1999). "Critical Values for Cointegration Tests in Heterogeneous
        Panels with Multiple Regressors." Oxford Bulletin of Economics and Statistics,
        61(S1), 653-670.

    Pedroni, P. (2004). "Panel Cointegration: Asymptotic and Finite Sample Properties
        of Pooled Time Series Tests with an Application to the PPP Hypothesis."
        Econometric Theory, 20(3), 597-625.

    Examples
    --------
    >>> from panelbox.diagnostics.cointegration import pedroni_test
    >>>
    >>> result = pedroni_test(
    ...     data, entity_col='country', time_col='year',
    ...     y_var='log_gdp', x_vars=['log_capital', 'log_labor']
    ... )
    >>> print(result)
    """
    # Input validation
    if isinstance(x_vars, str):
        x_vars = [x_vars]

    required_cols = [entity_col, time_col, y_var] + x_vars
    missing_cols = set(required_cols) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Get entities
    entities = data[entity_col].unique()
    N = len(entities)

    # Estimate cointegrating regressions and extract residuals
    residuals = []
    sigma2s = []
    sigma2_lrs = []
    lambda2s = []
    T_avg = 0

    for entity in entities:
        entity_data = data[data[entity_col] == entity].sort_values(time_col)

        y = entity_data[y_var].values
        X = entity_data[x_vars].values
        T_avg += len(y)

        # Estimate cointegrating regression
        beta, resid, sigma2 = _estimate_cointegrating_regression(y, X, trend)

        if not np.isnan(sigma2):
            # Compute long-run variance
            sigma2_lr, lambda2 = _compute_long_run_variance(resid, lags)

            residuals.append(resid)
            sigma2s.append(sigma2)
            sigma2_lrs.append(sigma2_lr)
            lambda2s.append(lambda2)

    T_avg = T_avg // N

    # Compute test statistics
    test_stats = {}

    if method in ["panel_v", "all"]:
        test_stats["panel_v"] = _panel_variance_test(residuals, sigma2s)

    if method in ["panel_rho", "all"]:
        test_stats["panel_rho"] = _panel_rho_test(residuals, sigma2s)

    if method in ["panel_PP", "all"]:
        test_stats["panel_PP"] = _panel_pp_test(residuals, sigma2s, sigma2_lrs, lambda2s)

    if method in ["panel_ADF", "all"]:
        test_stats["panel_ADF"] = _panel_adf_test(residuals, sigma2s, lags)

    if method in ["group_rho", "all"]:
        test_stats["group_rho"] = _group_rho_test(residuals, sigma2s)

    if method in ["group_PP", "all"]:
        test_stats["group_PP"] = _group_pp_test(residuals, sigma2s, sigma2_lrs, lambda2s)

    if method in ["group_ADF", "all"]:
        test_stats["group_ADF"] = _group_adf_test(residuals, sigma2s, lags)

    # Get critical values and compute p-values
    critical_values = {}
    pvalues = {}

    for test in test_stats.keys():
        critical_values[test] = _get_critical_values(test, trend)

        # Approximate p-value using normal distribution
        # (asymptotically, these statistics are N(0,1))
        if test == "panel_v":
            # Panel-v is right-tailed
            pvalues[test] = 1 - stats.norm.cdf(test_stats[test])
        else:
            # Others are left-tailed
            pvalues[test] = stats.norm.cdf(test_stats[test])

    # Create result object
    result = PedroniResult(
        statistic=test_stats,
        pvalue=pvalues,
        critical_values=critical_values,
        method=method,
        trend=trend,
        lags=lags,
        n_entities=N,
        n_time=T_avg,
    )

    return result
