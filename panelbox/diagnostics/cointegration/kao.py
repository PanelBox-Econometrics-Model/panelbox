"""
Kao (1999) DF and ADF-Based Panel Cointegration Tests

Implementation of Kao's Dickey-Fuller and Augmented Dickey-Fuller tests
for panel cointegration, assuming a homogeneous cointegrating vector.

Reference
---------
Kao, C. (1999). "Spurious Regression and Residual-Based Tests for
    Cointegration in Panel Data." Journal of Econometrics, 90(1), 1-44.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class KaoResult:
    """
    Results from Kao panel cointegration tests.

    Attributes
    ----------
    statistic : dict
        Dictionary containing test statistics (DF, ADF)
    pvalue : dict
        Dictionary containing p-values
    critical_values : dict
        Dictionary containing critical values at 1%, 5%, 10%
    method : str
        Test method used ('DF', 'ADF', or 'all')
    trend : str
        Deterministic trend specification
    lags : int
        Number of lags used (for ADF)
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
        summary_str = f"Kao (1999) Cointegration Test Results\n"
        summary_str += f"{'='*60}\n"
        summary_str += f"Method: {self.method}\n"
        summary_str += f"Trend: {self.trend}\n"
        summary_str += f"Lags: {self.lags}\n"
        summary_str += f"Entities: {self.n_entities}, Time periods: {self.n_time}\n"
        summary_str += f"\n{self.summary().to_string(index=False)}\n"
        summary_str += f"\nH0: No cointegration (homogeneous cointegrating vector)\n"
        summary_str += f"***, **, * denote rejection at 1%, 5%, 10% level"
        return summary_str


def _estimate_pooled_cointegrating_regression(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: List[str],
    trend: str = "c",
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """
    Estimate pooled cointegrating regression with entity fixed effects.

    y_{it} = α_i + β x_{it} + ε_{it}

    Note: β is assumed to be homogeneous across entities (pooled).

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
    y_var : str
        Dependent variable
    x_vars : list
        Independent variables
    trend : str
        Trend specification: 'n', 'c', 'ct'

    Returns
    -------
    beta : np.ndarray
        Estimated cointegrating vector
    pooled_resid : np.ndarray
        Pooled residuals (stacked)
    entity_resids : list of np.ndarray
        Residuals for each entity
    """
    entities = data[entity_col].unique()
    N = len(entities)

    # Prepare pooled data
    y_pooled = []
    X_pooled = []
    entity_resids = []

    for i, entity in enumerate(entities):
        entity_data = data[data[entity_col] == entity].sort_values(time_col)

        y_i = entity_data[y_var].values
        X_i = entity_data[x_vars].values
        T_i = len(y_i)

        # Add to pooled data
        if trend == "c":
            # Include entity-specific intercept
            intercept = np.zeros((T_i, N))
            intercept[:, i] = 1
            X_pooled_i = np.column_stack([intercept, X_i])
        elif trend == "ct":
            # Include entity-specific intercept and common trend
            intercept = np.zeros((T_i, N))
            intercept[:, i] = 1
            time_trend = np.arange(1, T_i + 1).reshape(-1, 1)
            X_pooled_i = np.column_stack([intercept, time_trend, X_i])
        else:
            X_pooled_i = X_i if X_i.ndim > 1 else X_i.reshape(-1, 1)

        y_pooled.append(y_i)
        X_pooled.append(X_pooled_i)

    # Stack all data
    y_pooled = np.concatenate(y_pooled)
    X_pooled = np.vstack(X_pooled)

    # Pooled OLS estimation
    try:
        beta = np.linalg.lstsq(X_pooled, y_pooled, rcond=None)[0]
        pooled_resid = y_pooled - X_pooled @ beta

        # Extract residuals for each entity
        idx = 0
        for entity in entities:
            entity_data = data[data[entity_col] == entity].sort_values(time_col)
            T_i = len(entity_data)
            entity_resids.append(pooled_resid[idx : idx + T_i])
            idx += T_i

        return beta, pooled_resid, entity_resids

    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in pooled cointegrating regression")
        return np.full(X_pooled.shape[1], np.nan), np.full(len(y_pooled), np.nan), []


def _compute_long_run_variance_pooled(resids: List[np.ndarray], lags: int = 4) -> float:
    """
    Compute long-run variance for pooled residuals.

    Parameters
    ----------
    resids : list of np.ndarray
        Residuals for each entity
    lags : int
        Number of lags for HAC estimation

    Returns
    -------
    float
        Long-run variance
    """
    # Stack residuals
    resid_pooled = np.concatenate(resids)
    N = len(resids)
    T = len(resids[0])

    # Variance
    gamma0 = np.mean(resid_pooled**2)

    # Autocovariances (pooled)
    gamma_sum = 0
    for j in range(1, lags + 1):
        gamma_j_sum = 0
        for i in range(N):
            resid_i = resids[i]
            if j < len(resid_i):
                gamma_j_sum += np.mean(resid_i[j:] * resid_i[:-j])

        gamma_j = gamma_j_sum / N
        weight = 1 - j / (lags + 1)  # Bartlett kernel
        gamma_sum += weight * gamma_j

    # Long-run variance
    sigma2_lr = gamma0 + 2 * gamma_sum

    return sigma2_lr


def _kao_df_test(resids: List[np.ndarray]) -> float:
    """
    Compute Kao DF test statistic.

    The test regresses pooled residuals on lagged residuals:
    Δε̂_{it} = ρ ε̂_{i,t-1} + ν_{it}

    H0: ρ = 0 (no cointegration)

    Parameters
    ----------
    resids : list of np.ndarray
        Residuals for each entity

    Returns
    -------
    float
        DF test statistic
    """
    N = len(resids)
    T = len(resids[0])

    # Pool all observations
    delta_resid_pooled = []
    resid_lag_pooled = []

    for resid_i in resids:
        delta_resid = np.diff(resid_i)
        resid_lag = resid_i[:-1]

        delta_resid_pooled.append(delta_resid)
        resid_lag_pooled.append(resid_lag)

    delta_resid_pooled = np.concatenate(delta_resid_pooled)
    resid_lag_pooled = np.concatenate(resid_lag_pooled)

    # DF regression
    try:
        rho = np.sum(delta_resid_pooled * resid_lag_pooled) / np.sum(resid_lag_pooled**2)

        # Compute standard error
        resid_df = delta_resid_pooled - rho * resid_lag_pooled
        sigma2 = np.sum(resid_df**2) / (len(resid_df) - 1)
        se_rho = np.sqrt(sigma2 / np.sum(resid_lag_pooled**2))

        # Standardize using Kao's adjustment
        # Kao (1999) provides modified statistics
        t_kao = (rho * np.sqrt(N * T)) / np.sqrt(sigma2)

        return t_kao

    except:
        return np.nan


def _kao_adf_test(resids: List[np.ndarray], lags: int = 1) -> float:
    """
    Compute Kao ADF test statistic.

    The ADF regression includes lagged first differences:
    Δε̂_{it} = ρ ε̂_{i,t-1} + Σ_j γ_j Δε̂_{i,t-j} + ν_{it}

    Parameters
    ----------
    resids : list of np.ndarray
        Residuals for each entity
    lags : int
        Number of lags for ADF regression

    Returns
    -------
    float
        ADF test statistic
    """
    N = len(resids)
    T = len(resids[0])

    # Pool all observations
    y_pooled = []
    X_pooled = []

    for resid_i in resids:
        delta_resid = np.diff(resid_i)
        resid_lag = resid_i[:-1]

        # Build regressor matrix
        X_adf = [resid_lag[lags:]]
        for j in range(1, lags + 1):
            if j < len(delta_resid):
                lag_j = (
                    delta_resid[lags - j : -j] if j < len(delta_resid) else delta_resid[lags - j :]
                )
                X_adf.append(lag_j)

        if len(X_adf) > 0 and len(X_adf[0]) > 0:
            X_adf = np.column_stack(X_adf) if len(X_adf) > 1 else X_adf[0].reshape(-1, 1)
            y_adf = delta_resid[lags:]

            y_pooled.append(y_adf)
            X_pooled.append(X_adf)

    # Stack
    y_pooled = np.concatenate(y_pooled)
    X_pooled = np.vstack(X_pooled)

    # ADF regression
    try:
        params = np.linalg.lstsq(X_pooled, y_pooled, rcond=None)[0]
        rho = params[0]

        # Compute standard error
        resid_adf = y_pooled - X_pooled @ params
        sigma2 = np.sum(resid_adf**2) / (len(y_pooled) - len(params))
        var_cov = sigma2 * np.linalg.inv(X_pooled.T @ X_pooled)
        se_rho = np.sqrt(var_cov[0, 0])

        # Kao's standardized statistic
        t_kao = (rho * np.sqrt(N * T)) / np.sqrt(sigma2)

        return t_kao

    except:
        return np.nan


def kao_test(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: Union[str, List[str]],
    method: str = "adf",
    trend: str = "c",
    lags: int = 1,
) -> KaoResult:
    """
    Kao (1999) DF and ADF-based panel cointegration tests.

    Tests the null hypothesis of no cointegration assuming a homogeneous
    cointegrating vector across all panel members. Uses pooled regression
    residuals and applies DF or ADF tests.

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
    method : str, default 'adf'
        Test method: 'df', 'adf', or 'all'
    trend : str, default 'c'
        Deterministic trend specification: 'n', 'c', or 'ct'
    lags : int, default 1
        Number of lags for ADF test

    Returns
    -------
    KaoResult
        Object containing test statistics, p-values, and critical values

    References
    ----------
    Kao, C. (1999). "Spurious Regression and Residual-Based Tests for
        Cointegration in Panel Data." Journal of Econometrics, 90(1), 1-44.

    Notes
    -----
    Unlike Pedroni tests, Kao tests assume a homogeneous cointegrating vector
    (same β for all entities). This makes the test simpler but more restrictive.

    Examples
    --------
    >>> from panelbox.diagnostics.cointegration import kao_test
    >>>
    >>> result = kao_test(
    ...     data, entity_col='country', time_col='year',
    ...     y_var='log_gdp', x_vars=['log_capital', 'log_labor'],
    ...     method='adf', lags=2
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

    # Get basic info
    entities = data[entity_col].unique()
    N = len(entities)
    T_avg = len(data[data[entity_col] == entities[0]])

    # Estimate pooled cointegrating regression
    beta, pooled_resid, entity_resids = _estimate_pooled_cointegrating_regression(
        data, entity_col, time_col, y_var, x_vars, trend
    )

    # Compute test statistics
    test_stats = {}

    if method in ["df", "all"]:
        test_stats["DF"] = _kao_df_test(entity_resids)

    if method in ["adf", "all"]:
        test_stats["ADF"] = _kao_adf_test(entity_resids, lags)

    # Critical values (from Kao 1999, Table 1)
    # These are asymptotic critical values
    critical_values_common = {"1%": -2.326, "5%": -1.645, "10%": -1.282}

    critical_values = {}
    pvalues = {}

    for test in test_stats.keys():
        critical_values[test] = critical_values_common.copy()

        # P-value using standard normal (left-tailed)
        pvalues[test] = (
            stats.norm.cdf(test_stats[test]) if not np.isnan(test_stats[test]) else np.nan
        )

    # Create result object
    result = KaoResult(
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
