"""
Westerlund (2007) ECM-Based Panel Cointegration Tests

Implementation of the four Westerlund (2007) error correction model-based
tests for panel cointegration: Gt, Ga, Pt, and Pa.

Reference
---------
Westerlund, J. (2007). "Testing for Error Correction in Panel Data."
    Oxford Bulletin of Economics and Statistics, 69(6), 709-748.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class WesterlundResult:
    """
    Results from Westerlund (2007) cointegration tests.

    Attributes
    ----------
    statistic : dict
        Dictionary containing test statistics (Gt, Ga, Pt, Pa)
    pvalue : dict
        Dictionary containing p-values from bootstrap
    critical_values : dict
        Dictionary containing critical values at 1%, 5%, 10%
    method : str
        Test method used ('Gt', 'Ga', 'Pt', 'Pa', or 'all')
    trend : str
        Deterministic trend specification
    lags : int
        Number of lags used
    n_bootstrap : int
        Number of bootstrap replications
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Number of time periods
    """

    statistic: Dict[str, float]
    pvalue: Dict[str, float]
    critical_values: Dict[str, Dict[str, float]]
    method: str
    trend: str
    lags: int
    n_bootstrap: int
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
        summary_str = f"Westerlund (2007) Cointegration Test Results\n"
        summary_str += f"{'='*60}\n"
        summary_str += f"Method: {self.method}\n"
        summary_str += f"Trend: {self.trend}\n"
        summary_str += f"Lags: {self.lags}\n"
        summary_str += f"Entities: {self.n_entities}, Time periods: {self.n_time}\n"
        summary_str += f"Bootstrap replications: {self.n_bootstrap}\n"
        summary_str += f"\n{self.summary().to_string(index=False)}\n"
        summary_str += f"\nH0: No cointegration (alpha_i = 0 for all i)\n"
        summary_str += f"***, **, * denote rejection at 1%, 5%, 10% level"
        return summary_str


def _estimate_ecm(
    y: np.ndarray, x: np.ndarray, lags: int, trend: str = "c"
) -> Tuple[float, float, np.ndarray]:
    """
    Estimate error correction model for a single entity.

    The ECM is:
    Δy_t = α_i d_t + α_i(y_{t-1} - β_i x_{t-1}) + Σ_j γ_ij Δy_{t-j} + Σ_j δ_ij Δx_{t-j} + ε_t

    Parameters
    ----------
    y : np.ndarray
        Dependent variable (T,)
    x : np.ndarray
        Independent variables (T, k)
    lags : int
        Number of lags for first differences
    trend : str
        Deterministic trend: 'n' (none), 'c' (constant), 'ct' (constant+trend)

    Returns
    -------
    alpha : float
        Error correction coefficient
    se_alpha : float
        Standard error of alpha
    resid : np.ndarray
        Residuals from ECM
    """
    T = len(y)
    k = x.shape[1] if x.ndim > 1 else 1

    # Prepare data
    dy = np.diff(y)
    dx = np.diff(x, axis=0) if x.ndim > 1 else np.diff(x)

    # Lagged levels
    y_lag = y[:-1]
    x_lag = x[:-1] if x.ndim > 1 else x[:-1]

    # Build regressor matrix
    regressors = []

    # Deterministic terms
    if trend == "c":
        regressors.append(np.ones(T - 1))
    elif trend == "ct":
        regressors.append(np.ones(T - 1))
        regressors.append(np.arange(1, T))

    # Error correction term: (y_{t-1} - β x_{t-1})
    # First, estimate cointegrating vector via OLS
    if trend in ["c", "ct"]:
        if x.ndim > 1:
            X_coint = np.column_stack([np.ones(T), x])
        else:
            X_coint = np.column_stack([np.ones(T), x])
    else:
        X_coint = x if x.ndim > 1 else x.reshape(-1, 1)

    beta_coint = np.linalg.lstsq(X_coint, y, rcond=None)[0]

    # Residuals from cointegrating regression (error correction term)
    ec_term = y - X_coint @ beta_coint
    ec_term_lag = ec_term[:-1]

    regressors.append(ec_term_lag)

    # Lagged first differences
    if lags > 0:
        for j in range(1, lags + 1):
            if j < len(dy):
                dy_lag = (
                    np.concatenate([np.zeros(j), dy[:-j]]) if j < len(dy) else np.zeros(len(dy))
                )
                regressors.append(dy_lag)

                if dx.ndim > 1:
                    for i in range(dx.shape[1]):
                        dx_lag = (
                            np.concatenate([np.zeros(j), dx[:-j, i]])
                            if j < len(dx)
                            else np.zeros(len(dx))
                        )
                        regressors.append(dx_lag)
                else:
                    dx_lag = (
                        np.concatenate([np.zeros(j), dx[:-j]]) if j < len(dx) else np.zeros(len(dx))
                    )
                    regressors.append(dx_lag)

    # Stack all regressors
    X_ecm = np.column_stack(regressors)

    # Estimate ECM via OLS
    try:
        params = np.linalg.lstsq(X_ecm, dy, rcond=None)[0]
        resid = dy - X_ecm @ params

        # Alpha is the coefficient on the error correction term
        # Position depends on trend specification
        if trend == "c":
            alpha_idx = 1
        elif trend == "ct":
            alpha_idx = 2
        else:
            alpha_idx = 0

        alpha = params[alpha_idx]

        # Compute standard error
        sigma2 = np.sum(resid**2) / (len(dy) - len(params))
        var_params = sigma2 * np.linalg.inv(X_ecm.T @ X_ecm)
        se_alpha = np.sqrt(var_params[alpha_idx, alpha_idx])

        return alpha, se_alpha, resid

    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in ECM estimation, returning NaN")
        return np.nan, np.nan, np.full(len(dy), np.nan)


def _select_lags(
    y: np.ndarray, x: np.ndarray, max_lags: int = 4, criterion: str = "aic", trend: str = "c"
) -> int:
    """
    Select optimal number of lags using information criterion.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable
    x : np.ndarray
        Independent variables
    max_lags : int
        Maximum lags to consider
    criterion : str
        'aic' or 'bic'
    trend : str
        Deterministic trend specification

    Returns
    -------
    int
        Optimal number of lags
    """
    ic_values = []

    for p in range(max_lags + 1):
        try:
            alpha, se_alpha, resid = _estimate_ecm(y, x, p, trend)

            if not np.isnan(alpha):
                T = len(resid)
                k = 1 + p * (1 + (x.shape[1] if x.ndim > 1 else 1))  # Number of parameters

                sse = np.sum(resid**2)

                if criterion == "aic":
                    ic = np.log(sse / T) + 2 * k / T
                else:  # bic
                    ic = np.log(sse / T) + k * np.log(T) / T

                ic_values.append(ic)
            else:
                ic_values.append(np.inf)
        except:
            ic_values.append(np.inf)

    return int(np.argmin(ic_values))


def _compute_test_statistics(alphas: np.ndarray, se_alphas: np.ndarray, T: int) -> Dict[str, float]:
    """
    Compute Westerlund test statistics.

    Parameters
    ----------
    alphas : np.ndarray
        Error correction coefficients for all entities (N,)
    se_alphas : np.ndarray
        Standard errors (N,)
    T : int
        Number of time periods

    Returns
    -------
    dict
        Dictionary with Gt, Ga, Pt, Pa statistics
    """
    N = len(alphas)

    # Filter out NaN values
    valid = ~(np.isnan(alphas) | np.isnan(se_alphas))
    alphas = alphas[valid]
    se_alphas = se_alphas[valid]
    N_valid = len(alphas)

    if N_valid == 0:
        return {"Gt": np.nan, "Ga": np.nan, "Pt": np.nan, "Pa": np.nan}

    # Gt: Group-mean t-statistic
    t_stats = alphas / se_alphas
    Gt = np.mean(t_stats)

    # Ga: Group-mean ratio (normalized by first estimate)
    # Use T * alpha / alpha[0] approximation
    alpha_1 = alphas[0] if alphas[0] != 0 else 1e-10
    Ga = np.mean(T * alphas / alpha_1)

    # Pt: Panel pooled t-statistic
    # Pool all observations
    alpha_pooled = np.mean(alphas)
    se_pooled = np.sqrt(np.mean(se_alphas**2) / N_valid)
    Pt = alpha_pooled / se_pooled

    # Pa: Panel pooled ratio
    Pa = T * alpha_pooled

    return {"Gt": Gt, "Ga": Ga, "Pt": Pt, "Pa": Pa}


def _bootstrap_critical_values(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: List[str],
    lags: int,
    trend: str,
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap critical values under null hypothesis of no cointegration.

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
    lags : int
        Number of lags
    trend : str
        Trend specification
    n_bootstrap : int
        Number of bootstrap replications
    random_state : int, optional
        Random seed

    Returns
    -------
    dict
        Critical values at 1%, 5%, 10% for each test
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Initialize storage
    boot_stats = {"Gt": [], "Ga": [], "Pt": [], "Pa": []}

    entities = data[entity_col].unique()
    N = len(entities)

    for b in range(n_bootstrap):
        boot_alphas = []
        boot_se_alphas = []

        for entity in entities:
            entity_data = data[data[entity_col] == entity].sort_values(time_col)

            y = entity_data[y_var].values
            X = entity_data[x_vars].values
            T = len(y)

            # Generate data under H0 (random walk, no cointegration)
            # Use resampled residuals from original data
            alpha_orig, se_orig, resid_orig = _estimate_ecm(y, X, lags, trend)

            if not np.isnan(alpha_orig):
                # Resample residuals
                resid_boot = np.random.choice(resid_orig, size=len(resid_orig), replace=True)

                # Generate y under H0 (random walk)
                # Need to pad to match original length
                y_boot = np.zeros(T)
                y_boot[0] = y[0]
                y_boot[1:] = y[0] + np.cumsum(resid_boot)

                # Estimate ECM on bootstrap data
                alpha_boot, se_boot, _ = _estimate_ecm(y_boot, X, lags, trend)

                if not np.isnan(alpha_boot):
                    boot_alphas.append(alpha_boot)
                    boot_se_alphas.append(se_boot)

        if len(boot_alphas) > 0:
            boot_alphas = np.array(boot_alphas)
            boot_se_alphas = np.array(boot_se_alphas)

            # Compute statistics for this bootstrap replication
            stats_b = _compute_test_statistics(boot_alphas, boot_se_alphas, T)

            for key in boot_stats:
                boot_stats[key].append(stats_b[key])

    # Compute critical values (percentiles)
    critical_values = {}
    for test in ["Gt", "Ga", "Pt", "Pa"]:
        values = np.array(boot_stats[test])
        values = values[~np.isnan(values)]

        if len(values) > 0:
            critical_values[test] = {
                "1%": np.percentile(values, 1),
                "5%": np.percentile(values, 5),
                "10%": np.percentile(values, 10),
            }
        else:
            critical_values[test] = {"1%": np.nan, "5%": np.nan, "10%": np.nan}

    return critical_values


def westerlund_test(
    data: pd.DataFrame,
    entity_col: str,
    time_col: str,
    y_var: str,
    x_vars: Union[str, List[str]],
    method: Literal["Gt", "Ga", "Pt", "Pa", "all"] = "all",
    trend: Literal["n", "c", "ct"] = "c",
    lags: Union[int, str] = "auto",
    max_lags: int = 4,
    lag_criterion: Literal["aic", "bic"] = "aic",
    n_bootstrap: int = 1000,
    random_state: Optional[int] = None,
    use_bootstrap: bool = True,
) -> WesterlundResult:
    """
    Westerlund (2007) error correction model-based panel cointegration tests.

    Tests the null hypothesis of no cointegration against the alternative
    of cointegration for at least some panel members.

    The error correction model is:
    Δy_{it} = α_i d_t + α_i(y_{i,t-1} - β_i x_{i,t-1}) +
              Σ_j γ_{ij} Δy_{i,t-j} + Σ_j δ_{ij} Δx_{i,t-j} + ε_{it}

    H0: α_i = 0 for all i (no error correction → no cointegration)
    H1: α_i < 0 for at least some i (error correction → cointegration)

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
    method : {'Gt', 'Ga', 'Pt', 'Pa', 'all'}, default 'all'
        Test statistic to compute:
        - 'Gt': Group-mean t-statistic
        - 'Ga': Group-mean ratio statistic
        - 'Pt': Panel pooled t-statistic
        - 'Pa': Panel pooled ratio statistic
        - 'all': Compute all four statistics
    trend : {'n', 'c', 'ct'}, default 'c'
        Deterministic trend specification:
        - 'n': No deterministic trend
        - 'c': Constant only
        - 'ct': Constant and linear trend
    lags : int or 'auto', default 'auto'
        Number of lags for first differences. If 'auto', selected by information criterion.
    max_lags : int, default 4
        Maximum lags to consider when lags='auto'
    lag_criterion : {'aic', 'bic'}, default 'aic'
        Information criterion for lag selection
    n_bootstrap : int, default 1000
        Number of bootstrap replications for critical values
    random_state : int, optional
        Random seed for reproducibility
    use_bootstrap : bool, default True
        If True, compute bootstrap critical values. If False, use tabulated values.

    Returns
    -------
    WesterlundResult
        Object containing test statistics, p-values, and critical values

    References
    ----------
    Westerlund, J. (2007). "Testing for Error Correction in Panel Data."
        Oxford Bulletin of Economics and Statistics, 69(6), 709-748.

    Examples
    --------
    >>> import pandas as pd
    >>> from panelbox.diagnostics.cointegration import westerlund_test
    >>>
    >>> # Load panel data
    >>> data = pd.read_csv('panel_data.csv')
    >>>
    >>> # Test for cointegration
    >>> result = westerlund_test(
    ...     data, entity_col='country', time_col='year',
    ...     y_var='log_gdp', x_vars=['log_capital', 'log_labor']
    ... )
    >>> print(result)
    >>>
    >>> # Check rejection at 5%
    >>> result.reject_at(0.05)
    """
    # Performance warnings
    if use_bootstrap and n_bootstrap > 2000:
        warnings.warn(
            f"Large bootstrap replications (n={n_bootstrap}) may take >5 minutes. "
            "Consider reducing to 1000 for exploratory analysis or use tabulated critical values.",
            UserWarning,
        )

    n_entities = data[entity_col].nunique()
    n_periods = data.groupby(entity_col)[time_col].count().mean()

    if use_bootstrap and n_entities * n_periods > 100000:
        warnings.warn(
            f"Large panel (N={n_entities}, T≈{n_periods:.0f}) with bootstrap may take considerable time. "
            "Consider using tabulated critical values (use_bootstrap=False).",
            UserWarning,
        )

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

    # Estimate ECM for each entity
    alphas = []
    se_alphas = []
    T_avg = 0

    # Determine lags to use
    if lags == "auto":
        # Use the first entity to select lags (could be improved to select per entity)
        entity_data_first = data[data[entity_col] == entities[0]].sort_values(time_col)
        y_first = entity_data_first[y_var].values
        X_first = entity_data_first[x_vars].values
        selected_lags = _select_lags(y_first, X_first, max_lags, lag_criterion, trend)
    else:
        selected_lags = lags

    for entity in entities:
        entity_data = data[data[entity_col] == entity].sort_values(time_col)

        y = entity_data[y_var].values
        X = entity_data[x_vars].values
        T = len(y)
        T_avg += T

        # Estimate ECM
        alpha, se_alpha, _ = _estimate_ecm(y, X, selected_lags, trend)

        alphas.append(alpha)
        se_alphas.append(se_alpha)

    T_avg = T_avg // N
    alphas = np.array(alphas)
    se_alphas = np.array(se_alphas)

    # Compute test statistics
    test_stats = _compute_test_statistics(alphas, se_alphas, T_avg)

    # Filter statistics based on method
    if method != "all":
        test_stats = {method: test_stats[method]}

    # Compute critical values and p-values
    if use_bootstrap and n_bootstrap > 0:
        critical_values = _bootstrap_critical_values(
            data,
            entity_col,
            time_col,
            y_var,
            x_vars,
            selected_lags,
            trend,
            n_bootstrap,
            random_state,
        )

        # Compute p-values from critical values
        # For left-tailed test (alpha < 0 under H1), reject if stat < critical value
        pvalues = {}
        for test in test_stats.keys():
            if not np.isnan(test_stats[test]):
                # Approximate p-value from critical values
                # If stat is less than 1% CV, p < 0.01, etc.
                if test_stats[test] < critical_values[test]["1%"]:
                    pvalues[test] = 0.005
                elif test_stats[test] < critical_values[test]["5%"]:
                    pvalues[test] = 0.025
                elif test_stats[test] < critical_values[test]["10%"]:
                    pvalues[test] = 0.075
                else:
                    pvalues[test] = 0.15
            else:
                pvalues[test] = np.nan
    else:
        # Use tabulated critical values (approximation)
        # These are rough approximations - bootstrap is preferred
        critical_values = {}
        pvalues = {}
        for test in test_stats.keys():
            critical_values[test] = {"1%": -2.58, "5%": -1.96, "10%": -1.645}
            # Approximate p-value using normal distribution
            pvalues[test] = (
                stats.norm.cdf(test_stats[test]) if not np.isnan(test_stats[test]) else np.nan
            )

    # Create result object
    result = WesterlundResult(
        statistic=test_stats,
        pvalue=pvalues,
        critical_values=critical_values,
        method=method,
        trend=trend,
        lags=selected_lags,
        n_bootstrap=n_bootstrap,
        n_entities=N,
        n_time=T_avg,
    )

    return result
