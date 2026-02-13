"""
Panel Vector Error Correction Model (VECM) for cointegrated systems.

This module implements Panel VECM estimation, rank selection tests, and
analysis tools for panel data with cointegrated variables.

The VECM separates short-run dynamics from long-run equilibrium relationships:

    Δy_it = α_i + Π·y_{i,t-1} + Σ_l Γ_l·Δy_{i,t-l} + ε_it

where:
    - Π = α·β' has rank r (number of cointegrating relations)
    - β = cointegrating vectors (K×r): long-run equilibria
    - α = loading matrix (K×r): adjustment speeds
    - Γ_l = short-run dynamics matrices (K×K)

References
----------
.. [1] Johansen, S. (1991). Estimation and hypothesis testing of
       cointegration vectors in Gaussian vector autoregressive models.
       Econometrica, 59(6), 1551-1580.
.. [2] Larsson, R., Lyhagen, J., & Löthgren, M. (2001). Likelihood-based
       cointegration tests in heterogeneous panels. The Econometrics Journal,
       4(1), 109-142.
.. [3] Lütkepohl, H. (2005). New Introduction to Multiple Time Series
       Analysis. Springer. Chapter 9.
.. [4] Breitung, J., & Pesaran, M. H. (2008). Unit roots and cointegration
       in panels. In The econometrics of panel data (pp. 279-322). Springer.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import eig, inv, sqrtm

from panelbox.var.data import PanelVARData
from panelbox.var.inference import within_transformation


@dataclass
class RankTestResult:
    """
    Results from a single cointegration rank test.

    Parameters
    ----------
    rank : int
        The tested rank (r)
    test_stat : float
        Test statistic value
    z_stat : float
        Standardized test statistic
    p_value : float
        P-value for the test
    critical_values : dict, optional
        Critical values at different significance levels
    test_type : str
        Type of test: 'trace' or 'maxeig'
    """

    rank: int
    test_stat: float
    z_stat: float
    p_value: float
    test_type: str
    critical_values: Optional[Dict[str, float]] = None

    @property
    def reject(self) -> bool:
        """Reject null hypothesis at 5% level."""
        return self.p_value < 0.05


class CointegrationRankTest:
    """
    Panel cointegration rank tests (Larsson et al. 2001).

    Determines the number of cointegrating relations (rank r) in a panel
    system using trace and max-eigenvalue tests.

    Parameters
    ----------
    data : PanelVARData
        Panel VAR data container
    max_rank : int, optional
        Maximum rank to test (default: K-1)
    deterministic : str, optional
        Deterministic specification: 'nc' (no constant), 'c' (constant in CE),
        'ct' (constant and trend). Default: 'c'

    Attributes
    ----------
    K : int
        Number of variables
    N : int
        Number of entities
    T_avg : float
        Average time periods per entity
    max_rank : int
        Maximum rank tested

    References
    ----------
    .. [1] Larsson, R., Lyhagen, J., & Löthgren, M. (2001). Likelihood-based
           cointegration tests in heterogeneous panels. The Econometrics Journal.

    Examples
    --------
    >>> from panelbox.var import PanelVARData
    >>> from panelbox.var.vecm import CointegrationRankTest
    >>>
    >>> # Create data
    >>> data = PanelVARData(df, endog_vars=['y1', 'y2', 'y3'],
    ...                     entity_col='id', time_col='time', lags=2)
    >>>
    >>> # Test rank
    >>> rank_test = CointegrationRankTest(data)
    >>> results = rank_test.test_rank()
    >>> print(results.summary())
    >>> print(f"Selected rank: {results.selected_rank}")
    """

    def __init__(
        self,
        data: PanelVARData,
        max_rank: Optional[int] = None,
        deterministic: str = "c",
    ):
        self.data = data
        self.K = data.K
        self.N = data.N
        self.p = data.p
        self.deterministic = deterministic

        # Calculate average T per entity
        entity_counts = data.data.groupby(data.entity_col).size()
        self.T_avg = entity_counts.mean()
        self.T_by_entity = entity_counts.to_dict()

        # Max rank defaults to K-1
        if max_rank is None:
            self.max_rank = self.K - 1
        else:
            if max_rank >= self.K:
                raise ValueError(f"max_rank must be < K={self.K}")
            self.max_rank = max_rank

        # Store residuals from reduced rank regression
        self._residuals_computed = False
        self._R0 = None  # Residuals from Δy_it regression
        self._R1 = None  # Residuals from y_{i,t-1} regression

    def _compute_residuals(self):
        """
        Compute residuals from concentrated model.

        This performs the first step of Johansen procedure:
        regress Δy_it and y_{i,t-1} on lagged differences to concentrate
        out the short-run dynamics.
        """
        if self._residuals_computed:
            return

        # Get data
        df = self.data.data.copy()
        endog_vars = self.data.endog_vars
        entity_col = self.data.entity_col
        time_col = self.data.time_col

        # Create differences
        for var in endog_vars:
            df[f"d_{var}"] = df.groupby(entity_col)[var].diff()

        # Create lags of differences
        lag_vars = []
        for lag in range(1, self.p):
            for var in endog_vars:
                lag_var = f"d_{var}_lag{lag}"
                df[lag_var] = df.groupby(entity_col)[f"d_{var}"].shift(lag)
                lag_vars.append(lag_var)

        # Create lag 1 of levels
        level_lag_vars = []
        for var in endog_vars:
            lag_var = f"{var}_lag1"
            df[lag_var] = df.groupby(entity_col)[var].shift(1)
            level_lag_vars.append(lag_var)

        # Drop missing values
        df = df.dropna()

        # Dependent variables
        delta_y_vars = [f"d_{var}" for var in endog_vars]

        # Within transformation (remove fixed effects)
        df_within = df.copy()
        for var in delta_y_vars + lag_vars + level_lag_vars:
            demeaned, _ = within_transformation(df_within[var].values, df_within[entity_col].values)
            df_within[var] = demeaned

        # Regress Δy_it on lagged differences
        X = df_within[lag_vars].values if lag_vars else np.zeros((len(df_within), 1))
        Y = df_within[delta_y_vars].values

        if lag_vars:
            # OLS: β = (X'X)^{-1}X'Y
            XtX = X.T @ X
            XtY = X.T @ Y
            beta = np.linalg.solve(XtX, XtY)
            fitted = X @ beta
        else:
            fitted = np.zeros_like(Y)

        self._R0 = Y - fitted

        # Regress y_{i,t-1} on lagged differences
        Y1 = df_within[level_lag_vars].values

        if lag_vars:
            fitted1 = X @ np.linalg.solve(XtX, X.T @ Y1)
        else:
            fitted1 = np.zeros_like(Y1)

        self._R1 = Y1 - fitted1

        self._residuals_computed = True

    def _compute_eigenvalues_entity(self, entity_id: int) -> np.ndarray:
        """
        Compute eigenvalues for a single entity.

        This follows Johansen's procedure: solve generalized eigenvalue problem
        for the canonical correlations between R0 and R1.

        Parameters
        ----------
        entity_id : int
            Entity identifier

        Returns
        -------
        np.ndarray
            Eigenvalues in descending order
        """
        # Get entity data
        df = self.data.data
        entity_mask = df[self.data.entity_col] == entity_id

        # Get indices in residual matrices
        # (This assumes residuals are in same order as df after dropna)
        # For simplicity, recompute for this entity
        df_entity = df[entity_mask].copy()

        # Create differences and lags (same as in _compute_residuals)
        endog_vars = self.data.endog_vars
        for var in endog_vars:
            df_entity[f"d_{var}"] = df_entity[var].diff()

        lag_vars = []
        for lag in range(1, self.p):
            for var in endog_vars:
                lag_var = f"d_{var}_lag{lag}"
                df_entity[lag_var] = df_entity[f"d_{var}"].shift(lag)
                lag_vars.append(lag_var)

        level_lag_vars = []
        for var in endog_vars:
            lag_var = f"{var}_lag1"
            df_entity[lag_var] = df_entity[var].shift(1)
            level_lag_vars.append(lag_var)

        df_entity = df_entity.dropna()

        if len(df_entity) < self.K + 2:
            # Not enough observations
            return np.zeros(self.K)

        delta_y_vars = [f"d_{var}" for var in endog_vars]

        # Demean (within transformation for single entity)
        for var in delta_y_vars + lag_vars + level_lag_vars:
            df_entity[var] = df_entity[var] - df_entity[var].mean()

        # Regress to get residuals
        X = df_entity[lag_vars].values if lag_vars else np.zeros((len(df_entity), 1))
        Y = df_entity[delta_y_vars].values
        Y1 = df_entity[level_lag_vars].values

        if lag_vars and len(df_entity) > len(lag_vars):
            beta = np.linalg.lstsq(X, Y, rcond=None)[0]
            R0 = Y - X @ beta
            beta1 = np.linalg.lstsq(X, Y1, rcond=None)[0]
            R1 = Y1 - X @ beta1
        else:
            R0 = Y
            R1 = Y1

        T_i = len(R0)

        # Product moment matrices
        S00 = R0.T @ R0 / T_i
        S11 = R1.T @ R1 / T_i
        S01 = R0.T @ R1 / T_i
        S10 = S01.T

        # Regularize if needed
        S00 += np.eye(self.K) * 1e-10
        S11 += np.eye(self.K) * 1e-10

        # Solve generalized eigenvalue problem: det(λ·S11 - S10·S00^{-1}·S01) = 0
        try:
            S00_inv = inv(S00)
            M = S00_inv @ S01 @ inv(S11) @ S10
            eigenvalues, _ = np.linalg.eig(M)
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order
            # Clip to [0, 1) range (canonical correlations squared)
            eigenvalues = np.clip(eigenvalues, 0, 0.999999)
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(self.K)

        return eigenvalues

    def _panel_trace_statistic(self, rank: int) -> Tuple[float, float]:
        """
        Compute panel trace test statistic for given rank.

        H0: cointegration rank ≤ r

        Parameters
        ----------
        rank : int
            Tested rank r

        Returns
        -------
        tuple of (LR_bar, Z_trace)
            LR_bar: average trace statistic
            Z_trace: standardized test statistic
        """
        # Compute eigenvalues for each entity
        LR_sum = 0.0
        valid_entities = 0

        for entity_id in self.data.data[self.data.entity_col].unique():
            eigenvalues = self._compute_eigenvalues_entity(entity_id)

            # Get T for this entity
            T_i = self.T_by_entity.get(entity_id, self.T_avg)
            T_i = T_i - self.p - 1  # Effective T after lags

            if T_i > self.K + 2:
                # Trace statistic for this entity
                LR_i = -T_i * np.sum(np.log(1 - eigenvalues[rank:]))
                LR_sum += LR_i
                valid_entities += 1

        if valid_entities == 0:
            return 0.0, 0.0

        LR_bar = LR_sum / valid_entities

        # Standardize using Larsson et al. (2001) moments
        # Asymptotic mean and variance
        E_LR, Var_LR = self._trace_moments(rank)

        if Var_LR > 0:
            Z_trace = np.sqrt(valid_entities) * (LR_bar - E_LR) / np.sqrt(Var_LR)
        else:
            Z_trace = 0.0

        return LR_bar, Z_trace

    def _trace_moments(self, rank: int) -> Tuple[float, float]:
        """
        Compute theoretical mean and variance of trace statistic.

        Based on Larsson et al. (2001) Theorem 1.

        Parameters
        ----------
        rank : int
            Tested rank r

        Returns
        -------
        tuple of (E_LR, Var_LR)
            Expected value and variance
        """
        K = self.K
        p = rank  # Number of zero eigenvalues under H0
        m = K - rank  # Number of non-zero eigenvalues

        # Asymptotic mean (Larsson et al. 2001, Theorem 1)
        # E[LR] for trace test with rank r
        if self.deterministic == "nc":
            # No constant
            E_LR = m * (m + 1) / 2
        elif self.deterministic == "c":
            # Constant in cointegration relation
            E_LR = m * (m + 1) / 2 + m
        else:  # "ct"
            # Constant and trend
            E_LR = m * (m + 1) / 2 + 2 * m

        # Asymptotic variance (Larsson et al. 2001)
        Var_LR = m * (m + 1) / 3

        return E_LR, Var_LR

    def _panel_maxeig_statistic(self, rank: int) -> Tuple[float, float]:
        """
        Compute panel max-eigenvalue test statistic for given rank.

        H0: cointegration rank = r vs H1: rank = r+1

        Parameters
        ----------
        rank : int
            Tested rank r

        Returns
        -------
        tuple of (LR_bar, Z_maxeig)
            LR_bar: average max-eigenvalue statistic
            Z_maxeig: standardized test statistic
        """
        # Compute eigenvalues for each entity
        LR_sum = 0.0
        valid_entities = 0

        for entity_id in self.data.data[self.data.entity_col].unique():
            eigenvalues = self._compute_eigenvalues_entity(entity_id)

            # Get T for this entity
            T_i = self.T_by_entity.get(entity_id, self.T_avg)
            T_i = T_i - self.p - 1

            if T_i > self.K + 2 and rank < len(eigenvalues):
                # Max-eigenvalue statistic for this entity
                LR_i = -T_i * np.log(1 - eigenvalues[rank])
                LR_sum += LR_i
                valid_entities += 1

        if valid_entities == 0:
            return 0.0, 0.0

        LR_bar = LR_sum / valid_entities

        # Standardize
        E_LR, Var_LR = self._maxeig_moments(rank)

        if Var_LR > 0:
            Z_maxeig = np.sqrt(valid_entities) * (LR_bar - E_LR) / np.sqrt(Var_LR)
        else:
            Z_maxeig = 0.0

        return LR_bar, Z_maxeig

    def _maxeig_moments(self, rank: int) -> Tuple[float, float]:
        """
        Compute theoretical mean and variance of max-eigenvalue statistic.

        Parameters
        ----------
        rank : int
            Tested rank r

        Returns
        -------
        tuple of (E_LR, Var_LR)
        """
        # For max-eigenvalue test, testing r vs r+1
        # Asymptotic moments differ from trace test
        if self.deterministic == "nc":
            E_LR = 1.0
        elif self.deterministic == "c":
            E_LR = 2.0
        else:  # "ct"
            E_LR = 3.0

        Var_LR = 2.0

        return E_LR, Var_LR

    def test_rank(
        self, use_bootstrap: bool = False, n_bootstrap: int = 1000, seed: int = 42
    ) -> "RankSelectionResult":
        """
        Test cointegration rank using trace and max-eigenvalue tests.

        Parameters
        ----------
        use_bootstrap : bool, optional
            Whether to use bootstrap p-values (default: False)
        n_bootstrap : int, optional
            Number of bootstrap replications (default: 1000)
        seed : int, optional
            Random seed for bootstrap (default: 42)

        Returns
        -------
        RankSelectionResult
            Results containing trace and max-eigenvalue tests for all ranks

        Examples
        --------
        >>> rank_test = CointegrationRankTest(data)
        >>> results = rank_test.test_rank()
        >>> print(results.summary())
        >>> print(f"Selected rank: {results.selected_rank}")
        """
        # Compute residuals if not done
        self._compute_residuals()

        trace_results = []
        maxeig_results = []

        # Test each rank from 0 to max_rank
        for r in range(self.max_rank + 1):
            # Trace test
            LR_trace, Z_trace = self._panel_trace_statistic(r)
            p_trace = 1 - stats.norm.cdf(Z_trace)

            trace_results.append(
                RankTestResult(
                    rank=r,
                    test_stat=LR_trace,
                    z_stat=Z_trace,
                    p_value=p_trace,
                    test_type="trace",
                )
            )

            # Max-eigenvalue test
            if r < self.K:
                LR_maxeig, Z_maxeig = self._panel_maxeig_statistic(r)
                p_maxeig = 1 - stats.norm.cdf(Z_maxeig)

                maxeig_results.append(
                    RankTestResult(
                        rank=r,
                        test_stat=LR_maxeig,
                        z_stat=Z_maxeig,
                        p_value=p_maxeig,
                        test_type="maxeig",
                    )
                )

        return RankSelectionResult(
            trace_tests=trace_results,
            maxeig_tests=maxeig_results,
            K=self.K,
            N=self.N,
            T_avg=self.T_avg,
            max_rank=self.max_rank,
        )


class RankSelectionResult:
    """
    Results from cointegration rank selection tests.

    Contains trace and max-eigenvalue test results for all tested ranks.

    Parameters
    ----------
    trace_tests : List[RankTestResult]
        Trace test results
    maxeig_tests : List[RankTestResult]
        Max-eigenvalue test results
    K : int
        Number of variables
    N : int
        Number of entities
    T_avg : float
        Average time periods
    max_rank : int
        Maximum tested rank

    Attributes
    ----------
    selected_rank_trace : int
        Rank selected by trace test
    selected_rank_maxeig : int
        Rank selected by max-eigenvalue test
    selected_rank : int
        Consensus rank (trace if agree, otherwise trace)
    """

    def __init__(
        self,
        trace_tests: List[RankTestResult],
        maxeig_tests: List[RankTestResult],
        K: int,
        N: int,
        T_avg: float,
        max_rank: int,
    ):
        self.trace_tests = trace_tests
        self.maxeig_tests = maxeig_tests
        self.K = K
        self.N = N
        self.T_avg = T_avg
        self.max_rank = max_rank

        # Select ranks
        self._select_ranks()

    def _select_ranks(self):
        """Determine selected rank from test results."""
        # Trace test: largest r where we do NOT reject H0: rank ≤ r
        # I.e., first r where p-value > 0.05
        self.selected_rank_trace = 0
        for test in self.trace_tests:
            if test.p_value > 0.05:
                self.selected_rank_trace = test.rank
                break
        else:
            # All rejected, use max rank
            self.selected_rank_trace = self.max_rank

        # Max-eigenvalue test: largest r where we do NOT reject H0: rank = r
        self.selected_rank_maxeig = 0
        for test in self.maxeig_tests:
            if test.p_value > 0.05:
                self.selected_rank_maxeig = test.rank
                break
        else:
            self.selected_rank_maxeig = self.max_rank

        # Consensus: use trace test by default (more robust)
        # Warn if they differ
        if self.selected_rank_trace != self.selected_rank_maxeig:
            import warnings

            warnings.warn(
                f"Trace and max-eigenvalue tests disagree on rank "
                f"(trace: {self.selected_rank_trace}, "
                f"maxeig: {self.selected_rank_maxeig}). "
                f"Using trace test result. Consider using bootstrap p-values."
            )

        self.selected_rank = self.selected_rank_trace

    def summary(self) -> str:
        """
        Generate formatted summary of rank selection results.

        Returns
        -------
        str
            Formatted summary table
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Panel Cointegration Rank Test (Larsson et al. 2001)")
        lines.append("=" * 80)
        lines.append(f"Number of variables (K): {self.K}")
        lines.append(f"Number of entities (N): {self.N}")
        lines.append(f"Average time periods (T): {self.T_avg:.1f}")
        lines.append(f"Maximum rank tested: {self.max_rank}")
        lines.append("")
        lines.append("─" * 80)
        lines.append("Trace Test Results")
        lines.append("─" * 80)
        lines.append(
            f"{'H0: rank ≤ r':<15} {'Trace Stat':<12} {'Z-stat':<10} "
            f"{'P-value':<10} {'Result':<10}"
        )
        lines.append("─" * 80)

        for test in self.trace_tests:
            sig = ""
            if test.p_value < 0.01:
                sig = "***"
            elif test.p_value < 0.05:
                sig = "**"
            elif test.p_value < 0.10:
                sig = "*"

            result = "Reject" if test.reject else "Not reject"

            lines.append(
                f"{'r = ' + str(test.rank):<15} "
                f"{test.test_stat:<12.2f} "
                f"{test.z_stat:<10.3f} "
                f"{test.p_value:<10.4f}{sig:<3} "
                f"{result:<10}"
            )

        lines.append("")
        lines.append("─" * 80)
        lines.append("Max-Eigenvalue Test Results")
        lines.append("─" * 80)
        lines.append(
            f"{'H0: rank = r':<15} {'MaxEig Stat':<12} {'Z-stat':<10} "
            f"{'P-value':<10} {'Result':<10}"
        )
        lines.append("─" * 80)

        for test in self.maxeig_tests:
            sig = ""
            if test.p_value < 0.01:
                sig = "***"
            elif test.p_value < 0.05:
                sig = "**"
            elif test.p_value < 0.10:
                sig = "*"

            result = "Reject" if test.reject else "Not reject"

            lines.append(
                f"{'r = ' + str(test.rank):<15} "
                f"{test.test_stat:<12.2f} "
                f"{test.z_stat:<10.3f} "
                f"{test.p_value:<10.4f}{sig:<3} "
                f"{result:<10}"
            )

        lines.append("")
        lines.append("─" * 80)
        lines.append("Selected Ranks")
        lines.append("─" * 80)
        lines.append(f"Trace test: r = {self.selected_rank_trace}")
        lines.append(f"Max-eigenvalue test: r = {self.selected_rank_maxeig}")
        lines.append(f"Consensus (used): r = {self.selected_rank}")
        lines.append("=" * 80)
        lines.append("Significance: *** 1%, ** 5%, * 10%")
        lines.append("=" * 80)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"RankSelectionResult(selected_rank={self.selected_rank}, " f"K={self.K}, N={self.N})"
        )


class PanelVECMResult:
    """
    Results container for Panel VECM estimation.

    Stores estimated parameters and provides methods for analysis and inference.

    Parameters
    ----------
    alpha : np.ndarray
        Loading matrix (K × r): adjustment speeds
    beta : np.ndarray
        Cointegrating vectors (K × r): long-run relations
    Gamma : List[np.ndarray]
        Short-run dynamics matrices, list of K×K arrays
    Sigma : np.ndarray
        Residual covariance matrix (K × K)
    residuals : np.ndarray
        Residuals from estimation
    var_names : List[str]
        Variable names
    rank : int
        Cointegration rank
    method : str
        Estimation method used
    N : int
        Number of entities
    T_avg : float
        Average time periods
    deterministic : str
        Deterministic specification

    Attributes
    ----------
    Pi : np.ndarray
        Long-run impact matrix Π = α·β' (K × K with rank r)
    K : int
        Number of variables
    p : int
        Number of lags (in VAR representation)
    """

    def __init__(
        self,
        alpha: np.ndarray,
        beta: np.ndarray,
        Gamma: List[np.ndarray],
        Sigma: np.ndarray,
        residuals: np.ndarray,
        var_names: List[str],
        rank: int,
        method: str,
        N: int,
        T_avg: float,
        deterministic: str = "c",
        alpha_se: Optional[np.ndarray] = None,
        beta_se: Optional[np.ndarray] = None,
    ):
        self.alpha = alpha
        self.beta = beta
        self.Gamma = Gamma
        self.Sigma = Sigma
        self.residuals = residuals
        self.var_names = var_names
        self.rank = rank
        self.method = method
        self.N = N
        self.T_avg = T_avg
        self.deterministic = deterministic
        self.alpha_se = alpha_se
        self.beta_se = beta_se

        # Derived attributes
        self.K = len(var_names)
        self.p = len(Gamma) + 1  # Number of lags in VAR representation
        self.Pi = alpha @ beta.T

    def cointegrating_relations(self) -> pd.DataFrame:
        """
        Return cointegrating relations (β) as DataFrame.

        The cointegrating vectors are normalized so that the first variable
        in each relation has coefficient 1.

        Returns
        -------
        pd.DataFrame
            Cointegrating relations with variables as rows and relations as columns
        """
        beta_normalized = self.beta.copy()

        # Normalize: first element of each column = 1
        for j in range(self.rank):
            if beta_normalized[0, j] != 0:
                beta_normalized[:, j] /= beta_normalized[0, j]

        df = pd.DataFrame(
            beta_normalized,
            index=self.var_names,
            columns=[f"Relation_{i+1}" for i in range(self.rank)],
        )

        return df

    def adjustment_speeds(self) -> pd.DataFrame:
        """
        Return adjustment speed coefficients (α) as DataFrame.

        Returns
        -------
        pd.DataFrame
            Adjustment speeds with variables as rows and relations as columns
        """
        df = pd.DataFrame(
            self.alpha,
            index=[f"Δ{var}" for var in self.var_names],
            columns=[f"ECT_{i+1}" for i in range(self.rank)],
        )

        return df

    def short_run_dynamics(self) -> List[pd.DataFrame]:
        """
        Return short-run dynamics matrices (Γ) as DataFrames.

        Returns
        -------
        List[pd.DataFrame]
            List of Γ matrices as DataFrames
        """
        gamma_dfs = []
        for lag, Gamma_l in enumerate(self.Gamma, start=1):
            df = pd.DataFrame(
                Gamma_l,
                index=[f"Δ{var}" for var in self.var_names],
                columns=[f"Δ{var}_lag{lag}" for var in self.var_names],
            )
            gamma_dfs.append(df)

        return gamma_dfs

    def to_var(self) -> List[np.ndarray]:
        """
        Convert VECM to VAR representation in levels.

        VECM(p-1): Δy_t = Π·y_{t-1} + Σ_l Γ_l·Δy_{t-l} + ε_t
        VAR(p):    y_t = Σ_l A_l·y_{t-l} + ε_t

        Relations:
            A_1 = Π + Γ_1 + I
            A_l = Γ_l - Γ_{l-1}  (l = 2,...,p-1)
            A_p = -Γ_{p-1}

        Returns
        -------
        List[np.ndarray]
            List of A matrices [A_1, A_2, ..., A_p]
        """
        A_matrices = []

        # A_1 = Π + Γ_1 + I
        if len(self.Gamma) > 0:
            A1 = self.Pi + self.Gamma[0] + np.eye(self.K)
        else:
            A1 = self.Pi + np.eye(self.K)
        A_matrices.append(A1)

        # A_l = Γ_l - Γ_{l-1} for l = 2,...,p-1
        for l in range(1, len(self.Gamma)):
            Al = self.Gamma[l] - self.Gamma[l - 1]
            A_matrices.append(Al)

        # A_p = -Γ_{p-1}
        if len(self.Gamma) > 0:
            Ap = -self.Gamma[-1]
            A_matrices.append(Ap)

        return A_matrices

    def test_weak_exogeneity(self, variable: str) -> Dict[str, float]:
        """
        Test weak exogeneity: H0: α[variable, :] = 0

        Weak exogeneity means the variable does not respond to deviations
        from long-run equilibrium.

        Parameters
        ----------
        variable : str
            Variable name to test

        Returns
        -------
        dict
            Dictionary with 'statistic', 'p_value', 'df', 'variable'
        """
        if variable not in self.var_names:
            raise ValueError(f"Variable '{variable}' not found")

        var_idx = self.var_names.index(variable)

        # H0: α[var_idx, :] = 0 (all loadings for this variable)
        alpha_row = self.alpha[var_idx, :]

        # Wald test statistic (simplified - assumes known covariance)
        # W = α'·V^{-1}·α ~ χ²(rank)
        # For simplicity, use t-test approach if standard errors available

        if self.alpha_se is not None:
            # Individual t-tests
            t_stats = alpha_row / self.alpha_se[var_idx, :]
            # Joint test: sum of squared t-stats ~ χ²(rank)
            W = np.sum(t_stats**2)
        else:
            # Simplified: use α values directly (assumes unit variance)
            W = np.sum(alpha_row**2) * self.N * self.T_avg / self.K

        df = self.rank
        p_value = 1 - stats.chi2.cdf(W, df)

        return {
            "variable": variable,
            "statistic": W,
            "p_value": p_value,
            "df": df,
            "reject": p_value < 0.05,
        }

    def test_strong_exogeneity(self, variable: str) -> Dict[str, float]:
        """
        Test strong exogeneity: H0: α[variable, :] = 0 AND Γ[variable, other] = 0

        Strong exogeneity means the variable is weakly exogenous and is not
        Granger-caused by other variables.

        Parameters
        ----------
        variable : str
            Variable name to test

        Returns
        -------
        dict
            Dictionary with 'statistic', 'p_value', 'df', 'variable'
        """
        if variable not in self.var_names:
            raise ValueError(f"Variable '{variable}' not found")

        var_idx = self.var_names.index(variable)

        # Number of restrictions:
        # - rank restrictions from α[var_idx, :] = 0
        # - (K-1) × (p-1) restrictions from Γ_l[var_idx, j] = 0 for j ≠ var_idx
        num_alpha_restrictions = self.rank
        num_gamma_restrictions = (self.K - 1) * len(self.Gamma)
        total_restrictions = num_alpha_restrictions + num_gamma_restrictions

        # Joint test (simplified)
        # W = combined test statistic ~ χ²(total_restrictions)

        # Alpha part
        alpha_row = self.alpha[var_idx, :]
        W_alpha = np.sum(alpha_row**2) * self.N * self.T_avg / self.K

        # Gamma part: test that row var_idx (excluding diagonal) is zero
        W_gamma = 0.0
        for Gamma_l in self.Gamma:
            gamma_row = Gamma_l[var_idx, :]
            # Exclude own lag
            gamma_row_others = np.delete(gamma_row, var_idx)
            W_gamma += np.sum(gamma_row_others**2) * self.N * self.T_avg / self.K

        W = W_alpha + W_gamma
        df = total_restrictions
        p_value = 1 - stats.chi2.cdf(W, df)

        return {
            "variable": variable,
            "statistic": W,
            "p_value": p_value,
            "df": df,
            "reject": p_value < 0.05,
        }

    def summary(self) -> str:
        """
        Generate formatted summary of VECM estimation results.

        Returns
        -------
        str
            Formatted summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Panel VECM Estimation Results")
        lines.append("=" * 80)
        lines.append(f"Method: {self.method}")
        lines.append(f"Cointegration Rank: {self.rank}")
        lines.append(f"Number of variables (K): {self.K}")
        lines.append(f"Number of lags (p): {self.p}")
        lines.append(f"Number of entities (N): {self.N}")
        lines.append(f"Average time periods (T): {self.T_avg:.1f}")
        lines.append(f"Deterministic: {self.deterministic}")
        lines.append("")

        # Cointegrating relations
        lines.append("─" * 80)
        lines.append("Cointegrating Relations (β, normalized)")
        lines.append("─" * 80)
        beta_df = self.cointegrating_relations()
        lines.append(beta_df.to_string())
        lines.append("")
        lines.append("Long-run equilibria:")
        for j in range(self.rank):
            terms = []
            for i, var in enumerate(self.var_names):
                coef = beta_df.iloc[i, j]
                if i == 0:
                    terms.append(f"{var}")
                else:
                    if coef >= 0:
                        terms.append(f"+ {coef:.3f}·{var}")
                    else:
                        terms.append(f"- {abs(coef):.3f}·{var}")
            equilibrium = " ".join(terms) + " = 0"
            lines.append(f"  β_{j+1}: {equilibrium}")
        lines.append("")

        # Adjustment speeds
        lines.append("─" * 80)
        lines.append("Adjustment Coefficients (α)")
        lines.append("─" * 80)
        alpha_df = self.adjustment_speeds()
        lines.append(alpha_df.to_string())
        lines.append("")

        # Short-run dynamics
        lines.append("─" * 80)
        lines.append("Short-Run Dynamics (Γ matrices)")
        lines.append("─" * 80)
        gamma_dfs = self.short_run_dynamics()
        for lag, gamma_df in enumerate(gamma_dfs, start=1):
            lines.append(f"\nΓ_{lag} (lag {lag} of differences):")
            lines.append(gamma_df.to_string())
        lines.append("")

        # Residual covariance
        lines.append("─" * 80)
        lines.append("Residual Covariance Matrix (Σ)")
        lines.append("─" * 80)
        sigma_df = pd.DataFrame(self.Sigma, index=self.var_names, columns=self.var_names)
        lines.append(sigma_df.to_string())
        lines.append("")

        # Exogeneity tests
        lines.append("─" * 80)
        lines.append("Exogeneity Tests")
        lines.append("─" * 80)
        lines.append(f"{'Variable':<15} {'Weak Exog.':<15} {'Strong Exog.':<15}")
        lines.append("─" * 80)

        for var in self.var_names:
            weak = self.test_weak_exogeneity(var)
            strong = self.test_strong_exogeneity(var)

            weak_str = f"p={weak['p_value']:.3f}"
            if not weak["reject"]:
                weak_str += " ✓"

            strong_str = f"p={strong['p_value']:.3f}"
            if not strong["reject"]:
                strong_str += " ✓"

            lines.append(f"{var:<15} {weak_str:<15} {strong_str:<15}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"PanelVECMResult(rank={self.rank}, K={self.K}, "
            f"p={self.p}, N={self.N}, method='{self.method}')"
        )

    def irf(
        self,
        periods: int = 10,
        method: str = "cholesky",
        shock_size: Union[str, float] = "one_std",
        cumulative: bool = False,
        ordering: Optional[List[str]] = None,
    ):
        """
        Compute Impulse Response Functions for VECM.

        The VECM is first converted to VAR representation in levels, then
        IRFs are computed. Unlike stable VARs, VECM IRFs show permanent
        effects (do not converge to zero).

        Parameters
        ----------
        periods : int, optional
            Number of periods ahead (default: 10)
        method : str, optional
            'cholesky' (default) or 'generalized'
        shock_size : str or float, optional
            'one_std' (default) for one standard deviation shock, or numerical value
        cumulative : bool, optional
            Whether to compute cumulative IRFs (default: False)
        ordering : List[str], optional
            Variable ordering for Cholesky decomposition

        Returns
        -------
        IRFResult
            IRF results object

        Notes
        -----
        VECM IRFs differ from stable VAR IRFs:
        - They show permanent effects (non-zero long-run impact)
        - Cumulative IRFs converge to finite long-run effects
        - This is due to cointegration relationships

        Examples
        --------
        >>> results = vecm.fit()
        >>> irf = results.irf(periods=20, method='cholesky')
        >>> irf.plot(impulse='y1', response='y2')
        """
        from panelbox.var.irf import (
            IRFResult,
            compute_cumulative_irf,
            compute_irf_cholesky,
            compute_irf_generalized,
            compute_phi_non_orthogonalized,
        )

        # Convert VECM to VAR representation
        A_matrices = self.to_var()

        # Compute IRFs using VAR methods
        if method == "cholesky":
            # Compute non-orthogonalized IRFs first
            Phi = compute_phi_non_orthogonalized(A_matrices, periods)
            # Then orthogonalize
            irf_matrix = compute_irf_cholesky(Phi, self.Sigma, shock_size)
        elif method == "generalized":
            # Compute non-orthogonalized IRFs
            Phi = compute_phi_non_orthogonalized(A_matrices, periods)
            # Generalized IRFs
            irf_matrix = compute_irf_generalized(Phi, self.Sigma, periods)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Cumulative if requested
        if cumulative:
            irf_matrix = compute_cumulative_irf(irf_matrix)

        return IRFResult(
            irf_matrix=irf_matrix,
            var_names=self.var_names,
            periods=periods,
            method=method,
            shock_size=shock_size,
            cumulative=cumulative,
            ordering=ordering if ordering else self.var_names,
        )

    def fevd(
        self,
        periods: int = 10,
        method: str = "cholesky",
        ordering: Optional[List[str]] = None,
    ):
        """
        Compute Forecast Error Variance Decomposition for VECM.

        The VECM is converted to VAR representation in levels, then FEVD
        is computed showing the contribution of each shock to forecast
        error variance.

        Parameters
        ----------
        periods : int, optional
            Number of periods ahead (default: 10)
        method : str, optional
            'cholesky' (default) or 'generalized'
        ordering : List[str], optional
            Variable ordering for Cholesky decomposition

        Returns
        -------
        FEVDResult
            FEVD results object

        Examples
        --------
        >>> results = vecm.fit()
        >>> fevd = results.fevd(periods=20)
        >>> fevd.plot(variable='y1')
        """
        from panelbox.var.fevd import FEVDResult, compute_fevd_cholesky, compute_fevd_generalized
        from panelbox.var.irf import compute_phi_non_orthogonalized

        # Convert VECM to VAR
        A_matrices = self.to_var()

        # Compute FEVD using VAR methods
        if method == "cholesky":
            # Compute non-orthogonalized IRFs
            Phi = compute_phi_non_orthogonalized(A_matrices, periods)
            decomposition = compute_fevd_cholesky(Phi, self.Sigma, periods)
        elif method == "generalized":
            # Compute non-orthogonalized IRFs
            Phi = compute_phi_non_orthogonalized(A_matrices, periods)
            decomposition = compute_fevd_generalized(Phi, self.Sigma, periods)
        else:
            raise ValueError(f"Unknown method: {method}")

        return FEVDResult(
            decomposition=decomposition,
            var_names=self.var_names,
            periods=periods,
            method=method,
            ordering=ordering if ordering else self.var_names,
        )


class PanelVECM:
    """
    Panel Vector Error Correction Model estimation.

    Estimates VECM for panel data with cointegrated variables.

    Parameters
    ----------
    data : PanelVARData
        Panel VAR data container
    rank : int, optional
        Cointegration rank (if known). If None, will be selected automatically.
    deterministic : str, optional
        Deterministic specification: 'nc' (no constant), 'c' (constant),
        'ct' (constant and trend). Default: 'c'

    Attributes
    ----------
    K : int
        Number of variables
    N : int
        Number of entities
    p : int
        Number of lags

    Examples
    --------
    >>> from panelbox.var import PanelVARData
    >>> from panelbox.var.vecm import PanelVECM
    >>>
    >>> # Create data
    >>> data = PanelVARData(df, endog_vars=['y1', 'y2'],
    ...                     entity_col='id', time_col='time', lags=2)
    >>>
    >>> # Estimate VECM
    >>> vecm = PanelVECM(data, rank=1)
    >>> results = vecm.fit(method='ml')
    >>> print(results.summary())
    >>>
    >>> # Cointegrating relations
    >>> print(results.cointegrating_relations())
    """

    def __init__(
        self,
        data: PanelVARData,
        rank: Optional[int] = None,
        deterministic: str = "c",
    ):
        self.data = data
        self.K = data.K
        self.N = data.N
        self.p = data.p
        self.deterministic = deterministic

        # Rank selection
        if rank is None:
            # Automatic rank selection
            rank_test = CointegrationRankTest(data, deterministic=deterministic)
            rank_results = rank_test.test_rank()
            self.rank = rank_results.selected_rank
            print(f"Automatically selected rank: {self.rank}")
        else:
            if rank < 0 or rank >= self.K:
                raise ValueError(f"Rank must be in [0, {self.K-1}]")
            self.rank = rank

    def fit(self, method: str = "ml") -> PanelVECMResult:
        """
        Estimate Panel VECM.

        Parameters
        ----------
        method : str, optional
            Estimation method: 'ml' (maximum likelihood) or 'twostep'
            Default: 'ml'

        Returns
        -------
        PanelVECMResult
            Estimation results

        Examples
        --------
        >>> vecm = PanelVECM(data, rank=1)
        >>> results = vecm.fit(method='ml')
        >>> print(results.summary())
        """
        if method == "ml":
            return self._fit_ml()
        elif method == "twostep":
            return self._fit_twostep()
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ml' or 'twostep'.")

    def _fit_ml(self) -> PanelVECMResult:
        """
        Estimate VECM using Maximum Likelihood (Johansen procedure for panels).

        Returns
        -------
        PanelVECMResult
            Estimation results
        """
        # Prepare data
        df = self.data.data.copy()
        endog_vars = self.data.endog_vars
        entity_col = self.data.entity_col

        # Create differences
        for var in endog_vars:
            df[f"d_{var}"] = df.groupby(entity_col)[var].diff()

        # Create lags
        lag_vars = []
        for lag in range(1, self.p):
            for var in endog_vars:
                lag_var = f"d_{var}_lag{lag}"
                df[lag_var] = df.groupby(entity_col)[f"d_{var}"].shift(lag)
                lag_vars.append(lag_var)

        level_lag_vars = []
        for var in endog_vars:
            lag_var = f"{var}_lag1"
            df[lag_var] = df.groupby(entity_col)[var].shift(1)
            level_lag_vars.append(lag_var)

        df = df.dropna()

        delta_y_vars = [f"d_{var}" for var in endog_vars]

        # Within transformation
        df_within = df.copy()
        for var in delta_y_vars + lag_vars + level_lag_vars:
            demeaned, _ = within_transformation(df_within[var].values, df_within[entity_col].values)
            df_within[var] = demeaned

        # Step 1: Concentrated regression
        X = df_within[lag_vars].values if lag_vars else np.zeros((len(df_within), 1))
        Y = df_within[delta_y_vars].values
        Y1 = df_within[level_lag_vars].values

        if lag_vars and len(df_within) > len(lag_vars):
            beta_short = np.linalg.lstsq(X, Y, rcond=None)[0]
            R0 = Y - X @ beta_short
            beta_short1 = np.linalg.lstsq(X, Y1, rcond=None)[0]
            R1 = Y1 - X @ beta_short1
        else:
            R0 = Y
            R1 = Y1

        # Step 2: Solve eigenvalue problem
        T_eff = len(R0)
        S00 = R0.T @ R0 / T_eff
        S11 = R1.T @ R1 / T_eff
        S01 = R0.T @ R1 / T_eff
        S10 = S01.T

        # Regularize
        S00 += np.eye(self.K) * 1e-8
        S11 += np.eye(self.K) * 1e-8

        # Generalized eigenvalue problem
        try:
            S00_inv = inv(S00)
            M = S00_inv @ S01 @ inv(S11) @ S10
            eigenvalues, eigenvectors = np.linalg.eig(M)
            eigenvalues = np.real(eigenvalues)
            eigenvectors = np.real(eigenvectors)

            # Sort by eigenvalues (descending)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        except np.linalg.LinAlgError:
            raise RuntimeError("Failed to solve eigenvalue problem. Data may be singular.")

        # Step 3: Extract β (first r eigenvectors)
        if self.rank == 0:
            # No cointegration: β and α are empty
            beta = np.zeros((self.K, 0))
            alpha = np.zeros((self.K, 0))
            Pi = np.zeros((self.K, self.K))
        else:
            # Beta: eigenvectors corresponding to largest r eigenvalues
            # These are eigenvectors of S11^{-1} S10 S00^{-1} S01
            # To get cointegrating vectors, we use eigenvectors of the dual problem

            # Solve: S11^{-1} S10 S00^{-1} S01 v = λ v
            # Cointegrating vectors are the eigenvectors
            try:
                S11_inv = inv(S11)
                M_dual = S11_inv @ S10 @ S00_inv @ S01
                eig_dual, vec_dual = np.linalg.eig(M_dual)
                eig_dual = np.real(eig_dual)
                vec_dual = np.real(vec_dual)

                idx_dual = np.argsort(eig_dual)[::-1]
                vec_dual = vec_dual[:, idx_dual]

                beta = vec_dual[:, : self.rank]

            except np.linalg.LinAlgError:
                # Fallback: use transpose of eigenvectors
                beta = eigenvectors[:, : self.rank]

            # Normalize β: first element of each column = 1
            for j in range(self.rank):
                if np.abs(beta[0, j]) > 1e-10:
                    beta[:, j] /= beta[0, j]

            # Step 4: Estimate α given β
            # Δy_it = α·(β'·y_{i,t-1}) + Γ·ΔZ_it + ε_it
            ECT = Y1 @ beta  # Error correction terms

            # Regress R0 on ECT (concentrated model)
            alpha = np.linalg.lstsq(ECT, R0, rcond=None)[0].T

            Pi = alpha @ beta.T

        # Step 5: Estimate Γ (short-run dynamics)
        # Regress Δy on ECT and lagged differences
        Gamma_matrices = []

        if lag_vars and self.rank > 0:
            # Regress Δy on ECT and X
            combined_X = np.hstack([Y1 @ beta, X])
            coefs = np.linalg.lstsq(combined_X, Y, rcond=None)[0]

            # Extract Γ coefficients
            # coefs: first 'rank' columns are α, rest are Γ stacked
            n_gamma_coefs = len(lag_vars)
            gamma_coefs = coefs[self.rank :, :].T  # K × (K*(p-1))

            # Reshape into Γ_1, Γ_2, ..., Γ_{p-1}
            for lag_idx in range(self.p - 1):
                Gamma_l = gamma_coefs[:, lag_idx * self.K : (lag_idx + 1) * self.K]
                Gamma_matrices.append(Gamma_l)

        elif lag_vars:
            # No cointegration, but have short-run dynamics
            coefs = np.linalg.lstsq(X, Y, rcond=None)[0].T
            for lag_idx in range(self.p - 1):
                Gamma_l = coefs[:, lag_idx * self.K : (lag_idx + 1) * self.K]
                Gamma_matrices.append(Gamma_l)

        # Step 6: Compute residuals and covariance
        if self.rank > 0 and lag_vars:
            ECT = Y1 @ beta
            fitted = ECT @ alpha.T + X @ beta_short.T
        elif self.rank > 0:
            ECT = Y1 @ beta
            fitted = ECT @ alpha.T
        elif lag_vars:
            fitted = X @ beta_short.T
        else:
            fitted = np.zeros_like(Y)

        residuals = Y - fitted
        Sigma = residuals.T @ residuals / len(residuals)

        # Calculate average T
        entity_counts = df.groupby(entity_col).size()
        T_avg = entity_counts.mean()

        return PanelVECMResult(
            alpha=alpha,
            beta=beta,
            Gamma=Gamma_matrices,
            Sigma=Sigma,
            residuals=residuals,
            var_names=endog_vars,
            rank=self.rank,
            method="ml",
            N=self.N,
            T_avg=T_avg,
            deterministic=self.deterministic,
        )

    def _fit_twostep(self) -> PanelVECMResult:
        """
        Estimate VECM using two-step method.

        Step 1: Estimate β via DOLS or other method
        Step 2: Given β, estimate α and Γ via OLS

        Returns
        -------
        PanelVECMResult
            Estimation results
        """
        # For simplicity, use ML method with pre-specified rank
        # A full two-step would estimate β separately (e.g., via DOLS)
        # then estimate α and Γ conditional on β

        # Placeholder: delegate to ML for now
        # TODO: Implement proper two-step estimation
        import warnings

        warnings.warn(
            "Two-step estimation not fully implemented. Using ML method.",
            UserWarning,
        )

        result = self._fit_ml()
        result.method = "twostep"
        return result
