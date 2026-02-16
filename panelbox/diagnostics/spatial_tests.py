"""
Spatial diagnostic tests for panel data models.

This module implements Lagrange Multiplier (LM) tests for spatial dependence in
panel data models, including both standard and robust versions of lag and error tests.

References
----------
Anselin, L. (1988). "Spatial Econometrics: Methods and Models."
    Kluwer Academic Publishers.
Anselin, L., Bera, A.K., Florax, R., & Yoon, M.J. (1996).
    "Simple diagnostic tests for spatial dependence."
    Regional Science and Urban Economics, 26(1), 77-104.
Anselin, L., & Rey, S.J. (2014). "Modern Spatial Econometrics in Practice:
    A Guide to GeoDa, GeoDaSpace and PySAL." GeoDa Press LLC.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class LMTestResult:
    """
    Results from an LM test for spatial dependence.

    Attributes
    ----------
    test_name : str
        Name of the test (e.g., 'LM-Lag', 'LM-Error')
    statistic : float
        Test statistic value
    pvalue : float
        P-value from chi-squared distribution
    df : int
        Degrees of freedom
    conclusion : str
        Automatic conclusion based on significance level
    """

    test_name: str
    statistic: float
    pvalue: float
    df: int
    conclusion: str

    def summary(self) -> str:
        """
        Print formatted summary of test results.

        Returns
        -------
        str
            Formatted summary string
        """
        return (
            f"{self.test_name}\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.pvalue:.4f}\n"
            f"  df: {self.df}\n"
            f"  Conclusion: {self.conclusion}"
        )


def lm_lag_test(residuals: np.ndarray, X: np.ndarray, W: np.ndarray, **kwargs) -> LMTestResult:
    """
    LM test for spatial lag dependence.

    Tests H0: rho = 0 in the spatial autoregressive model:
        y = rho*W*y + X*beta + epsilon

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals from non-spatial model
    X : np.ndarray
        Design matrix including intercept
    W : np.ndarray
        Spatial weight matrix (should be row-normalized).
        For panel data, W should be n_entities x n_entities, and the function
        will automatically expand it to handle the temporal dimension.

    Returns
    -------
    LMTestResult
        Test results with statistic, p-value, and conclusion

    References
    ----------
    Anselin (1988), equation 6.1
    Anselin et al. (1996), equation 8

    Notes
    -----
    The test statistic follows a chi-squared distribution with 1 degree of freedom
    under the null hypothesis of no spatial lag dependence.

    For panel data with T time periods and N spatial units, W is expanded
    to (N*T) x (N*T) by taking the Kronecker product with I_T.
    """
    n_obs = len(residuals)
    n_entities = W.shape[0]

    # Check if this is panel data
    if n_obs > n_entities:
        # Panel data: expand W using Kronecker product with identity matrix
        n_time = n_obs // n_entities
        if n_obs != n_entities * n_time:
            raise ValueError(
                f"Residuals length ({n_obs}) must be divisible by W dimension ({n_entities})"
            )
        # W_full = I_T ⊗ W
        I_T = np.eye(n_time)
        W_full = np.kron(I_T, W)
    else:
        W_full = W

    n = len(residuals)
    sigma2 = np.sum(residuals**2) / n

    # Compute trace terms
    W2 = W_full @ W_full
    tr_W2_Wt = np.trace(W_full @ W_full.T + W2)

    # Compute numerator: (e'*W*e / sigma2)^2
    Wy_resid = W_full @ residuals
    numerator = (residuals.T @ Wy_resid / sigma2) ** 2

    # Compute denominator with adjustment for X
    # D = tr(W'W + W^2) + tr(W*X*(X'X)^{-1}*X'*W')
    XtX_inv = np.linalg.inv(X.T @ X)
    M_adjustment = np.trace(W_full @ X @ XtX_inv @ X.T @ W_full.T)
    denominator = tr_W2_Wt + M_adjustment

    lm_stat = numerator / denominator
    pvalue = 1 - stats.chi2.cdf(lm_stat, 1)

    conclusion = "Reject H0: No spatial lag" if pvalue < 0.05 else "Cannot reject H0"

    return LMTestResult(
        test_name="LM-Lag",
        statistic=lm_stat,
        pvalue=pvalue,
        df=1,
        conclusion=conclusion,
    )


def lm_error_test(residuals: np.ndarray, X: np.ndarray, W: np.ndarray, **kwargs) -> LMTestResult:
    """
    LM test for spatial error dependence.

    Tests H0: lambda = 0 in the spatial error model:
        y = X*beta + epsilon
        epsilon = lambda*W*epsilon + u

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals from non-spatial model
    X : np.ndarray
        Design matrix including intercept
    W : np.ndarray
        Spatial weight matrix (should be row-normalized).
        For panel data, W should be n_entities x n_entities, and the function
        will automatically expand it to handle the temporal dimension.

    Returns
    -------
    LMTestResult
        Test results with statistic, p-value, and conclusion

    References
    ----------
    Anselin (1988), equation 6.2
    Anselin et al. (1996), equation 9

    Notes
    -----
    The test statistic follows a chi-squared distribution with 1 degree of freedom
    under the null hypothesis of no spatial error dependence.

    For panel data with T time periods and N spatial units, W is expanded
    to (N*T) x (N*T) by taking the Kronecker product with I_T.
    """
    n_obs = len(residuals)
    n_entities = W.shape[0]

    # Check if this is panel data
    if n_obs > n_entities:
        # Panel data: expand W using Kronecker product with identity matrix
        n_time = n_obs // n_entities
        if n_obs != n_entities * n_time:
            raise ValueError(
                f"Residuals length ({n_obs}) must be divisible by W dimension ({n_entities})"
            )
        # W_full = I_T ⊗ W
        I_T = np.eye(n_time)
        W_full = np.kron(I_T, W)
    else:
        W_full = W

    n = len(residuals)
    sigma2 = np.sum(residuals**2) / n

    # Compute trace term
    W2 = W_full @ W_full
    tr_W2_Wt = np.trace(W_full @ W_full.T + W2)

    # Compute LM statistic
    We = W_full @ residuals
    numerator = (residuals.T @ We / sigma2) ** 2
    denominator = tr_W2_Wt

    lm_stat = numerator / denominator
    pvalue = 1 - stats.chi2.cdf(lm_stat, 1)

    conclusion = "Reject H0: No spatial error" if pvalue < 0.05 else "Cannot reject H0"

    return LMTestResult(
        test_name="LM-Error",
        statistic=lm_stat,
        pvalue=pvalue,
        df=1,
        conclusion=conclusion,
    )


def robust_lm_lag_test(
    residuals: np.ndarray, X: np.ndarray, W: np.ndarray, **kwargs
) -> LMTestResult:
    """
    Robust LM test for spatial lag, controlling for spatial error.

    Tests H0: rho = 0 conditional on the presence of spatial error dependence.
    This test is robust to misspecification of the alternative hypothesis.

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals from non-spatial model
    X : np.ndarray
        Design matrix including intercept
    W : np.ndarray
        Spatial weight matrix (should be row-normalized).
        For panel data, W should be n_entities x n_entities, and the function
        will automatically expand it to handle the temporal dimension.

    Returns
    -------
    LMTestResult
        Test results with statistic, p-value, and conclusion

    References
    ----------
    Anselin et al. (1996), equation 12

    Notes
    -----
    The robust test is particularly useful when both spatial lag and error
    dependence may be present. It follows a chi-squared distribution with
    1 degree of freedom under the null hypothesis.

    For panel data with T time periods and N spatial units, W is expanded
    to (N*T) x (N*T) by taking the Kronecker product with I_T.
    """
    n_obs = len(residuals)
    n_entities = W.shape[0]

    # Check if this is panel data
    if n_obs > n_entities:
        # Panel data: expand W using Kronecker product with identity matrix
        n_time = n_obs // n_entities
        if n_obs != n_entities * n_time:
            raise ValueError(
                f"Residuals length ({n_obs}) must be divisible by W dimension ({n_entities})"
            )
        # W_full = I_T ⊗ W
        I_T = np.eye(n_time)
        W_full = np.kron(I_T, W)
    else:
        W_full = W

    n = len(residuals)
    sigma2 = np.sum(residuals**2) / n

    # Compute trace terms
    W2 = W_full @ W_full
    tr_W2_Wt = np.trace(W_full @ W_full.T + W2)

    # Compute components
    We = W_full @ residuals
    Wy_term = residuals.T @ (W_full @ residuals) / sigma2
    We_term = residuals.T @ We / sigma2

    # Adjustment for X in denominator
    XtX_inv = np.linalg.inv(X.T @ X)
    M_adjustment = np.trace(W_full @ X @ XtX_inv @ X.T @ W_full.T)
    D = tr_W2_Wt + M_adjustment

    # Robust statistic: (Wy_term - We_term)^2 / adjusted_denominator
    numerator = (Wy_term - We_term) ** 2
    denominator = D - tr_W2_Wt**2 / tr_W2_Wt

    lm_stat = numerator / denominator
    pvalue = 1 - stats.chi2.cdf(lm_stat, 1)

    conclusion = "Reject H0" if pvalue < 0.05 else "Cannot reject H0"

    return LMTestResult(
        test_name="Robust LM-Lag",
        statistic=lm_stat,
        pvalue=pvalue,
        df=1,
        conclusion=conclusion,
    )


def robust_lm_error_test(
    residuals: np.ndarray, X: np.ndarray, W: np.ndarray, **kwargs
) -> LMTestResult:
    """
    Robust LM test for spatial error, controlling for spatial lag.

    Tests H0: lambda = 0 conditional on the presence of spatial lag dependence.
    This test is robust to misspecification of the alternative hypothesis.

    Parameters
    ----------
    residuals : np.ndarray
        OLS residuals from non-spatial model
    X : np.ndarray
        Design matrix including intercept
    W : np.ndarray
        Spatial weight matrix (should be row-normalized).
        For panel data, W should be n_entities x n_entities, and the function
        will automatically expand it to handle the temporal dimension.

    Returns
    -------
    LMTestResult
        Test results with statistic, p-value, and conclusion

    References
    ----------
    Anselin et al. (1996), equation 13

    Notes
    -----
    The robust test is particularly useful when both spatial lag and error
    dependence may be present. It follows a chi-squared distribution with
    1 degree of freedom under the null hypothesis.

    For panel data with T time periods and N spatial units, W is expanded
    to (N*T) x (N*T) by taking the Kronecker product with I_T.
    """
    n_obs = len(residuals)
    n_entities = W.shape[0]

    # Check if this is panel data
    if n_obs > n_entities:
        # Panel data: expand W using Kronecker product with identity matrix
        n_time = n_obs // n_entities
        if n_obs != n_entities * n_time:
            raise ValueError(
                f"Residuals length ({n_obs}) must be divisible by W dimension ({n_entities})"
            )
        # W_full = I_T ⊗ W
        I_T = np.eye(n_time)
        W_full = np.kron(I_T, W)
    else:
        W_full = W

    n = len(residuals)
    sigma2 = np.sum(residuals**2) / n

    # Compute trace terms
    W2 = W_full @ W_full
    tr_W2_Wt = np.trace(W_full @ W_full.T + W2)

    # Compute components
    We = W_full @ residuals
    Wy_term = residuals.T @ (W_full @ residuals) / sigma2
    We_term = residuals.T @ We / sigma2

    # Adjustment for X
    XtX_inv = np.linalg.inv(X.T @ X)
    M_adjustment = np.trace(W_full @ X @ XtX_inv @ X.T @ W_full.T)
    D = tr_W2_Wt + M_adjustment

    # Robust statistic for error: (We_term - (tr_W2_Wt/D)*Wy_term)^2 / adjusted_denominator
    numerator = (We_term - (tr_W2_Wt / D) * Wy_term) ** 2
    denominator = tr_W2_Wt * (1 - tr_W2_Wt / D)

    lm_stat = numerator / denominator
    pvalue = 1 - stats.chi2.cdf(lm_stat, 1)

    conclusion = "Reject H0" if pvalue < 0.05 else "Cannot reject H0"

    return LMTestResult(
        test_name="Robust LM-Error",
        statistic=lm_stat,
        pvalue=pvalue,
        df=1,
        conclusion=conclusion,
    )


def run_lm_tests(
    model_result, W: np.ndarray, alpha: float = 0.05
) -> Dict[str, Union[LMTestResult, str, pd.DataFrame]]:
    """
    Run all LM tests and provide model recommendation using decision tree.

    This function implements the decision tree approach from Anselin & Rey (2014)
    to guide model selection based on LM test results.

    Parameters
    ----------
    model_result : regression result object
        OLS model result with residuals and exogenous variables.
        Must have attributes: resid, model.exog
    W : np.ndarray
        Spatial weight matrix (should be row-normalized)
    alpha : float, optional
        Significance level for hypothesis tests (default: 0.05)

    Returns
    -------
    dict
        Dictionary containing:
        - 'lm_lag': LMTestResult for LM-Lag test
        - 'lm_error': LMTestResult for LM-Error test
        - 'robust_lm_lag': LMTestResult for Robust LM-Lag test
        - 'robust_lm_error': LMTestResult for Robust LM-Error test
        - 'recommendation': str, recommended model type
        - 'reason': str, explanation for recommendation
        - 'summary': pd.DataFrame, summary table of all tests

    References
    ----------
    Anselin, L., & Rey, S.J. (2014). "Modern Spatial Econometrics in Practice"

    Notes
    -----
    Decision tree logic:
    1. If only LM-Lag is significant -> SAR (Spatial Lag Model)
    2. If only LM-Error is significant -> SEM (Spatial Error Model)
    3. If both significant -> use robust tests to discriminate
    4. If neither significant -> No spatial dependence
    """
    residuals = model_result.resid
    X = model_result.model.exog

    results = {}
    results["lm_lag"] = lm_lag_test(residuals, X, W)
    results["lm_error"] = lm_error_test(residuals, X, W)
    results["robust_lm_lag"] = robust_lm_lag_test(residuals, X, W)
    results["robust_lm_error"] = robust_lm_error_test(residuals, X, W)

    # Decision tree from Anselin & Rey (2014)
    lm_lag_sig = results["lm_lag"].pvalue < alpha
    lm_error_sig = results["lm_error"].pvalue < alpha

    if lm_lag_sig and not lm_error_sig:
        results["recommendation"] = "SAR (Spatial Lag Model)"
        results["reason"] = "LM-Lag significant, LM-Error not significant"
    elif not lm_lag_sig and lm_error_sig:
        results["recommendation"] = "SEM (Spatial Error Model)"
        results["reason"] = "LM-Error significant, LM-Lag not significant"
    elif lm_lag_sig and lm_error_sig:
        # Both significant, use robust tests
        robust_lag_sig = results["robust_lm_lag"].pvalue < alpha
        robust_error_sig = results["robust_lm_error"].pvalue < alpha

        if robust_lag_sig and not robust_error_sig:
            results["recommendation"] = "SAR (Spatial Lag Model)"
            results["reason"] = "Robust LM-Lag significant"
        elif not robust_lag_sig and robust_error_sig:
            results["recommendation"] = "SEM (Spatial Error Model)"
            results["reason"] = "Robust LM-Error significant"
        else:
            results["recommendation"] = "SDM or GNS"
            results["reason"] = "Both robust tests significant - consider Spatial Durbin Model"
    else:
        results["recommendation"] = "No spatial dependence"
        results["reason"] = "Neither LM test significant"

    # Create summary DataFrame
    test_list = [
        results["lm_lag"],
        results["lm_error"],
        results["robust_lm_lag"],
        results["robust_lm_error"],
    ]

    results["summary"] = pd.DataFrame(
        {
            "Test": [t.test_name for t in test_list],
            "Statistic": [t.statistic for t in test_list],
            "p-value": [t.pvalue for t in test_list],
            "Significant": [t.pvalue < alpha for t in test_list],
        }
    )

    return results


@dataclass
class MoranIResult:
    """Results from Moran's I test."""

    statistic: float
    expected_value: float
    variance: float
    z_score: float
    pvalue: float
    conclusion: str
    additional_info: Dict

    def summary(self) -> str:
        """Print formatted summary."""
        return (
            f"Moran's I Test\n"
            f"  I statistic: {self.statistic:.4f}\n"
            f"  Expected I: {self.expected_value:.4f}\n"
            f"  Variance: {self.variance:.6f}\n"
            f"  Z-score: {self.z_score:.4f}\n"
            f"  P-value: {self.pvalue:.4f}\n"
            f"  Conclusion: {self.conclusion}\n"
            f"  Additional info: {self.additional_info}"
        )


class MoranIPanelTest:
    """
    Global Moran's I test for panel data residuals.

    Can test autocorrelation by period or pooled.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals (NT x 1)
    W : np.ndarray
        Spatial weight matrix (N x N)
    entity_ids : array-like
        Entity identifiers
    time_ids : array-like
        Time identifiers

    References
    ----------
    Anselin, L. (1995). "Local Indicators of Spatial Association—LISA."
        Geographical Analysis, 27(2), 93-115.
    """

    def __init__(self, residuals: np.ndarray, W: np.ndarray, entity_ids, time_ids):
        self.residuals = residuals
        self.W = W
        self.entity_ids = np.array(entity_ids)
        self.time_ids = np.array(time_ids)
        self.n_entities = len(np.unique(entity_ids))
        self.n_periods = len(np.unique(time_ids))

    def run(self, method: str = "pooled") -> Union[MoranIResult, Dict[str, MoranIResult]]:
        """
        Run Moran's I test.

        Parameters
        ----------
        method : {'pooled', 'by_period', 'average'}
            How to compute test statistic:
            - 'pooled': use time-averaged residuals by entity
            - 'by_period': compute I for each period separately
            - 'average': average of I statistics across periods

        Returns
        -------
        MoranIResult or dict of MoranIResult
            Test results
        """
        if method == "pooled":
            return self._test_pooled()
        elif method == "by_period":
            return self._test_by_period()
        elif method == "average":
            return self._test_average()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _test_pooled(self) -> MoranIResult:
        """Test using all residuals pooled (time-averaged by entity)."""
        # Compute average residuals by entity
        resid_by_entity = {}
        for i, eid in enumerate(self.entity_ids):
            if eid not in resid_by_entity:
                resid_by_entity[eid] = []
            resid_by_entity[eid].append(self.residuals[i])

        avg_resid = np.array(
            [np.mean(resid_by_entity[eid]) for eid in sorted(resid_by_entity.keys())]
        )

        return self._compute_moran(avg_resid)

    def _test_by_period(self) -> Dict[str, MoranIResult]:
        """Test for each period separately."""
        results = {}
        unique_times = np.unique(self.time_ids)

        for t in unique_times:
            mask = self.time_ids == t
            resid_t = self.residuals[mask]
            results[str(t)] = self._compute_moran(resid_t)

        return results

    def _test_average(self) -> MoranIResult:
        """Compute average Moran's I across periods."""
        by_period = self._test_by_period()

        avg_stat = np.mean([r.statistic for r in by_period.values()])
        avg_pvalue = np.mean([r.pvalue for r in by_period.values()])
        avg_z = np.mean([r.z_score for r in by_period.values()])

        return MoranIResult(
            statistic=avg_stat,
            expected_value=-1 / (self.n_entities - 1),
            variance=np.var([r.statistic for r in by_period.values()]),
            z_score=avg_z,
            pvalue=avg_pvalue,
            conclusion=(
                "Average across periods shows significant spatial autocorrelation"
                if avg_pvalue < 0.05
                else "No significant spatial autocorrelation on average"
            ),
            additional_info={"n_periods": self.n_periods, "method": "average"},
        )

    def _compute_moran(self, z: np.ndarray) -> MoranIResult:
        """
        Compute Moran's I statistic.

        Parameters
        ----------
        z : np.ndarray
            Values to test (e.g., residuals for N entities)

        Returns
        -------
        MoranIResult
        """
        n = len(z)

        # Standardize
        z_mean = np.mean(z)
        z_std = (z - z_mean) / np.std(z)

        # Spatial lag
        Wz = self.W @ z_std

        # Moran's I
        S0 = np.sum(self.W)
        I = (n / S0) * (z_std.T @ Wz) / (z_std.T @ z_std)

        # Expected value under randomization
        E_I = -1 / (n - 1)

        # Variance (normality assumption)
        S1 = 0.5 * np.sum((self.W + self.W.T) ** 2)
        S2 = np.sum((np.sum(self.W, axis=0) + np.sum(self.W, axis=1)) ** 2)

        # Variance formula (normality assumption)
        num = n * ((n**2 - 3 * n + 3) * S1 - n * S2 + 3 * S0**2)
        denom = (n - 1) * (n - 2) * (n - 3) * S0**2

        Var_I = num / denom - E_I**2

        # Z-score and p-value
        z_score = (I - E_I) / np.sqrt(Var_I)
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

        return MoranIResult(
            statistic=I,
            expected_value=E_I,
            variance=Var_I,
            z_score=z_score,
            pvalue=pvalue,
            conclusion=(
                "Significant spatial autocorrelation"
                if pvalue < 0.05
                else "No significant spatial autocorrelation"
            ),
            additional_info={"n": n, "S0": S0},
        )


@dataclass
class LISAResult:
    """Results from Local Moran's I analysis."""

    local_i: np.ndarray
    pvalues: np.ndarray
    z_values: np.ndarray
    Wz_values: np.ndarray
    entity_ids: np.ndarray

    def get_clusters(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Classify observations into cluster types.

        Parameters
        ----------
        alpha : float
            Significance level

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - entity_id: Entity identifier
            - Ii: Local Moran's I statistic
            - pvalue: Pseudo p-value from permutation
            - cluster_type: Classification
                'HH' - High-High (hot spot)
                'LL' - Low-Low (cold spot)
                'HL' - High-Low (outlier)
                'LH' - Low-High (outlier)
                'Not significant' - Not statistically significant
        """
        n = len(self.local_i)
        cluster_types = []

        for i in range(n):
            if self.pvalues[i] >= alpha:
                cluster_types.append("Not significant")
            elif self.z_values[i] > 0 and self.Wz_values[i] > 0:
                cluster_types.append("HH")  # Hot spot
            elif self.z_values[i] < 0 and self.Wz_values[i] < 0:
                cluster_types.append("LL")  # Cold spot
            elif self.z_values[i] > 0 and self.Wz_values[i] < 0:
                cluster_types.append("HL")  # High outlier
            else:
                cluster_types.append("LH")  # Low outlier

        return pd.DataFrame(
            {
                "entity_id": self.entity_ids,
                "Ii": self.local_i,
                "pvalue": self.pvalues,
                "cluster_type": cluster_types,
            }
        )

    def summary(self, alpha: float = 0.05) -> str:
        """Print summary of LISA results."""
        clusters = self.get_clusters(alpha)
        counts = clusters["cluster_type"].value_counts()

        summary_text = "Local Moran's I (LISA) Results\n"
        summary_text += f"Total observations: {len(self.local_i)}\n"
        summary_text += f"Significance level: {alpha}\n\n"
        summary_text += "Cluster types:\n"
        for cluster_type, count in counts.items():
            pct = 100 * count / len(self.local_i)
            summary_text += f"  {cluster_type}: {count} ({pct:.1f}%)\n"

        return summary_text


class LocalMoranI:
    """
    Local Indicators of Spatial Association (LISA).

    Computes local Moran's I for each observation to identify
    spatial clusters and outliers.

    Parameters
    ----------
    values : np.ndarray
        Values to analyze (e.g., residuals or variable values)
    W : np.ndarray
        Spatial weight matrix (N x N)
    entity_ids : array-like
        Entity identifiers

    References
    ----------
    Anselin, L. (1995). "Local Indicators of Spatial Association—LISA."
        Geographical Analysis, 27(2), 93-115.
    """

    def __init__(self, values: np.ndarray, W: np.ndarray, entity_ids):
        self.values = np.array(values)
        self.W = W
        self.entity_ids = np.array(entity_ids)

    def run(self, permutations: int = 999) -> LISAResult:
        """
        Compute Local Moran's I statistics.

        Parameters
        ----------
        permutations : int
            Number of permutations for pseudo p-values
            Default is 999 (recommended minimum)

        Returns
        -------
        LISAResult
            Local Moran's I results with p-values
        """
        n = len(self.values)

        # Standardize
        z_mean = np.mean(self.values)
        z_std_dev = np.std(self.values)
        z = (self.values - z_mean) / z_std_dev

        # Local Moran's I for each observation
        Wz = self.W @ z
        Ii = z * Wz

        # Permutation inference
        pvalues = np.zeros(n)

        # Set random seed for reproducibility
        np.random.seed(42)

        for i in range(n):
            # Count more extreme values under permutation
            extreme_count = 0

            for _ in range(permutations):
                # Permute all values
                z_perm = np.random.permutation(z)

                # Compute spatial lag for permuted data
                Wz_perm = self.W @ z_perm

                # Local I for this permutation
                Ii_perm = z_perm[i] * Wz_perm[i]

                # Check if more extreme
                if np.abs(Ii_perm) >= np.abs(Ii[i]):
                    extreme_count += 1

            # Pseudo p-value
            pvalues[i] = (extreme_count + 1) / (permutations + 1)

        return LISAResult(
            local_i=Ii,
            pvalues=pvalues,
            z_values=z,
            Wz_values=Wz,
            entity_ids=self.entity_ids,
        )
