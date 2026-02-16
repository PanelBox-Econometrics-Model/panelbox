"""
Diagnostic functions for standard errors tutorials.

This module provides diagnostic tests for heteroskedasticity, autocorrelation,
spatial correlation, and clustering issues in panel data.

Functions
---------
test_heteroskedasticity : White test and Breusch-Pagan test
test_autocorrelation : Durbin-Watson and Breusch-Godfrey tests
test_spatial_correlation : Moran's I test
cluster_diagnostics : Check cluster size and count
check_pcse_conditions : Verify T > N for PCSE

Author: PanelBox Development Team
Date: 2026-02-16
Version: 1.0.0
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """
    Container for diagnostic test results.

    Attributes
    ----------
    statistic : float
        Test statistic value
    p_value : float
        P-value for the test
    critical_value : float, optional
        Critical value at given significance level
    conclusion : str
        Text conclusion (reject/fail to reject null)
    details : dict
        Additional test details
    """

    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    conclusion: str = ""
    details: Dict = None

    def __str__(self):
        output = [
            f"Test Statistic: {self.statistic:.4f}",
            f"P-value: {self.p_value:.4f}",
        ]
        if self.critical_value is not None:
            output.append(f"Critical Value: {self.critical_value:.4f}")
        output.append(f"Conclusion: {self.conclusion}")
        return "\n".join(output)


def test_heteroskedasticity(
    residuals: np.ndarray, X: np.ndarray, test_type: str = "white"
) -> TestResult:
    """
    Test for heteroskedasticity using White or Breusch-Pagan test.

    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals
    X : np.ndarray
        Regressor matrix (n x k)
    test_type : str, optional
        Type of test: 'white' or 'breusch_pagan'

    Returns
    -------
    result : TestResult
        Test results with statistic, p-value, and conclusion
    """
    n = len(residuals)
    k = X.shape[1]

    if test_type.lower() == "white":
        # White test: regress squared residuals on X, X^2, and cross-products
        residuals_sq = residuals**2

        # Create auxiliary regressors: X, X^2, and cross-products
        aux_X = [X]  # Original regressors

        # Add squared terms
        for i in range(k):
            aux_X.append(X[:, i : i + 1] ** 2)

        # Add cross-products (upper triangle only to avoid perfect multicollinearity)
        for i in range(k):
            for j in range(i + 1, k):
                aux_X.append((X[:, i] * X[:, j]).reshape(-1, 1))

        aux_X = np.hstack(aux_X)

        # Add constant
        aux_X = np.column_stack([np.ones(n), aux_X])

        # Auxiliary regression
        try:
            beta_aux = np.linalg.lstsq(aux_X, residuals_sq, rcond=None)[0]
            fitted_aux = aux_X @ beta_aux
            ss_explained = np.sum((fitted_aux - np.mean(residuals_sq)) ** 2)
            ss_total = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
            r_squared = ss_explained / ss_total if ss_total > 0 else 0

            # Test statistic: n * R^2 ~ chi-squared(p)
            statistic = n * r_squared
            df = aux_X.shape[1] - 1  # Degrees of freedom
            p_value = 1 - stats.chi2.cdf(statistic, df)

            conclusion = (
                "Reject H0: Heteroskedasticity detected"
                if p_value < 0.05
                else "Fail to reject H0: No heteroskedasticity detected"
            )

            return TestResult(
                statistic=statistic,
                p_value=p_value,
                conclusion=conclusion,
                details={"test": "White", "df": df, "r_squared": r_squared, "n_obs": n},
            )
        except np.linalg.LinAlgError:
            return TestResult(
                statistic=np.nan,
                p_value=np.nan,
                conclusion="Test failed: Matrix singularity issue",
                details={"test": "White", "error": "Singular matrix"},
            )

    elif test_type.lower() == "breusch_pagan":
        # Breusch-Pagan test: regress squared residuals on X only
        residuals_sq = residuals**2

        # Standardize squared residuals
        sigma_sq = np.var(residuals, ddof=k)
        g = residuals_sq / sigma_sq

        # Add constant to X
        X_const = np.column_stack([np.ones(n), X])

        # Auxiliary regression
        try:
            beta_aux = np.linalg.lstsq(X_const, g, rcond=None)[0]
            fitted_aux = X_const @ beta_aux
            ss_explained = np.sum((fitted_aux - np.mean(g)) ** 2)

            # Test statistic: 0.5 * ESS ~ chi-squared(k)
            statistic = 0.5 * ss_explained
            df = k
            p_value = 1 - stats.chi2.cdf(statistic, df)

            conclusion = (
                "Reject H0: Heteroskedasticity detected"
                if p_value < 0.05
                else "Fail to reject H0: No heteroskedasticity detected"
            )

            return TestResult(
                statistic=statistic,
                p_value=p_value,
                conclusion=conclusion,
                details={"test": "Breusch-Pagan", "df": df, "n_obs": n},
            )
        except np.linalg.LinAlgError:
            return TestResult(
                statistic=np.nan,
                p_value=np.nan,
                conclusion="Test failed: Matrix singularity issue",
                details={"test": "Breusch-Pagan", "error": "Singular matrix"},
            )

    else:
        raise ValueError(f"Unknown test type: {test_type}. Use 'white' or 'breusch_pagan'")


def test_autocorrelation(
    residuals: np.ndarray,
    X: Optional[np.ndarray] = None,
    test_type: str = "durbin_watson",
    lags: int = 1,
) -> TestResult:
    """
    Test for autocorrelation using Durbin-Watson or Breusch-Godfrey test.

    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals (should be ordered by time)
    X : np.ndarray, optional
        Regressor matrix (required for Breusch-Godfrey test)
    test_type : str, optional
        Type of test: 'durbin_watson' or 'breusch_godfrey'
    lags : int, optional
        Number of lags for Breusch-Godfrey test

    Returns
    -------
    result : TestResult
        Test results with statistic, p-value, and conclusion
    """
    n = len(residuals)

    if test_type.lower() == "durbin_watson":
        # Durbin-Watson test: d = sum((e_t - e_{t-1})^2) / sum(e_t^2)
        diff = np.diff(residuals)
        numerator = np.sum(diff**2)
        denominator = np.sum(residuals**2)

        statistic = numerator / denominator if denominator > 0 else np.nan

        # Interpretation: DW H 2(1 - Á)
        # DW = 2: no autocorrelation
        # DW = 0: strong positive autocorrelation
        # DW = 4: strong negative autocorrelation
        rho_hat = 1 - statistic / 2

        if statistic < 1.5:
            conclusion = "Positive autocorrelation detected"
        elif statistic > 2.5:
            conclusion = "Negative autocorrelation detected"
        else:
            conclusion = "No significant autocorrelation"

        return TestResult(
            statistic=statistic,
            p_value=np.nan,  # Exact p-values require d_L and d_U critical values
            conclusion=conclusion,
            details={
                "test": "Durbin-Watson",
                "rho_estimate": rho_hat,
                "interpretation": "DW H 2 indicates no autocorrelation",
                "n_obs": n,
            },
        )

    elif test_type.lower() == "breusch_godfrey":
        if X is None:
            raise ValueError("X matrix required for Breusch-Godfrey test")

        # Breusch-Godfrey test: regress residuals on X and lagged residuals
        # Create lagged residuals
        lagged_resids = []
        for lag in range(1, lags + 1):
            lagged = np.zeros(n)
            lagged[lag:] = residuals[:-lag]
            lagged_resids.append(lagged)

        lagged_resids = np.column_stack(lagged_resids)

        # Auxiliary regression (exclude first `lags` observations)
        X_aux = np.column_stack([np.ones(n), X, lagged_resids])[lags:]
        y_aux = residuals[lags:]
        n_aux = len(y_aux)

        try:
            beta_aux = np.linalg.lstsq(X_aux, y_aux, rcond=None)[0]
            fitted_aux = X_aux @ beta_aux
            ss_explained = np.sum((fitted_aux - np.mean(y_aux)) ** 2)
            ss_total = np.sum((y_aux - np.mean(y_aux)) ** 2)
            r_squared = ss_explained / ss_total if ss_total > 0 else 0

            # Test statistic: (n - lags) * R^2 ~ chi-squared(lags)
            statistic = n_aux * r_squared
            df = lags
            p_value = 1 - stats.chi2.cdf(statistic, df)

            conclusion = (
                "Reject H0: Autocorrelation detected"
                if p_value < 0.05
                else "Fail to reject H0: No autocorrelation detected"
            )

            return TestResult(
                statistic=statistic,
                p_value=p_value,
                conclusion=conclusion,
                details={
                    "test": "Breusch-Godfrey",
                    "lags": lags,
                    "df": df,
                    "r_squared": r_squared,
                    "n_obs": n_aux,
                },
            )
        except np.linalg.LinAlgError:
            return TestResult(
                statistic=np.nan,
                p_value=np.nan,
                conclusion="Test failed: Matrix singularity issue",
                details={"test": "Breusch-Godfrey", "error": "Singular matrix"},
            )

    else:
        raise ValueError(
            f"Unknown test type: {test_type}. Use 'durbin_watson' or 'breusch_godfrey'"
        )


def test_spatial_correlation(
    residuals: np.ndarray, coords: np.ndarray, weights_matrix: Optional[np.ndarray] = None
) -> TestResult:
    """
    Test for spatial correlation using Moran's I.

    Parameters
    ----------
    residuals : np.ndarray
        Regression residuals
    coords : np.ndarray
        Spatial coordinates (n x 2): [latitude, longitude] or [x, y]
    weights_matrix : np.ndarray, optional
        Spatial weights matrix (n x n). If None, uses inverse distance weights

    Returns
    -------
    result : TestResult
        Test results with statistic, p-value, and conclusion
    """
    n = len(residuals)

    # Create spatial weights matrix if not provided
    if weights_matrix is None:
        # Use inverse distance weights
        from scipy.spatial.distance import cdist

        distances = cdist(coords, coords, metric="euclidean")

        # Avoid division by zero on diagonal
        np.fill_diagonal(distances, 1)
        weights_matrix = 1 / distances
        np.fill_diagonal(weights_matrix, 0)

    # Row-standardize weights
    row_sums = weights_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    W = weights_matrix / row_sums

    # Demean residuals
    e = residuals - np.mean(residuals)

    # Compute Moran's I
    numerator = e.T @ W @ e
    denominator = e.T @ e

    I = (n / np.sum(W)) * (numerator / denominator) if denominator > 0 else 0

    # Expected value and variance under null (no spatial correlation)
    E_I = -1 / (n - 1)

    # Simplified variance calculation (assumes normality)
    S0 = np.sum(W)
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2)

    var_I = (n * S1 - n * S0**2 + 3 * S0**2) / ((n - 1) * (n + 1) * S0**2)

    # Z-score
    z_score = (I - E_I) / np.sqrt(var_I)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

    conclusion = (
        "Reject H0: Spatial correlation detected"
        if p_value < 0.05
        else "Fail to reject H0: No spatial correlation detected"
    )

    return TestResult(
        statistic=I,
        p_value=p_value,
        conclusion=conclusion,
        details={
            "test": "Moran's I",
            "expected_value": E_I,
            "z_score": z_score,
            "interpretation": "I > 0: positive spatial correlation, I < 0: negative",
            "n_obs": n,
        },
    )


def cluster_diagnostics(data: pd.DataFrame, cluster_var: str) -> Dict:
    """
    Compute diagnostics for cluster structure.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    cluster_var : str
        Name of clustering variable

    Returns
    -------
    diagnostics : dict
        Dictionary with cluster diagnostics:
        - n_clusters: Number of clusters
        - cluster_sizes: Array of cluster sizes
        - min_size: Minimum cluster size
        - max_size: Maximum cluster size
        - mean_size: Mean cluster size
        - median_size: Median cluster size
        - unbalanced: Whether clusters are unbalanced
    """
    cluster_sizes = data.groupby(cluster_var).size().values

    diagnostics = {
        "n_clusters": len(cluster_sizes),
        "cluster_sizes": cluster_sizes,
        "min_size": cluster_sizes.min(),
        "max_size": cluster_sizes.max(),
        "mean_size": cluster_sizes.mean(),
        "median_size": np.median(cluster_sizes),
        "std_size": cluster_sizes.std(),
        "unbalanced": (
            cluster_sizes.std() / cluster_sizes.mean() > 0.1 if cluster_sizes.mean() > 0 else False
        ),
    }

    return diagnostics


def check_pcse_conditions(data: pd.DataFrame, entity_var: str, time_var: str) -> Dict:
    """
    Check conditions for Panel-Corrected Standard Errors (PCSE).

    PCSE requires T e N (time periods >= entities) for proper estimation.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data
    entity_var : str
        Entity identifier variable
    time_var : str
        Time identifier variable

    Returns
    -------
    conditions : dict
        Dictionary with:
        - N: Number of entities
        - T: Number of time periods
        - T_per_entity: Time periods per entity (if balanced)
        - balanced: Whether panel is balanced
        - pcse_appropriate: Whether PCSE conditions are met (T >= N)
        - recommendation: Text recommendation
    """
    N = data[entity_var].nunique()
    T = data[time_var].nunique()

    # Check if panel is balanced
    entity_counts = data.groupby(entity_var).size()
    balanced = entity_counts.nunique() == 1
    T_per_entity = entity_counts.mode()[0] if len(entity_counts) > 0 else 0

    # PCSE condition
    pcse_appropriate = T >= N

    if pcse_appropriate:
        recommendation = "PCSE is appropriate (T >= N)"
    else:
        recommendation = f"PCSE may be problematic (T={T} < N={N}). Consider clustered SEs instead."

    conditions = {
        "N": N,
        "T": T,
        "T_per_entity": T_per_entity if balanced else "varies",
        "balanced": balanced,
        "pcse_appropriate": pcse_appropriate,
        "recommendation": recommendation,
    }

    return conditions


def compute_vif(X: np.ndarray, variable_names: Optional[list] = None) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors (VIF) to detect multicollinearity.

    VIF > 10 indicates problematic multicollinearity.

    Parameters
    ----------
    X : np.ndarray
        Regressor matrix (n x k), without constant
    variable_names : list, optional
        Names of variables

    Returns
    -------
    vif_df : pd.DataFrame
        DataFrame with VIF values for each variable
    """
    n, k = X.shape

    if variable_names is None:
        variable_names = [f"X{i+1}" for i in range(k)]

    vif_values = []

    for i in range(k):
        # Regress X_i on all other X variables
        X_i = X[:, i]
        X_other = np.delete(X, i, axis=1)

        # Add constant
        X_other_const = np.column_stack([np.ones(n), X_other])

        try:
            beta = np.linalg.lstsq(X_other_const, X_i, rcond=None)[0]
            fitted = X_other_const @ beta
            ss_res = np.sum((X_i - fitted) ** 2)
            ss_tot = np.sum((X_i - np.mean(X_i)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            vif = 1 / (1 - r_squared) if r_squared < 0.9999 else np.inf
        except np.linalg.LinAlgError:
            vif = np.inf

        vif_values.append(vif)

    vif_df = pd.DataFrame({"Variable": variable_names, "VIF": vif_values})

    vif_df["Interpretation"] = vif_df["VIF"].apply(
        lambda x: "Severe" if x > 10 else ("Moderate" if x > 5 else "Low")
    )

    return vif_df


# Export all functions
__all__ = [
    "TestResult",
    "test_heteroskedasticity",
    "test_autocorrelation",
    "test_spatial_correlation",
    "cluster_diagnostics",
    "check_pcse_conditions",
    "compute_vif",
]
