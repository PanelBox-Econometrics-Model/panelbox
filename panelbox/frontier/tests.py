"""
Statistical tests for stochastic frontier models.

This module provides hypothesis tests for comparing different SFA model
specifications, including the Hausman-type test for choosing between
True Fixed Effects and True Random Effects models.

References:
    Hausman, J. A. (1978).
        Specification tests in econometrics. Econometrica, 1251-1271.

    Greene, W. H. (2005).
        Fixed and random effects in stochastic frontier models.
        Journal of Productivity Analysis, 23(1), 7-32.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.linalg import inv, pinv


def hausman_test_tfe_tre(
    params_tfe: np.ndarray,
    params_tre: np.ndarray,
    vcov_tfe: np.ndarray,
    vcov_tre: np.ndarray,
    param_names: Optional[list] = None,
) -> Dict[str, any]:
    """Hausman-type test for choosing between TFE and TRE models.

    Tests the null hypothesis that TRE is consistent and efficient
    against the alternative that only TFE is consistent.

    H0: TRE is consistent and efficient (w_i ⊥ X)
    H1: Only TFE is consistent (w_i correlated with X)

    The test statistic is:
        H = (β̂_TFE - β̂_TRE)' [V(β̂_TFE) - V(β̂_TRE)]⁻¹ (β̂_TFE - β̂_TRE)

    Under H0, H ~ χ²(K) where K is the number of frontier parameters.

    Parameters:
        params_tfe: Parameter estimates from TFE model
        params_tre: Parameter estimates from TRE model
        vcov_tfe: Variance-covariance matrix from TFE
        vcov_tre: Variance-covariance matrix from TRE
        param_names: Names of parameters (optional)

    Returns:
        Dict with test results:
            - statistic: Test statistic value
            - df: Degrees of freedom
            - pvalue: P-value
            - conclusion: 'TFE' or 'TRE' based on p-value
            - difference: Parameter differences
            - std_difference: Standardized differences

    Notes:
        - Only frontier parameters (β) are compared
        - Variance parameters are excluded from test
        - If variance difference matrix is not positive definite,
          the test uses pseudo-inverse

    References:
        Hausman, J. A. (1978). Econometrica, 1251-1271.
        Greene, W. H. (2005). Journal of Productivity Analysis, 23(1), 7-32.
    """
    # Determine number of frontier parameters to compare
    # Exclude variance and heterogeneity parameters
    # Assume first k parameters are β (frontier)

    # Find common parameters (frontier coefficients)
    # TFE: [β, σ²_v, σ²_u]
    # TRE: [β, σ²_v, σ²_u, σ²_w]

    # Compare only β parameters
    n_beta = len(params_tfe) - 2  # Exclude σ²_v, σ²_u
    if len(params_tre) >= n_beta + 3:  # β, σ²_v, σ²_u, σ²_w
        n_beta_tre = len(params_tre) - 3
    else:
        n_beta_tre = len(params_tre) - 2

    n_compare = min(n_beta, n_beta_tre)

    # Extract frontier parameters
    beta_tfe = params_tfe[:n_compare]
    beta_tre = params_tre[:n_compare]

    # Extract variance-covariance matrices for frontier parameters
    V_tfe = vcov_tfe[:n_compare, :n_compare]
    V_tre = vcov_tre[:n_compare, :n_compare]

    # Difference in parameters
    diff = beta_tfe - beta_tre

    # Variance of difference: V(β̂_FE - β̂_RE) = V(β̂_FE) - V(β̂_RE)
    # Under H0, TRE is efficient, so V(TRE) ≤ V(TFE)
    V_diff = V_tfe - V_tre

    # Check if V_diff is positive definite
    try:
        # Try Cholesky decomposition (only works for PD matrices)
        np.linalg.cholesky(V_diff)
        is_positive_definite = True
    except np.linalg.LinAlgError:
        is_positive_definite = False
        warnings.warn(
            "Variance difference matrix is not positive definite. "
            "Using pseudo-inverse for Hausman test. "
            "Results may not be reliable.",
            UserWarning,
        )

    # Compute test statistic
    if is_positive_definite:
        # Use regular inverse
        try:
            V_diff_inv = inv(V_diff)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            V_diff_inv = pinv(V_diff)
            warnings.warn(
                "Variance difference matrix is singular. Using pseudo-inverse.", UserWarning
            )
    else:
        # Use pseudo-inverse
        V_diff_inv = pinv(V_diff)

    # Hausman statistic: H = diff' * V_diff_inv * diff
    hausman_stat = diff.T @ V_diff_inv @ diff

    # Degrees of freedom
    df = n_compare

    # P-value from chi-squared distribution
    pvalue = 1 - stats.chi2.cdf(hausman_stat, df)

    # Decision rule (typical significance level: 5%)
    if pvalue < 0.05:
        conclusion = "TFE"
        interpretation = (
            "Reject H0. Evidence suggests w_i is correlated with X. "
            "True Fixed Effects (TFE) model is preferred."
        )
    else:
        conclusion = "TRE"
        interpretation = (
            "Do not reject H0. No evidence of correlation between w_i and X. "
            "True Random Effects (TRE) model is preferred (more efficient)."
        )

    # Compute standardized differences for diagnostics
    # std_diff = diff / sqrt(diag(V_diff))
    std_diff = diff / np.sqrt(np.diag(V_diff) + 1e-10)

    # Prepare parameter comparison table
    param_comparison = []
    for i in range(n_compare):
        param_dict = {
            "parameter": param_names[i] if param_names else f"beta_{i}",
            "tfe_estimate": beta_tfe[i],
            "tre_estimate": beta_tre[i],
            "difference": diff[i],
            "std_difference": std_diff[i],
        }
        param_comparison.append(param_dict)

    return {
        "statistic": hausman_stat,
        "df": df,
        "pvalue": pvalue,
        "conclusion": conclusion,
        "interpretation": interpretation,
        "difference": diff,
        "std_difference": std_diff,
        "param_comparison": param_comparison,
        "is_positive_definite": is_positive_definite,
    }


def lr_test(loglik_restricted: float, loglik_unrestricted: float, df_diff: int) -> Dict[str, float]:
    """Likelihood ratio test for nested models.

    Tests whether additional parameters in the unrestricted model
    significantly improve the fit.

    H0: Restricted model is adequate
    H1: Unrestricted model is better

    Test statistic: LR = 2 * (LL_unrestricted - LL_restricted) ~ χ²(df_diff)

    Parameters:
        loglik_restricted: Log-likelihood of restricted model
        loglik_unrestricted: Log-likelihood of unrestricted model
        df_diff: Difference in degrees of freedom (number of restrictions)

    Returns:
        Dict with test results:
            - statistic: LR test statistic
            - df: Degrees of freedom
            - pvalue: P-value
            - conclusion: 'Reject H0' or 'Do not reject H0'

    Example:
        Test if σ²_w = 0 (TRE reduces to pooled SFA):
        >>> lr_result = lr_test(loglik_pooled, loglik_tre, df_diff=1)
    """
    # LR statistic
    lr_stat = 2 * (loglik_unrestricted - loglik_restricted)

    # Ensure non-negative (numerical precision issues)
    if lr_stat < 0:
        warnings.warn(
            f"LR statistic is negative ({lr_stat:.4f}). "
            "This usually indicates numerical issues. Setting to 0.",
            UserWarning,
        )
        lr_stat = 0

    # P-value
    pvalue = 1 - stats.chi2.cdf(lr_stat, df_diff)

    # Decision
    if pvalue < 0.05:
        conclusion = "Reject H0"
        interpretation = (
            "Unrestricted model provides significantly better fit. "
            "Additional parameters are justified."
        )
    else:
        conclusion = "Do not reject H0"
        interpretation = (
            "Restricted model is adequate. " "Additional parameters not statistically justified."
        )

    return {
        "statistic": lr_stat,
        "df": df_diff,
        "pvalue": pvalue,
        "conclusion": conclusion,
        "interpretation": interpretation,
    }


def wald_test(
    params: np.ndarray, vcov: np.ndarray, R: np.ndarray, r: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Wald test for linear restrictions on parameters.

    Tests H0: R*θ = r against H1: R*θ ≠ r

    Test statistic: W = (R*θ̂ - r)' [R*V*R']⁻¹ (R*θ̂ - r) ~ χ²(q)

    where q is the number of restrictions (rows of R).

    Parameters:
        params: Parameter estimates (k,)
        vcov: Variance-covariance matrix (k, k)
        R: Restriction matrix (q, k)
        r: Restriction vector (q,). If None, defaults to zeros.

    Returns:
        Dict with test results:
            - statistic: Wald test statistic
            - df: Degrees of freedom
            - pvalue: P-value
            - conclusion: 'Reject H0' or 'Do not reject H0'

    Example:
        Test if β_1 = β_2 = 0:
        >>> R = np.array([[0, 1, 0, 0], [0, 0, 1, 0]])  # Select β_1 and β_2
        >>> wald_result = wald_test(params, vcov, R)
    """
    if r is None:
        r = np.zeros(R.shape[0])

    # Compute restriction
    restriction = R @ params - r

    # Variance of restriction
    V_restriction = R @ vcov @ R.T

    # Inverse
    try:
        V_restriction_inv = inv(V_restriction)
    except np.linalg.LinAlgError:
        V_restriction_inv = pinv(V_restriction)
        warnings.warn("Restriction variance matrix is singular. Using pseudo-inverse.", UserWarning)

    # Wald statistic
    wald_stat = restriction.T @ V_restriction_inv @ restriction

    # Degrees of freedom
    df = R.shape[0]

    # P-value
    pvalue = 1 - stats.chi2.cdf(wald_stat, df)

    # Decision
    if pvalue < 0.05:
        conclusion = "Reject H0"
        interpretation = "Restrictions are not supported by the data."
    else:
        conclusion = "Do not reject H0"
        interpretation = "Restrictions are consistent with the data."

    return {
        "statistic": wald_stat,
        "df": df,
        "pvalue": pvalue,
        "conclusion": conclusion,
        "interpretation": interpretation,
    }


def heterogeneity_significance_test(sigma_w_sq: float, se_sigma_w_sq: float) -> Dict[str, float]:
    """Test if heterogeneity variance is significantly different from zero.

    For TRE model, tests H0: σ²_w = 0 vs H1: σ²_w > 0

    If σ²_w = 0, TRE reduces to pooled SFA model.

    Parameters:
        sigma_w_sq: Estimated variance of heterogeneity
        se_sigma_w_sq: Standard error of σ²_w estimate

    Returns:
        Dict with test results:
            - statistic: z-statistic
            - pvalue: One-sided p-value
            - conclusion: 'Significant' or 'Not significant'

    Note:
        This is a one-sided test (σ²_w > 0) since variance must be non-negative.
    """
    # Z-statistic
    z_stat = sigma_w_sq / se_sigma_w_sq

    # One-sided p-value (upper tail)
    pvalue = 1 - stats.norm.cdf(z_stat)

    # Decision
    if pvalue < 0.05:
        conclusion = "Significant"
        interpretation = (
            "Heterogeneity variance is significantly greater than zero. "
            "TRE model is preferred over pooled SFA."
        )
    else:
        conclusion = "Not significant"
        interpretation = (
            "Heterogeneity variance is not significantly different from zero. "
            "Pooled SFA may be adequate."
        )

    return {
        "statistic": z_stat,
        "pvalue": pvalue,
        "conclusion": conclusion,
        "interpretation": interpretation,
    }


def summary_model_comparison(
    results_dict: Dict[str, "SFResult"], test_type: str = "hausman"
) -> str:
    """Generate a summary table comparing multiple SFA models.

    Parameters:
        results_dict: Dictionary of model results {model_name: SFResult}
        test_type: Type of comparison ('hausman', 'lr', 'aic', 'bic')

    Returns:
        Formatted string with comparison table
    """
    # Extract information from each model
    comparison_data = []

    for model_name, result in results_dict.items():
        comparison_data.append(
            {
                "model": model_name,
                "loglik": result.loglik,
                "aic": result.aic if hasattr(result, "aic") else None,
                "bic": result.bic if hasattr(result, "bic") else None,
                "n_params": len(result.params),
            }
        )

    # Format table
    header = "Model Comparison"
    separator = "=" * 60
    lines = [separator, header, separator]

    # Add rows
    for data in comparison_data:
        line = (
            f"{data['model']:15s} | LL: {data['loglik']:10.2f} | "
            f"AIC: {data['aic']:10.2f} | BIC: {data['bic']:10.2f}"
        )
        lines.append(line)

    lines.append(separator)

    # Add test results if available
    if test_type == "hausman" and "TFE" in results_dict and "TRE" in results_dict:
        # Perform Hausman test
        tfe_result = results_dict["TFE"]
        tre_result = results_dict["TRE"]

        hausman_result = hausman_test_tfe_tre(
            tfe_result.params, tre_result.params, tfe_result.vcov, tre_result.vcov
        )

        lines.append("\nHausman Test (TFE vs TRE):")
        lines.append(f"  Statistic: {hausman_result['statistic']:.4f}")
        lines.append(f"  P-value: {hausman_result['pvalue']:.4f}")
        lines.append(f"  Conclusion: {hausman_result['conclusion']}")

    return "\n".join(lines)
