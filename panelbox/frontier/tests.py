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


def inefficiency_presence_test(
    loglik_sfa: float,
    loglik_ols: float,
    residuals_ols: np.ndarray,
    frontier_type: str = "production",
    distribution: str = "half_normal",
) -> Dict[str, any]:
    """Test for the presence of inefficiency (σ²_u > 0).

    Tests H0: σ²_u = 0 (OLS is sufficient) vs H1: σ²_u > 0 (SFA is needed)

    Since σ²_u is constrained to be non-negative (boundary parameter),
    the standard LR test distribution doesn't apply. Under H0, the LR
    statistic follows a mixed chi-square distribution (Kodde & Palm 1986):

    For half-normal or exponential:
        LR ~ ½χ²(0) + ½χ²(1)

    For truncated-normal (with μ parameter):
        LR ~ ½χ²(1) + ½χ²(2)

    Also performs a skewness test as diagnostic:
    - Production frontier: OLS residuals should have negative skewness
    - Cost frontier: OLS residuals should have positive skewness

    Parameters:
        loglik_sfa: Log-likelihood from SFA model
        loglik_ols: Log-likelihood from OLS model (H0)
        residuals_ols: OLS residuals for skewness test
        frontier_type: 'production' or 'cost'
        distribution: 'half_normal', 'exponential', or 'truncated_normal'

    Returns:
        Dict with test results:
            - lr_statistic: LR test statistic
            - df: Degrees of freedom
            - pvalue: P-value (corrected for mixed chi-square)
            - critical_values: Critical values at 10%, 5%, 1% levels
            - conclusion: 'SFA needed' or 'OLS sufficient'
            - skewness: Skewness of OLS residuals
            - skewness_warning: Warning if skewness has wrong sign

    References:
        Kodde, D. A., & Palm, F. C. (1986). Wald criteria for jointly testing
            equality and inequality restrictions. Econometrica, 1243-1248.
        Coelli, T. J. (1995). Estimators and hypothesis tests for a
            stochastic frontier function: A Monte Carlo analysis.
            Journal of Productivity Analysis, 6(4), 247-268.
    """
    # Compute LR statistic
    lr_stat = 2 * (loglik_sfa - loglik_ols)

    # Ensure non-negative (numerical precision)
    if lr_stat < -1e-6:
        warnings.warn(
            f"LR statistic is negative ({lr_stat:.6f}). "
            "SFA model may have failed to converge properly.",
            UserWarning,
        )
        lr_stat = 0

    # Determine distribution and compute p-value
    if distribution in ["half_normal", "exponential"]:
        # Mixed chi-square: ½χ²(0) + ½χ²(1)
        # P-value: P(½χ²(0) + ½χ²(1) > LR) = 0.5 * P(χ²(1) > LR)
        df = 1
        if lr_stat > 0:
            pvalue = 0.5 * (1 - stats.chi2.cdf(lr_stat, df=1))
        else:
            pvalue = 0.5  # LR = 0 implies p-value = 0.5

        # Kodde-Palm critical values for mixed ½χ²(0) + ½χ²(1)
        # From Table 1 of Kodde & Palm (1986)
        critical_values = {
            "10%": 2.706,
            "5%": 3.841,
            "1%": 6.635,
        }

    elif distribution == "truncated_normal":
        # Mixed chi-square: ½χ²(1) + ½χ²(2)
        # P-value: 0.5*P(χ²(1) > LR) + 0.5*P(χ²(2) > LR)
        df = 2  # For reporting purposes
        if lr_stat > 0:
            p1 = 1 - stats.chi2.cdf(lr_stat, df=1)
            p2 = 1 - stats.chi2.cdf(lr_stat, df=2)
            pvalue = 0.5 * p1 + 0.5 * p2
        else:
            pvalue = 0.5

        # Kodde-Palm critical values for mixed ½χ²(1) + ½χ²(2)
        critical_values = {
            "10%": 4.605,
            "5%": 5.991,
            "1%": 9.210,
        }

    else:
        warnings.warn(
            f"Unknown distribution '{distribution}'. "
            "Using standard chi-square test with df=1.",
            UserWarning,
        )
        df = 1
        pvalue = 1 - stats.chi2.cdf(lr_stat, df=1)
        critical_values = {
            "10%": stats.chi2.ppf(0.90, df=1),
            "5%": stats.chi2.ppf(0.95, df=1),
            "1%": stats.chi2.ppf(0.99, df=1),
        }

    # Decision
    if pvalue < 0.05:
        conclusion = "SFA needed"
        interpretation = (
            f"Reject H0 at 5% level (p-value = {pvalue:.4f}). "
            "Inefficiency term is statistically significant. "
            "SFA model is preferred over OLS."
        )
    else:
        conclusion = "OLS sufficient"
        interpretation = (
            f"Do not reject H0 (p-value = {pvalue:.4f}). "
            "No evidence of inefficiency. "
            "OLS regression may be adequate."
        )

    # Skewness test
    skewness = stats.skew(residuals_ols)
    skewness_warning = None

    if frontier_type.lower() == "production":
        # Production: inefficiency reduces output, residuals should be negatively skewed
        if skewness > 0:
            skewness_warning = (
                "WARNING: OLS residuals have positive skewness, "
                "but production frontier should have negative skewness. "
                "This suggests potential misspecification (wrong frontier type, "
                "outliers, or distributional assumption)."
            )
    elif frontier_type.lower() == "cost":
        # Cost: inefficiency increases cost, residuals should be positively skewed
        if skewness < 0:
            skewness_warning = (
                "WARNING: OLS residuals have negative skewness, "
                "but cost frontier should have positive skewness. "
                "This suggests potential misspecification (wrong frontier type, "
                "outliers, or distributional assumption)."
            )

    return {
        "lr_statistic": lr_stat,
        "df": df,
        "pvalue": pvalue,
        "critical_values": critical_values,
        "conclusion": conclusion,
        "interpretation": interpretation,
        "skewness": skewness,
        "skewness_warning": skewness_warning,
        "distribution": distribution,
        "test_type": "mixed_chi_square" if distribution in ["half_normal", "exponential", "truncated_normal"] else "standard_chi_square",
    }


def skewness_test(
    residuals: np.ndarray,
    frontier_type: str = "production"
) -> Dict[str, any]:
    """Preliminary skewness test for frontier model specification.

    This is a diagnostic test, not a formal hypothesis test. It checks whether
    OLS residuals have the expected sign of skewness for the specified frontier type.

    For production frontier: u reduces y, so ε = y - Xβ should be negatively skewed
    For cost frontier: u increases y, so ε = y - Xβ should be positively skewed

    Parameters:
        residuals: OLS residuals (y - Xβ)
        frontier_type: 'production' or 'cost'

    Returns:
        Dict with diagnostic results:
            - skewness: Sample skewness
            - expected_sign: Expected sign for frontier type
            - correct_sign: Boolean indicating if sign matches expectation
            - warning: Warning message if sign is incorrect

    References:
        Coelli, T. J. (1995). Journal of Productivity Analysis, 6(4), 247-268.
    """
    skew_value = stats.skew(residuals)

    if frontier_type.lower() == "production":
        expected_sign = "negative"
        correct_sign = skew_value < 0
        if not correct_sign:
            warning = (
                f"Skewness is {skew_value:.4f} (positive), but production "
                "frontier should have negative skewness. "
                "Consider: (1) wrong frontier type, (2) outliers, "
                "(3) incorrect distributional assumption."
            )
        else:
            warning = None

    elif frontier_type.lower() == "cost":
        expected_sign = "positive"
        correct_sign = skew_value > 0
        if not correct_sign:
            warning = (
                f"Skewness is {skew_value:.4f} (negative), but cost "
                "frontier should have positive skewness. "
                "Consider: (1) wrong frontier type, (2) outliers, "
                "(3) incorrect distributional assumption."
            )
        else:
            warning = None

    else:
        raise ValueError(f"Invalid frontier_type: {frontier_type}. Must be 'production' or 'cost'.")

    return {
        "skewness": skew_value,
        "expected_sign": expected_sign,
        "correct_sign": correct_sign,
        "warning": warning,
        "frontier_type": frontier_type,
    }


def vuong_test(
    loglik1: np.ndarray,
    loglik2: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
) -> Dict[str, any]:
    """Vuong (1989) test for non-nested model selection.

    Tests whether two non-nested models have significantly different
    predictive performance. Unlike LR test, Vuong test doesn't require
    models to be nested.

    H0: Both models are equally close to the true model
    H1: One model is closer to the true model

    Test statistic:
        V = (1/√N) × Σᵢ ln(L1_i/L2_i) / sd[ln(L1_i/L2_i)]

    Under H0: V ~ N(0, 1)

    Parameters:
        loglik1: Array of observation-level log-likelihoods for model 1
        loglik2: Array of observation-level log-likelihoods for model 2
        model1_name: Name of first model
        model2_name: Name of second model

    Returns:
        Dict with test results:
            - statistic: Vuong test statistic
            - pvalue_two_sided: Two-sided p-value
            - pvalue_model1: One-sided p-value for model 1 > model 2
            - pvalue_model2: One-sided p-value for model 2 > model 1
            - conclusion: Which model is preferred (or 'equivalent')
            - interpretation: Detailed interpretation

    References:
        Vuong, Q. H. (1989). Likelihood ratio tests for model selection and
            non-nested hypotheses. Econometrica, 307-333.

    Notes:
        - Requires observation-level log-likelihoods, not just total log-likelihood
        - Both models must be estimated on the same sample
        - Test assumes large N (asymptotic normality)
    """
    # Check inputs
    if len(loglik1) != len(loglik2):
        raise ValueError(
            f"Log-likelihood arrays must have same length. "
            f"Got {len(loglik1)} and {len(loglik2)}"
        )

    n = len(loglik1)

    if n < 30:
        warnings.warn(
            f"Vuong test requires large sample (N >> 30). N = {n} may be too small. "
            "Results may not be reliable.",
            UserWarning,
        )

    # Compute log-likelihood ratios at observation level
    llr_i = loglik1 - loglik2

    # Vuong statistic
    mean_llr = np.mean(llr_i)
    sd_llr = np.std(llr_i, ddof=1)

    if sd_llr < 1e-12:
        # Models are essentially identical
        vuong_stat = 0
    else:
        vuong_stat = np.sqrt(n) * mean_llr / sd_llr

    # P-values
    pvalue_two_sided = 2 * (1 - stats.norm.cdf(np.abs(vuong_stat)))
    pvalue_model1 = 1 - stats.norm.cdf(vuong_stat)  # P(V < v)
    pvalue_model2 = stats.norm.cdf(vuong_stat)      # P(V > v)

    # Decision (using 5% significance level)
    if pvalue_two_sided < 0.05:
        if vuong_stat > 0:
            conclusion = model1_name
            interpretation = (
                f"Reject H0 at 5% level (p = {pvalue_two_sided:.4f}). "
                f"{model1_name} is significantly preferred over {model2_name}."
            )
        else:
            conclusion = model2_name
            interpretation = (
                f"Reject H0 at 5% level (p = {pvalue_two_sided:.4f}). "
                f"{model2_name} is significantly preferred over {model1_name}."
            )
    else:
        conclusion = "equivalent"
        interpretation = (
            f"Do not reject H0 (p = {pvalue_two_sided:.4f}). "
            f"Models have equivalent fit. Use information criteria or other considerations."
        )

    return {
        "statistic": vuong_stat,
        "pvalue_two_sided": pvalue_two_sided,
        "pvalue_model1": pvalue_model1,
        "pvalue_model2": pvalue_model2,
        "conclusion": conclusion,
        "interpretation": interpretation,
        "n_obs": n,
        "mean_llr": mean_llr,
        "sd_llr": sd_llr,
    }


def compare_nested_distributions(
    loglik_restricted: float,
    loglik_unrestricted: float,
    dist_restricted: str,
    dist_unrestricted: str,
) -> Dict[str, any]:
    """Compare nested distributional specifications using LR test.

    For nested distributions (e.g., half-normal is nested in truncated-normal
    when μ = 0), performs likelihood ratio test.

    Common nested pairs:
        - half_normal ⊂ truncated_normal (H0: μ = 0)
        - exponential ⊂ gamma (H0: P = 1)

    Parameters:
        loglik_restricted: Log-likelihood of restricted model
        loglik_unrestricted: Log-likelihood of unrestricted model
        dist_restricted: Name of restricted distribution
        dist_unrestricted: Name of unrestricted distribution

    Returns:
        Dict with test results:
            - lr_statistic: LR test statistic
            - df: Degrees of freedom
            - pvalue: P-value
            - conclusion: Which distribution is preferred
            - interpretation: Detailed interpretation

    Notes:
        - Half-normal vs truncated-normal: df=1, tests μ=0, uses mixed chi-square
        - Other nested pairs: standard chi-square distribution
    """
    # Determine df and test type
    if (dist_restricted == "half_normal" and dist_unrestricted == "truncated_normal"):
        # H0: μ = 0 (boundary parameter)
        # Use mixed chi-square: ½χ²(0) + ½χ²(1)
        df = 1
        use_mixed = True
    elif (dist_restricted == "exponential" and dist_unrestricted == "gamma"):
        # H0: P = 1
        df = 1
        use_mixed = False
    else:
        warnings.warn(
            f"Nested relationship between '{dist_restricted}' and "
            f"'{dist_unrestricted}' may not be standard. "
            "Using df=1 and standard chi-square.",
            UserWarning,
        )
        df = 1
        use_mixed = False

    # Compute LR statistic
    lr_stat = 2 * (loglik_unrestricted - loglik_restricted)

    # Ensure non-negative
    if lr_stat < -1e-6:
        warnings.warn(
            f"LR statistic is negative ({lr_stat:.6f}). "
            "Unrestricted model should have higher log-likelihood.",
            UserWarning,
        )
        lr_stat = 0

    # Compute p-value
    if use_mixed and lr_stat > 0:
        # Mixed chi-square ½χ²(0) + ½χ²(1)
        pvalue = 0.5 * (1 - stats.chi2.cdf(lr_stat, df=1))
    elif lr_stat > 0:
        # Standard chi-square
        pvalue = 1 - stats.chi2.cdf(lr_stat, df=df)
    else:
        pvalue = 0.5 if use_mixed else 1.0

    # Decision
    if pvalue < 0.05:
        conclusion = dist_unrestricted
        interpretation = (
            f"Reject H0 at 5% level (p = {pvalue:.4f}). "
            f"{dist_unrestricted} provides significantly better fit than {dist_restricted}."
        )
    else:
        conclusion = dist_restricted
        interpretation = (
            f"Do not reject H0 (p = {pvalue:.4f}). "
            f"{dist_restricted} is adequate. Simpler model preferred."
        )

    return {
        "lr_statistic": lr_stat,
        "df": df,
        "pvalue": pvalue,
        "conclusion": conclusion,
        "interpretation": interpretation,
        "test_type": "mixed_chi_square" if use_mixed else "chi_square",
    }
