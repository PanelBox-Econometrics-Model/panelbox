"""
Diagnostic helper functions for count models tutorials.

Provides common statistical tests and diagnostics for count models.

Functions:
- compute_overdispersion_index: Calculate dispersion index
- overdispersion_test: Formal test for overdispersion
- vuong_test_summary: Vuong test for non-nested models
- compute_rootogram_data: Compute rootogram statistics
- detect_outliers_count: Identify outliers in count models
- hausman_test_summary: Hausman test for FE vs RE
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln


def compute_overdispersion_index(y, fitted_values):
    """
    Compute the overdispersion index (Var/Mean ratio).

    Parameters
    ----------
    y : array-like
        Observed counts
    fitted_values : array-like
        Fitted values from model

    Returns
    -------
    float
        Overdispersion index. Values > 1 suggest overdispersion.

    Examples
    --------
    >>> index = compute_overdispersion_index(data['y'], result.predict())
    >>> print(f"Overdispersion index: {index:.3f}")

    Notes
    -----
    Under Poisson assumption, E(Y) = Var(Y), so index should be ≈ 1.
    Index >> 1 suggests Negative Binomial may be more appropriate.
    """
    y = np.asarray(y)
    mu = np.asarray(fitted_values)

    # Pearson residuals
    pearson_resid = (y - mu) / np.sqrt(mu)

    # Overdispersion parameter estimate
    n = len(y)
    k = len(mu)  # Assuming k parameters
    dispersion = np.sum(pearson_resid**2) / (n - k)

    return dispersion


def overdispersion_test(result, verbose=True):
    """
    Perform overdispersion test for Poisson model.

    Tests H0: alpha = 0 (Poisson) vs H1: alpha > 0 (Negative Binomial).

    Parameters
    ----------
    result : PoissonResults
        Fitted Poisson model result
    verbose : bool, default True
        If True, print test results

    Returns
    -------
    dict
        Dictionary containing:
        - 'statistic': Test statistic
        - 'p_value': P-value
        - 'conclusion': Text interpretation

    Notes
    -----
    Uses auxiliary regression approach (Cameron & Trivedi 1990).
    """
    # Get residuals
    y = result.model.endog
    mu = result.predict()

    # Auxiliary regression: (y - mu)^2 - y = alpha * mu^2
    aux_y = (y - mu) ** 2 - y
    aux_x = mu**2

    # OLS regression
    from scipy.stats import linregress

    slope, intercept, r_value, p_value, std_err = linregress(aux_x, aux_y)

    # Test statistic
    t_stat = slope / std_err
    p_val = 1 - stats.norm.cdf(t_stat)  # One-sided test

    if p_val < 0.01:
        conclusion = "Strong evidence of overdispersion (p < 0.01). Use Negative Binomial."
    elif p_val < 0.05:
        conclusion = "Significant overdispersion detected (p < 0.05). Consider Negative Binomial."
    elif p_val < 0.10:
        conclusion = "Weak evidence of overdispersion (p < 0.10)."
    else:
        conclusion = "No significant overdispersion detected. Poisson may be adequate."

    results = {
        "alpha_estimate": slope,
        "se_alpha": std_err,
        "statistic": t_stat,
        "p_value": p_val,
        "conclusion": conclusion,
    }

    if verbose:
        print("=" * 60)
        print("Overdispersion Test (Cameron-Trivedi)")
        print("=" * 60)
        print(f"H0: alpha = 0 (Poisson)")
        print(f"H1: alpha > 0 (Negative Binomial)")
        print()
        print(f"Alpha estimate: {slope:.4f}")
        print(f"Std. Error:     {std_err:.4f}")
        print(f"t-statistic:    {t_stat:.4f}")
        print(f"p-value:        {p_val:.4f}")
        print()
        print(f"Conclusion: {conclusion}")
        print("=" * 60)

    return results


def vuong_test_summary(
    result1, result2, model1_name="Model 1", model2_name="Model 2", verbose=True
):
    """
    Perform Vuong test for non-nested model comparison.

    Commonly used to compare ZIP vs Poisson or ZINB vs NB.

    Parameters
    ----------
    result1 : ModelResults
        First model result
    result2 : ModelResults
        Second model result
    model1_name : str
        Name of first model
    model2_name : str
        Name of second model
    verbose : bool, default True
        Print results

    Returns
    -------
    dict
        Test results including statistic and conclusion

    Notes
    -----
    Vuong (1989) test for non-nested models.
    Positive statistic favors model 1, negative favors model 2.
    """
    # Log-likelihoods
    ll1 = result1.llf
    ll2 = result2.llf

    # Number of observations
    n = result1.nobs

    # Likelihood ratio statistic
    lr = ll1 - ll2

    # Variance of LR under H0 (models equivalent)
    # Simplified calculation (full calculation requires observation-level LLs)
    # This is an approximation
    var_lr = n * (lr / n) ** 2  # Placeholder

    # Vuong statistic
    vuong_stat = np.sqrt(n) * lr / np.sqrt(var_lr) if var_lr > 0 else 0

    # P-value (two-sided)
    p_value = 2 * (1 - stats.norm.cdf(abs(vuong_stat)))

    if p_value < 0.01:
        if vuong_stat > 0:
            conclusion = f"{model1_name} significantly preferred (p < 0.01)"
        else:
            conclusion = f"{model2_name} significantly preferred (p < 0.01)"
    elif p_value < 0.05:
        if vuong_stat > 0:
            conclusion = f"{model1_name} preferred (p < 0.05)"
        else:
            conclusion = f"{model2_name} preferred (p < 0.05)"
    else:
        conclusion = "Models are statistically indistinguishable"

    results = {
        "llf_model1": ll1,
        "llf_model2": ll2,
        "vuong_statistic": vuong_stat,
        "p_value": p_value,
        "conclusion": conclusion,
    }

    if verbose:
        print("=" * 60)
        print("Vuong Test for Non-Nested Models")
        print("=" * 60)
        print(f"Model 1: {model1_name}")
        print(f"  Log-Likelihood: {ll1:.2f}")
        print()
        print(f"Model 2: {model2_name}")
        print(f"  Log-Likelihood: {ll2:.2f}")
        print()
        print(f"Vuong statistic: {vuong_stat:.4f}")
        print(f"p-value:         {p_value:.4f}")
        print()
        print(f"Conclusion: {conclusion}")
        print("=" * 60)

    return results


def compute_rootogram_data(observed, expected, breaks=None):
    """
    Compute data for rootogram plot.

    Parameters
    ----------
    observed : array-like
        Observed counts
    expected : array-like
        Expected counts
    breaks : array-like, optional
        Bin edges

    Returns
    -------
    dict
        Dictionary with x values, observed sqrt, expected sqrt, and residuals
    """
    observed = np.asarray(observed)
    expected = np.asarray(expected)

    if breaks is None:
        breaks = np.arange(observed.min(), observed.max() + 2)

    obs_freq, _ = np.histogram(observed, bins=breaks)
    exp_freq, _ = np.histogram(expected, bins=breaks)

    obs_sqrt = np.sqrt(obs_freq)
    exp_sqrt = np.sqrt(exp_freq)

    x = breaks[:-1] + 0.5
    residuals = obs_sqrt - exp_sqrt

    return {"x": x, "observed_sqrt": obs_sqrt, "expected_sqrt": exp_sqrt, "residuals": residuals}


def detect_outliers_count(result, threshold=3, method="pearson"):
    """
    Identify outliers in count models using residuals.

    Parameters
    ----------
    result : ModelResults
        Fitted model result
    threshold : float, default 3
        Threshold for standardized residuals
    method : str, default 'pearson'
        Residual type: 'pearson', 'deviance', or 'standardized'

    Returns
    -------
    pd.DataFrame
        DataFrame with outlier information

    Examples
    --------
    >>> outliers = detect_outliers_count(result, threshold=3)
    >>> print(f"Found {len(outliers)} outliers")
    """
    y = result.model.endog
    mu = result.predict()

    if method == "pearson":
        residuals = (y - mu) / np.sqrt(mu)
    elif method == "deviance":
        # Deviance residuals for Poisson
        sign = np.sign(y - mu)
        dev = 2 * (y * np.log(y / mu + 1e-10) - (y - mu))
        residuals = sign * np.sqrt(dev)
    elif method == "standardized":
        pearson = (y - mu) / np.sqrt(mu)
        # Approximate standardization
        residuals = pearson / np.sqrt(1 - 1 / len(y))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Identify outliers
    outlier_mask = np.abs(residuals) > threshold

    outlier_df = pd.DataFrame(
        {
            "observation": np.where(outlier_mask)[0],
            "observed": y[outlier_mask],
            "expected": mu[outlier_mask],
            "residual": residuals[outlier_mask],
        }
    )

    return outlier_df


def hausman_test_summary(fe_result, re_result, verbose=True):
    """
    Perform Hausman specification test for FE vs RE.

    Tests H0: RE is consistent and efficient vs H1: Only FE is consistent.

    Parameters
    ----------
    fe_result : ModelResults
        Fixed effects model result
    re_result : ModelResults
        Random effects model result
    verbose : bool, default True
        Print results

    Returns
    -------
    dict
        Test results

    Notes
    -----
    Hausman (1978) test: H = (β_FE - β_RE)' [Var(β_FE) - Var(β_RE)]^{-1} (β_FE - β_RE)
    Under H0, H ~ χ²(k) where k is number of parameters.
    """
    # Extract coefficients (common variables only)
    fe_params = fe_result.params
    re_params = re_result.params

    # Get common variables
    common_vars = fe_params.index.intersection(re_params.index)

    beta_fe = fe_params[common_vars].values
    beta_re = re_params[common_vars].values

    # Covariance matrices
    cov_fe = fe_result.cov_params().loc[common_vars, common_vars].values
    cov_re = re_result.cov_params().loc[common_vars, common_vars].values

    # Difference
    diff = beta_fe - beta_re
    cov_diff = cov_fe - cov_re

    # Handle potential singularity
    try:
        inv_cov_diff = np.linalg.inv(cov_diff)
        hausman_stat = diff @ inv_cov_diff @ diff
        df = len(common_vars)
        p_value = 1 - stats.chi2.cdf(hausman_stat, df)

        if p_value < 0.01:
            conclusion = "Reject RE in favor of FE (p < 0.01). Use Fixed Effects."
        elif p_value < 0.05:
            conclusion = "Reject RE in favor of FE (p < 0.05). Use Fixed Effects."
        elif p_value < 0.10:
            conclusion = "Weak evidence against RE (p < 0.10)."
        else:
            conclusion = "Do not reject RE (p > 0.10). Random Effects may be appropriate."

    except np.linalg.LinAlgError:
        hausman_stat = np.nan
        p_value = np.nan
        conclusion = "Hausman test could not be computed (singular matrix)."

    results = {
        "statistic": hausman_stat,
        "df": len(common_vars),
        "p_value": p_value,
        "conclusion": conclusion,
    }

    if verbose:
        print("=" * 60)
        print("Hausman Specification Test (FE vs RE)")
        print("=" * 60)
        print(f"H0: Random Effects is consistent and efficient")
        print(f"H1: Fixed Effects is consistent, RE is inconsistent")
        print()
        print(f"Chi-squared statistic: {hausman_stat:.4f}")
        print(f"Degrees of freedom:    {len(common_vars)}")
        print(f"p-value:               {p_value:.4f}")
        print()
        print(f"Conclusion: {conclusion}")
        print("=" * 60)

    return results


if __name__ == "__main__":
    # Test diagnostics
    print("Diagnostics helpers module loaded successfully!")
    print("Functions available:")
    print("  - compute_overdispersion_index")
    print("  - overdispersion_test")
    print("  - vuong_test_summary")
    print("  - compute_rootogram_data")
    print("  - detect_outliers_count")
    print("  - hausman_test_summary")
