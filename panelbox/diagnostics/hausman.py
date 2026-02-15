"""
Hausman test for panel data models.

This module implements the Hausman specification test for comparing
Fixed Effects and Random Effects models.
"""

import warnings
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd


@dataclass
class HausmanTestResult:
    """
    Result container for Hausman test.

    Attributes
    ----------
    statistic : float
        Hausman test statistic
    pvalue : float
        P-value (chi-squared or bootstrap)
    df : int
        Degrees of freedom
    method : str
        Method used ('chi2' or 'bootstrap')
    interpretation : str
        Text interpretation of results
    common_vars : list
        Variables compared in test
    """

    statistic: float
    pvalue: float
    df: int
    method: str
    interpretation: str
    common_vars: list

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HausmanTestResult:\n"
            f"  Statistic: {self.statistic:.4f}\n"
            f"  P-value: {self.pvalue:.4f}\n"
            f"  DF: {self.df}\n"
            f"  Method: {self.method}\n"
            f"  {self.interpretation}"
        )

    def summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 60)
        print("Hausman Specification Test")
        print("=" * 60)
        print(f"H0: Random Effects estimator is consistent and efficient")
        print(f"H1: Random Effects estimator is inconsistent\n")

        print(f"Test statistic: {self.statistic:.4f}")
        print(f"P-value: {self.pvalue:.4f}")
        print(f"Degrees of freedom: {self.df}")
        print(f"Method: {self.method}")

        print("\n" + "-" * 40)
        print("Interpretation:")
        print(self.interpretation)

        if self.pvalue < 0.01:
            print("\nStrong evidence against Random Effects (p < 0.01)")
        elif self.pvalue < 0.05:
            print("\nEvidence against Random Effects (p < 0.05)")
        elif self.pvalue < 0.10:
            print("\nWeak evidence against Random Effects (p < 0.10)")
        else:
            print("\nNo evidence against Random Effects (p >= 0.10)")

        print("\nVariables compared: " + ", ".join(self.common_vars))
        print("=" * 60)


def hausman_test(fe_result, re_result, df_correction: bool = True) -> HausmanTestResult:
    """
    Hausman test for Fixed Effects vs Random Effects (linear models).

    Tests the null hypothesis that the Random Effects estimator is
    consistent (i.e., E[α_i | X_i] = 0).

    The test statistic is:
    H = (β_FE - β_RE)' [V(β_FE) - V(β_RE)]^{-1} (β_FE - β_RE)

    Under H0, H ~ χ²(k) where k is the number of common coefficients.

    Parameters
    ----------
    fe_result : PanelResults
        Results from Fixed Effects estimation
    re_result : PanelResults
        Results from Random Effects estimation
    df_correction : bool, default=True
        Whether to apply degrees of freedom correction

    Returns
    -------
    HausmanTestResult
        Test results with statistic, p-value, and interpretation

    Examples
    --------
    >>> # Fit Fixed Effects
    >>> fe_model = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> fe_results = fe_model.fit()
    >>>
    >>> # Fit Random Effects
    >>> re_model = pb.RandomEffects("y ~ x1 + x2", data, "entity", "time")
    >>> re_results = re_model.fit()
    >>>
    >>> # Hausman test
    >>> hausman = hausman_test(fe_results, re_results)
    >>> print(hausman.summary())

    Notes
    -----
    The test requires that:
    1. Both models are estimated on the same data
    2. The Fixed Effects estimator is consistent under both H0 and H1
    3. The Random Effects estimator is efficient under H0

    If the variance difference matrix is not positive definite,
    this suggests model misspecification.

    References
    ----------
    .. [1] Hausman, J. A. (1978). "Specification Tests in Econometrics."
           Econometrica, 46(6), 1251-1271.
    .. [2] Wooldridge, J. M. (2010). Econometric Analysis of Cross Section
           and Panel Data. MIT Press. Section 10.7.3.
    """
    # Get common variables (FE doesn't estimate time-invariant vars)
    fe_vars = set(fe_result.params.index)
    re_vars = set(re_result.params.index)
    common_vars = sorted(fe_vars & re_vars)

    if len(common_vars) == 0:
        raise ValueError("No common variables between FE and RE models")

    # Extract coefficients for common variables
    b_fe = fe_result.params[common_vars].values
    b_re = re_result.params[common_vars].values

    # Difference
    b_diff = b_fe - b_re

    # Extract covariance matrices for common variables
    V_fe = fe_result.cov_params.loc[common_vars, common_vars].values
    V_re = re_result.cov_params.loc[common_vars, common_vars].values

    # Variance of difference: V(β_FE - β_RE) = V(β_FE) - V(β_RE)
    # Under H0, Cov(β_FE, β_RE) = V(β_RE) because RE is efficient
    V_diff = V_fe - V_re

    # Check if V_diff is positive definite
    eigenvalues = np.linalg.eigvalsh(V_diff)
    if np.min(eigenvalues) < -1e-8:
        warnings.warn(
            "Variance difference matrix is not positive semi-definite. "
            "This may indicate model misspecification. "
            f"Minimum eigenvalue: {np.min(eigenvalues):.2e}",
            RuntimeWarning,
        )

    # Compute test statistic
    try:
        V_diff_inv = np.linalg.inv(V_diff)
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        V_diff_inv = np.linalg.pinv(V_diff)
        warnings.warn(
            "Variance difference matrix is singular. Using pseudo-inverse. "
            "Results may be unreliable.",
            RuntimeWarning,
        )

    # Hausman statistic
    H = b_diff @ V_diff_inv @ b_diff

    # Degrees of freedom
    df = len(common_vars)

    # P-value from chi-squared distribution
    from scipy.stats import chi2

    pvalue = 1 - chi2.cdf(H, df)

    # Interpretation
    if pvalue < 0.05:
        interpretation = (
            f"Reject H0: Random Effects estimator appears inconsistent (p={pvalue:.4f}).\n"
            "Fixed Effects is preferred. This suggests correlation between\n"
            "unobserved heterogeneity and regressors."
        )
    else:
        interpretation = (
            f"Fail to reject H0: No evidence against Random Effects (p={pvalue:.4f}).\n"
            "Random Effects can be used. This suggests no significant\n"
            "correlation between unobserved heterogeneity and regressors."
        )

    return HausmanTestResult(
        statistic=H,
        pvalue=pvalue,
        df=df,
        method="chi2",
        interpretation=interpretation,
        common_vars=common_vars,
    )


def hausman_test_discrete(
    fe_result, re_result, n_bootstrap: int = 999, seed: Optional[int] = None
) -> HausmanTestResult:
    """
    Hausman test for discrete panel models using bootstrap.

    For discrete models (Logit/Probit), the asymptotic distribution
    of the Hausman test is non-standard, so we use bootstrap.

    Parameters
    ----------
    fe_result : PanelResults
        Results from Fixed Effects Logit
    re_result : PanelResults
        Results from Random Effects Probit/Logit
    n_bootstrap : int, default=999
        Number of bootstrap replications
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    HausmanTestResult
        Test results with bootstrap p-value

    Examples
    --------
    >>> # Fixed Effects Logit
    >>> fe_logit = pb.FixedEffectsLogit("y ~ x1 + x2", data, "entity", "time")
    >>> fe_results = fe_logit.fit()
    >>>
    >>> # Random Effects Probit
    >>> re_probit = pb.RandomEffectsProbit("y ~ x1 + x2", data, "entity", "time")
    >>> re_results = re_probit.fit()
    >>>
    >>> # Hausman test with bootstrap
    >>> hausman = hausman_test_discrete(fe_results, re_results, n_bootstrap=500)
    >>> print(hausman.summary())

    Notes
    -----
    Bootstrap procedure:
    1. Resample entities (cluster bootstrap)
    2. Re-estimate both models
    3. Compute test statistic
    4. Repeat n_bootstrap times
    5. P-value = proportion of bootstrap statistics >= observed

    This is computationally intensive for large datasets.
    """
    if seed is not None:
        np.random.seed(seed)

    # Get common variables
    fe_vars = set(fe_result.params.index)
    re_vars = set(re_result.params.index)

    # For RE Probit, exclude sigma_alpha
    if "log_sigma_alpha" in re_vars:
        re_vars.remove("log_sigma_alpha")

    common_vars = sorted(fe_vars & re_vars)

    if len(common_vars) == 0:
        raise ValueError("No common variables between FE and RE models")

    # Observed test statistic
    b_fe_obs = fe_result.params[common_vars].values
    b_re_obs = re_result.params[common_vars].values
    b_diff_obs = b_fe_obs - b_re_obs

    # Get covariance matrices
    V_fe_obs = fe_result.cov_params.loc[common_vars, common_vars].values
    V_re_obs = re_result.cov_params.loc[common_vars, common_vars].values
    V_diff_obs = V_fe_obs - V_re_obs

    # Check for positive definiteness
    eigenvalues = np.linalg.eigvalsh(V_diff_obs)
    if np.min(eigenvalues) < -1e-8:
        warnings.warn(
            "Variance difference matrix is not positive semi-definite. "
            f"Minimum eigenvalue: {np.min(eigenvalues):.2e}",
            RuntimeWarning,
        )

    try:
        V_diff_inv = np.linalg.inv(V_diff_obs)
    except np.linalg.LinAlgError:
        V_diff_inv = np.linalg.pinv(V_diff_obs)
        warnings.warn("Using pseudo-inverse for variance difference matrix.")

    # Observed Hausman statistic
    H_obs = b_diff_obs @ V_diff_inv @ b_diff_obs

    # Bootstrap
    print(f"Running bootstrap with {n_bootstrap} replications...")
    boot_stats = []

    # Get entity IDs
    fe_model = fe_result.model
    re_model = re_result.model

    # Check if models have entity information
    if not hasattr(fe_model, "data") or not hasattr(re_model, "data"):
        raise ValueError("Models must have 'data' attribute for bootstrap")

    entity_col = fe_model.data.entity_col
    entities = fe_model.data.data[entity_col].unique()
    n_entities = len(entities)

    for b in range(n_bootstrap):
        if (b + 1) % 100 == 0:
            print(f"  Bootstrap iteration {b+1}/{n_bootstrap}")

        # Resample entities with replacement
        boot_entities = np.random.choice(entities, size=n_entities, replace=True)

        # Create bootstrap sample
        boot_data = []
        for entity in boot_entities:
            entity_data = fe_model.data.data[fe_model.data.data[entity_col] == entity].copy()
            boot_data.append(entity_data)

        boot_df = pd.concat(boot_data, ignore_index=True)

        try:
            # Re-estimate FE model
            fe_boot_model = type(fe_model)(
                fe_model.formula, boot_df, entity_col, fe_model.data.time_col
            )
            fe_boot_result = fe_boot_model.fit()

            # Re-estimate RE model
            if hasattr(re_model, "quadrature_points"):
                # Random Effects Probit
                re_boot_model = type(re_model)(
                    re_model.formula,
                    boot_df,
                    entity_col,
                    re_model.data.time_col,
                    quadrature_points=re_model.quadrature_points,
                )
            else:
                re_boot_model = type(re_model)(
                    re_model.formula, boot_df, entity_col, re_model.data.time_col
                )
            re_boot_result = re_boot_model.fit()

            # Compute test statistic for bootstrap sample
            b_fe_boot = fe_boot_result.params[common_vars].values
            b_re_boot = re_boot_result.params[common_vars].values
            b_diff_boot = b_fe_boot - b_re_boot

            V_fe_boot = fe_boot_result.cov_params.loc[common_vars, common_vars].values
            V_re_boot = re_boot_result.cov_params.loc[common_vars, common_vars].values
            V_diff_boot = V_fe_boot - V_re_boot

            try:
                V_diff_boot_inv = np.linalg.inv(V_diff_boot)
                H_boot = b_diff_boot @ V_diff_boot_inv @ b_diff_boot
                boot_stats.append(H_boot)
            except np.linalg.LinAlgError:
                # Skip this bootstrap sample if matrix is singular
                continue

        except Exception as e:
            # Skip failed bootstrap iterations
            continue

    if len(boot_stats) < n_bootstrap / 2:
        warnings.warn(
            f"Only {len(boot_stats)} out of {n_bootstrap} bootstrap "
            "iterations succeeded. Results may be unreliable.",
            RuntimeWarning,
        )

    # Compute bootstrap p-value
    boot_stats = np.array(boot_stats)
    pvalue = np.mean(boot_stats >= H_obs)

    # Interpretation
    if pvalue < 0.05:
        interpretation = (
            f"Reject H0: Random Effects appears inconsistent (bootstrap p={pvalue:.4f}).\n"
            "Fixed Effects is preferred."
        )
    else:
        interpretation = (
            f"Fail to reject H0: No evidence against Random Effects (bootstrap p={pvalue:.4f}).\n"
            "Random Effects can be used."
        )

    return HausmanTestResult(
        statistic=H_obs,
        pvalue=pvalue,
        df=len(common_vars),
        method="bootstrap",
        interpretation=interpretation,
        common_vars=common_vars,
    )


def mundlak_test(re_result) -> dict:
    """
    Mundlak test for Random Effects specification.

    Tests whether random effects are correlated with regressors by
    including group means as additional regressors.

    Parameters
    ----------
    re_result : PanelResults
        Results from Random Effects estimation

    Returns
    -------
    dict
        Test results with F-statistic and p-value

    Notes
    -----
    The Mundlak approach augments the RE model with group means:
    y_it = X_it'β + X̄_i'γ + α_i + ε_it

    Under H0: γ = 0 (no correlation between α_i and X_i)

    References
    ----------
    .. [1] Mundlak, Y. (1978). "On the Pooling of Time Series and
           Cross Section Data." Econometrica, 46(1), 69-85.
    """
    raise NotImplementedError(
        "Mundlak test not yet implemented. " "This will be added in a future release."
    )
