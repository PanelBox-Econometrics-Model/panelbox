"""
Encompassing Tests for Model Selection.

This module implements various encompassing tests to compare non-nested
and nested models. Encompasses means one model can explain the results
of another model.

References
----------
- Mizon, G.E., & Richard, J.F. (1986). "The Encompassing Principle and its
  Application to Testing Non-nested Hypotheses." Econometrica, 54(3), 657-678.
- Cox, D.R. (1961). "Tests of Separate Families of Hypotheses."
  Proceedings of the Fourth Berkeley Symposium, 1, 105-123.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class EncompassingResult:
    """
    Results from encompassing tests.

    Attributes
    ----------
    test_name : str
        Name of the test performed
    statistic : float
        Test statistic value
    pvalue : float
        P-value of the test
    df : float or None
        Degrees of freedom (for chi-square tests)
    null_hypothesis : str
        Description of the null hypothesis
    alternative : str
        Description of the alternative hypothesis
    model1_name : str
        Name of Model 1
    model2_name : str
        Name of Model 2
    additional_info : dict
        Additional test-specific information
    """

    test_name: str
    statistic: float
    pvalue: float
    df: Optional[float]
    null_hypothesis: str
    alternative: str
    model1_name: str
    model2_name: str
    additional_info: Dict[str, Any]

    def interpretation(self, alpha: float = 0.05) -> str:
        """
        Provide interpretation of test results.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level

        Returns
        -------
        str
            Interpretation of test results
        """
        reject = self.pvalue < alpha

        interpretation = f"{self.test_name}\n"
        interpretation += "=" * 60 + "\n\n"
        interpretation += f"H0: {self.null_hypothesis}\n"
        interpretation += f"Ha: {self.alternative}\n\n"
        interpretation += f"Test statistic: {self.statistic:.4f}\n"
        if self.df is not None:
            interpretation += f"Degrees of freedom: {self.df}\n"
        interpretation += f"p-value: {self.pvalue:.4f}\n\n"

        interpretation += "Decision:\n"
        interpretation += "-" * 60 + "\n"
        if reject:
            interpretation += f"REJECT H0 at {alpha} level (p={self.pvalue:.4f} < {alpha})\n"
            interpretation += f"{self.alternative}\n"
        else:
            interpretation += (
                f"FAIL TO REJECT H0 at {alpha} level (p={self.pvalue:.4f} >= {alpha})\n"
            )
            interpretation += f"{self.null_hypothesis}\n"

        return interpretation

    def summary(self) -> pd.DataFrame:
        """
        Return formatted summary table.

        Returns
        -------
        pd.DataFrame
            Summary of test results
        """
        data = {
            "Test": [self.test_name],
            "Statistic": [self.statistic],
            "p-value": [self.pvalue],
            "Null Hypothesis": [self.null_hypothesis],
        }

        if self.df is not None:
            data["df"] = [self.df]

        return pd.DataFrame(data)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.test_name} (p={self.pvalue:.4f})\n\n{self.interpretation()}"


def cox_test(
    result1, result2, model1_name: Optional[str] = None, model2_name: Optional[str] = None
) -> EncompassingResult:
    """
    Perform Cox test for non-nested models.

    The Cox test compares the log-likelihoods of two non-nested models
    adjusted for their variances. Model 1 encompasses Model 2 if the
    adjusted log-likelihood ratio favors Model 1.

    Parameters
    ----------
    result1 : fitted model result
        First model's fitted result. Must have .llf (log-likelihood)
    result2 : fitted model result
        Second model's fitted result
    model1_name : str, optional
        Name for Model 1 (default: 'Model 1')
    model2_name : str, optional
        Name for Model 2 (default: 'Model 2')

    Returns
    -------
    EncompassingResult
        Test results with interpretation

    Notes
    -----
    The Cox test statistic is:

    T = (llf₁ - llf₂) / sqrt(Var(llf₁ - llf₂))

    Under H0 that the models are equally good, T ~ N(0,1)

    References
    ----------
    Cox, D.R. (1961). "Tests of Separate Families of Hypotheses."
    Proceedings of the Fourth Berkeley Symposium, 1, 105-123.

    Examples
    --------
    >>> from statsmodels.regression.linear_model import OLS
    >>> # Fit two competing models
    >>> result1 = OLS(y, X1).fit()
    >>> result2 = OLS(y, X2).fit()
    >>> # Perform Cox test
    >>> cox_result = cox_test(result1, result2,
    ...                       model1_name='Linear',
    ...                       model2_name='Quadratic')
    >>> print(cox_result.interpretation())
    """
    # Set default names
    if model1_name is None:
        model1_name = "Model 1"
    if model2_name is None:
        model2_name = "Model 2"

    # Check for log-likelihood
    if not hasattr(result1, "llf") or not hasattr(result2, "llf"):
        raise AttributeError(
            "Both models must have log-likelihood (.llf attribute). "
            "Cox test requires likelihood-based estimation."
        )

    llf1 = result1.llf
    llf2 = result2.llf

    # Get number of observations
    n = len(result1.model.endog)

    # Compute log-likelihood difference
    llf_diff = llf1 - llf2

    # Estimate variance of log-likelihood difference
    # This is a simplified version; full Cox test requires computing
    # variance based on the models' information matrices
    # For now, we use an approximation based on the difference in residuals

    # Get residuals
    if hasattr(result1, "resid") and hasattr(result2, "resid"):
        resid1 = result1.resid
        resid2 = result2.resid

        # Variance approximation
        var_diff = np.var(np.log(resid1**2) - np.log(resid2**2), ddof=1)
        se_diff = np.sqrt(var_diff / n)
    else:
        # Fallback: use simple approximation based on sample size
        se_diff = np.sqrt(2 / n)
        warnings.warn("Could not compute exact variance. Using approximation.", UserWarning)

    # Test statistic
    if se_diff > 0:
        t_stat = llf_diff / se_diff
    else:
        t_stat = np.inf if llf_diff > 0 else -np.inf
        warnings.warn("Standard error is zero. Test may be unreliable.", UserWarning)

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return EncompassingResult(
        test_name="Cox Test for Non-nested Models",
        statistic=t_stat,
        pvalue=p_value,
        df=None,
        null_hypothesis=f"{model1_name} and {model2_name} fit equally well",
        alternative=f"Models have different fit quality",
        model1_name=model1_name,
        model2_name=model2_name,
        additional_info={"llf1": llf1, "llf2": llf2, "llf_diff": llf_diff, "se_diff": se_diff},
    )


def wald_encompassing_test(
    result_restricted,
    result_unrestricted,
    model_restricted_name: Optional[str] = None,
    model_unrestricted_name: Optional[str] = None,
) -> EncompassingResult:
    """
    Perform Wald encompassing test for nested models.

    Tests whether the unrestricted model significantly improves upon
    the restricted model using a Wald test on the additional parameters.

    Parameters
    ----------
    result_restricted : fitted model result
        Restricted (nested) model result
    result_unrestricted : fitted model result
        Unrestricted (full) model result
    model_restricted_name : str, optional
        Name for restricted model (default: 'Restricted')
    model_unrestricted_name : str, optional
        Name for unrestricted model (default: 'Unrestricted')

    Returns
    -------
    EncompassingResult
        Test results with interpretation

    Notes
    -----
    The Wald test statistic is:

    W = (β_r)' [Var(β_r)]^(-1) (β_r)

    where β_r are the restricted parameters in the unrestricted model.
    Under H0, W ~ χ²(df) where df is the number of restrictions.

    Examples
    --------
    >>> # Restricted model: y ~ x1
    >>> result_r = OLS(y, X_restricted).fit()
    >>> # Unrestricted model: y ~ x1 + x2 + x3
    >>> result_u = OLS(y, X_unrestricted).fit()
    >>> # Test if additional variables are significant
    >>> wald_result = wald_encompassing_test(result_r, result_u)
    >>> print(wald_result.interpretation())
    """
    # Set default names
    if model_restricted_name is None:
        model_restricted_name = "Restricted Model"
    if model_unrestricted_name is None:
        model_unrestricted_name = "Unrestricted Model"

    # Get number of parameters
    k_restricted = result_restricted.params.shape[0]
    k_unrestricted = result_unrestricted.params.shape[0]

    # Check that unrestricted has more parameters
    if k_unrestricted <= k_restricted:
        raise ValueError(
            f"Unrestricted model must have more parameters than restricted. "
            f"Got: unrestricted={k_unrestricted}, restricted={k_restricted}"
        )

    df = k_unrestricted - k_restricted

    # Compute Wald statistic
    # W = (RSS_r - RSS_u) / (σ²_u * df)
    # where σ²_u = RSS_u / (n - k_u)

    if hasattr(result_restricted, "ssr") and hasattr(result_unrestricted, "ssr"):
        # OLS models
        rss_r = result_restricted.ssr
        rss_u = result_unrestricted.ssr
        n = len(result_unrestricted.model.endog)

        sigma2_u = rss_u / (n - k_unrestricted)
        wald_stat = (rss_r - rss_u) / sigma2_u

    elif hasattr(result_restricted, "llf") and hasattr(result_unrestricted, "llf"):
        # Likelihood-based models - use LR statistic
        llf_r = result_restricted.llf
        llf_u = result_unrestricted.llf
        wald_stat = 2 * (llf_u - llf_r)

    else:
        raise AttributeError(
            "Models must have either .ssr (sum of squared residuals) "
            "or .llf (log-likelihood) attributes."
        )

    # P-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(wald_stat, df)

    return EncompassingResult(
        test_name="Wald Encompassing Test",
        statistic=wald_stat,
        pvalue=p_value,
        df=df,
        null_hypothesis=f"{model_restricted_name} is adequate (restrictions valid)",
        alternative=f"{model_unrestricted_name} significantly improves fit",
        model1_name=model_restricted_name,
        model2_name=model_unrestricted_name,
        additional_info={
            "k_restricted": k_restricted,
            "k_unrestricted": k_unrestricted,
            "num_restrictions": df,
        },
    )


def likelihood_ratio_test(
    result_restricted,
    result_unrestricted,
    model_restricted_name: Optional[str] = None,
    model_unrestricted_name: Optional[str] = None,
) -> EncompassingResult:
    """
    Perform likelihood ratio test for nested models.

    Tests whether the unrestricted model significantly improves upon
    the restricted model using the likelihood ratio statistic.

    Parameters
    ----------
    result_restricted : fitted model result
        Restricted (nested) model result. Must have .llf attribute.
    result_unrestricted : fitted model result
        Unrestricted (full) model result. Must have .llf attribute.
    model_restricted_name : str, optional
        Name for restricted model (default: 'Restricted')
    model_unrestricted_name : str, optional
        Name for unrestricted model (default: 'Unrestricted')

    Returns
    -------
    EncompassingResult
        Test results with interpretation

    Notes
    -----
    The likelihood ratio statistic is:

    LR = -2(llf_r - llf_u)

    Under H0 that the restrictions are valid, LR ~ χ²(df) where df is
    the number of restrictions.

    This test is only valid for nested models estimated by maximum likelihood.

    Examples
    --------
    >>> # Fit restricted and unrestricted models
    >>> result_r = Logit(y, X_restricted).fit()
    >>> result_u = Logit(y, X_unrestricted).fit()
    >>> # Perform LR test
    >>> lr_result = likelihood_ratio_test(result_r, result_u)
    >>> print(lr_result.interpretation())
    """
    # Set default names
    if model_restricted_name is None:
        model_restricted_name = "Restricted Model"
    if model_unrestricted_name is None:
        model_unrestricted_name = "Unrestricted Model"

    # Check for log-likelihood
    if not hasattr(result_restricted, "llf"):
        raise AttributeError(
            "Restricted model must have .llf (log-likelihood) attribute. "
            "LR test requires likelihood-based estimation."
        )
    if not hasattr(result_unrestricted, "llf"):
        raise AttributeError(
            "Unrestricted model must have .llf (log-likelihood) attribute. "
            "LR test requires likelihood-based estimation."
        )

    # Get log-likelihoods
    llf_r = result_restricted.llf
    llf_u = result_unrestricted.llf

    # Check that unrestricted has higher log-likelihood
    if llf_u < llf_r - 1e-6:  # allow small numerical differences
        warnings.warn(
            f"Unrestricted model has lower log-likelihood ({llf_u:.4f}) than "
            f"restricted model ({llf_r:.4f}). This should not happen for nested models. "
            f"Check that models are correctly specified.",
            UserWarning,
        )

    # Get number of parameters
    k_r = result_restricted.params.shape[0]
    k_u = result_unrestricted.params.shape[0]

    # Check nesting
    if k_u <= k_r:
        raise ValueError(
            f"Unrestricted model must have more parameters than restricted. "
            f"Got: unrestricted={k_u}, restricted={k_r}"
        )

    df = k_u - k_r

    # Compute LR statistic
    lr_stat = -2 * (llf_r - llf_u)

    # Ensure non-negative (can be slightly negative due to numerical issues)
    if lr_stat < 0:
        if lr_stat < -1e-6:
            warnings.warn(
                f"LR statistic is negative ({lr_stat:.6f}). "
                f"Setting to 0. Check model specification.",
                UserWarning,
            )
        lr_stat = 0.0

    # P-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(lr_stat, df)

    return EncompassingResult(
        test_name="Likelihood Ratio Test",
        statistic=lr_stat,
        pvalue=p_value,
        df=df,
        null_hypothesis=f"{model_restricted_name} is adequate (restrictions valid)",
        alternative=f"{model_unrestricted_name} significantly improves fit",
        model1_name=model_restricted_name,
        model2_name=model_unrestricted_name,
        additional_info={
            "llf_restricted": llf_r,
            "llf_unrestricted": llf_u,
            "llf_diff": llf_u - llf_r,
            "k_restricted": k_r,
            "k_unrestricted": k_u,
            "num_restrictions": df,
        },
    )
