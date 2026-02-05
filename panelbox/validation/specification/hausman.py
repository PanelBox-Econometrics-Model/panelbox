"""
Hausman specification test for panel data.

This module provides the Hausman test for choosing between Fixed Effects
and Random Effects models.
"""


import numpy as np
import pandas as pd
from scipy import stats

from panelbox.core.results import PanelResults


class HausmanTestResult:
    """
    Container for Hausman test results.

    Attributes
    ----------
    statistic : float
        Chi-squared test statistic
    pvalue : float
        P-value
    df : int
        Degrees of freedom
    conclusion : str
        Interpretation of test result
    fe_params : pd.Series
        Fixed effects coefficients
    re_params : pd.Series
        Random effects coefficients
    diff : pd.Series
        Difference in coefficients (FE - RE)
    """

    def __init__(
        self,
        statistic: float,
        pvalue: float,
        df: int,
        fe_params: pd.Series,
        re_params: pd.Series,
        diff: pd.Series,
        alpha: float = 0.05,
    ):
        self.statistic = statistic
        self.pvalue = pvalue
        self.df = df
        self.fe_params = fe_params
        self.re_params = re_params
        self.diff = diff
        self.alpha = alpha

        # Determine conclusion
        if pvalue < alpha:
            self.conclusion = (
                f"Reject H0 at {alpha*100:.0f}% level. " "Use Fixed Effects (RE is inconsistent)."
            )
            self.recommendation = "Fixed Effects"
        else:
            self.conclusion = (
                f"Fail to reject H0 at {alpha*100:.0f}% level. "
                "Random Effects is consistent and efficient."
            )
            self.recommendation = "Random Effects"

    def __str__(self) -> str:
        """String representation."""
        return self.summary()

    def __repr__(self) -> str:
        """Repr."""
        return f"HausmanTestResult(statistic={self.statistic:.3f}, pvalue={self.pvalue:.4f}, df={self.df})"

    def summary(self) -> str:
        """
        Generate formatted summary.

        Returns
        -------
        str
            Formatted test results
        """
        lines = []
        lines.append("=" * 70)
        lines.append("HAUSMAN SPECIFICATION TEST")
        lines.append("=" * 70)
        lines.append("")
        lines.append("H0: Random Effects is consistent (and efficient)")
        lines.append("H1: Random Effects is inconsistent (use Fixed Effects)")
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"{'Test Statistic (Chi2)':<30} {self.statistic:>15.4f}")
        lines.append(f"{'P-value':<30} {self.pvalue:>15.4f}")
        lines.append(f"{'Degrees of Freedom':<30} {self.df:>15}")
        lines.append("-" * 70)
        lines.append("")
        lines.append(f"Conclusion: {self.conclusion}")
        lines.append(f"Recommendation: {self.recommendation}")
        lines.append("")

        # Coefficient comparison table
        lines.append("=" * 70)
        lines.append("COEFFICIENT COMPARISON")
        lines.append("=" * 70)
        lines.append(
            f"{'Variable':<15} {'Fixed Effects':<15} {'Random Effects':<15} {'Difference':<15}"
        )
        lines.append("-" * 70)

        for var in self.fe_params.index:
            fe_coef = self.fe_params[var]
            re_coef = self.re_params[var]
            diff_coef = self.diff[var]

            lines.append(f"{var:<15} {fe_coef:>14.4f} {re_coef:>14.4f} {diff_coef:>14.4f}")

        lines.append("=" * 70)
        lines.append("")

        return "\n".join(lines)


class HausmanTest:
    """
    Hausman specification test for panel data.

    Tests the null hypothesis that the Random Effects estimator is consistent
    (and efficient) against the alternative that it is inconsistent.

    The test compares Fixed Effects (always consistent under standard assumptions)
    with Random Effects (consistent only if E[u_i | X_it] = 0).

    Parameters
    ----------
    fe_results : PanelResults
        Results from Fixed Effects estimation
    re_results : PanelResults
        Results from Random Effects estimation

    Examples
    --------
    >>> import panelbox as pb
    >>>
    >>> # Estimate both models
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year")
    >>> fe_results = fe.fit()
    >>>
    >>> re = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year")
    >>> re_results = re.fit()
    >>>
    >>> # Run Hausman test
    >>> hausman = pb.HausmanTest(fe_results, re_results)
    >>> result = hausman.run()
    >>> print(result)
    >>>
    >>> # Use result
    >>> if result.recommendation == "Fixed Effects":
    ...     final_results = fe_results
    >>> else:
    ...     final_results = re_results
    """

    def __init__(self, fe_results: PanelResults, re_results: PanelResults):
        if fe_results.model_type not in [
            "Fixed Effects",
            "Fixed Effects (Two-Way)",
            "Fixed Effects (Time)",
        ]:
            raise ValueError("First argument must be Fixed Effects results")

        if re_results.model_type not in ["Random Effects (GLS)", "Random Effects"]:
            raise ValueError("Second argument must be Random Effects results")

        self.fe_results = fe_results
        self.re_results = re_results

        # Find common coefficients (exclude Intercept for FE, keep for RE)
        # FE doesn't have intercept, RE does
        fe_vars = set(fe_results.params.index)
        re_vars = set(re_results.params.index) - {"Intercept"}  # Exclude intercept from comparison

        self.common_vars = sorted(fe_vars & re_vars)

        if len(self.common_vars) == 0:
            raise ValueError("No common variables found between FE and RE models")

    def run(self, alpha: float = 0.05) -> HausmanTestResult:
        """
        Run the Hausman test.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for test

        Returns
        -------
        HausmanTestResult
            Test results

        Notes
        -----
        The Hausman test statistic is:

            H = (b_FE - b_RE)' [Var(b_FE) - Var(b_RE)]^{-1} (b_FE - b_RE)

        which follows a chi-squared distribution with K degrees of freedom,
        where K is the number of coefficients being tested.

        Examples
        --------
        >>> result = hausman.run(alpha=0.05)
        >>> print(f"Chi2 statistic: {result.statistic:.3f}")
        >>> print(f"P-value: {result.pvalue:.4f}")
        >>> print(f"Recommendation: {result.recommendation}")
        """
        # Extract coefficients for common variables
        beta_fe = self.fe_results.params[self.common_vars].values
        beta_re = self.re_results.params[self.common_vars].values

        # Difference in coefficients
        diff = beta_fe - beta_re

        # Extract covariance matrices
        vcov_fe = self.fe_results.cov_params.loc[self.common_vars, self.common_vars].values
        vcov_re = self.re_results.cov_params.loc[self.common_vars, self.common_vars].values

        # Variance of difference: Var(b_FE - b_RE) = Var(b_FE) - Var(b_RE)
        # Under H0, RE is efficient, so this is the correct variance
        var_diff = vcov_fe - vcov_re

        # Check if var_diff is positive definite
        # If not, use the generalized inverse
        try:
            var_diff_inv = np.linalg.inv(var_diff)
        except np.linalg.LinAlgError:
            # Matrix is singular, use pseudo-inverse
            var_diff_inv = np.linalg.pinv(var_diff)

        # Hausman statistic: (b_FE - b_RE)' [Var(diff)]^{-1} (b_FE - b_RE)
        statistic = float(diff.T @ var_diff_inv @ diff)

        # Degrees of freedom
        df = len(self.common_vars)

        # P-value from chi-squared distribution
        pvalue = 1 - stats.chi2.cdf(statistic, df)

        # Create result object
        result = HausmanTestResult(
            statistic=statistic,
            pvalue=pvalue,
            df=df,
            fe_params=self.fe_results.params[self.common_vars],
            re_params=self.re_results.params[self.common_vars],
            diff=pd.Series(diff, index=self.common_vars),
            alpha=alpha,
        )

        return result
