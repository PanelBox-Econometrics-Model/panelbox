"""
Pesaran CD test for cross-sectional dependence in panel data.

References
----------
Pesaran, M. H. (2004). General diagnostic tests for cross section dependence
in panels. University of Cambridge, Faculty of Economics, Cambridge Working
Papers in Economics No. 0435.

Stata command: xtcd
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panelbox.core.results import PanelResults

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class PesaranCDTest(ValidationTest):
    """
    Pesaran CD test for cross-sectional dependence.

    Tests the null hypothesis of cross-sectional independence against the
    alternative of cross-sectional dependence.

    The test is based on the average of pairwise correlation coefficients
    of the residuals.

    H0: No cross-sectional dependence (residuals are independent across entities)
    H1: Cross-sectional dependence present

    Parameters
    ----------
    results : PanelResults
        Results from panel model estimation

    Notes
    -----
    **Test Statistic:**

    The Pesaran CD statistic is computed as:

        CD = √(2T / (N(N-1))) × Σᵢ<ⱼ ρ̂ᵢⱼ

    where:
    - N = number of entities
    - T = number of time periods
    - ρ̂ᵢⱼ = sample correlation of residuals between entities i and j

    Under H0, CD ~ N(0,1) asymptotically as N → ∞.

    **When to Use:**

    This test is particularly useful for:

    - **Checking model validity**: Detecting omitted spatial effects or
      common shocks not captured by time effects
    - **Panel VAR models**: Testing for cross-sectional independence
    - **Large N panels**: Works well even when T is small
    - **Pre-testing**: Before applying Driscoll-Kraay or panel-corrected
      standard errors

    **Advantages:**

    - Simple and computationally efficient (O(N²) complexity)
    - Valid for both balanced and unbalanced panels
    - Works well for small T, large N panels
    - Does not require normality assumptions
    - Robust to heteroskedasticity

    **Limitations:**

    1. **Requires T ≥ 3**: Need minimum time periods to compute correlations
    2. **Large N required**: Asymptotic distribution requires N → ∞
    3. **Not powerful for weak dependence**: May miss weak spatial patterns
    4. **Assumes no structural breaks**: Common shocks should be stable

    **Interpretation:**

    | CD Statistic | P-value | Interpretation |
    |--------------|---------|----------------|
    | \|CD\| < 1.645 | > 0.10 | No evidence of dependence |
    | 1.645 < \|CD\| < 1.96 | 0.05-0.10 | Weak evidence |
    | 1.96 < \|CD\| < 2.576 | 0.01-0.05 | Moderate dependence |
    | \|CD\| > 2.576 | < 0.01 | Strong dependence |

    **Average Correlation Guidelines:**

    - \|ρ̄\| < 0.1: Weak dependence (likely negligible)
    - 0.1 ≤ \|ρ̄\| < 0.3: Moderate dependence (consider robust SEs)
    - 0.3 ≤ \|ρ̄\| < 0.5: Strong dependence (require spatial models)
    - \|ρ̄\| ≥ 0.5: Very strong dependence (serious misspecification)

    **If Cross-Sectional Dependence is Detected:**

    1. **Include time fixed effects** to control for common shocks:
       ```python
       fe = pb.FixedEffects(data, "y", ["x1", "x2"], "firm", "year",
                           time_effects=True)
       ```

    2. **Use Driscoll-Kraay standard errors** (robust to cross-sectional
       and serial correlation):
       ```python
       result = fe.fit(cov_type="driscoll-kraay")
       ```

    3. **Use panel-corrected standard errors (PCSE)**:
       ```python
       result = fe.fit(cov_type="pcse")
       ```

    4. **Consider spatial panel models** if geographic structure is known

    5. **Add common correlated effects** (Pesaran CCE estimator)

    **Comparison with Stata:**

    This test is equivalent to Stata's `xtcd` command:

    ```stata
    xtreg y x1 x2, fe
    xtcd, pesaran
    ```

    References
    ----------
    .. [1] Pesaran, M. H. (2004). "General diagnostic tests for cross section
           dependence in panels." *University of Cambridge Working Paper*,
           No. 0435.

    .. [2] Pesaran, M. H. (2015). "Testing weak cross-sectional dependence in
           large panels." *Econometric Reviews*, 34(6-10), 1089-1117.

    .. [3] De Hoyos, R. E., & Sarafidis, V. (2006). "Testing for cross-sectional
           dependence in panel-data models." *Stata Journal*, 6(4), 482-496.

    See Also
    --------
    BreuschPaganLMTest : Lagrange Multiplier test for cross-sectional dependence
    FreesTest : Distribution-free test for cross-sectional dependence

    Examples
    --------
    **Basic usage:**

    >>> import panelbox as pb
    >>>
    >>> # Estimate model
    >>> fe = pb.FixedEffects(data, "y", ["x1", "x2"], "firm", "year")
    >>> results = fe.fit()
    >>>
    >>> # Test for cross-sectional dependence
    >>> from panelbox.validation.cross_sectional_dependence import PesaranCDTest
    >>> test = PesaranCDTest(results)
    >>> result = test.run()
    >>> print(result)

    **Interpreting results:**

    >>> print(f"CD statistic: {result.statistic:.3f}")
    >>> print(f"P-value: {result.pvalue:.4f}")
    >>>
    >>> # Check average correlation
    >>> avg_corr = result.metadata['avg_abs_correlation']
    >>> print(f"Average absolute correlation: {avg_corr:.3f}")
    >>>
    >>> if result.pvalue < 0.05:
    ...     print("Evidence of cross-sectional dependence")
    ...     if avg_corr < 0.3:
    ...         print("Use Driscoll-Kraay standard errors")
    ...     else:
    ...         print("Consider spatial panel model")
    >>> else:
    ...     print("No evidence of cross-sectional dependence")

    **Examining correlation patterns:**

    >>> # Access detailed correlation statistics
    >>> print(f"Number of entity pairs: {result.metadata['n_pairs']}")
    >>> print(f"Average correlation: {result.metadata['avg_correlation']:.3f}")
    >>> print(f"Max absolute correlation: {result.metadata['max_abs_correlation']:.3f}")
    >>> print(f"Range: [{result.metadata['min_correlation']:.3f}, "
    ...       f"{result.metadata['max_correlation']:.3f}]")

    **Testing with time effects:**

    >>> # Include time effects to control for common shocks
    >>> fe_te = pb.FixedEffects(data, "y", ["x1", "x2"], "firm", "year",
    ...                         time_effects=True)
    >>> results_te = fe_te.fit()
    >>>
    >>> # Re-test
    >>> test_te = PesaranCDTest(results_te)
    >>> result_te = test_te.run()
    >>> print(f"CD with time effects: {result_te.statistic:.3f}")
    >>> print(f"Reduction in dependence: "
    ...       f"{(1 - result_te.statistic/result.statistic)*100:.1f}%")
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Pesaran CD test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)

    def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult:
        """
        Run Pesaran CD test for cross-sectional dependence.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results

        Raises
        ------
        ValueError
            If required data indices are not available or T < 3
        """
        # Get residuals with entity and time information
        resid_df = self._prepare_residual_data()

        # Reshape residuals to wide format (entities as columns, time as rows)
        resid_wide = resid_df.pivot(index="time", columns="entity", values="resid")

        # Check minimum time periods
        T = len(resid_wide)
        if T < 3:
            raise ValueError(f"Pesaran CD test requires at least 3 time periods. Found: {T}")

        N = len(resid_wide.columns)

        # Compute pairwise correlations
        correlations = []
        T_ij_list = []  # Effective sample size for each pair

        for i, j in combinations(range(N), 2):
            # Get residuals for entities i and j
            e_i = resid_wide.iloc[:, i]
            e_j = resid_wide.iloc[:, j]

            # Drop missing values for this pair
            valid = ~(e_i.isna() | e_j.isna())
            e_i_valid = e_i[valid]
            e_j_valid = e_j[valid]

            T_ij = len(e_i_valid)

            if T_ij >= 3:  # Need at least 3 observations to compute correlation
                # Correlation coefficient
                rho_ij = np.corrcoef(e_i_valid, e_j_valid)[0, 1]
                correlations.append(rho_ij)
                T_ij_list.append(T_ij)

        if len(correlations) == 0:
            raise ValueError("No valid pairwise correlations could be computed")

        # Pesaran CD statistic
        # CD = sqrt(2T / (N(N-1))) * sum(rho_ij)
        rho_sum = np.sum(correlations)

        # Use average T for unbalanced panels
        T_avg = np.mean(T_ij_list) if len(T_ij_list) > 0 else T

        cd_stat = np.sqrt(2 * T_avg / (N * (N - 1))) * rho_sum

        # Under H0, CD ~ N(0,1)
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(cd_stat)))

        # Average absolute correlation
        avg_abs_corr = np.mean(np.abs(correlations))

        # Metadata
        metadata = {
            "n_entities": N,
            "n_time_periods": T,
            "n_pairs": len(correlations),
            "avg_correlation": np.mean(correlations),
            "avg_abs_correlation": avg_abs_corr,
            "max_abs_correlation": np.max(np.abs(correlations)),
            "min_correlation": np.min(correlations),
            "max_correlation": np.max(correlations),
        }

        result = ValidationTestResult(
            test_name="Pesaran CD Test for Cross-Sectional Dependence",
            statistic=cd_stat,
            pvalue=pvalue,
            null_hypothesis="No cross-sectional dependence",
            alternative_hypothesis="Cross-sectional dependence present",
            alpha=alpha,
            df=None,
            metadata=metadata,
        )

        return result

    def _prepare_residual_data(self) -> pd.DataFrame:
        """
        Prepare residual data with entity and time identifiers.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: entity, time, resid
        """
        if hasattr(self.results, "entity_index") and hasattr(self.results, "time_index"):
            # Ensure resid is 1D
            resid_flat = self.resid.ravel() if hasattr(self.resid, "ravel") else self.resid

            resid_df = pd.DataFrame(
                {
                    "entity": self.results.entity_index,
                    "time": self.results.time_index,
                    "resid": resid_flat,
                }
            )
        else:
            raise AttributeError(
                "Results object must have 'entity_index' and 'time_index' attributes. "
                "Please ensure your model stores these during estimation."
            )

        return resid_df
