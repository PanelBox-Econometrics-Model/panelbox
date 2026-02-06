"""
Modified Wald test for groupwise heteroskedasticity in fixed effects models.

References
----------
Greene, W. H. (2000). Econometric Analysis (4th ed.). Prentice Hall.

Baum, C. F. (2001). Residual diagnostics for cross-section time series
regression models. The Stata Journal, 1(1), 101-104.

Stata command: xttest3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panelbox.core.results import PanelResults

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class ModifiedWaldTest(ValidationTest):
    """
    Modified Wald test for groupwise heteroskedasticity.

    Tests the null hypothesis that the error variance is the same across
    all cross-sectional units (entities) against the alternative that
    variances differ across groups.

    H0: σ²_1 = σ²_2 = ... = σ²_N
    H1: σ²_i ≠ σ²_j for some i ≠ j

    This test is specifically designed for fixed effects panel models and
    is robust to serial correlation.

    Parameters
    ----------
    results : PanelResults
        Results from panel model estimation (preferably Fixed Effects)

    Notes
    -----
    **Test Statistic:**

    The Modified Wald statistic is computed as:

        W = Σᵢ₌₁ᴺ Tᵢ ln(σ̂²_pooled / σ̂²_i)

    where:
    - N = number of entities
    - Tᵢ = number of time periods for entity i
    - σ̂²_i = estimated variance for entity i
    - σ̂²_pooled = pooled variance across all entities

    Under H0, W ~ χ²(N).

    **When to Use:**

    This test is particularly useful for:

    - **Fixed Effects models**: Designed specifically for within estimator
    - **Detecting groupwise patterns**: Tests if different entities have
      different error variances
    - **Pre-testing**: Before choosing appropriate standard errors
    - **Model diagnostics**: Checking homoskedasticity assumption

    **Advantages:**

    - **Simple and intuitive**: Easy to implement and interpret
    - **Robust to serial correlation**: Unlike Breusch-Pagan test
    - **Handles unbalanced panels**: Weights by entity sample size
    - **FE-specific**: Exploits FE structure efficiently

    **Limitations:**

    1. **FE-specific**: Less appropriate for other estimators
    2. **Groupwise only**: Tests entity-level heteroskedasticity, not
       general heteroskedasticity patterns
    3. **Large N**: Chi-square approximation requires sufficient entities
    4. **Assumes independence**: Cross-sectional independence of errors

    **Interpretation:**

    | P-value | Variance Ratio | Interpretation |
    |---------|----------------|----------------|
    | < 0.01  | > 10 | Strong heteroskedasticity |
    | 0.01-0.05 | 5-10 | Moderate heteroskedasticity |
    | 0.05-0.10 | 2-5 | Weak heteroskedasticity |
    | > 0.10  | < 2 | No evidence of heteroskedasticity |

    **Variance Ratio:**

    The variance ratio (max variance / min variance) provides intuitive measure:

    - Ratio < 2: Relatively homogeneous variances
    - Ratio 2-5: Moderate heterogeneity
    - Ratio 5-10: Strong heterogeneity
    - Ratio > 10: Very strong heterogeneity

    **If Heteroskedasticity is Detected:**

    1. **Use robust standard errors**:
       ```python
       result = fe.fit(cov_type="robust")  # White's heteroskedasticity-robust
       ```

    2. **Use cluster-robust standard errors** (also handles serial correlation):
       ```python
       result = fe.fit(cov_type="clustered", cluster_entity=True)
       ```

    3. **Use Feasible GLS (FGLS)** to exploit heteroskedasticity:
       ```python
       from panelbox import FeasibleGLS
       fgls = FeasibleGLS(data, "y", ["x1", "x2"], "entity", "time")
       result_fgls = fgls.fit()
       ```

    4. **Transform variables** to stabilize variance:
       ```python
       data['log_y'] = np.log(data['y'])  # Log transformation
       ```

    **Relationship to Other Tests:**

    | Test | Null Hypothesis | Best For |
    |------|----------------|----------|
    | **Modified Wald** | Groupwise homoskedasticity | FE models |
    | Breusch-Pagan | General homoskedasticity | OLS, RE models |
    | White | General homoskedasticity | Cross-section |
    | Levene | Equal variances | ANOVA-type tests |

    **Practical Considerations:**

    - Always check variance ratio in addition to p-value
    - Small variance ratios with significant p-value suggest:
      - Large sample size detecting small deviations
      - May not be economically significant
    - Large variance ratios (> 10) indicate serious issue even if p > 0.05
    - Consider entity characteristics that may explain variance differences

    **Comparison with Stata:**

    This test is equivalent to Stata's `xttest3` command:

    ```stata
    xtreg y x1 x2, fe
    xttest3
    ```

    References
    ----------
    .. [1] Greene, W. H. (2000). "Econometric Analysis" (4th ed.).
           Prentice Hall. Chapter 14.

    .. [2] Baum, C. F. (2001). "Residual diagnostics for cross-section time
           series regression models." *The Stata Journal*, 1(1), 101-104.

    .. [3] Wooldridge, J. M. (2010). "Econometric Analysis of Cross Section
           and Panel Data" (2nd ed.). MIT Press. Chapter 10.

    See Also
    --------
    BreuschPaganTest : Lagrange Multiplier test for heteroskedasticity
    WhiteTest : General heteroskedasticity test

    Examples
    --------
    **Basic usage:**

    >>> import panelbox as pb
    >>>
    >>> # Estimate Fixed Effects model
    >>> fe = pb.FixedEffects(data, "y", ["x1", "x2"], "firm", "year")
    >>> results = fe.fit()
    >>>
    >>> # Test for groupwise heteroskedasticity
    >>> from panelbox.validation.heteroskedasticity import ModifiedWaldTest
    >>> test = ModifiedWaldTest(results)
    >>> result = test.run()
    >>> print(result)

    **Interpreting results:**

    >>> print(f"Wald statistic: {result.statistic:.3f}")
    >>> print(f"P-value: {result.pvalue:.4f}")
    >>>
    >>> # Check variance ratio
    >>> var_ratio = result.metadata['variance_ratio']
    >>> print(f"Variance ratio (max/min): {var_ratio:.2f}")
    >>>
    >>> if result.pvalue < 0.05:
    ...     print("Evidence of groupwise heteroskedasticity")
    ...     if var_ratio > 10:
    ...         print("Strong heterogeneity - use FGLS or robust SEs")
    ...     else:
    ...         print("Moderate heterogeneity - robust SEs sufficient")
    >>> else:
    ...     print("No evidence of heteroskedasticity")

    **Examining entity variances:**

    >>> # Access variance statistics
    >>> print(f"Pooled variance: {result.metadata['pooled_variance']:.4f}")
    >>> print(f"Min entity variance: {result.metadata['min_entity_var']:.4f}")
    >>> print(f"Max entity variance: {result.metadata['max_entity_var']:.4f}")
    >>>
    >>> # Compare to pooled
    >>> print(f"Max is {result.metadata['max_entity_var']/result.metadata['pooled_variance']:.1f}x pooled")

    **Using robust standard errors if heteroskedasticity detected:**

    >>> test_result = test.run()
    >>> if test_result.pvalue < 0.05:
    ...     # Re-estimate with robust SEs
    ...     results_robust = fe.fit(cov_type="robust")
    ...     print(results_robust.summary())

    **Comparing different variance correction methods:**

    >>> # Standard
    >>> result_std = fe.fit()
    >>> # Robust
    >>> result_robust = fe.fit(cov_type="robust")
    >>> # Clustered
    >>> result_cluster = fe.fit(cov_type="clustered", cluster_entity=True)
    >>>
    >>> # Compare standard errors for key coefficient
    >>> coef_name = "x1"
    >>> se_std = result_std.std_errors[coef_name]
    >>> se_robust = result_robust.std_errors[coef_name]
    >>> se_cluster = result_cluster.std_errors[coef_name]
    >>> print(f"SE ratio (robust/standard): {se_robust/se_std:.2f}")
    >>> print(f"SE ratio (cluster/standard): {se_cluster/se_std:.2f}")
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Modified Wald test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation (preferably Fixed Effects)
        """
        super().__init__(results)

        # Check if model is suitable
        if "Fixed Effects" not in self.model_type:
            import warnings

            warnings.warn(
                "Modified Wald test is designed for Fixed Effects models. "
                f"Current model: {self.model_type}"
            )

    def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult:
        """
        Run Modified Wald test for groupwise heteroskedasticity.

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
            If required data indices are not available
        """
        # Get residuals with entity information
        resid_df = self._prepare_residual_data()

        # Compute variance for each entity
        entity_vars = resid_df.groupby("entity")["resid"].var()
        entity_counts = resid_df.groupby("entity").size()

        n_entities = len(entity_vars)

        # Modified Wald statistic
        # sum over i of: (T_i - 1) * ln(sigma²_pooled) - ln(sigma²_i)
        # where sigma²_pooled is the pooled variance

        # Pooled variance (weighted by sample size)
        total_resid_sq = np.sum((resid_df["resid"] ** 2).values)
        total_obs = len(resid_df)
        k = len(self.params)  # number of parameters
        pooled_var = total_resid_sq / (total_obs - n_entities - k)

        # Wald statistic
        wald_stat = 0.0
        for entity in entity_vars.index:
            T_i = entity_counts[entity]
            sigma2_i = entity_vars[entity]

            if sigma2_i <= 0:
                continue

            wald_stat += T_i * np.log(pooled_var / sigma2_i)

        # Under H0, the statistic is approximately chi2(N)
        df = n_entities
        pvalue = 1 - stats.chi2.cdf(wald_stat, df)

        # Metadata
        metadata = {
            "n_entities": n_entities,
            "pooled_variance": pooled_var,
            "min_entity_var": entity_vars.min(),
            "max_entity_var": entity_vars.max(),
            "variance_ratio": (
                entity_vars.max() / entity_vars.min() if entity_vars.min() > 0 else np.inf
            ),
        }

        result = ValidationTestResult(
            test_name="Modified Wald Test for Groupwise Heteroskedasticity",
            statistic=wald_stat,
            pvalue=pvalue,
            null_hypothesis="Homoskedasticity (constant variance across entities)",
            alternative_hypothesis="Groupwise heteroskedasticity present",
            alpha=alpha,
            df=df,
            metadata=metadata,
        )

        return result

    def _prepare_residual_data(self) -> pd.DataFrame:
        """
        Prepare residual data with entity identifiers.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: entity, resid
        """
        if hasattr(self.results, "entity_index"):
            # Ensure resid is 1D
            resid_flat = self.resid.ravel() if hasattr(self.resid, "ravel") else self.resid

            resid_df = pd.DataFrame({"entity": self.results.entity_index, "resid": resid_flat})
        else:
            raise AttributeError(
                "Results object must have 'entity_index' attribute. "
                "Please ensure your model stores this during estimation."
            )

        return resid_df
