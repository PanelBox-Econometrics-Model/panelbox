"""
Wooldridge test for autocorrelation in panel data.

References
----------
Wooldridge, J. M. (2002). Econometric Analysis of Cross Section and Panel Data.
MIT Press, Section 10.4.1.

Stata command: xtserial
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from panelbox.core.results import PanelResults

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class WooldridgeARTest(ValidationTest):
    """
    Wooldridge test for first-order autocorrelation in panel data.

    This test is specifically designed for fixed effects models and tests
    for AR(1) autocorrelation in the idiosyncratic errors.

    The test is based on regressing the first-differenced residuals on their
    own lag and testing if the coefficient equals -0.5 (which is the value
    under H0 of no serial correlation).

    Parameters
    ----------
    results : PanelResults
        Results from panel model estimation (preferably Fixed Effects)

    Notes
    -----
    **Test Procedure:**

    The Wooldridge test implements the following steps:

    1. **Compute first differences of residuals:**

        Δe_it = e_it - e_{i,t-1}

    2. **Regress Δe_it on Δe_{i,t-1}:**

        Δe_it = β · Δe_{i,t-1} + ν_it

    3. **Test H0: β = -0.5**

    Under the null hypothesis of no serial correlation:

        Cov(Δe_it, Δe_{i,t-1}) = E[(e_it - e_{i,t-1})(e_{i,t-1} - e_{i,t-2})]
                                = -σ²_e

    And since Var(Δe_it) ≈ 2σ²_e, we expect β ≈ -0.5.

    **Test Statistic:**

    The F-statistic is computed as:

        F = [(β̂ + 0.5) / SE(β̂)]²  ~  F(1, N-1)

    where N is the number of entities.

    **When to Use:**

    This test is particularly useful for:

    - **Fixed Effects models**: Designed specifically for FE estimation
    - **GMM model validation**: Testing AR(1) assumption before GMM
    - **Dynamic panels**: Checking for residual autocorrelation after
      including lagged dependent variable
    - **Pre-testing**: Before using cluster-robust standard errors

    **Advantages:**

    - Simple to implement and interpret
    - Does not require strong distributional assumptions
    - Robust to heteroskedasticity
    - Works with unbalanced panels (requires T ≥ 3 for each entity)

    **Limitations:**

    1. **Only tests AR(1)**: Does not detect higher-order autocorrelation
    2. **Minimum T requirement**: Needs T ≥ 3 per entity
    3. **FE-specific**: Less powerful for other estimators
    4. **Small N issues**: F-distribution approximation may be poor with few entities

    **Interpretation:**

    | P-value | Decision | Interpretation |
    |---------|----------|----------------|
    | < 0.01  | Strong rejection | Strong AR(1) autocorrelation |
    | 0.01-0.05 | Rejection | Moderate autocorrelation |
    | 0.05-0.10 | Borderline | Consider robust SEs |
    | > 0.10  | Fail to reject | No evidence of AR(1) |

    **If Autocorrelation is Detected:**

    - Use **cluster-robust standard errors** (cluster by entity)
    - Use **Driscoll-Kraay standard errors** (robust to both serial
      correlation and cross-sectional dependence)
    - Consider **AR(1) error structure** in GLS estimation
    - For dynamic models, use **GMM estimators** (System GMM)

    **Comparison with Stata:**

    This test is equivalent to Stata's `xtserial` command:

    ```stata
    xtreg y x1 x2, fe
    xtserial y x1 x2
    ```

    References
    ----------
    .. [1] Wooldridge, J. M. (2002). "Econometric Analysis of Cross Section
           and Panel Data." MIT Press, Section 10.4.1.

    .. [2] Drukker, D. M. (2003). "Testing for serial correlation in linear
           panel-data models." *Stata Journal*, 3(2), 168-177.

    .. [3] Wooldridge, J. M. (2010). "Econometric Analysis of Cross Section
           and Panel Data" (2nd ed.). MIT Press, Chapter 10.

    See Also
    --------
    BaltagiWuTest : Alternative test for serial correlation in panel data
    BreuschGodfreyTest : Lagrange Multiplier test for serial correlation

    Examples
    --------
    **Basic usage with Fixed Effects:**

    >>> import panelbox as pb
    >>>
    >>> # Estimate Fixed Effects model
    >>> fe = pb.FixedEffects(data, "y", ["x1", "x2"], "firm", "year")
    >>> results = fe.fit()
    >>>
    >>> # Test for autocorrelation
    >>> from panelbox.validation.serial_correlation import WooldridgeARTest
    >>> test = WooldridgeARTest(results)
    >>> result = test.run()
    >>> print(result)

    **Interpreting results:**

    >>> print(f"F-statistic: {result.statistic:.3f}")
    >>> print(f"P-value: {result.pvalue:.4f}")
    >>>
    >>> if result.pvalue < 0.05:
    ...     print("Evidence of AR(1) autocorrelation")
    ...     print("Consider using cluster-robust standard errors")
    >>> else:
    ...     print("No evidence of autocorrelation")

    **Accessing additional information:**

    >>> # Estimated coefficient and its standard error
    >>> print(f"β̂ = {result.metadata['coefficient']:.4f}")
    >>> print(f"SE(β̂) = {result.metadata['std_error']:.4f}")
    >>> print(f"Expected under H0: β = -0.5")
    >>>
    >>> # Sample information
    >>> print(f"Number of entities: {result.metadata['n_entities']}")
    >>> print(f"Observations used: {result.metadata['n_obs_used']}")

    **Different significance level:**

    >>> # Test at 1% level
    >>> result_strict = test.run(alpha=0.01)
    >>> print(result_strict.conclusion)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Wooldridge AR test.

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
                "Wooldridge test is designed for Fixed Effects models. "
                f"Current model: {self.model_type}"
            )

    def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult:
        """
        Run Wooldridge test for AR(1) autocorrelation.

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
            If panel has fewer than 3 time periods
        """
        # Get residuals as DataFrame with entity and time info
        # We need to reconstruct the panel structure
        resid_df = self._prepare_residual_data()

        # Check minimum time periods
        min_T = resid_df.groupby("entity").size().min()
        if min_T < 3:
            raise ValueError(
                f"Wooldridge test requires at least 3 time periods. " f"Minimum found: {min_T}"
            )

        # Compute first differences of residuals
        resid_df = resid_df.sort_values(["entity", "time"])
        resid_df["resid_diff"] = resid_df.groupby("entity")["resid"].diff()
        resid_df["resid_diff_lag"] = resid_df.groupby("entity")["resid_diff"].shift(1)

        # Drop missing values (first two obs per entity are lost)
        resid_df = resid_df.dropna(subset=["resid_diff", "resid_diff_lag"])

        if len(resid_df) == 0:
            raise ValueError("No valid observations after differencing")

        # Regression: Δe_it on Δe_{i,t-1}
        y = resid_df["resid_diff"].values
        X = resid_df["resid_diff_lag"].values

        # OLS regression
        n = len(y)
        beta = np.sum(X * y) / np.sum(X * X)

        # Residuals
        fitted = beta * X
        resid_reg = y - fitted

        # Standard error of beta
        s2 = np.sum(resid_reg**2) / (n - 1)
        se_beta = np.sqrt(s2 / np.sum(X * X))

        # Test H0: beta = -0.5 (no serial correlation)
        # Under H0, if no autocorrelation, E[Δe_it * Δe_{i,t-1}] = -sigma²/2
        # So coefficient should be -0.5
        t_stat = (beta - (-0.5)) / se_beta

        # F statistic (F = t²)
        f_stat = t_stat**2

        # P-value from F distribution
        # Number of entities
        n_entities = resid_df["entity"].nunique()
        df_num = 1
        df_denom = n_entities - 1

        pvalue = 1 - stats.f.cdf(f_stat, df_num, df_denom)

        # Metadata
        metadata = {
            "coefficient": beta,
            "std_error": se_beta,
            "t_statistic": t_stat,
            "n_entities": n_entities,
            "n_obs_used": n,
        }

        result = ValidationTestResult(
            test_name="Wooldridge Test for Autocorrelation",
            statistic=f_stat,
            pvalue=pvalue,
            null_hypothesis="No first-order autocorrelation",
            alternative_hypothesis="First-order autocorrelation present",
            alpha=alpha,
            df=(df_num, df_denom),
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
        # Try to get entity and time from model metadata
        # This assumes the model stored the original data structure

        # For now, we'll try to extract from the results object
        # This requires that the model kept track of entity/time indices

        # If results has entity_index and time_index attributes
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
            # Fallback: try to reconstruct from model's data attribute
            # This assumes the results object has reference to the original model
            # which has the PanelData object

            # For now, raise informative error
            raise AttributeError(
                "Results object must have 'entity_index' and 'time_index' attributes. "
                "Please ensure your model stores these during estimation."
            )

        return resid_df
