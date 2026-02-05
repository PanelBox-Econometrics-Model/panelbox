"""
Wooldridge test for autocorrelation in panel data.

References
----------
Wooldridge, J. M. (2002). Econometric Analysis of Cross Section and Panel Data.
MIT Press, Section 10.4.1.

Stata command: xtserial
"""

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

    Notes
    -----
    The test statistic is approximately distributed as F(1, N-1) under the null
    of no first-order serial correlation.

    This test requires at least T >= 3 time periods.

    Examples
    --------
    >>> from panelbox.models.static.fixed_effects import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> from panelbox.validation.serial_correlation.wooldridge_ar import WooldridgeARTest
    >>> test = WooldridgeARTest(results)
    >>> result = test.run()
    >>> print(result)
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

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
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
