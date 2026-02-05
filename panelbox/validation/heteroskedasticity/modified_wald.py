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

    H0: sigma²_1 = sigma²_2 = ... = sigma²_N
    H1: sigma²_i ≠ sigma²_j for some i ≠ j

    This test is specifically designed for fixed effects panel models and
    is robust to serial correlation.

    Notes
    -----
    The test statistic follows a chi-squared distribution with N degrees
    of freedom under the null hypothesis, where N is the number of entities.

    This test requires that each entity has the same number of time periods
    (balanced panel) or can be adapted for unbalanced panels.

    Examples
    --------
    >>> from panelbox.models.static.fixed_effects import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest
    >>> test = ModifiedWaldTest(results)
    >>> result = test.run()
    >>> print(result)
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
