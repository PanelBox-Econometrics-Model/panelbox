"""
Breusch-Pagan LM test for cross-sectional dependence in panel data.

References
----------
Breusch, T. S., & Pagan, A. R. (1980). The Lagrange Multiplier Test and its
Applications to Model Specification in Econometrics. The Review of Economic
Studies, 47(1), 239-253.

Pesaran, M. H. (2004). General Diagnostic Tests for Cross Section Dependence
in Panels. Cambridge Working Papers in Economics No. 0435.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class BreuschPaganLMTest(ValidationTest):
    """
    Breusch-Pagan LM test for cross-sectional dependence.

    Tests the null hypothesis that residuals are cross-sectionally
    independent (no contemporaneous correlation across entities).

    H0: Corr(e_it, e_jt) = 0 for all i ≠ j
    H1: Some Corr(e_it, e_jt) ≠ 0

    The test is based on the sum of squared pairwise correlation
    coefficients of residuals.

    Notes
    -----
    The test statistic is:

    LM = T * sum_{i<j} rho_ij²

    where rho_ij is the sample correlation between residuals of
    entity i and entity j, and the sum is over all N(N-1)/2 pairs.

    Under H0, LM ~ Chi2(N(N-1)/2)

    This test is appropriate for panels with:
    - Fixed T (time periods)
    - N not too large (becomes over-sized as N → ∞)
    - For large N, use Pesaran CD test instead

    The test requires a balanced panel or will use pairwise complete
    observations for each entity pair.

    Examples
    --------
    >>> from panelbox.models.static.pooled_ols import PooledOLS
    >>> model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
    >>> results = model.fit()
    >>>
    >>> from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import BreuschPaganLMTest
    >>> test = BreuschPaganLMTest(results)
    >>> result = test.run()
    >>> print(result)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize Breusch-Pagan LM test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)

    def run(self, alpha: float = 0.05, **kwargs) -> ValidationTestResult:
        """
        Run Breusch-Pagan LM test for cross-sectional dependence.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level

        Returns
        -------
        ValidationTestResult
            Test results

        Warnings
        --------
        This test can be over-sized (reject H0 too often) when N is large.
        For large N (> 30), consider using the Pesaran CD test instead.

        Notes
        -----
        The test requires computing N(N-1)/2 pairwise correlations.
        For large N, this can be computationally intensive.
        """
        # Get residuals with entity and time structure
        resid_df = self._prepare_residual_data()

        # Create wide format: rows = time, columns = entities
        resid_wide = resid_df.pivot(index="time", columns="entity", values="resid")

        # Get dimensions
        T = resid_wide.shape[0]  # Number of time periods
        N = resid_wide.shape[1]  # Number of entities

        if N < 2:
            raise ValueError("Need at least 2 entities for cross-sectional dependence test")

        # Compute pairwise correlations
        # Use pairwise complete observations
        correlations: Union[list[float], np.ndarray] = []
        n_pairs = 0

        entity_list = list(resid_wide.columns)

        for i in range(N):
            for j in range(i + 1, N):
                entity_i = entity_list[i]
                entity_j = entity_list[j]

                # Get residuals for this pair (drop NaN)
                resid_i = resid_wide[entity_i].dropna()
                resid_j = resid_wide[entity_j].dropna()

                # Find common time periods
                common_times = resid_i.index.intersection(resid_j.index)

                if len(common_times) >= 3:  # Need at least 3 obs for correlation
                    resid_i_common = resid_i.loc[common_times]
                    resid_j_common = resid_j.loc[common_times]

                    # Compute correlation
                    rho_ij = np.corrcoef(resid_i_common, resid_j_common)[0, 1]

                    # Handle potential NaN from constant series
                    if not np.isnan(rho_ij):
                        correlations.append(rho_ij)
                        n_pairs += 1

        if n_pairs == 0:
            raise ValueError(
                "No valid pairwise correlations could be computed. "
                "Check for constant residuals or insufficient data."
            )

        # Compute LM statistic
        # LM = T * sum(rho_ij²)
        correlations = np.array(correlations)
        lm_stat = T * np.sum(correlations**2)

        # Degrees of freedom = number of pairs
        # For complete data: N(N-1)/2
        # For incomplete data: actual number of pairs
        df = n_pairs

        # P-value from chi-squared distribution
        pvalue = 1 - stats.chi2.cdf(lm_stat, df)

        # Metadata
        mean_abs_corr = np.mean(np.abs(correlations))
        max_abs_corr = np.max(np.abs(correlations))
        positive_corrs = np.sum(correlations > 0)
        negative_corrs = np.sum(correlations < 0)

        metadata = {
            "n_entities": int(N),
            "n_time_periods": int(T),
            "n_pairs": int(n_pairs),
            "n_pairs_expected": int(N * (N - 1) // 2),
            "mean_abs_correlation": float(mean_abs_corr),
            "max_abs_correlation": float(max_abs_corr),
            "n_positive_correlations": int(positive_corrs),
            "n_negative_correlations": int(negative_corrs),
            "warning": (
                "Test may be over-sized for large N. " "Consider Pesaran CD test if N > 30."
                if N > 30
                else None
            ),
        }

        result = ValidationTestResult(
            test_name="Breusch-Pagan LM Test for Cross-Sectional Dependence",
            statistic=lm_stat,
            pvalue=pvalue,
            null_hypothesis="No cross-sectional dependence (residuals independent across entities)",
            alternative_hypothesis="Cross-sectional dependence present",
            alpha=alpha,
            df=df,
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
            resid_flat = self.resid.ravel() if hasattr(self.resid, "ravel") else self.resid

            resid_df = pd.DataFrame(
                {
                    "entity": self.results.entity_index,
                    "time": self.results.time_index,
                    "resid": resid_flat,
                }
            )

            return resid_df
        else:
            raise AttributeError(
                "Results object must have 'entity_index' and 'time_index' attributes"
            )
