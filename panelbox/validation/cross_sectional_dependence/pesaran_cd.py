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

    Notes
    -----
    The test statistic is asymptotically distributed as N(0,1) under the null
    hypothesis. The test works well for both N large, T small and N large, T large
    panels.

    The test requires T > 3 and works best when N is reasonably large.

    Examples
    --------
    >>> from panelbox.models.static.fixed_effects import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> from panelbox.validation.cross_sectional.pesaran_cd import PesaranCDTest
    >>> test = PesaranCDTest(results)
    >>> result = test.run()
    >>> print(result)
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
