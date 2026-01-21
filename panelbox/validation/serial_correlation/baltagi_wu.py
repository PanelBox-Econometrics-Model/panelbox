"""
Baltagi-Wu LBI test for serial correlation in panel data.

LBI = Locally Best Invariant

References
----------
Baltagi, B. H., & Wu, P. X. (1999). Unequally Spaced Panel Data Regressions
with AR(1) Disturbances. Econometric Theory, 15(6), 814-823.

Baltagi, B. H., & Li, Q. (1995). Testing AR(1) Against MA(1) Disturbances
in an Error Component Model. Journal of Econometrics, 68(1), 133-151.
"""

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class BaltagiWuTest(ValidationTest):
    """
    Baltagi-Wu LBI test for first-order serial correlation in panel data.

    This test is designed for unbalanced panels and tests for AR(1)
    serial correlation in the idiosyncratic errors.

    H0: No first-order serial correlation (rho = 0)
    H1: AR(1) serial correlation present (rho ≠ 0)

    The test is based on a modified Durbin-Watson statistic that
    accounts for unbalanced panel structure.

    Notes
    -----
    The test statistic is:

    LBI = sum_i sum_t (e_it - e_{i,t-1})² / sum_i sum_t e_it²

    Under H0, LBI ≈ 2 (similar to Durbin-Watson).
    LBI < 2 suggests positive autocorrelation.
    LBI > 2 suggests negative autocorrelation.

    Unlike the standard Durbin-Watson test, the Baltagi-Wu test:
    - Works with unbalanced panels
    - Accounts for heterogeneous time series lengths
    - Provides asymptotic normal distribution under H0

    Examples
    --------
    >>> from panelbox.models.static.fixed_effects import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> from panelbox.validation.serial_correlation.baltagi_wu import BaltagiWuTest
    >>> test = BaltagiWuTest(results)
    >>> result = test.run()
    >>> print(result)
    """

    def __init__(self, results: 'PanelResults'):
        """
        Initialize Baltagi-Wu test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)

    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Run Baltagi-Wu test for serial correlation.

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
            If panel has fewer than 2 time periods per entity

        Notes
        -----
        The test uses asymptotic normality of the LBI statistic.
        For large N and T, (LBI - 2) is approximately normally distributed.

        The transformation to a standard normal test statistic uses:
        z = (LBI - 2) / sqrt(var(LBI))

        where var(LBI) is estimated from the panel structure.
        """
        # Get residuals with entity and time structure
        resid_df = self._prepare_residual_data()

        # Sort by entity and time
        resid_df = resid_df.sort_values(['entity', 'time'])

        # Compute lagged residuals within each entity
        resid_df['resid_lag'] = resid_df.groupby('entity')['resid'].shift(1)

        # Compute differences
        resid_df['resid_diff'] = resid_df['resid'] - resid_df['resid_lag']

        # Drop missing values (first observation of each entity)
        resid_df_clean = resid_df.dropna(subset=['resid_diff', 'resid_lag'])

        if len(resid_df_clean) == 0:
            raise ValueError(
                "No valid observations after computing differences. "
                "Ensure each entity has at least 2 time periods."
            )

        # Compute LBI statistic
        # LBI = sum(diff²) / sum(resid²)
        numerator = np.sum(resid_df_clean['resid_diff'] ** 2)
        denominator = np.sum(resid_df['resid'] ** 2)  # Use all residuals

        if denominator == 0:
            raise ValueError("Sum of squared residuals is zero (perfect fit)")

        lbi_stat = numerator / denominator

        # Compute approximate variance of LBI
        # Under H0: E[LBI] ≈ 2
        # Var(LBI) ≈ 4 / (N*T_bar) where T_bar is average time periods
        n_entities = resid_df['entity'].nunique()
        n_obs_total = len(resid_df)
        t_bar = n_obs_total / n_entities

        # More refined variance estimate
        # Account for unbalanced structure
        entity_counts = resid_df.groupby('entity').size()
        t_i = entity_counts.values

        # Variance formula for unbalanced panels
        # Var(LBI) ≈ 4 * sum(1/T_i) / N
        var_lbi = 4 * np.sum(1 / t_i) / n_entities

        # Standard error
        se_lbi = np.sqrt(var_lbi)

        # Test statistic: (LBI - 2) / SE(LBI)
        # Under H0, this is approximately N(0,1)
        if se_lbi == 0:
            raise ValueError("Standard error is zero")

        z_stat = (lbi_stat - 2) / se_lbi

        # Two-sided p-value
        pvalue = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        # Metadata
        # Estimate rho from LBI: rho ≈ 1 - LBI/2
        rho_estimate = 1 - lbi_stat / 2

        metadata = {
            'lbi_statistic': float(lbi_stat),
            'z_statistic': float(z_stat),
            'rho_estimate': float(rho_estimate),
            'n_entities': int(n_entities),
            'n_obs_total': int(n_obs_total),
            'n_obs_used': len(resid_df_clean),
            'avg_time_periods': float(t_bar),
            'min_time_periods': int(t_i.min()),
            'max_time_periods': int(t_i.max()),
            'variance_lbi': float(var_lbi),
            'se_lbi': float(se_lbi),
            'interpretation': (
                'LBI < 2: positive autocorrelation, '
                'LBI ≈ 2: no autocorrelation, '
                'LBI > 2: negative autocorrelation'
            )
        }

        result = ValidationTestResult(
            test_name="Baltagi-Wu LBI Test for Serial Correlation",
            statistic=z_stat,
            pvalue=pvalue,
            null_hypothesis="No first-order serial correlation",
            alternative_hypothesis="First-order serial correlation present",
            alpha=alpha,
            df=None,  # Asymptotic test, no df
            metadata=metadata
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
        if hasattr(self.results, 'entity_index') and hasattr(self.results, 'time_index'):
            resid_flat = self.resid.ravel() if hasattr(self.resid, 'ravel') else self.resid

            resid_df = pd.DataFrame({
                'entity': self.results.entity_index,
                'time': self.results.time_index,
                'resid': resid_flat
            })

            return resid_df
        else:
            raise AttributeError(
                "Results object must have 'entity_index' and 'time_index' attributes"
            )
