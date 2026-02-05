"""
GMM Specification Tests
========================

Specification tests for GMM models including Hansen J-test, Sargan test,
Arellano-Bond autocorrelation tests, and Difference-in-Hansen test.

Classes
-------
GMMTests : Specification tests for GMM models

References
----------
.. [1] Hansen, L. P. (1982). "Large Sample Properties of Generalized Method
       of Moments Estimators." Econometrica, 50(4), 1029-1054.

.. [2] Sargan, J. D. (1958). "The Estimation of Economic Relationships using
       Instrumental Variables." Econometrica, 26(3), 393-415.

.. [3] Arellano, M., & Bond, S. (1991). "Some Tests of Specification for Panel
       Data: Monte Carlo Evidence and an Application to Employment Equations."
       Review of Economic Studies, 58(2), 277-297.
"""

from typing import Tuple, Union

import numpy as np
from scipy import stats

from panelbox.gmm.results import TestResult


class GMMTests:
    """
    Specification tests for GMM models.

    Implements:
    - Hansen J-test of overidentifying restrictions
    - Sargan test (non-robust version)
    - Arellano-Bond AR(1) and AR(2) tests
    - Difference-in-Hansen test for instrument subsets

    Examples
    --------
    >>> tester = GMMTests()
    >>> hansen = tester.hansen_j_test(residuals, Z, W, n_params)
    >>> ar2 = tester.arellano_bond_ar_test(residuals_diff, order=2)
    """

    def hansen_j_test(
        self, residuals: np.ndarray, Z: np.ndarray, W: np.ndarray, n_params: int
    ) -> TestResult:
        """
        Hansen (1982) J-test of overidentifying restrictions.

        Tests the validity of the moment conditions (instrument validity).
        Under the null hypothesis that instruments are valid, the J-statistic
        follows a chi-square distribution.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals (n x 1)
        Z : np.ndarray
            Instrument matrix (n x n_instruments)
        W : np.ndarray
            GMM weight matrix (n_instruments x n_instruments)
        n_params : int
            Number of parameters estimated

        Returns
        -------
        TestResult
            Hansen J-test result

        Notes
        -----
        H0: All instruments are valid (E[Z'ε] = 0)
        Test statistic: J = n * (Z'ε)' W (Z'ε) ~ χ²(n_instruments - n_params)

        Interpretation:
        - p-value < 0.10: Reject H0 (instruments invalid)
        - p-value > 0.25: Suspicious (may indicate weak instruments)
        - 0.10 < p-value < 0.25: Good (instruments valid)

        References
        ----------
        Hansen, L. P. (1982). Econometrica, 50(4), 1029-1054.
        """
        # Remove missing values
        residuals = residuals.flatten() if residuals.ndim > 1 else residuals
        valid_mask = ~np.isnan(residuals)
        resid_clean = residuals[valid_mask]
        Z_clean = Z[valid_mask, :]

        n = len(resid_clean)
        n_instruments = Z_clean.shape[1]

        # Compute moment conditions: g_n = (1/n) Z'ε
        g_n = (Z_clean.T @ resid_clean) / n

        # Compute J statistic: J = n * g_n' W g_n
        J_stat = n * (g_n.T @ W @ g_n)

        # Degrees of freedom
        df = n_instruments - n_params

        if df <= 0:
            # Exactly identified or under-identified
            return TestResult(
                name="Hansen J-test",
                statistic=np.nan,
                pvalue=np.nan,
                df=df,
                distribution="chi2",
                null_hypothesis="All instruments are valid",
                conclusion="N/A (exactly/under-identified)",
                details={"message": "Test not applicable: model is exactly or under-identified"},
            )

        # P-value from chi-square distribution
        pvalue = 1 - stats.chi2.cdf(J_stat, df)

        return TestResult(
            name="Hansen J-test",
            statistic=J_stat,
            pvalue=pvalue,
            df=df,
            distribution="chi2",
            null_hypothesis="All instruments are valid (overid restrictions hold)",
            details={
                "n_instruments": n_instruments,
                "n_params": n_params,
                "overid_restrictions": df,
            },
        )

    def sargan_test(self, residuals: np.ndarray, Z: np.ndarray, n_params: int) -> TestResult:
        """
        Sargan (1958) test of overidentifying restrictions.

        Non-robust version of Hansen J-test. Only valid under homoskedasticity,
        but reported for compatibility with Stata xtabond2.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals (n x 1)
        Z : np.ndarray
            Instrument matrix (n x n_instruments)
        n_params : int
            Number of parameters estimated

        Returns
        -------
        TestResult
            Sargan test result

        Notes
        -----
        H0: All instruments are valid (under homoskedasticity)
        Test statistic: S = (Z'ε)' (Z'Z)^{-1} (Z'ε) ~ χ²(n_instruments - n_params)

        This is equivalent to Hansen J-test with W = (Z'Z)^{-1}.
        Not robust to heteroskedasticity - use Hansen J-test for robustness.

        References
        ----------
        Sargan, J. D. (1958). Econometrica, 26(3), 393-415.
        """
        # Remove missing values
        residuals = residuals.flatten() if residuals.ndim > 1 else residuals
        valid_mask = ~np.isnan(residuals)
        resid_clean = residuals[valid_mask]
        Z_clean = Z[valid_mask, :]

        len(resid_clean)
        n_instruments = Z_clean.shape[1]

        # Degrees of freedom
        df = n_instruments - n_params

        if df <= 0:
            return TestResult(
                name="Sargan test",
                statistic=np.nan,
                pvalue=np.nan,
                df=df,
                distribution="chi2",
                null_hypothesis="All instruments are valid (homoskedasticity)",
                conclusion="N/A (exactly/under-identified)",
                details={"message": "Test not applicable: model is exactly or under-identified"},
            )

        # Compute Sargan statistic
        # S = (Z'ε)' (Z'Z)^{-1} (Z'ε)
        Zte = Z_clean.T @ resid_clean
        ZtZ = Z_clean.T @ Z_clean

        try:
            ZtZ_inv = np.linalg.inv(ZtZ)
        except np.linalg.LinAlgError:
            ZtZ_inv = np.linalg.pinv(ZtZ)

        S_stat = Zte.T @ ZtZ_inv @ Zte

        # P-value from chi-square distribution
        pvalue = 1 - stats.chi2.cdf(S_stat, df)

        return TestResult(
            name="Sargan test",
            statistic=S_stat,
            pvalue=pvalue,
            df=df,
            distribution="chi2",
            null_hypothesis="All instruments valid (assumes homoskedasticity)",
            details={
                "n_instruments": n_instruments,
                "n_params": n_params,
                "note": "Not robust to heteroskedasticity. Use Hansen J-test for robustness.",
            },
        )

    def arellano_bond_ar_test(
        self, residuals_diff: np.ndarray, ids: np.ndarray, order: int = 1
    ) -> TestResult:
        """
        Arellano-Bond (1991) test for autocorrelation in residuals.

        Tests for serial correlation in the differenced residuals.
        Critical for validating moment conditions in GMM.

        Parameters
        ----------
        residuals_diff : np.ndarray
            First-differenced residuals (Δε_it)
        ids : np.ndarray
            Cross-sectional unit identifiers
        order : int
            Order of autocorrelation to test (1 or 2)

        Returns
        -------
        TestResult
            AR test result

        Notes
        -----
        H0: No autocorrelation of order `order` in differenced residuals

        AR(1) test:
        - Expected to REJECT (first-differencing induces MA(1))
        - Reported for completeness, not a failure if rejected

        AR(2) test:
        - Should NOT reject (critical test)
        - If AR(2) is rejected, moment conditions are invalid
        - This invalidates lagged levels as instruments

        Test statistic is asymptotically N(0,1).

        References
        ----------
        Arellano, M., & Bond, S. (1991). Review of Economic Studies, 58(2), 277-297.
        """
        # Remove missing values
        valid_mask = ~np.isnan(residuals_diff)
        resid_clean = residuals_diff[valid_mask]
        ids_clean = ids[valid_mask]

        # Compute lagged residuals by group
        unique_ids = np.unique(ids_clean)
        len(unique_ids)

        # Store products of residuals and their lags
        products: list[float] = []

        for group_id in unique_ids:
            # Get residuals for this group
            mask = ids_clean == group_id
            group_resid = resid_clean[mask]

            # Compute products: Δε_it * Δε_{i,t-order}
            for t in range(order, len(group_resid)):
                product = group_resid[t] * group_resid[t - order]
                products.append(product)

        if len(products) == 0:
            return TestResult(
                name=f"AR({order}) test",
                statistic=np.nan,
                pvalue=np.nan,
                df=None,
                distribution="normal",
                null_hypothesis=f"No AR({order}) in differenced residuals",
                conclusion="N/A (insufficient data)",
                details={"message": "Insufficient observations for AR test"},
            )

        products = np.array(products)

        # Compute test statistic
        # Under H0, E[Δε_it * Δε_{i,t-k}] = 0
        mean_product = np.mean(products)
        var_product = np.var(products, ddof=1)

        if var_product == 0:
            return TestResult(
                name=f"AR({order}) test",
                statistic=np.nan,
                pvalue=np.nan,
                df=None,
                distribution="normal",
                null_hypothesis=f"No AR({order}) in differenced residuals",
                conclusion="N/A (zero variance)",
                details={"message": "Zero variance in products"},
            )

        # Normalize by standard error
        se_product = np.sqrt(var_product / len(products))
        z_stat = mean_product / se_product

        # P-value from standard normal (two-sided test)
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        # Determine null hypothesis and expected result
        if order == 1:
            null_hyp = "No AR(1) in differenced residuals"
            details = {
                "note": "AR(1) rejection is EXPECTED due to MA(1) induced by differencing",
                "n_products": len(products),
            }
        else:
            null_hyp = f"No AR({order}) in differenced residuals"
            details = {
                "note": f"AR({order}) rejection indicates INVALID moment conditions",
                "n_products": len(products),
            }

        return TestResult(
            name=f"AR({order}) test",
            statistic=z_stat,
            pvalue=pvalue,
            df=None,
            distribution="normal",
            null_hypothesis=null_hyp,
            details=details,
        )

    def difference_in_hansen(
        self,
        residuals: np.ndarray,
        Z_full: np.ndarray,
        Z_subset: np.ndarray,
        W_full: np.ndarray,
        W_subset: np.ndarray,
        n_params: int,
        subset_name: str = "subset",
    ) -> TestResult:
        """
        Difference-in-Hansen test for instrument subsets.

        Tests the validity of a specific subset of instruments by comparing
        Hansen J statistics with and without the subset.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals (n x 1)
        Z_full : np.ndarray
            Full instrument matrix (n x n_instruments_full)
        Z_subset : np.ndarray
            Subset instrument matrix to test (n x n_instruments_subset)
        W_full : np.ndarray
            Weight matrix for full model
        W_subset : np.ndarray
            Weight matrix for model without subset
        n_params : int
            Number of parameters
        subset_name : str
            Name of instrument subset being tested

        Returns
        -------
        TestResult
            Difference-in-Hansen test result

        Notes
        -----
        H0: Subset of instruments is valid

        Test statistic: D = J_subset - J_full ~ χ²(df_subset - df_full)

        Common uses in System GMM:
        - Test validity of level instruments
        - Test validity of specific GMM-type instruments
        - Compare collapsed vs. uncollapsed instruments

        References
        ----------
        Roodman, D. (2009). Stata Journal, 9(1), 86-136.
        """
        # Remove missing values
        residuals = residuals.flatten() if residuals.ndim > 1 else residuals
        valid_mask = ~np.isnan(residuals)
        resid_clean = residuals[valid_mask]
        Z_full_clean = Z_full[valid_mask, :]  # 2D indexing
        Z_subset_clean = Z_subset[valid_mask, :]  # 2D indexing

        n = len(resid_clean)

        # Compute Hansen J for full model
        n_instruments_full = Z_full_clean.shape[1]
        df_full = n_instruments_full - n_params

        g_n_full = (Z_full_clean.T @ resid_clean) / n
        J_full = n * (g_n_full.T @ W_full @ g_n_full)

        # Compute Hansen J for model without subset
        n_instruments_subset = Z_subset_clean.shape[1]
        df_subset = n_instruments_subset - n_params

        g_n_subset = (Z_subset_clean.T @ resid_clean) / n
        J_subset = n * (g_n_subset.T @ W_subset @ g_n_subset)

        # Difference-in-Hansen statistic
        D_stat = J_subset - J_full
        df_diff = df_subset - df_full

        if df_diff <= 0:
            return TestResult(
                name=f"Diff-in-Hansen ({subset_name})",
                statistic=np.nan,
                pvalue=np.nan,
                df=df_diff,
                distribution="chi2",
                null_hypothesis=f"{subset_name} instruments are valid",
                conclusion="N/A (invalid df)",
                details={"message": "Invalid degrees of freedom for difference test"},
            )

        # P-value from chi-square distribution
        pvalue = 1 - stats.chi2.cdf(D_stat, df_diff)

        return TestResult(
            name=f"Diff-in-Hansen ({subset_name})",
            statistic=D_stat,
            pvalue=pvalue,
            df=df_diff,
            distribution="chi2",
            null_hypothesis=f"{subset_name} instruments are valid",
            details={"J_full": J_full, "J_subset": J_subset, "n_instruments_tested": df_diff},
        )

    def weak_instruments_test(self, X: np.ndarray, Z: np.ndarray) -> Tuple[float, bool]:
        """
        Simple weak instruments diagnostic.

        Computes the F-statistic from the first-stage regression of each
        endogenous variable on the instruments.

        Parameters
        ----------
        X : np.ndarray
            Endogenous regressors (n x k)
        Z : np.ndarray
            Instruments (n x n_instruments)

        Returns
        -------
        f_stat : float
            Minimum F-statistic across first-stage regressions
        weak : bool
            True if instruments may be weak (F < 10)

        Notes
        -----
        Rule of thumb: F-statistic > 10 suggests instruments are not weak.

        For more sophisticated weak instruments tests, see:
        - Stock & Yogo (2005) critical values
        - Montiel Olea & Pflueger (2013) effective F-statistic
        """
        # Remove missing values
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(Z).any(axis=1)
        X_clean = X[valid_mask]
        Z_clean = Z[valid_mask]

        n, k = X_clean.shape
        n_instruments = Z_clean.shape[1]

        # Compute F-statistics for each endogenous variable
        f_stats = []

        for j in range(k):
            x_j = X_clean[:, j]

            # First-stage regression: x_j = Z * π + u
            ZtZ = Z_clean.T @ Z_clean
            Ztx = Z_clean.T @ x_j

            try:
                ZtZ_inv = np.linalg.inv(ZtZ)
            except np.linalg.LinAlgError:
                ZtZ_inv = np.linalg.pinv(ZtZ)

            pi = ZtZ_inv @ Ztx
            x_j_fitted = Z_clean @ pi
            resid = x_j - x_j_fitted

            # Compute F-statistic
            # F = (R² / k) / ((1 - R²) / (n - k - 1))
            ss_total = np.sum((x_j - np.mean(x_j)) ** 2)
            ss_resid = np.sum(resid**2)
            r_squared = 1 - (ss_resid / ss_total)

            if r_squared >= 1.0:
                f_stat = np.inf
            else:
                f_stat = (r_squared / n_instruments) / ((1 - r_squared) / (n - n_instruments - 1))

            f_stats.append(f_stat)

        min_f_stat = np.min(f_stats)
        weak = min_f_stat < 10.0

        return min_f_stat, weak
