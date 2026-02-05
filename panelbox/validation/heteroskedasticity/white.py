"""
White test for heteroskedasticity in panel data.

References
----------
White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator
and a direct test for heteroskedasticity. Econometrica, 48(4), 817-838.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class WhiteTest(ValidationTest):
    """
    White test for heteroskedasticity.

    Tests the null hypothesis of homoskedasticity against a general
    alternative of heteroskedasticity. Unlike Breusch-Pagan, this test
    does not assume a specific functional form for the heteroskedasticity.

    H0: sigma²_i = sigma² (homoskedasticity)
    H1: sigma²_i depends on regressors in a general way

    The test regresses squared residuals on the original regressors,
    their squares, and cross-products (interactions).

    Notes
    -----
    The test statistic is n*R² from the auxiliary regression, which
    follows a chi-squared distribution under the null hypothesis.

    For models with many regressors, the test can have low power due
    to the large number of terms in the auxiliary regression.

    Examples
    --------
    >>> from panelbox.models.static.pooled_ols import PooledOLS
    >>> model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
    >>> results = model.fit()
    >>>
    >>> from panelbox.validation.heteroskedasticity.white import WhiteTest
    >>> test = WhiteTest(results)
    >>> result = test.run()
    >>> print(result)
    """

    def __init__(self, results: "PanelResults"):
        """
        Initialize White test.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)
        self._X = None
        if hasattr(results, "_model"):
            if hasattr(results._model, "_X_orig"):
                self._X = results._model._X_orig

    def run(self, alpha: float = 0.05, cross_terms: bool = True, **kwargs) -> ValidationTestResult:
        """
        Run White test for heteroskedasticity.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        cross_terms : bool, default=True
            If True, include cross-product terms in auxiliary regression
            If False, only include squared terms (less computationally intensive)

        Returns
        -------
        ValidationTestResult
            Test results

        Raises
        ------
        ValueError
            If design matrix is not available
        """
        # Get squared residuals
        resid_sq = self.resid**2

        # Get design matrix
        X = self._get_design_matrix()

        if X is None:
            raise ValueError(
                "Design matrix not available. "
                "White test requires access to the original regressors."
            )

        n = len(resid_sq)
        k_orig = X.shape[1]

        # Build augmented design matrix with squares and cross-products
        # Remove constant column if present (first column all 1s)
        if np.allclose(X[:, 0], 1.0):
            X_vars = X[:, 1:]  # Exclude constant
            has_constant = True
        else:
            X_vars = X
            has_constant = False

        k_vars = X_vars.shape[1]

        # Create list of columns for auxiliary regression
        aux_cols = [np.ones(n)] if has_constant or not has_constant else [np.ones(n)]

        # Add original variables
        for j in range(k_vars):
            aux_cols.append(X_vars[:, j])

        # Add squared terms
        for j in range(k_vars):
            aux_cols.append(X_vars[:, j] ** 2)

        # Add cross-product terms if requested
        if cross_terms and k_vars > 1:
            for i, j in combinations_with_replacement(range(k_vars), 2):
                if i < j:  # Only upper triangle (avoid duplicates)
                    aux_cols.append(X_vars[:, i] * X_vars[:, j])

        # Stack into matrix
        X_aux = np.column_stack(aux_cols)

        # Auxiliary regression: resid² on X_aux
        try:
            XtX = X_aux.T @ X_aux
            Xty = X_aux.T @ resid_sq
            beta_aux = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            beta_aux = np.linalg.lstsq(X_aux, resid_sq, rcond=None)[0]

        # Fitted values
        fitted_aux = X_aux @ beta_aux

        # R²
        mean_resid_sq = np.mean(resid_sq)
        SST = np.sum((resid_sq - mean_resid_sq) ** 2)
        SSR = np.sum((resid_sq - fitted_aux) ** 2)

        if SST > 0:
            R2_aux = 1 - SSR / SST
        else:
            R2_aux = 0.0

        # LM statistic
        lm_stat = n * R2_aux

        # Degrees of freedom = number of auxiliary regressors - 1 (for constant)
        df = X_aux.shape[1] - 1

        if df <= 0:
            df = 1

        # P-value
        pvalue = 1 - stats.chi2.cdf(lm_stat, df)

        # Metadata
        metadata = {
            "R2_auxiliary": R2_aux,
            "n_obs": n,
            "n_original_regressors": k_orig,
            "n_auxiliary_terms": X_aux.shape[1],
            "includes_cross_terms": cross_terms,
        }

        result = ValidationTestResult(
            test_name="White Test for Heteroskedasticity",
            statistic=lm_stat,
            pvalue=pvalue,
            null_hypothesis="Homoskedasticity (constant error variance)",
            alternative_hypothesis="General heteroskedasticity",
            alpha=alpha,
            df=df,
            metadata=metadata,
        )

        return result

    def _get_design_matrix(self) -> np.ndarray:
        """Get the design matrix X."""
        if self._X is not None:
            return self._X

        if hasattr(self.results, "_model"):
            model = self.results._model

            if hasattr(model, "formula_parser") and hasattr(model, "data"):
                try:
                    _, X = model.formula_parser.build_design_matrices(
                        model.data.data, return_type="array"
                    )
                    return X
                except Exception:
                    pass

        return None
