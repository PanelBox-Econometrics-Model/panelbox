"""
Breusch-Pagan LM test for heteroskedasticity in panel data.

References
----------
Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity
and random coefficient variation. Econometrica, 47(5), 1287-1294.

Greene, W. H. (2018). Econometric Analysis (8th ed.). Pearson.
"""

import numpy as np
import pandas as pd
from scipy import stats

from panelbox.validation.base import ValidationTest, ValidationTestResult


class BreuschPaganTest(ValidationTest):
    """
    Breusch-Pagan LM test for heteroskedasticity.
    
    Tests the null hypothesis that the error variance is constant
    (homoskedasticity) against the alternative that the variance
    is a function of the regressors.
    
    H0: sigma²_i = sigma² (homoskedasticity)
    H1: sigma²_i = h(X_i) (heteroskedasticity)
    
    The test regresses squared residuals on the original regressors
    and tests if the coefficients are jointly zero using an LM statistic.
    
    Notes
    -----
    The test statistic is n*R² from the auxiliary regression, which
    follows a chi-squared distribution with k degrees of freedom
    under the null hypothesis, where k is the number of regressors
    (excluding the constant).
    
    Examples
    --------
    >>> from panelbox.models.static.pooled_ols import PooledOLS
    >>> model = PooledOLS("y ~ x1 + x2", data, "entity", "time")
    >>> results = model.fit()
    >>> 
    >>> from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest
    >>> test = BreuschPaganTest(results)
    >>> result = test.run()
    >>> print(result)
    """
    
    def __init__(self, results: 'PanelResults'):
        """
        Initialize Breusch-Pagan test.
        
        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        super().__init__(results)
        
        # Store original design matrix if available
        # We'll need the original X matrix for the auxiliary regression
        self._X = None
        if hasattr(results, '_model'):
            if hasattr(results._model, '_X_orig'):
                self._X = results._model._X_orig
    
    def run(self, alpha: float = 0.05) -> ValidationTestResult:
        """
        Run Breusch-Pagan LM test for heteroskedasticity.
        
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
            If design matrix is not available
        
        Notes
        -----
        The test procedure:
        1. Estimate the original model and obtain residuals e
        2. Compute squared residuals e²
        3. Regress e² on the original regressors X
        4. Compute LM statistic = n*R² from this auxiliary regression
        5. Compare to chi-squared(k) distribution
        """
        # Get squared residuals
        resid_sq = self.resid ** 2
        
        # We need the design matrix X
        # Try to get it from the model or reconstruct it
        X = self._get_design_matrix()
        
        if X is None:
            raise ValueError(
                "Design matrix not available. "
                "Breusch-Pagan test requires access to the original regressors."
            )
        
        n = len(resid_sq)
        
        # Auxiliary regression: resid² on X
        # OLS: beta_aux = (X'X)^{-1} X'resid²
        try:
            XtX = X.T @ X
            Xty = X.T @ resid_sq
            beta_aux = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Singular matrix, use pseudo-inverse
            beta_aux = np.linalg.lstsq(X, resid_sq, rcond=None)[0]

        # Fitted values from auxiliary regression
        fitted_aux = X @ beta_aux

        # R² from auxiliary regression using explained sum of squares
        # This is more numerically stable than 1 - SSR/SST
        mean_resid_sq = np.mean(resid_sq)
        SST = np.sum((resid_sq - mean_resid_sq) ** 2)
        SSE = np.sum((fitted_aux - mean_resid_sq) ** 2)

        # R² = SSE/SST (explained variation / total variation)
        if SST > 0:
            R2_aux = SSE / SST
        else:
            R2_aux = 0.0

        # Ensure R² is in [0, 1]
        # Due to numerical errors, R² might be slightly negative or > 1
        R2_aux = np.clip(R2_aux, 0.0, 1.0)

        # LM statistic = n * R²
        # This must be non-negative
        lm_stat = n * R2_aux

        # Sanity check: LM statistic must be >= 0
        if lm_stat < 0:
            # This should never happen, but if it does, set to 0
            lm_stat = 0.0

        # Degrees of freedom = number of regressors (excluding constant)
        k = X.shape[1]

        # Check if X has a constant column (all 1s or all same value)
        # Check first and last columns as constant might be anywhere
        has_constant = False
        for col_idx in range(k):
            col = X[:, col_idx]
            if np.allclose(col, col[0]):
                has_constant = True
                break

        if has_constant:
            df = k - 1
        else:
            df = k

        if df <= 0:
            df = 1  # At least 1 df
        
        # P-value from chi-squared distribution
        pvalue = 1 - stats.chi2.cdf(lm_stat, df)
        
        # Metadata
        metadata = {
            'R2_auxiliary': R2_aux,
            'n_obs': n,
            'n_regressors': k
        }
        
        result = ValidationTestResult(
            test_name="Breusch-Pagan LM Test for Heteroskedasticity",
            statistic=lm_stat,
            pvalue=pvalue,
            null_hypothesis="Homoskedasticity (constant error variance)",
            alternative_hypothesis="Heteroskedasticity (variance depends on regressors)",
            alpha=alpha,
            df=df,
            metadata=metadata
        )
        
        return result
    
    def _get_design_matrix(self) -> np.ndarray:
        """
        Get the design matrix X.
        
        Returns
        -------
        np.ndarray or None
            Design matrix, or None if not available
        """
        # Try to get from stored _X
        if self._X is not None:
            return self._X
        
        # Try to get from model through results
        if hasattr(self.results, '_model'):
            model = self.results._model
            
            # Try to rebuild design matrices
            if hasattr(model, 'formula_parser') and hasattr(model, 'data'):
                try:
                    _, X = model.formula_parser.build_design_matrices(
                        model.data.data,
                        return_type='array'
                    )
                    return X
                except Exception:
                    pass
        
        return None
