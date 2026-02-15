"""
Diagnostic measures for quantile regression.

This module implements various diagnostic measures for quantile regression
models, including:
- Pseudo R-squared
- Goodness of fit tests
- Symmetry tests
- Residual analysis
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from panelbox.models.quantile import QuantileRegressionModel


class QuantileRegressionDiagnostics:
    """
    Diagnostic measures for quantile regression models.

    Parameters
    ----------
    model : QuantileRegressionModel
        Fitted quantile regression model
    params : ndarray
        Model parameters
    tau : float, default 0.5
        Quantile level

    Attributes
    ----------
    residuals : ndarray
        Model residuals
    tau : float
        Quantile level
    """

    def __init__(self, model: QuantileRegressionModel, params: np.ndarray, tau: float = 0.5):
        """Initialize diagnostics object."""
        self.model = model
        self.tau = tau
        self.params = params

        # Compute residuals
        self.residuals = model.endog - model.exog @ params

    def pseudo_r2(self) -> float:
        """
        Compute pseudo R² for quantile regression.

        The pseudo R² (or R¹) for quantile regression is based on
        the comparison of check loss with and without covariates:

            R¹ = 1 - (Check Loss with X) / (Check Loss without X)

        Returns
        -------
        float
            Pseudo R² value in [0, 1]

        Notes
        -----
        Values close to 0 indicate poor fit, values close to 1 indicate
        good fit. However, pseudo R² values are typically much lower
        than OLS R² and should not be directly compared.
        """
        # Check loss with covariates
        loss_with_x = np.sum((self.tau - (self.residuals < 0)) * self.residuals)

        # Check loss without covariates (intercept only)
        intercept_residuals = self.model.endog - np.median(self.model.endog)
        loss_without_x = np.sum((self.tau - (intercept_residuals < 0)) * intercept_residuals)

        # Pseudo R²
        if loss_without_x == 0:
            return 1.0 if loss_with_x == 0 else 0.0

        r2 = 1 - (loss_with_x / loss_without_x)

        return np.clip(r2, 0, 1)

    def goodness_of_fit(self, n_bins: int = 10) -> Dict[str, float]:
        """
        Compute goodness of fit statistics.

        Uses several diagnostics to assess overall model fit.

        Parameters
        ----------
        n_bins : int, default 10
            Number of bins for distribution tests

        Returns
        -------
        dict
            Dictionary with keys:
            - 'pseudo_r2': Pseudo R² statistic
            - 'mean_residual': Mean of residuals
            - 'median_residual': Median of residuals
            - 'quantile_count': Count of observations at quantile
            - 'sparsity': Estimated sparsity (density at quantile)
        """
        diagnostics = {}

        # Pseudo R²
        diagnostics["pseudo_r2"] = self.pseudo_r2()

        # Residual statistics
        diagnostics["mean_residual"] = np.mean(self.residuals)
        diagnostics["median_residual"] = np.median(self.residuals)

        # Count observations at/near the quantile
        # (residuals should be close to 0 for a fraction tau)
        near_zero = np.sum(np.abs(self.residuals) < np.std(self.residuals) * 0.1)
        diagnostics["quantile_count"] = near_zero / len(self.residuals)

        # Sparsity estimate
        diagnostics["sparsity"] = self._estimate_sparsity()

        return diagnostics

    def symmetry_test(self) -> Tuple[float, float]:
        """
        Test for symmetry of residuals around the quantile.

        For quantile τ, residuals should have the property that
        P(residual < 0) = τ and P(residual ≥ 0) = 1 - τ.

        Returns
        -------
        test_stat : float
            Test statistic
        pvalue : float
            P-value for the test

        Notes
        -----
        Uses a simple sign test comparing the fraction of negative
        residuals to the quantile level.
        """
        # Fraction of negative residuals
        frac_neg = np.mean(self.residuals < 0)

        # Expected fraction
        expected = self.tau

        # Test statistic: standardized difference
        # Using normal approximation for binomial
        n = len(self.residuals)
        se = np.sqrt(expected * (1 - expected) / n)

        if se > 0:
            z_stat = (frac_neg - expected) / se
        else:
            z_stat = 0

        # Two-tailed p-value
        pvalue = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

        return z_stat, pvalue

    def goodness_of_fit_test(self, n_bins: int = 10) -> Tuple[float, float]:
        """
        Perform chi-square goodness of fit test.

        Tests whether residuals follow the expected distribution
        under the null hypothesis of correct model specification.

        Parameters
        ----------
        n_bins : int, default 10
            Number of bins for the test

        Returns
        -------
        test_stat : float
            Chi-square test statistic
        pvalue : float
            P-value

        Notes
        -----
        Under correct specification, residuals should have quantile τ
        at position 0.
        """
        # Compute quantile positions
        quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]

        # Compute empirical quantiles from residuals
        emp_quantiles = np.quantile(self.residuals, quantiles)

        # Expected quantiles under standard normal (approximately)
        expected = stats.norm.ppf(quantiles)

        # Chi-square statistic (simplified)
        chi2_stat = np.sum((emp_quantiles - expected) ** 2 / (expected**2 + 1e-8))

        # P-value (df = n_bins - 1 - n_params)
        df = max(n_bins - 1 - self.model.n_params, 1)
        pvalue = 1 - stats.chi2.cdf(chi2_stat, df)

        return chi2_stat, pvalue

    def _estimate_sparsity(self) -> float:
        """
        Estimate sparsity parameter (density at the quantile).

        Returns
        -------
        float
            Estimated density at the quantile
        """
        n = len(self.residuals)

        # Silverman's rule of thumb for bandwidth
        h = 0.9 * np.std(self.residuals) * n ** (-0.2)

        # Count residuals near zero (within ±h)
        count = np.sum(np.abs(self.residuals) <= h)

        # Estimate density at zero
        if count > 0:
            f_hat = count / (n * 2 * h)
        else:
            # Fallback to normal approximation
            f_hat = stats.norm.pdf(0) / (np.std(self.residuals) + 1e-8)

        return f_hat

    def residual_quantiles(
        self, quantiles: np.ndarray = np.array([0.25, 0.5, 0.75])
    ) -> Dict[float, float]:
        """
        Compute quantiles of residuals.

        Parameters
        ----------
        quantiles : ndarray, default [0.25, 0.5, 0.75]
            Quantiles to compute

        Returns
        -------
        dict
            Dictionary with quantile levels as keys and values as values
        """
        result = {}
        for q in quantiles:
            result[q] = np.quantile(self.residuals, q)
        return result

    def summary(self) -> str:
        """
        Generate summary of diagnostics.

        Returns
        -------
        str
            Formatted summary string
        """
        gof = self.goodness_of_fit()
        sym_stat, sym_pval = self.symmetry_test()
        gof_stat, gof_pval = self.goodness_of_fit_test()

        summary = "\n" + "=" * 60 + "\n"
        summary += f"Quantile Regression Diagnostics (τ={self.tau})\n"
        summary += "=" * 60 + "\n\n"

        summary += f"Pseudo R²:                {gof['pseudo_r2']:.4f}\n"
        summary += f"Sparsity estimate:       {gof['sparsity']:.4f}\n"
        summary += f"Quantile count:          {gof['quantile_count']:.4f}\n\n"

        summary += "Residual Statistics:\n"
        summary += "-" * 60 + "\n"
        summary += f"Mean:                    {gof['mean_residual']:.4f}\n"
        summary += f"Median:                  {gof['median_residual']:.4f}\n"
        summary += f"Std Dev:                 {np.std(self.residuals):.4f}\n\n"

        summary += "Specification Tests:\n"
        summary += "-" * 60 + "\n"
        summary += f"Symmetry Test:           {sym_stat:.4f} (p={sym_pval:.4f})\n"
        summary += f"Goodness of Fit Test:    {gof_stat:.4f} (p={gof_pval:.4f})\n"
        summary += "=" * 60 + "\n"

        return summary
