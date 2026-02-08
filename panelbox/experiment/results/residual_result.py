"""
Residual Result Container.

This module provides a container class for residual diagnostic analysis results,
completing the result container trilogy (Validation, Comparison, Residual).
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy import stats

from .base import BaseResult


class ResidualResult(BaseResult):
    """
    Container for residual diagnostic analysis results.

    This class wraps model residuals and provides convenient access to
    diagnostic tests, summary statistics, and visualization-ready data.

    It integrates with:
    - ResidualDataTransformer for data transformation
    - ReportManager for HTML report generation
    - Residual diagnostic charts for visualization

    Parameters
    ----------
    model_results : PanelResults
        Fitted model results object
    residuals : np.ndarray, optional
        Residuals array. If None, extracted from model_results.resid
    fitted_values : np.ndarray, optional
        Fitted values array. If None, extracted from model_results.fittedvalues
    standardized_residuals : np.ndarray, optional
        Standardized residuals. If None, computed automatically
    timestamp : datetime, optional
        Timestamp of result creation
    metadata : dict, optional
        Additional metadata

    Examples
    --------
    Create from model results:

    >>> from panelbox import FixedEffects
    >>> from panelbox.experiment.results import ResidualResult
    >>>
    >>> model = FixedEffects('y ~ x1 + x2', data, entity_id='firm', time_id='year')
    >>> results = model.fit()
    >>>
    >>> residual_result = ResidualResult.from_model_results(results)
    >>> print(residual_result.summary())

    Access diagnostic tests:

    >>> # Shapiro-Wilk test for normality
    >>> stat, pvalue = residual_result.shapiro_test
    >>> print(f"Normality test p-value: {pvalue:.4f}")
    >>>
    >>> # Durbin-Watson test for autocorrelation
    >>> dw_stat = residual_result.durbin_watson
    >>> print(f"Durbin-Watson: {dw_stat:.4f}")

    Generate HTML report:

    >>> residual_result.save_html(
    ...     'residual_diagnostics.html',
    ...     test_type='residuals',
    ...     theme='professional'
    ... )

    Export to JSON:

    >>> residual_result.save_json('residuals.json')
    """

    def __init__(
        self,
        model_results: Any,
        residuals: Optional[np.ndarray] = None,
        fitted_values: Optional[np.ndarray] = None,
        standardized_residuals: Optional[np.ndarray] = None,
        timestamp=None,
        metadata=None,
    ):
        """
        Initialize ResidualResult.

        Parameters
        ----------
        model_results : PanelResults
            Fitted model results
        residuals : np.ndarray, optional
            Residuals. If None, extracted from model_results
        fitted_values : np.ndarray, optional
            Fitted values. If None, extracted from model_results
        standardized_residuals : np.ndarray, optional
            Standardized residuals. If None, computed
        timestamp : datetime, optional
            Creation timestamp
        metadata : dict, optional
            Additional metadata
        """
        super().__init__(timestamp=timestamp, metadata=metadata)

        self.model_results = model_results
        self.residuals = residuals if residuals is not None else self._extract_residuals()
        self.fitted_values = fitted_values if fitted_values is not None else self._extract_fitted()

        # Compute standardized residuals if not provided
        if standardized_residuals is not None:
            self.standardized_residuals = standardized_residuals
        else:
            self.standardized_residuals = self._compute_standardized_residuals()

    def _extract_residuals(self) -> np.ndarray:
        """Extract residuals from model results."""
        if hasattr(self.model_results, "resid"):
            return np.asarray(self.model_results.resid)
        elif hasattr(self.model_results, "residuals"):
            return np.asarray(self.model_results.residuals)
        else:
            raise ValueError("Could not extract residuals from model_results")

    def _extract_fitted(self) -> np.ndarray:
        """Extract fitted values from model results."""
        if hasattr(self.model_results, "fittedvalues"):
            return np.asarray(self.model_results.fittedvalues)
        elif hasattr(self.model_results, "fitted_values"):
            return np.asarray(self.model_results.fitted_values)
        elif hasattr(self.model_results, "predict"):
            return np.asarray(self.model_results.predict())
        else:
            raise ValueError("Could not extract fitted values from model_results")

    def _compute_standardized_residuals(self) -> np.ndarray:
        """Compute standardized residuals."""
        # Use residual standard error if available
        if hasattr(self.model_results, "scale"):
            scale = np.sqrt(self.model_results.scale)
        elif hasattr(self.model_results, "resid_std_err"):
            scale = self.model_results.resid_std_err
        else:
            # Fallback: use sample standard deviation
            scale = np.std(self.residuals, ddof=1)

        return self.residuals / scale if scale > 0 else self.residuals

    # ==================== Diagnostic Test Properties ====================

    @property
    def shapiro_test(self) -> tuple:
        """
        Shapiro-Wilk test for normality of residuals.

        Tests the null hypothesis that the residuals are normally distributed.

        Returns
        -------
        tuple
            (statistic, pvalue)
            - statistic: Test statistic
            - pvalue: Two-tailed p-value

        Examples
        --------
        >>> stat, pvalue = residual_result.shapiro_test
        >>> if pvalue < 0.05:
        ...     print("Residuals are not normally distributed")
        ... else:
        ...     print("Residuals appear normally distributed")

        Notes
        -----
        The Shapiro-Wilk test is sensitive to sample size. For large samples,
        even small deviations from normality may be significant.
        """
        stat, pvalue = stats.shapiro(self.residuals)
        return float(stat), float(pvalue)

    @property
    def durbin_watson(self) -> float:
        """
        Durbin-Watson test statistic for autocorrelation in residuals.

        The test statistic ranges from 0 to 4, where:
        - 2 indicates no autocorrelation
        - < 2 indicates positive autocorrelation
        - > 2 indicates negative autocorrelation

        Returns
        -------
        float
            Durbin-Watson test statistic

        Examples
        --------
        >>> dw = residual_result.durbin_watson
        >>> if dw < 1.5:
        ...     print("Positive autocorrelation detected")
        >>> elif dw > 2.5:
        ...     print("Negative autocorrelation detected")
        >>> else:
        ...     print("No significant autocorrelation")

        Notes
        -----
        DW test assumes residuals are ordered by time. For panel data,
        the test is computed on the entire residual series.
        """
        from statsmodels.stats.stattools import durbin_watson as dw_test

        return float(dw_test(self.residuals))

    @property
    def ljung_box(self, lags: int = 10) -> tuple:
        """
        Ljung-Box test for autocorrelation in residuals.

        Tests the null hypothesis that there is no autocorrelation
        up to lag order k.

        Parameters
        ----------
        lags : int, default 10
            Number of lags to test

        Returns
        -------
        tuple
            (statistic, pvalue)
            - statistic: Ljung-Box Q-statistic
            - pvalue: P-value for the test

        Examples
        --------
        >>> stat, pvalue = residual_result.ljung_box
        >>> if pvalue < 0.05:
        ...     print("Significant autocorrelation detected")

        Notes
        -----
        The Ljung-Box test is more general than the Durbin-Watson test
        and can detect autocorrelation at multiple lags.
        """
        from statsmodels.stats.diagnostic import acorr_ljungbox

        result = acorr_ljungbox(self.residuals, lags=lags, return_df=True)
        # Returns DataFrame with 'lb_stat' and 'lb_pvalue' columns
        # We take the last lag (most conservative test)
        return float(result["lb_stat"].iloc[-1]), float(result["lb_pvalue"].iloc[-1])

    @property
    def jarque_bera(self) -> tuple:
        """
        Jarque-Bera test for normality.

        Tests the null hypothesis that the residuals follow a normal distribution,
        based on sample skewness and kurtosis.

        Returns
        -------
        tuple
            (statistic, pvalue)

        Examples
        --------
        >>> stat, pvalue = residual_result.jarque_bera
        >>> if pvalue < 0.05:
        ...     print("Residuals deviate from normality")
        """
        jb_stat, jb_pvalue = stats.jarque_bera(self.residuals)
        return float(jb_stat), float(jb_pvalue)

    # ==================== Summary Statistics ====================

    @property
    def mean(self) -> float:
        """Mean of residuals (should be close to 0)."""
        return float(np.mean(self.residuals))

    @property
    def std(self) -> float:
        """Standard deviation of residuals."""
        return float(np.std(self.residuals, ddof=1))

    @property
    def skewness(self) -> float:
        """Skewness of residuals (should be close to 0 for normality)."""
        return float(stats.skew(self.residuals))

    @property
    def kurtosis(self) -> float:
        """Excess kurtosis of residuals (should be close to 0 for normality)."""
        return float(stats.kurtosis(self.residuals))

    @property
    def min(self) -> float:
        """Minimum residual value."""
        return float(np.min(self.residuals))

    @property
    def max(self) -> float:
        """Maximum residual value."""
        return float(np.max(self.residuals))

    # ==================== BaseResult Implementation ====================

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary for report generation.

        This method integrates with ResidualDataTransformer to produce
        visualization-ready data.

        Returns
        -------
        dict
            Complete residual diagnostic data including:
            - residuals, fitted_values, standardized_residuals
            - diagnostic test results
            - summary statistics
            - leverage and influence measures (if available)
            - time/entity structure (for panel data)

        Examples
        --------
        >>> data = residual_result.to_dict()
        >>> print(data.keys())
        dict_keys(['residuals', 'fitted', 'tests', 'summary', ...])
        """
        from panelbox.visualization.transformers.residuals import ResidualDataTransformer

        # Use transformer to get chart-ready data
        transformer = ResidualDataTransformer()
        chart_data = transformer.transform(self.model_results)

        # Add diagnostic test results
        shapiro_stat, shapiro_p = self.shapiro_test
        jb_stat, jb_p = self.jarque_bera
        lb_stat, lb_p = self.ljung_box

        chart_data["tests"] = {
            "shapiro_wilk": {
                "name": "Shapiro-Wilk Normality Test",
                "statistic": shapiro_stat,
                "pvalue": shapiro_p,
                "passed": shapiro_p >= 0.05,
            },
            "jarque_bera": {
                "name": "Jarque-Bera Normality Test",
                "statistic": jb_stat,
                "pvalue": jb_p,
                "passed": jb_p >= 0.05,
            },
            "durbin_watson": {
                "name": "Durbin-Watson Autocorrelation Test",
                "statistic": self.durbin_watson,
                "interpretation": self._interpret_dw(self.durbin_watson),
            },
            "ljung_box": {
                "name": "Ljung-Box Autocorrelation Test",
                "statistic": lb_stat,
                "pvalue": lb_p,
                "passed": lb_p >= 0.05,
            },
        }

        # Add summary statistics
        chart_data["summary"] = {
            "n_obs": len(self.residuals),
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
        }

        return chart_data

    def _interpret_dw(self, dw: float) -> str:
        """Interpret Durbin-Watson statistic."""
        if dw < 1.5:
            return "Positive autocorrelation"
        elif dw > 2.5:
            return "Negative autocorrelation"
        else:
            return "No significant autocorrelation"

    def summary(self) -> str:
        """
        Generate text summary of residual diagnostics.

        Returns
        -------
        str
            Formatted text summary including:
            - Summary statistics
            - Diagnostic test results
            - Interpretation

        Examples
        --------
        >>> print(residual_result.summary())
        Residual Diagnostic Analysis
        =============================

        Summary Statistics:
        ------------------
        Observations:        500
        Mean:                0.0002
        Std. Deviation:      0.9854
        Min:                -2.8431
        Max:                 3.1234
        Skewness:           -0.0234
        Kurtosis:            0.1234

        Diagnostic Tests:
        ----------------
        Shapiro-Wilk (Normality):        W = 0.998, p = 0.234 ✓ PASS
        Jarque-Bera (Normality):         JB = 2.34, p = 0.311 ✓ PASS
        Durbin-Watson (Autocorrelation): DW = 1.987 (No autocorrelation) ✓
        Ljung-Box (Autocorrelation):     Q = 12.3, p = 0.264 ✓ PASS

        Interpretation:
        --------------
        ✓ Residuals appear normally distributed
        ✓ No significant autocorrelation detected
        ✓ Model assumptions satisfied
        """
        shapiro_stat, shapiro_p = self.shapiro_test
        jb_stat, jb_p = self.jarque_bera
        lb_stat, lb_p = self.ljung_box
        dw = self.durbin_watson

        # Build summary string
        lines = [
            "Residual Diagnostic Analysis",
            "=" * 50,
            "",
            "Summary Statistics:",
            "-" * 50,
            f"Observations:        {len(self.residuals):>10d}",
            f"Mean:                {self.mean:>10.4f}",
            f"Std. Deviation:      {self.std:>10.4f}",
            f"Min:                 {self.min:>10.4f}",
            f"Max:                 {self.max:>10.4f}",
            f"Skewness:            {self.skewness:>10.4f}",
            f"Kurtosis:            {self.kurtosis:>10.4f}",
            "",
            "Diagnostic Tests:",
            "-" * 50,
            f"Shapiro-Wilk (Normality):        W = {shapiro_stat:.3f}, p = {shapiro_p:.3f} {self._pass_mark(shapiro_p)}",
            f"Jarque-Bera (Normality):         JB = {jb_stat:.2f}, p = {jb_p:.3f} {self._pass_mark(jb_p)}",
            f"Durbin-Watson (Autocorrelation): DW = {dw:.3f} ({self._interpret_dw(dw)})",
            f"Ljung-Box (Autocorrelation):     Q = {lb_stat:.2f}, p = {lb_p:.3f} {self._pass_mark(lb_p)}",
            "",
            "Interpretation:",
            "-" * 50,
        ]

        # Add interpretations
        normality_ok = shapiro_p >= 0.05 and jb_p >= 0.05
        autocorr_ok = 1.5 <= dw <= 2.5 and lb_p >= 0.05

        if normality_ok:
            lines.append("✓ Residuals appear normally distributed")
        else:
            lines.append("✗ Residuals may not be normally distributed")

        if autocorr_ok:
            lines.append("✓ No significant autocorrelation detected")
        else:
            lines.append("✗ Autocorrelation may be present")

        if normality_ok and autocorr_ok:
            lines.append("✓ Model assumptions satisfied")
        else:
            lines.append("⚠ Some model assumptions may be violated")

        return "\n".join(lines)

    def _pass_mark(self, pvalue: float, alpha: float = 0.05) -> str:
        """Return pass/fail mark based on p-value."""
        return "✓ PASS" if pvalue >= alpha else "✗ FAIL"

    # ==================== Factory Methods ====================

    @classmethod
    def from_model_results(cls, model_results: Any, **kwargs):
        """
        Create ResidualResult from model results.

        This is the recommended way to create a ResidualResult, as it
        automatically extracts all necessary data from the model results.

        Parameters
        ----------
        model_results : PanelResults
            Fitted model results object
        **kwargs
            Additional arguments passed to __init__ (e.g., metadata)

        Returns
        -------
        ResidualResult
            Configured ResidualResult instance

        Examples
        --------
        >>> from panelbox import FixedEffects
        >>> from panelbox.experiment.results import ResidualResult
        >>>
        >>> model = FixedEffects('y ~ x1 + x2', data, entity_id='firm', time_id='year')
        >>> results = model.fit()
        >>>
        >>> residual_result = ResidualResult.from_model_results(
        ...     results,
        ...     metadata={'model_type': 'fixed_effects'}
        ... )
        >>> print(residual_result.summary())
        """
        return cls(model_results=model_results, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ResidualResult(\n"
            f"  n_obs={len(self.residuals)},\n"
            f"  mean={self.mean:.4f},\n"
            f"  std={self.std:.4f},\n"
            f"  normality_tests={'PASS' if self.shapiro_test[1] >= 0.05 else 'FAIL'},\n"
            f"  autocorrelation={'None' if 1.5 <= self.durbin_watson <= 2.5 else 'Present'},\n"
            f"  timestamp={self.timestamp.isoformat()}\n"
            f")"
        )
