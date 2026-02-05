"""
Standard Error Comparison Tools

This module provides tools for comparing different types of standard errors
for the same model specification. It allows researchers to assess the impact
of different SE assumptions on inference.

Classes
-------
StandardErrorComparison
    Compare multiple standard error types for a given model

Examples
--------
>>> import panelbox as pb
>>> import pandas as pd
>>>
>>> # Fit model
>>> fe = pb.FixedEffects("y ~ x1 + x2", data, "entity", "time")
>>> results = fe.fit()
>>>
>>> # Compare all SE types
>>> comparison = pb.StandardErrorComparison(results)
>>> comp_df = comparison.compare_all()
>>> print(comp_df)
>>>
>>> # Plot comparison
>>> comparison.plot_comparison()

References
----------
- Petersen, M. A. (2009). Estimating standard errors in finance panel data sets:
  Comparing approaches. Review of Financial Studies, 22(1), 435-480.
- Thompson, S. B. (2011). Simple formulas for standard errors that cluster by
  both firm and time. Journal of Financial Economics, 99(1), 1-10.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class ComparisonResult:
    """
    Results from comparing multiple standard error types.

    Attributes
    ----------
    se_comparison : pd.DataFrame
        DataFrame with columns for each SE type and rows for each coefficient
    se_ratios : pd.DataFrame
        Ratios of each SE type relative to nonrobust (baseline)
    t_stats : pd.DataFrame
        t-statistics for each coefficient under each SE type
    p_values : pd.DataFrame
        p-values for each coefficient under each SE type
    ci_lower : pd.DataFrame
        Lower bounds of 95% confidence intervals
    ci_upper : pd.DataFrame
        Upper bounds of 95% confidence intervals
    significance : pd.DataFrame
        Significance indicators (*, **, ***) for each SE type
    summary_stats : pd.DataFrame
        Summary statistics across SE types
    """

    se_comparison: pd.DataFrame
    se_ratios: pd.DataFrame
    t_stats: pd.DataFrame
    p_values: pd.DataFrame
    ci_lower: pd.DataFrame
    ci_upper: pd.DataFrame
    significance: pd.DataFrame
    summary_stats: pd.DataFrame


class StandardErrorComparison:
    """
    Compare multiple standard error types for a panel data model.

    This class facilitates comparison of different robust standard error
    estimators to assess the sensitivity of inference to SE assumptions.

    Parameters
    ----------
    model_results : PanelResults
        Fitted model results object

    Attributes
    ----------
    model_results : PanelResults
        Original model results
    coef_names : list
        Coefficient names
    coefficients : np.ndarray
        Point estimates (same across all SE types)
    df_resid : int
        Residual degrees of freedom

    Methods
    -------
    compare_all(se_types=None, **kwargs)
        Compare all specified SE types
    compare_pair(se_type1, se_type2, **kwargs)
        Compare two specific SE types
    plot_comparison(result=None, alpha=0.05)
        Plot comparison of standard errors
    summary(result=None)
        Print summary of comparison

    Examples
    --------
    Compare all SE types:

    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>> comparison = StandardErrorComparison(results)
    >>> comp = comparison.compare_all()
    >>> print(comp.se_comparison)

    Compare specific pair:

    >>> comp = comparison.compare_pair('robust', 'clustered')
    >>> print(f"Max difference: {comp.se_ratios.max().max():.3f}")

    Plot comparison:

    >>> comparison.plot_comparison()

    References
    ----------
    - Petersen, M. A. (2009). Review of Financial Studies, 22(1), 435-480.
    - Thompson, S. B. (2011). Journal of Financial Economics, 99(1), 1-10.
    """

    def __init__(self, model_results):
        """
        Initialize comparison with fitted model results.

        Parameters
        ----------
        model_results : PanelResults
            Fitted model results from FixedEffects, RandomEffects, or PooledOLS
        """
        self.model_results = model_results
        self.coef_names = model_results.params.index.tolist()
        self.coefficients = model_results.params.values
        self.df_resid = model_results.df_resid

        # Extract model info for computing SEs
        self._extract_model_info()

    def _extract_model_info(self):
        """Extract model information for computing different SEs."""
        # Store original model object if available
        # Check multiple possible attribute names
        self.model = getattr(self.model_results, "_model", None)
        if self.model is None:
            self.model = getattr(self.model_results, "model", None)

        # If model not available, store what we need from results
        if self.model is None:
            # Store residuals and fitted values
            self.resid = self.model_results.resid
            self.fittedvalues = self.model_results.fittedvalues

            # Try to reconstruct design matrix from model_info
            # This is a fallback if we can't refit the model
            self._has_model = False
        else:
            self._has_model = True

    def compare_all(self, se_types: Optional[List[str]] = None, **kwargs) -> ComparisonResult:
        """
        Compare all specified standard error types.

        Parameters
        ----------
        se_types : list of str, optional
            List of SE types to compare. If None, uses default list:
            ['nonrobust', 'robust', 'hc3', 'clustered', 'twoway', 'driscoll_kraay']
        **kwargs : dict
            Additional parameters for specific SE types:
            - max_lags : int, for driscoll_kraay and newey_west
            - kernel : str, for driscoll_kraay and newey_west

        Returns
        -------
        ComparisonResult
            Object containing all comparison results

        Examples
        --------
        >>> comparison = StandardErrorComparison(results)
        >>> comp = comparison.compare_all()
        >>> print(comp.se_comparison)

        >>> # Custom SE types
        >>> comp = comparison.compare_all(['nonrobust', 'robust', 'clustered'])

        >>> # With parameters
        >>> comp = comparison.compare_all(
        ...     se_types=['driscoll_kraay', 'newey_west'],
        ...     max_lags=3
        ... )
        """
        if se_types is None:
            # Default list of SE types to compare
            se_types = ["nonrobust", "robust", "hc3", "clustered"]

            # Add advanced types if T is large enough
            if hasattr(self.model_results, "nobs") and self.model_results.nobs > 100:
                se_types.extend(["driscoll_kraay", "newey_west"])

        # Store standard errors for each type
        se_dict = {}

        for se_type in se_types:
            try:
                # Refit model with specific SE type
                if self.model is not None:
                    # Get SE-specific kwargs
                    se_kwargs = self._get_se_kwargs(se_type, **kwargs)
                    results = self.model.fit(cov_type=se_type, **se_kwargs)
                    se_dict[se_type] = results.std_errors.values
                else:
                    # Can't refit, skip this SE type
                    print(f"Warning: Cannot refit model for {se_type}")
                    continue
            except Exception as e:
                print(f"Warning: Failed to compute {se_type} SEs: {str(e)}")
                continue

        if not se_dict:
            raise ValueError("No SE types could be computed successfully")

        # Create comparison DataFrame
        se_comparison = pd.DataFrame(se_dict, index=self.coef_names)

        # Compute ratios relative to nonrobust (if available)
        if "nonrobust" in se_dict:
            se_ratios = se_comparison.div(se_comparison["nonrobust"], axis=0)
        else:
            # Use first SE type as baseline
            baseline = list(se_dict.keys())[0]
            se_ratios = se_comparison.div(se_comparison[baseline], axis=0)

        # Compute t-statistics
        t_stats = pd.DataFrame(
            {se_type: self.coefficients / se_dict[se_type] for se_type in se_dict.keys()},
            index=self.coef_names,
        )

        # Compute p-values (two-tailed)
        from scipy import stats

        p_values = pd.DataFrame(
            {
                se_type: 2 * (1 - stats.t.cdf(np.abs(t_stats[se_type]), self.df_resid))
                for se_type in se_dict.keys()
            },
            index=self.coef_names,
        )

        # Compute 95% confidence intervals
        t_crit = stats.t.ppf(0.975, self.df_resid)
        ci_lower = pd.DataFrame(
            {se_type: self.coefficients - t_crit * se_dict[se_type] for se_type in se_dict.keys()},
            index=self.coef_names,
        )
        ci_upper = pd.DataFrame(
            {se_type: self.coefficients + t_crit * se_dict[se_type] for se_type in se_dict.keys()},
            index=self.coef_names,
        )

        # Significance indicators
        significance = p_values.copy()
        significance = significance.map(self._significance_stars)

        # Summary statistics
        summary_stats = pd.DataFrame(
            {
                "mean_se": se_comparison.mean(axis=1),
                "std_se": se_comparison.std(axis=1),
                "min_se": se_comparison.min(axis=1),
                "max_se": se_comparison.max(axis=1),
                "range_se": se_comparison.max(axis=1) - se_comparison.min(axis=1),
                "cv_se": se_comparison.std(axis=1)
                / se_comparison.mean(axis=1),  # Coefficient of variation
            }
        )

        return ComparisonResult(
            se_comparison=se_comparison,
            se_ratios=se_ratios,
            t_stats=t_stats,
            p_values=p_values,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            significance=significance,
            summary_stats=summary_stats,
        )

    def compare_pair(self, se_type1: str, se_type2: str, **kwargs) -> ComparisonResult:
        """
        Compare two specific standard error types.

        Parameters
        ----------
        se_type1 : str
            First SE type (e.g., 'nonrobust')
        se_type2 : str
            Second SE type (e.g., 'clustered')
        **kwargs : dict
            Additional parameters for SE types

        Returns
        -------
        ComparisonResult
            Comparison results for the two SE types

        Examples
        --------
        >>> comp = comparison.compare_pair('robust', 'clustered')
        >>> print(comp.se_ratios)
        """
        return self.compare_all(se_types=[se_type1, se_type2], **kwargs)

    def plot_comparison(
        self,
        result: Optional[ComparisonResult] = None,
        alpha: float = 0.05,
        figsize: tuple = (12, 8),
    ):
        """
        Plot comparison of standard errors and confidence intervals.

        Parameters
        ----------
        result : ComparisonResult, optional
            Pre-computed comparison result. If None, computes comparison.
        alpha : float, default=0.05
            Significance level for confidence intervals
        figsize : tuple, default=(12, 8)
            Figure size (width, height)

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object

        Examples
        --------
        >>> comparison.plot_comparison()
        >>>
        >>> # Custom figure size
        >>> comparison.plot_comparison(figsize=(14, 10))

        Notes
        -----
        Requires matplotlib to be installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "Matplotlib is required for plotting. " "Install it with: pip install matplotlib"
            )

        if result is None:
            result = self.compare_all()

        n_coefs = len(self.coef_names)
        n_se_types = len(result.se_comparison.columns)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Plot 1: Standard Errors Comparison
        ax1 = axes[0]
        result.se_comparison.plot(kind="bar", ax=ax1)
        ax1.set_title("Standard Errors Comparison", fontsize=14, fontweight="bold")
        ax1.set_xlabel("Coefficient", fontsize=12)
        ax1.set_ylabel("Standard Error", fontsize=12)
        ax1.legend(title="SE Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Coefficient Estimates with Confidence Intervals
        ax2 = axes[1]
        x = np.arange(n_coefs)
        width = 0.8 / n_se_types

        for i, se_type in enumerate(result.se_comparison.columns):
            offset = (i - n_se_types / 2 + 0.5) * width
            ax2.errorbar(
                x + offset,
                self.coefficients,
                yerr=[
                    self.coefficients - result.ci_lower[se_type].values,
                    result.ci_upper[se_type].values - self.coefficients,
                ],
                fmt="o",
                label=se_type,
                capsize=5,
                capthick=2,
            )

        ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
        ax2.set_title(
            "Coefficient Estimates with 95% Confidence Intervals", fontsize=14, fontweight="bold"
        )
        ax2.set_xlabel("Coefficient", fontsize=12)
        ax2.set_ylabel("Estimate", fontsize=12)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.coef_names, rotation=45, ha="right")
        ax2.legend(title="SE Type", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    def summary(self, result: Optional[ComparisonResult] = None):
        """
        Print summary of standard error comparison.

        Parameters
        ----------
        result : ComparisonResult, optional
            Pre-computed comparison result. If None, computes comparison.

        Examples
        --------
        >>> comparison.summary()
        """
        if result is None:
            result = self.compare_all()

        print("=" * 80)
        print("STANDARD ERROR COMPARISON SUMMARY")
        print("=" * 80)
        print()

        print("Standard Errors by Type:")
        print("-" * 80)
        print(result.se_comparison.to_string())
        print()

        print("Standard Error Ratios (relative to baseline):")
        print("-" * 80)
        print(result.se_ratios.to_string(float_format=lambda x: f"{x:.3f}"))
        print()

        print("Significance Levels (* p<0.10, ** p<0.05, *** p<0.01):")
        print("-" * 80)

        # Combine coefficients with significance
        sig_table = pd.DataFrame({"Coefficient": self.coefficients})
        for col in result.significance.columns:
            sig_table[col] = result.significance[col]
        print(
            sig_table.to_string(
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)
            )
        )
        print()

        print("Summary Statistics Across SE Types:")
        print("-" * 80)
        print(result.summary_stats.to_string(float_format=lambda x: f"{x:.4f}"))
        print()

        # Inference sensitivity analysis
        print("Inference Sensitivity:")
        print("-" * 80)

        # Count significant coefficients by SE type
        sig_counts = (result.p_values < 0.05).sum()
        print("Coefficients significant at 5% level:")
        for se_type, count in sig_counts.items():
            print(f"  {se_type:20s}: {count}/{len(self.coef_names)}")
        print()

        # Identify coefficients with inconsistent inference
        sig_matrix = result.p_values < 0.05
        inconsistent = sig_matrix.sum(axis=1)
        inconsistent = inconsistent[
            (inconsistent > 0) & (inconsistent < len(result.p_values.columns))
        ]

        if len(inconsistent) > 0:
            print("⚠️  Coefficients with inconsistent inference across SE types:")
            for coef in inconsistent.index:
                sig_types = sig_matrix.loc[coef]
                sig_list = [st for st, is_sig in sig_types.items() if is_sig]
                nonsig_list = [st for st, is_sig in sig_types.items() if not is_sig]
                print(f"  {coef}:")
                print(f"    Significant with: {', '.join(sig_list)}")
                print(f"    Not significant with: {', '.join(nonsig_list)}")
        else:
            print("✓ Inference is consistent across all SE types")

        print()
        print("=" * 80)

    def _get_se_kwargs(self, se_type: str, **kwargs) -> Dict[str, Any]:
        """Get SE-specific keyword arguments."""
        se_kwargs = {}

        if se_type in ["driscoll_kraay", "newey_west"]:
            if "max_lags" in kwargs:
                se_kwargs["max_lags"] = kwargs["max_lags"]
            if "kernel" in kwargs:
                se_kwargs["kernel"] = kwargs["kernel"]

        return se_kwargs

    @staticmethod
    def _significance_stars(p_value: float) -> str:
        """Convert p-value to significance stars."""
        if p_value < 0.01:
            return "***"
        elif p_value < 0.05:
            return "**"
        elif p_value < 0.10:
            return "*"
        else:
            return ""
