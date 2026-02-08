"""
ComparisonResult - Container for model comparison results.

This module provides a concrete implementation of BaseResult for
model comparison results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from panelbox.experiment.results.base import BaseResult


class ComparisonResult(BaseResult):
    """
    Result container for model comparison.

    This class stores results from comparing multiple panel models and provides
    methods to save as HTML reports or JSON files.

    Parameters
    ----------
    models : dict
        Dictionary of {model_name: PanelResults}
    comparison_metrics : dict, optional
        Dictionary of comparison metrics (AIC, BIC, R², etc.)
    timestamp : datetime, optional
        Timestamp of comparison. If None, uses current time.
    metadata : dict, optional
        Additional metadata

    Attributes
    ----------
    models : dict
        Dictionary of fitted models
    comparison_metrics : dict
        Comparison metrics for all models
    timestamp : datetime
        When comparison was performed
    metadata : dict
        Additional metadata

    Examples
    --------
    >>> # After fitting multiple models
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year")
    >>> fe_results = fe.fit()
    >>> re = pb.RandomEffects("y ~ x1 + x2", data, "firm", "year")
    >>> re_results = re.fit()
    >>>
    >>> # Create ComparisonResult
    >>> comp_result = ComparisonResult(
    ...     models={'Fixed Effects': fe_results, 'Random Effects': re_results}
    ... )
    >>>
    >>> # Save as HTML
    >>> comp_result.save_html(
    ...     'comparison_report.html',
    ...     test_type='comparison',
    ...     theme='professional'
    ... )
    >>>
    >>> # Save as JSON
    >>> comp_result.save_json('comparison_result.json')
    >>>
    >>> # Get summary
    >>> print(comp_result.summary())
    >>>
    >>> # Find best model
    >>> best = comp_result.best_model('rsquared')
    """

    def __init__(
        self,
        models: Dict[str, Any],
        comparison_metrics: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ComparisonResult.

        Parameters
        ----------
        models : dict
            Dictionary of {model_name: PanelResults}
        comparison_metrics : dict, optional
            Pre-computed comparison metrics
        timestamp : datetime, optional
            Timestamp of comparison
        metadata : dict, optional
            Additional metadata
        """
        super().__init__(timestamp=timestamp, metadata=metadata)

        if not models:
            raise ValueError("Must provide at least one model for comparison")

        self.models = models
        self.comparison_metrics = comparison_metrics or self._compute_metrics()

    def _compute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute comparison metrics for all models.

        Returns
        -------
        dict
            Dictionary with metrics for each model
        """
        metrics = {}

        for name, results in self.models.items():
            model_metrics = {
                "rsquared": getattr(results, "rsquared", None),
                "rsquared_adj": getattr(results, "rsquared_adj", None),
                "fvalue": getattr(results, "f_statistic", {}).get("stat", None),
                "f_pvalue": getattr(results, "f_statistic", {}).get("pval", None),
                "nobs": getattr(results, "nobs", None),
                "df_model": getattr(results, "df_model", None),
                "df_resid": getattr(results, "df_resid", None),
            }

            # Add information criteria if available
            if hasattr(results, "loglik"):
                k = results.df_model + 1  # Number of parameters
                n = results.nobs
                ll = results.loglik

                # AIC = -2*LL + 2*k
                model_metrics["aic"] = -2 * ll + 2 * k

                # BIC = -2*LL + k*ln(n)
                import math

                model_metrics["bic"] = -2 * ll + k * math.log(n)

            metrics[name] = model_metrics

        return metrics

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert comparison result to dictionary.

        This method uses ComparisonDataTransformer to convert the comparison
        to a template-friendly format.

        Returns
        -------
        dict
            Dictionary with comparison data ready for reporting

        Examples
        --------
        >>> data = comp_result.to_dict()
        >>> print(data.keys())
        dict_keys(['models', 'comparison_table', 'summary', 'charts'])
        """
        from panelbox.visualization import create_comparison_charts
        from panelbox.visualization.transformers.comparison import ComparisonDataTransformer

        # Convert models dict to list and names
        model_names = list(self.models.keys())
        results_list = list(self.models.values())

        # Use ComparisonDataTransformer to extract data
        transformer = ComparisonDataTransformer()
        comparison_data = transformer.transform(results_list, names=model_names)

        # Create charts
        try:
            charts = create_comparison_charts(results_list, names=model_names)
        except Exception:
            charts = {}

        # Build final data structure
        data = {
            "models": model_names,
            "comparison_data": comparison_data,
            "comparison_metrics": self.comparison_metrics,
            "charts": charts,
            "summary": {
                "n_models": self.n_models,
                "timestamp": self.timestamp.isoformat(),
            },
        }

        return data

    def summary(self) -> str:
        """
        Generate text summary of comparison results.

        Returns
        -------
        str
            Formatted text summary

        Examples
        --------
        >>> print(comp_result.summary())
        Model Comparison Summary
        ========================
        Models Compared: 3
        ...
        """
        lines = []
        lines.append("=" * 80)
        lines.append("MODEL COMPARISON SUMMARY")
        lines.append("=" * 80)
        lines.append("")

        # Overview
        lines.append(f"Models Compared: {len(self.models)}")
        lines.append(f"Comparison Date: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Comparison table
        lines.append("Comparison Metrics:")
        lines.append("-" * 80)

        # Header
        header = f"{'Model':<25} {'R²':<10} {'R² Adj':<10} {'AIC':<12} {'BIC':<12}"
        lines.append(header)
        lines.append("-" * 80)

        # Rows
        for name, metrics in self.comparison_metrics.items():
            rsq = metrics.get("rsquared")
            rsq_adj = metrics.get("rsquared_adj")
            aic = metrics.get("aic")
            bic = metrics.get("bic")

            rsq_str = f"{rsq:.4f}" if rsq is not None else "N/A"
            rsq_adj_str = f"{rsq_adj:.4f}" if rsq_adj is not None else "N/A"
            aic_str = f"{aic:.2f}" if aic is not None else "N/A"
            bic_str = f"{bic:.2f}" if bic is not None else "N/A"

            row = f"{name:<25} {rsq_str:<10} {rsq_adj_str:<10} {aic_str:<12} {bic_str:<12}"
            lines.append(row)

        lines.append("-" * 80)
        lines.append("")

        # Best models
        lines.append("Best Models by Metric:")
        lines.append("-" * 80)

        # Best R²
        best_rsq = self.best_model("rsquared")
        if best_rsq:
            lines.append(f"  • Highest R²: {best_rsq}")

        # Best AIC (lowest)
        best_aic = self.best_model("aic", prefer_lower=True)
        if best_aic:
            lines.append(f"  • Lowest AIC: {best_aic}")

        # Best BIC (lowest)
        best_bic = self.best_model("bic", prefer_lower=True)
        if best_bic:
            lines.append(f"  • Lowest BIC: {best_bic}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)

    def best_model(self, metric: str, prefer_lower: bool = False) -> Optional[str]:
        """
        Find the best model according to a specific metric.

        Parameters
        ----------
        metric : str
            Metric to use ('rsquared', 'rsquared_adj', 'aic', 'bic', etc.)
        prefer_lower : bool, default False
            If True, lower values are better (e.g., for AIC, BIC)
            If False, higher values are better (e.g., for R²)

        Returns
        -------
        str or None
            Name of the best model, or None if metric not available

        Examples
        --------
        >>> comp_result.best_model('rsquared')
        'Fixed Effects'
        >>> comp_result.best_model('aic', prefer_lower=True)
        'Random Effects'
        """
        valid_models = {
            name: metrics.get(metric)
            for name, metrics in self.comparison_metrics.items()
            if metrics.get(metric) is not None
        }

        if not valid_models:
            return None

        if prefer_lower:
            best_model = min(valid_models.items(), key=lambda x: x[1])
        else:
            best_model = max(valid_models.items(), key=lambda x: x[1])

        return best_model[0]

    @property
    def model_names(self) -> List[str]:
        """
        Get list of model names.

        Returns
        -------
        list of str
            Names of all models in comparison

        Examples
        --------
        >>> comp_result.model_names
        ['Pooled OLS', 'Fixed Effects', 'Random Effects']
        """
        return list(self.models.keys())

    @property
    def n_models(self) -> int:
        """
        Get number of models being compared.

        Returns
        -------
        int
            Number of models

        Examples
        --------
        >>> comp_result.n_models
        3
        """
        return len(self.models)

    @classmethod
    def from_experiment(cls, experiment, model_names: Optional[List[str]] = None, **kwargs):
        """
        Create ComparisonResult from a PanelExperiment.

        This is a convenience factory method that extracts fitted models
        from a PanelExperiment and creates a comparison.

        Parameters
        ----------
        experiment : PanelExperiment
            Experiment with fitted models
        model_names : list of str, optional
            Names of models to compare. If None, compares all models.
        **kwargs
            Additional arguments passed to ComparisonResult.__init__()

        Returns
        -------
        ComparisonResult
            ComparisonResult container

        Examples
        --------
        >>> experiment = PanelExperiment(data, "y ~ x1 + x2", "firm", "year")
        >>> experiment.fit_model('pooled_ols', name='pooled')
        >>> experiment.fit_model('fixed_effects', name='fe')
        >>> experiment.fit_model('random_effects', name='re')
        >>>
        >>> # Compare all models
        >>> comp_result = ComparisonResult.from_experiment(experiment)
        >>>
        >>> # Compare specific models
        >>> comp_result = ComparisonResult.from_experiment(
        ...     experiment,
        ...     model_names=['fe', 're']
        ... )
        """
        all_models = experiment.list_models()

        if model_names is None:
            model_names = all_models
        else:
            # Validate model names
            invalid = set(model_names) - set(all_models)
            if invalid:
                raise ValueError(f"Models not found in experiment: {invalid}")

        # Extract models
        models = {name: experiment.get_model(name) for name in model_names}

        # Create ComparisonResult
        return cls(models=models, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ComparisonResult(\n"
            f"  n_models={self.n_models},\n"
            f"  models={self.model_names},\n"
            f"  timestamp={self.timestamp.isoformat()}\n"
            f")"
        )
