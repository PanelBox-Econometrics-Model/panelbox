"""
ComparisonTest - Runner for model comparison.

This module provides a test runner that compares multiple fitted panel
models and returns ComparisonResult objects.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from panelbox.experiment.results import ComparisonResult


class ComparisonTest:
    """
    Runner for comparing multiple panel models.

    This class compares fitted panel models by extracting metrics and
    coefficients, then returns a ComparisonResult. It integrates with
    PanelExperiment for easy multi-model comparison.

    Examples
    --------
    >>> from panelbox.experiment.tests import ComparisonTest
    >>> import panelbox as pb
    >>>
    >>> # Fit multiple models
    >>> data = pb.load_grunfeld()
    >>> experiment = pb.PanelExperiment(
    ...     data=data,
    ...     formula="invest ~ value + capital",
    ...     entity_col="firm",
    ...     time_col="year"
    ... )
    >>> ols_res = experiment.fit_model('pooled_ols', name='ols')
    >>> fe_res = experiment.fit_model('fixed_effects', name='fe')
    >>>
    >>> # Compare models
    >>> test_runner = ComparisonTest()
    >>> comparison_result = test_runner.run({
    ...     'ols': ols_res,
    ...     'fe': fe_res
    ... })
    >>>
    >>> # Save report
    >>> comparison_result.save_html('comparison.html', test_type='comparison')
    """

    def __init__(self):
        """Initialize ComparisonTest runner."""
        pass

    def run(
        self,
        models: Dict[str, Any],
        include_coefficients: bool = True,
        include_statistics: bool = True,
    ) -> ComparisonResult:
        """
        Compare multiple fitted models.

        Parameters
        ----------
        models : dict
            Dictionary of model results {name: results}
            At least 2 models are required for comparison
        include_coefficients : bool, default True
            Whether to extract and compare coefficients
        include_statistics : bool, default True
            Whether to extract and compare model statistics

        Returns
        -------
        ComparisonResult
            Result container with model comparison

        Raises
        ------
        ValueError
            If fewer than 2 models are provided
        TypeError
            If models is not a dictionary

        Examples
        --------
        >>> runner = ComparisonTest()
        >>>
        >>> # Compare two models
        >>> models = {'ols': ols_results, 'fe': fe_results}
        >>> result = runner.run(models)
        >>>
        >>> # Compare three models
        >>> models = {
        ...     'ols': ols_results,
        ...     'fe': fe_results,
        ...     're': re_results
        ... }
        >>> result = runner.run(models)
        >>>
        >>> # Get best model by AIC
        >>> best = result.best_model('aic', prefer_lower=True)
        """
        # Validate inputs
        if not isinstance(models, dict):
            raise TypeError("models must be a dictionary {name: results}")

        if len(models) < 2:
            raise ValueError(
                "ComparisonTest requires at least 2 models for comparison, " f"got {len(models)}"
            )

        # ComparisonResult will auto-compute metrics
        # We can pass pre-computed metrics if needed
        comparison_metrics = None
        if include_statistics or include_coefficients:
            # ComparisonResult auto-computes metrics in _compute_metrics()
            # So we just pass models and let it handle the rest
            pass

        # Create ComparisonResult
        # The ComparisonResult class will handle metric computation
        comparison_result = ComparisonResult(models=models, comparison_metrics=comparison_metrics)

        return comparison_result

    def _extract_metrics(self, models: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract goodness-of-fit metrics from all models.

        Parameters
        ----------
        models : dict
            Dictionary of models {name: results}

        Returns
        -------
        pd.DataFrame
            Metrics table with models as rows and metrics as columns
        """
        metrics_data = {}

        for name, results in models.items():
            model_metrics = {}

            # R-squared
            if hasattr(results, "rsquared"):
                model_metrics["rsquared"] = results.rsquared
            else:
                model_metrics["rsquared"] = None

            # Adjusted R-squared
            if hasattr(results, "rsquared_adj"):
                model_metrics["rsquared_adj"] = results.rsquared_adj
            elif hasattr(results, "rsquared"):
                model_metrics["rsquared_adj"] = results.rsquared
            else:
                model_metrics["rsquared_adj"] = None

            # AIC
            if hasattr(results, "aic"):
                model_metrics["aic"] = results.aic
            else:
                model_metrics["aic"] = None

            # BIC
            if hasattr(results, "bic"):
                model_metrics["bic"] = results.bic
            else:
                model_metrics["bic"] = None

            # Log-likelihood
            if hasattr(results, "loglik"):
                model_metrics["loglik"] = results.loglik
            else:
                model_metrics["loglik"] = None

            # F-statistic
            if hasattr(results, "f_statistic"):
                if hasattr(results.f_statistic, "stat"):
                    model_metrics["f_stat"] = results.f_statistic.stat
                else:
                    model_metrics["f_stat"] = results.f_statistic
            else:
                model_metrics["f_stat"] = None

            # Number of observations
            if hasattr(results, "nobs"):
                model_metrics["n_obs"] = int(results.nobs)
            else:
                model_metrics["n_obs"] = None

            # Number of parameters
            if hasattr(results, "params"):
                model_metrics["n_params"] = len(results.params)
            else:
                model_metrics["n_params"] = None

            metrics_data[name] = model_metrics

        # Convert to DataFrame (models as rows)
        metrics_df = pd.DataFrame(metrics_data).T

        return metrics_df

    def _extract_coefficients(self, models: Dict[str, Any]) -> pd.DataFrame:
        """
        Extract parameter estimates from all models.

        Parameters
        ----------
        models : dict
            Dictionary of models {name: results}

        Returns
        -------
        pd.DataFrame
            Coefficients table with parameters as rows and models as columns
        """
        coef_data = {}

        for name, results in models.items():
            if hasattr(results, "params"):
                coef_data[name] = results.params
            else:
                coef_data[name] = pd.Series(dtype=float)

        # Convert to DataFrame (parameters as rows, models as columns)
        coef_df = pd.DataFrame(coef_data)

        return coef_df

    def __repr__(self) -> str:
        """String representation of ComparisonTest."""
        return "ComparisonTest()"
