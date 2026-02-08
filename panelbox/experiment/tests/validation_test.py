"""
ValidationTest - Runner for validation tests.

This module provides a test runner that executes validation tests on
fitted panel models and returns ValidationResult objects.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from panelbox.experiment.results import ValidationResult


class ValidationTest:
    """
    Runner for validation tests on panel models.

    This class runs diagnostic tests on fitted panel models
    and returns a ValidationResult. It provides configurable test
    sets (quick, basic, full) and integrates with PanelExperiment.

    Examples
    --------
    >>> from panelbox.experiment.tests import ValidationTest
    >>> import panelbox as pb
    >>>
    >>> # Fit a model
    >>> data = pb.load_grunfeld()
    >>> experiment = pb.PanelExperiment(
    ...     data=data,
    ...     formula="invest ~ value + capital",
    ...     entity_col="firm",
    ...     time_col="year"
    ... )
    >>> results = experiment.fit_model('fixed_effects', name='fe')
    >>>
    >>> # Run validation tests
    >>> test_runner = ValidationTest()
    >>> validation_result = test_runner.run(results, config='full')
    >>>
    >>> # Save report
    >>> validation_result.save_html('validation.html', test_type='validation')
    """

    # Test configurations
    CONFIGS = {
        "quick": ["heteroskedasticity", "autocorrelation"],
        "basic": ["heteroskedasticity", "autocorrelation", "normality"],
        "full": ["heteroskedasticity", "autocorrelation", "normality", "hausman"],
    }

    def __init__(self):
        """Initialize ValidationTest runner."""
        pass

    def run(
        self, results: Any, tests: Optional[List[str]] = None, config: str = "basic"
    ) -> ValidationResult:
        """
        Run validation tests on model results.

        Parameters
        ----------
        results : PanelResults
            Fitted model results from linearmodels or panelbox models
        tests : list of str, optional
            Specific tests to run. If None, uses config.
            Available tests: 'heteroskedasticity', 'autocorrelation',
            'normality', 'hausman'
        config : str, default 'basic'
            Test configuration: 'quick', 'basic', or 'full'
            - 'quick': heteroskedasticity, autocorrelation (fastest)
            - 'basic': adds normality test
            - 'full': adds Hausman test

        Returns
        -------
        ValidationResult
            Result container with test results

        Raises
        ------
        ValueError
            If config is not 'quick', 'basic', or 'full'
        TypeError
            If results is not a valid model results object

        Examples
        --------
        >>> runner = ValidationTest()
        >>>
        >>> # Run with basic config
        >>> result = runner.run(model_results, config='basic')
        >>>
        >>> # Run specific tests
        >>> result = runner.run(
        ...     model_results,
        ...     tests=['heteroskedasticity', 'normality']
        ... )
        >>>
        >>> # Run full test suite
        >>> result = runner.run(model_results, config='full')
        """
        # Validate config
        if config not in self.CONFIGS:
            raise ValueError(
                f"config must be one of {list(self.CONFIGS.keys())}, " f"got '{config}'"
            )

        # Determine which tests to run
        if tests is None:
            tests = self.CONFIGS.get(config, self.CONFIGS["basic"])

        # Check if model has validate() method (panelbox models)
        if hasattr(results, "validate"):
            # Use built-in validation
            validation_report = results.validate(tests=tests)

            # Create ValidationResult
            validation_result = ValidationResult(
                validation_report=validation_report, model_results=results
            )

            return validation_result
        else:
            # Fall back to manual test execution for linearmodels results
            raise NotImplementedError(
                "ValidationTest runner currently requires panelbox models "
                "with a validate() method. For linearmodels results, "
                "use the model's built-in diagnostic methods."
            )

    def _extract_model_info(self, results: Any) -> Dict[str, Any]:
        """
        Extract model information from results.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        dict
            Model information including metrics, sample size, etc.
        """
        model_info = {
            "model_type": results.__class__.__name__,
            "n_obs": results.nobs if hasattr(results, "nobs") else None,
            "n_params": len(results.params) if hasattr(results, "params") else None,
        }

        # Add common metrics if available
        if hasattr(results, "rsquared"):
            model_info["rsquared"] = results.rsquared

        if hasattr(results, "rsquared_adj"):
            model_info["rsquared_adj"] = results.rsquared_adj
        elif hasattr(results, "rsquared"):
            model_info["rsquared_adj"] = results.rsquared

        if hasattr(results, "aic"):
            model_info["aic"] = results.aic

        if hasattr(results, "bic"):
            model_info["bic"] = results.bic

        if hasattr(results, "loglik"):
            model_info["log_likelihood"] = results.loglik

        if hasattr(results, "f_statistic"):
            model_info["f_statistic"] = results.f_statistic.stat

        return model_info

    def _extract_warnings(self, results: Any) -> List[str]:
        """
        Extract warnings from model results.

        Parameters
        ----------
        results : PanelResults
            Model results

        Returns
        -------
        list of str
            List of warning messages
        """
        warnings = []

        # Check for low R²
        if hasattr(results, "rsquared"):
            if results.rsquared < 0.3:
                warnings.append("Low R² (< 0.3). Model explains little variance in the data.")

        # Check for high condition number (multicollinearity indicator)
        if hasattr(results, "condition_number"):
            if results.condition_number > 30:
                warnings.append(
                    "High condition number (> 30). Potential multicollinearity detected."
                )

        # Check for small sample size
        if hasattr(results, "nobs"):
            if results.nobs < 30:
                warnings.append(f"Small sample size (n={results.nobs}). Results may be unreliable.")

        return warnings

    def __repr__(self) -> str:
        """String representation of ValidationTest."""
        return f"ValidationTest(\n" f"  configs={list(self.CONFIGS.keys())}\n" f")"
