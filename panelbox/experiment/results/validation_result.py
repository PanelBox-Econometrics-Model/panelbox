"""
ValidationResult - Container for validation test results.

This module provides a concrete implementation of BaseResult for
validation test results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from panelbox.experiment.results.base import BaseResult


class ValidationResult(BaseResult):
    """
    Result container for validation tests.

    This class stores results from panel model validation tests and provides
    methods to save as HTML reports or JSON files.

    Parameters
    ----------
    validation_report : ValidationReport
        ValidationReport object from model.validate()
    model_results : PanelResults, optional
        The model results that were validated
    timestamp : datetime, optional
        Timestamp of validation. If None, uses current time.
    metadata : dict, optional
        Additional metadata

    Attributes
    ----------
    validation_report : ValidationReport
        The validation report object
    model_results : PanelResults or None
        The model results
    timestamp : datetime
        When validation was performed
    metadata : dict
        Additional metadata

    Examples
    --------
    >>> # After fitting a model and running validation
    >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year")
    >>> results = fe.fit()
    >>> validation = results.validate(tests="default")
    >>>
    >>> # Create ValidationResult
    >>> val_result = ValidationResult(
    ...     validation_report=validation,
    ...     model_results=results
    ... )
    >>>
    >>> # Save as HTML
    >>> val_result.save_html(
    ...     'validation_report.html',
    ...     test_type='validation',
    ...     theme='professional'
    ... )
    >>>
    >>> # Save as JSON
    >>> val_result.save_json('validation_result.json')
    >>>
    >>> # Get summary
    >>> print(val_result.summary())
    """

    def __init__(
        self,
        validation_report,
        model_results=None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ValidationResult.

        Parameters
        ----------
        validation_report : ValidationReport
            Validation report from model.validate()
        model_results : PanelResults, optional
            Model results that were validated
        timestamp : datetime, optional
            Timestamp of validation
        metadata : dict, optional
            Additional metadata
        """
        super().__init__(timestamp=timestamp, metadata=metadata)

        self.validation_report = validation_report
        self.model_results = model_results

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation result to dictionary.

        This method uses ValidationTransformer to convert the ValidationReport
        to a template-friendly format.

        Returns
        -------
        dict
            Dictionary with validation data ready for reporting

        Examples
        --------
        >>> data = val_result.to_dict()
        >>> print(data.keys())
        dict_keys(['model_info', 'tests', 'summary', 'recommendations', 'charts'])
        """
        from panelbox.report.validation_transformer import ValidationTransformer

        # Use ValidationTransformer to convert report
        transformer = ValidationTransformer(self.validation_report)
        data = transformer.transform(include_charts=True, use_new_visualization=True)

        return data

    def summary(self) -> str:
        """
        Generate text summary of validation results.

        Returns
        -------
        str
            Formatted text summary

        Examples
        --------
        >>> print(val_result.summary())
        Validation Report Summary
        =========================
        Total Tests: 9
        Passed: 2
        Failed: 7
        ...
        """
        # Use ValidationReport's built-in summary
        return self.validation_report.summary(verbose=True)

    @property
    def total_tests(self) -> int:
        """
        Get total number of tests run.

        Returns
        -------
        int
            Total number of tests

        Examples
        --------
        >>> val_result.total_tests
        9
        """
        return (
            len(self.validation_report.specification_tests or {})
            + len(self.validation_report.serial_tests or {})
            + len(self.validation_report.het_tests or {})
            + len(self.validation_report.cd_tests or {})
        )

    @property
    def failed_tests(self) -> List[str]:
        """
        Get list of failed tests.

        Returns
        -------
        list of str
            Names of tests that failed

        Examples
        --------
        >>> val_result.failed_tests
        ['Wooldridge Test', 'Breusch-Pagan Test', ...]
        """
        return self.validation_report.get_failed_tests()

    @property
    def passed_tests(self) -> List[str]:
        """
        Get list of passed tests.

        Returns
        -------
        list of str
            Names of tests that passed

        Examples
        --------
        >>> val_result.passed_tests
        ['Hausman Test', 'Pesaran CD Test']
        """
        all_tests = []
        failed = set(self.failed_tests)

        # Collect all test names
        if self.validation_report.specification_tests:
            all_tests.extend(self.validation_report.specification_tests.keys())
        if self.validation_report.serial_tests:
            all_tests.extend(self.validation_report.serial_tests.keys())
        if self.validation_report.het_tests:
            all_tests.extend(self.validation_report.het_tests.keys())
        if self.validation_report.cd_tests:
            all_tests.extend(self.validation_report.cd_tests.keys())

        # Return tests not in failed set
        return [test for test in all_tests if test not in failed]

    @property
    def pass_rate(self) -> float:
        """
        Calculate pass rate.

        Returns
        -------
        float
            Pass rate (0.0 to 1.0)

        Examples
        --------
        >>> val_result.pass_rate
        0.222
        """
        total = self.total_tests
        if total == 0:
            return 0.0
        return len(self.passed_tests) / total

    @classmethod
    def from_model_results(
        cls,
        model_results,
        alpha: float = 0.05,
        tests: str = "default",
        verbose: bool = False,
        **kwargs,
    ):
        """
        Create ValidationResult from model results by running validation.

        This is a convenience factory method that runs validation and creates
        the result container in one step.

        Parameters
        ----------
        model_results : PanelResults
            Model results from fitting
        alpha : float, default 0.05
            Significance level for tests
        tests : str, default "default"
            Which tests to run
        verbose : bool, default False
            Whether to print progress
        **kwargs
            Additional arguments passed to ValidationResult.__init__()

        Returns
        -------
        ValidationResult
            ValidationResult container

        Examples
        --------
        >>> fe = pb.FixedEffects("y ~ x1 + x2", data, "firm", "year")
        >>> results = fe.fit()
        >>>
        >>> # Create ValidationResult directly
        >>> val_result = ValidationResult.from_model_results(
        ...     model_results=results,
        ...     alpha=0.05,
        ...     tests="default"
        ... )
        >>>
        >>> # Save report
        >>> val_result.save_html('report.html', test_type='validation')
        """
        # Run validation
        validation_report = model_results.validate(tests=tests, alpha=alpha, verbose=verbose)

        # Create ValidationResult
        return cls(validation_report=validation_report, model_results=model_results, **kwargs)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ValidationResult(\n"
            f"  total_tests={self.total_tests},\n"
            f"  passed={len(self.passed_tests)},\n"
            f"  failed={len(self.failed_tests)},\n"
            f"  pass_rate={self.pass_rate:.1%},\n"
            f"  timestamp={self.timestamp.isoformat()}\n"
            f")"
        )
