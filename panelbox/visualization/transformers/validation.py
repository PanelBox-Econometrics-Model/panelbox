"""
Data transformer for validation reports.

Converts ValidationReport objects into the data format expected by validation charts.
"""

from typing import Any, Dict, List


class ValidationDataTransformer:
    """
    Transform ValidationReport to chart-friendly format.

    Takes a ValidationReport object and converts it into a structured
    dictionary that can be consumed by validation chart APIs.

    Examples
    --------
    >>> from panelbox.validation import ValidationReport
    >>> from panelbox.visualization.transformers import ValidationDataTransformer
    >>>
    >>> validation_report = ValidationReport(...)
    >>> transformer = ValidationDataTransformer()
    >>> data = transformer.transform(validation_report)
    >>>
    >>> # Use with chart API
    >>> charts = create_validation_charts(data)
    """

    def transform(self, validation_report: Any) -> Dict[str, Any]:
        """
        Transform ValidationReport to chart data format.

        Parameters
        ----------
        validation_report : ValidationReport
            Validation report object with test results

        Returns
        -------
        dict
            Structured data for chart creation with keys:
            - 'tests': list of test dictionaries
            - 'categories': dict grouping tests by category
            - 'summary': overall statistics

        Examples
        --------
        >>> data = transformer.transform(validation_report)
        >>> print(data.keys())
        dict_keys(['tests', 'categories', 'summary', 'model_info'])
        """
        # Extract test results
        tests = self._extract_tests(validation_report)

        # Group by category
        categories = self._group_by_category(tests)

        # Compute summary statistics
        summary = self._compute_summary(tests)

        # Extract model information
        model_info = self._extract_model_info(validation_report)

        return {
            'tests': tests,
            'categories': categories,
            'summary': summary,
            'model_info': model_info
        }

    def _extract_tests(self, validation_report: Any) -> List[Dict[str, Any]]:
        """
        Extract individual test results.

        Parameters
        ----------
        validation_report : ValidationReport
            Validation report object

        Returns
        -------
        list of dict
            List of test dictionaries with standardized fields
        """
        tests = []

        # Extract from specification tests
        if hasattr(validation_report, 'specification_tests'):
            for name, result in validation_report.specification_tests.items():
                tests.append(self._format_test_result(name, result, 'Specification'))

        # Extract from serial correlation tests
        if hasattr(validation_report, 'serial_tests'):
            for name, result in validation_report.serial_tests.items():
                tests.append(self._format_test_result(name, result, 'Serial Correlation'))

        # Extract from heteroskedasticity tests
        if hasattr(validation_report, 'het_tests'):
            for name, result in validation_report.het_tests.items():
                tests.append(self._format_test_result(name, result, 'Heteroskedasticity'))

        # Extract from cross-sectional dependence tests
        if hasattr(validation_report, 'cd_tests'):
            for name, result in validation_report.cd_tests.items():
                tests.append(self._format_test_result(name, result, 'Cross-Sectional Dependence'))

        return tests

    def _format_test_result(self, name: str, result: Any, category: str) -> Dict[str, Any]:
        """
        Format a single test result into standardized dictionary.

        Parameters
        ----------
        name : str
            Test name
        result : Any
            Test result object (ValidationTestResult or HausmanTestResult)
        category : str
            Test category

        Returns
        -------
        dict
            Standardized test result dictionary
        """
        test_dict = {
            'name': name,
            'category': category,
            'statistic': getattr(result, 'statistic', None),
            'pvalue': getattr(result, 'pvalue', None),
            'df': getattr(result, 'df', None),
            'conclusion': getattr(result, 'conclusion', ''),
            'passed': not getattr(result, 'reject_null', True),  # Inverted logic
        }

        # Add alpha if available
        if hasattr(result, 'alpha'):
            test_dict['alpha'] = result.alpha

        # Add metadata if available
        if hasattr(result, 'metadata'):
            test_dict['metadata'] = result.metadata

        return test_dict

    def _group_by_category(self, tests: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group tests by category.

        Parameters
        ----------
        tests : list of dict
            List of test dictionaries

        Returns
        -------
        dict
            Dictionary mapping category names to lists of tests
        """
        categories = {}

        for test in tests:
            category = test['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(test)

        return categories

    def _compute_summary(self, tests: List[Dict]) -> Dict[str, Any]:
        """
        Compute summary statistics.

        Parameters
        ----------
        tests : list of dict
            List of test dictionaries

        Returns
        -------
        dict
            Summary statistics
        """
        if not tests:
            return {
                'total_tests': 0,
                'passed': 0,
                'failed': 0,
                'pass_rate': 0.0
            }

        total = len(tests)
        passed = sum(1 for t in tests if t.get('passed', False))
        failed = total - passed
        pass_rate = 100.0 * passed / total if total > 0 else 0.0

        return {
            'total_tests': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': pass_rate
        }

    def _extract_model_info(self, validation_report: Any) -> Dict[str, Any]:
        """
        Extract model information.

        Parameters
        ----------
        validation_report : ValidationReport
            Validation report object

        Returns
        -------
        dict
            Model information
        """
        model_info = {}

        if hasattr(validation_report, 'model_info'):
            model_info = validation_report.model_info.copy()

        return model_info
