"""
Validation Report Transformer.

Transforms ValidationReport objects into template-friendly data structures.
"""

from typing import Any, Dict, List


class ValidationTransformer:
    """
    Transforms ValidationReport into data suitable for HTML templates.

    Converts ValidationReport objects into structured dictionaries with
    all necessary data for rendering interactive and static reports.

    Parameters
    ----------
    validation_report : ValidationReport
        The validation report to transform

    Examples
    --------
    >>> from panelbox.validation import ValidationReport
    >>> report = ValidationReport(model_info={...}, specification_tests={...})
    >>> transformer = ValidationTransformer(report)
    >>> data = transformer.transform()
    >>> # Use data in report generation
    >>> report_mgr.generate_validation_report(data)
    """

    def __init__(self, validation_report):
        """Initialize transformer with validation report."""
        self.report = validation_report

    def transform(self, include_charts: bool = True, use_new_visualization: bool = True) -> Dict[str, Any]:
        """
        Transform validation report into template data.

        Parameters
        ----------
        include_charts : bool, default=True
            Include chart data for interactive reports
        use_new_visualization : bool, default=True
            Use new visualization system (PanelBox 0.5.0+).
            If True, returns pre-rendered chart HTML.
            If False, returns raw chart data (legacy mode).

        Returns
        -------
        dict
            Complete data structure for template rendering

        Examples
        --------
        >>> data = transformer.transform(include_charts=True)
        >>> print(data.keys())
        dict_keys(['model_info', 'tests', 'summary', 'recommendations', 'charts'])

        >>> # Use new visualization system (default)
        >>> data = transformer.transform(use_new_visualization=True)
        >>> # data['charts']['test_overview'] is pre-rendered HTML

        >>> # Use legacy mode for backward compatibility
        >>> data = transformer.transform(use_new_visualization=False)
        >>> # data['charts']['test_overview'] is raw data dict
        """
        data = {
            "model_info": self._transform_model_info(),
            "tests": self._transform_tests(),
            "summary": self._compute_summary(),
            "recommendations": self._generate_recommendations(),
        }

        if include_charts:
            data["charts"] = self._prepare_chart_data(use_new_visualization=use_new_visualization)

        return data

    def _transform_model_info(self) -> Dict[str, Any]:
        """
        Transform model information.

        Returns
        -------
        dict
            Model information for template
        """
        info = self.report.model_info.copy()

        # Add formatted strings
        if "nobs" in info:
            info["nobs_formatted"] = f"{info['nobs']:,}"

        if "n_entities" in info:
            info["n_entities_formatted"] = f"{info['n_entities']:,}"

        if "n_periods" in info and info.get("n_periods"):
            info["n_periods_formatted"] = f"{info['n_periods']}"

        return info

    def _transform_tests(self) -> List[Dict[str, Any]]:
        """
        Transform test results into table-ready format.

        Returns
        -------
        list of dict
            List of test results for template tables
        """
        tests = []

        # Process each category
        for category_name, category_tests in [
            ("Specification", self.report.specification_tests),
            ("Serial Correlation", self.report.serial_tests),
            ("Heteroskedasticity", self.report.het_tests),
            ("Cross-Sectional Dependence", self.report.cd_tests),
        ]:
            for test_name, test_result in category_tests.items():
                test_data = {
                    "category": category_name,
                    "name": test_name,
                    "statistic": test_result.statistic,
                    "statistic_formatted": f"{test_result.statistic:.3f}",
                    "pvalue": test_result.pvalue,
                    "pvalue_formatted": self._format_pvalue(test_result.pvalue),
                    "df": test_result.df,
                    "reject_null": test_result.reject_null,
                    "result": "REJECT" if test_result.reject_null else "ACCEPT",
                    "result_class": "reject" if test_result.reject_null else "accept",
                    "conclusion": test_result.conclusion,
                    "significance": self._get_significance_stars(test_result.pvalue),
                    "metadata": test_result.metadata or {},
                }

                tests.append(test_data)

        return tests

    def _compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics.

        Returns
        -------
        dict
            Summary statistics for dashboard
        """
        # Count tests
        total_tests = (
            len(self.report.specification_tests)
            + len(self.report.serial_tests)
            + len(self.report.het_tests)
            + len(self.report.cd_tests)
        )

        # Count failures by category
        failed_by_category = {
            "specification": 0,
            "serial": 0,
            "heteroskedasticity": 0,
            "cross_sectional": 0,
        }

        total_failed = 0

        for test_result in self.report.specification_tests.values():
            if test_result.reject_null:
                failed_by_category["specification"] += 1
                total_failed += 1

        for test_result in self.report.serial_tests.values():
            if test_result.reject_null:
                failed_by_category["serial"] += 1
                total_failed += 1

        for test_result in self.report.het_tests.values():
            if test_result.reject_null:
                failed_by_category["heteroskedasticity"] += 1
                total_failed += 1

        for test_result in self.report.cd_tests.values():
            if test_result.reject_null:
                failed_by_category["cross_sectional"] += 1
                total_failed += 1

        total_passed = total_tests - total_failed

        # Calculate pass rate
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        # Overall status
        if total_failed == 0:
            overall_status = "excellent"
            status_message = "All tests passed"
        elif total_failed <= 2:
            overall_status = "good"
            status_message = "Minor issues detected"
        elif total_failed <= 4:
            overall_status = "warning"
            status_message = "Several issues detected"
        else:
            overall_status = "critical"
            status_message = "Multiple issues detected"

        return {
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "pass_rate": pass_rate,
            "pass_rate_formatted": f"{pass_rate:.1f}%",
            "failed_by_category": failed_by_category,
            "overall_status": overall_status,
            "status_message": status_message,
            "has_issues": total_failed > 0,
        }

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate recommendations based on failed tests.

        Returns
        -------
        list of dict
            List of recommendations
        """
        recommendations = []

        # Check serial correlation
        serial_issues = [
            name for name, result in self.report.serial_tests.items() if result.reject_null
        ]

        if serial_issues:
            recommendations.append(
                {
                    "category": "Serial Correlation",
                    "severity": "high",
                    "issue": f"Detected serial correlation in {len(serial_issues)} test(s)",
                    "tests": serial_issues,
                    "suggestions": [
                        "Use clustered standard errors at the entity level",
                        "Consider HAC (Heteroskedasticity and Autocorrelation Consistent) errors",
                        "Add lagged dependent variable if appropriate",
                        "Review model dynamics and time structure",
                    ],
                }
            )

        # Check heteroskedasticity
        het_issues = [name for name, result in self.report.het_tests.items() if result.reject_null]

        if het_issues:
            recommendations.append(
                {
                    "category": "Heteroskedasticity",
                    "severity": "medium",
                    "issue": f"Detected heteroskedasticity in {len(het_issues)} test(s)",
                    "tests": het_issues,
                    "suggestions": [
                        "Use robust (White) standard errors",
                        "Consider weighted least squares (WLS)",
                        "Apply log transformation to dependent variable",
                        "Check for outliers and influential observations",
                    ],
                }
            )

        # Check cross-sectional dependence
        cd_issues = [name for name, result in self.report.cd_tests.items() if result.reject_null]

        if cd_issues:
            recommendations.append(
                {
                    "category": "Cross-Sectional Dependence",
                    "severity": "high",
                    "issue": f"Detected cross-sectional dependence in {len(cd_issues)} test(s)",
                    "tests": cd_issues,
                    "suggestions": [
                        "Use Driscoll-Kraay standard errors",
                        "Consider spatial econometric models if geographic data",
                        "Add time fixed effects to control common shocks",
                        "Use bootstrap methods robust to cross-sectional dependence",
                    ],
                }
            )

        # Check specification
        spec_issues = [
            name for name, result in self.report.specification_tests.items() if result.reject_null
        ]

        if spec_issues:
            recommendations.append(
                {
                    "category": "Model Specification",
                    "severity": "critical",
                    "issue": f"Specification concerns in {len(spec_issues)} test(s)",
                    "tests": spec_issues,
                    "suggestions": [
                        "Review model specification (Fixed vs Random Effects)",
                        "Consider alternative estimators",
                        "Add or remove control variables",
                        "Test for omitted variable bias",
                    ],
                }
            )

        # Sort by severity
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        recommendations.sort(key=lambda r: severity_order.get(r["severity"], 99))

        return recommendations

    def _prepare_chart_data(self, use_new_visualization: bool = True) -> Dict[str, Any]:
        """
        Prepare chart data or generate pre-rendered charts.

        Parameters
        ----------
        use_new_visualization : bool, default=True
            If True, use new visualization system and return HTML charts.
            If False, return raw data dicts (legacy mode for backward compatibility).

        Returns
        -------
        dict
            If use_new_visualization=True: {'test_overview': '<div>...</div>', ...}
            If use_new_visualization=False: {'test_overview': {data dict}, ...}

        Notes
        -----
        The new visualization system (use_new_visualization=True) uses the
        panelbox.visualization module to generate professional, themed charts.

        Legacy mode (use_new_visualization=False) returns raw data dicts for
        backward compatibility with old templates that use inline JavaScript.
        """
        if not use_new_visualization:
            # Legacy mode - return raw data dicts
            return self._prepare_chart_data_legacy()

        # NEW: Use visualization system
        try:
            from panelbox.visualization import create_validation_charts
        except ImportError:
            # Fallback if visualization module not available
            import warnings
            warnings.warn(
                "New visualization system not available. Falling back to legacy mode. "
                "Install panelbox.visualization or set use_new_visualization=False.",
                UserWarning
            )
            return self._prepare_chart_data_legacy()

        # Use the already-existing prepare_visualization_data() method
        viz_data = self.prepare_visualization_data()

        # Generate charts using new visualization system
        try:
            chart_objects = create_validation_charts(
                validation_data=viz_data,
                theme='professional',
                interactive=True,
                charts=['test_overview', 'pvalue_distribution', 'test_statistics'],
                include_html=False  # Get chart objects, not HTML strings yet
            )

            # Convert chart objects to HTML (div only, no full document)
            charts_html = {}
            for name, chart_obj in chart_objects.items():
                # Get just the div, not a full HTML document
                # include_plotlyjs=False to avoid duplicate Plotly library
                # full_html=False to get just the div
                charts_html[name] = chart_obj.to_html(
                    include_plotlyjs=False,
                    full_html=False,
                    div_id=f"chart-{name}"
                )

            return charts_html

        except Exception as e:
            # Fallback to legacy mode if chart generation fails
            import warnings
            warnings.warn(
                f"Failed to generate charts with new visualization system: {e}. "
                "Falling back to legacy mode.",
                UserWarning
            )
            return self._prepare_chart_data_legacy()

    def _prepare_chart_data_legacy(self) -> Dict[str, Any]:
        """
        Prepare raw chart data (legacy mode).

        This method is kept for backward compatibility with old templates
        that use inline JavaScript to render charts client-side.

        Returns
        -------
        dict
            Chart data dictionaries for manual Plotly rendering

        Notes
        -----
        This is the original implementation of _prepare_chart_data().
        It returns raw data that templates can pass to Plotly.newPlot().
        """
        charts = {}

        # 1. Test Results Overview (Bar Chart)
        test_categories = []
        passed_counts = []
        failed_counts = []

        for category_name, category_tests in [
            ("Specification", self.report.specification_tests),
            ("Serial Correlation", self.report.serial_tests),
            ("Heteroskedasticity", self.report.het_tests),
            ("Cross-Sectional Dep.", self.report.cd_tests),
        ]:
            if not category_tests:
                continue

            test_categories.append(category_name)

            passed = sum(1 for t in category_tests.values() if not t.reject_null)
            failed = sum(1 for t in category_tests.values() if t.reject_null)

            passed_counts.append(passed)
            failed_counts.append(failed)

        charts["test_overview"] = {
            "categories": test_categories,
            "passed": passed_counts,
            "failed": failed_counts,
        }

        # 2. P-value Distribution
        all_pvalues = []
        all_test_names = []

        for test_dict in [
            self.report.specification_tests,
            self.report.serial_tests,
            self.report.het_tests,
            self.report.cd_tests,
        ]:
            for name, result in test_dict.items():
                all_test_names.append(name)
                all_pvalues.append(result.pvalue)

        charts["pvalue_distribution"] = {"test_names": all_test_names, "pvalues": all_pvalues}

        # 3. Statistics by Test
        test_stats = []
        test_labels = []
        test_categories_list = []

        for category_name, test_dict in [
            ("Specification", self.report.specification_tests),
            ("Serial Correlation", self.report.serial_tests),
            ("Heteroskedasticity", self.report.het_tests),
            ("Cross-Sectional Dep.", self.report.cd_tests),
        ]:
            for name, result in test_dict.items():
                test_labels.append(name)
                test_stats.append(result.statistic)
                test_categories_list.append(category_name)

        charts["test_statistics"] = {
            "test_names": test_labels,
            "statistics": test_stats,
            "categories": test_categories_list,  # Added for new visualization system
            "pvalues": all_pvalues  # Added for size scaling in scatter plot
        }

        return charts

    def prepare_visualization_data(self) -> Dict[str, Any]:
        """
        Prepare data specifically for the new visualization system.

        This method creates a comprehensive data structure that can be
        passed directly to create_validation_charts() from panelbox.visualization.

        Returns
        -------
        dict
            Data structure compatible with panelbox.visualization API

        Examples
        --------
        >>> from panelbox.visualization import create_validation_charts
        >>> transformer = ValidationTransformer(validation_report)
        >>> viz_data = transformer.prepare_visualization_data()
        >>> charts = create_validation_charts(viz_data, theme='professional')

        Notes
        -----
        This is the recommended method for integrating with the new
        visualization system introduced in PanelBox 0.5.0+.
        """
        # Prepare test list with full metadata
        tests = []

        for category_name, test_dict in [
            ("Specification", self.report.specification_tests),
            ("Serial Correlation", self.report.serial_tests),
            ("Heteroskedasticity", self.report.het_tests),
            ("Cross-Sectional Dependence", self.report.cd_tests),
        ]:
            for name, result in test_dict.items():
                tests.append({
                    'name': name,
                    'category': category_name,
                    'statistic': result.statistic,
                    'pvalue': result.pvalue,
                    'df': result.df if hasattr(result, 'df') else None,
                    'conclusion': result.conclusion if hasattr(result, 'conclusion') else '',
                    'passed': not result.reject_null,
                    'alpha': result.alpha if hasattr(result, 'alpha') else 0.05,
                    'metadata': result.metadata if hasattr(result, 'metadata') else {}
                })

        return {
            'tests': tests,
            'model_info': self.report.model_info,
            'charts': self._prepare_chart_data_legacy()  # For backward compatibility - use legacy to avoid recursion
        }

    @staticmethod
    def _format_pvalue(pvalue: float) -> str:
        """
        Format p-value for display.

        Parameters
        ----------
        pvalue : float
            P-value to format

        Returns
        -------
        str
            Formatted p-value
        """
        if pvalue < 0.001:
            return f"{pvalue:.2e}"
        return f"{pvalue:.4f}"

    @staticmethod
    def _get_significance_stars(pvalue: float) -> str:
        """
        Get significance stars based on p-value.

        Parameters
        ----------
        pvalue : float
            P-value

        Returns
        -------
        str
            Significance stars
        """
        if pvalue < 0.001:
            return "***"
        elif pvalue < 0.01:
            return "**"
        elif pvalue < 0.05:
            return "*"
        elif pvalue < 0.1:
            return "."
        return ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary (alias for transform).

        Returns
        -------
        dict
            Complete data structure
        """
        return self.transform()

    def __repr__(self) -> str:
        """String representation."""
        n_tests = (
            len(self.report.specification_tests)
            + len(self.report.serial_tests)
            + len(self.report.het_tests)
            + len(self.report.cd_tests)
        )

        return f"ValidationTransformer(tests={n_tests})"
