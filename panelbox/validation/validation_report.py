"""
Validation report container.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from panelbox.validation.base import ValidationTestResult

logger = logging.getLogger(__name__)


class ValidationReport:
    """
    Container for validation test results.

    Attributes
    ----------
    model_info : dict
        Information about the model being validated
    specification_tests : dict
        Results of specification tests (Hausman, Mundlak, etc.)
    serial_tests : dict
        Results of serial correlation tests
    het_tests : dict
        Results of heteroskedasticity tests
    cd_tests : dict
        Results of cross-sectional dependence tests
    """

    def __init__(
        self,
        model_info: Dict[str, Any],
        specification_tests: Optional[Dict[str, ValidationTestResult]] = None,
        serial_tests: Optional[Dict[str, ValidationTestResult]] = None,
        het_tests: Optional[Dict[str, ValidationTestResult]] = None,
        cd_tests: Optional[Dict[str, ValidationTestResult]] = None,
    ):
        self.model_info = model_info
        self.specification_tests = specification_tests or {}
        self.serial_tests = serial_tests or {}
        self.het_tests = het_tests or {}
        self.cd_tests = cd_tests or {}

    def __str__(self) -> str:
        """String representation."""
        return self.summary()

    def __repr__(self) -> str:
        """Repr."""
        n_tests = (
            len(self.specification_tests)
            + len(self.serial_tests)
            + len(self.het_tests)
            + len(self.cd_tests)
        )
        return f"ValidationReport(model='{self.model_info.get('model_type')}', tests={n_tests})"

    def _test_categories(self):
        """Return the ordered list of (category_key, label, tests) tuples."""
        return [
            ("specification", "Specification", self.specification_tests),
            ("serial correlation", "Serial Correlation", self.serial_tests),
            ("heteroskedasticity", "Heteroskedasticity", self.het_tests),
            ("cross-sectional dependence", "Cross-Sectional Dep.", self.cd_tests),
        ]

    def _summary_as_dataframe(self):
        """Return summary as a pandas DataFrame."""
        import pandas as pd

        rows = []
        for _key, label, tests in self._test_categories():
            for name, result in tests.items():
                rows.append(
                    {
                        "category": label,
                        "test": name,
                        "statistic": result.statistic,
                        "pvalue": result.pvalue,
                        "reject": result.reject_null,
                        "conclusion": result.conclusion,
                    }
                )
        return pd.DataFrame(rows)

    def _build_header_lines(self) -> list[str]:
        """Build model information header."""
        return [
            "=" * 78,
            "MODEL VALIDATION REPORT",
            "=" * 78,
            "",
            "Model Information:",
            f"  Type:    {self.model_info.get('model_type', 'Unknown')}",
            f"  Formula: {self.model_info.get('formula', 'Unknown')}",
            f"  N obs:   {self.model_info.get('nobs', 'Unknown')}",
            f"  N entities: {self.model_info.get('n_entities', 'Unknown')}",
            "",
        ]

    def _build_summary_table(self) -> list[str]:
        """Build the summary table of all tests."""
        lines = [
            "=" * 78,
            "VALIDATION TESTS SUMMARY",
            "=" * 78,
            f"{'Test':<35} {'Statistic':<12} {'P-value':<10} {'Result':<10}",
            "-" * 78,
        ]

        section_labels = {
            "specification": "Specification Tests:",
            "serial correlation": "Serial Correlation Tests:",
            "heteroskedasticity": "Heteroskedasticity Tests:",
            "cross-sectional dependence": "Cross-Sectional Dependence Tests:",
        }
        for key, _label, tests in self._test_categories():
            if tests:
                lines.append("")
                lines.append(section_labels[key])
                for name, result in tests.items():
                    stat_str = f"{result.statistic:.3f}"
                    pval_str = f"{result.pvalue:.4f}"
                    verdict = "REJECT" if result.reject_null else "OK"
                    lines.append(f"  {name:<33} {stat_str:<12} {pval_str:<10} {verdict:<10}")

        lines.extend(["=" * 78, ""])
        return lines

    def _build_diagnostics(self) -> list[str]:
        """Build the diagnostics / recommendations section."""
        problems = []
        for key, _label, tests in self._test_categories():
            for name, result in tests.items():
                if result.reject_null:
                    problems.append((key, f"  - {name}: {key}"))

        lines = []
        if problems:
            lines.append("\u26a0\ufe0f  POTENTIAL ISSUES DETECTED:")
            lines.extend(p for _, p in problems)
            lines.append("")
            lines.append("Consider:")
            categories_with_issues = {cat for cat, _ in problems}
            recommendations = {
                "serial correlation": "  \u2022 Use clustered standard errors or HAC errors",
                "heteroskedasticity": "  \u2022 Use robust standard errors",
                "cross-sectional dependence": "  \u2022 Use Driscoll-Kraay standard errors",
                "specification": "  \u2022 Review model specification",
            }
            for cat, rec in recommendations.items():
                if cat in categories_with_issues:
                    lines.append(rec)
        else:
            lines.append("\u2713 No major issues detected in validation tests")

        lines.extend(["", "=" * 78, ""])
        return lines

    def _build_detailed_results(self) -> list[str]:
        """Build verbose detailed results section."""
        verbose_labels = {
            "specification": "SPECIFICATION TESTS",
            "serial correlation": "SERIAL CORRELATION TESTS",
            "heteroskedasticity": "HETEROSKEDASTICITY TESTS",
            "cross-sectional dependence": "CROSS-SECTIONAL DEPENDENCE TESTS",
        }
        lines = ["", "DETAILED TEST RESULTS", "=" * 78, ""]
        for key, _label, tests in self._test_categories():
            if tests:
                lines.append("")
                lines.append(verbose_labels[key])
                lines.append("-" * 78)
                for _name, result in tests.items():
                    lines.append("")
                    lines.append(result.summary())
        return lines

    def summary(self, verbose: bool = True, as_dataframe: bool = False):
        """
        Generate formatted summary of all validation tests.

        Parameters
        ----------
        verbose : bool, default=True
            If True, include full details of each test
            If False, show only summary table
        as_dataframe : bool, default=False
            If True, return a pandas DataFrame with one row per test instead
            of a formatted string.  The DataFrame has columns:
            ``category``, ``test``, ``statistic``, ``pvalue``, ``reject``

        Returns
        -------
        str or pandas.DataFrame
            Formatted validation report (str) or summary table (DataFrame)
        """
        if as_dataframe:
            return self._summary_as_dataframe()

        lines = self._build_header_lines()
        lines.extend(self._build_summary_table())
        lines.extend(self._build_diagnostics())

        has_tests = self.specification_tests or self.serial_tests or self.het_tests or self.cd_tests
        if verbose and has_tests:
            lines.extend(self._build_detailed_results())

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """
        Export validation report to dictionary.

        Returns
        -------
        dict
            Dictionary with all test results
        """
        result = {
            "model_info": self.model_info,
            "specification_tests": {},
            "serial_tests": {},
            "het_tests": {},
            "cd_tests": {},
        }

        # Helper to convert test result to dict
        def test_to_dict(test):
            """Convert a test result to a dictionary representation."""
            return {
                "statistic": test.statistic,
                "pvalue": test.pvalue,
                "df": test.df,
                "reject_null": test.reject_null,
                "conclusion": test.conclusion,
                "metadata": test.metadata,
            }

        for name, test in self.specification_tests.items():
            result["specification_tests"][name] = test_to_dict(test)

        for name, test in self.serial_tests.items():
            result["serial_tests"][name] = test_to_dict(test)

        for name, test in self.het_tests.items():
            result["het_tests"][name] = test_to_dict(test)

        for name, test in self.cd_tests.items():
            result["cd_tests"][name] = test_to_dict(test)

        return result

    def get_failed_tests(self) -> List[str]:
        """
        Get list of tests that rejected the null hypothesis.

        Returns
        -------
        list
            Names of tests that detected issues
        """
        failed = []

        for category, tests in [
            ("spec", self.specification_tests),
            ("serial", self.serial_tests),
            ("het", self.het_tests),
            ("cd", self.cd_tests),
        ]:
            for name, result in tests.items():
                if result.reject_null:
                    failed.append(f"{category}/{name}")

        return failed
