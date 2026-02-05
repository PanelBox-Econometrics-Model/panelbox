"""
Validation report container.
"""

from typing import Dict, List, Optional

from panelbox.validation.base import ValidationTestResult


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
        model_info: Dict[str, any],
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

    def summary(self, verbose: bool = True) -> str:
        """
        Generate formatted summary of all validation tests.

        Parameters
        ----------
        verbose : bool, default=True
            If True, include full details of each test
            If False, show only summary table

        Returns
        -------
        str
            Formatted validation report
        """
        lines = []
        lines.append("=" * 78)
        lines.append("MODEL VALIDATION REPORT")
        lines.append("=" * 78)
        lines.append("")

        # Model information
        lines.append("Model Information:")
        lines.append(f"  Type:    {self.model_info.get('model_type', 'Unknown')}")
        lines.append(f"  Formula: {self.model_info.get('formula', 'Unknown')}")
        lines.append(f"  N obs:   {self.model_info.get('nobs', 'Unknown')}")
        lines.append(f"  N entities: {self.model_info.get('n_entities', 'Unknown')}")
        lines.append("")

        # Summary table
        lines.append("=" * 78)
        lines.append("VALIDATION TESTS SUMMARY")
        lines.append("=" * 78)
        lines.append(f"{'Test':<35} {'Statistic':<12} {'P-value':<10} {'Result':<10}")
        lines.append("-" * 78)

        # Helper function to add test row
        def add_test_row(test_name, test_result):
            if test_result is None:
                return

            stat_str = f"{test_result.statistic:.3f}"
            pval_str = f"{test_result.pvalue:.4f}"
            result = "REJECT" if test_result.reject_null else "OK"

            lines.append(f"{test_name:<35} {stat_str:<12} {pval_str:<10} {result:<10}")

        # Specification tests
        if self.specification_tests:
            lines.append("")
            lines.append("Specification Tests:")
            for name, result in self.specification_tests.items():
                add_test_row(f"  {name}", result)

        # Serial correlation tests
        if self.serial_tests:
            lines.append("")
            lines.append("Serial Correlation Tests:")
            for name, result in self.serial_tests.items():
                add_test_row(f"  {name}", result)

        # Heteroskedasticity tests
        if self.het_tests:
            lines.append("")
            lines.append("Heteroskedasticity Tests:")
            for name, result in self.het_tests.items():
                add_test_row(f"  {name}", result)

        # Cross-sectional dependence tests
        if self.cd_tests:
            lines.append("")
            lines.append("Cross-Sectional Dependence Tests:")
            for name, result in self.cd_tests.items():
                add_test_row(f"  {name}", result)

        lines.append("=" * 78)
        lines.append("")

        # Diagnostics summary
        problems = []

        for category, tests in [
            ("specification", self.specification_tests),
            ("serial correlation", self.serial_tests),
            ("heteroskedasticity", self.het_tests),
            ("cross-sectional dependence", self.cd_tests),
        ]:
            for name, result in tests.items():
                if result.reject_null:
                    problems.append(f"  - {name}: {category}")

        if problems:
            lines.append("⚠️  POTENTIAL ISSUES DETECTED:")
            lines.extend(problems)
            lines.append("")
            lines.append("Consider:")

            if any("serial correlation" in p for p in problems):
                lines.append("  • Use clustered standard errors or HAC errors")

            if any("heteroskedasticity" in p for p in problems):
                lines.append("  • Use robust standard errors")

            if any("cross-sectional dependence" in p for p in problems):
                lines.append("  • Use Driscoll-Kraay standard errors")

            if any("specification" in p for p in problems):
                lines.append("  • Review model specification")
        else:
            lines.append("✓ No major issues detected in validation tests")

        lines.append("")
        lines.append("=" * 78)
        lines.append("")

        # Verbose output: detailed results
        if verbose and (
            self.specification_tests or self.serial_tests or self.het_tests or self.cd_tests
        ):
            lines.append("")
            lines.append("DETAILED TEST RESULTS")
            lines.append("=" * 78)
            lines.append("")

            for category, tests in [
                ("SPECIFICATION TESTS", self.specification_tests),
                ("SERIAL CORRELATION TESTS", self.serial_tests),
                ("HETEROSKEDASTICITY TESTS", self.het_tests),
                ("CROSS-SECTIONAL DEPENDENCE TESTS", self.cd_tests),
            ]:
                if tests:
                    lines.append("")
                    lines.append(category)
                    lines.append("-" * 78)
                    for name, result in tests.items():
                        lines.append("")
                        lines.append(result.summary())

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
