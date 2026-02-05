"""
Validation suite for panel models.
"""

import warnings
from typing import Dict, List, Union

from panelbox.core.results import PanelResults
from panelbox.validation.base import ValidationTestResult
from panelbox.validation.cross_sectional_dependence.breusch_pagan_lm import BreuschPaganLMTest
from panelbox.validation.cross_sectional_dependence.frees import FreesTest
from panelbox.validation.cross_sectional_dependence.pesaran_cd import PesaranCDTest
from panelbox.validation.heteroskedasticity.breusch_pagan import BreuschPaganTest
from panelbox.validation.heteroskedasticity.modified_wald import ModifiedWaldTest
from panelbox.validation.heteroskedasticity.white import WhiteTest
from panelbox.validation.serial_correlation.baltagi_wu import BaltagiWuTest
from panelbox.validation.serial_correlation.breusch_godfrey import BreuschGodfreyTest
from panelbox.validation.serial_correlation.wooldridge_ar import WooldridgeARTest

# Import tests
from panelbox.validation.specification.mundlak import MundlakTest
from panelbox.validation.specification.reset import RESETTest
from panelbox.validation.validation_report import ValidationReport


class ValidationSuite:
    """
    Suite of validation tests for panel models.

    This class provides a unified interface to run various diagnostic tests
    on panel model results.

    Parameters
    ----------
    results : PanelResults
        Results from panel model estimation

    Examples
    --------
    >>> from panelbox.models.static.fixed_effects import FixedEffects
    >>> fe = FixedEffects("y ~ x1 + x2", data, "entity", "time")
    >>> results = fe.fit()
    >>>
    >>> from panelbox.validation.validation_suite import ValidationSuite
    >>> suite = ValidationSuite(results)
    >>> report = suite.run(tests='all')
    >>> print(report)
    """

    def __init__(self, results: PanelResults):
        """
        Initialize validation suite.

        Parameters
        ----------
        results : PanelResults
            Results from panel model estimation
        """
        self.results = results
        self.model_type = results.model_type

    def run(
        self, tests: Union[str, List[str]] = "default", alpha: float = 0.05, verbose: bool = False
    ) -> ValidationReport:
        """
        Run validation tests.

        Parameters
        ----------
        tests : str or list of str, default='default'
            Which tests to run:
            - 'all': Run all available tests
            - 'default': Run recommended tests for this model type
            - 'serial': Serial correlation tests only
            - 'het': Heteroskedasticity tests only
            - 'cd': Cross-sectional dependence tests only
            - List of test names
        alpha : float, default=0.05
            Significance level for tests
        verbose : bool, default=False
            If True, print progress during testing

        Returns
        -------
        ValidationReport
            Report containing all test results
        """
        # Determine which tests to run
        tests_to_run = self._determine_tests(tests)

        # Run tests by category
        specification_tests = {}
        serial_tests = {}
        het_tests = {}
        cd_tests = {}

        # Specification tests
        if "specification" in tests_to_run:
            specification_tests = self.run_specification_tests(alpha, verbose)

        # Serial correlation tests
        if "serial" in tests_to_run:
            serial_tests = self.run_serial_correlation_tests(alpha, verbose)

        # Heteroskedasticity tests
        if "het" in tests_to_run:
            het_tests = self.run_heteroskedasticity_tests(alpha, verbose)

        # Cross-sectional dependence tests
        if "cd" in tests_to_run:
            cd_tests = self.run_cross_sectional_tests(alpha, verbose)

        # Create validation report
        model_info = {
            "model_type": self.model_type,
            "formula": self.results.formula,
            "nobs": self.results.nobs,
            "n_entities": self.results.n_entities,
            "n_periods": self.results.n_periods,
        }

        report = ValidationReport(
            model_info=model_info,
            specification_tests=specification_tests,
            serial_tests=serial_tests,
            het_tests=het_tests,
            cd_tests=cd_tests,
        )

        return report

    def run_specification_tests(
        self, alpha: float = 0.05, verbose: bool = False
    ) -> Dict[str, ValidationTestResult]:
        """
        Run specification tests.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        verbose : bool, default=False
            Print progress

        Returns
        -------
        dict
            Dictionary of test results
        """
        results = {}

        # Note: Hausman test requires two model results (FE and RE)
        # It cannot be run from a single results object
        # Users should run it separately

        # Mundlak test (for RE models)
        if "Random Effects" in self.model_type:
            try:
                if verbose:
                    print("Running Mundlak test...")
                test = MundlakTest(self.results)
                results["Mundlak"] = test.run(alpha)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Mundlak test failed: {e}")
                warnings.warn(f"Mundlak test failed: {e}")

        # RESET test (for all models)
        try:
            if verbose:
                print("Running RESET test...")
            test = RESETTest(self.results)
            results["RESET"] = test.run(alpha=alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: RESET test failed: {e}")
            warnings.warn(f"RESET test failed: {e}")

        # Chow test (for all models)
        # Note: Requires break_point parameter, skip by default
        # Users can run separately with specific break point

        return results

    def run_serial_correlation_tests(
        self, alpha: float = 0.05, verbose: bool = False
    ) -> Dict[str, ValidationTestResult]:
        """
        Run serial correlation tests.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        verbose : bool, default=False
            Print progress

        Returns
        -------
        dict
            Dictionary of test results
        """
        results = {}

        # Wooldridge test (for FE models)
        if "Fixed Effects" in self.model_type:
            try:
                if verbose:
                    print("Running Wooldridge AR test...")
                test = WooldridgeARTest(self.results)
                results["Wooldridge"] = test.run(alpha)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Wooldridge test failed: {e}")
                warnings.warn(f"Wooldridge test failed: {e}")

        # Breusch-Godfrey test (for all models)
        try:
            if verbose:
                print("Running Breusch-Godfrey test...")
            test = BreuschGodfreyTest(self.results)
            results["Breusch-Godfrey"] = test.run(lags=1, alpha=alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: Breusch-Godfrey test failed: {e}")
            warnings.warn(f"Breusch-Godfrey test failed: {e}")

        # Baltagi-Wu LBI test (for all models, especially unbalanced panels)
        try:
            if verbose:
                print("Running Baltagi-Wu LBI test...")
            test = BaltagiWuTest(self.results)
            results["Baltagi-Wu"] = test.run(alpha=alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: Baltagi-Wu test failed: {e}")
            warnings.warn(f"Baltagi-Wu test failed: {e}")

        return results

    def run_heteroskedasticity_tests(
        self, alpha: float = 0.05, verbose: bool = False
    ) -> Dict[str, ValidationTestResult]:
        """
        Run heteroskedasticity tests.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        verbose : bool, default=False
            Print progress

        Returns
        -------
        dict
            Dictionary of test results
        """
        results = {}

        # Modified Wald test (for FE models)
        if "Fixed Effects" in self.model_type:
            try:
                if verbose:
                    print("Running Modified Wald test...")
                test = ModifiedWaldTest(self.results)
                results["Modified Wald"] = test.run(alpha)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Modified Wald test failed: {e}")
                warnings.warn(f"Modified Wald test failed: {e}")

        # Breusch-Pagan LM test (for all models)
        try:
            if verbose:
                print("Running Breusch-Pagan test...")
            test = BreuschPaganTest(self.results)
            results["Breusch-Pagan"] = test.run(alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: Breusch-Pagan test failed: {e}")
            warnings.warn(f"Breusch-Pagan test failed: {e}")

        # White test (for all models)
        try:
            if verbose:
                print("Running White test...")
            test = WhiteTest(self.results)
            results["White"] = test.run(alpha, cross_terms=False)  # Without cross terms for speed
        except Exception as e:
            if verbose:
                print(f"  Warning: White test failed: {e}")
            warnings.warn(f"White test failed: {e}")

        return results

    def run_cross_sectional_tests(
        self, alpha: float = 0.05, verbose: bool = False
    ) -> Dict[str, ValidationTestResult]:
        """
        Run cross-sectional dependence tests.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level
        verbose : bool, default=False
            Print progress

        Returns
        -------
        dict
            Dictionary of test results
        """
        results = {}

        # Pesaran CD test (for all models, large N)
        try:
            if verbose:
                print("Running Pesaran CD test...")
            test = PesaranCDTest(self.results)
            results["Pesaran CD"] = test.run(alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: Pesaran CD test failed: {e}")
            warnings.warn(f"Pesaran CD test failed: {e}")

        # Breusch-Pagan LM test (for all models, small to moderate N)
        try:
            if verbose:
                print("Running Breusch-Pagan LM test...")
            test = BreuschPaganLMTest(self.results)
            results["Breusch-Pagan LM"] = test.run(alpha=alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: Breusch-Pagan LM test failed: {e}")
            warnings.warn(f"Breusch-Pagan LM test failed: {e}")

        # Frees test (non-parametric, robust to non-normality)
        try:
            if verbose:
                print("Running Frees test...")
            test = FreesTest(self.results)
            results["Frees"] = test.run(alpha=alpha)
        except Exception as e:
            if verbose:
                print(f"  Warning: Frees test failed: {e}")
            warnings.warn(f"Frees test failed: {e}")

        return results

    def _determine_tests(self, tests: Union[str, List[str]]) -> List[str]:
        """
        Determine which test categories to run.

        Parameters
        ----------
        tests : str or list
            Test specification

        Returns
        -------
        list
            List of test categories
        """
        if tests == "all":
            return ["specification", "serial", "het", "cd"]
        elif tests == "default":
            # Recommended tests based on model type
            if "Fixed Effects" in self.model_type:
                return ["serial", "het", "cd"]
            elif "Random Effects" in self.model_type:
                return ["cd"]
            else:  # Pooled OLS
                return ["het", "cd"]
        elif tests == "serial":
            return ["serial"]
        elif tests == "het":
            return ["het"]
        elif tests == "cd":
            return ["cd"]
        elif isinstance(tests, list):
            return tests
        else:
            raise ValueError(
                f"Invalid tests specification: {tests}. "
                "Use 'all', 'default', 'serial', 'het', 'cd', or a list of test names"
            )
