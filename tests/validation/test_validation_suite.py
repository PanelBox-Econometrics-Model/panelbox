"""
Tests for ValidationSuite class.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.validation_report import ValidationReport
from panelbox.validation.validation_suite import ValidationSuite


class MockPanelResults:
    """Mock PanelResults object for testing."""

    def __init__(self, model_type="Fixed Effects", n_entities=10, n_periods=10):
        """Initialize mock results."""
        self.model_type = model_type
        self.formula = "y ~ x1 + x2"
        self.nobs = n_entities * n_periods
        self.n_entities = n_entities
        self.n_periods = n_periods

        # Create synthetic data
        np.random.seed(42)
        self.resid = np.random.normal(0, 1, self.nobs)
        self.fittedvalues = np.random.normal(2, 0.5, self.nobs)
        self.params = np.array([1.0, 0.5, -0.3])

        # Create entity and time indices
        self.entity_index = np.repeat(np.arange(n_entities), n_periods)
        self.time_index = np.tile(np.arange(n_periods), n_entities)

        # Create data DataFrame
        self.data = pd.DataFrame(
            {
                "entity": self.entity_index,
                "time": self.time_index,
                "y": self.fittedvalues + self.resid,
                "x1": np.random.normal(0, 1, self.nobs),
                "x2": np.random.normal(0, 1, self.nobs),
            }
        )

        # Additional attributes needed by some tests
        self.exog = np.column_stack(
            [np.ones(self.nobs), self.data["x1"].values, self.data["x2"].values]
        )


class TestValidationSuiteInit:
    """Test ValidationSuite initialization."""

    def test_init_stores_results(self):
        """Test that initialization stores results."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        assert suite.results is results

    def test_init_extracts_model_type(self):
        """Test that initialization extracts model type."""
        results = MockPanelResults(model_type="Random Effects")
        suite = ValidationSuite(results)

        assert suite.model_type == "Random Effects"

    def test_init_different_model_types(self):
        """Test initialization with different model types."""
        for model_type in ["Fixed Effects", "Random Effects", "Pooled OLS"]:
            results = MockPanelResults(model_type=model_type)
            suite = ValidationSuite(results)
            assert suite.model_type == model_type


class TestValidationSuiteRun:
    """Test ValidationSuite run method."""

    def test_run_returns_validation_report(self):
        """Test that run returns ValidationReport."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="serial")
        assert isinstance(report, ValidationReport)

    def test_run_default_tests(self):
        """Test running default tests."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        report = suite.run(tests="default")
        assert isinstance(report, ValidationReport)

    def test_run_all_tests(self):
        """Test running all tests."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="all")
        assert isinstance(report, ValidationReport)

    def test_run_serial_tests_only(self):
        """Test running serial correlation tests only."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="serial")
        assert isinstance(report, ValidationReport)

    def test_run_het_tests_only(self):
        """Test running heteroskedasticity tests only."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="het")
        assert isinstance(report, ValidationReport)

    def test_run_cd_tests_only(self):
        """Test running cross-sectional dependence tests only."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="cd")
        assert isinstance(report, ValidationReport)

    def test_run_with_custom_alpha(self):
        """Test running tests with custom alpha level."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="serial", alpha=0.01)
        assert isinstance(report, ValidationReport)

    def test_run_verbose_mode(self):
        """Test running tests in verbose mode."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Should not raise
        report = suite.run(tests="serial", verbose=True)
        assert isinstance(report, ValidationReport)

    def test_run_with_list_of_tests(self):
        """Test running specific list of test categories."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests=["serial", "het"])
        assert isinstance(report, ValidationReport)

    def test_run_invalid_test_specification(self):
        """Test that invalid test specification raises error."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        with pytest.raises(ValueError):
            suite.run(tests="invalid_test_name")


class TestDetermineTests:
    """Test _determine_tests method."""

    def test_determine_tests_all(self):
        """Test determining all tests."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        tests = suite._determine_tests("all")
        assert "specification" in tests
        assert "serial" in tests
        assert "het" in tests
        assert "cd" in tests

    def test_determine_tests_default_fixed_effects(self):
        """Test default tests for Fixed Effects."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        tests = suite._determine_tests("default")
        assert "serial" in tests
        assert "het" in tests
        assert "cd" in tests

    def test_determine_tests_default_random_effects(self):
        """Test default tests for Random Effects."""
        results = MockPanelResults(model_type="Random Effects")
        suite = ValidationSuite(results)

        tests = suite._determine_tests("default")
        assert "cd" in tests

    def test_determine_tests_default_pooled_ols(self):
        """Test default tests for Pooled OLS."""
        results = MockPanelResults(model_type="Pooled OLS")
        suite = ValidationSuite(results)

        tests = suite._determine_tests("default")
        assert "het" in tests
        assert "cd" in tests

    def test_determine_tests_serial_only(self):
        """Test determining serial tests only."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        tests = suite._determine_tests("serial")
        assert tests == ["serial"]

    def test_determine_tests_het_only(self):
        """Test determining het tests only."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        tests = suite._determine_tests("het")
        assert tests == ["het"]

    def test_determine_tests_cd_only(self):
        """Test determining cd tests only."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        tests = suite._determine_tests("cd")
        assert tests == ["cd"]

    def test_determine_tests_custom_list(self):
        """Test determining tests from custom list."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        tests = suite._determine_tests(["serial", "het"])
        assert tests == ["serial", "het"]


class TestRunSpecificationTests:
    """Test run_specification_tests method."""

    def test_run_specification_tests_returns_dict(self):
        """Test that method returns a dictionary."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        spec_tests = suite.run_specification_tests()
        assert isinstance(spec_tests, dict)

    def test_run_specification_tests_with_verbose(self):
        """Test running specification tests in verbose mode."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        spec_tests = suite.run_specification_tests(verbose=True)
        assert isinstance(spec_tests, dict)

    def test_run_specification_tests_handles_errors(self):
        """Test that errors are handled gracefully."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Should not raise even if some tests fail
        spec_tests = suite.run_specification_tests()
        assert isinstance(spec_tests, dict)


class TestRunSerialCorrelationTests:
    """Test run_serial_correlation_tests method."""

    def test_run_serial_tests_returns_dict(self):
        """Test that method returns a dictionary."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        serial_tests = suite.run_serial_correlation_tests()
        assert isinstance(serial_tests, dict)

    def test_run_serial_tests_fixed_effects(self):
        """Test serial tests for Fixed Effects model."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        serial_tests = suite.run_serial_correlation_tests()
        # May include Wooldridge test
        assert isinstance(serial_tests, dict)

    def test_run_serial_tests_with_verbose(self):
        """Test running serial tests in verbose mode."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        serial_tests = suite.run_serial_correlation_tests(verbose=True)
        assert isinstance(serial_tests, dict)

    def test_run_serial_tests_handles_errors(self):
        """Test that errors are handled gracefully."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Should not raise even if some tests fail
        serial_tests = suite.run_serial_correlation_tests()
        assert isinstance(serial_tests, dict)


class TestRunHeteroskedasticityTests:
    """Test run_heteroskedasticity_tests method."""

    def test_run_het_tests_returns_dict(self):
        """Test that method returns a dictionary."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        het_tests = suite.run_heteroskedasticity_tests()
        assert isinstance(het_tests, dict)

    def test_run_het_tests_fixed_effects(self):
        """Test het tests for Fixed Effects model."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        het_tests = suite.run_heteroskedasticity_tests()
        # May include Modified Wald test
        assert isinstance(het_tests, dict)

    def test_run_het_tests_with_verbose(self):
        """Test running het tests in verbose mode."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        het_tests = suite.run_heteroskedasticity_tests(verbose=True)
        assert isinstance(het_tests, dict)

    def test_run_het_tests_handles_errors(self):
        """Test that errors are handled gracefully."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Should not raise even if some tests fail
        het_tests = suite.run_heteroskedasticity_tests()
        assert isinstance(het_tests, dict)


class TestRunCrossSectionalTests:
    """Test run_cross_sectional_tests method."""

    def test_run_cd_tests_returns_dict(self):
        """Test that method returns a dictionary."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        cd_tests = suite.run_cross_sectional_tests()
        assert isinstance(cd_tests, dict)

    def test_run_cd_tests_with_verbose(self):
        """Test running cd tests in verbose mode."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        cd_tests = suite.run_cross_sectional_tests(verbose=True)
        assert isinstance(cd_tests, dict)

    def test_run_cd_tests_handles_errors(self):
        """Test that errors are handled gracefully."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Should not raise even if some tests fail
        cd_tests = suite.run_cross_sectional_tests()
        assert isinstance(cd_tests, dict)


class TestIntegration:
    """Integration tests for ValidationSuite."""

    def test_full_workflow_fixed_effects(self):
        """Test complete workflow for Fixed Effects model."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        # Run all tests
        report = suite.run(tests="all", alpha=0.05)

        assert isinstance(report, ValidationReport)
        assert hasattr(report, "model_info")

    def test_full_workflow_random_effects(self):
        """Test complete workflow for Random Effects model."""
        results = MockPanelResults(model_type="Random Effects")
        suite = ValidationSuite(results)

        report = suite.run(tests="default")

        assert isinstance(report, ValidationReport)

    def test_sequential_test_runs(self):
        """Test running different test categories sequentially."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Run different tests sequentially
        report1 = suite.run(tests="serial")
        report2 = suite.run(tests="het")
        report3 = suite.run(tests="cd")

        assert isinstance(report1, ValidationReport)
        assert isinstance(report2, ValidationReport)
        assert isinstance(report3, ValidationReport)

    def test_different_alpha_levels(self):
        """Test running with different significance levels."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        for alpha in [0.01, 0.05, 0.10]:
            report = suite.run(tests="serial", alpha=alpha)
            assert isinstance(report, ValidationReport)

    def test_suite_with_small_sample(self):
        """Test suite with small sample size."""
        results = MockPanelResults(n_entities=5, n_periods=5)
        suite = ValidationSuite(results)

        # Should handle small samples gracefully
        report = suite.run(tests="default")
        assert isinstance(report, ValidationReport)

    def test_suite_with_large_sample(self):
        """Test suite with larger sample size."""
        results = MockPanelResults(n_entities=50, n_periods=20)
        suite = ValidationSuite(results)

        report = suite.run(tests="default")
        assert isinstance(report, ValidationReport)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_test_string(self):
        """Test that invalid test string raises ValueError."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        with pytest.raises(ValueError):
            suite.run(tests="nonexistent_test")

    def test_empty_test_list(self):
        """Test with empty test list."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        # Empty list should still return a report
        report = suite.run(tests=[])
        assert isinstance(report, ValidationReport)

    def test_verbose_false(self):
        """Test that verbose=False works."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="serial", verbose=False)
        assert isinstance(report, ValidationReport)

    def test_report_contains_model_info(self):
        """Test that report contains model information."""
        results = MockPanelResults()
        suite = ValidationSuite(results)

        report = suite.run(tests="serial")

        # Report should have model_info attribute
        assert hasattr(report, "model_info")


class TestValidationSuiteSpecificationTests:
    """Test specification tests with different model types."""

    def test_specification_tests_random_effects(self):
        """Test specification tests for Random Effects model."""
        results = MockPanelResults(model_type="Random Effects")
        suite = ValidationSuite(results)

        # Should attempt to run Mundlak test for RE models
        spec_tests = suite.run_specification_tests(alpha=0.05, verbose=True)
        assert isinstance(spec_tests, dict)

    def test_specification_tests_fixed_effects_verbose(self):
        """Test specification tests for FE model in verbose mode."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        spec_tests = suite.run_specification_tests(alpha=0.05, verbose=True)
        assert isinstance(spec_tests, dict)

    def test_specification_tests_pooled_ols(self):
        """Test specification tests for Pooled OLS."""
        results = MockPanelResults(model_type="Pooled OLS")
        suite = ValidationSuite(results)

        spec_tests = suite.run_specification_tests(alpha=0.05, verbose=True)
        assert isinstance(spec_tests, dict)


class TestValidationSuiteVerboseMode:
    """Test verbose mode in all test categories."""

    def test_serial_tests_verbose(self):
        """Test serial correlation tests in verbose mode."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        serial_tests = suite.run_serial_correlation_tests(alpha=0.05, verbose=True)
        assert isinstance(serial_tests, dict)

    def test_het_tests_verbose(self):
        """Test heteroskedasticity tests in verbose mode."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        het_tests = suite.run_heteroskedasticity_tests(alpha=0.05, verbose=True)
        assert isinstance(het_tests, dict)

    def test_cd_tests_verbose(self):
        """Test cross-sectional dependence tests in verbose mode."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        cd_tests = suite.run_cross_sectional_tests(alpha=0.05, verbose=True)
        assert isinstance(cd_tests, dict)

    def test_all_tests_verbose(self):
        """Test running all tests in verbose mode."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        report = suite.run(tests="all", alpha=0.05, verbose=True)
        assert isinstance(report, ValidationReport)

    def test_specification_tests_verbose(self):
        """Test specification tests in verbose mode with RE model."""
        results = MockPanelResults(model_type="Random Effects")
        suite = ValidationSuite(results)

        report = suite.run(tests="all", alpha=0.05, verbose=True)
        assert isinstance(report, ValidationReport)


class TestValidationSuiteCustomAlpha:
    """Test with custom alpha levels."""

    def test_all_tests_alpha_001(self):
        """Test all tests with alpha=0.01."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        report = suite.run(tests="all", alpha=0.01)
        assert isinstance(report, ValidationReport)

    def test_all_tests_alpha_010(self):
        """Test all tests with alpha=0.10."""
        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        report = suite.run(tests="all", alpha=0.10)
        assert isinstance(report, ValidationReport)


class TestExceptionHandlingVerbose:
    """Test that exception handling branches are covered when tests fail with verbose=True.

    These tests force individual diagnostic tests to raise exceptions by
    patching their constructors, then verify the suite handles errors
    gracefully (issues warnings and continues).
    """

    def test_mundlak_test_failure_verbose(self):
        """Test Mundlak test exception handler with verbose=True (lines 169-171)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Random Effects")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.MundlakTest",
            side_effect=RuntimeError("Mundlak forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                spec_tests = suite.run_specification_tests(alpha=0.05, verbose=True)

            assert isinstance(spec_tests, dict)
            assert "Mundlak" not in spec_tests
            # Check that a warning was issued
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Mundlak test failed" in msg for msg in warning_messages)

    def test_wooldridge_test_failure_verbose(self):
        """Test Wooldridge test exception handler with verbose=True (lines 217-220)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.WooldridgeARTest",
            side_effect=RuntimeError("Wooldridge forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                serial_tests = suite.run_serial_correlation_tests(alpha=0.05, verbose=True)

            assert isinstance(serial_tests, dict)
            assert "Wooldridge" not in serial_tests
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Wooldridge test failed" in msg for msg in warning_messages)

    def test_baltagi_wu_test_failure_verbose(self):
        """Test Baltagi-Wu test exception handler with verbose=True (lines 239-242)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Pooled OLS")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.BaltagiWuTest",
            side_effect=RuntimeError("Baltagi-Wu forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                serial_tests = suite.run_serial_correlation_tests(alpha=0.05, verbose=True)

            assert isinstance(serial_tests, dict)
            assert "Baltagi-Wu" not in serial_tests
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Baltagi-Wu test failed" in msg for msg in warning_messages)

    def test_modified_wald_test_failure_verbose(self):
        """Test Modified Wald test exception handler with verbose=True (lines 273-276)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.ModifiedWaldTest",
            side_effect=RuntimeError("Modified Wald forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                het_tests = suite.run_heteroskedasticity_tests(alpha=0.05, verbose=True)

            assert isinstance(het_tests, dict)
            assert "Modified Wald" not in het_tests
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Modified Wald test failed" in msg for msg in warning_messages)

    def test_pesaran_cd_test_failure_verbose(self):
        """Test Pesaran CD test exception handler with verbose=True (lines 330-333)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.PesaranCDTest",
            side_effect=RuntimeError("Pesaran CD forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                cd_tests = suite.run_cross_sectional_tests(alpha=0.05, verbose=True)

            assert isinstance(cd_tests, dict)
            assert "Pesaran CD" not in cd_tests
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Pesaran CD test failed" in msg for msg in warning_messages)

    def test_breusch_pagan_lm_test_failure_verbose(self):
        """Test Breusch-Pagan LM test exception handler with verbose=True (lines 341-344)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.BreuschPaganLMTest",
            side_effect=RuntimeError("Breusch-Pagan LM forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                cd_tests = suite.run_cross_sectional_tests(alpha=0.05, verbose=True)

            assert isinstance(cd_tests, dict)
            assert "Breusch-Pagan LM" not in cd_tests
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Breusch-Pagan LM test failed" in msg for msg in warning_messages)

    def test_frees_test_failure_verbose(self):
        """Test Frees test exception handler with verbose=True (lines 352-355)."""
        from unittest.mock import patch

        results = MockPanelResults(model_type="Fixed Effects")
        suite = ValidationSuite(results)

        with patch(
            "panelbox.validation.validation_suite.FreesTest",
            side_effect=RuntimeError("Frees forced failure"),
        ):
            import warnings as w

            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                cd_tests = suite.run_cross_sectional_tests(alpha=0.05, verbose=True)

            assert isinstance(cd_tests, dict)
            assert "Frees" not in cd_tests
            warning_messages = [str(warning.message) for warning in caught]
            assert any("Frees test failed" in msg for msg in warning_messages)
