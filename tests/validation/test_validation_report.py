"""
Tests for ValidationReport class.
"""

from panelbox.validation.base import ValidationTestResult
from panelbox.validation.validation_report import ValidationReport


class TestValidationReportInit:
    """Test ValidationReport initialization."""

    def test_init_basic(self):
        """Test basic initialization."""
        model_info = {"model_type": "Fixed Effects", "formula": "y ~ x", "nobs": 100}
        report = ValidationReport(model_info=model_info)

        assert report.model_info == model_info
        assert report.specification_tests == {}
        assert report.serial_tests == {}
        assert report.het_tests == {}
        assert report.cd_tests == {}

    def test_init_with_tests(self):
        """Test initialization with test results."""
        model_info = {"model_type": "Fixed Effects"}

        test1 = ValidationTestResult(
            test_name="Test 1",
            statistic=5.0,
            pvalue=0.01,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        test2 = ValidationTestResult(
            test_name="Test 2",
            statistic=2.0,
            pvalue=0.15,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info, serial_tests={"Wooldridge": test1}, het_tests={"White": test2}
        )

        assert len(report.serial_tests) == 1
        assert len(report.het_tests) == 1
        assert report.serial_tests["Wooldridge"] == test1


class TestValidationReportRepr:
    """Test string representations."""

    def test_str_method(self):
        """Test __str__ method."""
        model_info = {"model_type": "Fixed Effects"}
        report = ValidationReport(model_info=model_info)

        report_str = str(report)
        assert isinstance(report_str, str)
        assert len(report_str) > 0

    def test_repr_method(self):
        """Test __repr__ method."""
        model_info = {"model_type": "Random Effects"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Test": test})

        report_repr = repr(report)
        assert "ValidationReport" in report_repr
        assert "Random Effects" in report_repr
        assert "tests=1" in report_repr

    def test_repr_counts_all_tests(self):
        """Test that repr counts all test categories."""
        model_info = {"model_type": "FE"}

        test = ValidationTestResult(
            test_name="Test",
            statistic=1.0,
            pvalue=0.5,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            specification_tests={"spec": test},
            serial_tests={"serial": test},
            het_tests={"het": test},
            cd_tests={"cd": test},
        )

        report_repr = repr(report)
        assert "tests=4" in report_repr


class TestValidationReportSummary:
    """Test summary method."""

    def test_summary_basic(self):
        """Test basic summary generation."""
        model_info = {
            "model_type": "Fixed Effects",
            "formula": "y ~ x1 + x2",
            "nobs": 200,
            "n_entities": 10,
        }

        report = ValidationReport(model_info=model_info)
        summary = report.summary()

        assert "MODEL VALIDATION REPORT" in summary
        assert "Fixed Effects" in summary
        assert "y ~ x1 + x2" in summary
        assert "200" in summary

    def test_summary_with_tests(self):
        """Test summary with test results."""
        model_info = {"model_type": "FE", "formula": "y ~ x", "nobs": 100, "n_entities": 5}

        test_reject = ValidationTestResult(
            test_name="Serial Test",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="No serial correlation",
            alternative_hypothesis="Serial correlation present",
        )

        test_ok = ValidationTestResult(
            test_name="Het Test",
            statistic=1.0,
            pvalue=0.50,
            null_hypothesis="Homoskedasticity",
            alternative_hypothesis="Heteroskedasticity",
        )

        report = ValidationReport(
            model_info=model_info,
            serial_tests={"Wooldridge": test_reject},
            het_tests={"White": test_ok},
        )

        summary = report.summary()
        assert "Serial Correlation Tests" in summary
        assert "Heteroskedasticity Tests" in summary
        assert "REJECT" in summary
        assert "OK" in summary

    def test_summary_verbose_false(self):
        """Test summary with verbose=False."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Test": test})

        summary = report.summary(verbose=False)
        assert "VALIDATION TESTS SUMMARY" in summary
        assert "DETAILED TEST RESULTS" not in summary

    def test_summary_verbose_true(self):
        """Test summary with verbose=True."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Test": test})

        summary = report.summary(verbose=True)
        assert "DETAILED TEST RESULTS" in summary

    def test_summary_no_issues(self):
        """Test summary when no issues detected."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=1.0,
            pvalue=0.80,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Test": test})

        summary = report.summary()
        assert "No major issues detected" in summary

    def test_summary_with_issues(self):
        """Test summary when issues are detected."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Serial Test",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Wooldridge": test})

        summary = report.summary()
        assert "POTENTIAL ISSUES DETECTED" in summary
        assert "Wooldridge" in summary

    def test_summary_recommendations(self):
        """Test that summary includes recommendations."""
        model_info = {"model_type": "FE"}

        serial_test = ValidationTestResult(
            test_name="Serial",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        het_test = ValidationTestResult(
            test_name="Het",
            statistic=8.0,
            pvalue=0.002,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        cd_test = ValidationTestResult(
            test_name="CD",
            statistic=5.0,
            pvalue=0.01,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            serial_tests={"Serial": serial_test},
            het_tests={"Het": het_test},
            cd_tests={"CD": cd_test},
        )

        summary = report.summary()
        assert "clustered standard errors" in summary or "HAC errors" in summary
        assert "robust standard errors" in summary
        assert "Driscoll-Kraay" in summary


class TestValidationReportToDict:
    """Test to_dict method."""

    def test_to_dict_basic(self):
        """Test basic dictionary export."""
        model_info = {"model_type": "FE", "nobs": 100}
        report = ValidationReport(model_info=model_info)

        result = report.to_dict()

        assert isinstance(result, dict)
        assert result["model_info"] == model_info
        assert "specification_tests" in result
        assert "serial_tests" in result
        assert "het_tests" in result
        assert "cd_tests" in result

    def test_to_dict_with_tests(self):
        """Test dictionary export with tests."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=5.0,
            pvalue=0.03,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
            df=10,
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Wooldridge": test})

        result = report.to_dict()

        assert "Wooldridge" in result["serial_tests"]
        assert result["serial_tests"]["Wooldridge"]["statistic"] == 5.0
        assert result["serial_tests"]["Wooldridge"]["pvalue"] == 0.03
        assert result["serial_tests"]["Wooldridge"]["df"] == 10

    def test_to_dict_all_categories(self):
        """Test dictionary export with all test categories."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            specification_tests={"Spec": test},
            serial_tests={"Serial": test},
            het_tests={"Het": test},
            cd_tests={"CD": test},
        )

        result = report.to_dict()

        assert "Spec" in result["specification_tests"]
        assert "Serial" in result["serial_tests"]
        assert "Het" in result["het_tests"]
        assert "CD" in result["cd_tests"]


class TestValidationReportGetFailedTests:
    """Test get_failed_tests method."""

    def test_get_failed_tests_empty(self):
        """Test with no failed tests."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=1.0,
            pvalue=0.50,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Test": test})

        failed = report.get_failed_tests()
        assert failed == []

    def test_get_failed_tests_one_failure(self):
        """Test with one failed test."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Wooldridge": test})

        failed = report.get_failed_tests()
        assert len(failed) == 1
        assert "serial/Wooldridge" in failed

    def test_get_failed_tests_multiple_failures(self):
        """Test with multiple failed tests."""
        model_info = {"model_type": "FE"}

        test_fail = ValidationTestResult(
            test_name="Fail",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        test_pass = ValidationTestResult(
            test_name="Pass",
            statistic=1.0,
            pvalue=0.50,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            serial_tests={"Serial": test_fail},
            het_tests={"Het1": test_fail, "Het2": test_pass},
            cd_tests={"CD": test_fail},
        )

        failed = report.get_failed_tests()
        assert len(failed) == 3
        assert "serial/Serial" in failed
        assert "het/Het1" in failed
        assert "cd/CD" in failed
        assert "het/Het2" not in failed

    def test_get_failed_tests_all_categories(self):
        """Test failures in all categories."""
        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            specification_tests={"Spec": test},
            serial_tests={"Serial": test},
            het_tests={"Het": test},
            cd_tests={"CD": test},
        )

        failed = report.get_failed_tests()
        assert len(failed) == 4
        assert any("spec/" in f for f in failed)
        assert any("serial/" in f for f in failed)
        assert any("het/" in f for f in failed)
        assert any("cd/" in f for f in failed)


class TestSummaryAsDataFrame:
    """Test summary(as_dataframe=True) branch — lines 72-87 and 202."""

    def test_summary_as_dataframe_basic(self):
        """Test that summary(as_dataframe=True) returns a DataFrame."""
        import pandas as pd

        model_info = {"model_type": "FE", "formula": "y ~ x", "nobs": 100, "n_entities": 5}
        test = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"Wooldridge": test})
        df = report.summary(as_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "category" in df.columns
        assert "test" in df.columns
        assert "statistic" in df.columns
        assert "pvalue" in df.columns
        assert "reject" in df.columns
        assert "conclusion" in df.columns

    def test_summary_as_dataframe_multiple_categories(self):
        """Test as_dataframe with tests across multiple categories."""
        import pandas as pd

        model_info = {"model_type": "FE"}

        test_serial = ValidationTestResult(
            test_name="Serial",
            statistic=10.0,
            pvalue=0.001,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        test_het = ValidationTestResult(
            test_name="Het",
            statistic=2.0,
            pvalue=0.30,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        test_cd = ValidationTestResult(
            test_name="CD",
            statistic=5.0,
            pvalue=0.01,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            serial_tests={"Wooldridge": test_serial},
            het_tests={"White": test_het},
            cd_tests={"Pesaran": test_cd},
        )

        df = report.summary(as_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df["category"].unique()) == {
            "Serial Correlation",
            "Heteroskedasticity",
            "Cross-Sectional Dep.",
        }

    def test_summary_as_dataframe_values(self):
        """Test that as_dataframe returns correct values."""

        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="My Test",
            statistic=7.5,
            pvalue=0.003,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(
            model_info=model_info,
            specification_tests={"Hausman": test},
        )

        df = report.summary(as_dataframe=True)

        row = df.iloc[0]
        assert row["category"] == "Specification"
        assert row["test"] == "Hausman"
        assert row["statistic"] == 7.5
        assert row["pvalue"] == 0.003
        assert row["reject"] == True  # p < 0.05  # noqa: E712
        assert isinstance(row["conclusion"], str)

    def test_summary_as_dataframe_empty(self):
        """Test as_dataframe when no tests present."""
        import pandas as pd

        model_info = {"model_type": "FE"}
        report = ValidationReport(model_info=model_info)
        df = report.summary(as_dataframe=True)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_summary_as_dataframe_overrides_verbose(self):
        """Test that as_dataframe=True takes priority, ignoring verbose arg."""
        import pandas as pd

        model_info = {"model_type": "FE"}
        test = ValidationTestResult(
            test_name="Test",
            statistic=3.0,
            pvalue=0.05,
            null_hypothesis="H0",
            alternative_hypothesis="H1",
        )

        report = ValidationReport(model_info=model_info, serial_tests={"T": test})

        # Even with verbose=True, as_dataframe should return DataFrame
        result = report.summary(verbose=True, as_dataframe=True)
        assert isinstance(result, pd.DataFrame)


class TestIntegration:
    """Integration tests for ValidationReport."""

    def test_full_workflow(self):
        """Test complete workflow."""
        # Create model info
        model_info = {
            "model_type": "Fixed Effects",
            "formula": "y ~ x1 + x2",
            "nobs": 200,
            "n_entities": 10,
            "n_periods": 20,
        }

        # Create various test results
        serial_pass = ValidationTestResult(
            "Serial Test", 1.5, 0.22, "No serial correlation", "Serial correlation"
        )

        het_fail = ValidationTestResult(
            "Het Test", 12.3, 0.001, "Homoskedasticity", "Heteroskedasticity"
        )

        # Create report
        report = ValidationReport(
            model_info=model_info,
            serial_tests={"Wooldridge": serial_pass},
            het_tests={"White": het_fail},
        )

        # Test summary
        summary = report.summary()
        assert "Fixed Effects" in summary
        assert "REJECT" in summary  # Het test should reject
        assert "OK" in summary  # Serial test should pass

        # Test to_dict
        report_dict = report.to_dict()
        assert report_dict["model_info"]["nobs"] == 200

        # Test get_failed_tests
        failed = report.get_failed_tests()
        assert len(failed) == 1
        assert "het/White" in failed
