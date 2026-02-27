"""Tests for ModelValidator -- prediction sanity, GMM diagnostics, formatting."""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from panelbox.production.validation import ModelValidator


class TestCheckParams:
    def test_check_params_valid(self):
        """Test check_params with valid parameters."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.5, "const": 0.3})
        validator = ModelValidator(mock_results)
        check = validator.check_params()
        assert check["name"] == "parameter_validity"
        assert check["no_nan"] is True
        assert check["no_inf"] is True
        assert check["reasonable_magnitude"] is True

    def test_check_params_with_nan(self):
        """Test check_params flags NaN parameters."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": np.nan, "const": 0.3})
        validator = ModelValidator(mock_results)
        check = validator.check_params()
        assert check["no_nan"] is False

    def test_check_params_with_inf(self):
        """Test check_params flags Inf parameters."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": np.inf, "const": 0.3})
        validator = ModelValidator(mock_results)
        check = validator.check_params()
        assert check["no_inf"] is False

    def test_check_params_large_magnitude(self):
        """Test check_params flags large magnitude parameters."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 5000.0, "const": 0.3})
        validator = ModelValidator(mock_results)
        check = validator.check_params()
        assert check["reasonable_magnitude"] is False


class TestPredictSanity:
    def test_predict_sanity_no_data_skipped(self):
        """Test predict sanity check skipped when no data provided."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        validator = ModelValidator(mock_results, training_data=None)
        result = validator.check_predict_sanity()
        assert result["skipped"] is True
        assert result["name"] == "predict_sanity"

    def test_predict_sanity_with_test_data(self):
        """Test predict sanity with explicit test data."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        mock_results.predict.return_value = np.array([1.0, 2.0, 3.0])
        validator = ModelValidator(mock_results)
        test_data = pd.DataFrame({"x1": [1, 2, 3]})
        result = validator.check_predict_sanity(test_data=test_data)
        assert result["passed"] is True
        assert result["n_predictions"] == 3

    def test_predict_sanity_exception(self):
        """Test predict sanity handles exception gracefully."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        mock_results.predict.side_effect = Exception("predict failed")
        validator = ModelValidator(mock_results)
        test_data = pd.DataFrame({"x1": [1, 2, 3]})
        result = validator.check_predict_sanity(test_data=test_data)
        assert result["passed"] is False
        assert "predict failed" in result["error"]

    def test_predict_sanity_uses_training_data(self):
        """Test predict sanity uses training_data when test_data is None."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        mock_results.predict.return_value = np.array([1.0, 2.0])
        training = pd.DataFrame({"x1": [1, 2]})
        validator = ModelValidator(mock_results, training_data=training)
        result = validator.check_predict_sanity()
        assert result["passed"] is True
        mock_results.predict.assert_called_once_with(training)


class TestGMMDiagnostics:
    def test_gmm_diagnostics_non_gmm_returns_none(self):
        """Test GMM diagnostics returns None for non-GMM model."""
        mock_results = MagicMock(spec=["params"])
        mock_results.params = pd.Series({"x1": 1.0})
        validator = ModelValidator(mock_results)
        result = validator.check_gmm_diagnostics()
        assert result is None

    def test_gmm_diagnostics_with_gmm_results(self):
        """Test GMM diagnostics returns full report for GMM model."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        mock_results.hansen_j.pvalue = 0.50
        mock_results.ar2_test.pvalue = 0.30
        mock_results.instrument_ratio = 0.8
        validator = ModelValidator(mock_results)
        result = validator.check_gmm_diagnostics()
        assert result["name"] == "gmm_diagnostics"
        assert result["hansen_j_ok"] is True
        assert result["ar2_ok"] is True
        assert result["instrument_ratio_ok"] is True

    def test_gmm_diagnostics_failing(self):
        """Test GMM diagnostics flags failing tests."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        mock_results.hansen_j.pvalue = 0.01
        mock_results.ar2_test.pvalue = 0.02
        mock_results.instrument_ratio = 1.5
        validator = ModelValidator(mock_results)
        result = validator.check_gmm_diagnostics()
        assert result["hansen_j_ok"] is False
        assert result["ar2_ok"] is False
        assert result["instrument_ratio_ok"] is False


class TestRunAll:
    def test_run_all_passed(self):
        """Test run_all with passing results."""
        mock_results = MagicMock(spec=["params", "predict"])
        mock_results.params = pd.Series({"x1": 1.0, "const": 0.5})
        mock_results.predict.return_value = np.array([1.0, 2.0])
        training = pd.DataFrame({"x1": [1, 2]})
        validator = ModelValidator(mock_results, training_data=training)
        report = validator.run_all()
        assert report["passed"] is True
        assert "summary" in report

    def test_run_all_with_gmm(self):
        """Test run_all includes GMM checks when available."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        mock_results.predict.return_value = np.array([1.0])
        mock_results.hansen_j.pvalue = 0.50
        mock_results.ar2_test.pvalue = 0.30
        mock_results.instrument_ratio = 0.5
        training = pd.DataFrame({"x1": [1]})
        validator = ModelValidator(mock_results, training_data=training)
        report = validator.run_all()
        check_names = [c["name"] for c in report["checks"]]
        assert "gmm_diagnostics" in check_names


class TestFormatSummary:
    def test_format_summary_passed(self):
        """Test format_summary with all passed checks."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        validator = ModelValidator(mock_results)
        checks = [
            {"name": "test1", "passed": True},
            {"name": "test2", "passed": True},
        ]
        summary = validator._format_summary(checks)
        assert isinstance(summary, str)
        assert "PASSED" in summary

    def test_format_summary_skipped_and_failed(self):
        """Test format_summary with mixed statuses."""
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0})
        validator = ModelValidator(mock_results)
        checks = [
            {"name": "test1", "passed": True},
            {"name": "test2", "skipped": True, "reason": "No data"},
            {"name": "test3", "passed": False},
        ]
        summary = validator._format_summary(checks)
        assert isinstance(summary, str)
        assert "SKIPPED" in summary
        assert "FAILED" in summary
        assert "PASSED" in summary
