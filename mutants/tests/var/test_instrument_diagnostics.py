"""
Tests for instrument diagnostics in Panel VAR GMM.
"""

import numpy as np
import pytest

from panelbox.var.diagnostics import GMMDiagnostics
from panelbox.var.result import PanelVARGMMResult


class TestInstrumentDiagnosticsIntegration:
    """Test instrument diagnostics integration with PanelVARGMMResult."""

    @pytest.fixture
    def mock_gmm_result(self):
        """Create a mock GMM result for testing."""
        np.random.seed(42)

        # Setup basic dimensions
        K = 2  # 2 variables
        p = 1  # 1 lag
        N = 20  # 20 entities
        T = 10  # 10 time periods
        n_obs = N * T

        # Mock parameters
        n_params_per_eq = K * p + 1  # Lags + constant
        params_by_eq = [np.random.randn(n_params_per_eq) for _ in range(K)]
        std_errors_by_eq = [np.abs(np.random.randn(n_params_per_eq) * 0.1) for _ in range(K)]
        cov_by_eq = [np.eye(n_params_per_eq) * 0.01 for _ in range(K)]

        # Mock residuals and fitted values
        resid_by_eq = [np.random.randn(n_obs) for _ in range(K)]
        fitted_by_eq = [np.random.randn(n_obs) for _ in range(K)]

        # Mock instruments
        n_instruments = 15
        instruments = np.random.randn(n_obs, n_instruments)

        # Mock entity IDs
        entity_ids = np.repeat(np.arange(N), T)

        # Create result
        result = PanelVARGMMResult(
            params_by_eq=params_by_eq,
            std_errors_by_eq=std_errors_by_eq,
            cov_by_eq=cov_by_eq,
            resid_by_eq=resid_by_eq,
            fitted_by_eq=fitted_by_eq,
            endog_names=["y1", "y2"],
            exog_names=["L1.y1", "L1.y2", "const"],
            model_info={
                "lags": p,
                "method": "gmm-fod",
                "cov_type": "robust",
                "trend": "constant",
                "n_exog": 0,
            },
            data_info={"n_entities": N, "n_obs": n_obs},
            instruments=instruments,
            n_instruments=n_instruments,
            instrument_type="all",
            gmm_step="two-step",
            entity_ids=entity_ids,
            windmeijer_corrected=True,
        )

        return result

    def test_instrument_diagnostics_method_exists(self, mock_gmm_result):
        """Test that instrument_diagnostics method exists."""
        assert hasattr(mock_gmm_result, "instrument_diagnostics")

    def test_instrument_diagnostics_returns_string(self, mock_gmm_result):
        """Test that instrument_diagnostics returns a formatted string."""
        report = mock_gmm_result.instrument_diagnostics()
        assert isinstance(report, str)
        assert len(report) > 0

    def test_instrument_diagnostics_includes_key_info(self, mock_gmm_result):
        """Test that diagnostics report includes key information."""
        report = mock_gmm_result.instrument_diagnostics()

        # Should include instrument counts
        assert "Number of instruments" in report
        assert "Number of parameters" in report
        assert "Number of entities" in report

        # Should include ratios
        assert "Ratio instruments/N" in report
        assert "Ratio instruments/params" in report

        # Should include Hansen J test
        assert "Hansen J statistic" in report
        assert "p-value" in report

        # Should include diagnosis
        assert "DIAGNOSIS" in report

    def test_instrument_diagnostics_includes_ar_tests(self, mock_gmm_result):
        """Test that diagnostics includes AR tests when entity_ids available."""
        report = mock_gmm_result.instrument_diagnostics()

        # Should include AR tests
        assert "Serial Correlation Tests" in report
        assert "AR(1) test" in report
        assert "AR(2) test" in report

    def test_hansen_j_test_method(self, mock_gmm_result):
        """Test hansen_j_test method."""
        hansen_result = mock_gmm_result.hansen_j_test()

        assert "statistic" in hansen_result
        assert "p_value" in hansen_result
        assert "df" in hansen_result
        assert "interpretation" in hansen_result

        # Check values are valid
        assert not np.isnan(hansen_result["statistic"])
        assert 0 <= hansen_result["p_value"] <= 1
        assert hansen_result["df"] > 0

    def test_sargan_test_method(self, mock_gmm_result):
        """Test sargan_test method."""
        sargan_result = mock_gmm_result.sargan_test()

        assert "statistic" in sargan_result
        assert "p_value" in sargan_result
        assert "df" in sargan_result

        # Check values are valid
        assert not np.isnan(sargan_result["statistic"])
        assert 0 <= sargan_result["p_value"] <= 1

    def test_ar_test_methods(self, mock_gmm_result):
        """Test AR test methods."""
        # AR(1) test
        ar1_result = mock_gmm_result.ar_test(order=1)
        assert "statistic" in ar1_result
        assert "p_value" in ar1_result
        assert ar1_result["order"] == 1

        # AR(2) test
        ar2_result = mock_gmm_result.ar_test(order=2)
        assert "statistic" in ar2_result
        assert "p_value" in ar2_result
        assert ar2_result["order"] == 2

    def test_summary_includes_gmm_diagnostics(self, mock_gmm_result):
        """Test that summary includes GMM diagnostics."""
        summary = mock_gmm_result.summary()

        # Should include GMM section
        assert "GMM Estimation Details" in summary
        assert "GMM step: two-step" in summary
        assert "Instrument type: all" in summary
        assert "Number of instruments: 15" in summary
        assert "Windmeijer correction: Yes" in summary

        # Should include Hansen J test
        assert "Hansen J Test" in summary

        # Should include AR tests
        assert "Serial Correlation Tests" in summary
        assert "AR(1)" in summary
        assert "AR(2)" in summary

    def test_result_repr_includes_gmm_info(self, mock_gmm_result):
        """Test that __repr__ includes GMM-specific info."""
        repr_str = repr(mock_gmm_result)

        assert "PanelVARGMMResult" in repr_str
        assert "gmm_step" in repr_str
        assert "n_instruments" in repr_str


class TestInstrumentDiagnosticsWarnings:
    """Test warning system in instrument diagnostics."""

    def test_instrument_proliferation_warning(self):
        """Test warning when #instruments > #entities."""
        np.random.seed(42)

        N = 10  # Small N
        n_obs = 100
        n_instruments = 20  # More than N
        n_params = 5

        residuals = np.random.randn(n_obs, 2)
        instruments = np.random.randn(n_obs, n_instruments)
        entity_ids = np.repeat(np.arange(N), 10)

        diag = GMMDiagnostics(
            residuals=residuals,
            instruments=instruments,
            n_params=n_params,
            n_entities=N,
            entity_ids=entity_ids,
        )

        report = diag.format_diagnostics_report()

        # Should include warning
        assert "WARNING" in report or "warning" in report.lower()

    def test_high_instrument_ratio_warning(self):
        """Test warning when instruments/params ratio is high."""
        np.random.seed(42)

        N = 50
        n_obs = 500
        n_instruments = 40  # High ratio
        n_params = 10

        residuals = np.random.randn(n_obs)
        instruments = np.random.randn(n_obs, n_instruments)

        diag = GMMDiagnostics(
            residuals=residuals,
            instruments=instruments,
            n_params=n_params,
            n_entities=N,
            entity_ids=None,
        )

        report_dict = diag.instrument_diagnostics_report()

        # Ratio should be flagged
        assert report_dict["ratio_instr_params"] > 3
        assert len(report_dict["warnings"]) > 0


class TestInstrumentDiagnosticsSuggestions:
    """Test suggestion system in diagnostics."""

    def test_suggestions_for_proliferation(self):
        """Test that suggestions are provided when proliferation detected."""
        np.random.seed(42)

        N = 10
        n_obs = 100
        n_instruments = 25  # Excessive
        n_params = 5

        residuals = np.random.randn(n_obs)
        instruments = np.random.randn(n_obs, n_instruments)

        diag = GMMDiagnostics(
            residuals=residuals,
            instruments=instruments,
            n_params=n_params,
            n_entities=N,
            entity_ids=None,
        )

        report_dict = diag.instrument_diagnostics_report()

        # Should have suggestions
        assert len(report_dict["suggestions"]) > 0
        # Should suggest collapsed instruments
        suggestions_str = " ".join(report_dict["suggestions"])
        assert "collapsed" in suggestions_str.lower()

    def test_no_warnings_for_good_instruments(self):
        """Test that no warnings for well-specified instruments."""
        np.random.seed(42)

        N = 50
        n_obs = 500
        n_instruments = 12  # Reasonable count
        n_params = 6

        residuals = np.random.randn(n_obs)
        instruments = np.random.randn(n_obs, n_instruments)

        diag = GMMDiagnostics(
            residuals=residuals,
            instruments=instruments,
            n_params=n_params,
            n_entities=N,
            entity_ids=None,
        )

        report_dict = diag.instrument_diagnostics_report()

        # Ratios should be OK
        assert report_dict["ratio_instr_entities"] < 1
        assert report_dict["ratio_instr_params"] < 3

        # If Hansen J p-value is in good range, should have positive diagnosis
        if 0.10 <= report_dict["hansen_j"]["p_value"] <= 0.90:
            assert "valid" in report_dict["diagnosis"].lower()
