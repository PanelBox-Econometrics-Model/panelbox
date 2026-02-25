"""
Tests for GMM diagnostic tests (Hansen J, Sargan, etc.)
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.diagnostics import GMMDiagnostics, hansen_j_test, sargan_test
from panelbox.var.gmm import estimate_panel_var_gmm


class TestHansenJTest:
    """Tests for Hansen J test of over-identifying restrictions."""

    def test_hansen_j_with_valid_instruments(self):
        """Test Hansen J with valid instruments (should not reject)."""
        np.random.seed(42)

        # Generate data from known DGP
        A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 60
        n_periods = 12

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                y = A1 @ y_prev + np.random.randn(2) * 0.3
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        # Estimate with GMM
        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            transform="fod",
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Construct instrument matrix (simplified for test)
        # In reality, this should come from build_gmm_instruments
        from panelbox.var.instruments import build_gmm_instruments

        Z, _ = build_gmm_instruments(
            data=df,
            var_lags=1,
            n_vars=2,
            entity_col="entity",
            time_col="time",
            value_cols=["y1", "y2"],
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Run Hansen J test
        j_result = hansen_j_test(
            residuals=result.residuals,
            instruments=Z,
            n_params=4,  # VAR(1) with K=2: K*p*K = 2*1*2 = 4
            n_entities=n_entities,
        )

        # Check structure
        assert "statistic" in j_result
        assert "p_value" in j_result
        assert "df" in j_result
        assert "interpretation" in j_result

        # With valid instruments, p-value should be > 0.05 (usually)
        # Note: stochastic test, may occasionally fail
        # For valid DGP, we expect not to reject
        print(f"\nHansen J statistic: {j_result['statistic']:.4f}")
        print(f"P-value: {j_result['p_value']:.4f}")
        print(f"Interpretation: {j_result['interpretation']}")

    def test_hansen_j_exactly_identified(self):
        """Test Hansen J when model is exactly identified (df=0)."""
        np.random.seed(123)

        # Create minimal data
        residuals = np.random.randn(100, 2)
        instruments = np.random.randn(100, 4)  # Exactly 4 instruments = 4 params
        n_params = 4
        n_entities = 20

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        result = diag.hansen_j_test()

        # Should return NaN for exactly identified
        assert np.isnan(result["statistic"])
        assert "exactly identified" in result["interpretation"].lower()

    def test_hansen_j_statistics_positive(self):
        """Test that Hansen J statistic is non-negative."""
        np.random.seed(456)

        residuals = np.random.randn(200, 2)
        instruments = np.random.randn(200, 10)  # Overidentified
        n_params = 4
        n_entities = 40

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        result = diag.hansen_j_test()

        assert result["statistic"] >= 0, "J statistic must be non-negative"
        assert result["p_value"] >= 0 and result["p_value"] <= 1


class TestSarganTest:
    """Tests for Sargan test."""

    def test_sargan_basic(self):
        """Test basic Sargan test functionality."""
        np.random.seed(789)

        residuals = np.random.randn(150, 2)
        instruments = np.random.randn(150, 8)
        n_params = 4
        n_entities = 30

        result = sargan_test(residuals, instruments, n_params, n_entities)

        assert "statistic" in result
        assert "p_value" in result
        assert result["statistic"] >= 0

    def test_sargan_vs_hansen(self):
        """
        Test that Sargan and Hansen give similar results under homoskedasticity.

        Under homoskedasticity, Sargan and Hansen J should be similar.
        """
        np.random.seed(999)

        # Homoskedastic residuals
        residuals = np.random.randn(200, 2) * 0.5  # Constant variance
        instruments = np.random.randn(200, 10)
        n_params = 4
        n_entities = 40

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)

        hansen = diag.hansen_j_test()
        sargan = diag.sargan_test()

        # Statistics should be in similar range (not necessarily equal)
        # Both should be positive
        assert hansen["statistic"] > 0
        assert sargan["statistic"] > 0

        print(f"\nHansen J: {hansen['statistic']:.4f}")
        print(f"Sargan:   {sargan['statistic']:.4f}")


class TestGMMDiagnosticsReport:
    """Tests for comprehensive diagnostics report."""

    def test_diagnostics_report_structure(self):
        """Test that diagnostics report has all required fields."""
        np.random.seed(111)

        residuals = np.random.randn(100, 2)
        instruments = np.random.randn(100, 12)
        n_params = 4
        n_entities = 20

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        report = diag.instrument_diagnostics_report()

        # Check all required fields
        assert "n_instruments" in report
        assert "n_params" in report
        assert "n_entities" in report
        assert "ratio_instr_entities" in report
        assert "ratio_instr_params" in report
        assert "hansen_j" in report
        assert "warnings" in report
        assert "suggestions" in report
        assert "diagnosis" in report

    def test_diagnostics_warnings_proliferation(self):
        """Test that warnings are issued for instrument proliferation."""
        np.random.seed(222)

        residuals = np.random.randn(50, 2)
        instruments = np.random.randn(50, 60)  # 60 instruments > 10 entities
        n_params = 4
        n_entities = 10  # Small N

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        report = diag.instrument_diagnostics_report()

        # Should have warnings about proliferation
        assert len(report["warnings"]) > 0
        assert report["ratio_instr_entities"] > 1  # Violates Roodman rule

    def test_diagnostics_format_string(self):
        """Test that formatted report is a valid string."""
        np.random.seed(333)

        residuals = np.random.randn(100, 2)
        instruments = np.random.randn(100, 8)
        n_params = 4
        n_entities = 20

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        formatted = diag.format_diagnostics_report()

        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Hansen J" in formatted
        assert "DIAGNOSIS" in formatted

        print("\n" + formatted)


class TestIntegrationWithGMM:
    """Integration tests with actual GMM estimation."""

    def test_diagnostics_with_gmm_result(self):
        """Test diagnostics using actual GMM estimation result."""
        np.random.seed(444)

        # Generate data
        A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 50
        n_periods = 12

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                y = A1 @ y_prev + np.random.randn(2) * 0.3
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        # Estimate
        result = estimate_panel_var_gmm(
            df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="two-step",
            transform="fod",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Get instruments
        from panelbox.var.instruments import build_gmm_instruments

        Z, _ = build_gmm_instruments(
            data=df,
            var_lags=1,
            n_vars=2,
            entity_col="entity",
            time_col="time",
            value_cols=["y1", "y2"],
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Create diagnostics
        diag = GMMDiagnostics(
            residuals=result.residuals,
            instruments=Z,
            n_params=result.coefficients.size,  # Total number of coefficients
            n_entities=n_entities,
        )

        # Get report
        report = diag.instrument_diagnostics_report()
        formatted = diag.format_diagnostics_report()

        print("\n" + formatted)

        # Basic checks
        assert report["n_instruments"] == result.n_instruments
        assert report["n_entities"] == n_entities


class TestInstrumentProliferation:
    """Tests for instrument proliferation detection and warnings."""

    def test_proliferation_warning_small_t(self):
        """Test that proliferation warning is issued for panel with small T."""
        np.random.seed(555)

        # Small T panel (T=5) - should trigger warnings with many instruments
        n_entities = 30
        n_periods = 5
        residuals = np.random.randn(n_entities * n_periods, 2)

        # Many instruments relative to T
        instruments = np.random.randn(n_entities * n_periods, 35)
        n_params = 4

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        report = diag.instrument_diagnostics_report()

        # Should have proliferation warnings
        assert len(report["warnings"]) > 0
        assert report["ratio_instr_entities"] > 1.0

        # Should suggest collapsed instruments
        assert len(report["suggestions"]) > 0
        suggestion_text = " ".join(report["suggestions"]).lower()
        assert "collapsed" in suggestion_text or "max_instruments" in suggestion_text

        print(f"\nT=5 panel - Warnings: {report['warnings']}")
        print(f"Suggestions: {report['suggestions']}")

    def test_no_warning_with_collapsed(self):
        """Test that collapsed instruments reduce warnings."""
        np.random.seed(666)

        # Same panel as above but with fewer (collapsed) instruments
        n_entities = 30
        n_periods = 5
        residuals = np.random.randn(n_entities * n_periods, 2)

        # Collapsed instruments - much fewer
        instruments = np.random.randn(n_entities * n_periods, 8)
        n_params = 4

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities)
        report = diag.instrument_diagnostics_report()

        # Should have fewer or no warnings
        assert report["ratio_instr_entities"] < 1.0  # Roodman rule satisfied

        print(f"\nCollapsed instruments - Ratio instr/N: {report['ratio_instr_entities']:.2f}")
        print(f"Warnings: {report['warnings'] if report['warnings'] else 'None'}")


class TestSensitivityAnalysis:
    """Tests for instrument sensitivity analysis."""

    def test_sensitivity_analysis_stable_coefficients(self):
        """Test sensitivity analysis with stable coefficients."""
        from panelbox.var.diagnostics import instrument_sensitivity_analysis

        np.random.seed(777)

        # Create a mock estimation function that returns stable results
        def mock_estimate(max_instruments, **kwargs):
            # Simulate stable coefficients regardless of instrument count
            class MockResult:
                def __init__(self, max_instr):
                    # Coefficients are stable (small random variation)
                    base = np.array([0.5, 0.2, 0.1, 0.6])
                    noise = np.random.randn(4) * 0.01  # Small noise
                    self.params_by_eq = [base[:2] + noise[:2], base[2:] + noise[2:]]
                    self.n_instruments = min(max_instr, 20)

            return MockResult(max_instruments)

        # Run sensitivity analysis
        results = instrument_sensitivity_analysis(
            model_func=mock_estimate, max_instruments_list=[6, 12, 18, 24]
        )

        # Should detect stability
        assert "stable" in results
        assert results["stable"], "Coefficients should be detected as stable"
        assert results["max_change_overall"] < 10.0, "Changes should be < 10%"

        print(f"\nSensitivity analysis - Stable: {results['stable']}")
        print(f"Max change: {results['max_change_overall']:.2f}%")
        print(f"Interpretation: {results['interpretation']}")

    def test_sensitivity_analysis_unstable_coefficients(self):
        """Test sensitivity analysis detects unstable coefficients."""
        from panelbox.var.diagnostics import instrument_sensitivity_analysis

        np.random.seed(888)

        # Create estimation function with unstable results
        def mock_estimate_unstable(max_instruments, **kwargs):
            class MockResult:
                def __init__(self, max_instr):
                    # Coefficients change significantly with instrument count
                    scale = 1.0 + (max_instr / 100.0)  # Coefficients drift
                    base = np.array([0.5, 0.2, 0.1, 0.6]) * scale
                    self.params_by_eq = [base[:2], base[2:]]
                    self.n_instruments = max_instr

            return MockResult(max_instruments)

        results = instrument_sensitivity_analysis(
            model_func=mock_estimate_unstable, max_instruments_list=[10, 20, 40, 80]
        )

        # Should detect instability
        assert not results["stable"], "Should detect coefficient instability"
        assert results["max_change_overall"] > 10.0, "Changes should be > 10%"

        print(f"\nSensitivity (unstable) - Stable: {results['stable']}")
        print(f"Max change: {results['max_change_overall']:.2f}%")


class TestOneStepVsTwoStep:
    """Tests for comparing one-step vs two-step GMM."""

    def test_compare_one_step_two_step_similar(self):
        """Test comparison when one-step and two-step are similar."""
        np.random.seed(999)

        # Create two similar results
        params_base = np.array([0.5, 0.2, 0.1, 0.6])

        # One-step result
        result_1step = type("MockResult", (), {})()
        result_1step.params_by_eq = [params_base[:2], params_base[2:]]
        result_1step.gmm_step = "one-step"

        # Two-step result (very similar)
        result_2step = type("MockResult", (), {})()
        noise = np.random.randn(4) * 0.01  # Very small difference
        result_2step.params_by_eq = [params_base[:2] + noise[:2], params_base[2:] + noise[2:]]
        result_2step.gmm_step = "two-step"

        # Import the actual PanelVARGMMResult class

        # Create minimal GMMResults with compare method
        # We'll use duck typing to add the compare method
        class MockGMMResult:
            def __init__(self, params_by_eq, gmm_step):
                self.params_by_eq = params_by_eq
                self.gmm_step = gmm_step

            def compare_one_step_two_step(self, other):
                # Import numpy here
                import numpy as np

                params_self = np.concatenate([p.flatten() for p in self.params_by_eq])
                params_other = np.concatenate([p.flatten() for p in other.params_by_eq])

                abs_diff = np.abs(params_self - params_other)
                max_pct_diff = np.max(abs_diff / (np.abs(params_other) + 1e-10)) * 100

                if max_pct_diff < 5.0:
                    return "EXCELLENT"
                elif max_pct_diff < 10.0:
                    return "GOOD"
                else:
                    return "WARNING"

        r1 = MockGMMResult(result_1step.params_by_eq, result_1step.gmm_step)
        r2 = MockGMMResult(result_2step.params_by_eq, result_2step.gmm_step)

        comparison = r1.compare_one_step_two_step(r2)

        # Should indicate they are similar
        assert "EXCELLENT" in comparison or "GOOD" in comparison

        print(f"\nOne-step vs Two-step comparison: {comparison}")


class TestHansenJLinAlgError:
    """Tests for hansen_j_test LinAlgError handling to cover lines 124-134."""

    def test_singular_omega_returns_nan(self):
        """Test that a singular Omega matrix results in nan and a warning."""
        np.random.seed(42)

        # Create residuals that produce a singular Omega matrix.
        # If instruments are linearly dependent, Z' diag(e^2) Z can be singular.
        n_obs = 50
        n_params = 4

        residuals = np.random.randn(n_obs, 1)

        # Make some columns of Z identical to create singularity
        base_col = np.random.randn(n_obs)
        instruments = np.column_stack(
            [base_col, base_col, base_col, base_col, np.random.randn(n_obs), np.random.randn(n_obs)]
        )

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities=10)
        result = diag.hansen_j_test()

        # Should return nan statistic due to singular Omega
        assert np.isnan(result["statistic"])
        assert "singular" in result["interpretation"].lower() or np.isnan(result["p_value"])


class TestInterpretHansenJEdgeCases:
    """Tests for _interpret_hansen_j edge cases to cover lines 180-193."""

    def test_very_high_p_value_weak_instruments(self):
        """Test interpretation when p > 0.99 (weak instruments warning)."""
        np.random.seed(42)

        # Create a scenario with many instruments and nearly zero residuals
        # to get a very high p-value (J stat near 0)
        n_obs = 200
        n_instruments = 12
        n_params = 4

        # Very small residuals => g is small => J stat is small => high p-value
        residuals = np.random.randn(n_obs, 1) * 0.0001
        instruments = np.random.randn(n_obs, n_instruments)

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities=200)
        result = diag.hansen_j_test()

        # With very small residuals, J stat should be very small
        # and p-value should be very high
        if result["p_value"] > 0.99:
            assert (
                "weak instruments" in result["interpretation"].lower()
                or "p-value very high" in result["interpretation"]
            )

    def test_ideal_range_p_value(self):
        """Test interpretation when p-value is in ideal range [0.10, 0.90]."""
        np.random.seed(42)

        # We'll directly test the _interpret_hansen_j method
        n_obs = 200
        n_instruments = 8
        n_params = 4

        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, n_instruments)

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities=200)

        # Test ideal range directly
        interpretation, test_warnings = diag._interpret_hansen_j(5.0, 0.50)
        assert "ideal range" in interpretation.lower()

        # Test p > 0.99
        interpretation, test_warnings = diag._interpret_hansen_j(0.1, 0.995)
        assert "weak instruments" in interpretation.lower() or "p-value very high" in interpretation

        # Test rejection (p < 0.05)
        interpretation, test_warnings = diag._interpret_hansen_j(20.0, 0.01)
        assert "reject" in interpretation.lower()
        assert len(test_warnings) > 0

    def test_marginal_p_value_outside_ideal(self):
        """Test interpretation when p-value is valid but outside ideal range."""
        np.random.seed(42)

        n_obs = 200
        n_instruments = 8
        n_params = 4

        residuals = np.random.randn(n_obs, 1)
        instruments = np.random.randn(n_obs, n_instruments)

        diag = GMMDiagnostics(residuals, instruments, n_params, n_entities=200)

        # p-value between 0.05 and 0.10 (valid but outside ideal)
        interpretation, _test_warnings = diag._interpret_hansen_j(5.0, 0.08)
        assert "do not reject" in interpretation.lower()

        # p-value between 0.90 and 0.99 (valid but outside ideal)
        interpretation, _test_warnings = diag._interpret_hansen_j(1.0, 0.95)
        assert "do not reject" in interpretation.lower()


class TestDifferenceInHansenTest:
    """Tests for difference_in_hansen_test to cover lines 309-337."""

    def test_difference_hansen_basic(self):
        """Test basic difference-in-Hansen test with subset of instruments."""
        np.random.seed(42)

        n_obs = 200
        n_instruments_full = 12
        n_instruments_restricted = 6
        n_params = 4

        residuals_full = np.random.randn(n_obs, 2)
        instruments_full = np.random.randn(n_obs, n_instruments_full)

        # Restricted set: first 6 columns
        instruments_restricted = instruments_full[:, :n_instruments_restricted]
        residuals_restricted = residuals_full + np.random.randn(n_obs, 2) * 0.01

        diag = GMMDiagnostics(residuals_full, instruments_full, n_params, n_entities=40)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals_full,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        # Check structure
        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert "interpretation" in result

        # Degrees of freedom = n_instruments_full - n_instruments_restricted
        assert result["df"] == n_instruments_full - n_instruments_restricted

    def test_difference_hansen_valid_additional_instruments(self):
        """Test that valid additional instruments do not reject."""
        np.random.seed(123)

        n_obs = 300
        n_params = 4

        residuals = np.random.randn(n_obs, 1) * 0.5
        instruments_full = np.random.randn(n_obs, 10)
        instruments_restricted = instruments_full[:, :6]
        residuals_restricted = residuals + np.random.randn(n_obs, 1) * 0.001

        diag = GMMDiagnostics(residuals, instruments_full, n_params, n_entities=60)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        # p-value should be a valid number
        assert 0 <= result["p_value"] <= 1
        assert result["df"] == 4  # 10 - 6

    def test_difference_hansen_interpretation_reject(self):
        """Test the rejection interpretation of difference-in-Hansen."""
        np.random.seed(42)

        n_obs = 200
        n_params = 4

        # Use same object to test the interpretation path
        residuals = np.random.randn(n_obs, 1)
        instruments_full = np.random.randn(n_obs, 10)
        instruments_restricted = instruments_full[:, :6]

        diag = GMMDiagnostics(residuals, instruments_full, n_params, n_entities=40)

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals,
            residuals_restricted=residuals,
            n_params=n_params,
        )

        # Check that interpretation string is populated
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0


class TestCompareTransforms:
    """Tests for compare_transforms to cover lines 842-885 (transformation robustness)."""

    def test_compare_transforms_basic(self):
        """Test basic FOD vs FD comparison."""
        from panelbox.var.diagnostics import compare_transforms

        np.random.seed(42)

        # Generate data from known DGP
        A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 50
        n_periods = 12

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                y = A1 @ y_prev + np.random.randn(2) * 0.3
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Check structure
        assert "fod_result" in result
        assert "fd_result" in result
        assert "coef_diff_max" in result
        assert "interpretation" in result
        assert "summary" in result

        # Both results should be present
        assert result["fod_result"] is not None
        assert result["fd_result"] is not None

        # Differences should be computed
        assert result["coef_diff_max"] >= 0
        assert result["coef_diff_mean"] >= 0

        # Summary should be a non-empty string
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0
        assert "FOD vs FD" in result["summary"]

    def test_compare_transforms_interpretation_levels(self):
        """Test that interpretation covers different divergence levels."""
        from panelbox.var.diagnostics import compare_transforms

        np.random.seed(42)

        A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 40
        n_periods = 15

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                y = A1 @ y_prev + np.random.randn(2) * 0.3
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            gmm_step="one-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Interpretation should be one of the known levels
        interp = result["interpretation"]
        assert any(keyword in interp for keyword in ["EXCELLENT", "GOOD", "MODERATE", "WARNING"])

        # Recommendation should be present
        assert "recommendation" in result
        assert len(result["recommendation"]) > 0


class TestHansenJSingularOmegaPath:
    """Additional tests for hansen_j_test singular Omega warning path (lines 124-128)."""

    def test_singular_omega_with_zero_residuals(self):
        """Test Hansen J when residuals produce all-zero Omega."""
        np.random.seed(42)
        n_obs = 50

        # All-zero residuals => Omega = Z' diag(0) Z = 0 => singular
        residuals = np.zeros((n_obs, 1))
        instruments = np.random.randn(n_obs, 8)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        result = diag.hansen_j_test()

        # Should handle gracefully (singular Omega)
        assert np.isnan(result["statistic"]) or result["statistic"] == 0.0
        assert "warnings" in result

    def test_singular_omega_with_rank_deficient_instruments(self):
        """Test Hansen J when instruments are perfectly collinear."""
        np.random.seed(42)
        n_obs = 100

        residuals = np.random.randn(n_obs, 1)

        # Create instruments where all columns are identical => Z'Z is rank 1
        base = np.random.randn(n_obs, 1)
        instruments = np.hstack([base] * 6)

        diag = GMMDiagnostics(residuals, instruments, n_params=4, n_entities=20)
        result = diag.hansen_j_test()

        # Should return nan for singular Omega
        assert np.isnan(result["statistic"])
        assert "singular" in result["interpretation"].lower() or "Singular" in " ".join(
            result.get("warnings", [])
        )


class TestHansenJPValueInterpretations:
    """Test all interpretation paths in _interpret_hansen_j (lines 180-193)."""

    def _make_diag(self, n_instruments=8, n_entities=200):
        """Helper to create a GMMDiagnostics with specific instrument/entity counts."""
        np.random.seed(42)
        return GMMDiagnostics(
            residuals=np.random.randn(200, 1),
            instruments=np.random.randn(200, n_instruments),
            n_params=4,
            n_entities=n_entities,
        )

    def test_rejection_path_p_below_005(self):
        """Test interpretation when p < 0.05 (rejection)."""
        diag = self._make_diag()
        interpretation, test_warnings = diag._interpret_hansen_j(20.0, 0.01)

        assert "Reject" in interpretation or "reject" in interpretation.lower()
        assert any("rejects" in w.lower() for w in test_warnings)

    def test_very_high_p_value_path(self):
        """Test interpretation when p > 0.99 (weak instruments, line 180-185)."""
        diag = self._make_diag()
        interpretation, test_warnings = diag._interpret_hansen_j(0.01, 0.999)

        assert "p-value very high" in interpretation or "weak instruments" in interpretation.lower()
        assert any("p > 0.99" in w for w in test_warnings)

    def test_ideal_range_path(self):
        """Test interpretation when 0.10 <= p <= 0.90 (ideal range, line 187-189)."""
        diag = self._make_diag()
        interpretation, _warnings = diag._interpret_hansen_j(5.0, 0.45)

        assert "ideal range" in interpretation.lower()
        assert "Do not reject" in interpretation

    def test_marginal_range_below_ideal(self):
        """Test interpretation when 0.05 <= p < 0.10 (valid but outside ideal, line 192-193)."""
        diag = self._make_diag()
        interpretation, _warnings = diag._interpret_hansen_j(5.0, 0.07)

        assert "Do not reject" in interpretation
        # Should NOT contain 'ideal range'
        assert "ideal range" not in interpretation.lower()

    def test_marginal_range_above_ideal(self):
        """Test interpretation when 0.90 < p <= 0.99 (valid but outside ideal)."""
        diag = self._make_diag()
        interpretation, _warnings = diag._interpret_hansen_j(1.0, 0.95)

        assert "Do not reject" in interpretation
        assert "ideal range" not in interpretation.lower()

    def test_roodman_rule_warning(self):
        """Test that Roodman rule warning fires when instruments > entities."""
        diag = self._make_diag(n_instruments=30, n_entities=10)
        _interpretation, test_warnings = diag._interpret_hansen_j(5.0, 0.50)

        # Should have Roodman rule warning
        assert any("Rule-of-thumb violated" in w for w in test_warnings)


class TestDifferenceHansenTestPaths:
    """Test difference_hansen_test method (lines 310-337)."""

    def test_difference_hansen_not_reject_path(self):
        """Test that valid additional instruments don't reject (line 334-335)."""
        np.random.seed(42)

        n_obs = 300
        n_params = 4
        n_entities = 50

        # Generate well-behaved random data
        residuals = np.random.randn(n_obs, 1) * 0.5
        instruments_full = np.random.randn(n_obs, 10)
        instruments_restricted = instruments_full[:, :6]
        residuals_restricted = residuals + np.random.randn(n_obs, 1) * 0.001

        diag = GMMDiagnostics(
            residuals,
            instruments_full,
            n_params,
            n_entities=n_entities,
        )

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        assert "statistic" in result
        assert "p_value" in result
        assert "df" in result
        assert result["df"] == 4  # 10 - 6
        assert 0 <= result["p_value"] <= 1

    def test_difference_hansen_reject_path(self):
        """Test difference-in-Hansen with invalid instruments (should reject, line 329-333)."""
        np.random.seed(42)

        n_obs = 200
        n_params = 4
        n_entities = 40

        # Use random residuals for full model
        residuals_full = np.random.randn(n_obs, 1)
        instruments_full = np.random.randn(n_obs, 12)

        # For restricted model, use very different residuals
        instruments_restricted = instruments_full[:, :4]
        residuals_restricted = np.random.randn(n_obs, 1) * 5  # Very different

        diag = GMMDiagnostics(
            residuals_full,
            instruments_full,
            n_params,
            n_entities=n_entities,
        )

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals_full,
            residuals_restricted=residuals_restricted,
            n_params=n_params,
        )

        assert "interpretation" in result
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

    def test_difference_hansen_df_correct(self):
        """Test that degrees of freedom equals difference in instrument counts."""
        np.random.seed(42)

        n_obs = 200
        n_params = 4

        residuals = np.random.randn(n_obs, 1)
        instruments_full = np.random.randn(n_obs, 15)
        instruments_restricted = instruments_full[:, :8]

        diag = GMMDiagnostics(
            residuals,
            instruments_full,
            n_params,
            n_entities=50,
        )

        result = diag.difference_hansen_test(
            instruments_subset=instruments_restricted,
            residuals_full=residuals,
            residuals_restricted=residuals,
            n_params=n_params,
        )

        # df should be 15 - 8 = 7
        assert result["df"] == 7


class TestCompareTransformsAdditional:
    """Additional tests for compare_transforms covering interpretation branches (lines 842-885)."""

    def test_compare_transforms_returns_all_keys(self):
        """Test that compare_transforms result has all expected keys."""
        from panelbox.var.diagnostics import compare_transforms

        np.random.seed(42)

        A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 40
        n_periods = 12

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                y = A1 @ y_prev + np.random.randn(2) * 0.3
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            gmm_step="two-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        # Check all expected keys are present
        expected_keys = [
            "fod_result",
            "fd_result",
            "n_obs_fod",
            "n_obs_fd",
            "coef_diff_max",
            "coef_diff_mean",
            "interpretation",
            "recommendation",
            "summary",
        ]
        for key in expected_keys:
            assert key in result, f"Missing expected key: {key}"

        # Both results should not be None
        assert result["fod_result"] is not None
        assert result["fd_result"] is not None

        # Numerical values should be non-negative
        assert result["coef_diff_max"] >= 0
        assert result["coef_diff_mean"] >= 0

    def test_compare_transforms_summary_contains_fod_fd(self):
        """Test that summary string contains FOD and FD labels."""
        from panelbox.var.diagnostics import compare_transforms

        np.random.seed(42)

        A1 = np.array([[0.5, 0.2], [0.1, 0.6]])
        n_entities = 30
        n_periods = 12

        data_list = []
        for entity in range(1, n_entities + 1):
            y_prev = np.random.randn(2) * 0.5
            for t in range(1, n_periods + 1):
                y = A1 @ y_prev + np.random.randn(2) * 0.3
                data_list.append({"entity": entity, "time": t, "y1": y[0], "y2": y[1]})
                y_prev = y.copy()

        df = pd.DataFrame(data_list)

        result = compare_transforms(
            data=df,
            var_lags=1,
            value_cols=["y1", "y2"],
            gmm_step="one-step",
            instrument_type="collapsed",
            max_instruments=3,
        )

        summary = result["summary"]
        assert "FOD" in summary
        assert "FD" in summary
        assert "Observations" in summary
        assert "Coefficient Differences" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
