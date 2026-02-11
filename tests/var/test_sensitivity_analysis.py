"""
Tests for instrument sensitivity analysis in Panel VAR GMM.
"""

import numpy as np
import pytest

from panelbox.var.diagnostics import instrument_sensitivity_analysis


class TestSensitivityAnalysis:
    """Test instrument sensitivity analysis function."""

    def test_sensitivity_analysis_basic(self):
        """Test basic sensitivity analysis functionality."""
        np.random.seed(42)

        # Mock model function that returns different results based on max_instruments
        def mock_model_func(max_instruments, **kwargs):
            # Create a mock result object
            class MockResult:
                def __init__(self, n_instr):
                    # Coefficients that vary slightly with instrument count
                    noise = np.random.randn(6) * 0.01 * (n_instr / 10)
                    self.params_by_eq = [
                        np.array([0.5, 0.3, 0.1]) + noise[:3],
                        np.array([0.2, 0.4, 0.05]) + noise[3:],
                    ]
                    self.n_instruments = min(n_instr, max_instruments)

            return MockResult(max_instruments)

        # Run sensitivity analysis
        result = instrument_sensitivity_analysis(
            model_func=mock_model_func, max_instruments_list=[6, 12, 18, 24]
        )

        # Check structure
        assert "max_instruments" in result
        assert "n_instruments_actual" in result
        assert "coefficients" in result
        assert "coefficient_changes" in result
        assert "max_change_overall" in result
        assert "stable" in result
        assert "interpretation" in result

        # Check values
        assert len(result["max_instruments"]) == 4
        assert len(result["n_instruments_actual"]) == 4
        assert len(result["coefficients"]) > 0

    def test_sensitivity_stable_coefficients(self):
        """Test sensitivity analysis with stable coefficients."""
        np.random.seed(42)

        # Model with truly stable coefficients
        def stable_model_func(max_instruments, **kwargs):
            class MockResult:
                def __init__(self):
                    # Same coefficients regardless of instruments
                    self.params_by_eq = [np.array([0.5, 0.3, 0.1]), np.array([0.2, 0.4, 0.05])]
                    self.n_instruments = max_instruments

            return MockResult()

        result = instrument_sensitivity_analysis(
            model_func=stable_model_func, max_instruments_list=[6, 12, 18, 24]
        )

        # Coefficients should be stable (no change)
        assert result["stable"] is True
        assert result["max_change_overall"] < 1e-10  # Should be exactly zero

    def test_sensitivity_unstable_coefficients(self):
        """Test sensitivity analysis with unstable coefficients."""
        np.random.seed(42)

        # Model with unstable coefficients
        def unstable_model_func(max_instruments, **kwargs):
            class MockResult:
                def __init__(self, n_instr):
                    # Coefficients vary significantly
                    multiplier = n_instr / 10.0
                    self.params_by_eq = [
                        np.array([0.5 * multiplier, 0.3 * multiplier, 0.1]),
                    ]
                    self.n_instruments = n_instr

            return MockResult(max_instruments)

        result = instrument_sensitivity_analysis(
            model_func=unstable_model_func, max_instruments_list=[6, 12, 24]
        )

        # Coefficients should be unstable (>10% change)
        assert result["stable"] is False
        assert result["max_change_overall"] > 10.0

    def test_sensitivity_with_failure(self):
        """Test sensitivity analysis when some estimations fail."""
        np.random.seed(42)

        call_count = [0]

        def failing_model_func(max_instruments, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                # Second call fails
                raise ValueError("Estimation failed")

            class MockResult:
                def __init__(self):
                    self.params_by_eq = [np.array([0.5, 0.3, 0.1])]
                    self.n_instruments = max_instruments

            return MockResult()

        result = instrument_sensitivity_analysis(
            model_func=failing_model_func, max_instruments_list=[6, 12, 18]
        )

        # Should have warnings
        assert len(result["warnings"]) > 0
        # Should still have some successful estimates
        assert len(result["max_instruments"]) == 2  # 1st and 3rd succeeded

    def test_sensitivity_coefficient_extraction_from_params(self):
        """Test coefficient extraction when result has .params instead of .params_by_eq."""
        np.random.seed(42)

        def model_with_params(max_instruments, **kwargs):
            class MockResult:
                def __init__(self):
                    # Single params array instead of params_by_eq
                    self.params = np.array([0.5, 0.3, 0.1, 0.2])
                    self.n_instruments = max_instruments

            return MockResult()

        result = instrument_sensitivity_analysis(
            model_func=model_with_params, max_instruments_list=[6, 12]
        )

        # Should work with .params
        assert len(result["coefficients"]) == 4
        assert result["stable"] is True  # Should be stable

    def test_sensitivity_near_zero_coefficients(self):
        """Test handling of coefficients near zero."""
        np.random.seed(42)

        def near_zero_model(max_instruments, **kwargs):
            class MockResult:
                def __init__(self, n_instr):
                    # Very small coefficients
                    noise = np.random.randn(3) * 1e-12
                    self.params_by_eq = [np.array([1e-11, 1e-10, 1e-11]) + noise]
                    self.n_instruments = n_instr

            return MockResult(max_instruments)

        result = instrument_sensitivity_analysis(
            model_func=near_zero_model, max_instruments_list=[6, 12, 18]
        )

        # Should handle near-zero coefficients without division errors
        assert not np.isnan(result["max_change_overall"])
        assert result["max_change_overall"] >= 0

    def test_sensitivity_with_model_info(self):
        """Test when result has model_info dict with n_instruments."""
        np.random.seed(42)

        def model_with_info(max_instruments, **kwargs):
            class MockResult:
                def __init__(self, n_instr):
                    self.params_by_eq = [np.array([0.5, 0.3])]
                    self.model_info = {"n_instruments": n_instr}

            return MockResult(max_instruments)

        result = instrument_sensitivity_analysis(
            model_func=model_with_info, max_instruments_list=[6, 12]
        )

        # Should extract n_instruments from model_info
        assert result["n_instruments_actual"] == [6, 12]

    def test_sensitivity_interpretation_message(self):
        """Test interpretation messages are generated."""
        np.random.seed(42)

        def stable_model(max_instruments, **kwargs):
            class MockResult:
                def __init__(self):
                    self.params_by_eq = [np.array([0.5, 0.3])]
                    self.n_instruments = max_instruments

            return MockResult()

        result = instrument_sensitivity_analysis(
            model_func=stable_model, max_instruments_list=[6, 12, 18]
        )

        # Should have interpretation
        assert isinstance(result["interpretation"], str)
        assert len(result["interpretation"]) > 0

        # For stable coefficients, should mention stability
        assert "stable" in result["interpretation"].lower()


class TestSensitivityAnalysisEdgeCases:
    """Test edge cases in sensitivity analysis."""

    def test_single_instrument_count(self):
        """Test with only one instrument count (no comparison possible)."""
        np.random.seed(42)

        def mock_model(max_instruments, **kwargs):
            class MockResult:
                def __init__(self):
                    self.params_by_eq = [np.array([0.5, 0.3])]
                    self.n_instruments = max_instruments

            return MockResult()

        result = instrument_sensitivity_analysis(model_func=mock_model, max_instruments_list=[12])

        # Should work but have no change calculations
        assert len(result["max_instruments"]) == 1
        assert len(result["coefficient_changes"]) == 0

    def test_all_estimations_fail(self):
        """Test when all estimations fail."""

        def failing_model(max_instruments, **kwargs):
            raise RuntimeError("All estimations fail")

        result = instrument_sensitivity_analysis(
            model_func=failing_model, max_instruments_list=[6, 12, 18]
        )

        # Should have warnings
        assert len(result["warnings"]) == 3
        # Should have empty results
        assert len(result["max_instruments"]) == 0
        assert result["max_change_overall"] == 0.0

    def test_kwargs_passed_to_model(self):
        """Test that additional kwargs are passed to model function."""
        received_kwargs = {}

        def model_capturing_kwargs(max_instruments, **kwargs):
            received_kwargs.update(kwargs)

            class MockResult:
                def __init__(self):
                    self.params_by_eq = [np.array([0.5])]
                    self.n_instruments = max_instruments

            return MockResult()

        instrument_sensitivity_analysis(
            model_func=model_capturing_kwargs,
            max_instruments_list=[6],
            custom_param=42,
            another_param="test",
        )

        # Check kwargs were passed
        assert received_kwargs["custom_param"] == 42
        assert received_kwargs["another_param"] == "test"

    def test_empty_instrument_list(self):
        """Test with empty instrument list."""

        def mock_model(max_instruments, **kwargs):
            class MockResult:
                def __init__(self):
                    self.params_by_eq = [np.array([0.5])]
                    self.n_instruments = max_instruments

            return MockResult()

        result = instrument_sensitivity_analysis(model_func=mock_model, max_instruments_list=[])

        # Should return empty results
        assert len(result["max_instruments"]) == 0
        assert result["max_change_overall"] == 0.0
        assert result["stable"] is True  # Trivially stable

    def test_coefficient_percentage_change_calculation(self):
        """Test percentage change calculation is correct."""
        np.random.seed(42)

        def varying_model(max_instruments, **kwargs):
            class MockResult:
                def __init__(self, n_instr):
                    # Simple linear variation
                    if n_instr == 6:
                        coef = 1.0
                    elif n_instr == 12:
                        coef = 1.1  # 10% increase
                    else:
                        coef = 1.2  # 20% increase

                    self.params_by_eq = [np.array([coef])]
                    self.n_instruments = n_instr

            return MockResult(max_instruments)

        result = instrument_sensitivity_analysis(
            model_func=varying_model, max_instruments_list=[6, 12, 18]
        )

        # Check percentage change is approximately 20%
        assert abs(result["max_change_overall"] - 20.0) < 1.0
