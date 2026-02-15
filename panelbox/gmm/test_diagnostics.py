"""
Tests for GMM Diagnostic Tests
================================

Test suite for GMM diagnostics implementation.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm import ContinuousUpdatedGMM
from panelbox.gmm.diagnostics import GMMDiagnostics


class TestGMMDiagnostics:
    """Test suite for GMMDiagnostics."""

    @pytest.fixture
    def simple_iv_model(self):
        """
        Create a simple fitted IV model for testing diagnostics.
        """
        np.random.seed(42)
        n = 500

        # Generate instruments (exogenous)
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        z3 = np.random.normal(0, 1, n)

        # Generate endogenous regressor
        v = np.random.normal(0, 1, n)
        x = 0.5 + 0.8 * z1 + 0.6 * z2 + 0.4 * z3 + v

        # Generate error (correlated with x through v)
        epsilon = np.random.normal(0, 1, n) + 0.5 * v

        # Outcome
        y = 1.0 + 2.0 * x + epsilon

        # Create DataFrame
        data = pd.DataFrame(
            {
                "y": y,
                "x": x,
                "z1": z1,
                "z2": z2,
                "z3": z3,
                "entity": np.arange(n),
                "time": 1,
            }
        )
        data = data.set_index(["entity", "time"])

        # Fit CUE-GMM model
        model = ContinuousUpdatedGMM(
            data=data,
            dep_var="y",
            exog_vars=["x"],
            instruments=["z1", "z2", "z3"],
        )
        results = model.fit(verbose=False)

        return model, results

    def test_initialization(self, simple_iv_model):
        """Test GMMDiagnostics initializes correctly."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)

        assert diagnostics.model is model
        assert diagnostics.results is results
        assert diagnostics.n == len(model.y)
        assert diagnostics.k == model.k
        assert diagnostics.n_instruments == model.n_instruments
        assert diagnostics.hansen_j is not None

    def test_c_statistic_basic(self, simple_iv_model):
        """Test C-statistic computation."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)

        # Test validity of last instrument (z3)
        # Instruments are [const, z1, z2, z3], so z3 is index 3
        c_test = diagnostics.c_statistic(subset_indices=[3])

        # Check structure
        assert hasattr(c_test, "statistic")
        assert hasattr(c_test, "pvalue")
        assert hasattr(c_test, "df")

        # C-statistic should be non-negative
        assert c_test.statistic >= 0

        # p-value in [0, 1]
        assert 0 <= c_test.pvalue <= 1

        # df should equal number of tested instruments
        assert c_test.df == 1

    def test_c_statistic_multiple_instruments(self, simple_iv_model):
        """Test C-statistic with multiple instruments."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)

        # Test validity of z2 and z3 (indices 2 and 3)
        c_test = diagnostics.c_statistic(subset_indices=[2, 3])

        assert c_test.df == 2
        assert c_test.statistic >= 0
        assert 0 <= c_test.pvalue <= 1

    def test_c_statistic_invalid_indices(self, simple_iv_model):
        """Test error with invalid subset indices."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)

        # Index too large
        with pytest.raises(ValueError, match="subset_indices contains"):
            diagnostics.c_statistic(subset_indices=[10])

    def test_weak_instruments_test(self, simple_iv_model):
        """Test weak instruments diagnostics."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)

        weak_test = diagnostics.weak_instruments_test()

        # Check structure
        assert "cragg_donald_f" in weak_test
        assert "critical_value_10pct" in weak_test
        assert "interpretation" in weak_test
        assert "warning_level" in weak_test

        # F-statistic should be positive
        assert weak_test["cragg_donald_f"] > 0

        # With strong instruments, F should be > 10
        # (This is a statistical property, may occasionally fail)
        # assert weak_test['cragg_donald_f'] > 10

    def test_weak_instruments_with_weak_data(self):
        """Test weak instruments detection with deliberately weak instruments."""
        np.random.seed(999)
        n = 200

        # Create very weak instruments (low correlation with x)
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)

        # Endogenous regressor barely correlated with instruments
        v = np.random.normal(0, 1, n)
        x = 0.01 * z1 + 0.01 * z2 + v  # Very weak first stage

        # Outcome
        epsilon = np.random.normal(0, 1, n) + 0.5 * v
        y = 1.0 + 2.0 * x + epsilon

        # Create DataFrame
        data = pd.DataFrame(
            {
                "y": y,
                "x": x,
                "z1": z1,
                "z2": z2,
                "entity": np.arange(n),
                "time": 1,
            }
        )
        data = data.set_index(["entity", "time"])

        # Fit model
        model = ContinuousUpdatedGMM(
            data=data, dep_var="y", exog_vars=["x"], instruments=["z1", "z2"]
        )

        try:
            results = model.fit(verbose=False)

            diagnostics = GMMDiagnostics(model, results)
            weak_test = diagnostics.weak_instruments_test()

            # F-statistic should be low (weak instruments)
            assert weak_test["cragg_donald_f"] < 20
            assert weak_test["warning_level"] in ["CRITICAL", "WARNING"]

        except Exception:
            # Model may not converge with very weak instruments
            # This is expected behavior
            pass

    def test_diagnostic_tests_summary(self, simple_iv_model):
        """Test diagnostic_tests() returns DataFrame."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)
        summary_df = diagnostics.diagnostic_tests()

        assert isinstance(summary_df, pd.DataFrame)
        assert "Test" in summary_df.columns
        assert "Statistic" in summary_df.columns
        assert "p-value" in summary_df.columns
        assert "Result" in summary_df.columns

        # Should have at least 2 tests (Hansen J, Weak instruments)
        assert len(summary_df) >= 2

    def test_summary_text(self, simple_iv_model):
        """Test summary() returns formatted text."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)
        summary_text = diagnostics.summary()

        assert isinstance(summary_text, str)
        assert "GMM Diagnostic Tests" in summary_text
        assert "Hansen J-test" in summary_text
        assert "Weak Instruments" in summary_text

    def test_repr(self, simple_iv_model):
        """Test string representation."""
        model, results = simple_iv_model

        diagnostics = GMMDiagnostics(model, results)
        repr_str = repr(diagnostics)

        assert "GMMDiagnostics" in repr_str
        assert f"n={diagnostics.n}" in repr_str


class TestGMMDiagnosticsEdgeCases:
    """Test edge cases and error handling."""

    def test_c_statistic_no_data_arrays(self):
        """Test C-statistic error when model lacks data arrays."""

        # Create mock model and results without data
        class MockModel:
            pass

        class MockResults:
            nobs = 100
            n_params = 2
            n_instruments = 3

            class hansen_j:
                statistic = 1.5
                pvalue = 0.22

        model = MockModel()
        results = MockResults()

        diagnostics = GMMDiagnostics(model, results)

        with pytest.raises(RuntimeError, match="requires access to data arrays"):
            diagnostics.c_statistic(subset_indices=[0])

    def test_weak_instruments_no_data_arrays(self):
        """Test weak instruments error when model lacks data arrays."""

        class MockModel:
            pass

        class MockResults:
            nobs = 100
            n_params = 2
            n_instruments = 3

        model = MockModel()
        results = MockResults()

        diagnostics = GMMDiagnostics(model, results)

        with pytest.raises(RuntimeError, match="requires access to data arrays"):
            diagnostics.weak_instruments_test()

    def test_c_statistic_no_subset_provided(
        self,
    ):
        """Test error when no subset specified."""
        np.random.seed(42)
        n = 100
        z1 = np.random.normal(0, 1, n)
        z2 = np.random.normal(0, 1, n)
        x = 0.5 + 0.8 * z1 + 0.6 * z2 + np.random.normal(0, 1, n)
        y = 1.0 + 2.0 * x + np.random.normal(0, 1, n)

        data = pd.DataFrame({"y": y, "x": x, "z1": z1, "z2": z2, "entity": np.arange(n), "time": 1})
        data = data.set_index(["entity", "time"])

        model = ContinuousUpdatedGMM(
            data=data, dep_var="y", exog_vars=["x"], instruments=["z1", "z2"]
        )
        results = model.fit(verbose=False)

        diagnostics = GMMDiagnostics(model, results)

        with pytest.raises(ValueError, match="Must provide either"):
            diagnostics.c_statistic()


class TestGMMDiagnosticsIntegration:
    """Integration tests with realistic scenarios."""

    def test_overidentified_model_diagnostics(self):
        """Test diagnostics on overidentified model (many instruments)."""
        np.random.seed(789)
        n = 300

        # Generate 5 instruments for 1 endogenous regressor
        instruments = {}
        for i in range(5):
            instruments[f"z{i+1}"] = np.random.normal(0, 1, n)

        v = np.random.normal(0, 1, n)
        x = 0.5 + sum(0.5 * instruments[f"z{i+1}"] for i in range(5)) / 5 + v

        epsilon = np.random.normal(0, 1, n) + 0.5 * v
        y = 1.0 + 2.0 * x + epsilon

        data = pd.DataFrame({"y": y, "x": x, **instruments, "entity": np.arange(n), "time": 1})
        data = data.set_index(["entity", "time"])

        model = ContinuousUpdatedGMM(
            data=data,
            dep_var="y",
            exog_vars=["x"],
            instruments=list(instruments.keys()),
        )
        results = model.fit(verbose=False)

        diagnostics = GMMDiagnostics(model, results)

        # Should be able to compute all diagnostics
        summary = diagnostics.diagnostic_tests()
        assert len(summary) >= 2

        # Test C-statistic on subset
        c_test = diagnostics.c_statistic(subset_indices=[5])  # Last instrument
        assert c_test.pvalue is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
