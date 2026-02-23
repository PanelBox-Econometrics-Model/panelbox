"""
Tests for unified panel unit root test interface.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.unit_root import PanelUnitRootResult, panel_unit_root_test


class TestPanelUnitRootTest:
    """Test suite for unified panel unit root test."""

    @pytest.fixture
    def stationary_panel(self):
        """Generate a balanced panel of stationary series."""
        np.random.seed(42)
        data = []
        N = 10
        T = 100

        for i in range(N):
            y = np.zeros(T)
            y[0] = np.random.randn()
            for t in range(1, T):
                y[t] = 0.6 * y[t - 1] + np.random.randn()

            for t in range(T):
                data.append({"entity": i, "time": t, "y": y[t]})

        return pd.DataFrame(data)

    @pytest.fixture
    def unit_root_panel(self):
        """Generate a balanced panel with unit roots."""
        np.random.seed(123)
        data = []
        N = 10
        T = 100

        for i in range(N):
            y = np.random.randn(T).cumsum()

            for t in range(T):
                data.append({"entity": i, "time": t, "y": y[t]})

        return pd.DataFrame(data)

    def test_run_all_tests(self, stationary_panel):
        """Test running all available tests."""
        result = panel_unit_root_test(stationary_panel, "y", test="all", trend="c")

        assert isinstance(result, PanelUnitRootResult)
        assert len(result.results) >= 2  # At least hadri and breitung
        assert "hadri" in result.results
        assert "breitung" in result.results

    def test_run_single_test(self, stationary_panel):
        """Test running a single test."""
        result = panel_unit_root_test(stationary_panel, "y", test="hadri", trend="c")

        assert len(result.results) == 1
        assert "hadri" in result.results
        assert result.tests_run == ["hadri"]

    def test_run_multiple_tests(self, stationary_panel):
        """Test running multiple specific tests."""
        result = panel_unit_root_test(stationary_panel, "y", test=["hadri", "breitung"], trend="c")

        assert len(result.results) == 2
        assert "hadri" in result.results
        assert "breitung" in result.results

    def test_result_attributes(self, stationary_panel):
        """Test that result has expected attributes."""
        result = panel_unit_root_test(stationary_panel, "y", test="all")

        assert hasattr(result, "results")
        assert hasattr(result, "variable")
        assert hasattr(result, "n_entities")
        assert hasattr(result, "n_time")
        assert hasattr(result, "tests_run")

        assert result.variable == "y"
        assert result.n_entities == 10
        assert result.n_time == 100

    def test_summary_table(self, stationary_panel):
        """Test summary table generation."""
        result = panel_unit_root_test(stationary_panel, "y", test=["hadri", "breitung"], trend="c")

        summary = result.summary_table()

        assert isinstance(summary, str)
        assert "Panel Unit Root Test Summary" in summary
        assert "HADRI" in summary
        assert "BREITUNG" in summary
        assert "Statistic" in summary
        assert "P-value" in summary

    def test_interpretation(self, stationary_panel):
        """Test interpretation generation."""
        result = panel_unit_root_test(stationary_panel, "y", test=["hadri", "breitung"], trend="c")

        interpretation = result.interpretation()

        assert isinstance(interpretation, str)
        assert "Interpretation" in interpretation

    def test_invalid_test_name(self, stationary_panel):
        """Test that invalid test name raises error."""
        with pytest.raises(ValueError, match="Unknown test"):
            panel_unit_root_test(stationary_panel, "y", test="invalid_test")

    def test_repr(self, stationary_panel):
        """Test string representation."""
        result = panel_unit_root_test(stationary_panel, "y", test=["hadri", "breitung"])

        repr_str = repr(result)

        assert "PanelUnitRootResult" in repr_str
        assert "hadri" in repr_str or "breitung" in repr_str

    def test_different_trends(self, stationary_panel):
        """Test with different trend specifications."""
        result_c = panel_unit_root_test(stationary_panel, "y", test="hadri", trend="c")
        result_ct = panel_unit_root_test(stationary_panel, "y", test="hadri", trend="ct")

        assert result_c.results["hadri"].trend == "c"
        assert result_ct.results["hadri"].trend == "ct"

        # Results should differ
        assert result_c.results["hadri"].statistic != result_ct.results["hadri"].statistic
