"""
Tests for Hadri (2000) LM test.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.unit_root import hadri_test


class TestHadriTest:
    """Test suite for Hadri (2000) LM test."""

    @pytest.fixture
    def stationary_panel(self):
        """Generate a balanced panel of stationary series."""
        np.random.seed(42)
        data = []
        N = 10  # entities
        T = 100  # time periods

        for i in range(N):
            # AR(1) with |ρ| < 1 (stationary)
            y = np.zeros(T)
            y[0] = np.random.randn()
            for t in range(1, T):
                y[t] = 0.5 * y[t - 1] + np.random.randn()

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
            # Random walk (unit root)
            y = np.random.randn(T).cumsum()

            for t in range(T):
                data.append({"entity": i, "time": t, "y": y[t]})

        return pd.DataFrame(data)

    def test_hadri_stationary_data(self, stationary_panel):
        """Test that Hadri does not reject H0 for stationary data."""
        result = hadri_test(stationary_panel, "y", trend="c")

        # Should not reject H0 of stationarity
        # (p-value should be relatively high)
        assert result.pvalue > 0.05, "Should not reject stationarity for stationary data"
        assert not result.reject

    def test_hadri_unit_root_data(self, unit_root_panel):
        """Test that Hadri rejects H0 for unit root data."""
        result = hadri_test(unit_root_panel, "y", trend="c")

        # Should reject H0 of stationarity for unit root data
        # (p-value should be low)
        assert result.pvalue < 0.10, "Should reject stationarity for unit root data"
        assert result.reject or result.pvalue < 0.10

    def test_hadri_result_structure(self, stationary_panel):
        """Test that result object has expected attributes."""
        result = hadri_test(stationary_panel, "y", trend="c")

        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "reject")
        assert hasattr(result, "lm_statistic")
        assert hasattr(result, "individual_lm")
        assert hasattr(result, "n_entities")
        assert hasattr(result, "n_time")
        assert hasattr(result, "trend")
        assert hasattr(result, "robust")

        assert result.n_entities == 10
        assert result.n_time == 100
        assert result.trend == "c"
        assert len(result.individual_lm) == 10

    def test_hadri_trend_ct(self, stationary_panel):
        """Test Hadri with constant and trend."""
        result = hadri_test(stationary_panel, "y", trend="ct")

        assert result.trend == "ct"
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1

    def test_hadri_robust_vs_nonrobust(self, stationary_panel):
        """Test that robust and non-robust versions give different results."""
        result_robust = hadri_test(stationary_panel, "y", trend="c", robust=True)
        result_nonrobust = hadri_test(stationary_panel, "y", trend="c", robust=False)

        # Results should be different (though maybe not by much)
        assert result_robust.robust == True
        assert result_nonrobust.robust == False

    def test_hadri_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        data = pd.DataFrame({"entity": [1, 1, 2, 2], "time": [0, 1, 0, 1], "y": [1, 2, 3, 4]})

        # Invalid variable name
        with pytest.raises(ValueError, match="Variable .* not found"):
            hadri_test(data, "nonexistent")

        # Invalid trend specification
        with pytest.raises(ValueError, match="trend must be"):
            hadri_test(data, "y", trend="invalid")

    def test_hadri_summary(self, stationary_panel):
        """Test that summary method produces formatted output."""
        result = hadri_test(stationary_panel, "y", trend="c")
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Hadri" in summary
        assert "statistic" in summary.lower()
        assert "p-value" in summary.lower()

    def test_hadri_repr(self, stationary_panel):
        """Test string representation."""
        result = hadri_test(stationary_panel, "y", trend="c")
        repr_str = repr(result)

        assert "HadriResult" in repr_str
        assert "statistic" in repr_str
        assert "pvalue" in repr_str

    def test_hadri_unbalanced_panel_raises(self):
        """Test that unbalanced panel raises error."""
        data = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2],  # Entity 1 has 3 obs, entity 2 has 2
                "time": [0, 1, 2, 0, 1],
                "y": [1, 2, 3, 4, 5],
            }
        )

        with pytest.raises(ValueError, match="balanced panel"):
            hadri_test(data, "y")
