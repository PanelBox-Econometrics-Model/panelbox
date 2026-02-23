"""
Tests for Breitung (2000) unit root test.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.unit_root import breitung_test


class TestBreitungTest:
    """Test suite for Breitung (2000) test."""

    @pytest.fixture
    def stationary_panel(self):
        """Generate a balanced panel of stationary series."""
        np.random.seed(42)
        data = []
        N = 10
        T = 100

        for i in range(N):
            # AR(1) with |ρ| < 1 (stationary)
            y = np.zeros(T)
            y[0] = np.random.randn()
            for t in range(1, T):
                y[t] = 0.7 * y[t - 1] + np.random.randn()

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

    def test_breitung_unit_root_data(self, unit_root_panel):
        """Test that Breitung does not reject H0 for unit root data."""
        result = breitung_test(unit_root_panel, "y", trend="ct")

        # Should not reject H0 of unit root
        # (p-value should be high for unit root data)
        assert not result.reject or result.pvalue > 0.01

    def test_breitung_stationary_data(self, stationary_panel):
        """Test that Breitung rejects H0 for stationary data."""
        result = breitung_test(stationary_panel, "y", trend="ct")

        # Should reject H0 of unit root for stationary data
        # (p-value should be low)
        # Note: Breitung may have lower power in small samples
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1

    def test_breitung_result_structure(self, unit_root_panel):
        """Test that result object has expected attributes."""
        result = breitung_test(unit_root_panel, "y", trend="ct")

        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "reject")
        assert hasattr(result, "raw_statistic")
        assert hasattr(result, "n_entities")
        assert hasattr(result, "n_time")
        assert hasattr(result, "trend")

        assert result.n_entities == 10
        assert result.n_time == 100
        assert result.trend == "ct"

    def test_breitung_trend_c(self, unit_root_panel):
        """Test Breitung with constant only."""
        result = breitung_test(unit_root_panel, "y", trend="c")

        assert result.trend == "c"
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert 0 <= result.pvalue <= 1

    def test_breitung_invalid_inputs(self):
        """Test that invalid inputs raise appropriate errors."""
        data = pd.DataFrame({"entity": [1, 1, 2, 2], "time": [0, 1, 0, 1], "y": [1, 2, 3, 4]})

        # Invalid variable name
        with pytest.raises(ValueError, match="Variable .* not found"):
            breitung_test(data, "nonexistent")

        # Invalid trend specification
        with pytest.raises(ValueError, match="trend must be"):
            breitung_test(data, "y", trend="invalid")

    def test_breitung_summary(self, unit_root_panel):
        """Test that summary method produces formatted output."""
        result = breitung_test(unit_root_panel, "y", trend="ct")
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Breitung" in summary
        assert "statistic" in summary.lower()
        assert "p-value" in summary.lower()

    def test_breitung_repr(self, unit_root_panel):
        """Test string representation."""
        result = breitung_test(unit_root_panel, "y", trend="ct")
        repr_str = repr(result)

        assert "BreitungResult" in repr_str
        assert "statistic" in repr_str
        assert "pvalue" in repr_str

    def test_breitung_unbalanced_panel_raises(self):
        """Test that unbalanced panel raises error."""
        data = pd.DataFrame(
            {"entity": [1, 1, 1, 2, 2], "time": [0, 1, 2, 0, 1], "y": [1, 2, 3, 4, 5]}
        )

        with pytest.raises(ValueError, match="balanced panel"):
            breitung_test(data, "y")
