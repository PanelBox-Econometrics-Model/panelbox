"""
Tests for LLC (Levin-Lin-Chu) panel unit root test.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.validation.unit_root import LLCTest


class TestLLCTest:
    """Test suite for LLC panel unit root test."""

    @pytest.fixture
    def stationary_panel_data(self):
        """Generate stationary panel data."""
        np.random.seed(42)
        n_entities = 10
        n_time = 50

        data_list = []
        for i in range(n_entities):
            # AR(1) with |rho| < 1 (stationary)
            rho = 0.5
            y = np.zeros(n_time)
            y[0] = np.random.normal(0, 1)
            for t in range(1, n_time):
                y[t] = rho * y[t - 1] + np.random.normal(0, 1)

            entity_data = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
            data_list.append(entity_data)

        return pd.concat(data_list, ignore_index=True)

    @pytest.fixture
    def unit_root_panel_data(self):
        """Generate panel data with unit root."""
        np.random.seed(123)
        n_entities = 10
        n_time = 50

        data_list = []
        for i in range(n_entities):
            # Random walk (unit root)
            y = np.cumsum(np.random.normal(0, 1, n_time))

            entity_data = pd.DataFrame({"entity": i, "time": range(n_time), "y": y})
            data_list.append(entity_data)

        return pd.concat(data_list, ignore_index=True)

    @pytest.fixture
    def grunfeld_data(self):
        """Load Grunfeld dataset."""
        try:
            import panelbox as pb

            return pb.load_grunfeld()
        except:
            pytest.skip("Grunfeld dataset not available")

    def test_llc_initialization(self, stationary_panel_data):
        """Test LLC test initialization."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")

        assert llc.variable == "y"
        assert llc.entity_col == "entity"
        assert llc.time_col == "time"
        assert llc.lags == 1
        assert llc.trend == "c"
        assert llc.n_entities == 10
        assert llc.result is None

    def test_llc_invalid_variable(self, stationary_panel_data):
        """Test error when variable not found."""
        with pytest.raises(ValueError, match="Variable 'invalid' not found"):
            LLCTest(stationary_panel_data, "invalid", "entity", "time")

    def test_llc_invalid_entity_col(self, stationary_panel_data):
        """Test error when entity column not found."""
        with pytest.raises(ValueError, match="Entity column 'invalid' not found"):
            LLCTest(stationary_panel_data, "y", "invalid", "time")

    def test_llc_invalid_time_col(self, stationary_panel_data):
        """Test error when time column not found."""
        with pytest.raises(ValueError, match="Time column 'invalid' not found"):
            LLCTest(stationary_panel_data, "y", "entity", "invalid")

    def test_llc_invalid_trend(self, stationary_panel_data):
        """Test error with invalid trend specification."""
        with pytest.raises(ValueError, match="trend must be 'n', 'c', or 'ct'"):
            LLCTest(stationary_panel_data, "y", "entity", "time", trend="invalid")

    def test_llc_stationary_data(self, stationary_panel_data):
        """Test LLC on stationary data."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")
        result = llc.run()

        assert result is not None
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert result.lags == 1
        assert result.n_obs > 0
        assert result.n_entities == 10
        assert result.test_type == "LLC"
        assert result.deterministics == "Constant"

        # For stationary data, should reject null (low p-value)
        # Note: This is probabilistic, but with strong stationarity should work
        assert result.pvalue < 0.10  # 10% level

    def test_llc_unit_root_data(self, unit_root_panel_data):
        """Test LLC on unit root data."""
        llc = LLCTest(unit_root_panel_data, "y", "entity", "time", lags=1, trend="c")
        result = llc.run()

        assert result is not None
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)

        # For unit root data, should fail to reject null (high p-value)
        # Note: This is also probabilistic
        assert result.pvalue > 0.05  # 5% level

    def test_llc_no_trend(self, stationary_panel_data):
        """Test LLC with no deterministic terms."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="n")
        result = llc.run()

        assert result.deterministics == "None"
        assert result.statistic is not None

    def test_llc_constant_and_trend(self, stationary_panel_data):
        """Test LLC with constant and trend."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="ct")
        result = llc.run()

        assert result.deterministics == "Constant and Trend"
        assert result.statistic is not None

    def test_llc_auto_lag_selection(self, stationary_panel_data):
        """Test automatic lag selection."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=None, trend="c")
        result = llc.run()

        assert result.lags >= 0
        assert isinstance(result.lags, int)

    def test_llc_multiple_lags(self, stationary_panel_data):
        """Test LLC with multiple lags."""
        for lags in [0, 1, 2, 3]:
            llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=lags, trend="c")
            result = llc.run()

            assert result.lags == lags
            assert result.statistic is not None

    def test_llc_result_string_representation(self, stationary_panel_data):
        """Test string representation of results."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")
        result = llc.run()

        result_str = str(result)
        assert "Levin-Lin-Chu Panel Unit Root Test" in result_str
        assert f"Test statistic:" in result_str
        assert f"P-value:" in result_str
        assert f"Lags:" in result_str
        assert "H0:" in result_str
        assert "H1:" in result_str

    def test_llc_conclusion_property(self, stationary_panel_data):
        """Test conclusion property."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")
        result = llc.run()

        conclusion = result.conclusion
        assert isinstance(conclusion, str)
        assert "H0" in conclusion

    def test_llc_with_grunfeld_invest(self, grunfeld_data):
        """Test LLC on Grunfeld investment variable."""
        llc = LLCTest(grunfeld_data, "invest", "firm", "year", lags=1, trend="c")
        result = llc.run()

        assert result is not None
        assert result.n_entities > 0
        assert result.n_obs > 0
        print(f"\nGrunfeld 'invest' LLC test:")
        print(f"  Statistic: {result.statistic:.4f}")
        print(f"  P-value: {result.pvalue:.4f}")
        print(f"  Conclusion: {result.conclusion}")

    def test_llc_with_grunfeld_value(self, grunfeld_data):
        """Test LLC on Grunfeld firm value variable."""
        llc = LLCTest(grunfeld_data, "value", "firm", "year", lags=1, trend="c")
        result = llc.run()

        assert result is not None
        print(f"\nGrunfeld 'value' LLC test:")
        print(f"  Statistic: {result.statistic:.4f}")
        print(f"  P-value: {result.pvalue:.4f}")
        print(f"  Conclusion: {result.conclusion}")

    def test_llc_with_grunfeld_capital(self, grunfeld_data):
        """Test LLC on Grunfeld capital variable."""
        llc = LLCTest(grunfeld_data, "capital", "firm", "year", lags=1, trend="c")
        result = llc.run()

        assert result is not None
        print(f"\nGrunfeld 'capital' LLC test:")
        print(f"  Statistic: {result.statistic:.4f}")
        print(f"  P-value: {result.pvalue:.4f}")
        print(f"  Conclusion: {result.conclusion}")

    def test_llc_unbalanced_panel_warning(self):
        """Test warning for unbalanced panel."""
        # Create unbalanced panel
        data = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 3, 3, 3, 3],
                "time": [1, 2, 3, 1, 2, 1, 2, 3, 4],
                "y": np.random.randn(9),
            }
        )

        with pytest.warns(UserWarning, match="unbalanced"):
            llc = LLCTest(data, "y", "entity", "time", lags=0, trend="c")

    def test_llc_small_sample(self):
        """Test LLC with very small sample."""
        # Small panel that should still work
        data = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
                "y": np.random.randn(10),
            }
        )

        llc = LLCTest(data, "y", "entity", "time", lags=0, trend="c")
        result = llc.run()

        assert result is not None
        assert result.n_entities == 2

    def test_llc_insufficient_data(self):
        """Test error with insufficient data."""
        # Too little data for any meaningful test
        data = pd.DataFrame({"entity": [1, 1, 2, 2], "time": [1, 2, 1, 2], "y": [1, 2, 3, 4]})

        llc = LLCTest(data, "y", "entity", "time", lags=1, trend="c")

        with pytest.raises(ValueError, match="Insufficient data"):
            llc.run()

    def test_llc_deterministic_specifications(self, stationary_panel_data):
        """Test all deterministic specifications produce valid results."""
        trends = ["n", "c", "ct"]
        expected_dets = ["None", "Constant", "Constant and Trend"]

        for trend, expected_det in zip(trends, expected_dets):
            llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend=trend)
            result = llc.run()

            assert result.deterministics == expected_det
            assert not np.isnan(result.statistic)
            assert not np.isnan(result.pvalue)
            assert 0 <= result.pvalue <= 1

    def test_llc_reproducibility(self, stationary_panel_data):
        """Test that results are reproducible."""
        llc1 = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")
        result1 = llc1.run()

        llc2 = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")
        result2 = llc2.run()

        assert result1.statistic == result2.statistic
        assert result1.pvalue == result2.pvalue
        assert result1.lags == result2.lags

    def test_llc_result_attributes(self, stationary_panel_data):
        """Test all result attributes are present and valid."""
        llc = LLCTest(stationary_panel_data, "y", "entity", "time", lags=1, trend="c")
        result = llc.run()

        # Check all attributes exist
        assert hasattr(result, "statistic")
        assert hasattr(result, "pvalue")
        assert hasattr(result, "lags")
        assert hasattr(result, "n_obs")
        assert hasattr(result, "n_entities")
        assert hasattr(result, "test_type")
        assert hasattr(result, "deterministics")
        assert hasattr(result, "null_hypothesis")
        assert hasattr(result, "alternative_hypothesis")
        assert hasattr(result, "conclusion")

        # Check types
        assert isinstance(result.statistic, float)
        assert isinstance(result.pvalue, float)
        assert isinstance(result.lags, int)
        assert isinstance(result.n_obs, int)
        assert isinstance(result.n_entities, int)
        assert isinstance(result.test_type, str)
        assert isinstance(result.deterministics, str)
        assert isinstance(result.null_hypothesis, str)
        assert isinstance(result.alternative_hypothesis, str)
        assert isinstance(result.conclusion, str)

        # Check ranges
        assert 0 <= result.pvalue <= 1
        assert result.lags >= 0
        assert result.n_obs > 0
        assert result.n_entities > 0


class TestLLCTestIntegration:
    """Integration tests for LLC test."""

    def test_llc_typical_workflow(self):
        """Test typical workflow with LLC test."""
        # Generate data
        np.random.seed(999)
        n_entities = 5
        n_time = 30

        data_list = []
        for i in range(n_entities):
            # Mildly stationary process
            rho = 0.7
            y = np.zeros(n_time)
            y[0] = np.random.normal(0, 1)
            for t in range(1, n_time):
                y[t] = rho * y[t - 1] + np.random.normal(0, 0.5)

            data_list.append(pd.DataFrame({"id": i, "period": range(n_time), "variable": y}))

        data = pd.concat(data_list, ignore_index=True)

        # Run test with auto lag selection
        llc = LLCTest(data, "variable", "id", "period", lags=None, trend="c")
        result = llc.run()

        # Print results
        print("\n" + "=" * 70)
        print("Integration Test: Typical LLC Workflow")
        print("=" * 70)
        print(result)

        assert result is not None
        assert result.pvalue is not None

    def test_llc_comparison_different_trends(self):
        """Test LLC with different trend specifications."""
        np.random.seed(456)
        n_entities = 8
        n_time = 40

        data_list = []
        for i in range(n_entities):
            # Stationary with trend
            t = np.arange(n_time)
            y = 0.1 * t + np.random.randn(n_time)
            for i_t in range(1, n_time):
                y[i_t] += 0.3 * y[i_t - 1]

            data_list.append(pd.DataFrame({"firm": i, "year": range(n_time), "series": y}))

        data = pd.concat(data_list, ignore_index=True)

        print("\n" + "=" * 70)
        print("Integration Test: Comparing Different Trend Specifications")
        print("=" * 70)

        for trend, name in [("n", "No trend"), ("c", "Constant"), ("ct", "Constant and trend")]:
            llc = LLCTest(data, "series", "firm", "year", lags=1, trend=trend)
            result = llc.run()
            print(f"\n{name}:")
            print(f"  Statistic: {result.statistic:.4f}")
            print(f"  P-value: {result.pvalue:.4f}")
            print(f"  Conclusion: {result.conclusion}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
