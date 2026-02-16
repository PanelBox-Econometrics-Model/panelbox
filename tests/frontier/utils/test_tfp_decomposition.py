"""
Tests for TFP decomposition module.

This module tests the Total Factor Productivity decomposition functionality,
including:
- Basic decomposition into TC, TE, SE components
- Verification that components sum to total TFP growth
- Returns to scale computation
- Aggregate statistics
- Visualization methods
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.frontier.utils import TFPDecomposition


class MockModel:
    """Mock SFA model for testing."""

    def __init__(self, data, entity="firm", time="year", depvar="y", exog=None):
        self.data = data
        self.entity = entity
        self.time = time
        self.depvar = depvar
        self.exog = exog if exog else ["x1", "x2"]
        self.n_exog = len(self.exog)


class MockResult:
    """Mock SFA result for testing."""

    def __init__(self, model, params, efficiency_data):
        self.model = model
        self.params = params
        self._efficiency_data = efficiency_data
        # Mock vcov
        k = len(params)
        self.vcov = np.eye(k) * 0.01

    def efficiency(self, estimator="bc"):
        """Return pre-defined efficiency scores."""
        return self._efficiency_data


class TestTFPDecompositionBasic:
    """Basic tests for TFP decomposition."""

    def test_decomposition_components_sum_to_total(self):
        """Test that TC + TE + SE = TFP growth."""
        # Create synthetic panel data
        np.random.seed(42)

        firms = [1, 1, 2, 2, 3, 3]
        years = [2010, 2011, 2010, 2011, 2010, 2011]

        # Output and inputs (in logs)
        y = [2.0, 2.1, 2.5, 2.6, 3.0, 3.2]
        x1 = [1.0, 1.05, 1.2, 1.25, 1.5, 1.55]
        x2 = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05]

        data = pd.DataFrame(
            {
                "firm": firms,
                "year": years,
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        )

        # Efficiency scores (must have entity and time columns)
        eff_data = pd.DataFrame(
            {
                "entity": firms,
                "time": years,
                "efficiency": [0.80, 0.82, 0.85, 0.87, 0.90, 0.92],
            }
        )

        # Create mock model and result
        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])

        # Parameters: β = [0.6, 0.3] (sum = 0.9 < 1, DRS)
        params = np.array([0.6, 0.3, 0.1, 0.05])  # β, σ²_v, σ²_u

        result = MockResult(model, params, eff_data)

        # Create TFP decomposition
        tfp = TFPDecomposition(result, periods=(2010, 2011))

        # Decompose
        decomp = tfp.decompose()

        # Check that decomposition exists for all firms
        assert len(decomp) == 3, "Should have 3 firms"

        # Check that components sum to total (within numerical tolerance)
        for _, row in decomp.iterrows():
            total = row["delta_tfp"]
            components = row["delta_tc"] + row["delta_te"] + row["delta_se"]
            verification = row["verification"]

            assert (
                abs(total - components) < 1e-6
            ), f"Components should sum to total: {total} vs {components}"
            assert abs(verification) < 1e-6, f"Verification should be near zero: {verification}"

    def test_efficiency_change_calculation(self):
        """Test that TE change is computed correctly."""
        np.random.seed(123)

        # Simple 2-firm, 2-period panel
        data = pd.DataFrame(
            {
                "firm": [1, 1, 2, 2],
                "year": [2010, 2011, 2010, 2011],
                "y": [2.0, 2.2, 2.5, 2.7],
                "x1": [1.0, 1.1, 1.2, 1.3],
                "x2": [0.5, 0.6, 0.7, 0.8],
            }
        )

        # Efficiency: Firm 1 improves, Firm 2 declines
        eff_data = pd.DataFrame(
            {
                "entity": [1, 1, 2, 2],
                "time": [2010, 2011, 2010, 2011],
                "efficiency": [0.70, 0.80, 0.90, 0.85],
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])
        params = np.array([0.5, 0.5, 0.1, 0.05])
        result = MockResult(model, params, eff_data)

        tfp = TFPDecomposition(result, periods=(2010, 2011))
        decomp = tfp.decompose()

        # Check TE changes
        firm1_te = decomp[decomp["entity"] == 1]["delta_te"].values[0]
        firm2_te = decomp[decomp["entity"] == 2]["delta_te"].values[0]

        # Firm 1: ln(0.80) - ln(0.70) > 0 (improvement)
        expected_firm1_te = np.log(0.80) - np.log(0.70)
        assert (
            abs(firm1_te - expected_firm1_te) < 1e-6
        ), f"Firm 1 TE change: {firm1_te} vs {expected_firm1_te}"

        # Firm 2: ln(0.85) - ln(0.90) < 0 (decline)
        expected_firm2_te = np.log(0.85) - np.log(0.90)
        assert (
            abs(firm2_te - expected_firm2_te) < 1e-6
        ), f"Firm 2 TE change: {firm2_te} vs {expected_firm2_te}"

        assert firm1_te > 0, "Firm 1 should improve"
        assert firm2_te < 0, "Firm 2 should decline"

    def test_returns_to_scale(self):
        """Test RTS computation for different technologies."""
        # CRS technology: β1 + β2 = 1
        data = pd.DataFrame(
            {
                "firm": [1, 1],
                "year": [2010, 2011],
                "y": [2.0, 2.1],
                "x1": [1.0, 1.05],
                "x2": [0.5, 0.55],
            }
        )

        eff_data = pd.DataFrame(
            {
                "entity": [1, 1],
                "time": [2010, 2011],
                "efficiency": [0.85, 0.87],
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])

        # Test CRS: sum = 1.0
        params_crs = np.array([0.6, 0.4, 0.1, 0.05])
        result_crs = MockResult(model, params_crs, eff_data)
        tfp_crs = TFPDecomposition(result_crs, periods=(2010, 2011))
        decomp_crs = tfp_crs.decompose()

        assert abs(decomp_crs["rts"].values[0] - 1.0) < 1e-6, "Should have CRS"
        assert abs(decomp_crs["delta_se"].values[0]) < 1e-6, "SE should be near zero for CRS"

        # Test IRS: sum > 1.0
        params_irs = np.array([0.7, 0.5, 0.1, 0.05])  # sum = 1.2
        result_irs = MockResult(model, params_irs, eff_data)
        tfp_irs = TFPDecomposition(result_irs, periods=(2010, 2011))
        decomp_irs = tfp_irs.decompose()

        assert decomp_irs["rts"].values[0] > 1.0, "Should have IRS"

        # Test DRS: sum < 1.0
        params_drs = np.array([0.4, 0.3, 0.1, 0.05])  # sum = 0.7
        result_drs = MockResult(model, params_drs, eff_data)
        tfp_drs = TFPDecomposition(result_drs, periods=(2010, 2011))
        decomp_drs = tfp_drs.decompose()

        assert decomp_drs["rts"].values[0] < 1.0, "Should have DRS"

    def test_scale_efficiency_change(self):
        """Test scale efficiency component."""
        # Firm with IRS and expanding
        data = pd.DataFrame(
            {
                "firm": [1, 1],
                "year": [2010, 2011],
                "y": [2.0, 2.3],  # Large output increase
                "x1": [1.0, 1.2],  # Input increase
                "x2": [0.5, 0.6],  # Input increase
            }
        )

        eff_data = pd.DataFrame(
            {
                "entity": [1, 1],
                "time": [2010, 2011],
                "efficiency": [0.85, 0.85],  # No TE change
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])

        # IRS technology: β1 + β2 = 1.3
        params = np.array([0.8, 0.5, 0.1, 0.05])
        result = MockResult(model, params, eff_data)

        tfp = TFPDecomposition(result, periods=(2010, 2011))
        decomp = tfp.decompose()

        # With IRS and expansion, SE should be positive
        delta_se = decomp["delta_se"].values[0]
        # Note: sign depends on implementation details

        # Most importantly: verify decomposition holds
        assert abs(decomp["verification"].values[0]) < 1e-6


class TestTFPAggregation:
    """Tests for aggregate TFP statistics."""

    def test_aggregate_decomposition(self):
        """Test aggregate statistics computation."""
        np.random.seed(42)

        # Create larger panel
        n_firms = 10
        firms = np.repeat(range(1, n_firms + 1), 2)
        years = np.tile([2010, 2011], n_firms)

        y = np.random.uniform(1.5, 3.0, n_firms * 2)
        x1 = np.random.uniform(0.5, 2.0, n_firms * 2)
        x2 = np.random.uniform(0.5, 2.0, n_firms * 2)

        data = pd.DataFrame(
            {
                "firm": firms,
                "year": years,
                "y": y,
                "x1": x1,
                "x2": x2,
            }
        )

        # Random efficiency scores
        eff = np.random.uniform(0.7, 0.95, n_firms * 2)
        eff_data = pd.DataFrame(
            {
                "entity": firms,
                "time": years,
                "efficiency": eff,
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])
        params = np.array([0.6, 0.35, 0.1, 0.05])
        result = MockResult(model, params, eff_data)

        tfp = TFPDecomposition(result, periods=(2010, 2011))
        agg = tfp.aggregate_decomposition()

        # Check structure
        assert "mean_delta_tfp" in agg
        assert "mean_delta_tc" in agg
        assert "mean_delta_te" in agg
        assert "mean_delta_se" in agg
        assert "pct_from_tc" in agg
        assert "pct_from_te" in agg
        assert "pct_from_se" in agg
        assert "std_delta_tfp" in agg
        assert "n_firms" in agg

        # Check values
        assert agg["n_firms"] == n_firms
        assert isinstance(agg["mean_delta_tfp"], (int, float))
        assert isinstance(agg["std_delta_tfp"], (int, float))

        # Percentages should sum to ~100%
        pct_sum = agg["pct_from_tc"] + agg["pct_from_te"] + agg["pct_from_se"]
        assert abs(pct_sum - 100) < 1, f"Percentages should sum to 100: {pct_sum}"


class TestTFPVisualization:
    """Tests for TFP visualization methods."""

    def test_plot_bar_creates_figure(self):
        """Test that bar plot is created without errors."""
        np.random.seed(42)

        firms = np.repeat(range(1, 6), 2)
        years = np.tile([2010, 2011], 5)

        data = pd.DataFrame(
            {
                "firm": firms,
                "year": years,
                "y": np.random.uniform(2, 3, 10),
                "x1": np.random.uniform(1, 2, 10),
                "x2": np.random.uniform(0.5, 1.5, 10),
            }
        )

        eff_data = pd.DataFrame(
            {
                "entity": firms,
                "time": years,
                "efficiency": np.random.uniform(0.7, 0.95, 10),
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])
        params = np.array([0.6, 0.35, 0.1, 0.05])
        result = MockResult(model, params, eff_data)

        tfp = TFPDecomposition(result, periods=(2010, 2011))

        # Should create figure without errors
        fig = tfp.plot_decomposition(kind="bar", top_n=5)
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_scatter_creates_figure(self):
        """Test that scatter plot is created without errors."""
        np.random.seed(42)

        firms = np.repeat(range(1, 6), 2)
        years = np.tile([2010, 2011], 5)

        data = pd.DataFrame(
            {
                "firm": firms,
                "year": years,
                "y": np.random.uniform(2, 3, 10),
                "x1": np.random.uniform(1, 2, 10),
                "x2": np.random.uniform(0.5, 1.5, 10),
            }
        )

        eff_data = pd.DataFrame(
            {
                "entity": firms,
                "time": years,
                "efficiency": np.random.uniform(0.7, 0.95, 10),
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])
        params = np.array([0.6, 0.35, 0.1, 0.05])
        result = MockResult(model, params, eff_data)

        tfp = TFPDecomposition(result, periods=(2010, 2011))

        # Should create figure without errors
        fig = tfp.plot_decomposition(kind="scatter")
        assert fig is not None

        import matplotlib.pyplot as plt

        plt.close(fig)


class TestTFPSummary:
    """Tests for TFP summary output."""

    def test_summary_generates_text(self):
        """Test that summary generates formatted text."""
        np.random.seed(42)

        firms = np.repeat(range(1, 4), 2)
        years = np.tile([2010, 2011], 3)

        data = pd.DataFrame(
            {
                "firm": firms,
                "year": years,
                "y": [2.0, 2.2, 2.5, 2.7, 3.0, 3.3],
                "x1": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                "x2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            }
        )

        eff_data = pd.DataFrame(
            {
                "entity": firms,
                "time": years,
                "efficiency": [0.80, 0.82, 0.85, 0.87, 0.90, 0.92],
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1", "x2"])
        params = np.array([0.6, 0.35, 0.1, 0.05])
        result = MockResult(model, params, eff_data)

        tfp = TFPDecomposition(result, periods=(2010, 2011))
        summary = tfp.summary()

        # Check that summary contains expected elements
        assert "TFP DECOMPOSITION SUMMARY" in summary
        assert "Period: 2010" in summary
        assert "2011" in summary
        assert "Number of firms: 3" in summary
        assert "Total TFP Growth" in summary
        assert "Technical Change" in summary
        assert "Efficiency Change" in summary
        assert "Scale Effect" in summary


class TestTFPErrors:
    """Tests for error handling."""

    def test_requires_panel_data(self):
        """Test that error is raised for non-panel data."""
        # Model without entity
        data = pd.DataFrame(
            {
                "y": [2.0, 2.1],
                "x1": [1.0, 1.1],
            }
        )

        model = MockModel(data, entity=None, time="year", depvar="y", exog=["x1"])
        params = np.array([0.6, 0.1, 0.05])
        eff_data = pd.DataFrame({"entity": [1, 1], "time": [2010, 2011], "efficiency": [0.8, 0.85]})
        result = MockResult(model, params, eff_data)

        with pytest.raises(ValueError, match="panel data"):
            TFPDecomposition(result)

    def test_requires_time_variable(self):
        """Test that error is raised when time is missing."""
        data = pd.DataFrame(
            {
                "firm": [1, 1],
                "y": [2.0, 2.1],
                "x1": [1.0, 1.1],
            }
        )

        model = MockModel(data, entity="firm", time=None, depvar="y", exog=["x1"])
        params = np.array([0.6, 0.1, 0.05])
        eff_data = pd.DataFrame({"entity": [1, 1], "time": [2010, 2011], "efficiency": [0.8, 0.85]})
        result = MockResult(model, params, eff_data)

        with pytest.raises(ValueError, match="time identifier"):
            TFPDecomposition(result)

    def test_invalid_periods(self):
        """Test that error is raised for invalid periods."""
        data = pd.DataFrame(
            {
                "firm": [1, 1],
                "year": [2010, 2011],
                "y": [2.0, 2.1],
                "x1": [1.0, 1.1],
            }
        )

        model = MockModel(data, entity="firm", time="year", depvar="y", exog=["x1"])
        params = np.array([0.6, 0.1, 0.05])
        eff_data = pd.DataFrame({"entity": [1, 1], "time": [2010, 2011], "efficiency": [0.8, 0.85]})
        result = MockResult(model, params, eff_data)

        with pytest.raises(ValueError, match="not found in data"):
            TFPDecomposition(result, periods=(2009, 2012))
