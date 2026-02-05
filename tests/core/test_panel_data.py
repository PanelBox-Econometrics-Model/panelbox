"""
Tests for PanelData class.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.core.panel_data import PanelData


class TestPanelDataInitialization:
    """Tests for PanelData initialization and validation."""

    def test_init_balanced_panel(self, balanced_panel_data):
        """Test initialization with balanced panel data."""
        panel = PanelData(balanced_panel_data, "entity", "time")

        assert panel.n_entities == 10
        assert panel.n_periods == 5
        assert panel.n_obs == 50
        assert panel.is_balanced is True
        assert panel.entity_col == "entity"
        assert panel.time_col == "time"

    def test_init_unbalanced_panel(self, unbalanced_panel_data):
        """Test initialization with unbalanced panel data."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")

        assert panel.n_entities == 3
        assert panel.n_periods == 5  # max periods
        assert panel.n_obs == 12
        assert panel.is_balanced is False
        assert panel.min_periods == 3
        assert panel.avg_periods == 4.0

    def test_invalid_dataframe_type(self):
        """Test that non-DataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
            PanelData([1, 2, 3], "entity", "time")

    def test_invalid_entity_column(self, balanced_panel_data):
        """Test that invalid entity column raises ValueError."""
        with pytest.raises(ValueError, match="entity_col 'invalid' not found"):
            PanelData(balanced_panel_data, "invalid", "time")

    def test_invalid_time_column(self, balanced_panel_data):
        """Test that invalid time column raises ValueError."""
        with pytest.raises(ValueError, match="time_col 'invalid' not found"):
            PanelData(balanced_panel_data, "entity", "invalid")

    def test_data_is_sorted(self, balanced_panel_data):
        """Test that data is sorted by entity and time."""
        # Shuffle data
        shuffled = balanced_panel_data.sample(frac=1.0, random_state=42)

        panel = PanelData(shuffled, "entity", "time")

        # Check that data is sorted
        assert (panel.data["entity"].diff().fillna(0) >= 0).all()
        # Within each entity, time should be sorted
        for entity in panel.entities:
            entity_data = panel.data[panel.data["entity"] == entity]
            assert (entity_data["time"].diff().fillna(0) >= 0).all()


class TestDemeaning:
    """Tests for demeaning transformations."""

    def test_entity_demeaning(self, balanced_panel_data):
        """Test entity demeaning (within transformation)."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned = panel.demeaning(["y", "x1"], method="entity")

        # Check that entity means are zero (within rounding error)
        entity_means = demeaned.groupby("entity")[["y", "x1"]].mean()
        np.testing.assert_array_almost_equal(entity_means.values, 0, decimal=10)

    def test_time_demeaning(self, balanced_panel_data):
        """Test time demeaning."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned = panel.demeaning(["y", "x1"], method="time")

        # Check that time means are zero
        time_means = demeaned.groupby("time")[["y", "x1"]].mean()
        np.testing.assert_array_almost_equal(time_means.values, 0, decimal=10)

    def test_both_demeaning(self, balanced_panel_data):
        """Test two-way demeaning."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned = panel.demeaning(["y"], method="both")

        # After two-way demeaning, both entity and time means should be zero
        entity_means = demeaned.groupby("entity")["y"].mean()
        time_means = demeaned.groupby("time")["y"].mean()

        np.testing.assert_array_almost_equal(entity_means.values, 0, decimal=10)
        np.testing.assert_array_almost_equal(time_means.values, 0, decimal=10)

    def test_demeaning_all_numeric(self, balanced_panel_data):
        """Test demeaning with variables=None (all numeric)."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned = panel.demeaning(method="entity")

        # Should demean y, x1, x2 but not entity, time
        assert "y" in demeaned.columns
        assert "x1" in demeaned.columns
        assert "x2" in demeaned.columns

    def test_invalid_method(self, balanced_panel_data):
        """Test that invalid method raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="method must be"):
            panel.demeaning(["y"], method="invalid")

    def test_invalid_variable(self, balanced_panel_data):
        """Test that invalid variable raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="Variable 'invalid' not found"):
            panel.demeaning(["invalid"], method="entity")


class TestFirstDifference:
    """Tests for first differencing."""

    def test_first_difference(self, balanced_panel_data):
        """Test first difference transformation."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        diff_data = panel.first_difference(["y", "x1"])

        # Should have fewer observations (first period dropped for each entity)
        expected_obs = panel.n_obs - panel.n_entities
        assert len(diff_data) == expected_obs

        # Check that differences are computed correctly for one entity
        entity_1 = panel.data[panel.data["entity"] == 1].copy()
        entity_1_diff = diff_data[diff_data["entity"] == 1].copy()

        # Manual difference for second period
        manual_diff_y = entity_1.iloc[1]["y"] - entity_1.iloc[0]["y"]
        computed_diff_y = entity_1_diff.iloc[0]["y"]

        np.testing.assert_almost_equal(manual_diff_y, computed_diff_y)

    def test_first_difference_all_numeric(self, balanced_panel_data):
        """Test first difference with variables=None."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        diff_data = panel.first_difference()

        assert "y" in diff_data.columns
        assert "x1" in diff_data.columns
        assert "x2" in diff_data.columns


class TestLagLead:
    """Tests for lag and lead operations."""

    def test_single_lag(self, balanced_panel_data):
        """Test creating a single lag."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        lagged = panel.lag("y", lags=1)

        assert "L1.y" in lagged.columns

        # Check lag is correct for entity 1
        entity_1 = lagged[lagged["entity"] == 1].copy().reset_index(drop=True)

        # L1.y at period 1 should be y at period 0
        assert pd.isna(entity_1.loc[0, "L1.y"])  # First period is NaN
        np.testing.assert_almost_equal(entity_1.loc[1, "L1.y"], entity_1.loc[0, "y"])

    def test_multiple_lags(self, balanced_panel_data):
        """Test creating multiple lags."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        lagged = panel.lag("y", lags=[1, 2, 3])

        assert "L1.y" in lagged.columns
        assert "L2.y" in lagged.columns
        assert "L3.y" in lagged.columns

    def test_single_lead(self, balanced_panel_data):
        """Test creating a single lead."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        led = panel.lead("y", leads=1)

        assert "F1.y" in led.columns

        # Check lead is correct
        entity_1 = led[led["entity"] == 1].copy().reset_index(drop=True)

        # F1.y at period 0 should be y at period 1
        np.testing.assert_almost_equal(entity_1.loc[0, "F1.y"], entity_1.loc[1, "y"])

    def test_invalid_lag_order(self, balanced_panel_data):
        """Test that lag order < 1 raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="Lag order must be >= 1"):
            panel.lag("y", lags=0)


class TestBalance:
    """Tests for panel balancing."""

    def test_balance_already_balanced(self, balanced_panel_data):
        """Test that balancing a balanced panel returns same panel."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        balanced = panel.balance()

        assert balanced.n_entities == panel.n_entities
        assert balanced.n_obs == panel.n_obs
        assert balanced.is_balanced is True

    def test_balance_unbalanced(self, unbalanced_panel_data):
        """Test balancing an unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        balanced = panel.balance()

        # Only entity 1 has 5 periods (max)
        assert balanced.n_entities == 1
        assert balanced.n_obs == 5
        assert balanced.is_balanced is True


class TestSummary:
    """Tests for summary method."""

    def test_summary_balanced(self, balanced_panel_data):
        """Test summary for balanced panel."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        summary = panel.summary()

        assert "PANEL DATA SUMMARY" in summary
        assert "Balanced:               Yes" in summary
        assert "Number of entities:     10" in summary

    def test_summary_unbalanced(self, unbalanced_panel_data):
        """Test summary for unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        summary = panel.summary()

        assert "PANEL DATA SUMMARY" in summary
        assert "Balanced:               No" in summary
        assert "Min periods per entity" in summary

    def test_repr(self, balanced_panel_data):
        """Test __repr__ method."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        repr_str = repr(panel)

        assert "PanelData" in repr_str
        assert "Balanced" in repr_str
        assert "n_entities=10" in repr_str
