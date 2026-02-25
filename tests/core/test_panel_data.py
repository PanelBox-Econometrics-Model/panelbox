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

    def test_index_is_reset(self, balanced_panel_data):
        """Test that index is reset (0-based sequential) after sorting."""
        shuffled = balanced_panel_data.sample(frac=1.0, random_state=42)
        panel = PanelData(shuffled, "entity", "time")
        # Index should be 0, 1, 2, ..., n-1
        assert list(panel.data.index) == list(range(len(panel.data)))

    def test_dataframe_property(self, balanced_panel_data):
        """Test the dataframe property alias."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        pd.testing.assert_frame_equal(panel.dataframe, panel.data)


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
        """Test demeaning with variables=None (all numeric except identifiers)."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned = panel.demeaning(method="entity")

        # Should demean y, x1, x2 but not entity, time
        assert "y" in demeaned.columns
        assert "x1" in demeaned.columns
        assert "x2" in demeaned.columns

        # Entity means of demeaned variables should be zero
        entity_means = demeaned.groupby("entity")[["y", "x1", "x2"]].mean()
        np.testing.assert_array_almost_equal(entity_means.values, 0, decimal=10)

        # entity and time columns should NOT be demeaned (values unchanged)
        pd.testing.assert_series_equal(demeaned["entity"], panel.data["entity"], check_names=True)
        pd.testing.assert_series_equal(demeaned["time"], panel.data["time"], check_names=True)

    def test_demeaning_single_string_variable(self, balanced_panel_data):
        """Test demeaning with a single string variable (not list)."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned = panel.demeaning("y", method="entity")

        # Should demean y specifically
        entity_means = demeaned.groupby("entity")["y"].mean()
        np.testing.assert_array_almost_equal(entity_means.values, 0, decimal=10)

    def test_demeaning_default_method(self, balanced_panel_data):
        """Test that default method is 'entity'."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        demeaned_default = panel.demeaning(["y"])
        demeaned_entity = panel.demeaning(["y"], method="entity")
        pd.testing.assert_frame_equal(demeaned_default, demeaned_entity)

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
        """Test first difference with variables=None selects all numeric cols."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        diff_data = panel.first_difference()

        assert "y" in diff_data.columns
        assert "x1" in diff_data.columns
        assert "x2" in diff_data.columns

        # Should have n_obs - n_entities rows (first period dropped)
        expected_obs = panel.n_obs - panel.n_entities
        assert len(diff_data) == expected_obs

        # Verify actual difference values for all numeric columns
        entity_1 = panel.data[panel.data["entity"] == 1].copy()
        entity_1_diff = diff_data[diff_data["entity"] == 1].copy()
        for col in ["y", "x1", "x2"]:
            manual_diff = entity_1.iloc[1][col] - entity_1.iloc[0][col]
            np.testing.assert_almost_equal(manual_diff, entity_1_diff.iloc[0][col])

    def test_first_difference_single_string(self, balanced_panel_data):
        """Test first difference with single string variable."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        diff_data = panel.first_difference("y")

        # Should still have correct number of rows
        expected_obs = panel.n_obs - panel.n_entities
        assert len(diff_data) == expected_obs

        # y should be differenced
        entity_1 = panel.data[panel.data["entity"] == 1].copy()
        entity_1_diff = diff_data[diff_data["entity"] == 1].copy()
        manual_diff = entity_1.iloc[1]["y"] - entity_1.iloc[0]["y"]
        np.testing.assert_almost_equal(manual_diff, entity_1_diff.iloc[0]["y"])

    def test_first_difference_no_nan_rows(self, balanced_panel_data):
        """Test that first difference output has no NaN in differenced variables."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        diff_data = panel.first_difference(["y", "x1"])

        # No NaN values in the differenced columns
        assert not diff_data["y"].isna().any()
        assert not diff_data["x1"].isna().any()


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

    def test_lag_column_naming(self, balanced_panel_data):
        """Test that lag columns have correct naming convention."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        lagged = panel.lag("y", lags=[1, 2])
        assert "L1.y" in lagged.columns
        assert "L2.y" in lagged.columns
        # L2.y for entity 1 at period 2 should be y at period 0
        entity_1 = lagged[lagged["entity"] == 1].copy().reset_index(drop=True)
        np.testing.assert_almost_equal(entity_1.loc[2, "L2.y"], entity_1.loc[0, "y"])

    def test_lead_column_naming(self, balanced_panel_data):
        """Test that lead columns have correct naming convention."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        led = panel.lead("y", leads=[1, 2])
        assert "F1.y" in led.columns
        assert "F2.y" in led.columns
        # Last lead should be NaN
        entity_1 = led[led["entity"] == 1].copy().reset_index(drop=True)
        assert pd.isna(entity_1.iloc[-1]["F1.y"])

    def test_lag_invalid_variable(self, balanced_panel_data):
        """Test lag with invalid variable name."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
            panel.lag("nonexistent")

    def test_lead_invalid_variable(self, balanced_panel_data):
        """Test lead with invalid variable name."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
            panel.lead("nonexistent")

    def test_lead_invalid_order(self, balanced_panel_data):
        """Test that lead order < 1 raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        with pytest.raises(ValueError, match="Lead order must be >= 1"):
            panel.lead("y", leads=0)

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

    def test_summary_content_balanced(self, balanced_panel_data):
        """Test summary content has exact expected values for balanced panel."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        summary = panel.summary()

        assert "Total observations:     50" in summary
        assert "Number of time periods: 5" in summary
        assert "Periods per entity:     5" in summary
        assert "Entity identifier: entity" in summary
        assert "Time identifier:   time" in summary
        # Time range
        assert "2020 to 2024" in summary

    def test_summary_content_unbalanced(self, unbalanced_panel_data):
        """Test summary content for unbalanced panel with exact values."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        summary = panel.summary()

        assert "Number of entities:     3" in summary
        assert "Total observations:     12" in summary
        assert "Min periods per entity: 3" in summary
        assert "Max periods per entity: 5" in summary
        assert "Avg periods per entity: 4.0" in summary

    def test_repr(self, balanced_panel_data):
        """Test __repr__ method."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        repr_str = repr(panel)

        assert "PanelData" in repr_str
        assert "Balanced" in repr_str
        assert "n_entities=10" in repr_str
        assert "n_periods=5" in repr_str
        assert "n_obs=50" in repr_str

    def test_repr_unbalanced(self, unbalanced_panel_data):
        """Test __repr__ for unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        repr_str = repr(panel)
        assert "Unbalanced" in repr_str
        assert "n_entities=3" in repr_str


class TestFirstDifferenceValidation:
    """Additional tests for first_difference edge cases."""

    def test_first_difference_invalid_variable(self, balanced_panel_data):
        """Test that first_difference with invalid variable raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="Variable 'nonexistent' not found"):
            panel.first_difference(["nonexistent"])

    def test_first_difference_invalid_variable_single_string(self, balanced_panel_data):
        """Test that first_difference with single invalid string variable raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")

        with pytest.raises(ValueError, match="Variable 'bad_col' not found"):
            panel.first_difference("bad_col")


class TestUnbalancedPanelOperations:
    """Tests for operations on unbalanced panel data."""

    def test_lag_unbalanced(self, unbalanced_panel_data):
        """Test lag operation on unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        lagged = panel.lag("y", lags=1)

        assert "L1.y" in lagged.columns
        # First observation per entity should be NaN
        for entity in panel.entities:
            entity_data = lagged[lagged["entity"] == entity].reset_index(drop=True)
            assert pd.isna(entity_data.loc[0, "L1.y"])

    def test_lead_unbalanced(self, unbalanced_panel_data):
        """Test lead operation on unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        led = panel.lead("y", leads=1)

        assert "F1.y" in led.columns
        # Last observation per entity should be NaN
        for entity in panel.entities:
            entity_data = led[led["entity"] == entity].reset_index(drop=True)
            assert pd.isna(entity_data.iloc[-1]["F1.y"])

    def test_first_difference_unbalanced(self, unbalanced_panel_data):
        """Test first difference on unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        diff = panel.first_difference(["y"])

        # Entity 1 has 5 periods -> 4 diffs, Entity 2 has 4 -> 3, Entity 3 has 3 -> 2
        expected_obs = (5 - 1) + (4 - 1) + (3 - 1)
        assert len(diff) == expected_obs
        assert not diff["y"].isna().any()

    def test_demeaning_unbalanced(self, unbalanced_panel_data):
        """Test entity demeaning on unbalanced panel."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")
        demeaned = panel.demeaning(["y"], method="entity")

        entity_means = demeaned.groupby("entity")["y"].mean()
        np.testing.assert_array_almost_equal(entity_means.values, 0, decimal=10)

    def test_balance_removes_incomplete_entities(self, unbalanced_panel_data):
        """Test that balance() only keeps entities with full time series."""
        panel = PanelData(unbalanced_panel_data, "entity", "time")

        assert not panel.is_balanced
        balanced = panel.balance()
        assert balanced.is_balanced
        # Only entity 1 has all 5 periods
        assert balanced.n_entities == 1
        assert set(balanced.entities) == {1}

    def test_balance_already_balanced_returns_self(self, balanced_panel_data):
        """Test that balance() returns self when already balanced."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        balanced = panel.balance()
        assert balanced is panel  # Same object, not a copy


class TestEdgeCasesData:
    """Edge cases for PanelData."""

    def test_single_entity(self):
        """Test PanelData with a single entity."""
        data = pd.DataFrame(
            {
                "entity": [1, 1, 1],
                "time": [2020, 2021, 2022],
                "y": [10.0, 20.0, 30.0],
            }
        )
        panel = PanelData(data, "entity", "time")
        assert panel.n_entities == 1
        assert panel.n_periods == 3
        assert panel.is_balanced is True

    def test_single_period(self):
        """Test PanelData with a single period per entity."""
        data = pd.DataFrame(
            {
                "entity": [1, 2, 3],
                "time": [2020, 2020, 2020],
                "y": [10.0, 20.0, 30.0],
            }
        )
        panel = PanelData(data, "entity", "time")
        assert panel.n_entities == 3
        assert panel.n_periods == 1
        assert panel.is_balanced is True

    def test_non_numeric_entity_col(self):
        """Test PanelData with string entity identifiers."""
        data = pd.DataFrame(
            {
                "firm": ["A", "A", "B", "B"],
                "year": [2020, 2021, 2020, 2021],
                "y": [1.0, 2.0, 3.0, 4.0],
            }
        )
        panel = PanelData(data, "firm", "year")
        assert panel.n_entities == 2
        assert panel.is_balanced is True
        assert set(panel.entities) == {"A", "B"}

    def test_lag_multiple_list(self, balanced_panel_data):
        """Test creating multiple lags at once with a list."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        lagged = panel.lag("y", lags=[1, 2, 3])

        assert "L1.y" in lagged.columns
        assert "L2.y" in lagged.columns
        assert "L3.y" in lagged.columns

        entity_1 = lagged[lagged["entity"] == 1].reset_index(drop=True)
        # L3.y at index 3 should be y at index 0
        np.testing.assert_almost_equal(entity_1.loc[3, "L3.y"], entity_1.loc[0, "y"])

    def test_lead_multiple_list(self, balanced_panel_data):
        """Test creating multiple leads at once with a list."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        led = panel.lead("y", leads=[1, 2])

        assert "F1.y" in led.columns
        assert "F2.y" in led.columns

        entity_1 = led[led["entity"] == 1].reset_index(drop=True)
        # F2.y at index 0 should be y at index 2
        np.testing.assert_almost_equal(entity_1.loc[0, "F2.y"], entity_1.loc[2, "y"])

    def test_lead_negative_order_raises(self, balanced_panel_data):
        """Test that lead with negative order raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        with pytest.raises(ValueError, match="Lead order must be >= 1"):
            panel.lead("y", leads=-1)

    def test_lag_negative_order_raises(self, balanced_panel_data):
        """Test that lag with negative order raises ValueError."""
        panel = PanelData(balanced_panel_data, "entity", "time")
        with pytest.raises(ValueError, match="Lag order must be >= 1"):
            panel.lag("y", lags=-1)

    def test_input_data_not_mutated(self, balanced_panel_data):
        """Test that original DataFrame is not mutated by PanelData operations."""
        original = balanced_panel_data.copy()
        panel = PanelData(balanced_panel_data, "entity", "time")
        panel.demeaning(["y"])
        panel.first_difference(["y"])
        panel.lag("y")

        pd.testing.assert_frame_equal(balanced_panel_data, original)
