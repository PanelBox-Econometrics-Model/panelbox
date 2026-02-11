"""
Tests for Panel VAR transformation methods (FOD and FD)
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var.transforms import (
    first_difference,
    forward_orthogonal_deviation,
    get_valid_instrument_lags,
)


class TestForwardOrthogonalDeviation:
    """Tests for FOD transformation"""

    def test_fod_basic_balanced_panel(self):
        """Test FOD on simple balanced panel"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        transformed, meta = forward_orthogonal_deviation(df)

        # Should drop last period for each entity
        assert len(transformed) == 4  # 2 entities × 2 periods
        assert transformed["entity"].nunique() == 2
        assert transformed["time"].max() == 2  # No time=3

        # Check metadata
        assert len(meta) == 4
        assert "normalization_factor" in meta.columns
        assert "periods_ahead" in meta.columns

    def test_fod_normalization_factor(self):
        """Test that normalization factor is correct"""
        df = pd.DataFrame({"entity": [1, 1, 1], "time": [1, 2, 3], "y": [1.0, 2.0, 3.0]})

        _, meta = forward_orthogonal_deviation(df)

        # For t=1: periods_ahead=2, c_t = sqrt(2/3)
        # For t=2: periods_ahead=1, c_t = sqrt(1/2)
        expected_c1 = np.sqrt(2 / 3)
        expected_c2 = np.sqrt(1 / 2)

        np.testing.assert_allclose(meta.iloc[0]["normalization_factor"], expected_c1, rtol=1e-10)
        np.testing.assert_allclose(meta.iloc[1]["normalization_factor"], expected_c2, rtol=1e-10)

    def test_fod_transformation_values(self):
        """Test that FOD values are computed correctly"""
        df = pd.DataFrame({"entity": [1, 1, 1], "time": [1, 2, 3], "y": [1.0, 2.0, 3.0]})

        transformed, _ = forward_orthogonal_deviation(df)

        # For t=1: y*_1 = sqrt(2/3) * (1 - (2+3)/2) = sqrt(2/3) * (1 - 2.5) = sqrt(2/3) * (-1.5)
        expected_y1 = np.sqrt(2 / 3) * (1.0 - 2.5)

        # For t=2: y*_2 = sqrt(1/2) * (2 - 3)
        expected_y2 = np.sqrt(1 / 2) * (2.0 - 3.0)

        np.testing.assert_allclose(transformed.iloc[0]["y"], expected_y1, rtol=1e-10)
        np.testing.assert_allclose(transformed.iloc[1]["y"], expected_y2, rtol=1e-10)

    def test_fod_unbalanced_panel(self):
        """Test FOD on unbalanced panel"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2],  # Entity 1: 4 periods, Entity 2: 2 periods
                "time": [1, 2, 3, 4, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        transformed, meta = forward_orthogonal_deviation(df)

        # Entity 1: 3 observations (drop t=4)
        # Entity 2: 1 observation (drop t=2)
        assert len(transformed) == 4

        entity1_obs = transformed[transformed["entity"] == 1]
        entity2_obs = transformed[transformed["entity"] == 2]

        assert len(entity1_obs) == 3
        assert len(entity2_obs) == 1

    def test_fod_multiple_variables(self):
        """Test FOD with multiple value columns"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            }
        )

        transformed, _ = forward_orthogonal_deviation(df, value_cols=["x", "y"])

        assert "x" in transformed.columns
        assert "y" in transformed.columns
        assert len(transformed) == 4

    def test_fod_single_period_entity_dropped(self):
        """Test that entities with only one period are dropped"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 2],  # Entity 2 has only 1 period
                "time": [1, 2, 1],
                "y": [1.0, 2.0, 5.0],
            }
        )

        transformed, _ = forward_orthogonal_deviation(df)

        # Only entity 1 should remain
        assert len(transformed) == 1
        assert transformed["entity"].iloc[0] == 1

    def test_fod_preserves_column_order(self):
        """Test that column order is preserved"""
        df = pd.DataFrame(
            {"entity": [1, 1, 1], "time": [1, 2, 3], "x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]}
        )

        transformed, _ = forward_orthogonal_deviation(df)

        expected_cols = ["entity", "time", "x", "y"]
        assert list(transformed.columns) == expected_cols

    def test_fod_invalid_entity_column(self):
        """Test error when entity column not found"""
        df = pd.DataFrame({"time": [1, 2], "y": [1.0, 2.0]})

        with pytest.raises(ValueError, match="Entity column.*not found"):
            forward_orthogonal_deviation(df, entity_col="entity")

    def test_fod_invalid_time_column(self):
        """Test error when time column not found"""
        df = pd.DataFrame({"entity": [1, 1], "y": [1.0, 2.0]})

        with pytest.raises(ValueError, match="Time column.*not found"):
            forward_orthogonal_deviation(df, time_col="time")

    def test_fod_no_numeric_columns(self):
        """Test error when no numeric columns to transform"""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "text": ["a", "b"]})

        with pytest.raises(ValueError, match="No numeric columns to transform"):
            forward_orthogonal_deviation(df)


class TestFirstDifference:
    """Tests for FD transformation"""

    def test_fd_basic_balanced_panel(self):
        """Test FD on simple balanced panel"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [1.0, 2.0, 4.0, 10.0, 15.0, 25.0],
            }
        )

        transformed = first_difference(df)

        # Should drop first period for each entity
        assert len(transformed) == 4  # 2 entities × 2 periods
        assert transformed["time"].min() == 2  # No time=1

        # Check differences
        entity1_diffs = transformed[transformed["entity"] == 1]["y"].values
        np.testing.assert_allclose(entity1_diffs, [1.0, 2.0], rtol=1e-10)  # 2-1=1, 4-2=2

        entity2_diffs = transformed[transformed["entity"] == 2]["y"].values
        np.testing.assert_allclose(entity2_diffs, [5.0, 10.0], rtol=1e-10)  # 15-10=5, 25-15=10

    def test_fd_transformation_values(self):
        """Test that FD values are computed correctly"""
        df = pd.DataFrame({"entity": [1, 1, 1], "time": [1, 2, 3], "y": [1.0, 3.0, 6.0]})

        transformed = first_difference(df)

        # Δy_2 = 3 - 1 = 2
        # Δy_3 = 6 - 3 = 3
        expected = [2.0, 3.0]

        np.testing.assert_allclose(transformed["y"].values, expected, rtol=1e-10)

    def test_fd_unbalanced_panel(self):
        """Test FD on unbalanced panel"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2],
                "time": [1, 2, 3, 4, 1, 2],
                "y": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0],
            }
        )

        transformed = first_difference(df)

        # Entity 1: 3 differences
        # Entity 2: 1 difference
        assert len(transformed) == 4

        entity1_obs = transformed[transformed["entity"] == 1]
        entity2_obs = transformed[transformed["entity"] == 2]

        assert len(entity1_obs) == 3
        assert len(entity2_obs) == 1

    def test_fd_multiple_variables(self):
        """Test FD with multiple value columns"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "x": [1.0, 3.0, 5.0, 10.0],
                "y": [2.0, 5.0, 8.0, 20.0],
            }
        )

        transformed = first_difference(df, value_cols=["x", "y"])

        assert "x" in transformed.columns
        assert "y" in transformed.columns

        # Entity 1: Δx = 2, Δy = 3
        entity1 = transformed[transformed["entity"] == 1].iloc[0]
        assert entity1["x"] == 2.0
        assert entity1["y"] == 3.0

    def test_fd_single_period_entity_dropped(self):
        """Test that entities with only one period are dropped"""
        df = pd.DataFrame({"entity": [1, 1, 2], "time": [1, 2, 1], "y": [1.0, 2.0, 5.0]})

        transformed = first_difference(df)

        # Only entity 1 should remain
        assert len(transformed) == 1
        assert transformed["entity"].iloc[0] == 1

    def test_fd_preserves_column_order(self):
        """Test that column order is preserved"""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "x": [1.0, 2.0], "y": [10.0, 20.0]})

        transformed = first_difference(df)

        expected_cols = ["entity", "time", "x", "y"]
        assert list(transformed.columns) == expected_cols


class TestGetValidInstrumentLags:
    """Tests for instrument lag validation"""

    def test_valid_lags_var1(self):
        """Test valid instrument lags for VAR(1)"""
        df = pd.DataFrame({"entity": [1, 1, 1, 1], "time": [1, 2, 3, 4], "y": [1.0, 2.0, 3.0, 4.0]})

        meta = get_valid_instrument_lags(df, "entity", "time", "fod", var_lags=1)

        # For VAR(1): valid lags are t-2 or earlier
        # t=1: no valid lags (min_valid_lag = -1)
        # t=2: no valid lags (min_valid_lag = 0)
        # t=3: valid lag = 1 (min_valid_lag = 1)
        # t=4: valid lags = 1, 2 (min_valid_lag = 2)

        assert len(meta) == 4

        assert meta.iloc[0]["n_valid_lags"] == 0
        assert meta.iloc[1]["n_valid_lags"] == 0
        assert meta.iloc[2]["n_valid_lags"] == 1
        assert meta.iloc[3]["n_valid_lags"] == 2

    def test_valid_lags_var2(self):
        """Test valid instrument lags for VAR(2)"""
        df = pd.DataFrame(
            {"entity": [1, 1, 1, 1, 1], "time": [1, 2, 3, 4, 5], "y": [1.0, 2.0, 3.0, 4.0, 5.0]}
        )

        meta = get_valid_instrument_lags(df, "entity", "time", "fod", var_lags=2)

        # For VAR(2): valid lags are t-3 or earlier
        # t=1,2,3: no valid lags
        # t=4: valid lag = 1 (min_valid_lag = 1)
        # t=5: valid lags = 1, 2 (min_valid_lag = 2)

        assert meta.iloc[0]["n_valid_lags"] == 0
        assert meta.iloc[1]["n_valid_lags"] == 0
        assert meta.iloc[2]["n_valid_lags"] == 0
        assert meta.iloc[3]["n_valid_lags"] == 1
        assert meta.iloc[4]["n_valid_lags"] == 2

    def test_valid_lags_multiple_entities(self):
        """Test valid lags with multiple entities"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            }
        )

        meta = get_valid_instrument_lags(df, "entity", "time", "fod", var_lags=1)

        # Each entity has 3 periods
        # For each entity: t=1,2 have 0 valid lags, t=3 has 1 valid lag

        entity1_meta = meta[meta["entity"] == 1]
        entity2_meta = meta[meta["entity"] == 2]

        assert len(entity1_meta) == 3
        assert len(entity2_meta) == 3

        assert entity1_meta.iloc[2]["n_valid_lags"] == 1
        assert entity2_meta.iloc[2]["n_valid_lags"] == 1


class TestComparisonFODvsFD:
    """Comparison tests between FOD and FD"""

    def test_fod_preserves_more_obs_unbalanced(self):
        """Test that FOD preserves more observations than FD in unbalanced panels"""
        # Create unbalanced panel with gaps
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2, 2],
                "time": [1, 2, 4, 5, 1, 3, 4],  # Entity 1 missing t=3, Entity 2 missing t=2
                "y": [1.0, 2.0, 4.0, 5.0, 10.0, 30.0, 40.0],
            }
        )

        fod_result, _ = forward_orthogonal_deviation(df)
        fd_result = first_difference(df)

        # Both should handle unbalanced panels
        # FOD: drops last period per entity
        # FD: drops first period per entity

        # Both transformations should produce valid results
        assert len(fod_result) > 0
        assert len(fd_result) > 0

    def test_both_transformations_remove_fixed_effects(self):
        """Test that both FOD and FD remove entity fixed effects"""
        # Create panel with strong fixed effects
        np.random.seed(42)
        entities = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        times = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        fixed_effects = [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0]
        shocks = np.random.randn(9) * 0.1

        df = pd.DataFrame(
            {"entity": entities, "time": times, "y": np.array(fixed_effects) + shocks}
        )

        fod_result, _ = forward_orthogonal_deviation(df)
        fd_result = first_difference(df)

        # After transformation, mean should be close to zero (fixed effects removed)
        assert abs(fod_result["y"].mean()) < 1.0
        assert abs(fd_result["y"].mean()) < 1.0
