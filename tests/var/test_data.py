"""
Tests for PanelVARData class.

This module tests the critical data preparation functionality for Panel VAR,
including lag construction, gap detection, and cross-contamination prevention.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.var import PanelVARData


class TestPanelVARDataBasic:
    """Basic tests for PanelVARData initialization and properties."""

    def test_balanced_panel_creation(self):
        """Test creation with balanced panel (5×20×3)."""
        # Create balanced panel: 5 entities × 20 periods × 3 variables
        np.random.seed(42)
        n_entities = 5
        n_periods = 20
        n_vars = 3

        data = []
        for i in range(n_entities):
            for t in range(n_periods):
                row = {
                    "entity": f"E{i}",
                    "time": t,
                    "y1": np.random.randn(),
                    "y2": np.random.randn(),
                    "y3": np.random.randn(),
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Create PanelVARData with 1 lag
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2", "y3"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="constant",
        )

        # Check properties
        assert pvar_data.K == 3
        assert pvar_data.p == 1
        assert pvar_data.N == 5
        assert pvar_data.is_balanced is True
        # After 1 lag, we lose 1 observation per entity
        assert pvar_data.n_obs == 5 * (20 - 1)
        assert pvar_data.T_min == 19
        assert pvar_data.T_max == 19

    def test_unbalanced_panel_creation(self):
        """Test creation with unbalanced panel (T varies from 10 to 20)."""
        np.random.seed(42)
        n_entities = 5

        data = []
        for i in range(n_entities):
            # Each entity has different number of periods
            n_periods = 10 + i * 2  # 10, 12, 14, 16, 18
            for t in range(n_periods):
                row = {
                    "entity": f"E{i}",
                    "time": t,
                    "y1": np.random.randn(),
                    "y2": np.random.randn(),
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Create PanelVARData
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=2,
            trend="constant",
        )

        # Check properties
        assert pvar_data.K == 2
        assert pvar_data.p == 2
        assert pvar_data.N == 5
        assert pvar_data.is_balanced is False
        # T ranges from 10-2 to 18-2
        assert pvar_data.T_min == 8
        assert pvar_data.T_max == 16
        assert 11 < pvar_data.T_avg < 13

    def test_multiple_lags(self):
        """Test with multiple lags (p=1,2,3)."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(15):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        # Test p=1
        pvar_data_1 = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1
        )
        assert pvar_data_1.p == 1
        assert pvar_data_1.n_obs == 3 * (15 - 1)

        # Test p=2
        pvar_data_2 = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )
        assert pvar_data_2.p == 2
        assert pvar_data_2.n_obs == 3 * (15 - 2)

        # Test p=3
        pvar_data_3 = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=3
        )
        assert pvar_data_3.p == 3
        assert pvar_data_3.n_obs == 3 * (15 - 3)


class TestPanelVARDataLags:
    """Tests for lag construction and cross-contamination prevention."""

    def test_lags_do_not_cross_entities(self):
        """CRITICAL TEST: Verify lags don't cross entity boundaries."""
        # Create simple data where we can manually verify
        data = []
        for entity in ["A", "B"]:
            for t in range(5):
                row = {
                    "entity": entity,
                    "time": t,
                    "y": float(f"{ord(entity)}.{t}"),  # A.0, A.1, ..., B.0, B.1, ...
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Create with 1 lag
        pvar_data = PanelVARData(df, endog_vars=["y"], entity_col="entity", time_col="time", lags=1)

        df_with_lags = pvar_data.to_stacked()

        # For entity A, L1.y at time t should equal y at time t-1
        entity_a = df_with_lags[df_with_lags["entity"] == "A"]
        for idx, row in entity_a.iterrows():
            t = row["time"]
            if t >= 1:  # Skip first observation (no lag available)
                lag_value = row["L1.y"]
                expected_value = float(f"{ord('A')}.{t-1}")
                assert np.isclose(lag_value, expected_value), f"Entity A, time {t}: lag mismatch"

        # For entity B, L1.y at time t should equal y at time t-1 (within B, not A)
        entity_b = df_with_lags[df_with_lags["entity"] == "B"]
        for idx, row in entity_b.iterrows():
            t = row["time"]
            if t >= 1:
                lag_value = row["L1.y"]
                expected_value = float(f"{ord('B')}.{t-1}")
                assert np.isclose(lag_value, expected_value), f"Entity B, time {t}: lag mismatch"

    def test_lag_columns_created(self):
        """Test that lag columns are created correctly."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        # Create with 2 lags
        pvar_data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=2
        )

        df_with_lags = pvar_data.to_stacked()

        # Check that lag columns exist
        expected_lag_cols = ["L1.y1", "L1.y2", "L2.y1", "L2.y2"]
        for col in expected_lag_cols:
            assert col in df_with_lags.columns, f"Missing lag column: {col}"

    def test_equation_data_dimensions(self):
        """Test dimensions of equation_data() for different lags."""
        np.random.seed(42)
        data = []
        n_entities = 5
        n_periods = 20
        for i in range(n_entities):
            for t in range(n_periods):
                row = {
                    "entity": i,
                    "time": t,
                    "y1": np.random.randn(),
                    "y2": np.random.randn(),
                    "y3": np.random.randn(),
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Test with p=1
        pvar_data_1 = PanelVARData(
            df,
            endog_vars=["y1", "y2", "y3"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="constant",
        )
        y, X = pvar_data_1.equation_data(0)
        # y should have n_obs observations
        assert y.shape == (pvar_data_1.n_obs,)
        # X should have: K*p lags + 1 constant = 3*1 + 1 = 4
        assert X.shape == (pvar_data_1.n_obs, 4)

        # Test with p=2
        pvar_data_2 = PanelVARData(
            df,
            endog_vars=["y1", "y2", "y3"],
            entity_col="entity",
            time_col="time",
            lags=2,
            trend="constant",
        )
        y, X = pvar_data_2.equation_data(0)
        # X should have: K*p lags + 1 constant = 3*2 + 1 = 7
        assert X.shape == (pvar_data_2.n_obs, 7)

        # Test with p=3
        pvar_data_3 = PanelVARData(
            df,
            endog_vars=["y1", "y2", "y3"],
            entity_col="entity",
            time_col="time",
            lags=3,
            trend="both",
        )
        y, X = pvar_data_3.equation_data(1)
        # X should have: K*p lags + 1 constant + 1 trend = 3*3 + 2 = 11
        assert X.shape == (pvar_data_3.n_obs, 11)


class TestPanelVARDataGaps:
    """Tests for temporal gap detection."""

    def test_panel_with_internal_gap_raises_error(self):
        """Test that internal gaps are detected and raise ValueError."""
        # Create data with a gap: entity A has times [0,1,2,4,5] (missing 3)
        data = [
            {"entity": "A", "time": 0, "y": 1.0},
            {"entity": "A", "time": 1, "y": 2.0},
            {"entity": "A", "time": 2, "y": 3.0},
            {"entity": "A", "time": 4, "y": 4.0},  # Gap: missing time 3
            {"entity": "A", "time": 5, "y": 5.0},
        ]

        df = pd.DataFrame(data)

        # This should raise ValueError
        with pytest.raises(ValueError, match="internal temporal gaps"):
            PanelVARData(df, endog_vars=["y"], entity_col="entity", time_col="time", lags=1)

    def test_panel_without_gaps_succeeds(self):
        """Test that panels without gaps are accepted."""
        # Create continuous data
        data = [
            {"entity": "A", "time": 0, "y": 1.0},
            {"entity": "A", "time": 1, "y": 2.0},
            {"entity": "A", "time": 2, "y": 3.0},
            {"entity": "A", "time": 3, "y": 4.0},
            {"entity": "A", "time": 4, "y": 5.0},
        ]

        df = pd.DataFrame(data)

        # This should work
        pvar_data = PanelVARData(df, endog_vars=["y"], entity_col="entity", time_col="time", lags=1)
        assert pvar_data.N == 1
        assert pvar_data.n_obs == 4  # 5 - 1 lag


class TestPanelVARDataExogenous:
    """Tests for exogenous variables."""

    def test_with_exogenous_variables(self):
        """Test PanelVARData with exogenous variables."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {
                    "entity": i,
                    "time": t,
                    "y1": np.random.randn(),
                    "y2": np.random.randn(),
                    "x1": np.random.randn(),
                    "x2": np.random.randn(),
                }
                data.append(row)

        df = pd.DataFrame(data)

        # Create with exogenous variables
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            exog_vars=["x1", "x2"],
            lags=1,
            trend="constant",
        )

        # Check equation data dimensions
        y, X = pvar_data.equation_data(0)
        # X should have: K*p lags + n_exog + constant = 2*1 + 2 + 1 = 5
        assert X.shape[1] == 5

    def test_without_exogenous_variables(self):
        """Test PanelVARData without exogenous variables."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        # Create without exogenous variables
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="constant",
        )

        # Check equation data dimensions
        y, X = pvar_data.equation_data(0)
        # X should have: K*p lags + constant = 2*1 + 1 = 3
        assert X.shape[1] == 3


class TestPanelVARDataTrend:
    """Tests for trend specifications."""

    def test_trend_none(self):
        """Test with no deterministic terms."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1, trend="none"
        )

        y, X = pvar_data.equation_data(0)
        # X should have only lags: K*p = 2*1 = 2
        assert X.shape[1] == 2

    def test_trend_constant(self):
        """Test with constant only."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="constant",
        )

        y, X = pvar_data.equation_data(0)
        # X should have: K*p + 1 = 2*1 + 1 = 3
        assert X.shape[1] == 3
        # Last column should be all ones
        assert np.allclose(X[:, -1], 1.0)

    def test_trend_linear(self):
        """Test with linear trend only."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1, trend="trend"
        )

        y, X = pvar_data.equation_data(0)
        # X should have: K*p + 1 trend = 2*1 + 1 = 3
        assert X.shape[1] == 3

    def test_trend_both(self):
        """Test with constant and linear trend."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(10):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1, trend="both"
        )

        y, X = pvar_data.equation_data(0)
        # X should have: K*p + constant + trend = 2*1 + 2 = 4
        assert X.shape[1] == 4


class TestPanelVARDataValidation:
    """Tests for input validation."""

    def test_missing_columns_raises_error(self):
        """Test that missing columns raise appropriate errors."""
        df = pd.DataFrame({"entity": [1, 2], "time": [1, 2], "y1": [1.0, 2.0]})

        # Missing y2
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(df, endog_vars=["y1", "y2"], entity_col="entity", time_col="time", lags=1)

        # Missing entity column
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(df, endog_vars=["y1"], entity_col="wrong_entity", time_col="time", lags=1)

    def test_invalid_lags_raises_error(self):
        """Test that invalid lag values raise errors."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})

        with pytest.raises(ValueError, match="lags must be >= 1"):
            PanelVARData(df, endog_vars=["y1"], entity_col="entity", time_col="time", lags=0)

    def test_invalid_trend_raises_error(self):
        """Test that invalid trend specification raises error."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})

        with pytest.raises(ValueError, match="trend must be one of"):
            PanelVARData(
                df, endog_vars=["y1"], entity_col="entity", time_col="time", lags=1, trend="invalid"
            )


class TestPanelVARDataRegressorNames:
    """Tests for regressor name generation."""

    def test_regressor_names_simple(self):
        """Test regressor names for simple case."""
        np.random.seed(42)
        data = []
        for i in range(2):
            for t in range(5):
                row = {"entity": i, "time": t, "y1": np.random.randn(), "y2": np.random.randn()}
                data.append(row)

        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="constant",
        )

        names = pvar_data.get_regressor_names()
        expected = ["L1.y1", "L1.y2", "const"]
        assert names == expected

    def test_regressor_names_with_exog(self):
        """Test regressor names with exogenous variables."""
        np.random.seed(42)
        data = []
        for i in range(2):
            for t in range(5):
                row = {
                    "entity": i,
                    "time": t,
                    "y1": np.random.randn(),
                    "y2": np.random.randn(),
                    "x1": np.random.randn(),
                }
                data.append(row)

        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            exog_vars=["x1"],
            lags=2,
            trend="both",
        )

        names = pvar_data.get_regressor_names()
        expected = ["L1.y1", "L1.y2", "L2.y1", "L2.y2", "x1", "const", "trend"]
        assert names == expected
