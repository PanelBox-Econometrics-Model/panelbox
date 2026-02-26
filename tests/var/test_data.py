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
        for _idx, row in entity_a.iterrows():
            t = row["time"]
            if t >= 1:  # Skip first observation (no lag available)
                lag_value = row["L1.y"]
                expected_value = float(f"{ord('A')}.{t - 1}")
                assert np.isclose(lag_value, expected_value), f"Entity A, time {t}: lag mismatch"

        # For entity B, L1.y at time t should equal y at time t-1 (within B, not A)
        entity_b = df_with_lags[df_with_lags["entity"] == "B"]
        for _idx, row in entity_b.iterrows():
            t = row["time"]
            if t >= 1:
                lag_value = row["L1.y"]
                expected_value = float(f"{ord('B')}.{t - 1}")
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
        _y, X = pvar_data.equation_data(0)
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
        _y, X = pvar_data.equation_data(0)
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

        _y, X = pvar_data.equation_data(0)
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

        _y, X = pvar_data.equation_data(0)
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

        _y, X = pvar_data.equation_data(0)
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

        _y, X = pvar_data.equation_data(0)
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


class TestPanelVARDataInputValidation:
    """Tests for uncovered validation branches in PanelVARData._validate_inputs."""

    def test_data_not_dataframe_raises_type_error(self):
        """Line 166: data is not a DataFrame raises TypeError."""
        with pytest.raises(TypeError, match="must be a DataFrame"):
            PanelVARData(
                "not_a_dataframe",
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    def test_data_dict_raises_type_error(self):
        """Line 166: passing a dict (not DataFrame) raises TypeError."""
        with pytest.raises(TypeError, match="must be a DataFrame"):
            PanelVARData(
                {"y1": [1, 2], "entity": [1, 1], "time": [1, 2]},
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    def test_data_numpy_array_raises_type_error(self):
        """Line 166: passing a numpy array raises TypeError."""
        with pytest.raises(TypeError, match="must be a DataFrame"):
            PanelVARData(
                np.array([[1, 2], [3, 4]]),
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    def test_missing_time_col_raises_value_error(self):
        """Line 173: missing time_col in data raises ValueError."""
        df = pd.DataFrame({"y1": [1, 2, 3], "y2": [4, 5, 6], "entity": [1, 1, 1]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                entity_col="entity",
                time_col="nonexistent",
                lags=1,
            )

    def test_missing_entity_col_only(self):
        """Line 171: only entity_col missing (time_col exists)."""
        df = pd.DataFrame({"y1": [1.0, 2.0], "time": [1, 2]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                entity_col="nonexistent_entity",
                time_col="time",
                lags=1,
            )

    def test_missing_endog_var_columns(self):
        """Line 181: missing endogenous variable columns."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1", "nonexistent_var"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    def test_missing_exog_var_columns(self):
        """Line 188: missing exogenous variable columns."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                exog_vars=["nonexistent_exog"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    def test_multiple_missing_exog_columns(self):
        """Line 181/188: multiple missing exog columns all listed."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                exog_vars=["missing_x1", "missing_x2"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )


class TestPanelVARDataEquationDataEdgeCases:
    """Tests for uncovered branches in equation_data()."""

    def test_equation_index_too_large(self):
        """Line 427: k >= K raises ValueError."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(5):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )

        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(k=5)

    def test_equation_index_negative(self):
        """Line 427: k < 0 raises ValueError."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(5):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )

        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(k=-1)

    def test_equation_index_exactly_K(self):
        """Line 427: k == K (boundary) raises ValueError."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(5):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )

        # K=2, so k=2 is out of bounds
        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(k=2)

    def test_equation_data_no_constant_no_trend(self):
        """Line 469: empty X_list when trend='none' and include_constant=False."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(5):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="none",
        )

        # With lags=1, X should still have lag columns even with include_constant=False
        y, X = pvar_data.equation_data(0, include_constant=False)
        # K*p = 2*1 = 2 lag columns, no constant, no trend
        assert X.shape[1] == 2
        assert y.shape[0] == X.shape[0]

    def test_repr(self):
        """Test __repr__ output."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(5):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )

        r = repr(pvar_data)
        assert "PanelVARData" in r
        assert "K=2" in r
        assert "p=1" in r
        assert "N=3" in r


class TestPanelVARDataCoverageExtras:
    """Additional tests targeting uncovered lines in panelbox/var/data.py."""

    # ------------------------------------------------------------------ #
    # Line 188: empty endog_vars list → "Must have at least 1 endogenous variable"
    # ------------------------------------------------------------------ #
    def test_empty_endog_vars_raises_value_error(self):
        """Line 188: len(endog_vars) < 1 raises ValueError."""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1],
                "time": [1, 2, 3],
                "y1": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="at least 1 endogenous variable"):
            PanelVARData(
                df,
                endog_vars=[],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    # ------------------------------------------------------------------ #
    # Line 316: _verify_no_cross_contamination ValueError
    # This is difficult to trigger through the public API because
    # _construct_lags always builds correct lags. We test it by
    # manually corrupting the lag column after construction and
    # calling the verification method directly.
    # ------------------------------------------------------------------ #
    def test_cross_contamination_detection(self):
        """Line 316: Cross-contamination detected raises ValueError."""
        np.random.seed(42)
        data = []
        for entity in ["A", "B"]:
            for t in range(6):
                data.append(
                    {
                        "entity": entity,
                        "time": t,
                        "y": float(t) + (0.0 if entity == "A" else 100.0),
                    }
                )
        df = pd.DataFrame(data)

        # Create a valid PanelVARData first
        pvar_data = PanelVARData(
            df,
            endog_vars=["y"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )

        # Now corrupt the lag column so verification would fail
        # Swap a lagged value to create a mismatch
        corrupted_df = pvar_data.data_with_lags.copy()
        # Find entity A rows and corrupt L1.y
        mask_a = corrupted_df["entity"] == "A"
        a_indices = corrupted_df[mask_a].index
        if len(a_indices) > 1:
            # Set a lagged value to something clearly wrong
            corrupted_df.loc[a_indices[1], "L1.y"] = 999999.0

        # Overwrite and re-run verification → should raise
        pvar_data.data_with_lags = corrupted_df
        with pytest.raises(ValueError, match="Cross-contamination detected"):
            pvar_data._verify_no_cross_contamination()

    # ------------------------------------------------------------------ #
    # Line 427: equation_data with k out of range (additional boundary)
    # ------------------------------------------------------------------ #
    def test_equation_data_k_exactly_minus1(self):
        """Line 427: k = -1 raises ValueError."""
        np.random.seed(42)
        data = []
        for i in range(2):
            for t in range(5):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(k=-1)

    # ------------------------------------------------------------------ #
    # Line 469: empty X matrix path
    # With lags >= 1, lag columns always exist, so the truly empty path
    # (len(X_list) == 0) is unreachable. Instead we test the closest
    # scenario: trend='none' + include_constant=False, which produces
    # an X with only lag columns and no deterministic terms.
    # ------------------------------------------------------------------ #
    def test_equation_data_include_constant_false_trend_none(self):
        """Line 469 vicinity: minimal X with include_constant=False and trend='none'."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(8):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="none",
        )

        y, X = pvar_data.equation_data(0, include_constant=False)
        # Only lag columns: K*p = 1*1 = 1
        assert X.shape[1] == 1
        assert y.shape[0] == X.shape[0]

    # ------------------------------------------------------------------ #
    # Line 173: missing time_col specifically (entity_col present)
    # ------------------------------------------------------------------ #
    def test_missing_time_col_with_entity_present(self):
        """Line 173: time_col not in data columns."""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1],
                "y1": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                entity_col="entity",
                time_col="date",  # not present
                lags=1,
            )

    # ------------------------------------------------------------------ #
    # Line 181: missing exog variable column
    # ------------------------------------------------------------------ #
    def test_missing_exog_column(self):
        """Line 181: exog_vars column not found in data."""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1],
                "time": [1, 2, 3],
                "y1": [1.0, 2.0, 3.0],
            }
        )
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                exog_vars=["z_not_present"],
                entity_col="entity",
                time_col="time",
                lags=1,
            )

    # ------------------------------------------------------------------ #
    # equation_data with include_constant=True + trend='both' verifies
    # both constant and trend columns are added.
    # ------------------------------------------------------------------ #
    def test_equation_data_include_constant_true_trend_both(self):
        """Verify include_constant=True with trend='both' adds constant and trend."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(8):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="both",
        )

        _y, X = pvar_data.equation_data(0, include_constant=True)
        # K*p lags + constant + trend = 2*1 + 1 + 1 = 4
        assert X.shape[1] == 4
        # Second-to-last column should be all ones (constant)
        assert np.allclose(X[:, -2], 1.0)
        # Last column should be a trend (1, 2, 3, ...)
        assert X[0, -1] == 1.0
        assert X[1, -1] == 2.0

    def test_equation_data_include_constant_false_trend_both(self):
        """Verify include_constant=False suppresses constant and trend even when trend='both'."""
        np.random.seed(42)
        data = []
        for i in range(3):
            for t in range(8):
                data.append(
                    {
                        "entity": i,
                        "time": t,
                        "y1": np.random.randn(),
                        "y2": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="both",
        )

        _y, X = pvar_data.equation_data(0, include_constant=False)
        # Only lag columns: K*p = 2*1 = 2
        assert X.shape[1] == 2


class TestDataCoverage:
    """Tests targeting specific uncovered lines in panelbox/var/data.py."""

    # ------------------------------------------------------------------ #
    # Helper to create simple panel data
    # ------------------------------------------------------------------ #
    @staticmethod
    def _make_simple_data(n_entities=3, n_periods=8, vars_list=None):
        """Create simple balanced panel DataFrame."""
        if vars_list is None:
            vars_list = ["y1", "y2"]
        np.random.seed(42)
        rows = []
        for i in range(n_entities):
            for t in range(n_periods):
                row = {"entity": i, "time": t}
                for v in vars_list:
                    row[v] = np.random.randn()
                rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # Line 166: raise TypeError for non-DataFrame input
    # ------------------------------------------------------------------ #
    def test_non_dataframe_dict_raises_type_error(self):
        """Line 166: passing a dict raises TypeError."""
        with pytest.raises(TypeError, match="data must be a DataFrame"):
            PanelVARData(
                data={"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]},
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
            )

    def test_non_dataframe_list_raises_type_error(self):
        """Line 166: passing a list raises TypeError."""
        with pytest.raises(TypeError, match="data must be a DataFrame"):
            PanelVARData(
                data=[[1, 2], [3, 4]],
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
            )

    def test_non_dataframe_none_raises_type_error(self):
        """Line 166: passing None raises TypeError."""
        with pytest.raises(TypeError, match="data must be a DataFrame"):
            PanelVARData(
                data=None,
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
            )

    # ------------------------------------------------------------------ #
    # Line 173: entity_col missing from data
    # ------------------------------------------------------------------ #
    def test_missing_entity_col_raises_value_error(self):
        """Line 170-171: entity_col not in data columns."""
        df = pd.DataFrame({"time": [1, 2, 3], "y1": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                entity_col="entity",  # not in df
                time_col="time",
            )

    def test_missing_time_col_raises_value_error(self):
        """Line 172-173: time_col not in data columns."""
        df = pd.DataFrame({"entity": [1, 1, 1], "y1": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",  # not in df
            )

    def test_both_entity_and_time_missing(self):
        """Lines 170-173: both entity_col and time_col missing."""
        df = pd.DataFrame({"y1": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                entity_col="entity",
                time_col="time",
            )

    # ------------------------------------------------------------------ #
    # Line 181: missing exog_var in data
    # ------------------------------------------------------------------ #
    def test_missing_exog_var_raises_value_error(self):
        """Line 179-181: exog_vars column not found in data."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                exog_vars=["missing_var"],
                entity_col="entity",
                time_col="time",
            )

    def test_multiple_missing_exog_vars(self):
        """Line 179-181: multiple exog vars missing."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            PanelVARData(
                df,
                endog_vars=["y1"],
                exog_vars=["x1", "x2"],
                entity_col="entity",
                time_col="time",
            )

    # ------------------------------------------------------------------ #
    # Line 188: empty endog_vars list
    # ------------------------------------------------------------------ #
    def test_empty_endog_vars_raises_value_error(self):
        """Line 187-188: len(endog_vars) < 1 raises ValueError."""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0]})
        with pytest.raises(ValueError, match="at least 1 endogenous variable"):
            PanelVARData(
                df,
                endog_vars=[],
                entity_col="entity",
                time_col="time",
            )

    # ------------------------------------------------------------------ #
    # Line 316: cross-contamination detection
    # ------------------------------------------------------------------ #
    def test_cross_contamination_detection_via_corruption(self):
        """Line 316: corrupted lag data triggers cross-contamination ValueError."""
        np.random.seed(42)
        rows = []
        for entity in ["A", "B"]:
            for t in range(10):
                rows.append(
                    {
                        "entity": entity,
                        "time": t,
                        "y": float(t) + (0.0 if entity == "A" else 1000.0),
                    }
                )
        df = pd.DataFrame(rows)

        pvar_data = PanelVARData(
            df,
            endog_vars=["y"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )

        # Corrupt a lag value to simulate cross-contamination
        corrupted = pvar_data.data_with_lags.copy()
        mask_a = corrupted["entity"] == "A"
        a_idx = corrupted[mask_a].index
        # Set a lagged value to something obviously wrong
        corrupted.loc[a_idx[2], "L1.y"] = -999999.0

        pvar_data.data_with_lags = corrupted
        with pytest.raises(ValueError, match="Cross-contamination detected"):
            pvar_data._verify_no_cross_contamination()

    # ------------------------------------------------------------------ #
    # Line 427: equation_data with invalid k
    # ------------------------------------------------------------------ #
    def test_equation_index_negative(self):
        """Line 426-427: k < 0 raises ValueError."""
        df = self._make_simple_data()
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(-1)

    def test_equation_index_too_large(self):
        """Line 426-427: k >= K raises ValueError."""
        df = self._make_simple_data()
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(5)

    def test_equation_index_exactly_K(self):
        """Line 426-427: k == K (boundary) raises ValueError."""
        df = self._make_simple_data()
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
        )
        # K=2, so k=2 should be out of bounds
        with pytest.raises(ValueError, match="Equation index"):
            pvar_data.equation_data(2)

    # ------------------------------------------------------------------ #
    # Line 469: empty X matrix when no regressors
    # This line is unreachable through normal usage (lags >= 1 enforced),
    # so we monkey-patch _lags to 0 after construction to reach it.
    # ------------------------------------------------------------------ #
    def test_empty_x_matrix_when_no_regressors(self):
        """Line 469: X = np.empty((len(df), 0)) when X_list is empty."""
        df = self._make_simple_data()
        pvar_data = PanelVARData(
            df,
            endog_vars=["y1", "y2"],
            entity_col="entity",
            time_col="time",
            lags=1,
            trend="none",
        )

        # Monkey-patch _lags to 0 so no lag columns are generated
        pvar_data._lags = 0

        # With _lags=0, no exog, trend='none', include_constant=False
        # -> X_list is empty -> line 469 is reached
        y, X = pvar_data.equation_data(0, include_constant=False)
        assert X.shape == (pvar_data.n_obs, 0)
        assert y.shape == (pvar_data.n_obs,)
