"""
Tests for Panel VAR GMM instrument matrix construction
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from panelbox.var.instruments import PanelVARInstruments, build_gmm_instruments


class TestPanelVARInstrumentsBasic:
    """Basic tests for instrument construction"""

    def test_initialization(self):
        """Test initialization with valid parameters"""
        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        assert builder.var_lags == 1
        assert builder.n_vars == 2
        assert builder.instrument_type == "all"
        assert builder.max_instruments is None

    def test_invalid_instrument_type(self):
        """Test error on invalid instrument type"""
        with pytest.raises(ValueError, match="instrument_type must be"):
            PanelVARInstruments(var_lags=1, n_vars=2, instrument_type="invalid")

    def test_basic_all_instruments(self):
        """Test construction with all instruments on simple panel"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # For VAR(1): valid instruments start at t-2
        # t=3: instrument is t=1 (1 lag × 2 vars = 2 instruments)
        # t=4: instruments are t=1, t=2 (2 lags × 2 vars = 4 instruments)

        assert Z.shape[0] == 2  # 2 observations with valid instruments
        assert Z.shape[1] > 0  # Has instruments
        assert meta["instruments_type"] == "all"
        assert meta["total_instruments"] == Z.shape[1]

    def test_basic_collapsed_instruments(self):
        """Test construction with collapsed instruments"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "y2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2, instrument_type="collapsed")
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        assert Z.shape[0] > 0
        assert meta["instruments_type"] == "collapsed"
        assert "lag_depths" in meta


class TestInstrumentMatrixStructure:
    """Tests for instrument matrix structure and properties"""

    def test_all_instruments_structure(self):
        """Test structure of all instruments matrix"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 1],
                "time": [1, 2, 3, 4, 5],
                "y1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "y2": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2, instrument_type="all")
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # For VAR(1) with T=5, valid instruments:
        # t=3: lags [1] -> 1 lag × 2 vars = 2 instruments
        # t=4: lags [1, 2] -> 2 lags × 2 vars = 4 instruments
        # t=5: lags [1, 2, 3] -> 3 lags × 2 vars = 6 instruments

        # But all rows must have same number of columns (padded with zeros)
        assert Z.shape[0] == 3  # 3 observations with valid instruments

    def test_collapsed_reduces_instruments(self):
        """Test that collapsed can reduce instrument count vs all in large T scenarios"""
        # Create panel with large T to demonstrate collapse benefit
        df = pd.DataFrame(
            {
                "entity": [1] * 20 + [2] * 20,  # 2 entities
                "time": list(range(1, 21)) + list(range(1, 21)),  # T = 20
                "y1": np.arange(40, dtype=float),
                "y2": np.arange(40, dtype=float) * 10,
            }
        )

        # Without max_instruments, all instruments grows with T
        builder_all = PanelVARInstruments(var_lags=1, n_vars=2, instrument_type="all")
        Z_all, meta_all = builder_all.construct_instruments(df, value_cols=["y1", "y2"])

        # Collapsed with limited lags
        builder_collapsed = PanelVARInstruments(
            var_lags=1, n_vars=2, instrument_type="collapsed", max_instruments=3  # Limit to 3 lags
        )
        Z_collapsed, meta_collapsed = builder_collapsed.construct_instruments(
            df, value_cols=["y1", "y2"]
        )

        # With max_instruments, collapsed should be smaller
        assert meta_collapsed["total_instruments"] <= meta_all["total_instruments"]


class TestMaxInstruments:
    """Tests for max_instruments constraint"""

    def test_max_instruments_all(self):
        """Test max_instruments constraint with all instruments"""
        df = pd.DataFrame(
            {
                "entity": [1] * 10,
                "time": list(range(1, 11)),
                "y1": np.arange(10, dtype=float),
                "y2": np.arange(10, dtype=float) * 10,
            }
        )

        # Without constraint
        builder_unlimited = PanelVARInstruments(var_lags=1, n_vars=2, instrument_type="all")
        Z_unlimited, meta_unlimited = builder_unlimited.construct_instruments(
            df, value_cols=["y1", "y2"]
        )

        # With constraint
        builder_limited = PanelVARInstruments(
            var_lags=1, n_vars=2, instrument_type="all", max_instruments=2
        )
        Z_limited, meta_limited = builder_limited.construct_instruments(df, value_cols=["y1", "y2"])

        # Limited should have fewer or equal total instruments
        assert meta_limited["total_instruments"] <= meta_unlimited["total_instruments"]

    def test_max_instruments_collapsed(self):
        """Test max_instruments constraint with collapsed instruments"""
        df = pd.DataFrame(
            {
                "entity": [1] * 8,
                "time": list(range(1, 9)),
                "y1": np.arange(8, dtype=float),
                "y2": np.arange(8, dtype=float) * 10,
            }
        )

        builder = PanelVARInstruments(
            var_lags=1, n_vars=2, instrument_type="collapsed", max_instruments=3
        )
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # Should respect max_instruments
        n_lags_per_var = meta["n_instruments_per_variable"]
        assert n_lags_per_var <= 3


class TestMultipleEntities:
    """Tests with multiple entities"""

    def test_two_entities_balanced(self):
        """Test instrument construction with two balanced entities"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4],
                "y1": [1, 2, 3, 4, 5, 6, 7, 8],
                "y2": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # Each entity should contribute observations with valid instruments
        # For each entity with T=4, VAR(1): t=3,4 have valid instruments
        assert Z.shape[0] == 4  # 2 entities × 2 valid time periods

    def test_unbalanced_panel(self):
        """Test instrument construction with unbalanced panel"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 1, 2, 2, 2],  # Entity 1: 5 periods, Entity 2: 3 periods
                "time": [1, 2, 3, 4, 5, 1, 2, 3],
                "y1": [1, 2, 3, 4, 5, 10, 20, 30],
                "y2": [10, 20, 30, 40, 50, 100, 200, 300],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # Entity 1: t=3,4,5 have valid instruments
        # Entity 2: t=3 has valid instruments
        assert Z.shape[0] == 4


class TestInstrumentValidity:
    """Tests for instrument validity (predetermined)"""

    def test_instruments_are_predetermined(self):
        """Test that instruments are dated t-p-1 or earlier"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # Check metadata for valid lag ranges
        obs_meta = meta["observation_metadata"]
        for obs in obs_meta:
            t = obs["time"]
            if obs["lag_range"] is not None:
                max_lag_used = obs["lag_range"][1]
                # For VAR(1): max lag should be t-2 or earlier
                assert max_lag_used <= t - 2


class TestEdgeCases:
    """Tests for edge cases"""

    def test_insufficient_time_periods(self):
        """Test error when T too small for VAR(p)"""
        df = pd.DataFrame({"entity": [1, 1], "time": [1, 2], "y1": [1.0, 2.0], "y2": [10.0, 20.0]})

        builder = PanelVARInstruments(var_lags=1, n_vars=2)

        # For VAR(1), need at least t=3 for first valid instrument
        with pytest.raises(ValueError, match="No valid instruments"):
            builder.construct_instruments(df, value_cols=["y1", "y2"])

    def test_wrong_number_of_variables(self):
        """Test error when variable count mismatch"""
        df = pd.DataFrame({"entity": [1, 1, 1], "time": [1, 2, 3], "y1": [1.0, 2.0, 3.0]})

        builder = PanelVARInstruments(var_lags=1, n_vars=2)  # Expects 2 vars

        with pytest.raises(ValueError, match="Expected 2 variables, got 1"):
            builder.construct_instruments(df, value_cols=["y1"])

    def test_proliferation_warning(self):
        """Test warning when instrument count exceeds entity count"""
        # Create panel with many time periods but few entities
        df = pd.DataFrame(
            {
                "entity": [1] * 20,  # Only 1 entity
                "time": list(range(1, 21)),  # 20 time periods
                "y1": np.arange(20, dtype=float),
                "y2": np.arange(20, dtype=float) * 10,
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2, instrument_type="all")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

            # Should trigger proliferation warning
            assert len(w) > 0
            assert "exceeds number of entities" in str(w[0].message)


class TestConvenienceFunction:
    """Tests for build_gmm_instruments convenience function"""

    def test_build_gmm_instruments_all(self):
        """Test convenience function with all instruments"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        Z, meta = build_gmm_instruments(df, var_lags=1, n_vars=2, value_cols=["y1", "y2"])

        assert Z.shape[0] > 0
        assert meta["instruments_type"] == "all"

    def test_build_gmm_instruments_collapsed(self):
        """Test convenience function with collapsed instruments"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1, 2, 2, 2, 2],
                "time": [1, 2, 3, 4, 1, 2, 3, 4],
                "y1": [1, 2, 3, 4, 5, 6, 7, 8],
                "y2": [10, 20, 30, 40, 50, 60, 70, 80],
            }
        )

        Z, meta = build_gmm_instruments(
            df, var_lags=1, n_vars=2, value_cols=["y1", "y2"], instrument_type="collapsed"
        )

        assert Z.shape[0] > 0
        assert meta["instruments_type"] == "collapsed"


class TestMetadata:
    """Tests for instrument metadata"""

    def test_metadata_structure(self):
        """Test that metadata has required fields"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        # Required fields
        assert "total_instruments" in meta
        assert "n_instruments_per_variable" in meta
        assert "n_instruments_per_equation" in meta
        assert "instruments_type" in meta
        assert "observation_metadata" in meta

    def test_instrument_count_summary(self):
        """Test summary generation"""
        df = pd.DataFrame(
            {
                "entity": [1, 1, 1, 1],
                "time": [1, 2, 3, 4],
                "y1": [1.0, 2.0, 3.0, 4.0],
                "y2": [10.0, 20.0, 30.0, 40.0],
            }
        )

        builder = PanelVARInstruments(var_lags=1, n_vars=2)
        Z, meta = builder.construct_instruments(df, value_cols=["y1", "y2"])

        summary = builder.get_instrument_count_summary(meta)

        assert "GMM Instrument Matrix Summary" in summary
        assert "Instrument type:" in summary
        assert "Total instruments:" in summary
