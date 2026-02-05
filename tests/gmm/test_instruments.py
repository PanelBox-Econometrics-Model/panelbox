"""
Unit tests for GMM instrument generation.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm.instruments import EquationType, InstrumentBuilder, InstrumentSet, InstrumentStyle


class TestInstrumentSet:
    """Tests for InstrumentSet class."""

    def test_creation(self):
        """Test basic InstrumentSet creation."""
        Z = np.random.randn(100, 10)
        inst_set = InstrumentSet(
            Z=Z,
            variable_names=["y"],
            instrument_names=[f"y_L{i}" for i in range(2, 12)],
            equation="diff",
            style="gmm",
            collapsed=True,
        )

        assert inst_set.n_instruments == 10
        assert inst_set.n_obs == 100
        assert inst_set.equation == "diff"
        assert inst_set.style == "gmm"
        assert inst_set.collapsed

    def test_n_instruments_property(self):
        """Test n_instruments property."""
        Z = np.random.randn(50, 5)
        inst_set = InstrumentSet(Z=Z)
        assert inst_set.n_instruments == 5

    def test_n_obs_property(self):
        """Test n_obs property."""
        Z = np.random.randn(50, 5)
        inst_set = InstrumentSet(Z=Z)
        assert inst_set.n_obs == 50

    def test_repr(self):
        """Test string representation."""
        Z = np.random.randn(50, 5)
        inst_set = InstrumentSet(Z=Z, equation="diff", style="gmm", collapsed=True)
        r = repr(inst_set)

        assert "InstrumentSet" in r
        assert "n_instruments=5" in r
        assert "n_obs=50" in r
        assert "equation='diff'" in r


class TestInstrumentBuilder:
    """Tests for InstrumentBuilder class."""

    @pytest.fixture
    def simple_panel(self):
        """Create simple balanced panel data."""
        data = []
        np.random.seed(42)
        for i in range(1, 6):  # 5 groups
            for t in range(1, 5):  # 4 periods
                data.append({"id": i, "year": t, "y": np.random.randn(), "x": np.random.randn()})
        return pd.DataFrame(data)

    def test_initialization(self, simple_panel):
        """Test InstrumentBuilder initialization."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        assert builder.n_groups == 5
        assert builder.n_periods == 4
        assert len(builder.time_periods) == 4
        assert list(builder.time_periods) == [1, 2, 3, 4]

    def test_iv_style_instruments_basic(self, simple_panel):
        """Test basic IV-style instrument generation."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_iv_style_instruments(var="x", min_lag=0, max_lag=0, equation="diff")

        assert Z.n_instruments == 1
        assert Z.n_obs == 20  # 5 groups * 4 periods
        assert Z.style == "iv"
        assert not Z.collapsed
        assert Z.instrument_names == ["x_L0"]

    def test_iv_style_instruments_multiple_lags(self, simple_panel):
        """Test IV-style instruments with multiple lags."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_iv_style_instruments(var="x", min_lag=1, max_lag=2, equation="diff")

        assert Z.n_instruments == 2
        assert Z.instrument_names == ["x_L1", "x_L2"]

    def test_gmm_style_collapsed(self, simple_panel):
        """Test GMM-style collapsed instruments."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=99, equation="diff", collapse=True
        )

        assert Z.style == "gmm"
        assert Z.collapsed
        # With 4 periods, max available lag is 3 (n_periods - 1)
        # So lags 2 and 3 should be created
        assert Z.n_instruments == 2
        assert "y_L2_collapsed" in Z.instrument_names
        assert "y_L3_collapsed" in Z.instrument_names

    def test_gmm_style_collapsed_respects_max_lag(self, simple_panel):
        """Test that collapsed GMM instruments respect data limitations."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        # Request lags up to 99, but only 3 periods available
        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=99, equation="diff", collapse=True
        )

        # Should only create lags that exist in data
        assert Z.n_instruments <= builder.n_periods

    def test_gmm_style_non_collapsed(self, simple_panel):
        """Test GMM-style non-collapsed instruments."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=3, equation="diff", collapse=False
        )

        assert Z.style == "gmm"
        assert not Z.collapsed
        # Non-collapsed creates separate columns for each time*lag combination
        assert Z.n_instruments >= 2  # At least one per lag

    def test_combine_instruments(self, simple_panel):
        """Test combining multiple instrument sets."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z1 = builder.create_gmm_style_instruments(var="y", min_lag=2, max_lag=99, collapse=True)
        Z2 = builder.create_iv_style_instruments(var="x", min_lag=0, max_lag=0)

        Z_combined = builder.combine_instruments(Z1, Z2)

        assert Z_combined.n_instruments == Z1.n_instruments + Z2.n_instruments
        assert Z_combined.n_obs == Z1.n_obs
        assert len(Z_combined.variable_names) == len(Z1.variable_names) + len(Z2.variable_names)

    def test_instrument_count_analysis(self, simple_panel):
        """Test instrument count analysis."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(var="y", min_lag=2, max_lag=99, collapse=True)

        analysis = builder.instrument_count_analysis(Z)

        assert isinstance(analysis, pd.DataFrame)
        assert "Total instruments" in analysis.index
        assert "Groups" in analysis.index
        assert "Instrument ratio" in analysis.index

    def test_instrument_count_analysis_warning(self, simple_panel):
        """Test instrument count analysis with too many instruments."""
        # Create data with more instruments than groups
        data = []
        for i in range(1, 3):  # Only 2 groups
            for t in range(1, 10):  # Many periods
                data.append({"id": i, "year": t, "y": np.random.randn()})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        # Create many instruments (non-collapsed)
        Z = builder.create_gmm_style_instruments(var="y", min_lag=2, max_lag=99, collapse=False)

        analysis = builder.instrument_count_analysis(Z)

        # Should have warning about too many instruments
        if Z.n_instruments > builder.n_groups:
            assert "Warning" in analysis.index
            assert "Too many instruments" in str(analysis.loc["Warning"].values[0])

    def test_get_valid_obs_mask(self, simple_panel):
        """Test valid observation mask generation."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(var="y", min_lag=2, max_lag=99, collapse=True)

        mask = builder.get_valid_obs_mask(Z)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == Z.n_obs
        assert np.any(mask)  # At least some valid observations


class TestInstrumentGeneration:
    """Integration tests for instrument generation."""

    @pytest.fixture
    def unbalanced_panel(self):
        """Create unbalanced panel data."""
        data = []
        np.random.seed(42)
        # Group 1: periods 1-4
        for t in range(1, 5):
            data.append({"id": 1, "year": t, "y": np.random.randn()})
        # Group 2: periods 1, 3, 4 (missing 2)
        for t in [1, 3, 4]:
            data.append({"id": 2, "year": t, "y": np.random.randn()})
        # Group 3: periods 2-4
        for t in range(2, 5):
            data.append({"id": 3, "year": t, "y": np.random.randn()})

        return pd.DataFrame(data)

    def test_handles_unbalanced_panel(self, unbalanced_panel):
        """Test that instrument generation handles unbalanced panels."""
        builder = InstrumentBuilder(unbalanced_panel, "id", "year")

        # Should not crash with unbalanced data
        Z = builder.create_gmm_style_instruments(var="y", min_lag=2, max_lag=99, collapse=True)

        assert Z.n_instruments > 0
        # Should have NaN for missing observations
        assert np.any(np.isnan(Z.Z))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
