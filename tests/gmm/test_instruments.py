"""
Unit tests for GMM instrument generation.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm.instruments import InstrumentBuilder, InstrumentSet


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
        assert Z.instrument_names == ["D.x_L0"]

    def test_iv_style_instruments_multiple_lags(self, simple_panel):
        """Test IV-style instruments with multiple lags."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_iv_style_instruments(var="x", min_lag=1, max_lag=2, equation="diff")

        assert Z.n_instruments == 2
        assert Z.instrument_names == ["D.x_L1", "D.x_L2"]

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


class TestDatetimeTimeVariable:
    """Tests for datetime/PeriodDtype time variable conversion (lines 150-152)."""

    def test_datetime_time_variable(self):
        """Test InstrumentBuilder with datetime time column."""
        np.random.seed(42)
        data = []
        for i in range(1, 6):
            for t in range(4):
                data.append(
                    {
                        "id": i,
                        "date": pd.Timestamp("2020-01-01") + pd.DateOffset(years=t),
                        "y": np.random.randn(),
                        "x": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "date")

        # The datetime should be converted to numeric indices 0,1,2,3
        assert builder.n_groups == 5
        assert builder.n_periods == 4
        assert builder._time_mapping is not None
        assert len(builder._time_mapping) == 4
        # Mapping should map original datetime keys to sequential integers
        assert all(isinstance(v, (int, np.integer)) for v in builder._time_mapping.values())

        # Should work for instrument generation
        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=99, equation="diff", collapse=True
        )
        assert Z.n_instruments > 0

    def test_period_dtype_time_variable(self):
        """Test InstrumentBuilder with PeriodDtype time column."""
        np.random.seed(42)
        data = []
        periods = pd.period_range("2020", periods=4, freq="Y")
        for i in range(1, 6):
            for t in periods:
                data.append(
                    {
                        "id": i,
                        "period": t,
                        "y": np.random.randn(),
                        "x": np.random.randn(),
                    }
                )
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "period")

        assert builder.n_groups == 5
        assert builder.n_periods == 4
        assert builder._time_mapping is not None
        assert len(builder._time_mapping) == 4

        # Instruments should work fine after conversion
        Z = builder.create_iv_style_instruments(var="x", min_lag=1, max_lag=2, equation="diff")
        assert Z.n_instruments == 2


class TestIVLevelEquation:
    """Tests for IV-style level equation instruments (lines 216, 234)."""

    @pytest.fixture
    def simple_panel(self):
        """Create simple balanced panel data."""
        data = []
        np.random.seed(42)
        for i in range(1, 6):
            for t in range(1, 5):
                data.append({"id": i, "year": t, "y": np.random.randn(), "x": np.random.randn()})
        return pd.DataFrame(data)

    def test_iv_level_instrument_names(self, simple_panel):
        """Test that level equation IV instruments have correct names (line 216)."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_iv_style_instruments(var="x", min_lag=1, max_lag=2, equation="level")

        # Level instruments should NOT have the "D." prefix
        assert Z.instrument_names == ["x_L1", "x_L2"]
        assert Z.equation == "level"
        assert Z.n_instruments == 2

    def test_iv_level_instrument_values(self, simple_panel):
        """Test that level equation IV instruments use raw levels (line 234)."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z_level = builder.create_iv_style_instruments(
            var="x", min_lag=1, max_lag=1, equation="level"
        )
        Z_diff = builder.create_iv_style_instruments(var="x", min_lag=1, max_lag=1, equation="diff")

        # Level and diff should produce different values (levels vs first-differences)
        # At least some valid observations should differ
        valid_both = ~np.isnan(Z_level.Z[:, 0]) & ~np.isnan(Z_diff.Z[:, 0])
        if np.any(valid_both):
            # They should not be identical since level uses x_{t-k}
            # while diff uses Δx_{t-k} = x_{t-k} - x_{t-k-1}
            assert not np.allclose(
                Z_level.Z[valid_both, 0], Z_diff.Z[valid_both, 0], equal_nan=True
            )

    def test_iv_level_single_lag(self, simple_panel):
        """Test IV-style level equation with a single lag."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_iv_style_instruments(var="y", min_lag=0, max_lag=0, equation="level")

        assert Z.n_instruments == 1
        assert Z.instrument_names == ["y_L0"]
        assert Z.style == "iv"
        # Lag 0 means contemporaneous - should have valid values for all obs
        n_valid = np.sum(~np.isnan(Z.Z[:, 0]))
        assert n_valid == 20  # 5 groups * 4 periods


class TestGMMStandardLevelEquation:
    """Tests for GMM-style standard level equation (lines 345-348)."""

    @pytest.fixture
    def simple_panel(self):
        """Create simple balanced panel data."""
        data = []
        np.random.seed(42)
        for i in range(1, 6):
            for t in range(1, 7):  # 6 periods to allow more lags
                data.append({"id": i, "year": t, "y": np.random.randn(), "x": np.random.randn()})
        return pd.DataFrame(data)

    def test_gmm_standard_level_equation(self, simple_panel):
        """Test GMM standard instruments with level equation (lines 345-348).

        For level equation, instruments should use differences (Δx_{i,t-k}).
        """
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=3, equation="level", collapse=False
        )

        assert Z.equation == "level"
        assert Z.style == "gmm"
        assert not Z.collapsed
        assert Z.n_instruments > 0

        # Level equation instruments use differences: x_{t-k} - x_{t-k-1}
        # Some entries may be NaN where the t-k-1 value is unavailable
        assert Z.Z.shape[0] == len(simple_panel)

    def test_gmm_standard_level_vs_diff(self, simple_panel):
        """Test that level and diff equations produce different instruments."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z_diff = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=3, equation="diff", collapse=False
        )
        Z_level = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=3, equation="level", collapse=False
        )

        # Both should have the same number of instruments (same lag structure)
        assert Z_diff.n_instruments == Z_level.n_instruments

        # But the values should differ (levels vs differences)
        for col in range(min(Z_diff.n_instruments, Z_level.n_instruments)):
            valid = ~np.isnan(Z_diff.Z[:, col]) & ~np.isnan(Z_level.Z[:, col])
            if np.any(valid):
                assert not np.allclose(Z_diff.Z[valid, col], Z_level.Z[valid, col])


class TestGMMStandardMaxLagNone:
    """Tests for GMM standard with max_lag=None (line 309)."""

    @pytest.fixture
    def simple_panel(self):
        """Create simple balanced panel data."""
        data = []
        np.random.seed(42)
        for i in range(1, 6):
            for t in range(1, 7):  # 6 periods
                data.append({"id": i, "year": t, "y": np.random.randn(), "x": np.random.randn()})
        return pd.DataFrame(data)

    def test_gmm_standard_max_lag_none(self, simple_panel):
        """Test GMM standard with max_lag=None uses all available lags (line 309)."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=None, equation="diff", collapse=False
        )

        assert Z.style == "gmm"
        assert not Z.collapsed
        # With 6 periods (indices 0-5) and min_lag=2, max_lag=None means effectively infinite
        # For t_idx=2: lags [2]; t_idx=3: lags [2,3]; t_idx=4: lags [2,3,4]; t_idx=5: lags [2,3,4,5]
        # Total: 1 + 2 + 3 + 4 = 10 instrument columns
        assert Z.n_instruments > 0

    def test_gmm_standard_max_lag_none_vs_large(self, simple_panel):
        """Test that max_lag=None and max_lag=99 produce same results."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z_none = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=None, equation="diff", collapse=False
        )
        Z_large = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=99, equation="diff", collapse=False
        )

        # Both should produce the same instruments
        assert Z_none.n_instruments == Z_large.n_instruments
        np.testing.assert_array_equal(Z_none.Z, Z_large.Z)


class TestGMMCollapsedLevelEquation:
    """Tests for GMM collapsed level equation (lines 484-487)."""

    @pytest.fixture
    def simple_panel(self):
        """Create simple balanced panel data."""
        data = []
        np.random.seed(42)
        for i in range(1, 6):
            for t in range(1, 7):  # 6 periods
                data.append({"id": i, "year": t, "y": np.random.randn(), "x": np.random.randn()})
        return pd.DataFrame(data)

    def test_gmm_collapsed_level_equation(self, simple_panel):
        """Test collapsed GMM instruments with level equation (lines 484-487).

        For level equation, collapsed instruments use differences.
        """
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=99, equation="level", collapse=True
        )

        assert Z.equation == "level"
        assert Z.style == "gmm"
        assert Z.collapsed
        assert Z.n_instruments > 0
        # Instrument names should have collapsed suffix
        for name in Z.instrument_names:
            assert "_collapsed" in name

    def test_gmm_collapsed_level_vs_diff(self, simple_panel):
        """Test that collapsed level and diff produce different values."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z_diff = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=4, equation="diff", collapse=True
        )
        Z_level = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=4, equation="level", collapse=True
        )

        # Same number of instruments
        assert Z_diff.n_instruments == Z_level.n_instruments

        # But different values
        for col in range(Z_diff.n_instruments):
            valid = ~np.isnan(Z_diff.Z[:, col]) & ~np.isnan(Z_level.Z[:, col])
            if np.any(valid):
                assert not np.allclose(Z_diff.Z[valid, col], Z_level.Z[valid, col])


class TestGMMCollapsedMaxLagNone:
    """Tests for collapsed GMM with max_lag=None (line 441)."""

    @pytest.fixture
    def simple_panel(self):
        """Create simple balanced panel data."""
        data = []
        np.random.seed(42)
        for i in range(1, 6):
            for t in range(1, 7):  # 6 periods
                data.append({"id": i, "year": t, "y": np.random.randn(), "x": np.random.randn()})
        return pd.DataFrame(data)

    def test_gmm_collapsed_max_lag_none(self, simple_panel):
        """Test collapsed GMM with max_lag=None (line 441)."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=None, equation="diff", collapse=True
        )

        assert Z.collapsed
        # With 6 periods, max lag = 5 (n_periods - 1)
        # So we should get lags 2, 3, 4, 5 if coverage is sufficient
        assert Z.n_instruments > 0

    def test_gmm_collapsed_max_lag_none_vs_large(self, simple_panel):
        """Test that max_lag=None and max_lag=99 produce same collapsed results."""
        builder = InstrumentBuilder(simple_panel, "id", "year")

        Z_none = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=None, equation="diff", collapse=True
        )
        Z_large = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=99, equation="diff", collapse=True
        )

        assert Z_none.n_instruments == Z_large.n_instruments
        np.testing.assert_array_equal(Z_none.Z, Z_large.Z)


class TestLagAvailabilityAndCoverage:
    """Tests for _analyze_lag_availability and poor coverage warning (lines 409, 414, 452-462)."""

    def test_analyze_lag_availability_basic(self):
        """Test _analyze_lag_availability with balanced panel (lines 409, 414)."""
        np.random.seed(42)
        data = []
        for i in range(1, 6):
            for t in range(1, 7):
                data.append({"id": i, "year": t, "y": np.random.randn()})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        # Analyze which lags have sufficient coverage
        valid_lags = builder._analyze_lag_availability("y", min_lag=1, max_lag=5)

        # With balanced data and 6 periods, lags 1-5 should all have some coverage
        # Lag 1: all obs except t=1 (5 groups * 5 valid out of 6) = 83%
        # Lag 5: only t=6 has lag 5 (5 groups * 1 valid out of 6) = 16%
        assert len(valid_lags) > 0
        assert 1 in valid_lags  # Lag 1 should definitely have good coverage

    def test_analyze_lag_availability_high_lags_excluded(self):
        """Test that very high lags with low coverage are excluded."""
        np.random.seed(42)
        data = []
        # Create panel with only 3 periods - lag 3 would have 0% coverage
        for i in range(1, 11):
            for t in range(1, 4):
                data.append({"id": i, "year": t, "y": np.random.randn()})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        valid_lags = builder._analyze_lag_availability("y", min_lag=1, max_lag=5, min_coverage=0.10)

        # Lag 3, 4, 5 cannot exist with only 3 periods (would need t-3 at least)
        # Lag 3 requires periods >=4 which don't exist
        assert 4 not in valid_lags
        assert 5 not in valid_lags

    def test_analyze_lag_availability_nan_values(self):
        """Test _analyze_lag_availability correctly handles NaN values (line 409)."""
        np.random.seed(42)
        data = []
        for i in range(1, 6):
            for t in range(1, 7):
                val = np.random.randn()
                # Introduce NaN for some early values to reduce lag coverage
                if t <= 2:
                    val = np.nan
                data.append({"id": i, "year": t, "y": val})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        valid_lags = builder._analyze_lag_availability("y", min_lag=1, max_lag=4)

        # Lags that would reference t<=2 (NaN values) should have reduced coverage
        # The method should still return lags with sufficient non-NaN coverage
        assert isinstance(valid_lags, list)
        # Each valid lag should be in range [1, 4]
        for lag in valid_lags:
            assert 1 <= lag <= 4

    def test_poor_coverage_warning(self):
        """Test warning when no lags meet the coverage threshold (lines 452-462).

        Create a panel where all lag values are NaN so no lag meets 10% coverage.
        """
        np.random.seed(42)
        data = []
        # Create a panel where variable values are NaN for all early periods
        # so that no lag has >= 10% valid coverage
        for i in range(1, 4):  # 3 groups
            for t in range(1, 4):  # 3 periods
                # Make all values NaN so lagged values are always NaN
                data.append({"id": i, "year": t, "y": np.nan})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        # This should trigger the warning because all lag values are NaN
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Z = builder.create_gmm_style_instruments(
                var="y", min_lag=2, max_lag=99, equation="diff", collapse=True
            )
            # Check that a UserWarning about coverage threshold was raised
            coverage_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning) and "coverage threshold" in str(x.message)
            ]
            assert len(coverage_warnings) == 1
            assert "10% coverage threshold" in str(coverage_warnings[0].message)

        # Should still produce instruments (using fallback lags)
        assert Z.n_instruments > 0

    def test_poor_coverage_fallback_lags(self):
        """Test that the fallback provides min_lag and min_lag+1 when coverage is poor."""
        np.random.seed(42)
        data = []
        # 4 periods with all NaN y values
        for i in range(1, 4):
            for t in range(1, 5):
                data.append({"id": i, "year": t, "y": np.nan})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        import warnings

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            Z = builder.create_gmm_style_instruments(
                var="y", min_lag=2, max_lag=99, equation="diff", collapse=True
            )

        # Fallback should give lags 2 and 3 (min_lag and min_lag+1)
        assert Z.n_instruments == 2
        assert "y_L2_collapsed" in Z.instrument_names
        assert "y_L3_collapsed" in Z.instrument_names


class TestEmptyAvailableLags:
    """Tests for the empty available_lags continue branch (line 326)."""

    def test_gmm_standard_high_min_lag(self):
        """Test GMM standard where min_lag causes empty available_lags for some periods.

        With 4 periods and min_lag=3, only t_idx=3 has lags available.
        Periods t_idx=0,1,2 should trigger the `if not available_lags: continue` path.
        But t_idx < min_lag check skips t_idx=0,1,2 before checking available_lags.
        We need min_lag such that t_idx >= min_lag but range(min_lag, min(max_lag+1, t_idx+1)) is empty.

        Actually, looking at the code more carefully:
        - for t_idx=min_lag, range(min_lag, min(max_lag+1, t_idx+1)) = range(min_lag, min_lag+1) = [min_lag]
        So available_lags is never empty after the t_idx < min_lag check. The empty check is for
        safety/edge cases. Let's create a scenario with max_lag < min_lag to test this.
        """
        np.random.seed(42)
        data = []
        for i in range(1, 6):
            for t in range(1, 7):
                data.append({"id": i, "year": t, "y": np.random.randn()})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        # max_lag < min_lag: the range(min_lag, max_lag+1) will be empty
        # but the code uses min(max_lag+1, t_idx+1) which can also limit it
        # With min_lag=5, max_lag=4 (max_lag < min_lag): range(5, min(5, t_idx+1)) is empty for all t_idx
        # Actually the outer code sets max_lag = int(1e6) if None, but we can pass max_lag explicitly
        # Let's test with max_lag=1 and min_lag=2
        Z = builder.create_gmm_style_instruments(
            var="y", min_lag=2, max_lag=1, equation="diff", collapse=False
        )

        # No instruments should be created since max_lag < min_lag
        assert Z.n_instruments == 0


class TestCombineInstrumentsEmpty:
    """Tests for combine_instruments with no sets (line 525)."""

    def test_combine_instruments_no_sets_raises(self):
        """Test that combine_instruments with no arguments raises ValueError (line 525)."""
        np.random.seed(42)
        data = []
        for i in range(1, 6):
            for t in range(1, 5):
                data.append({"id": i, "year": t, "y": np.random.randn()})
        df = pd.DataFrame(data)

        builder = InstrumentBuilder(df, "id", "year")

        with pytest.raises(ValueError, match="Must provide at least one instrument set"):
            builder.combine_instruments()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
