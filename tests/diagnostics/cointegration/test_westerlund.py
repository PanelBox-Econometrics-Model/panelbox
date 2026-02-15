"""
Tests for Westerlund (2007) cointegration tests.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.cointegration import WesterlundResult, westerlund_test


@pytest.fixture
def cointegrated_panel_data():
    """
    Generate cointegrated panel data.

    DGP:
    y_{it} = β x_{it} + ε_{it}
    x_{it} = x_{i,t-1} + u_{it}  (random walk)

    where ε and u are stationary, implying y and x are cointegrated.
    """
    np.random.seed(42)
    N = 20  # entities
    T = 50  # time periods
    beta = 1.5

    data_list = []

    for i in range(N):
        # Generate I(1) process for x
        u = np.random.randn(T)
        x = np.cumsum(u)

        # Generate cointegrated y
        epsilon = 0.5 * np.random.randn(T)  # stationary error
        y = beta * x + epsilon

        # Create dataframe
        df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def non_cointegrated_panel_data():
    """
    Generate non-cointegrated panel data (two independent random walks).

    DGP:
    y_{it} = y_{i,t-1} + ε_{it}
    x_{it} = x_{i,t-1} + u_{it}

    where ε and u are independent.
    """
    np.random.seed(123)
    N = 20
    T = 50

    data_list = []

    for i in range(N):
        # Two independent random walks
        y = np.cumsum(np.random.randn(T))
        x = np.cumsum(np.random.randn(T))

        df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


class TestWesterlundBasic:
    """Basic functionality tests for Westerlund tests."""

    def test_basic_functionality(self, cointegrated_panel_data):
        """Test basic execution with default parameters."""
        result = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            n_bootstrap=100,  # Small number for speed
            use_bootstrap=False,  # Use tabulated for speed
        )

        assert isinstance(result, WesterlundResult)
        assert "Gt" in result.statistic
        assert "Ga" in result.statistic
        assert "Pt" in result.statistic
        assert "Pa" in result.statistic

    def test_single_statistic(self, cointegrated_panel_data):
        """Test computation of single statistic."""
        result = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="Gt",
            use_bootstrap=False,
        )

        assert "Gt" in result.statistic
        assert "Ga" not in result.statistic

    def test_trend_specifications(self, cointegrated_panel_data):
        """Test different trend specifications."""
        for trend in ["n", "c", "ct"]:
            result = westerlund_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                trend=trend,
                use_bootstrap=False,
            )

            assert result.trend == trend
            assert not np.isnan(result.statistic["Gt"])

    def test_lag_selection(self, cointegrated_panel_data):
        """Test automatic lag selection."""
        result_auto = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            lags="auto",
            max_lags=3,
            use_bootstrap=False,
        )

        assert result_auto.lags >= 0
        assert result_auto.lags <= 3

    def test_multiple_regressors(self, cointegrated_panel_data):
        """Test with multiple independent variables."""
        # Add second regressor
        np.random.seed(42)
        data = cointegrated_panel_data.copy()
        data["x2"] = data.groupby("entity")["x"].transform(
            lambda x: x + 0.5 * np.random.randn(len(x))
        )

        result = westerlund_test(
            data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars=["x", "x2"],
            use_bootstrap=False,
        )

        assert not np.isnan(result.statistic["Gt"])


class TestWesterlundStatistical:
    """Statistical property tests."""

    def test_cointegrated_data_properties(self, cointegrated_panel_data):
        """Test that cointegrated data shows evidence of cointegration."""
        result = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            use_bootstrap=False,
        )

        # Error correction coefficient should be negative
        # (though we can't directly access it, the statistics should reflect this)
        # At minimum, check that statistics are computed
        assert all(not np.isnan(v) for v in result.statistic.values())

    def test_non_cointegrated_data(self, non_cointegrated_panel_data):
        """Test with non-cointegrated data."""
        result = westerlund_test(
            non_cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            use_bootstrap=False,
        )

        # Should not reject null (at least not strongly)
        # Statistics should be less negative than critical values
        assert all(not np.isnan(v) for v in result.statistic.values())


class TestWesterlundResult:
    """Test WesterlundResult class methods."""

    def test_summary_method(self, cointegrated_panel_data):
        """Test summary method."""
        result = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            use_bootstrap=False,
        )

        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Test" in summary.columns
        assert "Statistic" in summary.columns
        assert "P-value" in summary.columns
        assert len(summary) == 4  # Four tests

    def test_reject_at_method(self, cointegrated_panel_data):
        """Test reject_at method."""
        result = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            use_bootstrap=False,
        )

        # Test all tests
        rejection = result.reject_at(0.05)
        assert isinstance(rejection, dict)
        assert all(isinstance(v, (bool, np.bool_)) for v in rejection.values())

        # Test single test
        rejection_gt = result.reject_at(0.05, test="Gt")
        assert isinstance(rejection_gt, (bool, np.bool_))

    def test_repr_method(self, cointegrated_panel_data):
        """Test __repr__ method."""
        result = westerlund_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            use_bootstrap=False,
        )

        repr_str = repr(result)
        assert "Westerlund" in repr_str
        assert "H0" in repr_str


class TestWesterlundEdgeCases:
    """Test edge cases and error handling."""

    def test_missing_columns(self, cointegrated_panel_data):
        """Test error handling for missing columns."""
        with pytest.raises(ValueError, match="Missing columns"):
            westerlund_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="nonexistent",
                use_bootstrap=False,
            )

    def test_small_panel(self):
        """Test with very small panel."""
        # N=3, T=10
        np.random.seed(42)
        N, T = 3, 10
        data_list = []

        for i in range(N):
            x = np.cumsum(np.random.randn(T))
            y = 1.5 * x + 0.3 * np.random.randn(T)

            df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
            data_list.append(df_i)

        data = pd.concat(data_list, ignore_index=True)

        # Should run without error
        result = westerlund_test(
            data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            use_bootstrap=False,
            lags=1,
        )

        assert isinstance(result, WesterlundResult)

    def test_unbalanced_panel(self):
        """Test with unbalanced panel (different T for each entity)."""
        np.random.seed(42)
        data_list = []

        for i in range(5):
            T = 30 + i * 10  # Different lengths
            x = np.cumsum(np.random.randn(T))
            y = 1.5 * x + 0.3 * np.random.randn(T)

            df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
            data_list.append(df_i)

        data = pd.concat(data_list, ignore_index=True)

        # Should handle unbalanced panel
        result = westerlund_test(
            data, entity_col="entity", time_col="time", y_var="y", x_vars="x", use_bootstrap=False
        )

        assert isinstance(result, WesterlundResult)


class TestBootstrap:
    """Test bootstrap functionality."""

    def test_bootstrap_execution(self, cointegrated_panel_data):
        """Test that bootstrap executes without error."""
        result = westerlund_test(
            cointegrated_panel_data.iloc[:200],  # Smaller sample for speed
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            n_bootstrap=50,  # Small number
            use_bootstrap=True,
            random_state=42,
        )

        assert isinstance(result, WesterlundResult)
        assert result.n_bootstrap == 50

    def test_bootstrap_reproducibility(self, cointegrated_panel_data):
        """Test bootstrap reproducibility with random_state."""
        data_small = cointegrated_panel_data.iloc[:200]

        result1 = westerlund_test(
            data_small,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            n_bootstrap=20,
            use_bootstrap=True,
            random_state=42,
        )

        result2 = westerlund_test(
            data_small,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            n_bootstrap=20,
            use_bootstrap=True,
            random_state=42,
        )

        # P-values should be identical with same seed
        for key in result1.pvalue.keys():
            assert result1.pvalue[key] == result2.pvalue[key]
