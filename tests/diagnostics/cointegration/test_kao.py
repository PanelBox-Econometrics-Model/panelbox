"""
Tests for Kao (1999) cointegration tests.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.cointegration import KaoResult, kao_test


@pytest.fixture
def cointegrated_panel_data():
    """Generate cointegrated panel data with homogeneous cointegrating vector."""
    np.random.seed(789)
    N = 10
    T = 80
    beta = 2.0  # Homogeneous across entities

    data_list = []

    for i in range(N):
        u = np.random.randn(T)
        x = np.cumsum(u)
        epsilon = 0.5 * np.random.randn(T)
        y = beta * x + epsilon  # Same beta for all entities

        df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def non_cointegrated_panel_data():
    """Generate non-cointegrated panel data."""
    np.random.seed(321)
    N = 10
    T = 80

    data_list = []

    for i in range(N):
        y = np.cumsum(np.random.randn(T))
        x = np.cumsum(np.random.randn(T))

        df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


class TestKaoBasic:
    """Basic functionality tests."""

    def test_basic_execution_df(self, cointegrated_panel_data):
        """Test basic execution with DF test."""
        result = kao_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="df",
        )

        assert isinstance(result, KaoResult)
        assert "DF" in result.statistic
        assert not np.isnan(result.statistic["DF"])

    def test_basic_execution_adf(self, cointegrated_panel_data):
        """Test basic execution with ADF test."""
        result = kao_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="adf",
            lags=2,
        )

        assert isinstance(result, KaoResult)
        assert "ADF" in result.statistic
        assert not np.isnan(result.statistic["ADF"])

    def test_all_methods(self, cointegrated_panel_data):
        """Test with method='all'."""
        result = kao_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="all",
            lags=1,
        )

        assert "DF" in result.statistic
        assert "ADF" in result.statistic
        assert not np.isnan(result.statistic["DF"])
        assert not np.isnan(result.statistic["ADF"])

    def test_trend_specifications(self, cointegrated_panel_data):
        """Test different trend specifications."""
        for trend in ["c", "ct"]:
            result = kao_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                trend=trend,
            )

            assert result.trend == trend
            assert not np.isnan(result.statistic["ADF"])

    def test_different_lags(self, cointegrated_panel_data):
        """Test different lag specifications."""
        for lags in [1, 2, 3]:
            result = kao_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                method="adf",
                lags=lags,
            )

            assert result.lags == lags
            assert not np.isnan(result.statistic["ADF"])

    def test_multiple_regressors(self, cointegrated_panel_data):
        """Test with multiple regressors (homogeneous cointegration)."""
        data = cointegrated_panel_data.copy()
        np.random.seed(42)

        # Add second regressor that's cointegrated with first
        data["x2"] = data.groupby("entity")["x"].transform(
            lambda x: x * 0.5 + 0.2 * np.random.randn(len(x))
        )

        result = kao_test(data, entity_col="entity", time_col="time", y_var="y", x_vars=["x", "x2"])

        assert not np.isnan(result.statistic["ADF"])


class TestKaoStatistical:
    """Statistical property tests."""

    def test_cointegrated_data(self, cointegrated_panel_data):
        """Test with cointegrated data."""
        result = kao_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        # Statistics should be computed
        assert all(not np.isnan(v) for v in result.statistic.values())

        # P-values should exist
        assert all(not np.isnan(v) for v in result.pvalue.values())

    def test_non_cointegrated_data(self, non_cointegrated_panel_data):
        """Test with non-cointegrated data."""
        result = kao_test(
            non_cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        # Should not strongly reject null
        assert all(not np.isnan(v) for v in result.statistic.values())


class TestKaoResult:
    """Test KaoResult class methods."""

    def test_summary_method(self, cointegrated_panel_data):
        """Test summary method."""
        result = kao_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="all",
        )

        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert "Test" in summary.columns
        assert "Statistic" in summary.columns
        assert "P-value" in summary.columns
        assert len(summary) == 2  # DF and ADF

    def test_reject_at_method(self, cointegrated_panel_data):
        """Test reject_at method."""
        result = kao_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="all",
        )

        # All tests
        rejection = result.reject_at(0.05)
        assert isinstance(rejection, dict)

        # Single test
        rejection_adf = result.reject_at(0.05, test="ADF")
        assert isinstance(rejection_adf, (bool, np.bool_))

    def test_repr_method(self, cointegrated_panel_data):
        """Test __repr__ method."""
        result = kao_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        repr_str = repr(result)
        assert "Kao" in repr_str
        assert "H0" in repr_str
        assert "homogeneous" in repr_str


class TestKaoEdgeCases:
    """Edge cases and error handling."""

    def test_missing_columns(self, cointegrated_panel_data):
        """Test error with missing columns."""
        with pytest.raises(ValueError, match="Missing columns"):
            kao_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="nonexistent",
            )

    def test_small_panel(self):
        """Test with small panel."""
        np.random.seed(42)
        N, T = 5, 25
        data_list = []

        for i in range(N):
            x = np.cumsum(np.random.randn(T))
            y = 2.0 * x + 0.5 * np.random.randn(T)

            df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
            data_list.append(df_i)

        data = pd.concat(data_list, ignore_index=True)

        result = kao_test(data, entity_col="entity", time_col="time", y_var="y", x_vars="x", lags=1)

        assert isinstance(result, KaoResult)

    def test_critical_values_presence(self, cointegrated_panel_data):
        """Test that critical values are provided."""
        result = kao_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="all",
        )

        for test in result.statistic.keys():
            assert test in result.critical_values
            assert "1%" in result.critical_values[test]
            assert "5%" in result.critical_values[test]
            assert "10%" in result.critical_values[test]


class TestKaoVsPedroni:
    """Compare Kao (homogeneous) vs Pedroni (heterogeneous) assumptions."""

    def test_homogeneous_vs_heterogeneous(self):
        """
        Kao assumes homogeneous cointegrating vector, Pedroni allows heterogeneity.
        Test with homogeneous data - both should work.
        """
        from panelbox.diagnostics.cointegration import pedroni_test

        np.random.seed(42)
        N, T = 10, 60
        beta = 1.5  # Homogeneous

        data_list = []
        for i in range(N):
            x = np.cumsum(np.random.randn(T))
            y = beta * x + 0.3 * np.random.randn(T)

            df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
            data_list.append(df_i)

        data = pd.concat(data_list, ignore_index=True)

        # Kao test (assumes homogeneity)
        kao_result = kao_test(data, entity_col="entity", time_col="time", y_var="y", x_vars="x")

        # Pedroni test (allows heterogeneity)
        pedroni_result = pedroni_test(
            data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        # Both should produce valid results
        assert not np.isnan(kao_result.statistic["ADF"])
        assert not np.isnan(pedroni_result.statistic["panel_ADF"])
