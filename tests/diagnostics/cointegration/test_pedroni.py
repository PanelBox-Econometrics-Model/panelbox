"""
Tests for Pedroni (1999, 2004) cointegration tests.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.cointegration import PedroniResult, pedroni_test


@pytest.fixture
def cointegrated_panel_data():
    """Generate cointegrated panel data."""
    np.random.seed(42)
    N = 15
    T = 60
    beta = 1.2

    data_list = []

    for i in range(N):
        u = np.random.randn(T)
        x = np.cumsum(u)
        epsilon = 0.4 * np.random.randn(T)
        y = beta * x + epsilon

        df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


@pytest.fixture
def non_cointegrated_panel_data():
    """Generate non-cointegrated panel data."""
    np.random.seed(456)
    N = 15
    T = 60

    data_list = []

    for i in range(N):
        y = np.cumsum(np.random.randn(T))
        x = np.cumsum(np.random.randn(T))

        df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
        data_list.append(df_i)

    return pd.concat(data_list, ignore_index=True)


class TestPedroniBasic:
    """Basic functionality tests."""

    def test_basic_execution(self, cointegrated_panel_data):
        """Test basic execution with all tests."""
        result = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        assert isinstance(result, PedroniResult)

        # Check all 7 statistics are present
        expected_tests = [
            "panel_v",
            "panel_rho",
            "panel_PP",
            "panel_ADF",
            "group_rho",
            "group_PP",
            "group_ADF",
        ]

        for test in expected_tests:
            assert test in result.statistic
            assert not np.isnan(result.statistic[test])

    def test_single_statistic(self, cointegrated_panel_data):
        """Test single statistic computation."""
        result = pedroni_test(
            cointegrated_panel_data,
            entity_col="entity",
            time_col="time",
            y_var="y",
            x_vars="x",
            method="panel_PP",
        )

        assert "panel_PP" in result.statistic
        assert "panel_v" not in result.statistic

    def test_panel_statistics(self, cointegrated_panel_data):
        """Test panel (within-dimension) statistics."""
        panel_tests = ["panel_v", "panel_rho", "panel_PP", "panel_ADF"]

        for test in panel_tests:
            result = pedroni_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                method=test,
            )

            assert test in result.statistic
            assert not np.isnan(result.statistic[test])

    def test_group_statistics(self, cointegrated_panel_data):
        """Test group (between-dimension) statistics."""
        group_tests = ["group_rho", "group_PP", "group_ADF"]

        for test in group_tests:
            result = pedroni_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                method=test,
            )

            assert test in result.statistic
            assert not np.isnan(result.statistic[test])

    def test_trend_specifications(self, cointegrated_panel_data):
        """Test different trend specifications."""
        for trend in ["c", "ct"]:
            result = pedroni_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                trend=trend,
            )

            assert result.trend == trend
            assert not np.isnan(result.statistic["panel_rho"])

    def test_lag_specification(self, cointegrated_panel_data):
        """Test different lag specifications."""
        for lags in [2, 4, 6]:
            result = pedroni_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="x",
                lags=lags,
            )

            assert result.lags == lags

    def test_multiple_regressors(self, cointegrated_panel_data):
        """Test with multiple regressors."""
        data = cointegrated_panel_data.copy()
        np.random.seed(42)
        data["x2"] = data.groupby("entity")["x"].transform(
            lambda x: x + 0.3 * np.random.randn(len(x))
        )

        result = pedroni_test(
            data, entity_col="entity", time_col="time", y_var="y", x_vars=["x", "x2"]
        )

        assert not np.isnan(result.statistic["panel_rho"])


class TestPedroniStatistical:
    """Statistical property tests."""

    def test_cointegrated_vs_non_cointegrated(
        self, cointegrated_panel_data, non_cointegrated_panel_data
    ):
        """Compare results for cointegrated vs non-cointegrated data."""
        result_coint = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        result_no_coint = pedroni_test(
            non_cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        # Cointegrated data should generally have more negative rho statistics
        # (though this is not guaranteed in finite samples)
        assert all(not np.isnan(v) for v in result_coint.statistic.values())
        assert all(not np.isnan(v) for v in result_no_coint.statistic.values())


class TestPedroniResult:
    """Test PedroniResult class methods."""

    def test_summary_method(self, cointegrated_panel_data):
        """Test summary method."""
        result = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        summary = result.summary()

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) == 7  # Seven tests
        assert "Test" in summary.columns
        assert "Statistic" in summary.columns
        assert "P-value" in summary.columns

    def test_reject_at_method(self, cointegrated_panel_data):
        """Test reject_at method."""
        result = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        # All tests
        rejection = result.reject_at(0.05)
        assert isinstance(rejection, dict)
        assert len(rejection) == 7

        # Single test
        rejection_single = result.reject_at(0.05, test="panel_rho")
        assert isinstance(rejection_single, (bool, np.bool_))

    def test_repr_method(self, cointegrated_panel_data):
        """Test __repr__ method."""
        result = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        repr_str = repr(result)
        assert "Pedroni" in repr_str
        assert "H0" in repr_str


class TestPedroniEdgeCases:
    """Edge cases and error handling."""

    def test_missing_columns(self, cointegrated_panel_data):
        """Test error with missing columns."""
        with pytest.raises(ValueError, match="Missing columns"):
            pedroni_test(
                cointegrated_panel_data,
                entity_col="entity",
                time_col="time",
                y_var="y",
                x_vars="missing_var",
            )

    def test_small_panel(self):
        """Test with small panel."""
        np.random.seed(42)
        N, T = 5, 20
        data_list = []

        for i in range(N):
            x = np.cumsum(np.random.randn(T))
            y = 1.5 * x + 0.3 * np.random.randn(T)

            df_i = pd.DataFrame({"entity": i, "time": range(T), "y": y, "x": x})
            data_list.append(df_i)

        data = pd.concat(data_list, ignore_index=True)

        result = pedroni_test(
            data, entity_col="entity", time_col="time", y_var="y", x_vars="x", lags=2
        )

        assert isinstance(result, PedroniResult)

    def test_critical_values_presence(self, cointegrated_panel_data):
        """Test that critical values are provided."""
        result = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        for test in result.statistic.keys():
            assert test in result.critical_values
            assert "1%" in result.critical_values[test]
            assert "5%" in result.critical_values[test]
            assert "10%" in result.critical_values[test]


class TestPedroniComparison:
    """Test comparison between different Pedroni tests."""

    def test_panel_vs_group_tests(self, cointegrated_panel_data):
        """Compare panel and group versions of tests."""
        result = pedroni_test(
            cointegrated_panel_data, entity_col="entity", time_col="time", y_var="y", x_vars="x"
        )

        # Both panel and group rho should exist
        assert "panel_rho" in result.statistic
        assert "group_rho" in result.statistic

        # They should be different (not identical)
        assert result.statistic["panel_rho"] != result.statistic["group_rho"]
