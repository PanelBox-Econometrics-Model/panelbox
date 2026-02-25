"""
Tests for Pedroni (1999, 2004) cointegration tests.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.cointegration import PedroniResult, pedroni_test
from panelbox.validation.cointegration.pedroni import PedroniTest, PedroniTestResult


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

        for test in result.statistic:
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


# ---------------------------------------------------------------------------
# Tests for panelbox.validation.cointegration.pedroni.PedroniTest class
# ---------------------------------------------------------------------------


@pytest.fixture
def coint_panel():
    """Generate cointegrated panel data for PedroniTest (validation module)."""
    np.random.seed(42)
    N = 10
    T = 50
    beta = 1.5

    rows = []
    for i in range(N):
        x = np.cumsum(np.random.randn(T))
        eps = 0.3 * np.random.randn(T)
        y = beta * x + eps
        for t in range(T):
            rows.append({"entity": i, "time": t, "y": y[t], "x": x[t]})

    return pd.DataFrame(rows)


@pytest.fixture
def non_coint_panel():
    """Generate non-cointegrated panel data for PedroniTest."""
    np.random.seed(456)
    N = 10
    T = 50

    rows = []
    for i in range(N):
        y = np.cumsum(np.random.randn(T))
        x = np.cumsum(np.random.randn(T))
        for t in range(T):
            rows.append({"entity": i, "time": t, "y": y[t], "x": x[t]})

    return pd.DataFrame(rows)


class TestPedroniTestClass:
    """Tests for the PedroniTest class in validation.cointegration.pedroni."""

    def test_basic_run(self, coint_panel):
        """Test basic run() with constant trend."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
            trend="c",
        )
        result = test.run()

        assert isinstance(result, PedroniTestResult)
        assert np.isfinite(result.panel_v)
        assert np.isfinite(result.panel_rho)
        assert np.isfinite(result.panel_pp)
        assert np.isfinite(result.panel_adf)
        assert np.isfinite(result.group_rho)
        assert np.isfinite(result.group_pp)
        assert np.isfinite(result.group_adf)
        assert result.n_entities == 10
        assert result.n_obs > 0
        assert result.trend == "Constant"

    def test_run_with_trend_ct(self, coint_panel):
        """Test run() with constant + trend."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
            trend="ct",
        )
        result = test.run()

        assert isinstance(result, PedroniTestResult)
        assert result.trend == "Constant and Trend"
        assert np.isfinite(result.panel_pp)

    def test_run_with_lags(self, coint_panel):
        """Test run() with explicit lags parameter."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
            lags=3,
        )
        result = test.run()

        assert isinstance(result, PedroniTestResult)
        assert np.isfinite(result.panel_adf)

    def test_run_multiple_independents(self, coint_panel):
        """Test run() with multiple independent variables."""
        data = coint_panel.copy()
        np.random.seed(99)
        data["x2"] = data["x"] * 0.5 + np.random.randn(len(data)) * 0.2
        test = PedroniTest(
            data=data,
            dependent="y",
            independents=["x", "x2"],
            entity_col="entity",
            time_col="time",
        )
        result = test.run()

        assert isinstance(result, PedroniTestResult)
        assert np.isfinite(result.group_pp)

    def test_pvalues_present(self, coint_panel):
        """Test that p-values are computed for all statistics."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
        )
        result = test.run()

        expected_keys = [
            "panel_v",
            "panel_rho",
            "panel_pp",
            "panel_adf",
            "group_rho",
            "group_pp",
            "group_adf",
        ]
        for key in expected_keys:
            assert key in result.pvalues, f"Missing p-value for {key}"
            assert 0.0 <= result.pvalues[key] <= 1.0

    def test_non_cointegrated_data(self, non_coint_panel):
        """Test with non-cointegrated data."""
        test = PedroniTest(
            data=non_coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
        )
        result = test.run()

        assert isinstance(result, PedroniTestResult)
        assert np.isfinite(result.panel_rho)

    def test_result_stored(self, coint_panel):
        """Test that result is stored in the test object."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
        )
        assert test.result is None
        result = test.run()
        assert test.result is result


class TestPedroniTestValidation:
    """Test input validation for PedroniTest."""

    def test_missing_dependent(self, coint_panel):
        """Test error when dependent variable is missing."""
        with pytest.raises(ValueError, match="not found"):
            PedroniTest(
                data=coint_panel,
                dependent="nonexistent",
                independents=["x"],
                entity_col="entity",
                time_col="time",
            )

    def test_missing_independent(self, coint_panel):
        """Test error when independent variable is missing."""
        with pytest.raises(ValueError, match="not found"):
            PedroniTest(
                data=coint_panel,
                dependent="y",
                independents=["nonexistent"],
                entity_col="entity",
                time_col="time",
            )

    def test_missing_entity_col(self, coint_panel):
        """Test error when entity column is missing."""
        with pytest.raises(ValueError, match="not found"):
            PedroniTest(
                data=coint_panel,
                dependent="y",
                independents=["x"],
                entity_col="bad_entity",
                time_col="time",
            )

    def test_missing_time_col(self, coint_panel):
        """Test error when time column is missing."""
        with pytest.raises(ValueError, match="not found"):
            PedroniTest(
                data=coint_panel,
                dependent="y",
                independents=["x"],
                entity_col="entity",
                time_col="bad_time",
            )

    def test_invalid_trend(self, coint_panel):
        """Test error for invalid trend specification."""
        with pytest.raises(ValueError, match="trend must be"):
            PedroniTest(
                data=coint_panel,
                dependent="y",
                independents=["x"],
                entity_col="entity",
                time_col="time",
                trend="invalid",
            )

    def test_string_independent_converted_to_list(self, coint_panel):
        """Test that a single string independent is converted to list."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents="x",
            entity_col="entity",
            time_col="time",
        )
        assert test.independents == ["x"]


class TestPedroniTestResultDisplay:
    """Test PedroniTestResult display methods."""

    def test_str_representation(self, coint_panel):
        """Test __str__ method of PedroniTestResult."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
        )
        result = test.run()

        result_str = str(result)
        assert "Pedroni Panel Cointegration Tests" in result_str
        assert "Panel v-statistic:" in result_str
        assert "Panel rho-statistic:" in result_str
        assert "Panel PP-statistic:" in result_str
        assert "Panel ADF-statistic:" in result_str
        assert "Group rho-statistic:" in result_str
        assert "Group PP-statistic:" in result_str
        assert "Group ADF-statistic:" in result_str
        assert "H0: No cointegration" in result_str
        assert "Conclusion:" in result_str

    def test_summary_conclusion_reject(self):
        """Test summary_conclusion when majority of tests reject H0."""
        result = PedroniTestResult(
            panel_v=5.0,
            panel_rho=-3.0,
            panel_pp=-4.0,
            panel_adf=-3.5,
            group_rho=-2.5,
            group_pp=-3.0,
            group_adf=-2.8,
            pvalues={
                "panel_v": 0.01,
                "panel_rho": 0.02,
                "panel_pp": 0.01,
                "panel_adf": 0.03,
                "group_rho": 0.04,
                "group_pp": 0.01,
                "group_adf": 0.02,
            },
            n_obs=500,
            n_entities=10,
            trend="Constant",
        )
        assert "Reject H0" in result.summary_conclusion
        assert "Evidence of cointegration" in result.summary_conclusion

    def test_summary_conclusion_fail_to_reject(self):
        """Test summary_conclusion when majority of tests fail to reject H0."""
        result = PedroniTestResult(
            panel_v=0.5,
            panel_rho=-0.3,
            panel_pp=-0.2,
            panel_adf=-0.4,
            group_rho=-0.1,
            group_pp=-0.3,
            group_adf=-0.2,
            pvalues={
                "panel_v": 0.60,
                "panel_rho": 0.70,
                "panel_pp": 0.80,
                "panel_adf": 0.50,
                "group_rho": 0.65,
                "group_pp": 0.55,
                "group_adf": 0.45,
            },
            n_obs=500,
            n_entities=10,
            trend="Constant",
        )
        assert "Fail to reject H0" in result.summary_conclusion
        assert "No evidence of cointegration" in result.summary_conclusion

    def test_str_with_trend_ct(self, coint_panel):
        """Test __str__ with constant and trend specification."""
        test = PedroniTest(
            data=coint_panel,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
            trend="ct",
        )
        result = test.run()
        result_str = str(result)
        assert "Constant and Trend" in result_str


class TestPedroniTestEdgeCases:
    """Edge cases for PedroniTest."""

    def test_short_entity_series(self):
        """Test behavior with very short entity time series."""
        np.random.seed(42)
        rows = []
        # Create entities with only 2 observations (< 3 should be skipped)
        for i in range(5):
            x = np.cumsum(np.random.randn(2))
            y = 1.5 * x + np.random.randn(2) * 0.1
            for t in range(2):
                rows.append({"entity": i, "time": t, "y": y[t], "x": x[t]})
        # Also add entities with enough data
        for i in range(5, 15):
            x = np.cumsum(np.random.randn(30))
            y = 1.5 * x + np.random.randn(30) * 0.1
            for t in range(30):
                rows.append({"entity": i, "time": t, "y": y[t], "x": x[t]})

        data = pd.DataFrame(rows)
        test = PedroniTest(
            data=data,
            dependent="y",
            independents=["x"],
            entity_col="entity",
            time_col="time",
        )
        result = test.run()
        assert isinstance(result, PedroniTestResult)
        assert np.isfinite(result.panel_pp)
