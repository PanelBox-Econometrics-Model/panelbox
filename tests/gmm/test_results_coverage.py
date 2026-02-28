"""
Coverage tests for panelbox.gmm.results module.

Targets uncovered lines: 112-115, 397, 408, 413, 426-428, 432-434, 446, 448,
454-457, 462, 501, 510, 526, 535->534, 573, 588->590, 616, 634, 643, 648,
692-693, 719->721.
"""

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm.results import GMMResults, TestResult

# ---------------------------------------------------------------------------
# Helpers to build GMMResults without running a real estimation
# ---------------------------------------------------------------------------


def _make_test_result(name="Hansen J-test", statistic=10.0, pvalue=0.18, df=5, conclusion="PASS"):
    return TestResult(
        name=name,
        statistic=statistic,
        pvalue=pvalue,
        df=df,
        conclusion=conclusion,
    )


def _make_gmm_results(
    param_names=None,
    param_values=None,
    se_values=None,
    pvalue_values=None,
    dep_var="y",
    exog_vars=None,
    id_var="id",
    time_var="time",
    n_lags=1,
    endogenous_vars=None,
    predetermined_vars=None,
    model_type="difference",
    transformation="fd",
    two_step=True,
    windmeijer_corrected=False,
    diff_hansen=None,
    nobs=100,
    n_groups=50,
    n_instruments=20,
):
    """Build a GMMResults object with sensible defaults."""
    if param_names is None:
        param_names = ["L1.y", "x1"]
    if param_values is None:
        param_values = [0.5] * len(param_names)
    if se_values is None:
        se_values = [0.1] * len(param_names)
    if pvalue_values is None:
        pvalue_values = [0.001] * len(param_names)
    if exog_vars is None:
        exog_vars = ["x1"]
    if endogenous_vars is None:
        endogenous_vars = []
    if predetermined_vars is None:
        predetermined_vars = []

    params = pd.Series(param_values, index=param_names)
    std_errors = pd.Series(se_values, index=param_names)
    tvalues = params / std_errors
    pvalues = pd.Series(pvalue_values, index=param_names)

    hansen = _make_test_result("Hansen J-test", 10.0, 0.18, 5)
    sargan = _make_test_result("Sargan test", 8.0, 0.25, 5)
    ar1 = _make_test_result("AR(1) test", -2.5, 0.01, None, "EXPECTED")
    ar2 = _make_test_result("AR(2) test", 1.2, 0.23, None)

    return GMMResults(
        params=params,
        std_errors=std_errors,
        tvalues=tvalues,
        pvalues=pvalues,
        nobs=nobs,
        n_groups=n_groups,
        n_instruments=n_instruments,
        n_params=len(param_names),
        hansen_j=hansen,
        sargan=sargan,
        ar1_test=ar1,
        ar2_test=ar2,
        diff_hansen=diff_hansen,
        vcov=np.eye(len(param_names)) * 0.01,
        converged=True,
        two_step=two_step,
        windmeijer_corrected=windmeijer_corrected,
        model_type=model_type,
        transformation=transformation,
        dep_var=dep_var,
        exog_vars=exog_vars,
        id_var=id_var,
        time_var=time_var,
        n_lags=n_lags,
        endogenous_vars=endogenous_vars,
        predetermined_vars=predetermined_vars,
    )


# ---------------------------------------------------------------------------
# Tests for TestResult._determine_conclusion generic branch (lines 112-115)
# ---------------------------------------------------------------------------


class TestTestResultGenericConclusion:
    """Cover the generic fallback in _determine_conclusion (lines 112-115)."""

    def test_generic_reject_below_005(self):
        """Generic test name with pvalue < 0.05 should return REJECT."""
        result = TestResult(
            name="Wald test",
            statistic=5.0,
            pvalue=0.03,
        )
        assert result.conclusion == "REJECT"

    def test_generic_pass_above_005(self):
        """Generic test name with pvalue >= 0.05 should return PASS."""
        result = TestResult(
            name="Wald test",
            statistic=2.0,
            pvalue=0.15,
        )
        assert result.conclusion == "PASS"

    def test_generic_boundary_exactly_005(self):
        """Generic test name with pvalue == 0.05 should return PASS (not < 0.05)."""
        result = TestResult(
            name="Custom test",
            statistic=3.0,
            pvalue=0.05,
        )
        assert result.conclusion == "PASS"


# ---------------------------------------------------------------------------
# Tests for predict() method
# ---------------------------------------------------------------------------


class TestPredictBranches:
    """Cover all uncovered branches in GMMResults.predict()."""

    def test_predict_raises_when_dep_var_empty(self):
        """Line 397: predict() raises ValueError when dep_var is empty."""
        results = _make_gmm_results(dep_var="")
        df = pd.DataFrame({"x1": [1, 2]})
        with pytest.raises(ValueError, match="Cannot predict"):
            results.predict(df)

    def test_predict_lag_column_found_directly(self):
        """Line 408: predict() uses pre-computed lag column from new_data."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "L1.y": [1.0, 2.0, 1.5, 2.5],
                "x1": [0.1, 0.2, 0.3, 0.4],
            }
        )
        preds = results.predict(df)
        expected = 0.5 * df["L1.y"].values + 0.3 * df["x1"].values
        np.testing.assert_allclose(preds, expected)

    def test_predict_lag_column_computed_via_shift(self):
        """Line 413: predict() computes lag via groupby shift when dep_var present."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1, 1, 2, 2, 2],
                "time": [1, 2, 3, 1, 2, 3],
                "y": [10.0, 20.0, 30.0, 5.0, 15.0, 25.0],
                "x1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            }
        )
        preds = results.predict(df)
        lagged = df.groupby("id")["y"].shift(1).values
        expected = 0.5 * lagged + 0.3 * df["x1"].values
        # First obs per entity will be NaN because lag is NaN
        assert np.isnan(preds[0])
        assert np.isnan(preds[3])
        np.testing.assert_allclose(preds[1], expected[1])
        np.testing.assert_allclose(preds[2], expected[2])

    def test_predict_lag_column_missing_raises(self):
        """Lines 413 fallthrough: predict() raises when neither lag col nor dep_var found."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "x1": [0.1, 0.2],
                # No "y" and no "L1.y" column
            }
        )
        with pytest.raises(ValueError, match="Cannot find lag"):
            results.predict(df)

    def test_predict_endogenous_variable_not_found(self):
        """Lines 426-428: predict() raises when endogenous variable missing."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "endo1"],
            param_values=[0.5, 0.3, 0.2],
            endogenous_vars=["endo1"],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
                # Missing "endo1"
            }
        )
        with pytest.raises(ValueError, match="Endogenous variable 'endo1' not found"):
            results.predict(df)

    def test_predict_predetermined_variable_not_found(self):
        """Lines 432-434: predict() raises when predetermined variable missing."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "pre1"],
            param_values=[0.5, 0.3, 0.1],
            predetermined_vars=["pre1"],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
                # Missing "pre1"
            }
        )
        with pytest.raises(ValueError, match="Predetermined variable 'pre1' not found"):
            results.predict(df)

    def test_predict_cons_parameter(self):
        """Line 446: predict() adds ones column for _cons parameter."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "_cons"],
            param_values=[0.5, 0.3, 1.0],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
            }
        )
        preds = results.predict(df)
        expected = 0.5 * np.array([1.0, 2.0]) + 0.3 * np.array([0.1, 0.2]) + 1.0
        np.testing.assert_allclose(preds, expected)

    def test_predict_param_in_new_data_columns(self):
        """Line 448: predict() picks up extra parameter directly from new_data columns."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "extra_col"],
            param_values=[0.5, 0.3, 0.7],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
                "extra_col": [3.0, 4.0],
            }
        )
        preds = results.predict(df)
        expected = (
            0.5 * np.array([1.0, 2.0]) + 0.3 * np.array([0.1, 0.2]) + 0.7 * np.array([3.0, 4.0])
        )
        np.testing.assert_allclose(preds, expected)

    def test_predict_year_time_dummy_reconstruction(self):
        """Lines 454-457: predict() reconstructs year_ dummies from time_var."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "year_2020", "year_2021"],
            param_values=[0.5, 0.3, 0.1, 0.2],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [2019, 2020, 2021],
                "L1.y": [1.0, 2.0, 3.0],
                "x1": [0.1, 0.2, 0.3],
            }
        )
        preds = results.predict(df)
        # For row 0 (time=2019): year_2020=0, year_2021=0
        # For row 1 (time=2020): year_2020=1, year_2021=0
        # For row 2 (time=2021): year_2020=0, year_2021=1
        expected_0 = 0.5 * 1.0 + 0.3 * 0.1 + 0.1 * 0 + 0.2 * 0
        expected_1 = 0.5 * 2.0 + 0.3 * 0.2 + 0.1 * 1 + 0.2 * 0
        expected_2 = 0.5 * 3.0 + 0.3 * 0.3 + 0.1 * 0 + 0.2 * 1
        np.testing.assert_allclose(preds[0], expected_0)
        np.testing.assert_allclose(preds[1], expected_1)
        np.testing.assert_allclose(preds[2], expected_2)

    def test_predict_year_dummy_fallback_on_valueerror(self):
        """Lines 454-457: predict() falls back to zeros on ValueError for year_ parsing."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "year_abc"],
            param_values=[0.5, 0.3, 0.9],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [2020, 2021],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
            }
        )
        preds = results.predict(df)
        # year_abc cannot be parsed to int, fallback to zeros
        expected = 0.5 * np.array([1.0, 2.0]) + 0.3 * np.array([0.1, 0.2])
        np.testing.assert_allclose(preds, expected)

    def test_predict_unknown_remaining_param_zeros(self):
        """Lines 456-457: predict() uses zeros for unrecognized remaining params."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "unknown_param"],
            param_values=[0.5, 0.3, 999.0],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
            }
        )
        preds = results.predict(df)
        # unknown_param is filled with zeros, so its coefficient has no effect
        expected = 0.5 * np.array([1.0, 2.0]) + 0.3 * np.array([0.1, 0.2])
        np.testing.assert_allclose(preds, expected)

    def test_predict_dimension_mismatch(self):
        """Line 462: predict() raises ValueError on dimension mismatch."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "x2"],
            param_values=[0.5, 0.3, 0.2],
            exog_vars=["x1", "x2"],
        )
        # Monkey-patch params to have an extra param that won't be matched
        # to trigger the dimension mismatch. We add a 4th param manually.
        results.params = pd.Series(
            [0.5, 0.3, 0.2, 0.1],
            index=["L1.y", "x1", "x2", "x3"],
        )
        results.n_params = 4
        # exog_vars only has x1, x2. x3 is not in remaining_params logic
        # because n_structural = 1 (n_lags) + 2 (exog) = 3
        # remaining_params = ["x3"] which won't be found -> zeros column
        # X will have 4 columns: L1.y, x1, x2, zeros_for_x3
        # and params also has 4, so no mismatch here.
        # Instead, force the mismatch by making exog_vars shorter.
        results.exog_vars = ["x1"]
        # n_structural = 1 + 1 = 2, remaining = ["x2", "x3"]
        # x2 is in new_data, x3 is not -> zeros
        # X has 4 cols, params has 4 -> still matches.
        # We need to actually trigger the mismatch. Let's add more params.
        results.params = pd.Series(
            [0.5, 0.3, 0.2, 0.1, 0.05],
            index=["L1.y", "x1", "x2", "x3", "x4"],
        )
        results.exog_vars = ["x1"]
        results.endogenous_vars = ["x2"]
        # n_structural = 1 + 1 + 1 + 0 = 3
        # remaining = ["x3", "x4"]
        # Both x3, x4 not in data and not year_ and not _cons -> zeros
        # X has 5 cols, params has 5. Still matches.
        # To actually trigger mismatch, we need the X to have different cols.
        # The only way is if n_lags loop adds different number of columns.
        # Let's set n_lags=2 but params only has 1 lag name.
        results2 = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
            n_lags=2,  # Will try to add 2 lag columns
        )
        # This means X will have 3 columns (L1.y shift 1, L1.y shift 2, x1)
        # but params only has 2 entries -> mismatch
        df = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [1, 2, 3],
                "y": [1.0, 2.0, 3.0],
                "x1": [0.1, 0.2, 0.3],
            }
        )
        with pytest.raises(ValueError, match="Feature matrix has"):
            results2.predict(df)

    def test_predict_with_endogenous_and_predetermined(self):
        """Successful predict() with endogenous and predetermined variables."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "endo1", "pre1"],
            param_values=[0.5, 0.3, 0.2, 0.1],
            endogenous_vars=["endo1"],
            predetermined_vars=["pre1"],
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [1, 2],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
                "endo1": [0.5, 0.6],
                "pre1": [0.3, 0.4],
            }
        )
        preds = results.predict(df)
        expected = (
            0.5 * np.array([1.0, 2.0])
            + 0.3 * np.array([0.1, 0.2])
            + 0.2 * np.array([0.5, 0.6])
            + 0.1 * np.array([0.3, 0.4])
        )
        np.testing.assert_allclose(preds, expected)


# ---------------------------------------------------------------------------
# Tests for forecast() method
# ---------------------------------------------------------------------------


class TestForecastBranches:
    """Cover all uncovered branches in GMMResults.forecast()."""

    def test_forecast_raises_when_dep_var_empty(self):
        """Line 501: forecast() raises ValueError when dep_var is empty."""
        results = _make_gmm_results(dep_var="")
        with pytest.raises(ValueError, match="Cannot forecast"):
            results.forecast(
                last_obs={1: [10.0]},
                future_exog=pd.DataFrame({"id": [1], "time": [3], "x1": [0.5]}),
                steps=1,
            )

    def test_forecast_ar_coefficient_not_found(self):
        """Line 510: forecast() raises ValueError when AR coefficient not found."""
        results = _make_gmm_results(
            param_names=["wrong_name", "x1"],
            param_values=[0.5, 0.3],
            dep_var="y",
            n_lags=1,
        )
        future = pd.DataFrame({"id": [1], "time": [3], "x1": [0.5]})
        with pytest.raises(ValueError, match=r"AR coefficient 'L1\.y' not found"):
            results.forecast(last_obs={1: [10.0]}, future_exog=future, steps=1)

    def test_forecast_entity_not_enough_future_periods(self):
        """Line 526: forecast() raises when entity doesn't have enough future periods."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        future = pd.DataFrame(
            {
                "id": [1],
                "time": [3],
                "x1": [0.5],
            }
        )
        # Ask for 3 steps but only 1 future period available
        with pytest.raises(ValueError, match="expected 3 future periods, got 1"):
            results.forecast(last_obs={1: [10.0]}, future_exog=future, steps=3)

    def test_forecast_basic_one_step(self):
        """Lines 534-535: forecast() single step with AR(1) model."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        future = pd.DataFrame(
            {
                "id": [1],
                "time": [3],
                "x1": [1.0],
            }
        )
        fc = results.forecast(last_obs={1: [10.0]}, future_exog=future, steps=1)
        # y_hat = 0.5 * 10.0 + 0.3 * 1.0 = 5.3
        assert len(fc) == 1
        np.testing.assert_allclose(fc["forecast"].values[0], 5.3)

    def test_forecast_multi_step(self):
        """Lines 534-535: forecast() multi-step feeds predictions back."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        future = pd.DataFrame(
            {
                "id": [1, 1, 1],
                "time": [3, 4, 5],
                "x1": [1.0, 1.0, 1.0],
            }
        )
        fc = results.forecast(last_obs={1: [10.0]}, future_exog=future, steps=3)
        assert len(fc) == 3

        # Step 1: y_hat = 0.5 * 10.0 + 0.3 * 1.0 = 5.3
        y1 = 5.3
        np.testing.assert_allclose(fc.iloc[0]["forecast"], y1)

        # Step 2: y_hat = 0.5 * 5.3 + 0.3 * 1.0 = 2.95
        y2 = 0.5 * y1 + 0.3
        np.testing.assert_allclose(fc.iloc[1]["forecast"], y2)

        # Step 3: y_hat = 0.5 * 2.95 + 0.3 * 1.0 = 1.775
        y3 = 0.5 * y2 + 0.3
        np.testing.assert_allclose(fc.iloc[2]["forecast"], y3)

    def test_forecast_multiple_entities(self):
        """Forecast for multiple entities."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
        )
        future = pd.DataFrame(
            {
                "id": [1, 1, 2, 2],
                "time": [3, 4, 3, 4],
                "x1": [1.0, 1.0, 2.0, 2.0],
            }
        )
        fc = results.forecast(
            last_obs={1: [10.0], 2: [20.0]},
            future_exog=future,
            steps=2,
        )
        assert len(fc) == 4
        # Entity 1, step 1: 0.5 * 10 + 0.3 * 1.0 = 5.3
        assert fc[(fc["id"] == 1)].iloc[0]["forecast"] == pytest.approx(5.3)
        # Entity 2, step 1: 0.5 * 20 + 0.3 * 2.0 = 10.6
        assert fc[(fc["id"] == 2)].iloc[0]["forecast"] == pytest.approx(10.6)

    def test_forecast_ar2_model(self):
        """Forecast with AR(2) model uses two lags."""
        results = _make_gmm_results(
            param_names=["L1.y", "L2.y", "x1"],
            param_values=[0.5, 0.2, 0.3],
            n_lags=2,
        )
        future = pd.DataFrame(
            {
                "id": [1],
                "time": [5],
                "x1": [1.0],
            }
        )
        fc = results.forecast(
            last_obs={1: [8.0, 10.0]},  # y_{T-1}=8, y_T=10
            future_exog=future,
            steps=1,
        )
        # y_hat = 0.5 * 10.0 + 0.2 * 8.0 + 0.3 * 1.0 = 5.0 + 1.6 + 0.3 = 6.9
        np.testing.assert_allclose(fc["forecast"].values[0], 6.9)


# ---------------------------------------------------------------------------
# Tests for summary() method
# ---------------------------------------------------------------------------


class TestSummaryBranches:
    """Cover all uncovered branches in GMMResults.summary()."""

    def test_summary_fod_transformation(self):
        """Line 573: summary() appends '(FOD)' when transformation='fod'."""
        results = _make_gmm_results(
            model_type="difference",
            transformation="fod",
        )
        summary = results.summary()
        assert "(FOD)" in summary

    def test_summary_windmeijer_corrected(self):
        """Lines 588-590: summary() shows 'Windmeijer' in GMM type line."""
        results = _make_gmm_results(
            two_step=True,
            windmeijer_corrected=True,
        )
        summary = results.summary()
        assert "Windmeijer" in summary
        assert "Standard errors: Windmeijer (2005) corrected" in summary

    def test_summary_not_windmeijer_corrected(self):
        """Line 648: summary() shows 'Robust' when not windmeijer corrected."""
        results = _make_gmm_results(
            two_step=True,
            windmeijer_corrected=False,
        )
        summary = results.summary()
        assert "Standard errors: Robust" in summary

    def test_summary_system_model_type(self):
        """Line 634: summary() shows diff_hansen for system model."""
        diff_h = _make_test_result("Difference-in-Hansen", 5.0, 0.30, 3, "PASS")
        results = _make_gmm_results(
            model_type="system",
            transformation="fd",
            diff_hansen=diff_h,
        )
        summary = results.summary()
        assert "System GMM" in summary
        assert "Difference-in-Hansen" in summary

    def test_summary_non_difference_no_fd_line(self):
        """Line 634/643: summary() for system model shows neither
        'First-differences' nor 'Forward Orthogonal Deviations' for fd transformation."""
        results = _make_gmm_results(
            model_type="system",
            transformation="fd",
        )
        summary = results.summary()
        # model_type != 'difference', so "First-differences" line is skipped
        # transformation != 'fod' either, so FOD line is also skipped
        assert "Transformation: First-differences" not in summary
        assert "Transformation: Forward Orthogonal Deviations" not in summary

    def test_summary_system_fod_transformation(self):
        """Line 643: summary() for system model with fod shows FOD transformation."""
        results = _make_gmm_results(
            model_type="system",
            transformation="fod",
        )
        summary = results.summary()
        assert "Transformation: Forward Orthogonal Deviations" in summary

    def test_summary_significance_star_single(self):
        """Line 616: summary() shows single star for 0.01 <= p < 0.05."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
            pvalue_values=[0.03, 0.001],  # one star and three stars
        )
        summary = results.summary()
        # The summary should contain significance stars
        assert "*" in summary

    def test_summary_one_step(self):
        """Lines 588->590: summary() shows 'One-step' when two_step=False."""
        results = _make_gmm_results(
            two_step=False,
            windmeijer_corrected=False,
        )
        summary = results.summary()
        assert "One-step" in summary


# ---------------------------------------------------------------------------
# Tests for to_latex() method
# ---------------------------------------------------------------------------


class TestToLatexBranches:
    """Cover all uncovered branches in GMMResults.to_latex()."""

    def test_to_latex_significance_stars_triple(self):
        """Lines 692-693: to_latex() shows *** for p < 0.001."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1"],
            param_values=[0.5, 0.3],
            pvalue_values=[0.0001, 0.03],  # *** and *
        )
        latex = results.to_latex()
        assert r"^{***}" in latex
        assert r"^{*}" in latex

    def test_to_latex_significance_star_single(self):
        """Lines 692-693: to_latex() shows * for 0.01 <= p < 0.05."""
        results = _make_gmm_results(
            param_names=["L1.y"],
            param_values=[0.5],
            se_values=[0.1],
            pvalue_values=[0.04],
            exog_vars=[],
        )
        latex = results.to_latex()
        assert r"^{*}" in latex
        # Should NOT have ** or ***
        assert r"^{***}" not in latex
        assert r"^{**}" not in latex

    def test_to_latex_windmeijer_corrected(self):
        """Lines 719-721: to_latex() appends Windmeijer info."""
        results = _make_gmm_results(
            two_step=True,
            windmeijer_corrected=True,
        )
        latex = results.to_latex()
        assert "Windmeijer corrected" in latex

    def test_to_latex_not_windmeijer(self):
        """Lines 719->721: to_latex() without Windmeijer just shows GMM type."""
        results = _make_gmm_results(
            two_step=True,
            windmeijer_corrected=False,
        )
        latex = results.to_latex()
        assert "Two-step" in latex
        assert "Windmeijer" not in latex

    def test_to_latex_one_step(self):
        """Lines 719->721: to_latex() shows 'One-step' when two_step=False."""
        results = _make_gmm_results(
            two_step=False,
            windmeijer_corrected=False,
        )
        latex = results.to_latex()
        assert "One-step" in latex


# ---------------------------------------------------------------------------
# Additional edge cases for completeness
# ---------------------------------------------------------------------------


class TestPredictTimeDummyKeyError:
    """Cover the KeyError path in year_ dummy reconstruction (line 454)."""

    def test_predict_year_dummy_keyerror_fallback(self):
        """Line 454: predict() falls back to zeros when time_var not in new_data."""
        results = _make_gmm_results(
            param_names=["L1.y", "x1", "year_2020"],
            param_values=[0.5, 0.3, 0.9],
            time_var="nonexistent_time_col",
        )
        df = pd.DataFrame(
            {
                "id": [1, 1],
                "time": [2020, 2021],
                "L1.y": [1.0, 2.0],
                "x1": [0.1, 0.2],
            }
        )
        # time_var="nonexistent_time_col" is not in df -> KeyError -> zeros
        preds = results.predict(df)
        expected = 0.5 * np.array([1.0, 2.0]) + 0.3 * np.array([0.1, 0.2])
        np.testing.assert_allclose(preds, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
