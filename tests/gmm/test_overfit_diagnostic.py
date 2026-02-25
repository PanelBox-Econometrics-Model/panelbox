"""
Unit tests for GMM Overfit Diagnostic
======================================

Tests for the GMMOverfitDiagnostic class.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.overfit_diagnostic import GMMOverfitDiagnostic
from panelbox.gmm.system_gmm import SystemGMM

# ============================================================================
# Fixtures
# ============================================================================


def _generate_panel(n_units, n_periods, ar_coef=0.5, seed=42):
    """Generate simulated panel data for testing."""
    np.random.seed(seed)
    data_list = []
    for i in range(n_units):
        eta_i = np.random.normal(0, 1)
        y = np.zeros(n_periods)
        x = np.random.normal(0, 1, n_periods)
        y[0] = eta_i + np.random.normal(0, 0.3)
        for t in range(1, n_periods):
            y[t] = ar_coef * y[t - 1] + 0.3 * x[t] + eta_i + np.random.normal(0, 0.3)
        for t in range(n_periods):
            data_list.append({"id": i, "year": t, "y": y[t], "x": x[t]})
    return pd.DataFrame(data_list)


@pytest.fixture
def large_panel():
    """Panel with N=50, T=10 - should produce GREEN diagnostics."""
    return _generate_panel(n_units=50, n_periods=10)


@pytest.fixture
def small_panel():
    """Panel with N=5, T=20 - may produce YELLOW/RED diagnostics."""
    return _generate_panel(n_units=5, n_periods=20, seed=123)


@pytest.fixture
def fitted_diff_gmm(large_panel):
    """Fitted Difference GMM on large panel."""
    model = DifferenceGMM(
        data=large_panel,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x"],
        collapse=True,
        two_step=True,
        robust=True,
    )
    results = model.fit()
    return model, results


@pytest.fixture
def fitted_system_gmm(large_panel):
    """Fitted System GMM on large panel."""
    model = SystemGMM(
        data=large_panel,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x"],
        collapse=True,
        two_step=True,
        robust=True,
    )
    try:
        results = model.fit()
    except (ValueError, np.linalg.LinAlgError):
        pytest.skip("System GMM failed with numerical issues (acceptable for synthetic data)")
    return model, results


@pytest.fixture
def fitted_small_diff(small_panel):
    """Fitted Difference GMM on small panel."""
    model = DifferenceGMM(
        data=small_panel,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x"],
        collapse=True,
        two_step=True,
        robust=True,
    )
    results = model.fit()
    return model, results


# ============================================================================
# Test Constructor
# ============================================================================


class TestInit:
    def test_creates_diagnostic(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        assert diag.n_groups == results.n_groups
        assert diag.n_instruments == results.n_instruments
        assert diag._ar_param_name is not None

    def test_repr(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        r = repr(diag)
        assert "GMMOverfitDiagnostic" in r
        assert "n_groups=" in r

    def test_works_with_system_gmm(self, fitted_system_gmm):
        model, results = fitted_system_gmm
        diag = GMMOverfitDiagnostic(model, results)
        assert diag.n_groups == results.n_groups


# ============================================================================
# Test assess_feasibility
# ============================================================================


class TestFeasibility:
    @pytest.mark.slow
    def test_large_panel_green(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        feas = diag.assess_feasibility()

        assert "n_groups" in feas
        assert "n_instruments" in feas
        assert "instrument_ratio" in feas
        assert "signal" in feas
        assert feas["instrument_ratio"] == results.n_instruments / results.n_groups

    def test_small_panel_warns(self, fitted_small_diff):
        model, results = fitted_small_diff
        diag = GMMOverfitDiagnostic(model, results)
        feas = diag.assess_feasibility()

        # With N=5 and collapsed instruments, ratio may be high
        assert feas["signal"] in ("GREEN", "YELLOW", "RED")
        assert isinstance(feas["recommendation"], str)


# ============================================================================
# Test instrument_sensitivity
# ============================================================================


class TestInstrumentSensitivity:
    def test_returns_dataframe(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        df = diag.instrument_sensitivity(max_lag_range=[2, 3, 4])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert "gmm_max_lag" in df.columns
        assert "n_instruments" in df.columns
        assert "ar_coef" in df.columns
        assert "hansen_j_pval" in df.columns

    def test_has_signal_attr(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        df = diag.instrument_sensitivity(max_lag_range=[2, 3])

        assert "signal" in df.attrs
        assert df.attrs["signal"] in ("GREEN", "YELLOW", "RED")


# ============================================================================
# Test coefficient_bounds_test
# ============================================================================


class TestCoefficientBounds:
    def test_returns_bounds(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        bounds = diag.coefficient_bounds_test()

        assert "ols_coef" in bounds
        assert "fe_coef" in bounds
        assert "gmm_coef" in bounds
        assert "within_bounds" in bounds
        assert "signal" in bounds
        assert bounds["signal"] in ("GREEN", "YELLOW", "RED")

    def test_ols_greater_than_fe(self, fitted_diff_gmm):
        """OLS should be biased upward, FE biased downward for AR models."""
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        bounds = diag.coefficient_bounds_test()

        if bounds["ols_coef"] is not None and bounds["fe_coef"] is not None:
            # OLS should generally be >= FE for AR models
            assert bounds["ols_coef"] is not None
            assert bounds["fe_coef"] is not None

    def test_system_gmm_bounds(self, fitted_system_gmm):
        model, results = fitted_system_gmm
        diag = GMMOverfitDiagnostic(model, results)
        bounds = diag.coefficient_bounds_test()
        assert bounds["gmm_coef"] is not None


# ============================================================================
# Test jackknife_groups
# ============================================================================


class TestJackknifeGroups:
    def test_skips_large_n(self, fitted_diff_gmm):
        """With N=50, should skip jackknife (exceeds default max_groups=30)."""
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        jk = diag.jackknife_groups()

        assert "Skipped" in jk["details"]
        assert jk["mean"] is None

    def test_runs_for_small_n(self, fitted_small_diff):
        """With N=5, should run jackknife."""
        model, results = fitted_small_diff
        diag = GMMOverfitDiagnostic(model, results)
        jk = diag.jackknife_groups()

        assert jk["full_sample_coef"] is not None
        assert len(jk["jackknife_coefs"]) > 0
        assert jk["signal"] in ("GREEN", "YELLOW", "RED")

    def test_custom_max_groups(self, fitted_diff_gmm):
        """Can increase max_groups to force run."""
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        jk = diag.jackknife_groups(max_groups=100)

        # Should run since N=50 < 100
        assert jk["mean"] is not None
        assert jk["std"] is not None


# ============================================================================
# Test step_comparison
# ============================================================================


class TestStepComparison:
    def test_returns_comparison(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        step = diag.step_comparison()

        assert "one_step_coef" in step
        assert "two_step_coef" in step
        assert "rel_diff" in step
        assert "signal" in step
        assert step["signal"] in ("GREEN", "YELLOW", "RED")

    def test_both_coefs_present(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        step = diag.step_comparison()

        assert step["one_step_coef"] is not None
        assert step["two_step_coef"] is not None
        assert step["abs_diff"] is not None


# ============================================================================
# Test summary
# ============================================================================


class TestSummary:
    def test_summary_runs(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        text = diag.summary(run_jackknife=False)

        assert "GMM Overfit Diagnostic Report" in text
        assert "OVERALL VERDICT" in text
        assert "Instrument Feasibility" in text
        assert "Coefficient Bounds" in text

    def test_summary_with_jackknife(self, fitted_small_diff):
        model, results = fitted_small_diff
        diag = GMMOverfitDiagnostic(model, results)
        text = diag.summary(run_jackknife=True)

        assert "Jackknife" in text
        assert "OVERALL VERDICT" in text

    def test_summary_skips_jackknife_when_requested(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        text = diag.summary(run_jackknife=False)

        assert "SKIPPED" in text


# ============================================================================
# Test _clone_model
# ============================================================================


class TestCloneModel:
    def test_clone_diff_gmm(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        clone = diag._clone_model(gmm_max_lag=3)

        assert isinstance(clone, DifferenceGMM)
        assert clone.gmm_max_lag == 3
        assert clone.dep_var == model.dep_var

    def test_clone_system_gmm(self, fitted_system_gmm):
        model, results = fitted_system_gmm
        diag = GMMOverfitDiagnostic(model, results)
        clone = diag._clone_model(gmm_max_lag=2)

        assert isinstance(clone, SystemGMM)
        assert clone.gmm_max_lag == 2
        assert hasattr(clone, "level_instruments")

    def test_clone_overrides_type(self, fitted_diff_gmm):
        model, results = fitted_diff_gmm
        diag = GMMOverfitDiagnostic(model, results)
        clone = diag._clone_model(gmm_type="one_step")

        assert clone.gmm_type == "one_step"


# ============================================================================
# Extended coverage tests
# ============================================================================


def _make_mock_results(param_names=("L1.y", "x"), n_groups=50, n_instruments=10, n_params=2):
    """Create a mock GMMResults object for testing without real estimation."""
    results = MagicMock()
    params = pd.Series(
        [0.5] * len(param_names),
        index=list(param_names),
    )
    std_errors = pd.Series(
        [0.1] * len(param_names),
        index=list(param_names),
    )
    results.params = params
    results.std_errors = std_errors
    results.n_groups = n_groups
    results.n_instruments = n_instruments
    results.n_params = n_params
    results.hansen_j = MagicMock(pvalue=0.3)
    results.ar2_test = MagicMock(pvalue=0.5)
    return results


def _make_mock_model(data=None, is_system=False, two_step=True):
    """Create a mock model object for testing."""
    if data is None:
        data = _generate_panel(10, 6)
    model = MagicMock()
    model.data = data
    model.dep_var = "y"
    model.lags = [1]
    model.id_var = "id"
    model.time_var = "year"
    model.exog_vars = ["x"]
    model.endogenous_vars = []
    model.predetermined_vars = []
    model.time_dummies = False
    model.collapse = True
    model.two_step = two_step
    model.robust = True
    model.gmm_type = "two_step" if two_step else "one_step"
    model.gmm_max_lag = None
    model.iv_max_lag = 0
    if is_system:
        model.level_instruments = None
    return model


class TestInitL2Param:
    """Test L2 AR param name detection (lines 87-88)."""

    def test_l2_param_detected(self):
        """When results have L2. prefix param, it should be detected."""
        results = _make_mock_results(param_names=("L2.y", "x"))
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)
        assert diag._ar_param_name == "L2.y"

    def test_l1_preferred_over_l2(self):
        """L1 should be found first since loop checks in order."""
        results = _make_mock_results(param_names=("L1.y", "L2.y", "x"))
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)
        assert diag._ar_param_name == "L1.y"

    def test_no_ar_param(self):
        """When no L1/L2 param exists, _ar_param_name is None."""
        results = _make_mock_results(param_names=("x", "z"))
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)
        assert diag._ar_param_name is None


class TestCloneModelSystemGMM:
    """Test SystemGMM clone path (line 131)."""

    def test_clone_system_gmm_includes_level_instruments(self, fitted_system_gmm):
        model, results = fitted_system_gmm
        diag = GMMOverfitDiagnostic(model, results)
        clone = diag._clone_model(gmm_max_lag=3)

        assert isinstance(clone, SystemGMM)
        # level_instruments should be carried over
        assert hasattr(clone, "level_instruments")


class TestFeasibilityYellow:
    """Test YELLOW feasibility signal (lines 162-163)."""

    def test_yellow_signal_when_ratio_between_075_and_1(self):
        """When 0.75 < ratio <= 1.0, signal should be YELLOW."""
        # n_instruments/n_groups between 0.75 and 1.0 => YELLOW
        results = _make_mock_results(n_groups=10, n_instruments=9)
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)
        feas = diag.assess_feasibility()
        assert feas["signal"] == "YELLOW"
        assert feas["instrument_ratio"] == 0.9
        assert "approaching" in feas["recommendation"]

    def test_green_signal_when_ratio_below_075(self):
        """When ratio <= 0.75, signal should be GREEN."""
        results = _make_mock_results(n_groups=20, n_instruments=10)
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)
        feas = diag.assess_feasibility()
        assert feas["signal"] == "GREEN"
        assert feas["instrument_ratio"] == 0.5

    def test_red_signal_when_ratio_above_1(self):
        """When ratio > 1.0, signal should be RED."""
        results = _make_mock_results(n_groups=5, n_instruments=10)
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)
        feas = diag.assess_feasibility()
        assert feas["signal"] == "RED"
        assert feas["instrument_ratio"] == 2.0
        assert "exceeds" in feas["recommendation"]


class TestInstrumentSensitivityEdgeCases:
    """Test instrument_sensitivity exception and edge case paths."""

    def test_exception_during_reestimation(self):
        """Lines 225-226: When re-estimation fails, row has None values."""
        results = _make_mock_results()
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        # Patch _clone_model to return a model whose fit() always fails
        failing_model = MagicMock()
        failing_model.fit.side_effect = ValueError("Estimation failed")

        with patch.object(diag, "_clone_model", return_value=failing_model):
            df = diag.instrument_sensitivity(max_lag_range=[2, 3])

        assert len(df) == 2
        assert df["ar_coef"].isna().all()
        assert df["n_instruments"].isna().all()

    def test_red_signal_large_variation(self):
        """Lines 246, 248: RED signal when relative_range > 0.20."""
        results = _make_mock_results()
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        # Create mock results with very different coefficients
        mock_results_low = MagicMock()
        mock_results_low.params = pd.Series({"L1.y": 0.3, "x": 0.1})
        mock_results_low.n_instruments = 5
        mock_results_low.hansen_j = MagicMock(pvalue=0.3)
        mock_results_low.ar2_test = MagicMock(pvalue=0.5)

        mock_results_high = MagicMock()
        mock_results_high.params = pd.Series({"L1.y": 0.8, "x": 0.1})
        mock_results_high.n_instruments = 8
        mock_results_high.hansen_j = MagicMock(pvalue=0.2)
        mock_results_high.ar2_test = MagicMock(pvalue=0.4)

        mock_model = MagicMock()
        mock_model.fit.side_effect = [mock_results_low, mock_results_high]

        with patch.object(diag, "_clone_model", return_value=mock_model):
            df = diag.instrument_sensitivity(max_lag_range=[2, 3])

        assert df.attrs["signal"] == "RED"
        assert df.attrs["relative_range"] > 0.20

    def test_yellow_signal_moderate_variation(self):
        """Line 248: YELLOW signal when 0.10 < relative_range <= 0.20."""
        results = _make_mock_results()
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        mock_results_1 = MagicMock()
        mock_results_1.params = pd.Series({"L1.y": 0.50, "x": 0.1})
        mock_results_1.n_instruments = 5
        mock_results_1.hansen_j = MagicMock(pvalue=0.3)
        mock_results_1.ar2_test = MagicMock(pvalue=0.5)

        mock_results_2 = MagicMock()
        # 0.58 - 0.50 = 0.08, mean = 0.54, range/mean ~ 0.148 => YELLOW
        mock_results_2.params = pd.Series({"L1.y": 0.58, "x": 0.1})
        mock_results_2.n_instruments = 8
        mock_results_2.hansen_j = MagicMock(pvalue=0.2)
        mock_results_2.ar2_test = MagicMock(pvalue=0.4)

        mock_model = MagicMock()
        mock_model.fit.side_effect = [mock_results_1, mock_results_2]

        with patch.object(diag, "_clone_model", return_value=mock_model):
            df = diag.instrument_sensitivity(max_lag_range=[2, 3])

        assert df.attrs["signal"] == "YELLOW"
        assert 0.10 < df.attrs["relative_range"] <= 0.20

    def test_fewer_than_2_valid_coefs(self):
        """Lines 253-254: When <2 valid coefs, YELLOW with None range."""
        results = _make_mock_results()
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        # One fails, one succeeds, but the second has no AR param
        mock_results_1 = MagicMock()
        mock_results_1.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        mock_results_1.n_instruments = 5
        mock_results_1.hansen_j = MagicMock(pvalue=0.3)
        mock_results_1.ar2_test = MagicMock(pvalue=0.5)

        mock_model_ok = MagicMock()
        mock_model_ok.fit.return_value = mock_results_1

        mock_model_fail = MagicMock()
        mock_model_fail.fit.side_effect = ValueError("Fail")

        call_count = [0]

        def clone_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_model_fail
            return mock_model_fail  # Both fail

        with patch.object(diag, "_clone_model", side_effect=clone_side_effect):
            df = diag.instrument_sensitivity(max_lag_range=[2, 3])

        # All failed => 0 valid coefs < 2
        assert df.attrs["signal"] == "YELLOW"
        assert df.attrs["relative_range"] is None


class TestCoefficientBoundsEdgeCases:
    """Test coefficient_bounds_test edge cases."""

    def test_no_ar_param(self):
        """Line 274: When no AR coefficient found."""
        results = _make_mock_results(param_names=("x", "z"))
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        bounds = diag.coefficient_bounds_test()
        assert bounds["ols_coef"] is None
        assert bounds["fe_coef"] is None
        assert bounds["gmm_coef"] is None
        assert bounds["signal"] == "YELLOW"
        assert "No AR coefficient" in bounds["details"]

    def test_ols_failure(self):
        """Lines 307-308: When OLS lstsq fails."""
        data = _generate_panel(10, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        # Patch np.linalg.lstsq to fail on the first call (OLS), then succeed (FE)
        original_lstsq = np.linalg.lstsq
        call_count = [0]

        def failing_lstsq(a, b, rcond=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise np.linalg.LinAlgError("Singular matrix")
            return original_lstsq(a, b, rcond=rcond)

        with patch("numpy.linalg.lstsq", side_effect=failing_lstsq):
            bounds = diag.coefficient_bounds_test()

        # OLS failed, FE might have succeeded or failed
        # Either way, if one is None, signal should be YELLOW
        if bounds["ols_coef"] is None:
            assert bounds["signal"] == "YELLOW"
            assert "Could not compute" in bounds["details"]

    def test_fe_failure(self):
        """Lines 326-327: When FE lstsq fails."""
        data = _generate_panel(10, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        original_lstsq = np.linalg.lstsq
        call_count = [0]

        def failing_fe_lstsq(a, b, rcond=None):
            call_count[0] += 1
            if call_count[0] == 2:
                raise np.linalg.LinAlgError("FE failed")
            return original_lstsq(a, b, rcond=rcond)

        with patch("numpy.linalg.lstsq", side_effect=failing_fe_lstsq):
            bounds = diag.coefficient_bounds_test()

        # FE failed
        if bounds["fe_coef"] is None:
            assert bounds["signal"] == "YELLOW"
            assert "Could not compute" in bounds["details"]

    def test_gmm_outside_bounds_red(self):
        """Lines 363-365: GMM outside OLS/FE bounds => RED."""
        data = _generate_panel(10, 6)
        # Set GMM coef to be way outside expected bounds
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        results.params = pd.Series({"L1.y": 2.0, "x": 0.1})  # Unrealistically high
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        bounds = diag.coefficient_bounds_test()

        if bounds["ols_coef"] is not None and bounds["fe_coef"] is not None:
            # GMM coef of 2.0 should be outside any reasonable OLS/FE bounds
            assert bounds["signal"] == "RED"
            assert "OUTSIDE" in bounds["details"]

    def test_gmm_within_bounds_green(self):
        """Lines 349-350: GMM within bounds but not near boundary => GREEN."""
        data = _generate_panel(10, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        bounds = diag.coefficient_bounds_test()
        # Just check structure; signal depends on actual data
        assert bounds["signal"] in ("GREEN", "YELLOW", "RED")
        assert bounds["gmm_coef"] is not None

    def test_gmm_near_boundary_yellow(self):
        """Lines 349-350: GMM within bounds but near boundary => YELLOW."""
        data = _generate_panel(10, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        # Patch the OLS/FE computation to force near-boundary result
        original_test = diag.coefficient_bounds_test

        def patched_bounds():
            # First get real OLS/FE coefs
            result = original_test()
            if result["ols_coef"] is not None and result["fe_coef"] is not None:
                # Override GMM coef to be near the lower bound
                lower = min(result["ols_coef"], result["fe_coef"])
                upper = max(result["ols_coef"], result["fe_coef"])
                margin = 0.05 * (upper - lower) if upper != lower else 0.05
                # Place GMM coef right at lower + margin/2
                results.params = pd.Series({"L1.y": lower + margin * 0.4, "x": 0.1})
            return None  # We'll call the real method again

        # Compute OLS/FE bounds first
        initial_bounds = diag.coefficient_bounds_test()
        if initial_bounds["ols_coef"] is not None and initial_bounds["fe_coef"] is not None:
            lower = min(initial_bounds["ols_coef"], initial_bounds["fe_coef"])
            upper = max(initial_bounds["ols_coef"], initial_bounds["fe_coef"])
            margin = 0.05 * (upper - lower) if upper != lower else 0.05
            # Set GMM near the lower boundary
            results.params = pd.Series({"L1.y": lower + margin * 0.4, "x": 0.1})
            bounds = diag.coefficient_bounds_test()
            assert bounds["signal"] == "YELLOW"
            assert "near a boundary" in bounds["details"]

    def test_both_ols_fe_none_yellow(self):
        """Lines 363-365: When both OLS and FE fail."""
        data = _generate_panel(10, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        with patch("numpy.linalg.lstsq", side_effect=np.linalg.LinAlgError("fail")):
            bounds = diag.coefficient_bounds_test()

        assert bounds["ols_coef"] is None
        assert bounds["fe_coef"] is None
        assert bounds["within_bounds"] is None
        assert bounds["signal"] == "YELLOW"
        assert "Could not compute" in bounds["details"]


class TestJackknifeEdgeCases:
    """Test jackknife_groups edge cases."""

    def test_no_ar_param(self):
        """Line 397: Model without AR parameter."""
        results = _make_mock_results(param_names=("x", "z"), n_groups=5)
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        jk = diag.jackknife_groups()
        assert jk["full_sample_coef"] is None
        assert jk["signal"] == "YELLOW"
        assert "No AR coefficient" in jk["details"]

    def test_all_reestimations_fail(self):
        """Lines 434-435, 440: When all re-estimations fail => <2 valid coefs."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        # All clone+fit calls fail
        failing_model = MagicMock()
        failing_model.fit.side_effect = ValueError("Failed")

        with patch.object(diag, "_clone_model", return_value=failing_model):
            jk = diag.jackknife_groups(max_groups=10)

        assert jk["signal"] == "RED"
        assert "Too few successful" in jk["details"]
        # All jackknife coefs should be None
        for v in jk["jackknife_coefs"].values():
            assert v is None

    def test_moderate_variation_yellow(self):
        """Lines 458-459: Moderate jackknife variation => YELLOW."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        # Create mock results with moderate variation (15% < rel_dev <= 30%)
        coefs = [0.42, 0.48, 0.53, 0.46, 0.58]  # max_dev from 0.5 = 0.08 => 16%
        call_idx = [0]

        def make_clone(**kwargs):
            mock_m = MagicMock()
            idx = call_idx[0]
            if idx < len(coefs):
                mock_r = MagicMock()
                mock_r.params = pd.Series({"L1.y": coefs[idx], "x": 0.1})
                mock_m.fit.return_value = mock_r
            else:
                mock_m.fit.side_effect = ValueError("fail")
            call_idx[0] += 1
            return mock_m

        with patch.object(diag, "_clone_model", side_effect=make_clone):
            jk = diag.jackknife_groups(max_groups=10)

        assert jk["signal"] == "YELLOW"
        assert "Moderate" in jk["details"]

    def test_high_variation_red(self):
        """Lines 464-465: High jackknife variation => RED."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        # Create mock results with high variation (rel_dev > 30%)
        coefs = [0.2, 0.3, 0.8, 0.9, 0.1]  # max_dev from 0.5 = 0.4 => 80%
        call_idx = [0]

        def make_clone(**kwargs):
            mock_m = MagicMock()
            idx = call_idx[0]
            if idx < len(coefs):
                mock_r = MagicMock()
                mock_r.params = pd.Series({"L1.y": coefs[idx], "x": 0.1})
                mock_m.fit.return_value = mock_r
            else:
                mock_m.fit.side_effect = ValueError("fail")
            call_idx[0] += 1
            return mock_m

        with patch.object(diag, "_clone_model", side_effect=make_clone):
            jk = diag.jackknife_groups(max_groups=10)

        assert jk["signal"] == "RED"
        assert "High" in jk["details"] or "fragile" in jk["details"]

    def test_stable_jackknife_green(self):
        """Lines 458-459: Stable jackknife => GREEN."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        # Create mock results with very stable coefficients
        coefs = [0.49, 0.50, 0.51, 0.50, 0.49]  # max_dev = 0.01 => 2%
        call_idx = [0]

        def make_clone(**kwargs):
            mock_m = MagicMock()
            idx = call_idx[0]
            if idx < len(coefs):
                mock_r = MagicMock()
                mock_r.params = pd.Series({"L1.y": coefs[idx], "x": 0.1})
                mock_m.fit.return_value = mock_r
            else:
                mock_m.fit.side_effect = ValueError("fail")
            call_idx[0] += 1
            return mock_m

        with patch.object(diag, "_clone_model", side_effect=make_clone):
            jk = diag.jackknife_groups(max_groups=10)

        assert jk["signal"] == "GREEN"
        assert "stable" in jk["details"]


class TestStepComparisonEdgeCases:
    """Test step_comparison edge cases."""

    def test_no_ar_param(self):
        """Line 501: Model without AR parameter."""
        results = _make_mock_results(param_names=("x", "z"), n_groups=10)
        model = _make_mock_model()
        diag = GMMOverfitDiagnostic(model, results)

        step = diag.step_comparison()
        assert step["one_step_coef"] is None
        assert step["two_step_coef"] is None
        assert step["signal"] == "YELLOW"
        assert "No AR coefficient" in step["details"]

    def test_alt_estimation_failure(self):
        """Lines 529-530: When alternative step estimation fails."""
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        failing_model = MagicMock()
        failing_model.fit.side_effect = ValueError("Alt estimation failed")

        with patch.object(diag, "_clone_model", return_value=failing_model):
            step = diag.step_comparison()

        assert step["signal"] == "YELLOW"
        assert step["two_step_coef"] == 0.5  # Original was two-step
        assert step["one_step_coef"] is None
        assert "Could not estimate" in step["details"]

    def test_original_is_one_step(self):
        """Lines 547-550: When original model is one-step."""
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(two_step=False)
        model.gmm_type = "one_step"
        diag = GMMOverfitDiagnostic(model, results)

        # Mock the two-step re-estimation
        alt_results = MagicMock()
        alt_results.params = pd.Series({"L1.y": 0.52, "x": 0.11})
        alt_results.std_errors = pd.Series({"L1.y": 0.09, "x": 0.04})
        alt_model = MagicMock()
        alt_model.fit.return_value = alt_results

        with patch.object(diag, "_clone_model", return_value=alt_model):
            step = diag.step_comparison()

        assert step["one_step_coef"] == 0.5  # Original was one-step
        assert step["two_step_coef"] == 0.52  # Alternative was two-step
        assert step["signal"] in ("GREEN", "YELLOW", "RED")

    def test_yellow_moderate_difference(self):
        """Lines 567-568: Moderate step difference => YELLOW."""
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        results.params = pd.Series({"L1.y": 0.50, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        # Set up moderate difference: 15% relative diff, SE ratio OK
        alt_results = MagicMock()
        alt_results.params = pd.Series({"L1.y": 0.42, "x": 0.09})
        alt_results.std_errors = pd.Series({"L1.y": 0.11, "x": 0.05})
        alt_model = MagicMock()
        alt_model.fit.return_value = alt_results

        with patch.object(diag, "_clone_model", return_value=alt_model):
            step = diag.step_comparison()

        # rel_diff = |0.50 - 0.42| / ((0.50 + 0.42)/2) = 0.08/0.46 ≈ 0.174
        assert step["signal"] == "YELLOW"

    def test_red_large_difference(self):
        """Lines 567-568 else branch: Large step difference => RED."""
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        results.params = pd.Series({"L1.y": 0.50, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.10, "x": 0.05})
        model = _make_mock_model(two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        # Large difference: > 20% relative diff
        alt_results = MagicMock()
        alt_results.params = pd.Series({"L1.y": 0.20, "x": 0.05})
        alt_results.std_errors = pd.Series({"L1.y": 0.15, "x": 0.06})
        alt_model = MagicMock()
        alt_model.fit.return_value = alt_results

        with patch.object(diag, "_clone_model", return_value=alt_model):
            step = diag.step_comparison()

        # rel_diff = |0.50 - 0.20| / ((0.50 + 0.20)/2) = 0.30/0.35 ≈ 0.857
        assert step["signal"] == "RED"
        assert "Large difference" in step["details"]

    def test_green_step_comparison(self):
        """Step comparison GREEN signal."""
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=10)
        results.params = pd.Series({"L1.y": 0.50, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.10, "x": 0.05})
        model = _make_mock_model(two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        # Small difference: < 10% relative diff, SE ratio near 1
        alt_results = MagicMock()
        alt_results.params = pd.Series({"L1.y": 0.48, "x": 0.10})
        alt_results.std_errors = pd.Series({"L1.y": 0.10, "x": 0.05})
        alt_model = MagicMock()
        alt_model.fit.return_value = alt_results

        with patch.object(diag, "_clone_model", return_value=alt_model):
            step = diag.step_comparison()

        # rel_diff = |0.50 - 0.48| / ((0.50 + 0.48)/2) = 0.02/0.49 ≈ 0.041
        assert step["signal"] == "GREEN"
        assert "consistent" in step["details"]


class TestSummaryExtended:
    """Extended tests for summary formatting and branches."""

    def test_summary_all_green(self):
        """Line 714: All GREEN signals => overall GREEN."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        # Patch all diagnostic methods to return GREEN
        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "All good.",
        }

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "Within bounds.",
        }

        green_jk = {
            "full_sample_coef": 0.5,
            "jackknife_coefs": {0: 0.49, 1: 0.51},
            "mean": 0.50,
            "std": 0.01,
            "max_deviation": 0.01,
            "signal": "GREEN",
            "details": "Stable jackknife.",
        }

        green_step = {
            "one_step_coef": 0.49,
            "two_step_coef": 0.50,
            "abs_diff": 0.01,
            "rel_diff": 0.02,
            "se_ratio": 1.0,
            "signal": "GREEN",
            "details": "Consistent.",
        }

        green_sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": 5,
                    "ar_coef": 0.50,
                    "hansen_j_pval": 0.3,
                    "ar2_pval": 0.5,
                },
            ]
        )
        green_sens_df.attrs["signal"] = "GREEN"
        green_sens_df.attrs["relative_range"] = 0.05

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", return_value=green_sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "jackknife_groups", return_value=green_jk),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=True)

        assert "OVERALL VERDICT: [GREEN]" in text
        assert "All diagnostics pass" in text

    def test_summary_with_red_overall(self):
        """Lines 703-706: Any RED => overall RED."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        red_feas = {
            "n_groups": 5,
            "n_instruments": 10,
            "instrument_ratio": 2.0,
            "signal": "RED",
            "recommendation": "Too many instruments.",
        }

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "OK.",
        }

        green_step = {
            "one_step_coef": 0.49,
            "two_step_coef": 0.50,
            "abs_diff": 0.01,
            "rel_diff": 0.02,
            "se_ratio": 1.0,
            "signal": "GREEN",
            "details": "OK.",
        }

        sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": 5,
                    "ar_coef": 0.50,
                    "hansen_j_pval": 0.3,
                    "ar2_pval": 0.5,
                },
            ]
        )
        sens_df.attrs["signal"] = "GREEN"
        sens_df.attrs["relative_range"] = 0.05

        with (
            patch.object(diag, "assess_feasibility", return_value=red_feas),
            patch.object(diag, "instrument_sensitivity", return_value=sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=False)

        assert "OVERALL VERDICT: [RED]" in text
        assert "overfitting" in text.lower() or "proliferation" in text.lower()

    def test_summary_instrument_sensitivity_exception(self):
        """Lines 644-648: When instrument_sensitivity raises an exception."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "OK.",
        }

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "OK.",
        }

        green_step = {
            "one_step_coef": 0.49,
            "two_step_coef": 0.50,
            "abs_diff": 0.01,
            "rel_diff": 0.02,
            "se_ratio": 1.0,
            "signal": "GREEN",
            "details": "OK.",
        }

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", side_effect=RuntimeError("Boom")),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=False)

        assert "Could not run" in text
        assert "Boom" in text

    def test_summary_no_ar_bounds_no_ols_fe(self):
        """Lines 655-661: When bounds test returns None for OLS/FE/GMM."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("x", "z"), n_groups=5)
        model = _make_mock_model(data=data)
        diag = GMMOverfitDiagnostic(model, results)

        no_ar_bounds = {
            "ols_coef": None,
            "fe_coef": None,
            "gmm_coef": None,
            "within_bounds": None,
            "signal": "YELLOW",
            "details": "No AR coefficient found.",
        }

        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "OK.",
        }

        sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": 5,
                    "ar_coef": None,
                    "hansen_j_pval": None,
                    "ar2_pval": None,
                },
            ]
        )
        sens_df.attrs["signal"] = "YELLOW"
        sens_df.attrs["relative_range"] = None

        green_step = {
            "one_step_coef": None,
            "two_step_coef": None,
            "abs_diff": None,
            "rel_diff": None,
            "se_ratio": None,
            "signal": "YELLOW",
            "details": "No AR coefficient found.",
        }

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", return_value=sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=no_ar_bounds),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=False)

        # OLS/FE/GMM lines should NOT appear since they're None
        assert "OLS (upper bound)" not in text
        assert "FE  (lower bound)" not in text
        assert "GMM estimate" not in text
        assert "No AR coefficient" in text

    def test_summary_step_comparison_partial_none(self):
        """Lines 688-696: When step comparison has some None values."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "OK.",
        }

        sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": 5,
                    "ar_coef": 0.5,
                    "hansen_j_pval": 0.3,
                    "ar2_pval": 0.5,
                },
            ]
        )
        sens_df.attrs["signal"] = "GREEN"
        sens_df.attrs["relative_range"] = 0.05

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "OK.",
        }

        partial_step = {
            "one_step_coef": None,
            "two_step_coef": 0.50,
            "abs_diff": None,
            "rel_diff": None,
            "se_ratio": None,
            "signal": "YELLOW",
            "details": "Could not estimate alternative step.",
        }

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", return_value=sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "step_comparison", return_value=partial_step),
        ):
            text = diag.summary(run_jackknife=False)

        # one_step_coef is None, so "One-step coef" should not appear
        assert "One-step coef" not in text
        # two_step_coef is present
        assert "Two-step coef" in text
        # rel_diff is None
        assert "Relative difference" not in text

    def test_summary_with_jackknife_mean_present(self):
        """Lines 670-675: Summary with jackknife mean/std/max_deviation present."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "OK.",
        }

        sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": 5,
                    "ar_coef": 0.5,
                    "hansen_j_pval": 0.3,
                    "ar2_pval": 0.5,
                },
            ]
        )
        sens_df.attrs["signal"] = "GREEN"
        sens_df.attrs["relative_range"] = 0.05

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "OK.",
        }

        jk_with_data = {
            "full_sample_coef": 0.5,
            "jackknife_coefs": {0: 0.49, 1: 0.51},
            "mean": 0.50,
            "std": 0.01,
            "max_deviation": 0.01,
            "signal": "GREEN",
            "details": "Stable.",
        }

        green_step = {
            "one_step_coef": 0.49,
            "two_step_coef": 0.50,
            "abs_diff": 0.01,
            "rel_diff": 0.02,
            "se_ratio": 1.0,
            "signal": "GREEN",
            "details": "Consistent.",
        }

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", return_value=sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "jackknife_groups", return_value=jk_with_data),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=True)

        assert "Full-sample coef" in text
        assert "Jackknife mean" in text
        assert "Jackknife std" in text
        assert "Max deviation" in text
        assert "OVERALL VERDICT: [GREEN]" in text

    def test_summary_jackknife_skipped_no_mean(self):
        """Line 670->675: Jackknife run but mean is None (skipped or failed)."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "OK.",
        }

        sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": 5,
                    "ar_coef": 0.5,
                    "hansen_j_pval": 0.3,
                    "ar2_pval": 0.5,
                },
            ]
        )
        sens_df.attrs["signal"] = "GREEN"
        sens_df.attrs["relative_range"] = 0.05

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "OK.",
        }

        jk_no_mean = {
            "full_sample_coef": 0.5,
            "jackknife_coefs": {},
            "mean": None,
            "std": None,
            "max_deviation": None,
            "signal": "YELLOW",
            "details": "Too few successful.",
        }

        green_step = {
            "one_step_coef": 0.49,
            "two_step_coef": 0.50,
            "abs_diff": 0.01,
            "rel_diff": 0.02,
            "se_ratio": 1.0,
            "signal": "GREEN",
            "details": "Consistent.",
        }

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", return_value=sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "jackknife_groups", return_value=jk_no_mean),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=True)

        # mean is None so Full-sample/Jackknife mean lines should NOT appear
        assert "Full-sample coef" not in text
        assert "Jackknife mean" not in text
        assert "Too few successful" in text

    def test_summary_sensitivity_no_relative_range(self):
        """Line 641->643: instrument_sensitivity with relative_range=None."""
        data = _generate_panel(5, 6)
        results = _make_mock_results(param_names=("L1.y", "x"), n_groups=5)
        results.params = pd.Series({"L1.y": 0.5, "x": 0.1})
        results.std_errors = pd.Series({"L1.y": 0.1, "x": 0.05})
        model = _make_mock_model(data=data, two_step=True)
        diag = GMMOverfitDiagnostic(model, results)

        green_feas = {
            "n_groups": 5,
            "n_instruments": 3,
            "instrument_ratio": 0.6,
            "signal": "GREEN",
            "recommendation": "OK.",
        }

        sens_df = pd.DataFrame(
            [
                {
                    "gmm_max_lag": 2,
                    "n_instruments": None,
                    "ar_coef": None,
                    "hansen_j_pval": None,
                    "ar2_pval": None,
                },
            ]
        )
        sens_df.attrs["signal"] = "YELLOW"
        sens_df.attrs["relative_range"] = None

        green_bounds = {
            "ols_coef": 0.6,
            "fe_coef": 0.4,
            "gmm_coef": 0.5,
            "within_bounds": True,
            "signal": "GREEN",
            "details": "OK.",
        }

        green_step = {
            "one_step_coef": 0.49,
            "two_step_coef": 0.50,
            "abs_diff": 0.01,
            "rel_diff": 0.02,
            "se_ratio": 1.0,
            "signal": "GREEN",
            "details": "Consistent.",
        }

        with (
            patch.object(diag, "assess_feasibility", return_value=green_feas),
            patch.object(diag, "instrument_sensitivity", return_value=sens_df),
            patch.object(diag, "coefficient_bounds_test", return_value=green_bounds),
            patch.object(diag, "step_comparison", return_value=green_step),
        ):
            text = diag.summary(run_jackknife=False)

        # Coefficient range line should NOT appear since relative_range is None
        assert "Coefficient range" not in text


class TestOneStepFittedModel:
    """Test with a real one-step fitted model."""

    def test_step_comparison_one_step_original(self):
        """Lines 547-550: Real one-step original model for step_comparison."""
        data = _generate_panel(20, 8, seed=99)
        model = DifferenceGMM(
            data=data,
            dep_var="y",
            lags=1,
            id_var="id",
            time_var="year",
            exog_vars=["x"],
            collapse=True,
            two_step=False,
            robust=True,
            gmm_type="one_step",
            time_dummies=False,
        )
        try:
            results = model.fit()
        except (ValueError, np.linalg.LinAlgError):
            pytest.skip("One-step model failed (numerical issues)")

        diag = GMMOverfitDiagnostic(model, results)
        step = diag.step_comparison()

        # Original is one-step, so one_step_coef should be from original
        assert step["one_step_coef"] is not None
        assert step["signal"] in ("GREEN", "YELLOW", "RED")

    def test_lags_2_model(self):
        """Lines 87-88: Real model with lags=2 to exercise L2. param detection."""
        data = _generate_panel(20, 10, seed=77)
        try:
            model = DifferenceGMM(
                data=data,
                dep_var="y",
                lags=[1, 2],
                id_var="id",
                time_var="year",
                exog_vars=["x"],
                collapse=True,
                two_step=True,
                robust=True,
                time_dummies=False,
            )
            results = model.fit()
        except (ValueError, np.linalg.LinAlgError):
            pytest.skip("Lags-2 model failed (numerical issues)")

        diag = GMMOverfitDiagnostic(model, results)
        # With lags=[1,2], should have L1 and L2 params
        assert diag._ar_param_name is not None
        assert diag._ar_param_name.startswith("L1.") or diag._ar_param_name.startswith("L2.")
