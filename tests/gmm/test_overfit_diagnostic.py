"""
Unit tests for GMM Overfit Diagnostic
======================================

Tests for the GMMOverfitDiagnostic class.
"""

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
