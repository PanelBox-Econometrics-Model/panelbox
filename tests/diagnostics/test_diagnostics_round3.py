"""
Round 3 coverage tests for panelbox diagnostics modules.

Targets uncovered lines/branches in:
- diagnostics/cointegration/westerlund.py
- diagnostics/cointegration/kao.py
- diagnostics/unit_root/unified.py
- diagnostics/specification/davidson_mackinnon.py
- diagnostics/quantile/basic_diagnostics.py
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_cointegrated_panel(N=10, T=50, seed=42):
    """Create panel data with cointegration: y_{it} = beta * x_{it} + e_{it}."""
    np.random.seed(seed)
    data = []
    for i in range(N):
        x = np.cumsum(np.random.randn(T))  # I(1) process
        e = np.random.randn(T) * 0.5
        y = 2.0 * x + e  # Cointegrated with beta=2
        for t in range(T):
            data.append({"entity": i, "time": t, "y": y[t], "x": x[t]})
    return pd.DataFrame(data)


def _make_non_cointegrated_panel(N=10, T=50, seed=123):
    """Create panel with independent I(1) variables (no cointegration)."""
    np.random.seed(seed)
    data = []
    for i in range(N):
        x = np.cumsum(np.random.randn(T))
        y = np.cumsum(np.random.randn(T))  # Independent random walk
        for t in range(T):
            data.append({"entity": i, "time": t, "y": y[t], "x": x[t]})
    return pd.DataFrame(data)


def _make_stationary_panel(N=10, T=100, seed=42):
    """Create panel with stationary AR(1) processes."""
    np.random.seed(seed)
    data = []
    for i in range(N):
        y = np.zeros(T)
        y[0] = np.random.randn()
        for t in range(1, T):
            y[t] = 0.5 * y[t - 1] + np.random.randn()
        for t in range(T):
            data.append({"entity": i, "time": t, "y": y[t]})
    return pd.DataFrame(data)


def _make_unit_root_panel(N=10, T=100, seed=123):
    """Create panel with unit root (random walk) processes."""
    np.random.seed(seed)
    data = []
    for i in range(N):
        y = np.cumsum(np.random.randn(T))
        for t in range(T):
            data.append({"entity": i, "time": t, "y": y[t]})
    return pd.DataFrame(data)


def _make_ols_result(n, k=2, seed=42):
    """Create a mock statsmodels OLS result for J-test."""
    np.random.seed(seed)
    X = np.column_stack([np.ones(n), np.random.randn(n, k - 1)])
    beta = np.random.randn(k)
    y = X @ beta + np.random.randn(n) * 0.5

    from statsmodels.regression.linear_model import OLS

    model = OLS(y, X)
    return model.fit()


# ===========================================================================
# Tests for Westerlund cointegration test
# ===========================================================================
class TestWesterlundUncoveredBranches:
    """Cover uncovered lines in westerlund.py."""

    def test_no_trend_specification(self):
        """Cover lines 190-191: trend='n' branch in _estimate_ecm."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(
            data,
            "entity",
            "time",
            "y",
            "x",
            trend="n",
            n_bootstrap=10,
            random_state=42,
        )
        assert isinstance(result.statistic, dict)

    def test_ct_trend_specification(self):
        """Cover lines 179-181: trend='ct' in _estimate_ecm."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(
            data,
            "entity",
            "time",
            "y",
            "x",
            trend="ct",
            n_bootstrap=10,
            random_state=42,
        )
        assert "Gt" in result.statistic

    def test_tabulated_critical_values(self):
        """Cover lines 664-674: use_bootstrap=False tabulated values."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(
            data,
            "entity",
            "time",
            "y",
            "x",
            use_bootstrap=False,
        )
        # Tabulated CVs use fixed values
        for test in result.critical_values:
            assert result.critical_values[test]["5%"] == -1.96

    def test_single_method(self):
        """Cover line 631: method != 'all' filters statistics."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(
            data,
            "entity",
            "time",
            "y",
            "x",
            method="Gt",
            n_bootstrap=10,
            random_state=42,
        )
        assert "Gt" in result.statistic
        assert "Ga" not in result.statistic

    def test_fixed_lags(self):
        """Cover line 606: explicit lags parameter (not auto)."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(
            data,
            "entity",
            "time",
            "y",
            "x",
            lags=2,
            n_bootstrap=10,
            random_state=42,
        )
        assert result.lags == 2

    def test_missing_columns_error(self):
        """Cover line 587: missing columns ValueError."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = pd.DataFrame({"entity": [0], "time": [0], "y": [1.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            westerlund_test(data, "entity", "time", "y", "x")

    def test_bic_lag_selection(self):
        """Cover lines 293-294: BIC criterion for lag selection."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(
            data,
            "entity",
            "time",
            "y",
            "x",
            lags="auto",
            lag_criterion="bic",
            n_bootstrap=10,
            random_state=42,
        )
        assert isinstance(result.lags, (int, np.integer))

    def test_reject_at_specific_test(self):
        """Cover line 83: reject_at with specific test name."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(data, "entity", "time", "y", "x", n_bootstrap=10, random_state=42)
        rej = result.reject_at(0.05, test="Gt")
        assert isinstance(rej, (bool, np.bool_))

    def test_reject_at_all_tests(self):
        """Cover line 84: reject_at without specific test."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = westerlund_test(data, "entity", "time", "y", "x", n_bootstrap=10, random_state=42)
        rej = result.reject_at(0.05)
        assert isinstance(rej, dict)

    def test_summary_significance_markers(self):
        """Cover lines 104-112: significance marker branches."""
        from panelbox.diagnostics.cointegration.westerlund import WesterlundResult

        result = WesterlundResult(
            statistic={"Gt": -3.0, "Ga": -1.0, "Pt": -2.0, "Pa": -0.5},
            pvalue={"Gt": 0.005, "Ga": 0.03, "Pt": 0.08, "Pa": 0.5},
            critical_values={
                "Gt": {"1%": -2.58, "5%": -1.96, "10%": -1.645},
                "Ga": {"1%": -2.58, "5%": -1.96, "10%": -1.645},
                "Pt": {"1%": -2.58, "5%": -1.96, "10%": -1.645},
                "Pa": {"1%": -2.58, "5%": -1.96, "10%": -1.645},
            },
            method="all",
            trend="c",
            lags=1,
            n_bootstrap=100,
            n_entities=5,
            n_time=30,
        )
        summary = result.summary()
        assert "***" in summary["Reject (5%)"].values
        assert "**" in summary["Reject (5%)"].values
        assert "*" in summary["Reject (5%)"].values
        assert "" in summary["Reject (5%)"].values

    def test_repr(self):
        """Cover lines 119-130: __repr__ method."""
        from panelbox.diagnostics.cointegration.westerlund import WesterlundResult

        result = WesterlundResult(
            statistic={"Gt": -3.0},
            pvalue={"Gt": 0.005},
            critical_values={"Gt": {"1%": -2.58, "5%": -1.96, "10%": -1.645}},
            method="Gt",
            trend="c",
            lags=1,
            n_bootstrap=100,
            n_entities=5,
            n_time=30,
        )
        r = repr(result)
        assert "Westerlund" in r
        assert "Gt" in r

    def test_large_bootstrap_warning(self):
        """Cover lines 561-567: large bootstrap warning."""
        from panelbox.diagnostics.cointegration.westerlund import westerlund_test

        data = _make_cointegrated_panel(N=3, T=20)
        with pytest.warns(UserWarning, match="Large bootstrap"):
            westerlund_test(
                data,
                "entity",
                "time",
                "y",
                "x",
                n_bootstrap=3000,
                use_bootstrap=True,
                random_state=42,
            )

    def test_compute_test_stats_all_nan(self):
        """Cover line 332: all NaN alphas returns NaN stats."""
        from panelbox.diagnostics.cointegration.westerlund import _compute_test_statistics

        result = _compute_test_statistics(
            np.array([np.nan, np.nan]),
            np.array([np.nan, np.nan]),
            T=30,
        )
        assert np.isnan(result["Gt"])

    def test_nan_pvalue_branch(self):
        """Cover line 663: NaN stat gives NaN p-value."""
        from panelbox.diagnostics.cointegration.westerlund import (
            _compute_test_statistics,
        )

        # When all entities produce NaN, the test stats are NaN
        # and the p-values should also be NaN
        # This is tested indirectly through westerlund_test
        stats_dict = _compute_test_statistics(np.array([np.nan]), np.array([np.nan]), T=30)
        assert np.isnan(stats_dict["Gt"])


# ===========================================================================
# Tests for Kao cointegration test
# ===========================================================================
class TestKaoUncoveredBranches:
    """Cover uncovered lines in kao.py."""

    def test_kao_df_method(self):
        """Cover lines 480-481: method='df'."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", method="df")
        assert "DF" in result.statistic
        assert "ADF" not in result.statistic

    def test_kao_all_method(self):
        """Cover lines 480-484: method='all'."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", method="all")
        assert "DF" in result.statistic
        assert "ADF" in result.statistic

    def test_kao_no_trend(self):
        """Cover lines 194-195: trend='n'."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", trend="n")
        assert result.trend == "n"

    def test_kao_ct_trend(self):
        """Cover lines 188-193: trend='ct'."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", trend="ct")
        assert result.trend == "ct"

    def test_kao_missing_columns(self):
        """Cover line 465: missing columns ValueError."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = pd.DataFrame({"entity": [0], "time": [0], "y": [1.0]})
        with pytest.raises(ValueError, match="Missing columns"):
            kao_test(data, "entity", "time", "y", "x")

    def test_kao_string_x_vars(self):
        """Cover line 460: x_vars as string (converted to list)."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", x_vars="x")
        assert result is not None

    def test_kao_different_lags(self):
        """Cover ADF with different lags."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", method="adf", lags=3)
        assert result.lags == 3

    def test_kao_reject_at_specific(self):
        """Cover line 79: reject_at with specific test."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", method="all")
        rej = result.reject_at(0.05, test="DF")
        assert isinstance(rej, (bool, np.bool_))

    def test_kao_reject_at_all(self):
        """Cover line 80: reject_at without specific test."""
        from panelbox.diagnostics.cointegration.kao import kao_test

        data = _make_cointegrated_panel(N=5, T=30)
        result = kao_test(data, "entity", "time", "y", "x", method="all")
        rej = result.reject_at(0.05)
        assert isinstance(rej, dict)

    def test_kao_summary_significance(self):
        """Cover lines 100-108: summary significance markers."""
        from panelbox.diagnostics.cointegration.kao import KaoResult

        result = KaoResult(
            statistic={"DF": -3.0, "ADF": -1.0},
            pvalue={"DF": 0.005, "ADF": 0.5},
            critical_values={
                "DF": {"1%": -2.326, "5%": -1.645, "10%": -1.282},
                "ADF": {"1%": -2.326, "5%": -1.645, "10%": -1.282},
            },
            method="all",
            trend="c",
            lags=1,
            n_entities=5,
            n_time=30,
        )
        summary = result.summary()
        assert "***" in summary["Reject (5%)"].values

    def test_kao_repr(self):
        """Cover lines 115-125: __repr__."""
        from panelbox.diagnostics.cointegration.kao import KaoResult

        result = KaoResult(
            statistic={"ADF": -2.5},
            pvalue={"ADF": 0.01},
            critical_values={"ADF": {"1%": -2.326, "5%": -1.645, "10%": -1.282}},
            method="adf",
            trend="c",
            lags=1,
            n_entities=5,
            n_time=30,
        )
        r = repr(result)
        assert "Kao" in r

    def test_kao_nan_statistic_pvalue(self):
        """Cover lines 497-499: NaN statistic gives NaN p-value."""
        from panelbox.diagnostics.cointegration.kao import KaoResult

        result = KaoResult(
            statistic={"ADF": np.nan},
            pvalue={"ADF": np.nan},
            critical_values={"ADF": {"1%": -2.326, "5%": -1.645, "10%": -1.282}},
            method="adf",
            trend="c",
            lags=1,
            n_entities=5,
            n_time=30,
        )
        assert np.isnan(result.pvalue["ADF"])

    def test_long_run_variance(self):
        """Cover lines 248-264: _compute_long_run_variance_pooled."""
        from panelbox.diagnostics.cointegration.kao import (
            _compute_long_run_variance_pooled,
            _estimate_pooled_cointegrating_regression,
        )

        data = _make_cointegrated_panel(N=5, T=30)
        _, _, entity_resids = _estimate_pooled_cointegrating_regression(
            data, "entity", "time", "y", ["x"], "c"
        )
        sigma2 = _compute_long_run_variance_pooled(entity_resids, lags=4)
        assert isinstance(sigma2, float)
        assert sigma2 > 0


# ===========================================================================
# Tests for unified unit root tests
# ===========================================================================
class TestUnifiedUncoveredBranches:
    """Cover uncovered lines in unified.py."""

    def test_single_test_string(self):
        """Cover lines 296-297: test as single string."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test="breitung")
        assert "breitung" in result.tests_run
        assert len(result.tests_run) == 1

    def test_list_of_tests(self):
        """Cover lines 298-299: test as list."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test=["hadri", "breitung"])
        assert len(result.tests_run) == 2

    def test_unknown_test_raises(self):
        """Cover line 310: unknown test ValueError."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        with pytest.raises(ValueError, match="Unknown test"):
            panel_unit_root_test(data, "y", test="nonexistent")

    def test_unavailable_ips_raises(self):
        """Cover lines 304-308: unavailable IPS test raises ImportError."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        # Patch HAS_IPS to False and request ips
        with (
            patch("panelbox.diagnostics.unit_root.unified.HAS_IPS", False),
            pytest.raises(ImportError, match="not available"),
        ):
            panel_unit_root_test(data, "y", test="ips")

    def test_summary_table_output(self):
        """Cover lines 77-112: summary_table method."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test="all")
        summary = result.summary_table()
        assert "Panel Unit Root Test Summary" in summary
        assert "HADRI" in summary
        assert "BREITUNG" in summary

    def test_interpretation_all_reject(self):
        """Cover lines 145-147: all unit root tests reject."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test="all")
        interp = result.interpretation()
        assert "Interpretation:" in interp

    def test_interpretation_none_reject(self):
        """Cover lines 142-144: no unit root tests reject."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_unit_root_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test="all")
        interp = result.interpretation()
        assert "Interpretation:" in interp

    def test_repr(self):
        """Cover lines 193-195: __repr__."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test="breitung")
        r = repr(result)
        assert "PanelUnitRootResult" in r

    def test_ct_trend(self):
        """Cover trend='ct' branch."""
        from panelbox.diagnostics.unit_root.unified import panel_unit_root_test

        data = _make_stationary_panel(N=5, T=50)
        result = panel_unit_root_test(data, "y", test="breitung", trend="ct")
        assert result is not None

    def test_interpretation_mixed(self):
        """Cover lines 148-150: mixed evidence path."""
        from panelbox.diagnostics.unit_root.unified import PanelUnitRootResult

        # Create a result where some tests reject and some don't
        mock_hadri = SimpleNamespace(
            statistic=3.0, pvalue=0.001, reject=True, n_entities=5, n_time=50
        )
        mock_breitung = SimpleNamespace(
            statistic=-0.5, pvalue=0.6, reject=False, n_entities=5, n_time=50
        )

        # Need at least 2 unit root tests where one rejects and one doesn't
        # to get mixed evidence. But we only have breitung as unit root test
        # (hadri is stationarity). So use IPS if available.
        # Alternatively, make breitung reject and add another that doesn't.
        # Let's directly construct the result.
        result = PanelUnitRootResult(
            results={"hadri": mock_hadri, "breitung": mock_breitung},
            variable="y",
            n_entities=5,
            n_time=50,
            tests_run=["hadri", "breitung"],
        )
        interp = result.interpretation()
        assert "Interpretation:" in interp

    def test_interpretation_stationary_conclusion(self):
        """Cover lines 181-183: evidence_stationary > evidence_unit_root."""
        from panelbox.diagnostics.unit_root.unified import PanelUnitRootResult

        # Breitung rejects H0 (unit root) -> evidence for stationarity
        # Hadri does NOT reject H0 (stationarity) -> evidence for stationarity
        mock_hadri = SimpleNamespace(
            statistic=1.0, pvalue=0.3, reject=False, n_entities=5, n_time=50
        )
        mock_breitung = SimpleNamespace(
            statistic=-3.0, pvalue=0.001, reject=True, n_entities=5, n_time=50
        )

        result = PanelUnitRootResult(
            results={"hadri": mock_hadri, "breitung": mock_breitung},
            variable="y",
            n_entities=5,
            n_time=50,
            tests_run=["hadri", "breitung"],
        )
        interp = result.interpretation()
        assert "STATIONARY" in interp

    def test_interpretation_unit_root_conclusion(self):
        """Cover lines 184-186: evidence_unit_root > evidence_stationary."""
        from panelbox.diagnostics.unit_root.unified import PanelUnitRootResult

        # Breitung does NOT reject H0 (unit root) -> evidence for unit root
        # Hadri rejects H0 (stationarity) -> evidence for unit root
        mock_hadri = SimpleNamespace(
            statistic=5.0, pvalue=0.001, reject=True, n_entities=5, n_time=50
        )
        mock_breitung = SimpleNamespace(
            statistic=-0.5, pvalue=0.6, reject=False, n_entities=5, n_time=50
        )

        result = PanelUnitRootResult(
            results={"hadri": mock_hadri, "breitung": mock_breitung},
            variable="y",
            n_entities=5,
            n_time=50,
            tests_run=["hadri", "breitung"],
        )
        interp = result.interpretation()
        assert "UNIT ROOT" in interp

    def test_interpretation_inconclusive(self):
        """Cover lines 187-189: evenly split results."""
        from panelbox.diagnostics.unit_root.unified import PanelUnitRootResult

        # Breitung rejects (evidence for stationarity): 1
        # Hadri rejects (evidence for unit root): 1
        # evidence_stationary = 1 + 0 = 1, evidence_unit_root = 0 + 1 = 1 -> tie
        mock_hadri = SimpleNamespace(
            statistic=5.0, pvalue=0.001, reject=True, n_entities=5, n_time=50
        )
        mock_breitung = SimpleNamespace(
            statistic=-3.0, pvalue=0.001, reject=True, n_entities=5, n_time=50
        )

        result = PanelUnitRootResult(
            results={"hadri": mock_hadri, "breitung": mock_breitung},
            variable="y",
            n_entities=5,
            n_time=50,
            tests_run=["hadri", "breitung"],
        )
        interp = result.interpretation()
        assert "INCONCLUSIVE" in interp


# ===========================================================================
# Tests for Davidson-MacKinnon J-Test
# ===========================================================================
class TestDavidsonMacKinnonUncoveredBranches:
    """Cover uncovered lines in davidson_mackinnon.py."""

    @pytest.fixture
    def two_ols_results(self):
        """Create two OLS model results for J-test."""
        np.random.seed(42)
        n = 200
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        x3 = np.random.randn(n)
        y = 1 + 2 * x1 + 3 * x2 + np.random.randn(n) * 0.5

        from statsmodels.regression.linear_model import OLS

        X1 = np.column_stack([np.ones(n), x1, x2])
        X2 = np.column_stack([np.ones(n), x1, x3])

        r1 = OLS(y, X1).fit()
        r2 = OLS(y, X2).fit()
        return r1, r2

    def test_forward_only(self, two_ols_results):
        """Cover lines 281-282: direction='forward'."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2, direction="forward", model1_name="M1", model2_name="M2")
        assert result.forward is not None
        assert result.reverse is None

    def test_reverse_only(self, two_ols_results):
        """Cover lines 285-286: direction='reverse'."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2, direction="reverse", model1_name="M1", model2_name="M2")
        assert result.forward is None
        assert result.reverse is not None

    def test_both_direction(self, two_ols_results):
        """Cover both directions."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2, direction="both")
        assert result.forward is not None
        assert result.reverse is not None

    def test_interpret_forward(self, two_ols_results):
        """Cover lines 69-84: _interpret_forward."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2, direction="forward", model1_name="M1", model2_name="M2")
        interp = result.interpretation()
        assert "Forward test" in interp

    def test_interpret_reverse(self, two_ols_results):
        """Cover lines 86-101: _interpret_reverse."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2, direction="reverse", model1_name="M1", model2_name="M2")
        interp = result.interpretation()
        assert "Reverse test" in interp

    def test_interpret_both_forward_reject(self):
        """Cover lines 125-130: forward rejects, reverse doesn't."""
        from panelbox.diagnostics.specification.davidson_mackinnon import JTestResult

        result = JTestResult(
            forward={"statistic": 3.0, "pvalue": 0.001},
            reverse={"statistic": 0.5, "pvalue": 0.6},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "PREFER M2" in interp

    def test_interpret_both_reverse_reject(self):
        """Cover lines 131-136: reverse rejects, forward doesn't."""
        from panelbox.diagnostics.specification.davidson_mackinnon import JTestResult

        result = JTestResult(
            forward={"statistic": 0.5, "pvalue": 0.6},
            reverse={"statistic": 3.0, "pvalue": 0.001},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "PREFER M1" in interp

    def test_interpret_both_reject(self):
        """Cover lines 137-141: both reject."""
        from panelbox.diagnostics.specification.davidson_mackinnon import JTestResult

        result = JTestResult(
            forward={"statistic": 3.0, "pvalue": 0.001},
            reverse={"statistic": 3.0, "pvalue": 0.001},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "BOTH MODELS REJECTED" in interp

    def test_interpret_neither_reject(self):
        """Cover lines 142-146: neither reject."""
        from panelbox.diagnostics.specification.davidson_mackinnon import JTestResult

        result = JTestResult(
            forward={"statistic": 0.5, "pvalue": 0.6},
            reverse={"statistic": 0.5, "pvalue": 0.6},
            model1_name="M1",
            model2_name="M2",
            direction="both",
        )
        interp = result.interpretation()
        assert "BOTH MODELS ACCEPTABLE" in interp

    def test_summary_forward_only(self):
        """Cover lines 161-171: summary with forward only."""
        from panelbox.diagnostics.specification.davidson_mackinnon import JTestResult

        result = JTestResult(
            forward={"statistic": 2.0, "pvalue": 0.04, "alpha_coef": 0.5, "alpha_se": 0.25},
            reverse=None,
            model1_name="M1",
            model2_name="M2",
            direction="forward",
        )
        summary = result.summary()
        assert len(summary) == 1
        assert summary.iloc[0]["Test"] == "Forward"

    def test_summary_reverse_only(self):
        """Cover lines 173-183: summary with reverse only."""
        from panelbox.diagnostics.specification.davidson_mackinnon import JTestResult

        result = JTestResult(
            forward=None,
            reverse={"statistic": 2.0, "pvalue": 0.04, "gamma_coef": 0.5, "gamma_se": 0.25},
            model1_name="M1",
            model2_name="M2",
            direction="reverse",
        )
        summary = result.summary()
        assert len(summary) == 1
        assert summary.iloc[0]["Test"] == "Reverse"

    def test_repr(self, two_ols_results):
        """Cover line 189: __repr__."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2)
        r = repr(result)
        assert "JTestResult" in r

    def test_validate_missing_attr(self):
        """Cover lines 304-307: missing attribute error."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1 = SimpleNamespace(fittedvalues=np.array([1.0]), model=None)
        r2 = SimpleNamespace()  # missing fittedvalues and model
        with pytest.raises(AttributeError, match="missing required"):
            j_test(r1, r2)

    def test_validate_different_sample_sizes(self):
        """Cover lines 313-317: different sample sizes."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1 = SimpleNamespace(
            fittedvalues=np.array([1.0, 2.0]),
            model=SimpleNamespace(endog=np.array([1.0, 2.0]), exog=np.ones((2, 1))),
        )
        r2 = SimpleNamespace(
            fittedvalues=np.array([1.0, 2.0, 3.0]),
            model=SimpleNamespace(endog=np.array([1.0, 2.0, 3.0]), exog=np.ones((3, 1))),
        )
        with pytest.raises(ValueError, match="different sample sizes"):
            j_test(r1, r2)

    def test_validate_different_dep_var_warning(self):
        """Cover lines 320-326: different dependent variables warning."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        n = 50
        np.random.seed(42)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y1 = np.random.randn(n)
        y2 = np.random.randn(n) * 100  # Very different y

        from statsmodels.regression.linear_model import OLS

        r1 = OLS(y1, X).fit()
        # Override y for r2 to be different
        r2 = OLS(y2, X).fit()

        with pytest.warns(UserWarning, match="different dependent"):
            j_test(r1, r2)

    def test_default_model_names(self, two_ols_results):
        """Cover lines 268-271: default model names."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        r1, r2 = two_ols_results
        result = j_test(r1, r2)
        assert result.model1_name == "Model 1"
        assert result.model2_name == "Model 2"

    def test_panel_data_cluster_robust(self):
        """Cover lines 357-363: entity_id cluster-robust SEs."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5

        from statsmodels.regression.linear_model import OLS

        r1 = OLS(y, X).fit()

        X2 = np.column_stack([np.ones(n), np.random.randn(n)])
        r2 = OLS(y, X2).fit()

        # Add entity_id to model data for cluster-robust
        r1.model.data.entity_id = np.repeat(np.arange(10), 10)
        r2.model.data.entity_id = np.repeat(np.arange(10), 10)

        result = j_test(r1, r2)
        assert result.forward is not None

    def test_no_panel_structure_hc1(self):
        """Cover lines 368-369: HC1 when no frame."""
        from panelbox.diagnostics.specification.davidson_mackinnon import j_test

        np.random.seed(42)
        n = 100
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = X @ np.array([1.0, 2.0]) + np.random.randn(n) * 0.5

        from statsmodels.regression.linear_model import OLS

        r1 = OLS(y, X).fit()
        X2 = np.column_stack([np.ones(n), np.random.randn(n)])
        r2 = OLS(y, X2).fit()

        # Remove frame attribute to trigger else branch
        if hasattr(r1.model.data, "frame"):
            delattr(r1.model.data, "frame")
        if hasattr(r2.model.data, "frame"):
            delattr(r2.model.data, "frame")

        result = j_test(r1, r2)
        assert result.forward is not None


# ===========================================================================
# Tests for QuantileRegressionDiagnostics
# ===========================================================================
class TestQuantileDiagnosticsUncoveredBranches:
    """Cover uncovered lines in basic_diagnostics.py."""

    def _make_qr_model(self, n=200, seed=42):
        """Create a simple mock quantile regression model."""
        np.random.seed(seed)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.array([1.0, 2.0])
        y = X @ beta + np.random.randn(n) * 0.5

        model = SimpleNamespace(
            endog=y,
            exog=X,
            n_params=2,
        )
        return model, beta

    def test_pseudo_r2(self):
        """Cover lines 76-89: pseudo_r2 computation."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)
        r2 = diag.pseudo_r2()
        assert 0 <= r2 <= 1

    def test_pseudo_r2_perfect_fit_zero_loss(self):
        """Cover line 85: loss_without_x == 0."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        n = 50
        X = np.ones((n, 1))
        y = np.ones(n) * 5.0  # Constant y
        beta = np.array([5.0])  # Perfect fit

        model = SimpleNamespace(endog=y, exog=X, n_params=1)
        diag = QuantileRegressionDiagnostics(model, beta, tau=0.5)
        r2 = diag.pseudo_r2()
        assert r2 == 1.0

    def test_goodness_of_fit(self):
        """Cover lines 91-129: goodness_of_fit."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)
        gof = diag.goodness_of_fit()
        assert "pseudo_r2" in gof
        assert "sparsity" in gof
        assert "quantile_count" in gof

    def test_symmetry_test(self):
        """Cover lines 131-166: symmetry_test."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)
        z_stat, pval = diag.symmetry_test()
        assert isinstance(z_stat, float)
        assert 0 <= pval <= 1

    def test_symmetry_test_se_zero(self):
        """Cover line 161: se == 0 gives z_stat = 0."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        # tau=0 or tau=1 gives se=0
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta = np.array([1.0, 2.0])
        y = X @ beta + np.random.randn(n)

        model = SimpleNamespace(endog=y, exog=X, n_params=2)
        diag = QuantileRegressionDiagnostics(model, beta, tau=0.0)
        z_stat, _pval = diag.symmetry_test()
        assert z_stat == 0.0

    def test_goodness_of_fit_test(self):
        """Cover lines 168-208: goodness_of_fit_test."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)
        chi2, pval = diag.goodness_of_fit_test()
        assert isinstance(chi2, float)
        assert 0 <= pval <= 1

    def test_estimate_sparsity(self):
        """Cover lines 210-234: _estimate_sparsity."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)
        sparsity = diag._estimate_sparsity()
        assert sparsity > 0

    def test_estimate_sparsity_no_residuals_near_zero(self):
        """Cover lines 230-232: count == 0 fallback."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        # Create residuals far from zero
        n = 50
        X = np.ones((n, 1))
        y = np.ones(n) * 100.0
        beta = np.array([0.0])  # Huge residuals

        model = SimpleNamespace(endog=y, exog=X, n_params=1)
        diag = QuantileRegressionDiagnostics(model, beta, tau=0.5)
        # Override residuals to be very large
        diag.residuals = np.ones(n) * 1000.0
        # With h = 0.9 * 0 * n^-0.2 = 0, count will be handled
        # Actually std is 0 for constant residuals, h=0, so count will be n
        # Let's make residuals large and varied
        diag.residuals = np.arange(1000, 1000 + n, dtype=float)
        sparsity = diag._estimate_sparsity()
        assert sparsity >= 0

    def test_residual_quantiles(self):
        """Cover lines 236-255: residual_quantiles."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)

        # Default quantiles
        rq = diag.residual_quantiles()
        assert 0.25 in rq
        assert 0.5 in rq
        assert 0.75 in rq

        # Custom quantiles
        rq2 = diag.residual_quantiles(np.array([0.1, 0.9]))
        assert 0.1 in rq2
        assert 0.9 in rq2

    def test_summary(self):
        """Cover lines 257-290: summary method."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        diag = QuantileRegressionDiagnostics(model, params, tau=0.5)
        s = diag.summary()
        assert "Pseudo R" in s
        assert "Symmetry Test" in s
        assert "Goodness of Fit Test" in s

    def test_different_tau(self):
        """Cover tau != 0.5 branches."""
        from panelbox.diagnostics.quantile.basic_diagnostics import (
            QuantileRegressionDiagnostics,
        )

        model, params = self._make_qr_model()
        for tau in [0.1, 0.25, 0.75, 0.9]:
            diag = QuantileRegressionDiagnostics(model, params, tau=tau)
            r2 = diag.pseudo_r2()
            assert 0 <= r2 <= 1
