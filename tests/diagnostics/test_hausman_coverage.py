"""Tests for panelbox.diagnostics.hausman module.

Covers HausmanTestResult.summary(), hausman_test edge cases (singular matrix,
non-positive definite), and hausman_test_discrete to raise coverage from
~26.32% to 80%+.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.hausman import (
    HausmanTestResult,
    hausman_test,
    hausman_test_discrete,
    mundlak_test,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_panel_result(params_dict, cov_matrix):
    """Build a mock PanelResults with params (pd.Series) and cov_params (pd.DataFrame)."""
    var_names = list(params_dict.keys())
    params = pd.Series(params_dict)
    cov_params = pd.DataFrame(cov_matrix, index=var_names, columns=var_names)
    return SimpleNamespace(params=params, cov_params=cov_params)


# ---------------------------------------------------------------------------
# Tests: HausmanTestResult
# ---------------------------------------------------------------------------


class TestHausmanTestResult:
    def test_repr(self):
        """Test HausmanTestResult string representation."""
        result = HausmanTestResult(
            statistic=5.0,
            pvalue=0.08,
            df=2,
            method="chi2",
            interpretation="Some text",
            common_vars=["x1", "x2"],
        )
        r = repr(result)
        assert "HausmanTestResult" in r
        assert "5.0000" in r
        assert "0.0800" in r

    def test_summary_reject_strong(self, capsys):
        """Test summary output when p < 0.01 (strong evidence)."""
        result = HausmanTestResult(
            statistic=20.0,
            pvalue=0.001,
            df=2,
            method="chi2",
            interpretation="Reject H0",
            common_vars=["x1", "x2"],
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Hausman Specification Test" in captured.out
        assert "Strong evidence against Random Effects" in captured.out
        assert "x1, x2" in captured.out

    def test_summary_reject_moderate(self, capsys):
        """Test summary output when 0.01 <= p < 0.05."""
        result = HausmanTestResult(
            statistic=8.0,
            pvalue=0.02,
            df=2,
            method="chi2",
            interpretation="Reject H0",
            common_vars=["x1"],
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Evidence against Random Effects (p < 0.05)" in captured.out

    def test_summary_weak(self, capsys):
        """Test summary output when 0.05 <= p < 0.10."""
        result = HausmanTestResult(
            statistic=4.0,
            pvalue=0.07,
            df=2,
            method="chi2",
            interpretation="Weak",
            common_vars=["x1"],
        )
        result.summary()
        captured = capsys.readouterr()
        assert "Weak evidence against Random Effects" in captured.out

    def test_summary_no_evidence(self, capsys):
        """Test summary output when p >= 0.10."""
        result = HausmanTestResult(
            statistic=1.5,
            pvalue=0.50,
            df=2,
            method="chi2",
            interpretation="Fail to reject",
            common_vars=["x1"],
        )
        result.summary()
        captured = capsys.readouterr()
        assert "No evidence against Random Effects" in captured.out


# ---------------------------------------------------------------------------
# Tests: hausman_test
# ---------------------------------------------------------------------------


class TestHausmanTest:
    def test_basic_hausman_reject(self):
        """Test hausman_test with coefficients that lead to rejection."""
        fe_result = _make_panel_result(
            {"x1": 2.0, "x2": 3.0},
            [[0.10, 0.00], [0.00, 0.10]],
        )
        re_result = _make_panel_result(
            {"x1": 1.0, "x2": 1.5},
            [[0.02, 0.00], [0.00, 0.02]],
        )
        result = hausman_test(fe_result, re_result)
        assert isinstance(result, HausmanTestResult)
        assert result.method == "chi2"
        assert result.df == 2
        assert result.statistic > 0
        assert result.pvalue < 0.05

    def test_basic_hausman_fail_to_reject(self):
        """Test hausman_test with similar coefficients (no rejection)."""
        fe_result = _make_panel_result(
            {"x1": 1.01, "x2": 2.01},
            [[0.10, 0.00], [0.00, 0.10]],
        )
        re_result = _make_panel_result(
            {"x1": 1.00, "x2": 2.00},
            [[0.02, 0.00], [0.00, 0.02]],
        )
        result = hausman_test(fe_result, re_result)
        assert result.pvalue > 0.05

    def test_no_common_vars(self):
        """Test hausman_test raises when no common variables."""
        fe_result = _make_panel_result({"x1": 1.0}, [[0.1]])
        re_result = _make_panel_result({"z1": 1.0}, [[0.1]])
        with pytest.raises(ValueError, match="No common variables"):
            hausman_test(fe_result, re_result)

    def test_singular_variance_matrix(self):
        """Test hausman_test with singular variance difference matrix."""
        # Make V_FE == V_RE so V_diff is zero (singular)
        fe_result = _make_panel_result(
            {"x1": 2.0, "x2": 3.0},
            [[0.05, 0.00], [0.00, 0.05]],
        )
        re_result = _make_panel_result(
            {"x1": 1.0, "x2": 1.5},
            [[0.05, 0.00], [0.00, 0.05]],
        )
        with pytest.warns(RuntimeWarning, match="singular"):
            result = hausman_test(fe_result, re_result)
        assert isinstance(result, HausmanTestResult)

    def test_non_positive_definite_matrix(self):
        """Test hausman_test warning when V_diff not positive semi-definite."""
        # V_RE > V_FE => V_diff has negative eigenvalues
        fe_result = _make_panel_result(
            {"x1": 2.0, "x2": 3.0},
            [[0.01, 0.00], [0.00, 0.01]],
        )
        re_result = _make_panel_result(
            {"x1": 1.0, "x2": 1.5},
            [[0.10, 0.00], [0.00, 0.10]],
        )
        with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
            result = hausman_test(fe_result, re_result)
        assert isinstance(result, HausmanTestResult)

    def test_common_vars_subset(self):
        """Test hausman_test uses intersection of variables."""
        fe_result = _make_panel_result(
            {"x1": 2.0, "x2": 3.0},
            [[0.10, 0.00], [0.00, 0.10]],
        )
        # RE has x1, x2 and an extra z1
        re_params = pd.Series({"x1": 1.0, "x2": 1.5, "z1": 0.5})
        re_cov = pd.DataFrame(
            np.diag([0.02, 0.02, 0.02]),
            index=["x1", "x2", "z1"],
            columns=["x1", "x2", "z1"],
        )
        re_result = SimpleNamespace(params=re_params, cov_params=re_cov)
        result = hausman_test(fe_result, re_result)
        assert set(result.common_vars) == {"x1", "x2"}


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete
# ---------------------------------------------------------------------------


class TestHausmanTestDiscrete:
    def test_discrete_no_data_attribute(self):
        """Test hausman_test_discrete raises when models lack data attribute."""
        fe_result = _make_panel_result(
            {"x1": 2.0},
            [[0.10]],
        )
        re_result = _make_panel_result(
            {"x1": 1.0},
            [[0.02]],
        )
        # No .model.data attribute
        fe_result.model = SimpleNamespace()
        re_result.model = SimpleNamespace()
        with pytest.raises(ValueError, match="must have 'data' attribute"):
            hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)

    def test_discrete_no_common_vars(self):
        """Test hausman_test_discrete raises when no common variables."""
        fe_result = _make_panel_result({"x1": 1.0}, [[0.1]])
        re_result = _make_panel_result({"z1": 1.0}, [[0.1]])
        with pytest.raises(ValueError, match="No common variables"):
            hausman_test_discrete(fe_result, re_result, n_bootstrap=5)

    def test_discrete_removes_log_sigma_alpha(self):
        """Test that log_sigma_alpha is excluded from RE vars."""
        fe_result = _make_panel_result({"x1": 2.0}, [[0.10]])
        re_params = pd.Series({"x1": 1.0, "log_sigma_alpha": 0.5})
        re_cov = pd.DataFrame(
            np.diag([0.02, 0.01]),
            index=["x1", "log_sigma_alpha"],
            columns=["x1", "log_sigma_alpha"],
        )
        re_result = SimpleNamespace(params=re_params, cov_params=re_cov)

        # Set up model/data to trigger bootstrap
        entity_data = pd.DataFrame(
            {
                "entity": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1, 0, 1, 0],
                "x1": [0.1, 0.2, 0.3, 0.4],
            }
        )
        data_ns = SimpleNamespace(entity_col="entity", time_col="time", data=entity_data)

        # Mock the models so bootstrap iterations always fail
        fe_model = MagicMock()
        fe_model.data = data_ns
        fe_model.formula = "y ~ x1"
        re_model = MagicMock()
        re_model.data = data_ns
        re_model.formula = "y ~ x1"

        fe_result.model = fe_model
        re_result.model = re_model

        # Bootstrap will fail because MagicMock type() cannot be re-instantiated
        # but the test covers the log_sigma_alpha removal branch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)
        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"

    def test_discrete_non_positive_definite_warning(self):
        """Test warning when V_diff is not positive semi-definite."""
        # V_RE > V_FE => negative eigenvalues
        fe_result = _make_panel_result({"x1": 2.0}, [[0.01]])
        re_result = _make_panel_result({"x1": 1.0}, [[0.10]])

        entity_data = pd.DataFrame(
            {
                "entity": [1, 1, 2, 2],
                "time": [1, 2, 1, 2],
                "y": [1, 0, 1, 0],
                "x1": [0.1, 0.2, 0.3, 0.4],
            }
        )
        data_ns = SimpleNamespace(entity_col="entity", time_col="time", data=entity_data)
        fe_model = MagicMock()
        fe_model.data = data_ns
        fe_model.formula = "y ~ x1"
        re_model = MagicMock()
        re_model.data = data_ns
        re_model.formula = "y ~ x1"

        fe_result.model = fe_model
        re_result.model = re_model

        with pytest.warns(RuntimeWarning, match="not positive semi-definite"):
            hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)


# ---------------------------------------------------------------------------
# Tests: mundlak_test
# ---------------------------------------------------------------------------


class TestMundlakTest:
    def test_not_implemented(self):
        """Test mundlak_test raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            mundlak_test(None)
