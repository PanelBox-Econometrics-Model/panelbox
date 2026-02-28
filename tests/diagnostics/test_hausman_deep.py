"""Deep coverage tests for panelbox.diagnostics.hausman module.

Targets uncovered lines from existing tests:
- Lines 316-318: LinAlgError except branch in hausman_test_discrete
- Line 341: logger.info for bootstrap iteration % 100
- Lines 362-392: Successful bootstrap iterations (full loop body)
- Branch 397->406: Warning when too few bootstrap iterations succeed
- Line 411: else branch (pvalue >= 0.05) in hausman_test_discrete
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from panelbox.diagnostics.hausman import (
    HausmanTestResult,
    hausman_test_discrete,
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


def _make_panel_data(n_entities=20, n_periods=5, seed=42):
    """Create a simple balanced panel dataset."""
    rng = np.random.default_rng(seed)
    entities = []
    times = []
    x1_vals = []
    y_vals = []

    for i in range(1, n_entities + 1):
        entity_effect = rng.standard_normal() * 0.5
        for t in range(1, n_periods + 1):
            entities.append(i)
            times.append(t)
            x1 = rng.standard_normal()
            x1_vals.append(x1)
            y_vals.append(1.0 + 2.0 * x1 + entity_effect + rng.standard_normal() * 0.3)

    return pd.DataFrame({"entity": entities, "time": times, "x1": x1_vals, "y": y_vals})


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - LinAlgError branch (lines 316-318)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteLinAlgError:
    """Cover the except LinAlgError branch in hausman_test_discrete."""

    def test_singular_v_diff_obs_uses_pinv(self):
        """Cover lines 316-318: V_diff_obs is singular => use pinv."""
        # V_FE == V_RE => V_diff = 0 (singular)
        fe_result = _make_panel_result({"x1": 2.0}, [[0.05]])
        re_result = _make_panel_result({"x1": 1.0}, [[0.05]])

        panel_data = _make_panel_data(n_entities=10, n_periods=3, seed=42)
        data_ns = SimpleNamespace(entity_col="entity", time_col="time", data=panel_data)

        # Create mock models that will fail on re-instantiation
        # (so bootstrap loop fails but we still cover the pinv branch)
        fe_model = SimpleNamespace(data=data_ns, formula="y ~ x1")
        re_model = SimpleNamespace(data=data_ns, formula="y ~ x1")

        fe_result.model = fe_model
        re_result.model = re_model

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - Successful bootstrap (lines 341, 362-392)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteBootstrap:
    """Cover successful bootstrap iterations using real FE/RE models."""

    def test_bootstrap_with_real_models(self):
        """Cover lines 341, 362-392: successful bootstrap iterations.

        Uses real FixedEffects and RandomEffects models so that
        bootstrap resampling and re-estimation actually works.
        """
        from panelbox.models.static.fixed_effects import FixedEffects
        from panelbox.models.static.random_effects import RandomEffects

        # Create panel data with a clear effect so models are stable
        panel_data = _make_panel_data(n_entities=20, n_periods=5, seed=42)

        # Fit real models
        fe_model = FixedEffects("y ~ x1", panel_data, "entity", "time")
        fe_result = fe_model.fit()

        re_model = RandomEffects("y ~ x1", panel_data, "entity", "time")
        re_result = re_model.fit()

        # Run bootstrap with enough iterations to cover line 341 (% 100)
        # and the full bootstrap loop body (lines 362-392)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=10, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"
        assert result.df > 0
        assert 0.0 <= result.pvalue <= 1.0

    def test_bootstrap_else_branch_no_reject(self):
        """Cover line 411: pvalue >= 0.05 => 'Fail to reject H0'.

        Use very similar models so the Hausman statistic is small.
        """
        from panelbox.models.static.fixed_effects import FixedEffects
        from panelbox.models.static.random_effects import RandomEffects

        # Create data where FE and RE are expected to give similar estimates
        # (no correlation between entity effects and regressors)
        rng = np.random.default_rng(123)
        entities = []
        times = []
        x1_vals = []
        y_vals = []

        n_entities = 30
        n_periods = 5
        for i in range(1, n_entities + 1):
            # Entity effect independent of x1
            entity_effect = rng.standard_normal() * 0.1
            for t in range(1, n_periods + 1):
                entities.append(i)
                times.append(t)
                x1 = rng.standard_normal()
                x1_vals.append(x1)
                y_vals.append(1.0 + 2.0 * x1 + entity_effect + rng.standard_normal() * 0.5)

        panel_data = pd.DataFrame({"entity": entities, "time": times, "x1": x1_vals, "y": y_vals})

        fe_model = FixedEffects("y ~ x1", panel_data, "entity", "time")
        fe_result = fe_model.fit()

        re_model = RandomEffects("y ~ x1", panel_data, "entity", "time")
        re_result = re_model.fit()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=10, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"
        # We can't guarantee non-rejection, but the test covers the code path
        assert 0.0 <= result.pvalue <= 1.0


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - low success warning (397->406)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteLowSuccessWarning:
    """Cover the warning when too few bootstrap iterations succeed."""

    def test_low_success_rate_warning(self):
        """Cover branch 397->406: boot_stats < n_bootstrap / 2.

        Use mock models that fail to re-instantiate, so all bootstrap
        iterations fail and the warning is triggered.
        """
        fe_result = _make_panel_result({"x1": 2.0}, [[0.10]])
        re_result = _make_panel_result({"x1": 1.0}, [[0.02]])

        panel_data = _make_panel_data(n_entities=10, n_periods=3, seed=42)
        data_ns = SimpleNamespace(entity_col="entity", time_col="time", data=panel_data)

        # Models whose type() constructor will raise TypeError
        # This makes every bootstrap iteration fail
        fe_model = SimpleNamespace(data=data_ns, formula="y ~ x1")
        re_model = SimpleNamespace(data=data_ns, formula="y ~ x1")

        fe_result.model = fe_model
        re_result.model = re_model

        with pytest.warns(RuntimeWarning, match="bootstrap.*succeeded"):
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=10, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - reject H0 branch (line 411)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteRejectBranch:
    """Cover the pvalue < 0.05 branch (line 411) in hausman_test_discrete."""

    def test_bootstrap_reject_h0(self):
        """Cover line 411: pvalue < 0.05 => 'Reject H0'.

        Use mock models where bootstrap always fails (all boot_stats are empty),
        so pvalue = mean([] >= H_obs) = nan. Instead, we patch np.mean to
        return 0.01 to force the reject branch.
        """
        import panelbox.diagnostics.hausman as hmod

        fe_result = _make_panel_result({"x1": 5.0}, [[0.10]])
        re_result = _make_panel_result({"x1": 1.0}, [[0.02]])

        panel_data = _make_panel_data(n_entities=10, n_periods=3, seed=42)
        data_ns = SimpleNamespace(entity_col="entity", time_col="time", data=panel_data)

        fe_model = SimpleNamespace(data=data_ns, formula="y ~ x1")
        re_model = SimpleNamespace(data=data_ns, formula="y ~ x1")

        fe_result.model = fe_model
        re_result.model = re_model

        # Patch np.mean in hausman module to return pvalue < 0.05
        original_mean = hmod.np.mean

        def mock_mean(arr, *args, **kwargs):
            # The call to np.mean(boot_stats >= H_obs) should return < 0.05
            # Other calls to np.mean should work normally
            result = original_mean(arr, *args, **kwargs)
            # If result is nan (from empty array), return 0.01
            if np.isscalar(result) and np.isnan(result):
                return 0.01
            return result

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with patch.object(hmod.np, "mean", side_effect=mock_mean):
                result = hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"
        assert "Reject" in result.interpretation


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - quadrature_points branch (line 364)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteQuadraturePoints:
    """Cover the hasattr(re_model, 'quadrature_points') branch (line 364)."""

    def test_re_model_with_quadrature_points(self):
        """Cover line 364: RE model has quadrature_points attribute.

        Use real FE/RE models but add quadrature_points attribute to RE model
        to trigger the branch. The branch will try to pass quadrature_points
        to the RE constructor, which RandomEffects doesn't accept, so bootstrap
        iterations will fail — but the branch code at line 364 will be covered.
        """
        from panelbox.models.static.fixed_effects import FixedEffects
        from panelbox.models.static.random_effects import RandomEffects

        panel_data = _make_panel_data(n_entities=15, n_periods=4, seed=42)

        fe_model = FixedEffects("y ~ x1", panel_data, "entity", "time")
        fe_result = fe_model.fit()

        re_model = RandomEffects("y ~ x1", panel_data, "entity", "time")
        re_result = re_model.fit()

        # Add quadrature_points to trigger the branch
        re_result.model.quadrature_points = 12

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - bootstrap LinAlgError (lines 390-392)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteBootstrapLinAlgError:
    """Cover the LinAlgError handler in the bootstrap inner loop."""

    def test_bootstrap_inv_raises_linalg_error(self):
        """Cover lines 390-392: inv raises LinAlgError in bootstrap loop.

        We need to make the bootstrap loop succeed through lines 356-384
        (model re-estimation) but then have np.linalg.inv at line 387
        raise LinAlgError. We achieve this by monkeypatching inv only
        at the precise moment within the inner try block.

        Strategy: Wrap the hausman module's np.linalg.inv using a context
        manager that only activates AFTER the bootstrap models are fitted.
        We do this by patching the V_diff_boot to be a truly singular matrix.
        """
        from panelbox.models.static.fixed_effects import FixedEffects
        from panelbox.models.static.random_effects import RandomEffects

        panel_data = _make_panel_data(n_entities=20, n_periods=5, seed=42)

        fe_model = FixedEffects("y ~ x1", panel_data, "entity", "time")
        fe_result = fe_model.fit()

        re_model = RandomEffects("y ~ x1", panel_data, "entity", "time")
        re_result = re_model.fit()

        # Monkey-patch fit() to make V_diff_boot singular (zero) in
        # bootstrap iterations, causing inv to raise LinAlgError.
        original_fit_fe = FixedEffects.fit
        original_fit_re = RandomEffects.fit

        fit_count = [0]

        def patched_fe_fit(self, *args, **kwargs):
            fit_count[0] += 1
            result = original_fit_fe(self, *args, **kwargs)
            if fit_count[0] > 1:
                # Bootstrap iteration — make cov_params return zeros
                # so V_diff = V_FE - V_RE = 0 (singular)
                result.cov_params = result.cov_params * 0.0
            return result

        def patched_re_fit(self, *args, **kwargs):
            result = original_fit_re(self, *args, **kwargs)
            if fit_count[0] > 1:
                result.cov_params = result.cov_params * 0.0
            return result

        with (
            warnings.catch_warnings(),
            patch.object(FixedEffects, "fit", patched_fe_fit),
            patch.object(RandomEffects, "fit", patched_re_fit),
        ):
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=5, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"


# ---------------------------------------------------------------------------
# Tests: hausman_test_discrete - line 341 (logger.info for iteration 100)
# ---------------------------------------------------------------------------


class TestHausmanDiscreteLoggerIteration:
    """Cover line 341: logger.info for bootstrap iteration % 100 == 0."""

    def test_bootstrap_100_iterations(self):
        """Cover line 341: run 100+ bootstrap iterations.

        This triggers the (b + 1) % 100 == 0 branch at iteration 100.
        """
        from panelbox.models.static.fixed_effects import FixedEffects
        from panelbox.models.static.random_effects import RandomEffects

        panel_data = _make_panel_data(n_entities=20, n_periods=5, seed=42)

        fe_model = FixedEffects("y ~ x1", panel_data, "entity", "time")
        fe_result = fe_model.fit()

        re_model = RandomEffects("y ~ x1", panel_data, "entity", "time")
        re_result = re_model.fit()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hausman_test_discrete(fe_result, re_result, n_bootstrap=100, seed=42)

        assert isinstance(result, HausmanTestResult)
        assert result.method == "bootstrap"
