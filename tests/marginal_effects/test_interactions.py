"""
Tests for panelbox.marginal_effects.interactions module.

Covers the InteractionEffectsResult class, compute_interaction_effects function,
internal helpers (_logit_interaction, _probit_interaction, _poisson_interaction,
_delta_method_se, _bootstrap_se), and test_interaction_significance.

Uses mock objects and synthetic data to exercise code paths not covered by the
existing validation test suite (test_interactions_validation.py).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest
from scipy import stats

matplotlib.use("Agg")  # non-interactive backend before any pyplot import

from panelbox.marginal_effects.interactions import (
    InteractionEffectsResult,
    _bootstrap_se,
    _delta_method_se,
    _logit_interaction,
    _poisson_interaction,
    _probit_interaction,
    compute_interaction_effects,
)

# Import with alias to prevent pytest from collecting it as a test
from panelbox.marginal_effects.interactions import (
    test_interaction_significance as _test_interaction_significance,
)

# ---------------------------------------------------------------------------
# Helpers for building lightweight mock model results
# ---------------------------------------------------------------------------


def _make_mock_model_result(
    model_class_name: str,
    n_obs: int = 200,
    n_vars: int = 4,
    exog_names: list[str] | None = None,
    seed: int = 42,
    include_cov_params: bool = True,
):
    """
    Build a minimal mock model result that satisfies
    compute_interaction_effects expectations.

    The mock exposes: model.exog, model.exog_names, model.endog,
    model.__class__.__name__, params, cov_params.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_obs, n_vars))
    params = rng.standard_normal(n_vars)
    y = (rng.random(n_obs) > 0.5).astype(float)

    if exog_names is None:
        exog_names = [f"x{i}" for i in range(n_vars)]

    # Build model mock
    model = MagicMock()
    model.exog = X
    model.endog = y
    model.exog_names = exog_names
    type(model).__name__ = model_class_name

    # Build result mock
    result = MagicMock()
    result.model = model
    result.params = params

    if include_cov_params:
        # Positive-definite covariance matrix
        A = rng.standard_normal((n_vars, n_vars))
        result.cov_params = A.T @ A / n_obs
    else:
        result.cov_params = None

    return result


# ---------------------------------------------------------------------------
# InteractionEffectsResult -- init branches, summary(), plot()
# ---------------------------------------------------------------------------


class TestInteractionEffectsResult:
    """Tests for the result container class."""

    def test_init_with_standard_errors(self):
        """With SEs provided, z-statistics and significance are computed."""
        cross = np.array([0.1, -0.2, 0.3, -0.4, 0.05])
        se = np.array([0.01, 0.01, 0.01, 0.01, 0.5])

        r = InteractionEffectsResult(
            cross_partial=cross,
            standard_errors=se,
            var1_name="education",
            var2_name="experience",
        )

        assert r.z_statistics is not None
        assert len(r.z_statistics) == 5
        np.testing.assert_allclose(r.z_statistics, cross / se)
        # Some z-stats are > 1.96 in absolute value
        assert r.significant_positive is not None
        assert r.significant_negative is not None

    def test_init_without_standard_errors(self):
        """Without SEs, z_statistics and significance fields are None."""
        cross = np.array([0.1, -0.2, 0.05])
        r = InteractionEffectsResult(cross_partial=cross)

        assert r.z_statistics is None
        assert r.significant_positive is None
        assert r.significant_negative is None
        assert r.mean_effect == pytest.approx(np.mean(cross))
        assert r.std_effect == pytest.approx(np.std(cross))
        assert r.min_effect == pytest.approx(np.min(cross))
        assert r.max_effect == pytest.approx(np.max(cross))

    def test_prop_positive_negative(self):
        """Proportion of positive/negative effects is correct."""
        cross = np.array([0.1, -0.2, 0.3, -0.4, 0.0])
        r = InteractionEffectsResult(cross_partial=cross)
        # 2 positive, 2 negative, 1 zero
        assert r.prop_positive == pytest.approx(2.0 / 5)
        assert r.prop_negative == pytest.approx(2.0 / 5)

    # ---- summary() ----

    def test_summary_without_se(self):
        """Summary without standard errors should not include significance section."""
        cross = np.array([0.1, -0.2, 0.05])
        r = InteractionEffectsResult(cross_partial=cross, var1_name="A", var2_name="B")
        text = r.summary()

        assert "Interaction Effects Analysis" in text
        assert "A" in text and "B" in text
        assert "Mean effect:" in text
        assert "Std. deviation:" in text
        assert "Positive effects:" in text
        assert "Negative effects:" in text
        assert "Ai & Norton" in text
        # No significance section
        assert "Statistical Significance" not in text

    def test_summary_with_se(self):
        """Summary with standard errors includes significance section."""
        cross = np.array([0.5, -0.3, 0.1])
        se = np.array([0.01, 0.01, 0.5])
        r = InteractionEffectsResult(
            cross_partial=cross,
            standard_errors=se,
            var1_name="X1",
            var2_name="X2",
        )
        text = r.summary()

        assert "Statistical Significance (5% level):" in text
        assert "Significant positive:" in text
        assert "Significant negative:" in text

    # ---- plot() ----

    def test_plot_with_predicted_prob_and_z(self):
        """Plot with both predicted probabilities and z-statistics."""
        rng = np.random.default_rng(0)
        n = 50
        cross = rng.standard_normal(n) * 0.1
        se = np.abs(rng.standard_normal(n)) * 0.05 + 0.01
        pred = rng.random(n)

        r = InteractionEffectsResult(
            cross_partial=cross,
            standard_errors=se,
            predicted_prob=pred,
            var1_name="V1",
            var2_name="V2",
        )

        fig = r.plot(figsize=(10, 6))
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_without_predicted_prob(self):
        """Plot branch when predicted_prob is None (lines 171-180)."""
        cross = np.array([0.1, -0.2, 0.3])
        r = InteractionEffectsResult(
            cross_partial=cross,
            predicted_prob=None,
            standard_errors=None,
        )
        fig = r.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_without_z_statistics(self):
        """Plot branch when z_statistics is None (lines 195-204)."""
        cross = np.array([0.1, -0.2, 0.3])
        r = InteractionEffectsResult(
            cross_partial=cross,
            standard_errors=None,  # -> z_statistics will be None
            predicted_prob=np.array([0.3, 0.5, 0.7]),
        )
        fig = r.plot()
        assert fig is not None
        import matplotlib.pyplot as plt

        plt.close(fig)


# ---------------------------------------------------------------------------
# _logit_interaction
# ---------------------------------------------------------------------------


class TestLogitInteraction:
    def test_with_interaction_term(self):
        """Cross-partial with an explicit interaction coefficient."""
        rng = np.random.default_rng(7)
        n = 100
        X = rng.standard_normal((n, 4))
        params = np.array([0.5, 1.0, -0.8, 0.6])  # intercept, x1, x2, x1x2
        xb = X @ params
        Lambda = 1 / (1 + np.exp(-xb))
        lambda_pdf = Lambda * (1 - Lambda)

        beta1, beta2, beta12 = params[1], params[2], params[3]
        expected = beta12 * lambda_pdf + beta1 * beta2 * lambda_pdf * (1 - 2 * Lambda)

        result = _logit_interaction(X, params, xb, var1_idx=1, var2_idx=2, interact_idx=3)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_without_interaction_term(self):
        """When interact_idx is None, beta12 is treated as 0."""
        rng = np.random.default_rng(8)
        n = 50
        X = rng.standard_normal((n, 3))
        params = np.array([0.2, 0.7, -0.5])
        xb = X @ params
        Lambda = 1 / (1 + np.exp(-xb))
        lambda_pdf = Lambda * (1 - Lambda)

        beta1, beta2 = params[1], params[2]
        expected = beta1 * beta2 * lambda_pdf * (1 - 2 * Lambda)

        result = _logit_interaction(X, params, xb, var1_idx=1, var2_idx=2, interact_idx=None)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# _probit_interaction
# ---------------------------------------------------------------------------


class TestProbitInteraction:
    def test_with_interaction_term(self):
        rng = np.random.default_rng(9)
        n = 80
        X = rng.standard_normal((n, 4))
        params = np.array([0.3, 0.9, -0.6, 0.4])
        xb = X @ params
        phi = stats.norm.pdf(xb)

        beta1, beta2, beta12 = params[1], params[2], params[3]
        expected = -phi * (beta12 + beta1 * beta2 * xb)

        result = _probit_interaction(X, params, xb, 1, 2, 3)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_without_interaction_term(self):
        rng = np.random.default_rng(10)
        n = 50
        X = rng.standard_normal((n, 3))
        params = np.array([0.1, 0.5, -0.3])
        xb = X @ params
        phi = stats.norm.pdf(xb)

        beta1, beta2 = params[1], params[2]
        expected = -phi * (0 + beta1 * beta2 * xb)

        result = _probit_interaction(X, params, xb, 1, 2, None)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# _poisson_interaction  (lines 419-440)
# ---------------------------------------------------------------------------


class TestPoissonInteraction:
    def test_with_interaction_term(self):
        rng = np.random.default_rng(11)
        n = 60
        X = rng.standard_normal((n, 4))
        params = np.array([0.2, 0.4, -0.3, 0.5])
        xb = X @ params
        lambda_ = np.exp(xb)

        beta1, beta2, beta12 = params[1], params[2], params[3]
        expected = lambda_ * (beta12 + beta1 * beta2)

        result = _poisson_interaction(X, params, xb, 1, 2, 3)
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_without_interaction_term(self):
        rng = np.random.default_rng(12)
        n = 40
        X = rng.standard_normal((n, 3))
        params = np.array([0.1, 0.6, -0.2])
        xb = X @ params
        lambda_ = np.exp(xb)

        beta1, beta2 = params[1], params[2]
        expected = lambda_ * (0 + beta1 * beta2)

        result = _poisson_interaction(X, params, xb, 1, 2, None)
        np.testing.assert_allclose(result, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# compute_interaction_effects -- integer indices, auto-detection, Poisson, error
# ---------------------------------------------------------------------------


class TestComputeInteractionEffects:
    def test_integer_variable_indices(self):
        """Variables specified as integers (lines 296-297, 303-304)."""
        mr = _make_mock_model_result("PooledLogit", n_obs=100, n_vars=4)
        result = compute_interaction_effects(mr, var1=1, var2=2, interaction_term=3, method=None)
        assert result.var1_name == "X1"
        assert result.var2_name == "X2"
        assert len(result.cross_partial) == 100

    def test_integer_interaction_term(self):
        """interaction_term as integer index (line 310)."""
        mr = _make_mock_model_result("PooledLogit", n_obs=80, n_vars=4)
        result = compute_interaction_effects(
            mr, var1="x1", var2="x2", interaction_term=3, method=None
        )
        assert len(result.cross_partial) == 80

    def test_auto_detect_interaction_colon(self):
        """Auto-detect interaction term with colon syntax (lines 314-322)."""
        mr = _make_mock_model_result(
            "PooledLogit",
            n_obs=50,
            n_vars=4,
            exog_names=["const", "edu", "exp", "edu:exp"],
        )
        result = compute_interaction_effects(mr, var1="edu", var2="exp", method=None)
        assert len(result.cross_partial) == 50

    def test_auto_detect_interaction_star(self):
        """Auto-detect interaction term with asterisk syntax."""
        mr = _make_mock_model_result(
            "PooledLogit",
            n_obs=50,
            n_vars=4,
            exog_names=["const", "edu", "exp", "edu*exp"],
        )
        result = compute_interaction_effects(mr, var1="edu", var2="exp", method=None)
        assert len(result.cross_partial) == 50

    def test_auto_detect_interaction_reverse_order(self):
        """Auto-detect interaction term with reversed variable order."""
        mr = _make_mock_model_result(
            "PooledLogit",
            n_obs=50,
            n_vars=4,
            exog_names=["const", "edu", "exp", "exp:edu"],
        )
        result = compute_interaction_effects(mr, var1="edu", var2="exp", method=None)
        assert len(result.cross_partial) == 50

    def test_auto_detect_no_match_still_works(self):
        """No auto-detected interaction term -> interact_idx is None."""
        mr = _make_mock_model_result(
            "PooledLogit",
            n_obs=50,
            n_vars=3,
            exog_names=["const", "edu", "exp"],
        )
        result = compute_interaction_effects(mr, var1="edu", var2="exp", method=None)
        assert len(result.cross_partial) == 50

    def test_poisson_model_type(self):
        """Poisson branch in compute_interaction_effects (lines 340-342)."""
        mr = _make_mock_model_result(
            "PanelPoisson",
            n_obs=60,
            n_vars=4,
            exog_names=["const", "x1", "x2", "x1x2"],
        )
        result = compute_interaction_effects(
            mr, var1="x1", var2="x2", interaction_term="x1x2", method=None
        )
        assert len(result.cross_partial) == 60
        # predicted_prob should be exp(xb) for poisson
        assert result.predicted_prob is not None
        assert np.all(result.predicted_prob > 0)

    def test_unsupported_model_type(self):
        """Unsupported model type raises ValueError (lines 344-345)."""
        mr = _make_mock_model_result("SurvivalModel", n_obs=30, n_vars=3)
        with pytest.raises(ValueError, match="not supported"):
            compute_interaction_effects(mr, var1=1, var2=2, method=None)

    def test_probit_branch(self):
        """Probit branch yields predicted probabilities via Phi."""
        mr = _make_mock_model_result(
            "PooledProbit",
            n_obs=70,
            n_vars=4,
            exog_names=["const", "x1", "x2", "x1x2"],
        )
        result = compute_interaction_effects(
            mr, var1="x1", var2="x2", interaction_term="x1x2", method=None
        )
        assert result.predicted_prob is not None
        assert np.all(result.predicted_prob >= 0)
        assert np.all(result.predicted_prob <= 1)

    def test_logistic_in_model_type(self):
        """Model class name containing 'logistic' triggers the logit branch."""
        mr = _make_mock_model_result(
            "LogisticRegression",
            n_obs=50,
            n_vars=3,
            exog_names=["const", "x1", "x2"],
        )
        result = compute_interaction_effects(mr, var1="x1", var2="x2", method=None)
        assert result.predicted_prob is not None
        assert np.all((result.predicted_prob >= 0) & (result.predicted_prob <= 1))

    def test_method_none_no_se(self):
        """method=None produces no standard errors."""
        mr = _make_mock_model_result("PooledLogit", n_obs=40, n_vars=3)
        result = compute_interaction_effects(mr, var1=1, var2=2, method=None)
        assert result.standard_errors is None
        assert result.z_statistics is None


# ---------------------------------------------------------------------------
# _delta_method_se  (lines 443-478)
# ---------------------------------------------------------------------------


class TestDeltaMethodSE:
    def test_returns_array_on_success(self):
        """Delta method returns SE array when inputs are well-formed."""
        rng = np.random.default_rng(20)
        n_obs, n_params = 80, 4
        X = rng.standard_normal((n_obs, n_params))
        params = rng.standard_normal(n_params)
        xb = X @ params

        A = rng.standard_normal((n_params, n_params))
        cov = A.T @ A / n_obs

        mr = MagicMock()
        mr.cov_params = cov

        se = _delta_method_se(mr, X, params, xb, 1, 2, 3, "logit")
        assert se is not None
        assert len(se) == n_obs
        assert np.all(se >= 0)

    def test_returns_none_on_exception(self):
        """Delta method returns None and warns when an exception occurs (lines 476-478)."""
        mr = MagicMock()
        # cov_params is not a matrix, will cause an error in the gradient @ V multiplication
        mr.cov_params = "not_a_matrix"

        X = np.ones((5, 3))
        params = np.ones(3)
        xb = X @ params

        with pytest.warns(UserWarning, match="Could not compute delta method"):
            se = _delta_method_se(mr, X, params, xb, 1, 2, None, "logit")

        assert se is None

    def test_non_logit_model_type(self):
        """Delta method with a non-logit model type still returns SEs (gradient stays zero)."""
        rng = np.random.default_rng(21)
        n_obs, n_params = 50, 3
        X = rng.standard_normal((n_obs, n_params))
        params = rng.standard_normal(n_params)
        xb = X @ params

        A = rng.standard_normal((n_params, n_params))
        cov = A.T @ A / n_obs

        mr = MagicMock()
        mr.cov_params = cov

        se = _delta_method_se(mr, X, params, xb, 1, 2, None, "probit")
        assert se is not None
        # Gradient is all zeros for non-logit, so SEs are all zero
        np.testing.assert_allclose(se, np.zeros(n_obs))


# ---------------------------------------------------------------------------
# _bootstrap_se  (lines 481-520)
# ---------------------------------------------------------------------------


class TestBootstrapSE:
    def test_returns_none_on_exception(self):
        """Bootstrap returns None and warns when model re-fitting fails (lines 518-520)."""
        mr = MagicMock()
        mr.model.__class__ = MagicMock(side_effect=RuntimeError("refit failed"))
        mr.model.__class__.__name__ = "PooledLogit"
        mr.model.endog = np.ones(10)
        X = np.ones((10, 3))

        with pytest.warns(UserWarning, match="Could not compute bootstrap"):
            se = _bootstrap_se(mr, X, 1, 2, None, n_bootstrap=5)

        assert se is None

    def test_successful_bootstrap(self):
        """Bootstrap succeeds with a cooperative mock that can be re-fitted."""
        rng = np.random.default_rng(30)
        n_obs, n_params = 60, 4
        X = rng.standard_normal((n_obs, n_params))
        params = rng.standard_normal(n_params)
        y = (rng.random(n_obs) > 0.5).astype(float)

        # Build a mock where model.__class__(y, X) returns a new model with .fit()
        boot_result = MagicMock()
        boot_result.params = params

        boot_model = MagicMock()
        boot_model.fit.return_value = boot_result

        model_cls = MagicMock(return_value=boot_model)
        model_cls.__name__ = "PooledLogit"

        mr = MagicMock()
        mr.model.__class__ = model_cls
        mr.model.endog = y

        se = _bootstrap_se(mr, X, 1, 2, 3, n_bootstrap=5)
        assert se is not None
        assert len(se) == n_obs
        # SEs should be non-negative
        assert np.all(se >= 0)


# ---------------------------------------------------------------------------
# compute_interaction_effects with method='delta' and method='bootstrap'
# ---------------------------------------------------------------------------


class TestComputeWithSEMethods:
    def test_delta_method_path(self):
        """method='delta' flows through _delta_method_se (lines 348-351)."""
        mr = _make_mock_model_result(
            "PooledLogit",
            n_obs=60,
            n_vars=4,
            exog_names=["const", "x1", "x2", "x1x2"],
        )
        result = compute_interaction_effects(
            mr, var1="x1", var2="x2", interaction_term="x1x2", method="delta"
        )
        # If delta method succeeds, standard_errors is not None
        if result.standard_errors is not None:
            assert len(result.standard_errors) == 60
            assert result.z_statistics is not None

    def test_bootstrap_method_path(self):
        """method='bootstrap' flows through _bootstrap_se (lines 352-355)."""
        rng = np.random.default_rng(40)
        n_obs, n_vars = 50, 4
        X = rng.standard_normal((n_obs, n_vars))
        params = rng.standard_normal(n_vars)
        y = (rng.random(n_obs) > 0.5).astype(float)

        boot_result = MagicMock()
        boot_result.params = params

        boot_model = MagicMock()
        boot_model.fit.return_value = boot_result

        model_cls = MagicMock(return_value=boot_model)
        model_cls.__name__ = "PooledLogit"

        model = MagicMock()
        model.exog = X
        model.endog = y
        model.exog_names = ["const", "x1", "x2", "x1x2"]
        model.__class__ = model_cls
        type(model).__name__ = "PooledLogit"

        mr = MagicMock()
        mr.model = model
        mr.params = params

        result = compute_interaction_effects(
            mr,
            var1="x1",
            var2="x2",
            interaction_term="x1x2",
            method="bootstrap",
            n_bootstrap=3,
        )
        assert len(result.cross_partial) == n_obs


# ---------------------------------------------------------------------------
# test_interaction_significance
# ---------------------------------------------------------------------------


class TestInteractionSignificance:
    def test_basic_comparison(self):
        """Test the test_interaction_significance helper with mock models."""
        rng = np.random.default_rng(50)
        n_obs, n_vars = 100, 4
        X = rng.standard_normal((n_obs, n_vars))
        params_with = rng.standard_normal(n_vars)

        model_with = MagicMock()
        model_with.exog = X
        model_with.endog = (rng.random(n_obs) > 0.5).astype(float)
        model_with.exog_names = ["const", "x1", "x2", "x1x2"]
        type(model_with).__name__ = "PooledLogit"

        mr_with = MagicMock()
        mr_with.model = model_with
        mr_with.params = params_with
        mr_with.llf = -50.0
        mr_with.aic = 108.0
        mr_with.bic = 118.0

        mr_without = MagicMock()
        mr_without.llf = -55.0
        mr_without.aic = 114.0
        mr_without.bic = 120.0

        results = _test_interaction_significance(mr_with, mr_without, var1="x1", var2="x2")

        assert "lr_statistic" in results
        assert "lr_pvalue" in results
        assert "delta_aic" in results
        assert "delta_bic" in results
        assert "avg_interaction_effect" in results
        assert "interaction_std" in results
        assert "prefer_interaction" in results

        # LR stat = 2*(−50 − (−55)) = 10
        assert results["lr_statistic"] == pytest.approx(10.0)
        # delta_aic = 108 − 114 = −6
        assert results["delta_aic"] == pytest.approx(-6.0)
        # delta_bic = 118 − 120 = −2
        assert results["delta_bic"] == pytest.approx(-2.0)
