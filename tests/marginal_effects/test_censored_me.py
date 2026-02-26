"""
Tests for censored (Tobit) marginal effects.

Tests Average Marginal Effects (AME) and Marginal Effects at Means (MEM)
for Tobit models with conditional, unconditional, and probability predictions.
Uses mock objects to avoid fitting real Tobit models.
"""

import numpy as np
import pytest
from scipy.stats import norm

from panelbox.marginal_effects.censored_me import (
    _inverse_mills_ratio,
    _mills_ratio_derivative,
    compute_tobit_ame,
    compute_tobit_mem,
)
from panelbox.marginal_effects.discrete_me import MarginalEffectsResult

# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------


class MockTobitModel:
    """Lightweight mock of a fitted Tobit model."""

    def __init__(self, X, beta, sigma, exog_names, censoring_point=0.0):
        self.exog = X
        self.beta = np.asarray(beta, dtype=float)
        self.sigma = sigma
        self.exog_names = exog_names
        self.censoring_point = censoring_point
        self.censoring_type = "left"


class MockTobitResult:
    """Lightweight mock of a Tobit model result object."""

    def __init__(self, model):
        self.model = model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_tobit_data():
    """
    Simple dataset with 200 observations and 2 regressors.

    Returns (X, beta, sigma, exog_names).
    """
    rng = np.random.default_rng(42)
    n = 200
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([1.0, 0.5])
    sigma = 1.0
    exog_names = ["const", "x1"]
    return X, beta, sigma, exog_names


@pytest.fixture
def simple_result(simple_tobit_data):
    """MockTobitResult wrapping the simple dataset."""
    X, beta, sigma, exog_names = simple_tobit_data
    model = MockTobitModel(X, beta, sigma, exog_names, censoring_point=0.0)
    return MockTobitResult(model)


@pytest.fixture
def multi_var_result():
    """MockTobitResult with 3 regressors (const, x1, x2)."""
    rng = np.random.default_rng(123)
    n = 300
    X = np.column_stack(
        [
            np.ones(n),
            rng.standard_normal(n),
            rng.standard_normal(n),
        ]
    )
    beta = np.array([0.5, 1.0, -0.8])
    sigma = 1.5
    exog_names = ["const", "x1", "x2"]
    model = MockTobitModel(X, beta, sigma, exog_names, censoring_point=0.0)
    return MockTobitResult(model)


@pytest.fixture
def result_with_cov():
    """MockTobitResult including a covariance matrix for delta-method SEs."""
    rng = np.random.default_rng(99)
    n = 150
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    beta = np.array([1.0, 0.5])
    sigma = 1.0
    exog_names = ["const", "x1"]
    model = MockTobitModel(X, beta, sigma, exog_names)
    # Covariance matrix for [beta_0, beta_1, sigma] -- 3x3
    model.cov_params = np.diag([0.01, 0.02, 0.005])
    return MockTobitResult(model)


# ---------------------------------------------------------------------------
# Tests: _inverse_mills_ratio
# ---------------------------------------------------------------------------


class TestInverseMillsRatio:
    """Tests for _inverse_mills_ratio."""

    def test_at_zero(self):
        """IMR at z=0: phi(0)/Phi(0) = (1/sqrt(2*pi)) / 0.5 ~ 0.7979."""
        imr = _inverse_mills_ratio(np.array([0.0]))
        expected = norm.pdf(0.0) / norm.cdf(0.0)
        np.testing.assert_allclose(imr, expected, rtol=1e-10)
        np.testing.assert_allclose(imr, 0.7978845608, rtol=1e-5)

    def test_large_positive_z(self):
        """For large positive z, Phi(z) -> 1 and phi(z) -> 0, so IMR -> 0."""
        imr = _inverse_mills_ratio(np.array([5.0, 10.0]))
        assert np.all(imr >= 0)
        assert np.all(imr < 0.01)

    def test_large_negative_z_asymptotic(self):
        """For very negative z (Phi(z) < 1e-10), use asymptotic approx IMR ~ -z."""
        z_vals = np.array([-10.0, -20.0])
        imr = _inverse_mills_ratio(z_vals)
        # Asymptotic: IMR ~ -z when z << 0
        np.testing.assert_allclose(imr, -z_vals, rtol=0.1)

    def test_always_positive(self):
        """IMR should be positive for all z (phi and Phi are both positive)."""
        z = np.linspace(-5, 5, 100)
        imr = _inverse_mills_ratio(z)
        assert np.all(imr > 0)

    def test_monotonically_decreasing(self):
        """IMR is a decreasing function of z."""
        z = np.linspace(-4, 4, 200)
        imr = _inverse_mills_ratio(z)
        # Each successive value should be <= previous
        assert np.all(np.diff(imr) <= 1e-10)

    def test_scalar_like_input(self):
        """Accept a scalar (float) that gets coerced to 0-d array."""
        imr = _inverse_mills_ratio(0.0)
        expected = norm.pdf(0.0) / norm.cdf(0.0)
        np.testing.assert_allclose(imr, expected, rtol=1e-10)

    def test_safe_handling_very_negative(self):
        """
        When Phi(z) < 1e-10 the safe branch triggers.
        Verify no NaN / Inf is returned.
        """
        z = np.array([-50.0, -100.0])
        imr = _inverse_mills_ratio(z)
        assert np.all(np.isfinite(imr))
        # Asymptotic: should be close to -z
        np.testing.assert_allclose(imr, -z, rtol=0.01)


# ---------------------------------------------------------------------------
# Tests: _mills_ratio_derivative
# ---------------------------------------------------------------------------


class TestMillsRatioDerivative:
    """Tests for _mills_ratio_derivative."""

    def test_always_negative(self):
        """The derivative of IMR is always negative."""
        z = np.linspace(-4, 4, 200)
        deriv = _mills_ratio_derivative(z)
        assert np.all(deriv < 0)

    def test_at_zero(self):
        """d/dz[lambda(z)] at z=0 should be a known negative value."""
        deriv = _mills_ratio_derivative(np.array([0.0]))
        # lambda(0) ~ 0.7979
        lam = norm.pdf(0) / norm.cdf(0)
        expected = -lam * (0.0 + lam)
        np.testing.assert_allclose(deriv, expected, rtol=1e-10)
        assert deriv[0] < 0

    def test_matches_formula(self):
        """Verify d/dz[lambda(z)] = -lambda(z) * (z + lambda(z))."""
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        deriv = _mills_ratio_derivative(z)
        lam = _inverse_mills_ratio(z)
        expected = -lam * (z + lam)
        np.testing.assert_allclose(deriv, expected, rtol=1e-12)

    def test_bounded_between_minus_one_and_zero(self):
        """
        For moderate z the derivative should lie in (-1, 0).
        (It asymptotes to -1 for z -> -inf and to 0 for z -> +inf.)
        """
        z = np.linspace(-3, 3, 100)
        deriv = _mills_ratio_derivative(z)
        assert np.all(deriv > -1.0 - 1e-10)
        assert np.all(deriv < 0)


# ---------------------------------------------------------------------------
# Tests: compute_tobit_ame
# ---------------------------------------------------------------------------


class TestComputeTobitAME:
    """Tests for compute_tobit_ame."""

    # ---- unconditional ----

    def test_ame_unconditional_returns_result(self, simple_result):
        """AME unconditional returns a MarginalEffectsResult."""
        res = compute_tobit_ame(simple_result, which="unconditional")
        assert isinstance(res, MarginalEffectsResult)
        assert res.me_type == "AME_UNCONDITIONAL"

    def test_ame_unconditional_all_vars(self, simple_result):
        """AME unconditional computes for all variables by default."""
        res = compute_tobit_ame(simple_result, which="unconditional")
        assert "const" in res.marginal_effects.index
        assert "x1" in res.marginal_effects.index

    def test_ame_unconditional_attenuation(self, simple_result):
        """
        Unconditional ME should be attenuated relative to beta.
        ME_k = beta_k * mean(Phi(z)), and mean(Phi(z)) < 1.
        """
        res = compute_tobit_ame(simple_result, which="unconditional")
        beta = simple_result.model.beta
        for i, var in enumerate(simple_result.model.exog_names):
            me = res.marginal_effects[var]
            # |ME| <= |beta_k| because 0 < Phi(z) < 1
            assert abs(me) <= abs(beta[i]) + 1e-10

    def test_ame_unconditional_sign_matches_beta(self, simple_result):
        """Unconditional ME has the same sign as the coefficient."""
        res = compute_tobit_ame(simple_result, which="unconditional")
        beta = simple_result.model.beta
        for i, var in enumerate(simple_result.model.exog_names):
            me = res.marginal_effects[var]
            if abs(beta[i]) > 1e-10:
                assert np.sign(me) == np.sign(beta[i])

    # ---- conditional ----

    def test_ame_conditional_returns_result(self, simple_result):
        """AME conditional returns a MarginalEffectsResult."""
        res = compute_tobit_ame(simple_result, which="conditional")
        assert isinstance(res, MarginalEffectsResult)
        assert res.me_type == "AME_CONDITIONAL"

    def test_ame_conditional_all_vars(self, simple_result):
        """AME conditional computes for all variables by default."""
        res = compute_tobit_ame(simple_result, which="conditional")
        assert len(res.marginal_effects) == 2

    def test_ame_conditional_less_than_beta(self, simple_result):
        """
        Conditional ME: beta_k * (1 - lambda(z)*(z + lambda(z))).
        The factor (1 - ...) is in (0, 1), so |ME| < |beta|.
        """
        res = compute_tobit_ame(simple_result, which="conditional")
        beta = simple_result.model.beta
        for i, var in enumerate(simple_result.model.exog_names):
            me = res.marginal_effects[var]
            assert abs(me) <= abs(beta[i]) + 1e-10

    # ---- probability ----

    def test_ame_probability_returns_result(self, simple_result):
        """AME probability returns a MarginalEffectsResult."""
        res = compute_tobit_ame(simple_result, which="probability")
        assert isinstance(res, MarginalEffectsResult)
        assert res.me_type == "AME_PROBABILITY"

    def test_ame_probability_sign_matches_beta(self, simple_result):
        """Probability ME has the same sign as beta."""
        res = compute_tobit_ame(simple_result, which="probability")
        beta = simple_result.model.beta
        for i, var in enumerate(simple_result.model.exog_names):
            me = res.marginal_effects[var]
            if abs(beta[i]) > 1e-10:
                assert np.sign(me) == np.sign(beta[i])

    def test_ame_probability_values(self, simple_result):
        """Probability ME = (beta_k / sigma) * mean(phi(z))."""
        model = simple_result.model
        X = model.exog
        beta = model.beta
        sigma = model.sigma
        z = (X @ beta - model.censoring_point) / sigma
        phi_z = norm.pdf(z)

        res = compute_tobit_ame(simple_result, which="probability")
        for i, var in enumerate(model.exog_names):
            expected = (beta[i] / sigma) * np.mean(phi_z)
            np.testing.assert_allclose(res.marginal_effects[var], expected, rtol=1e-10)

    # ---- varlist ----

    def test_ame_varlist(self, multi_var_result):
        """AME with varlist computes only for requested variables."""
        res = compute_tobit_ame(multi_var_result, which="unconditional", varlist=["x1"])
        assert "x1" in res.marginal_effects.index
        assert "x2" not in res.marginal_effects.index
        assert "const" not in res.marginal_effects.index

    def test_ame_varlist_nonexistent_skipped(self, simple_result):
        """Variables not in the model are silently skipped."""
        res = compute_tobit_ame(simple_result, which="unconditional", varlist=["nonexistent"])
        assert len(res.marginal_effects) == 0

    # ---- standard errors ----

    def test_ame_se_nan_without_cov(self, simple_result):
        """When no cov_params, standard errors should be NaN."""
        res = compute_tobit_ame(simple_result, which="unconditional")
        for var in simple_result.model.exog_names:
            assert np.isnan(res.std_errors[var])

    def test_ame_se_with_cov(self, result_with_cov):
        """When cov_params is provided, SE computation is attempted."""
        res = compute_tobit_ame(result_with_cov, which="unconditional")
        # SE is either a finite number or NaN (if delta method fails);
        # just verify no exception is raised and results exist.
        for var in result_with_cov.model.exog_names:
            assert var in res.std_errors.index


# ---------------------------------------------------------------------------
# Tests: compute_tobit_mem
# ---------------------------------------------------------------------------


class TestComputeTobitMEM:
    """Tests for compute_tobit_mem."""

    # ---- unconditional ----

    def test_mem_unconditional(self, simple_result):
        """MEM unconditional returns a MarginalEffectsResult with at_values."""
        res = compute_tobit_mem(simple_result, which="unconditional")
        assert isinstance(res, MarginalEffectsResult)
        assert res.me_type == "MEM_UNCONDITIONAL"
        assert res.at_values is not None

    def test_mem_unconditional_formula(self, simple_result):
        """Verify MEM_unconditional = beta_k * Phi(z_bar)."""
        model = simple_result.model
        X = model.exog
        beta = model.beta
        sigma = model.sigma
        X_mean = np.mean(X, axis=0)
        z_bar = (X_mean @ beta - model.censoring_point) / sigma

        res = compute_tobit_mem(simple_result, which="unconditional")
        for i, var in enumerate(model.exog_names):
            expected = beta[i] * norm.cdf(z_bar)
            np.testing.assert_allclose(res.marginal_effects[var], expected, rtol=1e-10)

    # ---- conditional ----

    def test_mem_conditional(self, simple_result):
        """MEM conditional returns the correct type."""
        res = compute_tobit_mem(simple_result, which="conditional")
        assert isinstance(res, MarginalEffectsResult)
        assert res.me_type == "MEM_CONDITIONAL"

    def test_mem_conditional_formula(self, simple_result):
        """Verify MEM_conditional = beta_k * (1 - lambda(z_bar)*(z_bar + lambda(z_bar)))."""
        model = simple_result.model
        X = model.exog
        beta = model.beta
        sigma = model.sigma
        X_mean = np.mean(X, axis=0)
        z_bar = (X_mean @ beta - model.censoring_point) / sigma
        lam = _inverse_mills_ratio(np.array([z_bar]))[0]

        res = compute_tobit_mem(simple_result, which="conditional")
        for i, var in enumerate(model.exog_names):
            expected = beta[i] * (1 - lam * (z_bar + lam))
            np.testing.assert_allclose(res.marginal_effects[var], expected, rtol=1e-10)

    # ---- probability ----

    def test_mem_probability(self, simple_result):
        """MEM probability returns the correct type."""
        res = compute_tobit_mem(simple_result, which="probability")
        assert isinstance(res, MarginalEffectsResult)
        assert res.me_type == "MEM_PROBABILITY"

    def test_mem_probability_formula(self, simple_result):
        """Verify MEM_probability = (beta_k / sigma) * phi(z_bar)."""
        model = simple_result.model
        X = model.exog
        beta = model.beta
        sigma = model.sigma
        X_mean = np.mean(X, axis=0)
        z_bar = (X_mean @ beta - model.censoring_point) / sigma

        res = compute_tobit_mem(simple_result, which="probability")
        for i, var in enumerate(model.exog_names):
            expected = (beta[i] / sigma) * norm.pdf(z_bar)
            np.testing.assert_allclose(res.marginal_effects[var], expected, rtol=1e-10)

    # ---- at_values ----

    def test_mem_at_values_match_X_means(self, simple_result):
        """at_values should equal the column means of X."""
        res = compute_tobit_mem(simple_result, which="unconditional")
        X_mean = np.mean(simple_result.model.exog, axis=0)
        for i, var in enumerate(simple_result.model.exog_names):
            np.testing.assert_allclose(res.at_values[var], X_mean[i], rtol=1e-10)

    # ---- varlist ----

    def test_mem_varlist(self, multi_var_result):
        """MEM with varlist computes only for requested variables."""
        res = compute_tobit_mem(multi_var_result, which="probability", varlist=["x2"])
        assert "x2" in res.marginal_effects.index
        assert "x1" not in res.marginal_effects.index

    # ---- standard errors ----

    def test_mem_se_nan_without_cov(self, simple_result):
        """Without cov_params, SE should be NaN."""
        res = compute_tobit_mem(simple_result, which="conditional")
        for var in simple_result.model.exog_names:
            assert np.isnan(res.std_errors[var])

    def test_mem_se_with_cov(self, result_with_cov):
        """When cov_params is provided, SE computation is attempted."""
        res = compute_tobit_mem(result_with_cov, which="conditional")
        for var in result_with_cov.model.exog_names:
            assert var in res.std_errors.index


# ---------------------------------------------------------------------------
# Tests: Error handling & edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_which_ame(self, simple_result):
        """Invalid 'which' parameter raises ValueError for AME."""
        with pytest.raises(ValueError, match="which must be"):
            compute_tobit_ame(simple_result, which="invalid")

    def test_invalid_which_mem(self, simple_result):
        """Invalid 'which' parameter raises ValueError for MEM."""
        with pytest.raises(ValueError, match="which must be"):
            compute_tobit_mem(simple_result, which="bogus")

    def test_right_censoring_ame(self):
        """Right censoring raises NotImplementedError for AME."""
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.standard_normal(50)])
        model = MockTobitModel(X, [1.0, 0.5], 1.0, ["const", "x1"])
        model.censoring_type = "right"
        result = MockTobitResult(model)
        with pytest.raises(NotImplementedError, match="left censoring"):
            compute_tobit_ame(result, which="unconditional")

    def test_right_censoring_mem(self):
        """Right censoring raises NotImplementedError for MEM."""
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.standard_normal(50)])
        model = MockTobitModel(X, [1.0, 0.5], 1.0, ["const", "x1"])
        model.censoring_type = "right"
        result = MockTobitResult(model)
        with pytest.raises(NotImplementedError, match="left censoring"):
            compute_tobit_mem(result, which="unconditional")

    def test_model_without_beta_ame(self):
        """Model without beta raises ValueError for AME."""

        class _BareModel:
            censoring_point = 0.0
            censoring_type = "left"
            exog = np.ones((10, 1))
            exog_names = ["const"]

        class _BareResult:
            model = _BareModel()

        with pytest.raises(ValueError, match="fitted"):
            compute_tobit_ame(_BareResult(), which="unconditional")

    def test_model_without_beta_mem(self):
        """Model without beta raises ValueError for MEM."""

        class _BareModel:
            censoring_point = 0.0
            censoring_type = "left"
            exog = np.ones((10, 1))
            exog_names = ["const"]

        class _BareResult:
            model = _BareModel()

        with pytest.raises(ValueError, match="fitted"):
            compute_tobit_mem(_BareResult(), which="unconditional")

    def test_sigma_eps_fallback(self):
        """Model with sigma_eps (no sigma) is accepted."""
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(50), rng.standard_normal(50)])

        class _Model:
            exog = X
            beta = np.array([1.0, 0.5])
            sigma_eps = 1.0
            exog_names = ["const", "x1"]
            censoring_point = 0.0
            censoring_type = "left"

        class _Result:
            model = _Model()

        res = compute_tobit_ame(_Result(), which="unconditional")
        assert isinstance(res, MarginalEffectsResult)

    def test_result_without_model_attr(self):
        """
        If result has no .model attribute, the code should treat result itself
        as the model (line 178: model = result).
        """
        rng = np.random.default_rng(7)
        n = 40
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])

        class _DirectModel:
            exog = X
            beta = np.array([0.5, 0.3])
            sigma = 1.0
            exog_names = ["const", "x1"]
            censoring_point = 0.0
            censoring_type = "left"

        # Pass the model directly (no .model wrapper)
        res = compute_tobit_ame(_DirectModel(), which="unconditional")
        assert isinstance(res, MarginalEffectsResult)

    def test_exog_df_branch(self):
        """If model has exog_df (DataFrame), that branch is used."""
        import pandas as pd

        rng = np.random.default_rng(5)
        n = 60
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        df = pd.DataFrame(X, columns=["const", "x1"])

        class _Model:
            exog_df = df
            beta = np.array([1.0, 0.5])
            sigma = 1.0
            censoring_point = 0.0
            censoring_type = "left"

        class _Result:
            model = _Model()

        res_ame = compute_tobit_ame(_Result(), which="unconditional")
        assert "const" in res_ame.marginal_effects.index

        res_mem = compute_tobit_mem(_Result(), which="conditional")
        assert "x1" in res_mem.marginal_effects.index

    def test_model_without_exog_names(self):
        """If exog_names is missing, auto-generated names x0, x1, ... are used."""

        class _Model:
            exog = np.column_stack([np.ones(30), np.random.default_rng(0).standard_normal(30)])
            beta = np.array([1.0, 0.5])
            sigma = 1.0
            censoring_point = 0.0
            censoring_type = "left"

        class _Result:
            model = _Model()

        res = compute_tobit_ame(_Result(), which="unconditional")
        assert "x0" in res.marginal_effects.index
        assert "x1" in res.marginal_effects.index


# ---------------------------------------------------------------------------
# Tests: Relationships between AME / MEM / types
# ---------------------------------------------------------------------------


class TestRelationships:
    """Tests verifying mathematical relationships between ME types."""

    def test_unconditional_and_conditional_differ(self, simple_result):
        """
        Unconditional and conditional AME should generally differ.
        Unconditional uses Phi(z) as scaling while conditional uses
        (1 - lambda(z)*(z + lambda(z))), so they are different quantities.
        """
        res_uncond = compute_tobit_ame(simple_result, which="unconditional")
        res_cond = compute_tobit_ame(simple_result, which="conditional")
        # They should not be identical
        any_differ = False
        for var in simple_result.model.exog_names:
            if not np.isclose(
                res_uncond.marginal_effects[var],
                res_cond.marginal_effects[var],
                rtol=1e-5,
            ):
                any_differ = True
        assert any_differ

    def test_mem_equals_ame_for_constant_X(self):
        """
        When all observations have identical X, MEM should equal AME
        because the average equals the evaluation at means.
        """
        n = 100
        X = np.tile([1.0, 2.0], (n, 1))
        beta = np.array([0.5, 1.0])
        model = MockTobitModel(X, beta, 1.0, ["const", "x1"])
        result = MockTobitResult(model)

        for which_type in ["unconditional", "conditional", "probability"]:
            ame = compute_tobit_ame(result, which=which_type)
            mem = compute_tobit_mem(result, which=which_type)
            for var in model.exog_names:
                np.testing.assert_allclose(
                    ame.marginal_effects[var],
                    mem.marginal_effects[var],
                    rtol=1e-10,
                    err_msg=f"AME != MEM for {var} (which={which_type})",
                )

    def test_different_censoring_points(self):
        """Changing the censoring point changes the marginal effects."""
        rng = np.random.default_rng(55)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([1.0, 0.5])

        model_c0 = MockTobitModel(X, beta, 1.0, ["const", "x1"], censoring_point=0.0)
        model_c1 = MockTobitModel(X, beta, 1.0, ["const", "x1"], censoring_point=1.0)

        res_c0 = compute_tobit_ame(MockTobitResult(model_c0), which="unconditional")
        res_c1 = compute_tobit_ame(MockTobitResult(model_c1), which="unconditional")

        # Higher censoring point => fewer uncensored obs => lower unconditional ME
        assert abs(res_c1.marginal_effects["x1"]) < abs(res_c0.marginal_effects["x1"])

    def test_scaling_with_sigma(self):
        """
        Probability ME scales inversely with sigma:
        ME_prob = (beta_k / sigma) * mean(phi(z)).
        Doubling sigma roughly halves the magnitude (z also changes though).
        """
        rng = np.random.default_rng(77)
        n = 200
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        beta = np.array([2.0, 1.0])

        res_s1 = compute_tobit_ame(
            MockTobitResult(MockTobitModel(X, beta, 1.0, ["const", "x1"])),
            which="probability",
        )
        res_s3 = compute_tobit_ame(
            MockTobitResult(MockTobitModel(X, beta, 3.0, ["const", "x1"])),
            which="probability",
        )
        # With larger sigma the probability ME is smaller
        assert abs(res_s3.marginal_effects["x1"]) < abs(res_s1.marginal_effects["x1"])
