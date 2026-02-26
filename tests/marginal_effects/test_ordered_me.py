"""
Test suite for ordered model marginal effects.

Tests AME/MEM computations, the OrderedMarginalEffectsResult container,
and associated helper properties/methods for ordered logit/probit models.
"""

import matplotlib

matplotlib.use("Agg")

import io
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.marginal_effects.ordered_me import (
    OrderedMarginalEffectsResult,
    compute_ordered_ame,
    compute_ordered_mem,
)

# ---------------------------------------------------------------------------
# Mock objects that satisfy the interface expected by ordered_me
# ---------------------------------------------------------------------------


class MockOrderedModel:
    """Mock ordered model with exog data and category info."""

    def __init__(self, X, n_categories, exog_names):
        self.exog = np.asarray(X)
        self.n_categories = n_categories
        self.exog_names = list(exog_names)


class MockOrderedLogitModel(MockOrderedModel):
    """Mock ordered *Logit* model -- class name contains 'Logit'."""

    pass


class MockOrderedResult:
    """Mock result object returned by an ordered model fit."""

    def __init__(self, model, beta, cutpoints, cov_params=None):
        self.model = model
        self.beta = np.asarray(beta, dtype=float)
        self.cutpoints = np.asarray(cutpoints, dtype=float)
        self.params = np.concatenate([self.beta, self.cutpoints])
        n_params = len(self.params)
        self.cov_params = cov_params if cov_params is not None else np.eye(n_params) * 0.01


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def probit_data():
    """Create a simple dataset and mock probit-style result."""
    np.random.seed(42)
    n_obs = 100
    n_vars = 2
    n_categories = 3  # 3 outcomes -> 2 cutpoints

    X = np.random.randn(n_obs, n_vars)
    exog_names = ["x1", "x2"]

    beta = np.array([0.5, -0.3])
    cutpoints = np.array([-0.5, 0.8])

    model = MockOrderedModel(X, n_categories, exog_names)
    result = MockOrderedResult(model, beta, cutpoints)
    return result


@pytest.fixture
def logit_data():
    """Create a simple dataset and mock logit-style result."""
    np.random.seed(123)
    n_obs = 80
    n_vars = 2
    n_categories = 4  # 4 outcomes -> 3 cutpoints

    X = np.random.randn(n_obs, n_vars)
    exog_names = ["income", "age"]

    beta = np.array([0.4, 0.2])
    cutpoints = np.array([-1.0, 0.0, 1.0])

    model = MockOrderedLogitModel(X, n_categories, exog_names)
    result = MockOrderedResult(model, beta, cutpoints)
    return result


@pytest.fixture
def single_var_data():
    """Create a minimal one-variable scenario."""
    np.random.seed(99)
    n_obs = 50
    n_categories = 2  # binary-like: 2 categories, 1 cutpoint

    X = np.random.randn(n_obs, 1)
    exog_names = ["z"]

    beta = np.array([1.0])
    cutpoints = np.array([0.0])

    model = MockOrderedModel(X, n_categories, exog_names)
    result = MockOrderedResult(model, beta, cutpoints)
    return result


# ===========================================================================
# Tests: compute_ordered_ame
# ===========================================================================


class TestComputeOrderedAME:
    """Tests for the average marginal effects (AME) function."""

    def test_ame_returns_result_object(self, probit_data):
        """AME should return an OrderedMarginalEffectsResult."""
        res = compute_ordered_ame(probit_data)
        assert isinstance(res, OrderedMarginalEffectsResult)

    def test_ame_shape_probit(self, probit_data):
        """AME DataFrame should have shape (n_vars, n_categories)."""
        res = compute_ordered_ame(probit_data)
        n_vars = probit_data.model.exog.shape[1]
        n_cats = probit_data.model.n_categories
        assert res.marginal_effects.shape == (n_vars, n_cats)
        assert res.std_errors.shape == (n_vars, n_cats)

    def test_ame_shape_logit(self, logit_data):
        """AME with logit PDF should have correct shape."""
        res = compute_ordered_ame(logit_data)
        n_vars = logit_data.model.exog.shape[1]
        n_cats = logit_data.model.n_categories
        assert res.marginal_effects.shape == (n_vars, n_cats)
        assert res.std_errors.shape == (n_vars, n_cats)

    def test_ame_sum_to_zero_probit(self, probit_data):
        """AME across categories must sum to zero (probit)."""
        res = compute_ordered_ame(probit_data)
        assert res.verify_sum_to_zero(tol=1e-8)

    def test_ame_sum_to_zero_logit(self, logit_data):
        """AME across categories must sum to zero (logit)."""
        res = compute_ordered_ame(logit_data)
        assert res.verify_sum_to_zero(tol=1e-8)

    def test_ame_se_positive(self, probit_data):
        """Standard errors must be strictly positive."""
        res = compute_ordered_ame(probit_data)
        assert np.all(res.std_errors.values > 0)

    def test_ame_type_label(self, probit_data):
        """The me_type attribute should be 'AME'."""
        res = compute_ordered_ame(probit_data)
        assert res.me_type == "AME"

    def test_ame_at_values_is_none(self, probit_data):
        """AME should not store at_values."""
        res = compute_ordered_ame(probit_data)
        assert res.at_values is None

    def test_ame_varlist_subset(self, probit_data):
        """Passing a subset varlist should limit output rows."""
        res = compute_ordered_ame(probit_data, varlist=["x1"])
        assert res.marginal_effects.shape[0] == 1
        assert "x1" in res.marginal_effects.index

    def test_ame_varlist_excess(self, probit_data):
        """Varlist longer than n_vars should be safely truncated."""
        res = compute_ordered_ame(probit_data, varlist=["x1", "x2", "x3_extra"])
        # x3_extra at index 2 is >= n_vars=2, so skipped
        assert res.marginal_effects.shape[0] == 2

    def test_ame_no_exog_names(self):
        """When model has no exog_names, fall back to x0, x1, ..."""
        np.random.seed(7)
        X = np.random.randn(60, 2)
        model = MockOrderedModel(X, 3, ["a", "b"])
        # Remove exog_names to test fallback
        del model.exog_names

        beta = np.array([0.3, -0.1])
        cutpoints = np.array([-0.2, 0.5])
        result = MockOrderedResult(model, beta, cutpoints)

        res = compute_ordered_ame(result)
        assert "x0" in res.marginal_effects.index
        assert "x1" in res.marginal_effects.index

    def test_ame_single_var(self, single_var_data):
        """AME with a single variable and 2 categories."""
        res = compute_ordered_ame(single_var_data)
        assert res.marginal_effects.shape == (1, 2)
        assert res.verify_sum_to_zero(tol=1e-8)

    def test_ame_finite_values(self, probit_data):
        """All marginal effects and SEs should be finite."""
        res = compute_ordered_ame(probit_data)
        assert np.all(np.isfinite(res.marginal_effects.values))
        assert np.all(np.isfinite(res.std_errors.values))


# ===========================================================================
# Tests: compute_ordered_mem
# ===========================================================================


class TestComputeOrderedMEM:
    """Tests for the marginal effects at means (MEM) function."""

    def test_mem_returns_result_object(self, probit_data):
        """MEM should return an OrderedMarginalEffectsResult."""
        res = compute_ordered_mem(probit_data)
        assert isinstance(res, OrderedMarginalEffectsResult)

    def test_mem_shape_probit(self, probit_data):
        """MEM shape should be (n_vars, n_categories)."""
        res = compute_ordered_mem(probit_data)
        n_vars = probit_data.model.exog.shape[1]
        n_cats = probit_data.model.n_categories
        assert res.marginal_effects.shape == (n_vars, n_cats)
        assert res.std_errors.shape == (n_vars, n_cats)

    def test_mem_shape_logit(self, logit_data):
        """MEM with logit PDF should have correct shape."""
        res = compute_ordered_mem(logit_data)
        n_vars = logit_data.model.exog.shape[1]
        n_cats = logit_data.model.n_categories
        assert res.marginal_effects.shape == (n_vars, n_cats)

    def test_mem_sum_to_zero_probit(self, probit_data):
        """MEM across categories must sum to zero (probit)."""
        res = compute_ordered_mem(probit_data)
        assert res.verify_sum_to_zero(tol=1e-10)

    def test_mem_sum_to_zero_logit(self, logit_data):
        """MEM across categories must sum to zero (logit)."""
        res = compute_ordered_mem(logit_data)
        assert res.verify_sum_to_zero(tol=1e-10)

    def test_mem_se_positive(self, probit_data):
        """Standard errors must be strictly positive."""
        res = compute_ordered_mem(probit_data)
        assert np.all(res.std_errors.values > 0)

    def test_mem_type_label(self, probit_data):
        """The me_type should be 'MEM'."""
        res = compute_ordered_mem(probit_data)
        assert res.me_type == "MEM"

    def test_mem_at_values_populated(self, probit_data):
        """MEM should store at_values (the means)."""
        res = compute_ordered_mem(probit_data)
        assert res.at_values is not None
        assert isinstance(res.at_values, dict)
        for name in probit_data.model.exog_names:
            assert name in res.at_values

    def test_mem_at_values_match_means(self, probit_data):
        """Stored at_values should match column means of exog."""
        res = compute_ordered_mem(probit_data)
        X_mean = probit_data.model.exog.mean(axis=0)
        for i, name in enumerate(probit_data.model.exog_names):
            assert_allclose(res.at_values[name], X_mean[i], atol=1e-12)

    def test_mem_varlist_subset(self, probit_data):
        """Passing a subset varlist should limit output rows."""
        res = compute_ordered_mem(probit_data, varlist=["x2"])
        assert res.marginal_effects.shape[0] == 1
        assert "x2" in res.marginal_effects.index

    def test_mem_no_exog_names(self):
        """Fall back to x0, x1, ... when exog_names is missing."""
        np.random.seed(11)
        X = np.random.randn(40, 3)
        model = MockOrderedModel(X, 3, ["a", "b", "c"])
        del model.exog_names

        beta = np.array([0.1, -0.2, 0.3])
        cutpoints = np.array([0.0, 1.0])
        result = MockOrderedResult(model, beta, cutpoints)

        res = compute_ordered_mem(result)
        assert "x0" in res.marginal_effects.index

    def test_mem_finite_values(self, logit_data):
        """All marginal effects and SEs should be finite."""
        res = compute_ordered_mem(logit_data)
        assert np.all(np.isfinite(res.marginal_effects.values))
        assert np.all(np.isfinite(res.std_errors.values))


# ===========================================================================
# Tests: OrderedMarginalEffectsResult container
# ===========================================================================


class TestOrderedMarginalEffectsResult:
    """Tests for the result container class."""

    @pytest.fixture
    def ame_result(self, probit_data):
        """Pre-computed AME result for container tests."""
        return compute_ordered_ame(probit_data)

    @pytest.fixture
    def mem_result(self, probit_data):
        """Pre-computed MEM result for container tests."""
        return compute_ordered_mem(probit_data)

    # --- z_stats & pvalues ---

    def test_z_stats_shape(self, ame_result):
        """z_stats should have the same shape as marginal_effects."""
        z = ame_result.z_stats
        assert z.shape == ame_result.marginal_effects.shape

    def test_z_stats_finite(self, ame_result):
        """z_stats should be finite."""
        z = ame_result.z_stats
        assert np.all(np.isfinite(z.values))

    def test_z_stats_computation(self, ame_result):
        """z_stats should equal ME / SE."""
        expected = ame_result.marginal_effects / ame_result.std_errors
        pd.testing.assert_frame_equal(ame_result.z_stats, expected)

    def test_pvalues_shape(self, ame_result):
        """pvalues should have the same shape as marginal_effects."""
        pv = ame_result.pvalues
        assert pv.shape == ame_result.marginal_effects.shape

    def test_pvalues_range(self, ame_result):
        """pvalues must be between 0 and 1."""
        pv = ame_result.pvalues
        assert np.all(pv.values >= 0)
        assert np.all(pv.values <= 1)

    def test_pvalues_finite(self, ame_result):
        """pvalues should be finite."""
        pv = ame_result.pvalues
        assert np.all(np.isfinite(pv.values))

    # --- verify_sum_to_zero ---

    def test_verify_sum_to_zero_passes(self, ame_result):
        """verify_sum_to_zero should return True for real AME."""
        assert ame_result.verify_sum_to_zero()

    def test_verify_sum_to_zero_fails_on_bad_data(self, probit_data):
        """verify_sum_to_zero should return False for nonsense data."""
        me_df = pd.DataFrame(
            {"Category_0": [1.0, 2.0], "Category_1": [3.0, 4.0]},
            index=["x1", "x2"],
        )
        se_df = pd.DataFrame(
            {"Category_0": [0.1, 0.1], "Category_1": [0.1, 0.1]},
            index=["x1", "x2"],
        )
        bad_result = OrderedMarginalEffectsResult(me_df, se_df, probit_data, me_type="ame")
        assert not bad_result.verify_sum_to_zero()

    def test_verify_custom_tolerance(self, ame_result):
        """Custom tolerance should be respected."""
        # With a very tight tolerance and real data this should still pass
        assert ame_result.verify_sum_to_zero(tol=1e-6)

    # --- n_categories ---

    def test_n_categories(self, ame_result):
        """n_categories should match the number of columns."""
        assert ame_result.n_categories == ame_result.marginal_effects.shape[1]

    # --- __repr__ ---

    def test_repr_contains_type(self, ame_result):
        """repr should mention the ME type."""
        r = repr(ame_result)
        assert "AME" in r

    def test_repr_contains_counts(self, ame_result):
        """repr should contain n_variables and n_categories."""
        r = repr(ame_result)
        assert "n_variables=" in r
        assert "n_categories=" in r
        assert "OrderedMarginalEffectsResult" in r

    def test_repr_mem(self, mem_result):
        """repr of MEM result should show MEM type."""
        r = repr(mem_result)
        assert "MEM" in r

    # --- summary ---

    def test_summary_produces_output(self, ame_result):
        """summary() should print to stdout."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            ame_result.summary()
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert len(output) > 0
        assert "AME" in output

    def test_summary_contains_category_count(self, ame_result):
        """summary should mention number of categories."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            ame_result.summary()
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "Number of categories" in output

    def test_summary_contains_se(self, ame_result):
        """summary should contain a Standard Errors section."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            ame_result.summary()
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "Standard Errors" in output

    def test_summary_verify_message(self, ame_result):
        """summary should report sum-to-zero verification."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            ame_result.summary()
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        # Real AME sums to zero, so we expect the verified message
        assert "sum to zero" in output.lower() or "verified" in output.lower()

    def test_summary_significance_codes(self, ame_result):
        """summary should contain significance legend."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            ame_result.summary()
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "Significance codes" in output

    def test_summary_mem_at_values(self, mem_result):
        """MEM summary should print 'Evaluated at' section."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            mem_result.summary()
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert "Evaluated at" in output

    def test_summary_custom_alpha(self, ame_result):
        """summary(alpha=0.10) should still run without error."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            ame_result.summary(alpha=0.10)
        finally:
            sys.stdout = old_stdout
        output = captured.getvalue()
        assert len(output) > 0

    # --- plot ---

    def test_plot_returns_axis(self, ame_result):
        """plot() should return a matplotlib axis."""
        ax = ame_result.plot("x1")
        assert ax is not None
        plt.close("all")

    def test_plot_with_provided_axis(self, ame_result):
        """plot() should use the axis when one is provided."""
        _fig, ax = plt.subplots()
        returned_ax = ame_result.plot("x1", ax=ax)
        assert returned_ax is ax
        plt.close("all")

    def test_plot_invalid_variable(self, ame_result):
        """plot() should raise ValueError for unknown variable."""
        with pytest.raises(ValueError, match="not found"):
            ame_result.plot("nonexistent_var")
        plt.close("all")

    def test_plot_each_variable(self, ame_result):
        """plot() should work for every variable in the result."""
        for var in ame_result.marginal_effects.index:
            ax = ame_result.plot(var)
            assert ax is not None
        plt.close("all")

    def test_plot_kwargs_forwarded(self, ame_result):
        """Extra kwargs should be forwarded to the bar call."""
        ax = ame_result.plot("x1", alpha=0.5, edgecolor="black")
        assert ax is not None
        plt.close("all")


# ===========================================================================
# Tests: AME vs MEM comparison
# ===========================================================================


class TestAMEvsMEM:
    """Compare AME and MEM for consistency."""

    def test_ame_and_mem_shapes_match(self, probit_data):
        """AME and MEM should have the same shape."""
        ame = compute_ordered_ame(probit_data)
        mem = compute_ordered_mem(probit_data)
        assert ame.marginal_effects.shape == mem.marginal_effects.shape

    def test_ame_and_mem_generally_different(self, probit_data):
        """AME and MEM values should generally differ."""
        ame = compute_ordered_ame(probit_data)
        mem = compute_ordered_mem(probit_data)
        # They should not be exactly equal in general
        assert not np.allclose(ame.marginal_effects.values, mem.marginal_effects.values, atol=1e-12)

    def test_both_sum_to_zero(self, probit_data):
        """Both AME and MEM should satisfy the sum-to-zero property."""
        ame = compute_ordered_ame(probit_data)
        mem = compute_ordered_mem(probit_data)
        assert ame.verify_sum_to_zero(tol=1e-8)
        assert mem.verify_sum_to_zero(tol=1e-10)


# ===========================================================================
# Tests: Edge cases and model with _pdf attribute
# ===========================================================================


class TestEdgeCases:
    """Edge-case and special-path tests."""

    def test_model_with_custom_pdf(self):
        """When model has _pdf attribute, it should be used."""
        np.random.seed(77)
        n_obs, n_vars, n_cats = 50, 2, 3
        X = np.random.randn(n_obs, n_vars)
        exog_names = ["v1", "v2"]

        model = MockOrderedModel(X, n_cats, exog_names)
        # Attach a custom _pdf (e.g. Cauchy-like)
        model._pdf = lambda z: 1 / (np.pi * (1 + z**2))

        beta = np.array([0.3, -0.2])
        cutpoints = np.array([-0.5, 0.5])
        result = MockOrderedResult(model, beta, cutpoints)

        ame = compute_ordered_ame(result)
        assert ame.marginal_effects.shape == (n_vars, n_cats)
        assert np.all(np.isfinite(ame.marginal_effects.values))
        # Sum to zero still holds for any symmetric PDF
        assert ame.verify_sum_to_zero(tol=1e-6)

    def test_model_with_custom_pdf_mem(self):
        """MEM should also use _pdf when present."""
        np.random.seed(78)
        n_obs, n_vars, n_cats = 40, 1, 2
        X = np.random.randn(n_obs, n_vars)

        model = MockOrderedModel(X, n_cats, ["w"])
        model._pdf = lambda z: 1 / (np.pi * (1 + z**2))

        beta = np.array([0.5])
        cutpoints = np.array([0.0])
        result = MockOrderedResult(model, beta, cutpoints)

        mem = compute_ordered_mem(result)
        assert mem.marginal_effects.shape == (1, 2)
        assert mem.verify_sum_to_zero(tol=1e-8)

    def test_model_with_transform_cutpoints(self):
        """When model has _transform_cutpoints, SE computation should use it."""
        np.random.seed(88)
        n_obs, n_vars, n_cats = 60, 2, 3
        X = np.random.randn(n_obs, n_vars)

        model = MockOrderedModel(X, n_cats, ["a", "b"])
        # _transform_cutpoints that applies cumulative-sum transformation
        model._transform_cutpoints = lambda c: np.cumsum(np.abs(c))

        beta = np.array([0.2, -0.1])
        cutpoints = np.array([0.3, 0.5])
        result = MockOrderedResult(model, beta, cutpoints)

        ame = compute_ordered_ame(result)
        assert ame.marginal_effects.shape == (n_vars, n_cats)
        assert np.all(np.isfinite(ame.std_errors.values))
        assert np.all(ame.std_errors.values > 0)

    def test_mem_with_transform_cutpoints(self):
        """MEM path should also respect _transform_cutpoints."""
        np.random.seed(89)
        n_obs, n_vars, n_cats = 40, 1, 2
        X = np.random.randn(n_obs, n_vars)

        model = MockOrderedModel(X, n_cats, ["q"])
        model._transform_cutpoints = lambda c: c  # identity

        beta = np.array([0.7])
        cutpoints = np.array([0.0])
        result = MockOrderedResult(model, beta, cutpoints)

        mem = compute_ordered_mem(result)
        assert mem.marginal_effects.shape == (1, 2)
        assert np.all(np.isfinite(mem.std_errors.values))

    def test_result_without_cov_params(self):
        """If result lacks cov_params, the fallback identity*0.01 is used."""
        np.random.seed(90)
        n_obs, n_vars, n_cats = 30, 2, 3
        X = np.random.randn(n_obs, n_vars)
        model = MockOrderedModel(X, n_cats, ["p", "q"])

        beta = np.array([0.1, -0.1])
        cutpoints = np.array([-0.3, 0.3])
        result = MockOrderedResult(model, beta, cutpoints)

        # Remove cov_params to test fallback
        del result.cov_params

        ame = compute_ordered_ame(result)
        assert np.all(np.isfinite(ame.std_errors.values))
        assert np.all(ame.std_errors.values > 0)

    def test_mem_without_cov_params(self):
        """MEM path should also use fallback when cov_params is missing."""
        np.random.seed(91)
        n_obs, n_vars, n_cats = 30, 1, 2
        X = np.random.randn(n_obs, n_vars)
        model = MockOrderedModel(X, n_cats, ["r"])

        beta = np.array([0.5])
        cutpoints = np.array([0.0])
        result = MockOrderedResult(model, beta, cutpoints)
        del result.cov_params

        mem = compute_ordered_mem(result)
        assert np.all(np.isfinite(mem.std_errors.values))


# ===========================================================================
# Tests: Logit vs Probit PDF selection
# ===========================================================================


class TestPDFSelection:
    """Tests that the correct PDF is chosen based on the model class name."""

    def test_logit_uses_logistic_pdf(self, logit_data):
        """Model with 'Logit' in class name should use logistic PDF."""
        res = compute_ordered_ame(logit_data)
        # The logistic PDF has fatter tails, so effects can differ
        # Just verify it produces valid results
        assert np.all(np.isfinite(res.marginal_effects.values))
        assert res.verify_sum_to_zero(tol=1e-8)

    def test_probit_uses_normal_pdf(self, probit_data):
        """Model without 'Logit' in class name should use normal PDF."""
        res = compute_ordered_ame(probit_data)
        assert np.all(np.isfinite(res.marginal_effects.values))
        assert res.verify_sum_to_zero(tol=1e-8)

    def test_logit_and_probit_differ(self):
        """AME from logit and probit on same data should differ."""
        np.random.seed(55)
        n_obs, n_vars, n_cats = 60, 2, 3
        X = np.random.randn(n_obs, n_vars)

        beta = np.array([0.5, -0.3])
        cutpoints = np.array([-0.5, 0.8])

        # Probit
        model_probit = MockOrderedModel(X, n_cats, ["x1", "x2"])
        res_probit = MockOrderedResult(model_probit, beta, cutpoints)
        ame_probit = compute_ordered_ame(res_probit)

        # Logit
        model_logit = MockOrderedLogitModel(X, n_cats, ["x1", "x2"])
        res_logit = MockOrderedResult(model_logit, beta, cutpoints)
        ame_logit = compute_ordered_ame(res_logit)

        # Effects should be somewhat different due to different PDFs
        assert not np.allclose(
            ame_probit.marginal_effects.values,
            ame_logit.marginal_effects.values,
            atol=1e-6,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
