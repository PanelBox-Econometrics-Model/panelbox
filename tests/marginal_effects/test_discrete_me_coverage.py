"""
Deep coverage tests for panelbox.marginal_effects.discrete_me.

Targets uncovered lines: OrderedMarginalEffectsResult (plot, summary,
verify_sum_to_zero, z_stats, pvalues, __repr__), compute_ordered_ame,
compute_ordered_mem, MarginalEffectsResult branches (summary with
at_values, significance stars).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from panelbox.marginal_effects.discrete_me import (
    MarginalEffectsResult,
    OrderedMarginalEffectsResult,
    compute_ame,
    compute_mem,
    compute_mer,
    compute_ordered_ame,
    compute_ordered_mem,
)
from panelbox.models.discrete.binary import PooledLogit, PooledProbit


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def logit_panel_data():
    """Panel data suitable for logit estimation."""
    np.random.seed(99)
    n, t = 80, 5
    entity = np.repeat(np.arange(n), t)
    time = np.tile(np.arange(t), n)
    x1 = np.random.randn(n * t)
    x2 = np.random.binomial(1, 0.4, n * t).astype(float)
    x3 = np.random.randn(n * t)
    eta = 0.3 + 0.6 * x1 + 0.9 * x2 - 0.4 * x3
    prob = 1 / (1 + np.exp(-eta))
    y = np.random.binomial(1, prob)
    return pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture
def probit_panel_data():
    """Panel data suitable for probit estimation."""
    np.random.seed(77)
    n, t = 80, 4
    entity = np.repeat(np.arange(n), t)
    time = np.tile(np.arange(t), n)
    x1 = np.random.randn(n * t)
    x2 = np.random.binomial(1, 0.5, n * t).astype(float)
    from scipy.stats import norm

    eta = 0.2 + 0.5 * x1 + 0.7 * x2
    prob = norm.cdf(eta)
    y = np.random.binomial(1, prob)
    return pd.DataFrame({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})


@pytest.fixture
def ordered_logit_model():
    """Fitted OrderedLogit model for marginal effects tests."""
    from panelbox.models.discrete import OrderedLogit

    np.random.seed(42)
    N = 300
    K = 2
    J = 4
    beta_true = np.array([0.5, -0.3])
    cutpoints_true = np.array([-1.0, 0.0, 1.0])
    X = np.random.randn(N, K)
    linear_pred = X @ beta_true
    y = np.zeros(N, dtype=int)
    for i in range(N):
        y_star = linear_pred[i] + np.random.logistic(0, 1)
        if y_star <= cutpoints_true[0]:
            y[i] = 0
        elif y_star <= cutpoints_true[1]:
            y[i] = 1
        elif y_star <= cutpoints_true[2]:
            y[i] = 2
        else:
            y[i] = 3
    groups = np.arange(N)
    model = OrderedLogit(endog=y, exog=X, groups=groups, n_categories=J)
    model.exog_names = ["x1", "x2"]
    model.fit(maxiter=500)
    return model


@pytest.fixture
def ordered_probit_model():
    """Fitted OrderedProbit model for marginal effects tests."""
    from panelbox.models.discrete import OrderedProbit

    np.random.seed(55)
    N = 300
    K = 2
    J = 3
    beta_true = np.array([0.4, -0.5])
    cutpoints_true = np.array([-0.5, 0.8])
    X = np.random.randn(N, K)
    linear_pred = X @ beta_true
    y = np.zeros(N, dtype=int)
    for i in range(N):
        y_star = linear_pred[i] + np.random.normal(0, 1)
        if y_star <= cutpoints_true[0]:
            y[i] = 0
        elif y_star <= cutpoints_true[1]:
            y[i] = 1
        else:
            y[i] = 2
    groups = np.arange(N)
    model = OrderedProbit(endog=y, exog=X, groups=groups, n_categories=J)
    model.exog_names = ["x1", "x2"]
    model.fit(maxiter=500)
    return model


# ---------------------------------------------------------------------------
# MarginalEffectsResult — branches not covered
# ---------------------------------------------------------------------------


class TestMarginalEffectsResultBranches:
    """Cover uncovered branches of MarginalEffectsResult."""

    def test_summary_with_at_values(self, logit_panel_data):
        """Cover lines 158-161: summary prints at_values when present (MER)."""
        model = PooledLogit("y ~ x1 + x2 + x3", logit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        mer = compute_mer(result, at={"x1": 0.5, "x3": -1.0})
        assert mer.at_values is not None
        df = mer.summary()
        assert isinstance(df, pd.DataFrame)
        assert "MER" in df.columns

    def test_significance_stars_all_levels(self):
        """Cover lines 118-127: all significance star levels."""
        me = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
        # p < 0.001 -> ***, p < 0.01 -> **, p < 0.05 -> *, p < 0.1 -> ., else ""
        se = {"a": 0.01, "b": 0.1, "c": 0.35, "d": 0.55, "e": 10.0}
        result = MarginalEffectsResult(me, se, parent_result=None, me_type="ame")
        df = result.summary()
        # Just check that stars column exists and summary has correct shape
        assert "" in df.columns
        assert len(df) == 5

    def test_pvalues_not_series_branch(self):
        """Cover lines 76-77 and 132-136: pvalues when not pd.Series."""
        me = pd.Series({"x": 2.0})
        se = pd.Series({"x": 0.5})
        result = MarginalEffectsResult(me, se, parent_result=None, me_type="ame")
        pvals = result.pvalues
        assert isinstance(pvals, pd.Series)
        assert 0 < pvals["x"] < 1

    def test_repr(self):
        """Cover line 167-169: __repr__."""
        me = pd.Series({"x1": 0.1, "x2": 0.2})
        se = pd.Series({"x1": 0.01, "x2": 0.02})
        result = MarginalEffectsResult(me, se, parent_result=None, me_type="ame")
        s = repr(result)
        assert "MarginalEffectsResult" in s
        assert "n_effects=2" in s


# ---------------------------------------------------------------------------
# AME with varlist filtering
# ---------------------------------------------------------------------------


class TestAMEVarlist:
    """Cover branches for varlist parameter in compute_ame."""

    def test_ame_with_explicit_varlist(self, logit_panel_data):
        """Cover line 246: var not in exog_names -> continue."""
        model = PooledLogit("y ~ x1 + x2 + x3", logit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        # Include a non-existent variable name
        ame = compute_ame(result, varlist=["x1", "nonexistent"])
        assert "x1" in ame.marginal_effects.index
        assert "nonexistent" not in ame.marginal_effects.index

    def test_ame_probit_family(self, probit_panel_data):
        """Cover probit family branch in _ame_continuous (lines 305-307)."""
        model = PooledProbit("y ~ x1 + x2", probit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        ame = compute_ame(result, varlist=["x1"])
        assert "x1" in ame.marginal_effects.index
        assert np.isfinite(ame.marginal_effects["x1"])

    def test_ame_probit_binary_var(self, probit_panel_data):
        """Cover probit branch in _ame_discrete (lines 354-356)."""
        model = PooledProbit("y ~ x1 + x2", probit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        ame = compute_ame(result)
        # x2 is binary -> discrete difference with probit
        assert "x2" in ame.marginal_effects.index
        assert np.isfinite(ame.marginal_effects["x2"])


# ---------------------------------------------------------------------------
# MEM branches
# ---------------------------------------------------------------------------


class TestMEMBranches:
    """Cover branches in compute_mem."""

    def test_mem_probit_binary_discrete_diff(self, probit_panel_data):
        """Cover lines 457-462: binary var discrete diff in MEM with probit."""
        model = PooledProbit("y ~ x1 + x2", probit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        mem = compute_mem(result)
        # x2 is binary -> takes the discrete diff branch at means
        assert "x2" in mem.marginal_effects.index
        assert np.isfinite(mem.marginal_effects["x2"])

    def test_mem_with_varlist(self, logit_panel_data):
        """Cover varlist filtering in compute_mem."""
        model = PooledLogit("y ~ x1 + x2 + x3", logit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        mem = compute_mem(result, varlist=["x1", "missing_var"])
        assert "x1" in mem.marginal_effects.index
        assert "missing_var" not in mem.marginal_effects.index


# ---------------------------------------------------------------------------
# MER branches
# ---------------------------------------------------------------------------


class TestMERBranches:
    """Cover branches in compute_mer."""

    def test_mer_probit_binary_discrete_diff(self, probit_panel_data):
        """Cover lines 568-570: binary var discrete diff in MER with probit."""
        model = PooledProbit("y ~ x1 + x2", probit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        mer = compute_mer(result, at={"x1": 0.0})
        # x2 is binary -> discrete diff branch at representative
        assert "x2" in mer.marginal_effects.index
        assert np.isfinite(mer.marginal_effects["x2"])

    def test_mer_probit_continuous(self, probit_panel_data):
        """Cover lines 613-614: continuous var in MER with probit."""
        model = PooledProbit("y ~ x1 + x2", probit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        mer = compute_mer(result, at={"x1": 1.0, "x2": 1})
        assert "x1" in mer.marginal_effects.index
        assert np.isfinite(mer.marginal_effects["x1"])

    def test_mer_with_varlist(self, logit_panel_data):
        """Cover varlist filtering in compute_mer."""
        model = PooledLogit("y ~ x1 + x2 + x3", logit_panel_data, "entity", "time")
        result = model.fit(cov_type="nonrobust")
        mer = compute_mer(result, at={"x1": 0.0}, varlist=["x1", "ghost"])
        assert "x1" in mer.marginal_effects.index
        assert "ghost" not in mer.marginal_effects.index


# ---------------------------------------------------------------------------
# OrderedMarginalEffectsResult
# ---------------------------------------------------------------------------


class TestOrderedMarginalEffectsResult:
    """Cover OrderedMarginalEffectsResult (lines 630-784)."""

    @pytest.fixture
    def ordered_me_result(self):
        """Create an OrderedMarginalEffectsResult for testing."""
        me_data = pd.DataFrame(
            {"x1": [0.1, -0.05, -0.05], "x2": [-0.2, 0.1, 0.1]},
            index=["Category_0", "Category_1", "Category_2"],
        )
        se_data = pd.DataFrame(
            {"x1": [0.02, 0.01, 0.01], "x2": [0.03, 0.02, 0.02]},
            index=["Category_0", "Category_1", "Category_2"],
        )
        return OrderedMarginalEffectsResult(
            marginal_effects=me_data,
            std_errors=se_data,
            parent_result=None,
            me_type="ame",
        )

    def test_z_stats(self, ordered_me_result):
        """Cover line 666: z_stats property."""
        z = ordered_me_result.z_stats
        assert isinstance(z, pd.DataFrame)
        assert z.shape == (3, 2)
        # z = ME / SE
        expected_z = ordered_me_result.marginal_effects / ordered_me_result.std_errors
        pd.testing.assert_frame_equal(z, expected_z)

    def test_pvalues(self, ordered_me_result):
        """Cover lines 671-672: pvalues property."""
        pvals = ordered_me_result.pvalues
        assert isinstance(pvals, (pd.DataFrame, np.ndarray))
        # All p-values should be between 0 and 1
        assert np.all(np.array(pvals) >= 0)
        assert np.all(np.array(pvals) <= 1)

    def test_verify_sum_to_zero_true(self, ordered_me_result):
        """Cover lines 691-692: verify_sum_to_zero returns True."""
        assert ordered_me_result.verify_sum_to_zero(tol=1e-10)

    def test_verify_sum_to_zero_false(self):
        """Cover verify_sum_to_zero returning False branch."""
        me_data = pd.DataFrame(
            {"x1": [0.1, 0.2, 0.3]},
            index=["Cat_0", "Cat_1", "Cat_2"],
        )
        se_data = pd.DataFrame(
            {"x1": [0.01, 0.01, 0.01]},
            index=["Cat_0", "Cat_1", "Cat_2"],
        )
        result = OrderedMarginalEffectsResult(me_data, se_data, None, me_type="ame")
        assert not result.verify_sum_to_zero(tol=1e-10)

    def test_plot(self, ordered_me_result):
        """Cover lines 711-739: plot method."""
        ordered_me_result.plot("x1", figsize=(8, 5), include_ci=True, alpha=0.05)
        # Also test without CI
        ordered_me_result.plot("x2", include_ci=False)

    def test_summary(self, ordered_me_result):
        """Cover lines 755-778: summary method."""
        df = ordered_me_result.summary()
        assert isinstance(df, pd.DataFrame)
        # Should have columns for ME and SE for each variable
        assert "x1_ME" in df.columns
        assert "x1_SE" in df.columns

    def test_summary_with_at_values(self):
        """Cover lines 758-761: summary prints at_values."""
        me_data = pd.DataFrame({"x1": [0.1, -0.1]}, index=["Cat_0", "Cat_1"])
        se_data = pd.DataFrame({"x1": [0.01, 0.01]}, index=["Cat_0", "Cat_1"])
        result = OrderedMarginalEffectsResult(
            me_data,
            se_data,
            None,
            me_type="mem",
            at_values={"x1": 0.5},
        )
        df = result.summary()
        assert isinstance(df, pd.DataFrame)

    def test_summary_not_sum_to_zero(self):
        """Cover line 767: warning when ME don't sum to zero."""
        me_data = pd.DataFrame({"x1": [0.3, 0.3]}, index=["Cat_0", "Cat_1"])
        se_data = pd.DataFrame({"x1": [0.01, 0.01]}, index=["Cat_0", "Cat_1"])
        result = OrderedMarginalEffectsResult(me_data, se_data, None, me_type="ame")
        df = result.summary()
        assert isinstance(df, pd.DataFrame)

    def test_repr(self, ordered_me_result):
        """Cover lines 782-784: __repr__."""
        s = repr(ordered_me_result)
        assert "OrderedMarginalEffectsResult" in s
        assert "categories=3" in s
        assert "variables=2" in s


# ---------------------------------------------------------------------------
# compute_ordered_ame (lines 810-899)
# ---------------------------------------------------------------------------


class TestComputeOrderedAME:
    """Cover compute_ordered_ame function."""

    def test_ordered_ame_logit(self, ordered_logit_model):
        """Cover compute_ordered_ame with OrderedLogit (lines 852-861)."""
        result = compute_ordered_ame(ordered_logit_model)
        assert isinstance(result, OrderedMarginalEffectsResult)
        assert result.me_type == "AME"
        # Should have n_categories rows
        assert result.marginal_effects.shape[0] == ordered_logit_model.n_categories
        # Effects should sum to ~0 across categories
        assert result.verify_sum_to_zero(tol=1e-6)

    def test_ordered_ame_probit(self, ordered_probit_model):
        """Cover compute_ordered_ame with OrderedProbit (lines 863-872)."""
        result = compute_ordered_ame(ordered_probit_model)
        assert isinstance(result, OrderedMarginalEffectsResult)
        assert result.marginal_effects.shape[0] == ordered_probit_model.n_categories
        assert result.verify_sum_to_zero(tol=1e-6)

    def test_ordered_ame_with_varlist(self, ordered_logit_model):
        """Cover varlist filtering in compute_ordered_ame (line 837-838)."""
        result = compute_ordered_ame(ordered_logit_model, varlist=["x1"])
        assert "x1" in result.marginal_effects.columns
        assert result.marginal_effects.shape[1] == 1

    def test_ordered_ame_unfitted_model_raises(self):
        """Cover line 811: ValueError when model not fitted."""
        from panelbox.models.discrete import OrderedLogit

        model = OrderedLogit(
            endog=np.array([0, 1, 2, 0, 1, 2]),
            exog=np.random.randn(6, 2),
            groups=np.arange(6),
            n_categories=3,
        )
        with pytest.raises(ValueError, match="Model must be fitted"):
            compute_ordered_ame(model)

    def test_ordered_ame_has_bse(self, ordered_logit_model):
        """Cover lines 884-889: SE computation when model has bse."""
        result = compute_ordered_ame(ordered_logit_model)
        # bse exists -> SEs should be finite
        if hasattr(ordered_logit_model, "bse") and ordered_logit_model.bse is not None:
            assert np.all(np.isfinite(result.std_errors.values))

    def test_ordered_ame_no_bse(self):
        """Cover lines 891-892: SE = nan when model has no bse."""
        from panelbox.models.discrete import OrderedLogit

        np.random.seed(100)
        N, K, J = 200, 2, 3
        X = np.random.randn(N, K)
        beta = np.array([0.4, -0.2])
        cutpoints = np.array([-0.5, 0.5])
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = X[i] @ beta + np.random.logistic(0, 1)
            if y_star <= cutpoints[0]:
                y[i] = 0
            elif y_star <= cutpoints[1]:
                y[i] = 1
            else:
                y[i] = 2
        groups = np.arange(N)
        model = OrderedLogit(endog=y, exog=X, groups=groups, n_categories=J)
        model.exog_names = ["x1", "x2"]
        model.fit(maxiter=500)
        # Remove bse to trigger the else branch
        original_bse = model.bse
        del model.bse
        result = compute_ordered_ame(model)
        assert np.all(np.isnan(result.std_errors.values))
        # Restore
        model.bse = original_bse


# ---------------------------------------------------------------------------
# compute_ordered_mem (lines 920-991)
# ---------------------------------------------------------------------------


class TestComputeOrderedMEM:
    """Cover compute_ordered_mem function."""

    def test_ordered_mem_logit(self, ordered_logit_model):
        """Cover compute_ordered_mem with OrderedLogit (lines 959-965)."""
        result = compute_ordered_mem(ordered_logit_model)
        assert isinstance(result, OrderedMarginalEffectsResult)
        assert result.me_type == "MEM"
        assert result.at_values is not None
        assert result.marginal_effects.shape[0] == ordered_logit_model.n_categories

    def test_ordered_mem_probit(self, ordered_probit_model):
        """Cover compute_ordered_mem with OrderedProbit (lines 967-973)."""
        result = compute_ordered_mem(ordered_probit_model)
        assert isinstance(result, OrderedMarginalEffectsResult)
        assert result.me_type == "MEM"
        assert result.at_values is not None

    def test_ordered_mem_with_varlist(self, ordered_logit_model):
        """Cover varlist filtering."""
        result = compute_ordered_mem(ordered_logit_model, varlist=["x2"])
        assert "x2" in result.marginal_effects.columns
        assert result.marginal_effects.shape[1] == 1

    def test_ordered_mem_unfitted_model_raises(self):
        """Cover line 921: ValueError when model not fitted."""
        from panelbox.models.discrete import OrderedLogit

        model = OrderedLogit(
            endog=np.array([0, 1, 2, 0, 1, 2]),
            exog=np.random.randn(6, 2),
            groups=np.arange(6),
            n_categories=3,
        )
        with pytest.raises(ValueError, match="Model must be fitted"):
            compute_ordered_mem(model)

    def test_ordered_mem_has_bse(self, ordered_logit_model):
        """Cover lines 978-979: SE computation when model has bse."""
        result = compute_ordered_mem(ordered_logit_model)
        if hasattr(ordered_logit_model, "bse") and ordered_logit_model.bse is not None:
            assert np.all(np.isfinite(result.std_errors.values))

    def test_ordered_mem_no_bse(self):
        """Cover lines 980-981: SE = nan when model has no bse."""
        from panelbox.models.discrete import OrderedLogit

        np.random.seed(101)
        N, K, J = 200, 2, 3
        X = np.random.randn(N, K)
        beta = np.array([0.3, -0.4])
        cutpoints = np.array([-0.5, 0.5])
        y = np.zeros(N, dtype=int)
        for i in range(N):
            y_star = X[i] @ beta + np.random.logistic(0, 1)
            if y_star <= cutpoints[0]:
                y[i] = 0
            elif y_star <= cutpoints[1]:
                y[i] = 1
            else:
                y[i] = 2
        groups = np.arange(N)
        model = OrderedLogit(endog=y, exog=X, groups=groups, n_categories=J)
        model.exog_names = ["x1", "x2"]
        model.fit(maxiter=500)
        del model.bse
        result = compute_ordered_mem(model)
        assert np.all(np.isnan(result.std_errors.values))


# ---------------------------------------------------------------------------
# Import error branch for ordered models (lines 26-28)
# ---------------------------------------------------------------------------


class TestImportBranches:
    """Test import-related branches."""

    def test_ordered_imports_exist(self):
        """Verify that OrderedLogit and OrderedProbit can be imported."""
        from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit

        assert OrderedLogit is not None
        assert OrderedProbit is not None
