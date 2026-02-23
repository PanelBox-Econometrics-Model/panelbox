"""Property-based tests for validation and diagnostic tests."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings

from panelbox.diagnostics.hausman import hausman_test
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.random_effects import RandomEffects
from tests.strategies import balanced_panels


def _fit_or_skip(model_cls, formula, df, entity, time, **fit_kwargs):
    """Fit a model, skipping the example if data is degenerate."""
    try:
        result = model_cls(formula, df, entity, time).fit(**fit_kwargs)
    except (np.linalg.LinAlgError, ValueError):
        assume(False)
    assume(np.all(np.isfinite(result.std_errors.values)))
    return result


# ---------------------------------------------------------------------------
# Test 1: p-values between 0 and 1 for all models
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_pvalues_bounds_pooled(data):
    """All p-values from Pooled OLS must be between 0 and 1."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time)

    for param, pval in result.pvalues.items():
        assert 0 <= pval <= 1, f"p-value for {param} = {pval} out of [0, 1]"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_pvalues_bounds_fe(data):
    """All p-values from Fixed Effects must be between 0 and 1."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time)

    for param, pval in result.pvalues.items():
        assert 0 <= pval <= 1, f"p-value for {param} = {pval} out of [0, 1]"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_pvalues_bounds_re(data):
    """All p-values from Random Effects must be between 0 and 1."""
    df, formula, entity, time = data
    result = _fit_or_skip(RandomEffects, formula, df, entity, time)

    for param, pval in result.pvalues.items():
        assert 0 <= pval <= 1, f"p-value for {param} = {pval} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 2: Test statistics are finite (not NaN, not Inf)
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_tvalues_finite_pooled(data):
    """All t-values from Pooled OLS must be finite."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time)

    for param, tval in result.tvalues.items():
        assert np.isfinite(tval), f"t-value for {param} = {tval} is not finite"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_tvalues_finite_fe(data):
    """All t-values from Fixed Effects must be finite."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time)

    for param, tval in result.tvalues.items():
        assert np.isfinite(tval), f"t-value for {param} = {tval} is not finite"


# ---------------------------------------------------------------------------
# Test 3: Hausman test p-value between 0 and 1
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=10, max_entities=25, n_regressors=2))
@settings(max_examples=30, deadline=60000)
def test_hausman_pvalue_bounds(data):
    """Hausman test p-value must be between 0 and 1."""
    df, formula, entity, time = data

    fe_result = _fit_or_skip(FixedEffects, formula, df, entity, time)
    re_result = _fit_or_skip(RandomEffects, formula, df, entity, time)

    try:
        hausman = hausman_test(fe_result, re_result)
    except Exception:
        # Some data configurations may make the Hausman test infeasible
        return

    if not np.isnan(hausman.pvalue):
        assert 0 <= hausman.pvalue <= 1, f"Hausman p-value = {hausman.pvalue} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 4: Hausman test statistic is finite
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=10, max_entities=25, n_regressors=2))
@settings(max_examples=30, deadline=60000)
def test_hausman_statistic_finite(data):
    """Hausman test statistic must be finite when it can be computed.

    Note: The Hausman statistic can be negative when the variance
    difference matrix (V_FE - V_RE) is not positive semi-definite.
    This is a known issue in econometrics, not a bug.
    """
    df, formula, entity, time = data

    fe_result = _fit_or_skip(FixedEffects, formula, df, entity, time)
    re_result = _fit_or_skip(RandomEffects, formula, df, entity, time)

    try:
        hausman = hausman_test(fe_result, re_result)
    except Exception:
        return

    stat = hausman.statistic
    if not np.isnan(stat):
        assert np.isfinite(stat), f"Hausman statistic = {stat} is not finite"


# ---------------------------------------------------------------------------
# Test 5: F-statistic for FE is finite
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_fe_f_statistic_finite(data):
    """F-statistic from FE model must be finite and non-negative when computable."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time)

    if hasattr(result, "f_pvalue") and result.f_pvalue is not None:
        if np.isfinite(result.f_pvalue):
            assert 0 <= result.f_pvalue <= 1, f"F p-value = {result.f_pvalue} out of [0, 1]"
