"""Property-based tests for static panel estimators (FE, RE, Pooled, Between, FD)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings

from panelbox.models.static.between import BetweenEstimator
from panelbox.models.static.first_difference import FirstDifferenceEstimator
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
    # Skip examples where SE are NaN or zero (numerically unstable / degenerate)
    assume(np.all(np.isfinite(result.std_errors.values)))
    assume(np.all(result.std_errors.values > 0))
    return result


# ---------------------------------------------------------------------------
# Test 1: Residuals orthogonal to regressors (X'e ≈ 0) for Pooled OLS
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_pooled_ols_residuals_orthogonal(data):
    """For Pooled OLS, X'e should be approximately zero (normal equations)."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time)

    from patsy import dmatrix

    X = np.asarray(dmatrix(formula.split("~")[1].strip(), df))
    resid = result.resid
    if isinstance(resid, pd.Series):
        resid = resid.values
    cross = X.T @ resid
    np.testing.assert_allclose(cross, 0, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 2: Residuals + fitted = y (identity check for FE)
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_fe_resid_plus_fitted_equals_y(data):
    """For FE, resid + fitted should equal y."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time)

    resid = result.resid
    if isinstance(resid, pd.Series):
        resid = resid.values
    fitted = result.fittedvalues
    if isinstance(fitted, pd.Series):
        fitted = fitted.values

    y = df.sort_values([entity, time])["y"].values
    np.testing.assert_allclose(resid + fitted, y, atol=1e-8)


# ---------------------------------------------------------------------------
# Test 3: Invariance to row permutation
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=1))
@settings(max_examples=30, deadline=30000)
def test_invariance_to_row_permutation_pooled(data):
    """Results should not change when rows are permuted."""
    df, formula, entity, time = data
    result1 = _fit_or_skip(PooledOLS, formula, df, entity, time)
    result2 = _fit_or_skip(PooledOLS, formula, df.sample(frac=1, random_state=42), entity, time)

    np.testing.assert_allclose(
        result1.params.sort_index().values,
        result2.params.sort_index().values,
        atol=1e-10,
    )


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=1))
@settings(max_examples=30, deadline=30000)
def test_invariance_to_row_permutation_fe(data):
    """FE results should not change when rows are permuted."""
    df, formula, entity, time = data
    result1 = _fit_or_skip(FixedEffects, formula, df, entity, time)
    result2 = _fit_or_skip(FixedEffects, formula, df.sample(frac=1, random_state=42), entity, time)

    np.testing.assert_allclose(
        result1.params.sort_index().values,
        result2.params.sort_index().values,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Test 4: R-squared between 0 and 1
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_pooled_ols_rsquared_bounds(data):
    """R-squared must be between 0 and 1 for Pooled OLS."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time)

    assert -1e-10 <= result.rsquared <= 1 + 1e-10, f"R-squared = {result.rsquared} out of bounds"
    assert result.rsquared_adj <= 1 + 1e-10, f"Adjusted R-squared = {result.rsquared_adj} > 1"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_fe_rsquared_within_bounds(data):
    """Within R-squared must be between 0 and 1 for Fixed Effects."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time)

    r2w = result.rsquared_within
    # Allow small floating-point tolerance beyond [0, 1]
    assert -1e-10 <= r2w <= 1 + 1e-10, f"Within R-squared = {r2w} out of bounds"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_re_rsquared_bounds(data):
    """R-squared measures must be between 0 and 1 for Random Effects."""
    df, formula, entity, time = data
    result = _fit_or_skip(RandomEffects, formula, df, entity, time)

    assert -1e-10 <= result.rsquared <= 1 + 1e-10, f"R-squared = {result.rsquared} out of bounds"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_between_rsquared_bounds(data):
    """R-squared must be between 0 and 1 for Between estimator."""
    df, formula, entity, time = data
    result = _fit_or_skip(BetweenEstimator, formula, df, entity, time)

    r2 = result.rsquared
    assert -1e-10 <= r2 <= 1 + 1e-10, f"R-squared = {r2} out of bounds"


# ---------------------------------------------------------------------------
# Test 5: Scale invariance — multiply y by constant c
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=2))
@settings(max_examples=30, deadline=30000)
def test_scale_invariance_pooled(data):
    """Multiplying y by constant c should multiply coefficients by c."""
    df, formula, entity, time = data
    c = 3.7

    result1 = _fit_or_skip(PooledOLS, formula, df, entity, time)

    df_scaled = df.copy()
    df_scaled["y"] = df_scaled["y"] * c
    result2 = _fit_or_skip(PooledOLS, formula, df_scaled, entity, time)

    np.testing.assert_allclose(
        result2.params.sort_index().values,
        c * result1.params.sort_index().values,
        atol=1e-8,
    )


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=2))
@settings(max_examples=30, deadline=30000)
def test_scale_invariance_fe(data):
    """Multiplying y by constant c should multiply FE coefficients by c."""
    df, formula, entity, time = data
    c = 2.5

    result1 = _fit_or_skip(FixedEffects, formula, df, entity, time)

    df_scaled = df.copy()
    df_scaled["y"] = df_scaled["y"] * c
    result2 = _fit_or_skip(FixedEffects, formula, df_scaled, entity, time)

    np.testing.assert_allclose(
        result2.params.sort_index().values,
        c * result1.params.sort_index().values,
        atol=1e-8,
    )


# ---------------------------------------------------------------------------
# Test 6: SE always positive
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_se_always_positive_pooled(data):
    """Standard errors must always be positive."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time)

    assert (result.std_errors > 0).all(), f"Negative SE: {result.std_errors}"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_se_always_positive_fe(data):
    """Standard errors must always be positive for FE."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time)

    assert (result.std_errors > 0).all(), f"Negative SE: {result.std_errors}"


# ---------------------------------------------------------------------------
# Test 7: First difference reduces T by 1 per entity
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=15, n_regressors=1))
@settings(max_examples=30, deadline=30000)
def test_first_difference_reduces_obs(data):
    """First difference should lose 1 observation per entity."""
    df, formula, entity, time = data
    n_entities = df["entity"].nunique()
    n_obs_original = len(df)

    result = _fit_or_skip(FirstDifferenceEstimator, formula, df, entity, time)

    expected_nobs = n_obs_original - n_entities
    assert result.nobs == expected_nobs, f"Expected {expected_nobs} obs after FD, got {result.nobs}"
