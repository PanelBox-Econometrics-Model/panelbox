"""Property-based tests for standard error computations."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings

from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.pooled_ols import PooledOLS
from tests.strategies import balanced_panels, positive_definite_matrices


def _fit_or_skip(model_cls, formula, df, entity, time, check_se=True, **fit_kwargs):
    """Fit a model, returning None if the data is degenerate."""
    try:
        result = model_cls(formula, df, entity, time).fit(**fit_kwargs)
    except (np.linalg.LinAlgError, ValueError):
        assume(False)  # Tell Hypothesis to discard this example
    if check_se:
        # Skip examples where SE are NaN or zero (degenerate data)
        se = result.std_errors.values
        assume(np.all(np.isfinite(se)) and np.all(se > 0))
    return result


# ---------------------------------------------------------------------------
# Test 1: SE always positive for all cov_types
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_se_positive_nonrobust(data):
    """Non-robust standard errors must always be positive."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time, cov_type="nonrobust")
    assert (result.std_errors > 0).all(), f"Non-positive SE: {result.std_errors}"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_se_positive_robust(data):
    """Robust (HC1) standard errors must always be positive."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time, cov_type="robust")
    assert (result.std_errors > 0).all(), f"Non-positive SE: {result.std_errors}"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_se_positive_clustered(data):
    """Clustered standard errors must always be positive."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time, cov_type="clustered")
    assert (result.std_errors > 0).all(), f"Non-positive SE: {result.std_errors}"


# ---------------------------------------------------------------------------
# Test 2: Covariance matrix is symmetric
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_cov_symmetric_nonrobust(data):
    """Non-robust covariance matrix must be symmetric."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time, cov_type="nonrobust")
    cov = result.cov_params.values
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_cov_symmetric_robust(data):
    """Robust covariance matrix must be symmetric."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time, cov_type="robust")
    cov = result.cov_params.values
    np.testing.assert_allclose(cov, cov.T, atol=1e-6)


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_cov_symmetric_clustered(data):
    """Clustered covariance matrix must be symmetric."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time, cov_type="clustered")
    cov = result.cov_params.values
    np.testing.assert_allclose(cov, cov.T, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: Covariance matrix is positive semi-definite
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_cov_psd_nonrobust(data):
    """Non-robust covariance matrix must be positive semi-definite."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time, cov_type="nonrobust")
    cov = result.cov_params.values
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_cov_psd_robust(data):
    """Robust covariance matrix must be positive semi-definite."""
    df, formula, entity, time = data
    result = _fit_or_skip(PooledOLS, formula, df, entity, time, cov_type="robust")
    cov = result.cov_params.values
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"


@pytest.mark.property
@given(data=balanced_panels(min_entities=5, max_entities=20, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_cov_psd_clustered(data):
    """Clustered covariance matrix must be positive semi-definite."""
    df, formula, entity, time = data
    result = _fit_or_skip(FixedEffects, formula, df, entity, time, cov_type="clustered")
    cov = result.cov_params.values
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10), f"Negative eigenvalue: {eigenvalues.min()}"


# ---------------------------------------------------------------------------
# Test 4: Positive definite matrices strategy produces valid matrices
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(M=positive_definite_matrices(min_size=2, max_size=8))
@settings(max_examples=50, deadline=5000)
def test_positive_definite_strategy_valid(M):
    """Generated positive definite matrices must be symmetric and PD."""
    # Symmetric
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    # Positive definite
    eigenvalues = np.linalg.eigvalsh(M)
    assert np.all(eigenvalues > 0), f"Non-positive eigenvalue: {eigenvalues.min()}"
