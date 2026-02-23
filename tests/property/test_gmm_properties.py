"""Property-based tests for GMM estimators (Difference GMM, System GMM)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from panelbox.gmm.difference_gmm import DifferenceGMM
from panelbox.gmm.system_gmm import SystemGMM


def _make_dynamic_panel(n_entities, n_periods, seed=None):
    """Generate a simple dynamic panel for GMM testing.

    Creates y_{it} = rho * y_{i,t-1} + beta * x_{it} + alpha_i + eps_{it}
    with known DGP parameters.
    """
    rng = np.random.RandomState(seed)

    alpha = rng.randn(n_entities) * 0.5
    rho = 0.5
    beta = 1.0

    records = []
    for i in range(n_entities):
        y_prev = rng.randn() * 2  # Initial y
        for t in range(n_periods):
            x = rng.randn() * 3
            eps = rng.randn() * 0.5
            y = rho * y_prev + beta * x + alpha[i] + eps
            records.append({"id": i, "year": t, "y": y, "x1": x})
            y_prev = y

    return pd.DataFrame(records)


@st.composite
def dynamic_panel_data(draw, min_n=10, max_n=50, min_t=5, max_t=10):
    """Hypothesis strategy for generating dynamic panels."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    t = draw(st.integers(min_value=min_t, max_value=max_t))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    return _make_dynamic_panel(n, t, seed=seed)


# ---------------------------------------------------------------------------
# Test 1: Hansen J statistic >= 0
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_hansen_j_nonnegative_diff_gmm(df):
    """Hansen J statistic must be non-negative for Difference GMM."""
    model = DifferenceGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    j_stat = result.hansen_j.statistic
    # Hansen J can be NaN when df <= 0 (under-identified)
    if not np.isnan(j_stat):
        assert j_stat >= -1e-10, f"Hansen J = {j_stat} < 0"


@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_hansen_j_nonnegative_sys_gmm(df):
    """Hansen J statistic must be non-negative for System GMM."""
    model = SystemGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    j_stat = result.hansen_j.statistic
    if not np.isnan(j_stat):
        assert j_stat >= -1e-10, f"Hansen J = {j_stat} < 0"


# ---------------------------------------------------------------------------
# Test 2: Hansen J p-value between 0 and 1
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_hansen_j_pvalue_bounds(df):
    """Hansen J p-value must be between 0 and 1."""
    model = DifferenceGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    p = result.hansen_j.pvalue
    if not np.isnan(p):
        assert 0 <= p <= 1, f"Hansen J p-value = {p} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 3: AR test p-values between 0 and 1
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_ar_test_pvalues_bounds(df):
    """AR(1) and AR(2) test p-values must be between 0 and 1."""
    model = DifferenceGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    for test_name, test_result in [("AR(1)", result.ar1_test), ("AR(2)", result.ar2_test)]:
        if test_result is not None:
            p = test_result.pvalue
            if not np.isnan(p):
                assert 0 <= p <= 1, f"{test_name} p-value = {p} out of [0, 1]"


# ---------------------------------------------------------------------------
# Test 4: SE always positive for GMM
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_gmm_se_positive(df):
    """Standard errors must be positive for GMM estimators."""
    model = DifferenceGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    se = result.std_errors
    assert (se > 0).all(), f"Non-positive SE found: {se}"


# ---------------------------------------------------------------------------
# Test 5: Number of instruments reported correctly
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_gmm_n_instruments_positive(df):
    """Number of instruments must be positive and >= number of parameters."""
    model = DifferenceGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    assert result.n_instruments > 0, "n_instruments should be > 0"
    assert result.n_instruments >= result.n_params, (
        f"n_instruments ({result.n_instruments}) < n_params ({result.n_params})"
    )


# ---------------------------------------------------------------------------
# Test 6: Covariance matrix is symmetric
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(df=dynamic_panel_data(min_n=15, max_n=40, min_t=6, max_t=8))
@settings(max_examples=30, deadline=60000)
def test_gmm_vcov_symmetric(df):
    """GMM covariance matrix must be symmetric."""
    model = DifferenceGMM(
        data=df,
        dep_var="y",
        lags=1,
        id_var="id",
        time_var="year",
        exog_vars=["x1"],
        time_dummies=False,
        collapse=True,
        two_step=True,
        robust=True,
    )
    result = model.fit()

    vcov = result.vcov
    np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
