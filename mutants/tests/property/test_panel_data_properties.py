"""Property-based tests for PanelData container."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings

from panelbox.core.panel_data import PanelData
from tests.strategies import balanced_panels, unbalanced_panels


# ---------------------------------------------------------------------------
# Test 1: Demeaning is idempotent — demean(demean(X)) == demean(X)
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=3, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_demeaning_idempotent(data):
    """Applying entity demeaning twice should give the same result as once."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    vars_to_demean = [c for c in df.columns if c.startswith("x") or c == "y"]

    demeaned1 = panel.demeaning(variables=vars_to_demean, method="entity")

    # Create a new PanelData from the demeaned result and demean again
    panel2 = PanelData(demeaned1, entity_col=entity, time_col=time)
    demeaned2 = panel2.demeaning(variables=vars_to_demean, method="entity")

    for var in vars_to_demean:
        np.testing.assert_allclose(
            demeaned1[var].values,
            demeaned2[var].values,
            atol=1e-10,
            err_msg=f"Demeaning not idempotent for variable {var}",
        )


@pytest.mark.property
@given(data=balanced_panels(min_entities=3, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_demeaning_idempotent_time(data):
    """Applying time demeaning twice should give the same result as once."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    vars_to_demean = [c for c in df.columns if c.startswith("x") or c == "y"]

    demeaned1 = panel.demeaning(variables=vars_to_demean, method="time")

    panel2 = PanelData(demeaned1, entity_col=entity, time_col=time)
    demeaned2 = panel2.demeaning(variables=vars_to_demean, method="time")

    for var in vars_to_demean:
        np.testing.assert_allclose(
            demeaned1[var].values,
            demeaned2[var].values,
            atol=1e-10,
            err_msg=f"Time demeaning not idempotent for variable {var}",
        )


# ---------------------------------------------------------------------------
# Test 2: First difference reduces T by 1 per entity
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=3, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_first_difference_reduces_obs(data):
    """First difference should drop 1 observation per entity."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    n_entities = panel.n_entities
    n_obs_original = panel.n_obs

    vars_to_diff = [c for c in df.columns if c.startswith("x") or c == "y"]
    diff_df = panel.first_difference(variables=vars_to_diff)

    expected = n_obs_original - n_entities
    assert len(diff_df) == expected, f"Expected {expected} obs after FD, got {len(diff_df)}"


@pytest.mark.property
@given(data=unbalanced_panels(min_entities=3, max_entities=10))
@settings(max_examples=30, deadline=30000)
def test_first_difference_unbalanced(data):
    """First difference on unbalanced panels drops 1 obs per entity."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    n_entities = panel.n_entities
    n_obs_original = panel.n_obs

    vars_to_diff = ["y", "x1"]
    diff_df = panel.first_difference(variables=vars_to_diff)

    expected = n_obs_original - n_entities
    assert len(diff_df) == expected, f"Expected {expected} obs after FD, got {len(diff_df)}"


# ---------------------------------------------------------------------------
# Test 3: Roundtrip — PanelData preserves data content
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=3, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_roundtrip_panel_data(data):
    """PanelData should preserve all data columns and values."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    # The internal data is sorted by entity+time, so sort the original too
    df_sorted = df.sort_values([entity, time]).reset_index(drop=True)
    result = panel.data.reset_index(drop=True)

    # Same columns
    assert set(result.columns) == set(df_sorted.columns)

    # Same values (after sorting)
    for col in df_sorted.columns:
        np.testing.assert_allclose(
            result[col].values.astype(float),
            df_sorted[col].values.astype(float),
            atol=1e-15,
            err_msg=f"Data mismatch in column {col}",
        )


# ---------------------------------------------------------------------------
# Test 4: Panel structure properties
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=3, max_entities=15, n_regressors=1))
@settings(max_examples=50, deadline=30000)
def test_balanced_panel_structure(data):
    """Balanced panels must report correct structure."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    assert panel.is_balanced
    assert panel.n_obs == panel.n_entities * panel.n_periods
    assert panel.n_entities == df[entity].nunique()
    assert panel.n_periods == df[time].nunique()


# ---------------------------------------------------------------------------
# Test 5: Demeaned data has zero mean within groups
# ---------------------------------------------------------------------------
@pytest.mark.property
@given(data=balanced_panels(min_entities=3, max_entities=15, n_regressors=2))
@settings(max_examples=50, deadline=30000)
def test_entity_demeaned_has_zero_group_means(data):
    """After entity demeaning, group means should be zero."""
    df, _formula, entity, time = data
    panel = PanelData(df, entity_col=entity, time_col=time)

    vars_to_demean = [c for c in df.columns if c.startswith("x") or c == "y"]
    demeaned = panel.demeaning(variables=vars_to_demean, method="entity")

    # Check that entity means of demeaned variables are ~0
    group_means = demeaned.groupby(entity)[vars_to_demean].mean()
    for var in vars_to_demean:
        np.testing.assert_allclose(
            group_means[var].values,
            0,
            atol=1e-10,
            err_msg=f"Entity mean not zero for {var} after demeaning",
        )
