"""
Tests for panelbox.utils.data module.

Covers uncovered lines 65-108 (check_panel_data branches) and 169-185 (panel_to_dict).
"""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from panelbox.utils.data import check_panel_data, panel_to_dict


class TestCheckPanelDataConversions:
    """Test check_panel_data input conversion branches."""

    def test_entity_id_as_series(self):
        """Line 65: entity_id pd.Series is converted to ndarray."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = pd.Series([1, 1, 2])
        time_id = np.array([1, 2, 1])

        _y_out, _X_out, eid_out, _tid_out, _w_out = check_panel_data(
            y, X, entity_id=entity_id, time_id=time_id
        )

        assert isinstance(eid_out, np.ndarray)
        assert_allclose(eid_out, [1, 1, 2])

    def test_time_id_as_series(self):
        """Line 67: time_id pd.Series is converted to ndarray."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = pd.Series([10, 20, 10])

        _y_out, _X_out, _eid_out, tid_out, _w_out = check_panel_data(
            y, X, entity_id=entity_id, time_id=time_id
        )

        assert isinstance(tid_out, np.ndarray)
        assert_allclose(tid_out, [10, 20, 10])

    def test_weights_as_series(self):
        """Lines 68-69: weights pd.Series is converted to ndarray."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2, 1])
        weights = pd.Series([0.5, 1.0, 1.5])

        _y_out, _X_out, _eid_out, _tid_out, w_out = check_panel_data(
            y, X, entity_id=entity_id, time_id=time_id, weights=weights
        )

        assert isinstance(w_out, np.ndarray)
        assert_allclose(w_out, [0.5, 1.0, 1.5])

    def test_y_multidimensional_flattened(self):
        """Lines 76-77: y with ndim > 1 is flattened."""
        y = np.array([[1.0], [2.0], [3.0]])  # 2D
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2, 1])

        y_out, _X_out, _eid_out, _tid_out, _w_out = check_panel_data(
            y, X, entity_id=entity_id, time_id=time_id
        )

        assert y_out.ndim == 1
        assert_allclose(y_out, [1.0, 2.0, 3.0])

    def test_y_as_dataframe(self):
        """Line 60-61: y as DataFrame is converted."""
        y = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2, 1])

        y_out, _X_out, _eid_out, _tid_out, _w_out = check_panel_data(
            y, X, entity_id=entity_id, time_id=time_id
        )

        assert isinstance(y_out, np.ndarray)
        assert y_out.ndim == 1
        assert_allclose(y_out, [1.0, 2.0, 3.0])


class TestCheckPanelDataDefaults:
    """Test check_panel_data default entity/time ID creation."""

    def test_entity_id_none_creates_default(self):
        """Lines 85-87: entity_id=None creates zeros."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])

        _y_out, _X_out, eid_out, _tid_out, _w_out = check_panel_data(
            y, X, entity_id=None, time_id=np.array([1, 2, 3])
        )

        assert_allclose(eid_out, [0, 0, 0])
        assert eid_out.dtype == int

    def test_time_id_none_creates_default(self):
        """Lines 91-93: time_id=None creates sequential integers."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])

        _y_out, _X_out, _eid_out, tid_out, _w_out = check_panel_data(
            y, X, entity_id=np.array([1, 1, 1]), time_id=None
        )

        assert_allclose(tid_out, [0, 1, 2])
        assert tid_out.dtype == int

    def test_both_ids_none(self):
        """Both entity_id and time_id default when None."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([[1.0], [2.0], [3.0], [4.0]])

        _y_out, _X_out, eid_out, tid_out, _w_out = check_panel_data(y, X)

        assert_allclose(eid_out, [0, 0, 0, 0])
        assert_allclose(tid_out, [0, 1, 2, 3])


class TestCheckPanelDataValidation:
    """Test check_panel_data dimension validation errors."""

    def test_x_y_dimension_mismatch(self):
        """Lines 81-82: X and y dimension mismatch raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0]])  # Only 2 rows

        with pytest.raises(ValueError, match="X has 2 observations but y has 3"):
            check_panel_data(y, X)

    def test_entity_id_length_mismatch(self):
        """Lines 98-99: entity_id length mismatch raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 2])  # Only 2 elements

        with pytest.raises(ValueError, match="entity_id has 2 elements but y has 3"):
            check_panel_data(y, X, entity_id=entity_id)

    def test_time_id_length_mismatch(self):
        """Lines 100-101: time_id length mismatch raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2])  # Only 2 elements

        with pytest.raises(ValueError, match="time_id has 2 elements but y has 3"):
            check_panel_data(y, X, entity_id=entity_id, time_id=time_id)

    def test_weights_length_mismatch(self):
        """Lines 105-106: weights length mismatch raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2, 1])
        weights = np.array([1.0, 1.0])  # Only 2 elements

        with pytest.raises(ValueError, match="weights has 2 elements but y has 3"):
            check_panel_data(y, X, entity_id=entity_id, time_id=time_id, weights=weights)

    def test_negative_weights(self):
        """Lines 107-108: negative weights raises ValueError."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2, 1])
        weights = np.array([1.0, -0.5, 1.0])

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            check_panel_data(y, X, entity_id=entity_id, time_id=time_id, weights=weights)

    def test_zero_weights_allowed(self):
        """Weights of zero should be allowed (non-negative)."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])
        entity_id = np.array([1, 1, 2])
        time_id = np.array([1, 2, 1])
        weights = np.array([1.0, 0.0, 1.0])

        _y_out, _X_out, _eid_out, _tid_out, w_out = check_panel_data(
            y, X, entity_id=entity_id, time_id=time_id, weights=weights
        )

        assert_allclose(w_out, [1.0, 0.0, 1.0])

    def test_weights_none_returns_none(self):
        """Lines 103+: weights=None returns None."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[1.0], [2.0], [3.0]])

        _y_out, _X_out, _eid_out, _tid_out, w_out = check_panel_data(y, X)

        assert w_out is None


class TestPanelToDict:
    """Test panel_to_dict function (lines 169-185)."""

    def test_basic_conversion(self):
        """Test basic conversion to dict format."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        entity_id = np.array([1, 1, 2, 2])
        time_id = np.array([1, 2, 1, 2])

        result = panel_to_dict(y, X, entity_id, time_id)

        assert set(result.keys()) == {1, 2}
        assert_allclose(result[1]["y"], [1.0, 2.0])
        assert_allclose(result[2]["y"], [3.0, 4.0])
        assert_allclose(result[1]["X"], [[10.0], [20.0]])
        assert_allclose(result[2]["X"], [[30.0], [40.0]])
        assert_allclose(result[1]["time"], [1, 2])
        assert_allclose(result[2]["time"], [1, 2])

    def test_with_weights(self):
        """Lines 175-176: weights are included when provided."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        X = np.array([[10.0], [20.0], [30.0], [40.0]])
        entity_id = np.array([1, 1, 2, 2])
        time_id = np.array([1, 2, 1, 2])
        weights = np.array([0.5, 1.0, 1.5, 2.0])

        result = panel_to_dict(y, X, entity_id, time_id, weights=weights)

        assert "weights" in result[1]
        assert "weights" in result[2]
        assert_allclose(result[1]["weights"], [0.5, 1.0])
        assert_allclose(result[2]["weights"], [1.5, 2.0])

    def test_without_weights(self):
        """Weights key should be absent when weights=None."""
        y = np.array([1.0, 2.0])
        X = np.array([[10.0], [20.0]])
        entity_id = np.array([1, 1])
        time_id = np.array([1, 2])

        result = panel_to_dict(y, X, entity_id, time_id, weights=None)

        assert "weights" not in result[1]

    def test_sorting_by_time(self):
        """Lines 179-181: data is sorted by time within each entity."""
        # Provide data out of time order
        y = np.array([2.0, 1.0, 4.0, 3.0])
        X = np.array([[20.0], [10.0], [40.0], [30.0]])
        entity_id = np.array([1, 1, 2, 2])
        time_id = np.array([2, 1, 2, 1])

        result = panel_to_dict(y, X, entity_id, time_id)

        # Should be sorted by time
        assert_allclose(result[1]["time"], [1, 2])
        assert_allclose(result[1]["y"], [1.0, 2.0])
        assert_allclose(result[1]["X"], [[10.0], [20.0]])

        assert_allclose(result[2]["time"], [1, 2])
        assert_allclose(result[2]["y"], [3.0, 4.0])
        assert_allclose(result[2]["X"], [[30.0], [40.0]])

    def test_sorting_with_weights(self):
        """Sorting also applies to weights."""
        y = np.array([2.0, 1.0])
        X = np.array([[20.0], [10.0]])
        entity_id = np.array([1, 1])
        time_id = np.array([2, 1])
        weights = np.array([0.8, 0.2])

        result = panel_to_dict(y, X, entity_id, time_id, weights=weights)

        assert_allclose(result[1]["weights"], [0.2, 0.8])  # Sorted by time

    def test_multiple_entities(self):
        """Test with three entities."""
        n = 9
        y = np.arange(1.0, n + 1)
        X = np.arange(10.0, 10 + n).reshape(-1, 1)
        entity_id = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        time_id = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])

        result = panel_to_dict(y, X, entity_id, time_id)

        assert len(result) == 3
        assert_allclose(result[1]["y"], [1.0, 2.0, 3.0])
        assert_allclose(result[3]["y"], [7.0, 8.0, 9.0])

    def test_single_entity(self):
        """Test with a single entity."""
        y = np.array([1.0, 2.0, 3.0])
        X = np.array([[10.0], [20.0], [30.0]])
        entity_id = np.array([1, 1, 1])
        time_id = np.array([1, 2, 3])

        result = panel_to_dict(y, X, entity_id, time_id)

        assert len(result) == 1
        assert_allclose(result[1]["y"], [1.0, 2.0, 3.0])
