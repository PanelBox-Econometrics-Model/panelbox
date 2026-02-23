"""
Tests for predict infrastructure (PanelResults._build_design_matrix, SerializableMixin).
"""

import tempfile

import numpy as np
import pandas as pd
import pytest

from panelbox.core.results import PanelResults
from panelbox.core.serialization import SerializableMixin


class _DummyResultsWithAttrs(SerializableMixin):
    """Module-level dummy class for pickle tests (local classes can't be pickled)."""

    def __init__(self):
        self.params = pd.Series([1.0, 2.0], index=["a", "b"])
        self.value = 42


class _DummyResultsEmpty(SerializableMixin):
    """Module-level dummy class for pickle tests (local classes can't be pickled)."""

    pass


class TestBuildDesignMatrix:
    """Test _build_design_matrix helper."""

    @pytest.fixture
    def sample_results(self):
        """Create a minimal PanelResults with formula."""
        params = pd.Series([1.0, 2.0, 3.0], index=["Intercept", "x1", "x2"])
        std_errors = pd.Series([0.1, 0.2, 0.3], index=["Intercept", "x1", "x2"])
        cov = pd.DataFrame(np.eye(3) * 0.01, index=params.index, columns=params.index)
        resid = np.zeros(10)
        fitted = np.ones(10)
        model_info = {"model_type": "PooledOLS", "formula": "y ~ x1 + x2"}
        data_info = {"nobs": 10, "n_entities": 2, "df_model": 2, "df_resid": 7}

        return PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov,
            resid=resid,
            fittedvalues=fitted,
            model_info=model_info,
            data_info=data_info,
        )

    def test_predict_no_newdata_returns_fitted(self, sample_results):
        """predict() without newdata returns fittedvalues."""
        preds = sample_results.predict()
        assert np.allclose(preds, sample_results.fittedvalues)

    def test_build_design_matrix_from_formula_string(self, sample_results):
        """_build_design_matrix reconstructs from formula string."""
        newdata = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [0, 0]})
        X = sample_results._build_design_matrix(newdata)
        assert X.shape[0] == 2
        assert X.shape[1] == 3  # intercept + x1 + x2

    def test_predict_newdata_computes_xb(self, sample_results):
        """predict(newdata) returns X @ beta."""
        newdata = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "y": [0, 0]})
        preds = sample_results.predict(newdata)
        # Expected: 1*1 + 2*1 + 3*3 = 12, 1*1 + 2*2 + 3*4 = 17
        expected = np.array([12.0, 17.0])
        assert np.allclose(preds, expected)

    def test_build_design_matrix_no_formula_raises(self):
        """_build_design_matrix raises ValueError when no formula is available."""
        params = pd.Series([1.0], index=["x1"])
        std_errors = pd.Series([0.1], index=["x1"])
        cov = pd.DataFrame([[0.01]], index=params.index, columns=params.index)
        model_info = {"model_type": "PooledOLS", "formula": ""}
        data_info = {"nobs": 10, "n_entities": 2, "df_model": 1, "df_resid": 8}

        results = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov,
            resid=np.zeros(10),
            fittedvalues=np.ones(10),
            model_info=model_info,
            data_info=data_info,
        )

        with pytest.raises(ValueError, match="no formula available"):
            results._build_design_matrix(pd.DataFrame({"x1": [1.0]}))


class TestSerializableMixin:
    """Test SerializableMixin save/load."""

    def test_pickle_roundtrip(self):
        """Save/load via pickle preserves all attributes."""
        obj = _DummyResultsWithAttrs()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            obj.save(f.name)
            loaded = _DummyResultsWithAttrs.load(f.name)

        assert np.allclose(loaded.params.values, obj.params.values)
        assert loaded.value == 42

    def test_save_creates_parent_dirs(self):
        """save() creates parent directories if needed."""
        obj = _DummyResultsEmpty()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/subdir/model.pkl"
            obj.save(path)
            _DummyResultsEmpty.load(path)

    def test_load_nonexistent_raises(self):
        """load() on missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SerializableMixin.load("/nonexistent/path.pkl")
