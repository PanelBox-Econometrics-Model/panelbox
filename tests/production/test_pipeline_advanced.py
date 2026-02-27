"""Advanced tests for PanelPipeline -- error paths and uncovered methods."""

import json
import pickle
import warnings
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.production.pipeline import PanelPipeline

# -- forecast error paths (Etapa 3) ------------------------------------------


class TestForecast:
    def test_forecast_unfitted_raises(self):
        """Test forecast raises when pipeline is not fitted."""
        pipeline = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1 + x2",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        with pytest.raises(RuntimeError, match="not fitted"):
            pipeline.forecast()

    def test_forecast_no_method_raises(self, fitted_pipeline):
        """Test forecast raises when model has no forecast method."""
        with pytest.raises(AttributeError, match="does not support forecast"):
            fitted_pipeline.forecast(steps=5)


# -- load error paths (Etapa 4) -----------------------------------------------


class TestLoad:
    def test_load_nonexistent_file_raises(self):
        """Test load raises for nonexistent file."""
        with pytest.raises(FileNotFoundError, match="not found"):
            PanelPipeline.load("/nonexistent/path/model.pkl")

    def test_load_wrong_type_raises(self, tmp_path):
        """Test load raises for wrong file content type."""
        bad_file = tmp_path / "bad_model.pkl"
        with open(bad_file, "wb") as f:
            pickle.dump({"not": "a pipeline"}, f)
        with pytest.raises(TypeError, match="expected PanelPipeline"):
            PanelPipeline.load(str(bad_file))


# -- to_json and summary (Etapa 5) -------------------------------------------


class TestJsonAndSummary:
    def test_to_json_returns_string(self, fitted_pipeline):
        """Test to_json returns valid JSON string."""
        json_str = fitted_pipeline.to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)
        assert "model_class" in data

    def test_to_json_writes_file(self, fitted_pipeline, tmp_path):
        """Test to_json writes to file."""
        filepath = tmp_path / "pipeline.json"
        fitted_pipeline.to_json(filepath=str(filepath))
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert isinstance(data, dict)

    def test_summary_fitted(self, fitted_pipeline):
        """Test summary on fitted pipeline returns string with model info."""
        summary = fitted_pipeline.summary()
        assert isinstance(summary, str)
        assert "PanelPipeline" in summary
        assert "PooledOLS" in summary
        assert "Parameters:" in summary

    def test_summary_unfitted(self):
        """Test summary on unfitted pipeline."""
        pipeline = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1 + x2",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        summary = pipeline.summary()
        assert isinstance(summary, str)
        assert "No" in summary

    def test_to_dict_with_std_errors(self, fitted_pipeline):
        """Test to_dict includes std_errors when available."""
        result = fitted_pipeline.to_dict()
        assert isinstance(result, dict)
        assert "params" in result
        if hasattr(fitted_pipeline.results, "std_errors"):
            assert "std_errors" in result


# -- compare and refit (Etapa 6) ----------------------------------------------


class TestCompareAndRefit:
    def test_compare_unfitted_raises(self, fitted_pipeline):
        """Test compare raises when one pipeline is not fitted."""
        unfitted = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1 + x2",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        with pytest.raises(RuntimeError, match="must be fitted"):
            unfitted.compare(fitted_pipeline)

    def test_refit_large_change_warns(self, fitted_pipeline, sample_panel_data):
        """Test refit warns when coefficients change significantly."""
        modified_data = sample_panel_data.copy()
        np.random.seed(99)
        modified_data["y"] = modified_data["y"] * 100 + np.random.randn(len(modified_data)) * 50
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fitted_pipeline.refit(modified_data)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "Large parameter change" in str(user_warnings[0].message)


# -- validate GMM paths (Etapa 7) ---------------------------------------------


class TestValidateGMM:
    def _make_pipeline_with_mock_results(self, hansen_pval, ar2_pval, instr_ratio=None):
        """Create a PanelPipeline with mocked GMM results."""
        pipeline = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        mock_results = MagicMock()
        mock_results.params = pd.Series({"x1": 1.0, "const": 0.5})

        hansen_mock = MagicMock()
        hansen_mock.pvalue = hansen_pval
        mock_results.hansen_j = hansen_mock

        ar2_mock = MagicMock()
        ar2_mock.pvalue = ar2_pval
        mock_results.ar2_test = ar2_mock

        if instr_ratio is not None:
            mock_results.instrument_ratio = instr_ratio
        else:
            del mock_results.instrument_ratio

        pipeline.results = mock_results
        pipeline.fit_timestamp = "2026-01-01T00:00:00"
        return pipeline

    def test_validate_gmm_bad_hansen(self):
        """Test validation flags bad Hansen test (p < 0.10)."""
        pipeline = self._make_pipeline_with_mock_results(hansen_pval=0.001, ar2_pval=0.50)
        report = pipeline.validate()
        assert not report["passed"]
        assert any("Hansen J rejected" in w for w in report["warnings"])

    def test_validate_gmm_bad_ar2(self):
        """Test validation flags bad AR(2) test (p < 0.10)."""
        pipeline = self._make_pipeline_with_mock_results(hansen_pval=0.50, ar2_pval=0.01)
        report = pipeline.validate()
        assert not report["passed"]
        assert any("AR(2) rejected" in w for w in report["warnings"])

    def test_validate_gmm_instrument_ratio(self):
        """Test validation warns about high instrument ratio."""
        pipeline = self._make_pipeline_with_mock_results(
            hansen_pval=0.50, ar2_pval=0.50, instr_ratio=2.5
        )
        report = pipeline.validate()
        assert not report["passed"]
        assert any("Too many instruments" in w for w in report["warnings"])

    def test_validate_unfitted(self):
        """Test validate on unfitted pipeline returns not passed."""
        pipeline = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        report = pipeline.validate()
        assert not report["passed"]
        assert "Model not fitted" in report["warnings"]


# -- repr (bonus) -------------------------------------------------------------


class TestRepr:
    def test_repr_fitted(self, fitted_pipeline):
        """Test repr for fitted pipeline."""
        r = repr(fitted_pipeline)
        assert "fitted" in r
        assert "PooledOLS" in r

    def test_repr_unfitted(self):
        """Test repr for unfitted pipeline."""
        pipeline = PanelPipeline(
            model_class=PooledOLS,
            model_params={
                "formula": "y ~ x1",
                "entity_col": "entity",
                "time_col": "time",
            },
        )
        r = repr(pipeline)
        assert "not fitted" in r
