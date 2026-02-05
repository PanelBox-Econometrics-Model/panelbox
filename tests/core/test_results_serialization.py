"""
Tests for PanelResults serialization and deserialization methods.

This test suite covers:
- to_dict() method
- to_json() method (with and without file)
- save() method (pickle and json formats)
- load() method
- Round-trip serialization/deserialization
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from panelbox.core.results import PanelResults


@pytest.fixture
def sample_results():
    """Create sample PanelResults for testing."""
    # Create sample data
    n = 100
    k = 3

    params = pd.Series([1.5, -0.8, 2.3], index=["x1", "x2", "x3"])
    std_errors = pd.Series([0.2, 0.15, 0.25], index=["x1", "x2", "x3"])
    cov_params = pd.DataFrame(
        np.array([[0.04, 0.001, 0.002], [0.001, 0.0225, 0.001], [0.002, 0.001, 0.0625]]),
        index=["x1", "x2", "x3"],
        columns=["x1", "x2", "x3"],
    )

    resid = np.random.randn(n)
    fittedvalues = np.random.randn(n)

    model_info = {
        "model_type": "FixedEffects",
        "formula": "y ~ x1 + x2 + x3",
        "cov_type": "robust",
        "cov_kwds": {"use_correction": True},
    }

    data_info = {
        "nobs": n,
        "n_entities": 10,
        "n_periods": 10,
        "df_model": k,
        "df_resid": n - k - 1,
        "entity_index": pd.RangeIndex(10),
        "time_index": pd.RangeIndex(10),
    }

    rsquared_dict = {
        "rsquared": 0.75,
        "rsquared_adj": 0.72,
        "rsquared_within": 0.68,
        "rsquared_between": 0.82,
        "rsquared_overall": 0.73,
    }

    return PanelResults(
        params=params,
        std_errors=std_errors,
        cov_params=cov_params,
        resid=resid,
        fittedvalues=fittedvalues,
        model_info=model_info,
        data_info=data_info,
        rsquared_dict=rsquared_dict,
    )


class TestToDict:
    """Test to_dict() method."""

    def test_to_dict_basic(self, sample_results):
        """Test that to_dict returns a dictionary."""
        result_dict = sample_results.to_dict()
        assert isinstance(result_dict, dict)

    def test_to_dict_keys(self, sample_results):
        """Test that to_dict includes all expected keys."""
        result_dict = sample_results.to_dict()
        expected_keys = {
            "params",
            "std_errors",
            "tvalues",
            "pvalues",
            "cov_params",
            "resid",
            "fittedvalues",
            "model_info",
            "sample_info",
            "rsquared",
        }
        assert set(result_dict.keys()) == expected_keys

    def test_to_dict_params(self, sample_results):
        """Test that params are correctly converted."""
        result_dict = sample_results.to_dict()
        assert isinstance(result_dict["params"], dict)
        assert result_dict["params"]["x1"] == 1.5
        assert result_dict["params"]["x2"] == -0.8
        assert result_dict["params"]["x3"] == 2.3

    def test_to_dict_arrays_to_lists(self, sample_results):
        """Test that numpy arrays are converted to lists."""
        result_dict = sample_results.to_dict()
        assert isinstance(result_dict["resid"], list)
        assert isinstance(result_dict["fittedvalues"], list)
        assert len(result_dict["resid"]) == 100
        assert len(result_dict["fittedvalues"]) == 100

    def test_to_dict_cov_params(self, sample_results):
        """Test that covariance matrix is properly converted."""
        result_dict = sample_results.to_dict()
        assert isinstance(result_dict["cov_params"], dict)
        assert "values" in result_dict["cov_params"]
        assert "index" in result_dict["cov_params"]
        assert "columns" in result_dict["cov_params"]
        assert isinstance(result_dict["cov_params"]["values"], list)

    def test_to_dict_model_info(self, sample_results):
        """Test that model info is correctly included."""
        result_dict = sample_results.to_dict()
        model_info = result_dict["model_info"]
        assert model_info["model_type"] == "FixedEffects"
        assert model_info["formula"] == "y ~ x1 + x2 + x3"
        assert model_info["cov_type"] == "robust"

    def test_to_dict_sample_info(self, sample_results):
        """Test that sample info is correctly included."""
        result_dict = sample_results.to_dict()
        sample_info = result_dict["sample_info"]
        assert sample_info["nobs"] == 100
        assert sample_info["n_entities"] == 10
        assert sample_info["n_periods"] == 10
        assert isinstance(sample_info["nobs"], int)
        assert isinstance(sample_info["n_entities"], int)

    def test_to_dict_rsquared(self, sample_results):
        """Test that R-squared values are correctly included."""
        result_dict = sample_results.to_dict()
        rsq = result_dict["rsquared"]
        assert rsq["rsquared"] == 0.75
        assert rsq["rsquared_adj"] == 0.72
        assert rsq["rsquared_within"] == 0.68
        assert rsq["rsquared_between"] == 0.82
        assert rsq["rsquared_overall"] == 0.73


class TestToJson:
    """Test to_json() method."""

    def test_to_json_string(self, sample_results):
        """Test that to_json returns a JSON string."""
        json_str = sample_results.to_json()
        assert isinstance(json_str, str)
        # Verify it's valid JSON
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_to_json_with_file(self, sample_results):
        """Test saving JSON to file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save to file
            json_str = sample_results.to_json(filepath)

            # Verify file exists
            assert Path(filepath).exists()

            # Load and verify content
            with open(filepath, "r") as f:
                loaded_data = json.load(f)

            assert loaded_data["model_info"]["model_type"] == "FixedEffects"
            assert loaded_data["sample_info"]["nobs"] == 100
        finally:
            # Clean up
            Path(filepath).unlink(missing_ok=True)

    def test_to_json_indent(self, sample_results):
        """Test JSON indentation."""
        json_str_2 = sample_results.to_json(indent=2)
        json_str_4 = sample_results.to_json(indent=4)

        # 4-space indent should be longer
        assert len(json_str_4) > len(json_str_2)

    def test_to_json_parseable(self, sample_results):
        """Test that JSON output is parseable and contains expected data."""
        json_str = sample_results.to_json()
        data = json.loads(json_str)

        # Check key structures
        assert "params" in data
        assert "model_info" in data
        assert data["params"]["x1"] == 1.5
        assert data["model_info"]["formula"] == "y ~ x1 + x2 + x3"


class TestSave:
    """Test save() method."""

    def test_save_pickle(self, sample_results):
        """Test saving as pickle."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Save
            sample_results.save(filepath, format="pickle")

            # Verify file exists
            assert Path(filepath).exists()

            # Verify file is not empty
            assert Path(filepath).stat().st_size > 0
        finally:
            # Clean up
            Path(filepath).unlink(missing_ok=True)

    def test_save_json(self, sample_results):
        """Test saving as JSON."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            # Save
            sample_results.save(filepath, format="json")

            # Verify file exists
            assert Path(filepath).exists()

            # Load and verify it's valid JSON
            with open(filepath, "r") as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert data["model_info"]["model_type"] == "FixedEffects"
        finally:
            # Clean up
            Path(filepath).unlink(missing_ok=True)

    def test_save_invalid_format(self, sample_results):
        """Test that invalid format raises error."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            filepath = f.name

        try:
            with pytest.raises(ValueError, match="not supported"):
                sample_results.save(filepath, format="invalid_format")
        finally:
            # Clean up
            Path(filepath).unlink(missing_ok=True)

    def test_save_path_object(self, sample_results):
        """Test that Path objects work for filepath."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = Path(f.name)

        try:
            sample_results.save(filepath)
            assert filepath.exists()
        finally:
            filepath.unlink(missing_ok=True)


class TestLoad:
    """Test load() classmethod."""

    def test_load_basic(self, sample_results):
        """Test basic loading of pickle file."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Save and load
            sample_results.save(filepath)
            loaded_results = PanelResults.load(filepath)

            # Verify type
            assert isinstance(loaded_results, PanelResults)

            # Verify basic attributes
            assert loaded_results.model_type == sample_results.model_type
            assert loaded_results.nobs == sample_results.nobs
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            PanelResults.load("/nonexistent/path/file.pkl")

    def test_load_path_object(self, sample_results):
        """Test loading with Path object."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = Path(f.name)

        try:
            sample_results.save(filepath)
            loaded_results = PanelResults.load(filepath)
            assert isinstance(loaded_results, PanelResults)
        finally:
            filepath.unlink(missing_ok=True)


class TestRoundTrip:
    """Test round-trip serialization/deserialization."""

    def test_roundtrip_pickle(self, sample_results):
        """Test that pickle round-trip preserves all data."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Save and load
            sample_results.save(filepath)
            loaded_results = PanelResults.load(filepath)

            # Compare parameters
            pd.testing.assert_series_equal(loaded_results.params, sample_results.params)
            pd.testing.assert_series_equal(loaded_results.std_errors, sample_results.std_errors)
            pd.testing.assert_series_equal(loaded_results.tvalues, sample_results.tvalues)
            pd.testing.assert_series_equal(loaded_results.pvalues, sample_results.pvalues)

            # Compare covariance matrix
            pd.testing.assert_frame_equal(loaded_results.cov_params, sample_results.cov_params)

            # Compare arrays
            np.testing.assert_array_equal(loaded_results.resid, sample_results.resid)
            np.testing.assert_array_equal(loaded_results.fittedvalues, sample_results.fittedvalues)

            # Compare attributes
            assert loaded_results.model_type == sample_results.model_type
            assert loaded_results.formula == sample_results.formula
            assert loaded_results.cov_type == sample_results.cov_type
            assert loaded_results.nobs == sample_results.nobs
            assert loaded_results.n_entities == sample_results.n_entities
            assert loaded_results.n_periods == sample_results.n_periods
            assert loaded_results.df_model == sample_results.df_model
            assert loaded_results.df_resid == sample_results.df_resid

            # Compare R-squared values
            assert loaded_results.rsquared == sample_results.rsquared
            assert loaded_results.rsquared_adj == sample_results.rsquared_adj
            assert loaded_results.rsquared_within == sample_results.rsquared_within
            assert loaded_results.rsquared_between == sample_results.rsquared_between
            assert loaded_results.rsquared_overall == sample_results.rsquared_overall
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_roundtrip_summary(self, sample_results):
        """Test that summary() works after loading."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Get original summary
            original_summary = sample_results.summary()

            # Save, load, get new summary
            sample_results.save(filepath)
            loaded_results = PanelResults.load(filepath)
            loaded_summary = loaded_results.summary()

            # Summaries should be identical
            assert original_summary == loaded_summary
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_roundtrip_conf_int(self, sample_results):
        """Test that conf_int() works after loading."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Get original conf_int
            original_ci = sample_results.conf_int()

            # Save, load, get new conf_int
            sample_results.save(filepath)
            loaded_results = PanelResults.load(filepath)
            loaded_ci = loaded_results.conf_int()

            # CIs should be identical
            pd.testing.assert_frame_equal(original_ci, loaded_ci)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_roundtrip_to_dict(self, sample_results):
        """Test that to_dict() works after loading."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            # Get original dict
            original_dict = sample_results.to_dict()

            # Save, load, get new dict
            sample_results.save(filepath)
            loaded_results = PanelResults.load(filepath)
            loaded_dict = loaded_results.to_dict()

            # Compare keys
            assert set(original_dict.keys()) == set(loaded_dict.keys())

            # Compare some values
            assert original_dict["model_info"] == loaded_dict["model_info"]
            assert original_dict["sample_info"] == loaded_dict["sample_info"]
        finally:
            Path(filepath).unlink(missing_ok=True)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_nan_rsquared(self):
        """Test handling of NaN R-squared values."""
        params = pd.Series([1.0], index=["x1"])
        std_errors = pd.Series([0.1], index=["x1"])
        cov_params = pd.DataFrame([[0.01]], index=["x1"], columns=["x1"])
        resid = np.array([1.0, 2.0, 3.0])
        fittedvalues = np.array([1.5, 2.5, 3.5])

        model_info = {
            "model_type": "Test",
            "formula": "y ~ x1",
            "cov_type": "nonrobust",
            "cov_kwds": {},
        }

        data_info = {"nobs": 3, "n_entities": 1, "n_periods": 3, "df_model": 1, "df_resid": 1}

        # Create results with NaN R-squared (no rsquared_dict)
        results = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=resid,
            fittedvalues=fittedvalues,
            model_info=model_info,
            data_info=data_info,
            rsquared_dict=None,
        )

        # Test to_dict with NaN values
        result_dict = results.to_dict()
        assert result_dict["rsquared"]["rsquared"] is None
        assert result_dict["rsquared"]["rsquared_adj"] is None

        # Test to_json doesn't fail
        json_str = results.to_json()
        assert isinstance(json_str, str)

        # Test round-trip
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            results.save(filepath)
            loaded_results = PanelResults.load(filepath)
            assert np.isnan(loaded_results.rsquared)
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_none_n_periods(self):
        """Test handling of None n_periods."""
        params = pd.Series([1.0], index=["x1"])
        std_errors = pd.Series([0.1], index=["x1"])
        cov_params = pd.DataFrame([[0.01]], index=["x1"], columns=["x1"])
        resid = np.array([1.0, 2.0])
        fittedvalues = np.array([1.5, 2.5])

        model_info = {
            "model_type": "Test",
            "formula": "y ~ x1",
            "cov_type": "nonrobust",
            "cov_kwds": {},
        }

        data_info = {
            "nobs": 2,
            "n_entities": 2,
            "n_periods": None,  # None value
            "df_model": 1,
            "df_resid": 0,
        }

        results = PanelResults(
            params=params,
            std_errors=std_errors,
            cov_params=cov_params,
            resid=resid,
            fittedvalues=fittedvalues,
            model_info=model_info,
            data_info=data_info,
        )

        # Test to_dict
        result_dict = results.to_dict()
        assert result_dict["sample_info"]["n_periods"] is None

        # Test round-trip
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            results.save(filepath)
            loaded_results = PanelResults.load(filepath)
            assert loaded_results.n_periods is None
        finally:
            Path(filepath).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
