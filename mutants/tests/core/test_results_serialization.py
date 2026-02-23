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
        # Verify exact index and column names
        assert result_dict["cov_params"]["index"] == ["x1", "x2", "x3"]
        assert result_dict["cov_params"]["columns"] == ["x1", "x2", "x3"]

    def test_to_dict_tvalues(self, sample_results):
        """Test that t-values are correctly included in dict."""
        result_dict = sample_results.to_dict()
        assert isinstance(result_dict["tvalues"], dict)
        # t-value for x1: 1.5 / 0.2 = 7.5
        assert result_dict["tvalues"]["x1"] == pytest.approx(7.5)
        # t-value for x2: -0.8 / 0.15 = -5.333
        assert result_dict["tvalues"]["x2"] == pytest.approx(-0.8 / 0.15)

    def test_to_dict_pvalues(self, sample_results):
        """Test that p-values are correctly included in dict."""
        result_dict = sample_results.to_dict()
        assert isinstance(result_dict["pvalues"], dict)
        # All p-values should be between 0 and 1
        for _key, val in result_dict["pvalues"].items():
            assert 0 <= val <= 1

    def test_to_dict_df_values(self, sample_results):
        """Test that df_model and df_resid are exactly correct."""
        result_dict = sample_results.to_dict()
        assert result_dict["sample_info"]["df_model"] == 3
        assert result_dict["sample_info"]["df_resid"] == 96

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
            sample_results.to_json(filepath)

            # Verify file exists
            assert Path(filepath).exists()

            # Load and verify content
            with open(filepath) as f:
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
            with open(filepath) as f:
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

    def test_save_pickle_has_metadata(self, sample_results):
        """Test that pickle save includes version and timestamp metadata."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name

        try:
            sample_results.save(filepath, format="pickle")
            loaded = PanelResults.load(filepath)
            assert hasattr(loaded, "_panelbox_version")
            assert hasattr(loaded, "_save_timestamp")
            assert loaded._panelbox_version != ""
            assert loaded._save_timestamp != ""
        finally:
            Path(filepath).unlink(missing_ok=True)

    def test_save_json_content(self, sample_results):
        """Test that JSON save produces valid, parseable content."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            sample_results.save(filepath, format="json")
            with open(filepath) as f:
                data = json.load(f)
            # Verify specific values
            assert data["params"]["x1"] == 1.5
            assert data["sample_info"]["nobs"] == 100
            assert data["rsquared"]["rsquared"] == 0.75
        finally:
            Path(filepath).unlink(missing_ok=True)


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


class TestSummaryOutput:
    """Test summary() output format and content."""

    def test_summary_contains_model_type(self, sample_results):
        """Test that summary includes model type."""
        summary = sample_results.summary()
        assert "FixedEffects Estimation Results" in summary

    def test_summary_contains_formula(self, sample_results):
        """Test that summary includes formula."""
        summary = sample_results.summary()
        assert "Formula: y ~ x1 + x2 + x3" in summary

    def test_summary_contains_nobs(self, sample_results):
        """Test that summary includes observation count."""
        summary = sample_results.summary()
        assert "No. Observations:" in summary
        assert "100" in summary

    def test_summary_contains_coefficients(self, sample_results):
        """Test that summary includes exact coefficient values."""
        summary = sample_results.summary()
        assert "x1" in summary
        assert "x2" in summary
        assert "x3" in summary
        # Check coefficient values appear
        assert "1.5000" in summary
        assert "-0.8000" in summary
        assert "2.3000" in summary

    def test_summary_contains_rsquared(self, sample_results):
        """Test that summary includes R-squared values."""
        summary = sample_results.summary()
        assert "R-squared:" in summary
        assert "0.7500" in summary

    def test_summary_custom_title(self, sample_results):
        """Test summary with custom title."""
        summary = sample_results.summary(title="My Custom Title")
        assert "My Custom Title" in summary
        assert "FixedEffects Estimation Results" not in summary

    def test_summary_significance_stars(self, sample_results):
        """Test that summary includes significance markers."""
        summary = sample_results.summary()
        assert "Signif. codes:" in summary

    def test_summary_conf_intervals(self, sample_results):
        """Test that summary includes confidence intervals."""
        summary = sample_results.summary()
        assert "[0.025" in summary
        assert "0.975]" in summary

    def test_summary_n_entities(self, sample_results):
        """Test that summary shows number of entities."""
        summary = sample_results.summary()
        assert "No. Entities:" in summary
        assert "10" in summary

    def test_summary_degrees_of_freedom(self, sample_results):
        """Test that summary shows degrees of freedom."""
        summary = sample_results.summary()
        assert "Degrees of Freedom:" in summary
        assert "96" in summary  # 100 - 3 - 1


class TestConfInt:
    """Test conf_int() method."""

    def test_conf_int_default_alpha(self, sample_results):
        """Test confidence intervals with default alpha=0.05."""
        ci = sample_results.conf_int()
        assert "lower" in ci.columns
        assert "upper" in ci.columns
        assert len(ci) == 3  # 3 parameters
        # Lower should be less than upper for all parameters
        assert (ci["lower"] < ci["upper"]).all()

    def test_conf_int_contains_params(self, sample_results):
        """Test that confidence intervals contain the point estimates."""
        ci = sample_results.conf_int()
        for var in sample_results.params.index:
            assert ci.loc[var, "lower"] < sample_results.params[var]
            assert ci.loc[var, "upper"] > sample_results.params[var]

    def test_conf_int_custom_alpha(self, sample_results):
        """Test CI with different alpha levels."""
        ci_95 = sample_results.conf_int(alpha=0.05)
        ci_99 = sample_results.conf_int(alpha=0.01)
        # 99% CI should be wider than 95%
        width_95 = ci_95["upper"] - ci_95["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert (width_99 > width_95).all()


class TestInitAttributes:
    """Test that __init__ correctly sets all attributes."""

    def test_tvalues_computed_correctly(self, sample_results):
        """Test that t-values are params / std_errors."""
        expected_t = sample_results.params / sample_results.std_errors
        pd.testing.assert_series_equal(sample_results.tvalues, expected_t)

    def test_pvalues_are_valid(self, sample_results):
        """Test that p-values are between 0 and 1."""
        assert (sample_results.pvalues >= 0).all()
        assert (sample_results.pvalues <= 1).all()

    def test_model_type_attribute(self, sample_results):
        """Test model_type is correctly set."""
        assert sample_results.model_type == "FixedEffects"

    def test_formula_attribute(self, sample_results):
        """Test formula is correctly set."""
        assert sample_results.formula == "y ~ x1 + x2 + x3"

    def test_cov_type_attribute(self, sample_results):
        """Test cov_type is correctly set."""
        assert sample_results.cov_type == "robust"

    def test_nobs_attribute(self, sample_results):
        """Test nobs is exactly 100."""
        assert sample_results.nobs == 100

    def test_n_entities_attribute(self, sample_results):
        """Test n_entities is exactly 10."""
        assert sample_results.n_entities == 10

    def test_n_periods_attribute(self, sample_results):
        """Test n_periods is exactly 10."""
        assert sample_results.n_periods == 10

    def test_df_model_attribute(self, sample_results):
        """Test df_model is exactly 3."""
        assert sample_results.df_model == 3

    def test_df_resid_attribute(self, sample_results):
        """Test df_resid is exactly 96."""
        assert sample_results.df_resid == 96

    def test_rsquared_attributes(self, sample_results):
        """Test R-squared values are exact."""
        assert sample_results.rsquared == 0.75
        assert sample_results.rsquared_adj == 0.72
        assert sample_results.rsquared_within == 0.68
        assert sample_results.rsquared_between == 0.82
        assert sample_results.rsquared_overall == 0.73

    def test_ssr_attribute(self, sample_results):
        """Test that SSR is computed correctly."""
        expected_ssr = float(np.sum(sample_results.resid**2))
        assert sample_results.ssr == pytest.approx(expected_ssr)

    def test_repr_output(self, sample_results):
        """Test repr contains key info."""
        repr_str = repr(sample_results)
        assert "FixedEffects" in repr_str
        assert "nobs=100" in repr_str
        assert "k_params=3" in repr_str


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
