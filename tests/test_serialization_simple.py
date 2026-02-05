"""
Simple test script for serialization functionality (no pytest required).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from panelbox.core.results import PanelResults


def create_sample_results():
    """Create sample PanelResults for testing."""
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

    data_info = {"nobs": n, "n_entities": 10, "n_periods": 10, "df_model": k, "df_resid": n - k - 1}

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


def test_to_dict():
    """Test to_dict() method."""
    print("Testing to_dict()...")
    results = create_sample_results()
    result_dict = results.to_dict()

    assert isinstance(result_dict, dict), "to_dict() should return a dict"
    assert "params" in result_dict, "Missing 'params' key"
    assert "std_errors" in result_dict, "Missing 'std_errors' key"
    assert "model_info" in result_dict, "Missing 'model_info' key"
    assert result_dict["params"]["x1"] == 1.5, "Incorrect params value"
    assert isinstance(result_dict["resid"], list), "resid should be a list"
    print("  ✓ to_dict() works correctly")


def test_to_json():
    """Test to_json() method."""
    print("Testing to_json()...")
    results = create_sample_results()

    # Test JSON string generation
    json_str = results.to_json()
    assert isinstance(json_str, str), "to_json() should return a string"

    # Verify it's valid JSON
    data = json.loads(json_str)
    assert isinstance(data, dict), "JSON should parse to a dict"
    assert data["model_info"]["model_type"] == "FixedEffects", "Incorrect model_type in JSON"
    print("  ✓ to_json() works correctly")


def test_save_load_pickle():
    """Test save() and load() with pickle format."""
    print("Testing save/load (pickle)...")
    results = create_sample_results()

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Save
        results.save(filepath, format="pickle")
        assert Path(filepath).exists(), "Pickle file not created"

        # Load
        loaded_results = PanelResults.load(filepath)
        assert isinstance(loaded_results, PanelResults), "Loaded object is not PanelResults"

        # Verify data
        pd.testing.assert_series_equal(loaded_results.params, results.params)
        assert loaded_results.model_type == results.model_type
        assert loaded_results.nobs == results.nobs
        assert loaded_results.rsquared == results.rsquared

        print("  ✓ save/load (pickle) works correctly")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_save_json():
    """Test save() with JSON format."""
    print("Testing save (JSON)...")
    results = create_sample_results()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Save
        results.save(filepath, format="json")
        assert Path(filepath).exists(), "JSON file not created"

        # Load and verify
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["model_info"]["model_type"] == "FixedEffects"
        assert data["sample_info"]["nobs"] == 100
        print("  ✓ save (JSON) works correctly")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_roundtrip():
    """Test complete round-trip serialization."""
    print("Testing round-trip serialization...")
    results = create_sample_results()

    # Get original summary
    original_summary = results.summary()

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Save and load
        results.save(filepath)
        loaded_results = PanelResults.load(filepath)

        # Compare summaries
        loaded_summary = loaded_results.summary()
        assert original_summary == loaded_summary, "Summaries differ after round-trip"

        # Compare conf_int
        original_ci = results.conf_int()
        loaded_ci = loaded_results.conf_int()
        pd.testing.assert_frame_equal(original_ci, loaded_ci)

        print("  ✓ Round-trip serialization works correctly")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_to_json_file():
    """Test to_json() with file output."""
    print("Testing to_json() with file...")
    results = create_sample_results()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Save to file
        results.to_json(filepath)
        assert Path(filepath).exists(), "JSON file not created"

        # Load and verify
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["model_info"]["formula"] == "y ~ x1 + x2 + x3"
        print("  ✓ to_json() with file works correctly")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_edge_cases():
    """Test edge cases."""
    print("Testing edge cases...")

    # Test with NaN R-squared
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

    data_info = {
        "nobs": 3,
        "n_entities": 1,
        "n_periods": None,  # Test None value
        "df_model": 1,
        "df_resid": 1,
    }

    results = PanelResults(
        params=params,
        std_errors=std_errors,
        cov_params=cov_params,
        resid=resid,
        fittedvalues=fittedvalues,
        model_info=model_info,
        data_info=data_info,
        rsquared_dict=None,  # Test NaN R-squared
    )

    # Test to_dict with NaN values
    result_dict = results.to_dict()
    assert result_dict["rsquared"]["rsquared"] is None
    assert result_dict["sample_info"]["n_periods"] is None

    # Test to_json doesn't fail
    json_str = results.to_json()
    assert isinstance(json_str, str)

    print("  ✓ Edge cases handled correctly")


def main():
    """Run all tests."""
    print("=" * 70)
    print("Testing PanelResults Serialization")
    print("=" * 70)
    print()

    tests = [
        test_to_dict,
        test_to_json,
        test_save_load_pickle,
        test_save_json,
        test_roundtrip,
        test_to_json_file,
        test_edge_cases,
    ]

    failed = 0
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            failed += 1

    print()
    print("=" * 70)
    if failed == 0:
        print("✓ All tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 70)

    return failed


if __name__ == "__main__":
    sys.exit(main())
