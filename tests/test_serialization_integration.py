"""
Integration test for serialization with real models.

This test creates real panel models, fits them, saves/loads results,
and verifies everything works correctly.
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

import panelbox as pb


def test_fixed_effects_serialization():
    """Test serialization with FixedEffects model."""
    print("Testing FixedEffects serialization...")

    # Load data
    data = pb.load_grunfeld()

    # Fit model
    fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    results = fe.fit(cov_type="robust")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Save
        results.save(filepath)

        # Load
        loaded_results = pb.PanelResults.load(filepath)

        # Compare
        assert loaded_results.model_type == "Fixed Effects"
        assert loaded_results.nobs == results.nobs
        pd.testing.assert_series_equal(loaded_results.params, results.params)

        # Test that methods work
        assert loaded_results.summary() == results.summary()

        print("  ✓ FixedEffects serialization works")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_pooled_ols_serialization():
    """Test serialization with PooledOLS model."""
    print("Testing PooledOLS serialization...")

    data = pb.load_grunfeld()

    # Fit model
    pooled = pb.PooledOLS("invest ~ value + capital", data, "firm", "year")
    results = pooled.fit(cov_type="clustered")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Save and load
        results.save(filepath)
        loaded_results = pb.PanelResults.load(filepath)

        # Verify
        assert loaded_results.model_type == "Pooled OLS"
        pd.testing.assert_series_equal(loaded_results.params, results.params)

        print("  ✓ PooledOLS serialization works")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_between_estimator_serialization():
    """Test serialization with BetweenEstimator."""
    print("Testing BetweenEstimator serialization...")

    data = pb.load_grunfeld()

    # Fit model
    be = pb.BetweenEstimator("invest ~ value + capital", data, "firm", "year")
    results = be.fit(cov_type="robust")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Save and load
        results.save(filepath)
        loaded_results = pb.PanelResults.load(filepath)

        # Verify
        assert loaded_results.model_type == "Between Estimator"
        pd.testing.assert_series_equal(loaded_results.params, results.params)

        print("  ✓ BetweenEstimator serialization works")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_first_difference_serialization():
    """Test serialization with FirstDifferenceEstimator."""
    print("Testing FirstDifferenceEstimator serialization...")

    data = pb.load_grunfeld()

    # Fit model
    fd = pb.FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year")
    results = fd.fit(cov_type="clustered")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Save and load
        results.save(filepath)
        loaded_results = pb.PanelResults.load(filepath)

        # Verify
        assert loaded_results.model_type == "First Difference"
        pd.testing.assert_series_equal(loaded_results.params, results.params)

        print("  ✓ FirstDifferenceEstimator serialization works")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_json_export():
    """Test JSON export functionality."""
    print("Testing JSON export...")

    data = pb.load_grunfeld()
    fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    results = fe.fit()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        # Export to JSON
        results.save(filepath, format="json")

        # Load and verify
        with open(filepath, "r") as f:
            data = json.load(f)

        assert data["model_info"]["model_type"] == "Fixed Effects"
        assert data["model_info"]["formula"] == "invest ~ value + capital"
        assert "params" in data
        assert "value" in data["params"]
        assert "capital" in data["params"]

        print("  ✓ JSON export works")
    finally:
        Path(filepath).unlink(missing_ok=True)


def test_multiple_models_save_load():
    """Test saving and loading multiple models."""
    print("Testing multiple models save/load...")

    data = pb.load_grunfeld()

    # Fit multiple models
    models = {
        "pooled": pb.PooledOLS("invest ~ value + capital", data, "firm", "year"),
        "fe": pb.FixedEffects("invest ~ value + capital", data, "firm", "year"),
        "between": pb.BetweenEstimator("invest ~ value + capital", data, "firm", "year"),
        "fd": pb.FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year"),
    }

    results_dict = {}
    for name, model in models.items():
        results_dict[name] = model.fit()

    temp_files = {}
    try:
        # Save all
        for name, results in results_dict.items():
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                filepath = f.name
                temp_files[name] = filepath
                results.save(filepath)

        # Load all and verify
        for name, filepath in temp_files.items():
            loaded = pb.PanelResults.load(filepath)
            original = results_dict[name]

            # Compare key attributes
            pd.testing.assert_series_equal(loaded.params, original.params)
            assert loaded.model_type == original.model_type
            assert loaded.nobs == original.nobs

        print("  ✓ Multiple models save/load works")
    finally:
        # Clean up
        for filepath in temp_files.values():
            Path(filepath).unlink(missing_ok=True)


def test_to_dict_all_models():
    """Test to_dict() on all model types."""
    print("Testing to_dict() on all models...")

    data = pb.load_grunfeld()

    models = [
        pb.PooledOLS("invest ~ value + capital", data, "firm", "year"),
        pb.FixedEffects("invest ~ value + capital", data, "firm", "year"),
        pb.BetweenEstimator("invest ~ value + capital", data, "firm", "year"),
        pb.FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year"),
    ]

    for model in models:
        results = model.fit()
        result_dict = results.to_dict()

        # Verify basic structure
        assert isinstance(result_dict, dict)
        assert "params" in result_dict
        assert "model_info" in result_dict
        assert "sample_info" in result_dict
        assert isinstance(result_dict["params"], dict)

    print("  ✓ to_dict() works on all models")


def main():
    """Run all integration tests."""
    print("=" * 70)
    print("Integration Tests: Serialization with Real Models")
    print("=" * 70)
    print()

    tests = [
        test_fixed_effects_serialization,
        test_pooled_ols_serialization,
        test_between_estimator_serialization,
        test_first_difference_serialization,
        test_json_export,
        test_multiple_models_save_load,
        test_to_dict_all_models,
    ]

    failed = 0
    for test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
            import traceback

            traceback.print_exc()
            failed += 1

    print()
    print("=" * 70)
    if failed == 0:
        print("✓ All integration tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 70)

    return failed


if __name__ == "__main__":
    sys.exit(main())
