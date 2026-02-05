"""
Tests for PanelBox CLI.

This test suite covers the command-line interface functionality.
"""

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import panelbox as pb
from panelbox.cli.main import main


def create_test_data():
    """Create test CSV data."""
    data = pb.load_grunfeld()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        filepath = f.name
    data.to_csv(filepath, index=False)
    return filepath


def test_cli_help():
    """Test CLI help command."""
    import pytest

    print("Testing CLI help...")

    # Test main help - should raise SystemExit(0)
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0, "Help should exit with code 0"
    print("  ✓ Main help works")


def test_estimate_command_basic():
    """Test basic estimate command."""
    print("Testing estimate command (basic)...")

    data_file = create_test_data()
    output_file = tempfile.mktemp(suffix=".pkl")

    try:
        # Run estimate
        argv = [
            "estimate",
            "--data",
            data_file,
            "--model",
            "fe",
            "--formula",
            "invest ~ value + capital",
            "--entity",
            "firm",
            "--time",
            "year",
            "--output",
            output_file,
            "--no-summary",
        ]

        exit_code = main(argv)
        assert exit_code == 0, f"Exit code should be 0, got {exit_code}"
        assert Path(output_file).exists(), "Output file should exist"

        # Load and verify results
        results = pb.PanelResults.load(output_file)
        assert results.model_type == "Fixed Effects"
        assert len(results.params) == 2  # value and capital

        print("  ✓ Basic estimate works")
    finally:
        Path(data_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_estimate_different_models():
    """Test estimate with different models."""
    print("Testing estimate with different models...")

    data_file = create_test_data()

    models = ["pooled", "fe", "between", "fd"]
    expected_types = ["Pooled OLS", "Fixed Effects", "Between Estimator", "First Difference"]

    try:
        for model, expected_type in zip(models, expected_types):
            output_file = tempfile.mktemp(suffix=".pkl")

            argv = [
                "estimate",
                "--data",
                data_file,
                "--model",
                model,
                "--formula",
                "invest ~ value + capital",
                "--entity",
                "firm",
                "--time",
                "year",
                "--output",
                output_file,
                "--no-summary",
            ]

            exit_code = main(argv)
            assert exit_code == 0, f"Exit code for {model} should be 0"

            results = pb.PanelResults.load(output_file)
            assert (
                results.model_type == expected_type
            ), f"Model type should be {expected_type}, got {results.model_type}"

            Path(output_file).unlink()

        print(f"  ✓ All {len(models)} models work")
    finally:
        Path(data_file).unlink(missing_ok=True)


def test_estimate_cov_types():
    """Test estimate with different covariance types."""
    print("Testing estimate with different covariance types...")

    data_file = create_test_data()

    cov_types = ["nonrobust", "robust", "clustered"]

    try:
        for cov_type in cov_types:
            output_file = tempfile.mktemp(suffix=".pkl")

            argv = [
                "estimate",
                "--data",
                data_file,
                "--model",
                "pooled",
                "--formula",
                "invest ~ value + capital",
                "--entity",
                "firm",
                "--time",
                "year",
                "--cov-type",
                cov_type,
                "--output",
                output_file,
                "--no-summary",
            ]

            exit_code = main(argv)
            assert exit_code == 0, f"Exit code for {cov_type} should be 0"

            results = pb.PanelResults.load(output_file)
            assert (
                results.cov_type == cov_type
            ), f"Cov type should be {cov_type}, got {results.cov_type}"

            Path(output_file).unlink()

        print(f"  ✓ All {len(cov_types)} covariance types work")
    finally:
        Path(data_file).unlink(missing_ok=True)


def test_estimate_json_format():
    """Test estimate with JSON output format."""
    print("Testing estimate with JSON format...")

    data_file = create_test_data()
    output_file = tempfile.mktemp(suffix=".json")

    try:
        argv = [
            "estimate",
            "--data",
            data_file,
            "--model",
            "fe",
            "--formula",
            "invest ~ value + capital",
            "--entity",
            "firm",
            "--time",
            "year",
            "--output",
            output_file,
            "--format",
            "json",
            "--no-summary",
        ]

        exit_code = main(argv)
        assert exit_code == 0, "Exit code should be 0"
        assert Path(output_file).exists(), "JSON file should exist"

        # Verify it's valid JSON
        import json

        with open(output_file, "r") as f:
            data = json.load(f)

        assert "params" in data
        assert "model_info" in data

        print("  ✓ JSON format works")
    finally:
        Path(data_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_info_data():
    """Test info command with data."""
    print("Testing info command (data)...")

    data_file = create_test_data()

    try:
        argv = ["info", "--data", data_file, "--entity", "firm", "--time", "year"]

        exit_code = main(argv)
        assert exit_code == 0, "Exit code should be 0"

        print("  ✓ Info with data works")
    finally:
        Path(data_file).unlink(missing_ok=True)


def test_info_results():
    """Test info command with results."""
    print("Testing info command (results)...")

    data_file = create_test_data()
    output_file = tempfile.mktemp(suffix=".pkl")

    try:
        # First estimate
        argv_estimate = [
            "estimate",
            "--data",
            data_file,
            "--model",
            "fe",
            "--formula",
            "invest ~ value + capital",
            "--entity",
            "firm",
            "--time",
            "year",
            "--output",
            output_file,
            "--no-summary",
        ]

        exit_code = main(argv_estimate)
        assert exit_code == 0

        # Then info
        argv_info = ["info", "--results", output_file]

        exit_code = main(argv_info)
        assert exit_code == 0, "Exit code should be 0"

        print("  ✓ Info with results works")
    finally:
        Path(data_file).unlink(missing_ok=True)
        Path(output_file).unlink(missing_ok=True)


def test_estimate_verbose():
    """Test estimate with verbose flag."""
    print("Testing estimate with verbose...")

    data_file = create_test_data()

    try:
        argv = [
            "estimate",
            "--data",
            data_file,
            "--model",
            "fe",
            "--formula",
            "invest ~ value + capital",
            "--entity",
            "firm",
            "--time",
            "year",
            "--verbose",
            "--no-summary",
        ]

        exit_code = main(argv)
        assert exit_code == 0, "Exit code should be 0"

        print("  ✓ Verbose flag works")
    finally:
        Path(data_file).unlink(missing_ok=True)


def test_estimate_error_missing_column():
    """Test that missing column produces error."""
    print("Testing error handling (missing column)...")

    data_file = create_test_data()

    try:
        argv = [
            "estimate",
            "--data",
            data_file,
            "--model",
            "fe",
            "--formula",
            "invest ~ value + capital",
            "--entity",
            "nonexistent",  # Wrong column
            "--time",
            "year",
            "--no-summary",
        ]

        exit_code = main(argv)
        assert exit_code == 1, "Exit code should be 1 for error"

        print("  ✓ Error handling works")
    finally:
        Path(data_file).unlink(missing_ok=True)


def main_test():
    """Run all tests."""
    print("=" * 70)
    print("Testing PanelBox CLI")
    print("=" * 70)
    print()

    tests = [
        test_cli_help,
        test_estimate_command_basic,
        test_estimate_different_models,
        test_estimate_cov_types,
        test_estimate_json_format,
        test_info_data,
        test_info_results,
        test_estimate_verbose,
        test_estimate_error_missing_column,
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
        print("✓ All CLI tests passed!")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 70)

    return failed


if __name__ == "__main__":
    sys.exit(main_test())
