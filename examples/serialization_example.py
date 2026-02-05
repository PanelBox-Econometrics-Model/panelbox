"""
Example: Serialization and Persistence of PanelResults

This example demonstrates how to save and load estimation results
using the serialization functionality in PanelBox.
"""

import tempfile
from pathlib import Path

import panelbox as pb


def example_basic_save_load():
    """Basic save and load example."""
    print("=" * 70)
    print("Example 1: Basic Save and Load")
    print("=" * 70)

    # Load data
    data = pb.load_grunfeld()

    # Fit model
    fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    results = fe.fit(cov_type="robust")

    print("\nOriginal results:")
    print(results.summary())

    # Save results
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        results.save(filepath)
        print(f"\n✓ Results saved to: {filepath}")

        # Load results
        loaded_results = pb.PanelResults.load(filepath)
        print("\n✓ Results loaded successfully")

        # Verify they are identical
        print(f"\nOriginal R²: {results.rsquared:.4f}")
        print(f"Loaded R²:   {loaded_results.rsquared:.4f}")
        print(f"Match: {results.rsquared == loaded_results.rsquared}")
    finally:
        Path(filepath).unlink(missing_ok=True)


def example_json_export():
    """JSON export example."""
    print("\n" + "=" * 70)
    print("Example 2: JSON Export")
    print("=" * 70)

    data = pb.load_grunfeld()
    pooled = pb.PooledOLS("invest ~ value + capital", data, "firm", "year")
    results = pooled.fit()

    # Export to JSON string
    json_str = results.to_json()
    print("\nJSON preview (first 300 chars):")
    print(json_str[:300] + "...")

    # Save to file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name

    try:
        results.save(filepath, format="json")
        print(f"\n✓ JSON saved to: {filepath}")

        # Read back
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        print(f"\n✓ JSON loaded, contains {len(data.keys())} top-level keys")
        print(f"  Keys: {', '.join(data.keys())}")
    finally:
        Path(filepath).unlink(missing_ok=True)


def example_to_dict():
    """Dictionary conversion example."""
    print("\n" + "=" * 70)
    print("Example 3: Dictionary Conversion")
    print("=" * 70)

    data = pb.load_grunfeld()
    be = pb.BetweenEstimator("invest ~ value + capital", data, "firm", "year")
    results = be.fit()

    # Convert to dictionary
    results_dict = results.to_dict()

    print("\nResults as dictionary:")
    print(f"  Model type: {results_dict['model_info']['model_type']}")
    print(f"  Formula: {results_dict['model_info']['formula']}")
    print(f"  Observations: {results_dict['sample_info']['nobs']}")
    print(f"  Parameters: {list(results_dict['params'].keys())}")
    print(f"  R²: {results_dict['rsquared']['rsquared']:.4f}")


def example_compare_models():
    """Compare multiple models by saving and loading."""
    print("\n" + "=" * 70)
    print("Example 4: Compare Multiple Models")
    print("=" * 70)

    data = pb.load_grunfeld()

    # Fit multiple models
    models = {
        "Pooled OLS": pb.PooledOLS("invest ~ value + capital", data, "firm", "year"),
        "Fixed Effects": pb.FixedEffects("invest ~ value + capital", data, "firm", "year"),
        "Between": pb.BetweenEstimator("invest ~ value + capital", data, "firm", "year"),
        "First Diff": pb.FirstDifferenceEstimator("invest ~ value + capital", data, "firm", "year"),
    }

    # Fit and save all models
    print("\nFitting and saving models...")
    temp_files = {}
    for name, model in models.items():
        results = model.fit()
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            filepath = f.name
            temp_files[name] = filepath
            results.save(filepath)
        print(f"  ✓ {name:15s} saved to {Path(filepath).name}")

    # Load and compare
    print("\nLoading and comparing models:")
    print(f"{'Model':<15s} {'R² Within':<12s} {'R² Between':<12s} {'R² Overall':<12s}")
    print("-" * 55)

    try:
        for name, filepath in temp_files.items():
            results = pb.PanelResults.load(filepath)

            r2_within = (
                f"{results.rsquared_within:.4f}" if not pd.isna(results.rsquared_within) else "N/A"
            )
            r2_between = (
                f"{results.rsquared_between:.4f}"
                if not pd.isna(results.rsquared_between)
                else "N/A"
            )
            r2_overall = (
                f"{results.rsquared_overall:.4f}"
                if not pd.isna(results.rsquared_overall)
                else "N/A"
            )

            print(f"{name:<15s} {r2_within:<12s} {r2_between:<12s} {r2_overall:<12s}")
    finally:
        # Clean up
        for filepath in temp_files.values():
            Path(filepath).unlink(missing_ok=True)


def example_workflow():
    """Real-world workflow example."""
    print("\n" + "=" * 70)
    print("Example 5: Real-World Workflow")
    print("=" * 70)

    # Step 1: Estimate model
    print("\nStep 1: Estimate model")
    data = pb.load_grunfeld()
    fe = pb.FixedEffects("invest ~ value + capital", data, "firm", "year")
    results = fe.fit(cov_type="robust")
    print(f"  ✓ Model estimated (R² = {results.rsquared:.4f})")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name

    try:
        # Step 2: Save results for later
        print("\nStep 2: Save results")
        results.save(filepath)
        print(f"  ✓ Results saved to disk")

        # Step 3: Simulate closing and reopening session
        print("\nStep 3: Load results in new session")
        loaded_results = pb.PanelResults.load(filepath)
        print(f"  ✓ Results loaded")

        # Step 4: Continue analysis with loaded results
        print("\nStep 4: Continue analysis")
        ci = loaded_results.conf_int()
        print("\n95% Confidence Intervals:")
        print(ci)

        # Step 5: Export to JSON for sharing
        json_file = filepath.replace(".pkl", ".json")
        loaded_results.save(json_file, format="json")
        print(f"\n  ✓ Also exported to JSON: {Path(json_file).name}")

        Path(json_file).unlink(missing_ok=True)
    finally:
        Path(filepath).unlink(missing_ok=True)


def main():
    """Run all examples."""
    import pandas as pd

    globals()["pd"] = pd  # For example 4

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " PanelBox Serialization Examples ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    example_basic_save_load()
    example_json_export()
    example_to_dict()
    example_compare_models()
    example_workflow()

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
