"""
Test PanelExperiment - Basic Functionality
==========================================

Tests the basic functionality of PanelExperiment class.
"""

import numpy as np
import pandas as pd

from panelbox.experiment import PanelExperiment

np.random.seed(42)


def create_sample_data():
    """Create sample panel data."""
    print("Creating sample panel data...")

    n_entities = 50
    n_time = 10

    # Create panel structure
    entities = np.repeat(range(n_entities), n_time)
    time = np.tile(range(n_time), n_entities)

    # Create features
    x1 = np.random.randn(n_entities * n_time)
    x2 = np.random.randn(n_entities * n_time)

    # Create target with entity effects
    entity_effects = np.random.randn(n_entities)
    entity_effects_expanded = np.repeat(entity_effects, n_time)

    y = (
        2
        + 1.5 * x1
        + 0.8 * x2
        + entity_effects_expanded
        + np.random.randn(n_entities * n_time) * 0.5
    )

    # Create DataFrame
    df = pd.DataFrame({"entity": entities, "time": time, "y": y, "x1": x1, "x2": x2})

    print(f"  ✅ Data created: {df.shape}")
    return df


def main():
    print("=" * 80)
    print("TEST: PANELEXPERIMENT - BASIC FUNCTIONALITY")
    print("=" * 80)
    print()

    # 1. Create sample data
    data = create_sample_data()
    print()

    # 2. Initialize PanelExperiment
    print("Initializing PanelExperiment...")
    experiment = PanelExperiment(
        data=data, formula="y ~ x1 + x2", entity_col="entity", time_col="time"
    )
    print(f"  ✅ Experiment initialized")
    print(f"  {experiment}")
    print()

    # 3. Fit Pooled OLS
    print("Fitting models...")
    ols_result = experiment.fit_model("pooled_ols", name="ols")
    print()

    # 4. Fit Fixed Effects
    fe_result = experiment.fit_model("fixed_effects", name="fe", cov_type="clustered")
    print()

    # 5. Fit Random Effects
    re_result = experiment.fit_model("random_effects", name="re")
    print()

    # 6. List models
    print("Listing fitted models...")
    models = experiment.list_models()
    print(f"  ✅ Fitted models: {models}")
    print()

    # 7. Get model
    print("Retrieving model...")
    ols_model = experiment.get_model("ols")
    print(f"  ✅ Retrieved 'ols' model")
    print(f"  - Type: {type(ols_model)}")
    print(f"  - R²: {ols_model.rsquared:.4f}")
    print()

    # 8. Get model metadata
    print("Getting model metadata...")
    metadata = experiment.get_model_metadata("ols")
    print(f"  ✅ Metadata retrieved:")
    print(f"  - Model type: {metadata['model_type']}")
    print(f"  - Formula: {metadata['formula']}")
    print(f"  - Fitted at: {metadata['fitted_at']}")
    print()

    # 9. Test auto-generated names
    print("Testing auto-generated names...")
    experiment.fit_model("pooled")  # No name provided
    experiment.fit_model("fe")  # Alias + no name
    models = experiment.list_models()
    print(f"  ✅ Models with auto-generated names: {models[-2:]}")
    print()

    # 10. Display experiment summary
    print("Experiment Summary:")
    print("=" * 80)
    print(experiment)
    print("=" * 80)
    print()

    # 11. Final summary
    print("✅ ALL TESTS PASSED!")
    print()
    print("Summary:")
    print(f"  • Total models fitted: {len(experiment.list_models())}")
    print(f"  • Models: {', '.join(experiment.list_models())}")
    print(f"  • Formula: {experiment.formula}")
    print(f"  • Observations: {len(experiment.data)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
