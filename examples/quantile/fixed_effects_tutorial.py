"""
Tutorial: Fixed Effects Quantile Regression for Panel Data
===========================================================

This tutorial demonstrates how to use panelbox for fixed effects quantile
regression, covering both the Koenker (2004) penalty method and the
Canay (2011) two-step estimator.

Author: PanelBox Development Team
Date: 2024
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from panelbox.diagnostics.quantile.heterogeneity import HeterogeneityTests
from panelbox.models.quantile.canay import CanayTwoStep
from panelbox.models.quantile.comparison import FEQuantileComparison
from panelbox.models.quantile.fixed_effects import FixedEffectsQuantile
from panelbox.utils.data import PanelData

# Set style for better plots
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")


# =============================================================================
# 1. GENERATE EXAMPLE DATA
# =============================================================================


def generate_panel_data(n_entities=50, n_time=20, fe_type="location_shift"):
    """
    Generate panel data with fixed effects.

    Parameters
    ----------
    n_entities : int
        Number of cross-sectional units
    n_time : int
        Number of time periods
    fe_type : str
        Type of fixed effects:
        - 'location_shift': Pure location shifters (Canay assumption holds)
        - 'heterogeneous': FE affect different quantiles differently
    """
    np.random.seed(42)

    n = n_entities * n_time

    # Panel structure
    entity_ids = np.repeat(np.arange(n_entities), n_time)
    time_ids = np.tile(np.arange(n_time), n_entities)

    # Generate covariates
    education = np.random.normal(12, 3, n)  # Years of education
    experience = np.random.uniform(0, 30, n)  # Years of experience
    experience_sq = experience**2 / 100  # Experience squared

    # Generate fixed effects
    if fe_type == "location_shift":
        # Pure location shifters - affect all quantiles equally
        ability = np.random.normal(0, 1, n_entities)  # Unobserved ability
        ability_expanded = np.repeat(ability, n_time)

        # Wage equation with location shift FE
        log_wage = (
            3.0
            + 0.08 * education  # Base wage
            + 0.05 * experience  # Return to education (8% per year)
            - 0.1 * experience_sq  # Return to experience
            + ability_expanded  # Diminishing returns
            + np.random.normal(0, 0.3, n)  # Fixed effect (location shift)  # Error term
        )

    else:  # heterogeneous
        # FE affect variance too (violates Canay assumption)
        ability_location = np.random.normal(0, 1, n_entities)
        ability_scale = np.random.uniform(0.5, 1.5, n_entities)

        log_wage = []
        for i in range(n_entities):
            mask = entity_ids == i
            n_i = np.sum(mask)

            # Entity-specific error variance
            errors = np.random.normal(0, 0.3 * ability_scale[i], n_i)

            # Entity-specific wage
            wage_i = (
                3.0
                + 0.08 * education[mask]
                + 0.05 * experience[mask]
                - 0.1 * experience_sq[mask]
                + ability_location[i]
                + errors  # Location effect  # Heteroskedastic errors
            )
            log_wage.extend(wage_i)

        log_wage = np.array(log_wage)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "log_wage": log_wage,
            "education": education,
            "experience": experience,
            "experience_sq": experience_sq,
            "entity_id": entity_ids,
            "time_id": time_ids,
        }
    )

    # Create PanelData object
    panel_df = df[["log_wage", "education", "experience", "experience_sq"]]
    panel_data = PanelData(panel_df, entity_col="entity_id", time_col="time_id")
    panel_data.entity_ids = df["entity_id"]
    panel_data.time_ids = df["time_id"]

    return panel_data, df


# =============================================================================
# 2. FIXED EFFECTS QR WITH PENALTY METHOD (KOENKER 2004)
# =============================================================================

print("=" * 70)
print("FIXED EFFECTS QUANTILE REGRESSION TUTORIAL")
print("=" * 70)

# Generate data with location shift FE
print("\n1. Generating panel data with fixed effects...")
panel_data, df = generate_panel_data(n_entities=50, n_time=20, fe_type="location_shift")
print(f"   Data shape: {df.shape}")
print(f"   Entities: {panel_data.entity_ids.nunique()}")
print(f"   Time periods: {panel_data.time_ids.nunique()}")

print("\n2. Fixed Effects QR with Koenker (2004) Penalty Method")
print("-" * 50)

# Estimate at multiple quantiles
tau_list = [0.1, 0.25, 0.5, 0.75, 0.9]
print(f"   Estimating at quantiles: {tau_list}")

# Method 1: Fixed lambda
print("\n   a) Fixed penalty parameter (λ = 0.5):")
fe_model = FixedEffectsQuantile(
    panel_data,
    formula="log_wage ~ education + experience + experience_sq",
    tau=tau_list,
    lambda_fe=0.5,
)

fe_result_fixed = fe_model.fit(verbose=False)

# Display results for median
print(f"\n   Results for τ = 0.5 (median):")
median_result = fe_result_fixed.results[0.5]
print(f"   Converged: {median_result.converged}")
print(f"   Lambda: {median_result.lambda_fe:.4f}")
print(f"   Coefficients:")
for i, name in enumerate(["Intercept", "Education", "Experience", "Experience²"]):
    print(f"     {name:12s}: {median_result.params[i]:8.4f} ({median_result.bse[i]:.4f})")

print(f"\n   Fixed Effects Distribution:")
print(f"     Mean:     {np.mean(median_result.fixed_effects):8.4f}")
print(f"     Std Dev:  {np.std(median_result.fixed_effects):8.4f}")
print(f"     # Shrunk to zero: {np.sum(np.abs(median_result.fixed_effects) < 1e-6)}")

# Method 2: Automatic lambda selection via CV
print("\n   b) Automatic λ selection via cross-validation:")
fe_model_auto = FixedEffectsQuantile(
    panel_data,
    formula="log_wage ~ education + experience + experience_sq",
    tau=0.5,  # Single quantile for CV
    lambda_fe="auto",
)

print("   Running 5-fold cross-validation...")
fe_result_auto = fe_model_auto.fit(cv_folds=5, verbose=False)

print(f"   Optimal λ selected: {fe_result_auto.results[0.5].lambda_fe:.4f}")
print(f"   CV results stored in: fe_model_auto.cv_results_")

# Plot shrinkage path
print("\n   c) Visualizing shrinkage path:")
fig = fe_model.plot_shrinkage_path(
    tau=0.5, var_names=["Intercept", "Education", "Experience", "Experience²"]
)
plt.suptitle("Coefficient Shrinkage Path for Fixed Effects QR")
plt.tight_layout()
plt.show()


# =============================================================================
# 3. CANAY (2011) TWO-STEP ESTIMATOR
# =============================================================================

print("\n3. Canay (2011) Two-Step Estimator")
print("-" * 50)

canay_model = CanayTwoStep(
    panel_data, formula="log_wage ~ education + experience + experience_sq", tau=tau_list
)

print("   Step 1: Estimating fixed effects via within-OLS...")
canay_result = canay_model.fit(se_adjustment="two-step", verbose=True)

print("\n   Fixed Effects from Step 1:")
print(f"     Number of entities: {len(canay_result.fixed_effects)}")
print(f"     FE mean: {np.mean(canay_result.fixed_effects):.4f}")
print(f"     FE std:  {np.std(canay_result.fixed_effects):.4f}")

# Test location shift assumption
print("\n   Testing location shift assumption:")
test_result = canay_model.test_location_shift(tau_grid=[0.1, 0.5, 0.9])

# Visualize fixed effects distribution
fig = canay_result.plot_fixed_effects_distribution()
plt.suptitle("Canay Two-Step: Fixed Effects Distribution")
plt.tight_layout()
plt.show()


# =============================================================================
# 4. COMPARING ESTIMATORS
# =============================================================================

print("\n4. Comparing Fixed Effects Estimators")
print("-" * 50)

comparison = FEQuantileComparison(
    panel_data, formula="log_wage ~ education + experience + experience_sq", tau=0.5
)

print("   Running comparison of methods...")
comp_results = comparison.compare_all(
    methods=["pooled", "canay", "penalty"], lambda_fe=0.5, verbose=True  # For penalty method
)

# Visualize comparison
fig = comp_results.plot_comparison()
plt.suptitle("Comparison of Fixed Effects QR Estimators")
plt.tight_layout()
plt.show()

# Coefficient correlation matrix
fig, corr_matrix = comp_results.coefficient_correlation_matrix()
plt.suptitle("Coefficient Correlation Across Methods")
plt.tight_layout()
plt.show()


# =============================================================================
# 5. TESTING FOR HETEROGENEOUS EFFECTS
# =============================================================================

print("\n5. Testing for Heterogeneous Effects Across Quantiles")
print("-" * 50)

# Need results at multiple quantiles
print("   Estimating model at multiple quantiles for testing...")
test_model = CanayTwoStep(
    panel_data,
    formula="log_wage ~ education + experience + experience_sq",
    tau=[0.1, 0.25, 0.5, 0.75, 0.9],
)
test_result = test_model.fit(verbose=False)

# Initialize heterogeneity tests
hetero_tests = HeterogeneityTests(test_result)

# Test 1: Slope equality across quantiles
print("\n   a) Testing slope equality for education:")
slope_test = hetero_tests.test_slope_equality(var_idx=1)  # Education is index 1
slope_test.summary()

# Test 2: Joint equality of all coefficients
print("\n   b) Joint test of coefficient equality:")
joint_test = hetero_tests.test_joint_equality()
joint_test.summary()

# Test 3: Monotonicity test for experience
print("\n   c) Testing monotonicity of experience effect:")
mono_test = hetero_tests.test_monotonicity(var_idx=2)  # Experience is index 2
mono_test.summary()
fig = mono_test.plot()
plt.tight_layout()
plt.show()

# Test 4: Interquantile range test
print("\n   d) Interquantile range test (heteroskedasticity):")
iqr_stat, iqr_pval = hetero_tests.interquantile_range_test()

# Visualize coefficient paths
print("\n   e) Visualizing coefficient paths across quantiles:")
fig = hetero_tests.plot_coefficient_paths(
    var_names=["Intercept", "Education", "Experience", "Experience²"]
)
plt.tight_layout()
plt.show()


# =============================================================================
# 6. BOOTSTRAP INFERENCE
# =============================================================================

print("\n6. Bootstrap Inference for Robustness")
print("-" * 50)

print("   Running bootstrap comparison (100 replications)...")
boot_stats = comparison.bootstrap_comparison(n_boot=100, methods=["canay", "penalty"])

print("\n   Bootstrap results:")
for method in boot_stats:
    print(f"\n   {method}:")
    print(f"     Coverage: {boot_stats[method]['coverage']:.1%}")
    print(f"     Mean coefficients: {boot_stats[method]['mean']}")
    print(f"     Std errors:        {boot_stats[method]['std']}")


# =============================================================================
# 7. PERFORMANCE ANALYSIS
# =============================================================================

print("\n7. Performance Analysis")
print("-" * 50)

# Test with larger data
print("   Generating larger dataset for performance testing...")
large_panel, _ = generate_panel_data(n_entities=100, n_time=50)

from panelbox.optimization.quantile.penalized import PerformanceMonitor

monitor = PerformanceMonitor()

print("   Profiling methods...")
results = monitor.compare_implementations(
    large_panel,
    formula="log_wage ~ education + experience + experience_sq",
    tau=0.5,
    methods=["canay", "penalty"],
)

monitor.print_report()


# =============================================================================
# 8. HANDLING VIOLATIONS OF ASSUMPTIONS
# =============================================================================

print("\n8. Handling Violations of Location Shift Assumption")
print("-" * 50)

# Generate data that violates location shift
print("   Generating data with heterogeneous fixed effects...")
hetero_panel, _ = generate_panel_data(n_entities=40, n_time=15, fe_type="heterogeneous")

# Test with Canay
print("\n   a) Canay estimator (assumes location shift):")
canay_hetero = CanayTwoStep(
    hetero_panel, formula="log_wage ~ education + experience + experience_sq", tau=[0.1, 0.5, 0.9]
)
canay_hetero_result = canay_hetero.fit(verbose=False)

# Test location shift assumption
print("   Testing location shift assumption...")
ls_test = canay_hetero.test_location_shift()

if ls_test.p_value < 0.05:
    print("\n   ⚠ Warning: Location shift assumption violated!")
    print("   Consider using penalty method instead.")

    # Use penalty method
    print("\n   b) Fixed Effects QR with penalty (robust to violation):")
    fe_hetero = FixedEffectsQuantile(
        hetero_panel,
        formula="log_wage ~ education + experience + experience_sq",
        tau=[0.1, 0.5, 0.9],
        lambda_fe="auto",
    )
    fe_hetero_result = fe_hetero.fit(cv_folds=3, verbose=False)

    print("   Penalty method results at τ=0.5:")
    print(f"     Optimal λ: {fe_hetero_result.results[0.5].lambda_fe:.4f}")
    print(f"     Converged: {fe_hetero_result.results[0.5].converged}")


# =============================================================================
# 9. PRACTICAL RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 70)
print("PRACTICAL RECOMMENDATIONS")
print("=" * 70)

print(
    """
1. CHOOSING BETWEEN METHODS:

   - Canay Two-Step:
     * Fast and simple
     * Good for large T (> 20)
     * Requires location shift assumption
     * Use test_location_shift() to validate

   - Koenker Penalty Method:
     * More flexible
     * Better for small T
     * Robust to heterogeneous effects
     * Requires lambda selection (use CV)

2. LAMBDA SELECTION FOR PENALTY METHOD:
   - Start with lambda='auto' for CV selection
   - Larger lambda → more shrinkage
   - Very small lambda → numerical instability
   - Monitor number of zero fixed effects

3. DIAGNOSTIC CHECKS:
   - Always test location shift assumption for Canay
   - Check coefficient stability across quantiles
   - Use heterogeneity tests to justify QR over OLS
   - Bootstrap for inference with small samples

4. PERFORMANCE TIPS:
   - Canay is 5-10x faster than penalty method
   - Use warm starts for multiple lambda values
   - Consider Numba optimization for large panels
   - Profile with PerformanceMonitor for method selection

5. COMMON PITFALLS:
   - Ignoring location shift test warnings
   - Using Canay with small T (< 10)
   - Not adjusting SE for two-step estimation
   - Over-interpreting FE estimates with large lambda
"""
)

print("\nTutorial completed successfully!")
print("=" * 70)
