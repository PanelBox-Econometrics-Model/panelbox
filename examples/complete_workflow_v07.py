"""
Complete Workflow Example - PanelBox v0.7.0
============================================

This example demonstrates the complete workflow for panel data analysis using
PanelBox v0.7.0, including all three result containers:
- ValidationResult: Model specification tests
- ComparisonResult: Model comparison and selection
- ResidualResult: Residual diagnostics (NEW in v0.7.0!)

The example uses the Grunfeld investment dataset and demonstrates:
1. Data loading and preparation
2. Model fitting with PanelExperiment
3. Model validation
4. Model comparison
5. Residual diagnostics
6. HTML report generation

Author: PanelBox Development Team
Date: 2026-02-08
"""

import numpy as np
import pandas as pd

import panelbox as pb

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 80)
print("PanelBox v0.7.0 - Complete Workflow Example")
print("=" * 80)
print()

# =============================================================================
# Step 1: Load and Prepare Data
# =============================================================================
print("Step 1: Loading Grunfeld investment dataset...")
print("-" * 80)

# Load the dataset
data = pb.load_grunfeld()
print(f"Dataset shape: {data.shape}")
print(f"Panel structure: {data['firm'].nunique()} firms, {data['year'].nunique()} years")
print(f"\nFirst few rows:")
print(data.head())
print()

# =============================================================================
# Step 2: Create PanelExperiment
# =============================================================================
print("Step 2: Creating PanelExperiment...")
print("-" * 80)

# Create experiment with investment as dependent variable
experiment = pb.PanelExperiment(
    data=data, formula="invest ~ value + capital", entity_col="firm", time_col="year"
)

print("✓ Experiment created successfully")
print(f"  Formula: invest ~ value + capital")
print(f"  Entity column: firm")
print(f"  Time column: year")
print()

# =============================================================================
# Step 3: Fit Multiple Models
# =============================================================================
print("Step 3: Fitting multiple panel models...")
print("-" * 80)

# Fit all three basic models at once
experiment.fit_all_models(names=["pooled", "fe", "re"])

print("✓ Models fitted:")
for model_name in experiment.list_models():
    model = experiment.get_model(model_name)
    print(f"  - {model_name}: R² = {model.rsquared:.4f}")
print()

# =============================================================================
# Step 4: Validate Model Specification (ValidationResult)
# =============================================================================
print("Step 4: Validating Fixed Effects model...")
print("-" * 80)

# Validate the fixed effects model
validation_result = experiment.validate_model("fe")

print("✓ ValidationResult created")
print("\nValidation Summary:")
print(validation_result.summary())
print()

# Save validation HTML report
validation_html = "validation_report_v07.html"
validation_result.save_html(validation_html, test_type="validation")
print(f"✓ Validation report saved to: {validation_html}")
print()

# =============================================================================
# Step 5: Compare Models (ComparisonResult)
# =============================================================================
print("Step 5: Comparing all models...")
print("-" * 80)

# Compare all fitted models
comparison_result = experiment.compare_models(["pooled", "fe", "re"])

print("✓ ComparisonResult created")
best_by_aic = comparison_result.best_model("aic", prefer_lower=True)
best_by_r2 = comparison_result.best_model("rsquared_adj", prefer_lower=False)
print(f"\nBest model by AIC: {best_by_aic}")
print(f"Best model by Adjusted R²: {best_by_r2}")
print()

# Show comparison summary
print("Comparison Summary:")
print(comparison_result.summary())
print()

# Save comparison HTML report
comparison_html = "comparison_report_v07.html"
comparison_result.save_html(comparison_html, test_type="comparison")
print(f"✓ Comparison report saved to: {comparison_html}")
print()

# =============================================================================
# Step 6: Analyze Residuals (ResidualResult) - NEW in v0.7.0!
# =============================================================================
print("Step 6: Analyzing residuals (NEW in v0.7.0!)...")
print("-" * 80)

# Analyze residuals of the best model (fixed effects)
residual_result = experiment.analyze_residuals("fe")

print("✓ ResidualResult created")
print("\nResidual Diagnostics Summary:")
print(residual_result.summary())
print()

# Access individual diagnostic tests
print("Individual Test Results:")
print("-" * 40)

# Shapiro-Wilk test for normality
stat, pvalue = residual_result.shapiro_test
print(f"Shapiro-Wilk Test:")
print(f"  Statistic: {stat:.6f}")
print(f"  P-value: {pvalue:.6f}")
print(
    f"  Interpretation: {'Residuals are normal' if pvalue > 0.05 else 'Residuals are NOT normal'}"
)
print()

# Jarque-Bera test for normality
stat, pvalue = residual_result.jarque_bera
print(f"Jarque-Bera Test:")
print(f"  Statistic: {stat:.6f}")
print(f"  P-value: {pvalue:.6f}")
print(
    f"  Interpretation: {'Residuals are normal' if pvalue > 0.05 else 'Residuals are NOT normal'}"
)
print()

# Durbin-Watson test for autocorrelation
dw = residual_result.durbin_watson
print(f"Durbin-Watson Statistic: {dw:.6f}")
if dw < 1.5:
    interpretation = "Positive autocorrelation detected"
elif dw > 2.5:
    interpretation = "Negative autocorrelation detected"
else:
    interpretation = "No significant autocorrelation"
print(f"  Interpretation: {interpretation}")
print()

# Ljung-Box test for autocorrelation
stat, pvalue = residual_result.ljung_box
print(f"Ljung-Box Test (10 lags):")
print(f"  Statistic: {stat:.6f}")
print(f"  P-value: {pvalue:.6f}")
print(
    f"  Interpretation: {'No serial correlation' if pvalue > 0.05 else 'Serial correlation detected'}"
)
print()

# Summary statistics
print("Summary Statistics:")
print("-" * 40)
print(f"  Mean: {residual_result.mean:.6f} (should be ≈ 0)")
print(f"  Std Dev: {residual_result.std:.6f}")
print(f"  Skewness: {residual_result.skewness:.6f} (should be ≈ 0 for normality)")
print(f"  Kurtosis: {residual_result.kurtosis:.6f} (should be ≈ 3 for normality)")
print(f"  Min: {residual_result.min:.6f}")
print(f"  Max: {residual_result.max:.6f}")
print()

# Save residual diagnostics HTML report
residuals_html = "residuals_report_v07.html"
residual_result.save_html(residuals_html, test_type="residuals")
print(f"✓ Residual diagnostics report saved to: {residuals_html}")
print()

# =============================================================================
# Step 7: Export Results to JSON
# =============================================================================
print("Step 7: Exporting results to JSON...")
print("-" * 80)

# Export all results to JSON for further analysis
validation_result.save_json("validation_results_v07.json")
comparison_result.save_json("comparison_results_v07.json")
residual_result.save_json("residual_results_v07.json")

print("✓ JSON files created:")
print("  - validation_results_v07.json")
print("  - comparison_results_v07.json")
print("  - residual_results_v07.json")
print()

# =============================================================================
# Summary
# =============================================================================
print("=" * 80)
print("Workflow Complete!")
print("=" * 80)
print()
print("Summary of Generated Files:")
print("-" * 40)
print("HTML Reports:")
print(f"  ✓ {validation_html} - Validation diagnostics")
print(f"  ✓ {comparison_html} - Model comparison")
print(f"  ✓ {residuals_html} - Residual diagnostics (NEW!)")
print()
print("JSON Exports:")
print("  ✓ validation_results_v07.json")
print("  ✓ comparison_results_v07.json")
print("  ✓ residual_results_v07.json")
print()
print("=" * 80)
print("Complete Result Container Trilogy:")
print("=" * 80)
print()
print("1. ValidationResult - Model specification tests")
print("   Created via: experiment.validate_model('fe')")
print("   Tests: Hausman, heteroskedasticity, autocorrelation, etc.")
print()
print("2. ComparisonResult - Model comparison and selection")
print("   Created via: experiment.compare_models(['pooled', 'fe', 're'])")
print("   Provides: Best model selection, AIC/BIC comparison")
print()
print("3. ResidualResult - Residual diagnostics (NEW in v0.7.0!)")
print("   Created via: experiment.analyze_residuals('fe')")
print("   Tests: Shapiro-Wilk, Jarque-Bera, Durbin-Watson, Ljung-Box")
print()
print("=" * 80)
print("Key Features Demonstrated:")
print("=" * 80)
print()
print("✓ One-liner workflows:")
print("  - experiment.fit_all_models()")
print("  - experiment.validate_model(name)")
print("  - experiment.compare_models(names)")
print("  - experiment.analyze_residuals(name) [NEW!]")
print()
print("✓ Professional HTML reports with interactive charts")
print("✓ JSON export for programmatic analysis")
print("✓ Comprehensive diagnostic tests")
print("✓ Clean, production-ready API")
print()
print("=" * 80)
print("Next Steps:")
print("=" * 80)
print()
print("1. Open the HTML reports in your browser to see interactive visualizations")
print("2. Inspect the JSON files for programmatic analysis")
print("3. Modify the formula and run your own experiments")
print("4. Explore advanced features:")
print("   - GMM models for dynamic panels")
print("   - Robust standard errors")
print("   - Bootstrap inference")
print("   - Sensitivity analysis")
print()
print("Documentation: https://github.com/PanelBox-Econometrics-Model/panelbox")
print("=" * 80)
