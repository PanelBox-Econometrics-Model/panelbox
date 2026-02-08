#!/usr/bin/env python3
"""Test script to execute the complete workflow from tutorial notebook."""

import sys

sys.path.insert(0, ".")

import warnings

import numpy as np
import pandas as pd

import panelbox as pb

warnings.filterwarnings("ignore")

print(f"PanelBox version: {pb.__version__}")
print(f"Expected: 0.8.0 or higher")

# Load data
print("\n=== Loading Grunfeld dataset ===")
data = pb.load_grunfeld()
print(f"Dataset shape: {data.shape}")
print(f"Panel structure: {data['firm'].nunique()} firms, {data['year'].nunique()} years")

# Create experiment
print("\n=== Creating PanelExperiment ===")
experiment = pb.PanelExperiment(
    data=data, formula="invest ~ value + capital", entity_col="firm", time_col="year"
)
print(experiment)

# Fit models
print("\n=== Fitting models ===")
experiment.fit_model("pooled_ols", name="ols")
print("✓ Pooled OLS fitted")
experiment.fit_model("fixed_effects", name="fe")
print("✓ Fixed Effects fitted")
experiment.fit_model("random_effects", name="re")
print("✓ Random Effects fitted")

print(f"\nModels fitted: {experiment.list_models()}")

# Model summaries
print("\n=== Model Summaries ===")
for model_name in experiment.list_models():
    results = experiment.get_model(model_name)
    print(f"\nModel: {model_name.upper()}")
    print(f"  R²: {results.rsquared:.4f}")
    print(f"  Adj. R²: {getattr(results, 'rsquared_adj', results.rsquared):.4f}")
    if hasattr(results, "aic"):
        print(f"  AIC: {results.aic:.2f}")
    if hasattr(results, "bic"):
        print(f"  BIC: {results.bic:.2f}")

# ValidationTest runner
print("\n=== ValidationTest Runner (v0.8.0) ===")
from panelbox.experiment.tests import ValidationTest

validation_runner = ValidationTest()
print(f"Available configs: {list(validation_runner.CONFIGS.keys())}")

validation_result = experiment.validate_model("fe")
print("✓ Validation completed")
if hasattr(validation_result.validation_report, "__len__"):
    print(f"  Tests run: {len(validation_result.validation_report)}")
else:
    print(f"  Tests run: {len(validation_result.validation_report.to_dict())}")

# Save validation report
val_path = validation_result.save_html(
    "validation_report_tutorial.html", test_type="validation", theme="professional"
)
print(f"✓ Validation report saved: {val_path}")

# ComparisonTest runner
print("\n=== ComparisonTest Runner (v0.8.0) ===")
from panelbox.experiment.tests import ComparisonTest

comparison_runner = ComparisonTest()
comparison_result = experiment.compare_models(["ols", "fe", "re"])
print("✓ Comparison completed")
print(f"  Models compared: {len(comparison_result.models)}")

# Best models
best_aic = comparison_result.best_model("aic", prefer_lower=True)
print(f"  Best by AIC: {best_aic}")

best_r2 = comparison_result.best_model("rsquared_adj", prefer_lower=False)
print(f"  Best by Adj. R²: {best_r2}")

# Save comparison report
comp_path = comparison_result.save_html(
    "comparison_report_tutorial.html", test_type="comparison", theme="professional"
)
print(f"✓ Comparison report saved: {comp_path}")

# Residual diagnostics
print("\n=== Residual Diagnostics (v0.7.0) ===")
residual_result = experiment.analyze_residuals("fe")
print("✓ Residual analysis completed")

# Diagnostic tests
stat, pvalue = residual_result.shapiro_test
print(f"  Shapiro-Wilk: stat={stat:.6f}, p={pvalue:.6f}")

dw = residual_result.durbin_watson
print(f"  Durbin-Watson: {dw:.6f}")

stat, pvalue = residual_result.jarque_bera
print(f"  Jarque-Bera: stat={stat:.6f}, p={pvalue:.6f}")

stat, pvalue = residual_result.ljung_box
print(f"  Ljung-Box: stat={stat:.6f}, p={pvalue:.6f}")

# Save residuals report
res_path = residual_result.save_html(
    "residuals_report_tutorial.html", test_type="residuals", theme="professional"
)
print(f"✓ Residuals report saved: {res_path}")

# Master report
print("\n=== Master Report Generation (v0.8.0) ===")
master_path = experiment.save_master_report(
    "master_report_tutorial.html",
    theme="professional",
    title="PanelBox v0.8.0 - Complete Analysis Tutorial",
    reports=[
        {
            "type": "validation",
            "title": "Fixed Effects Validation",
            "description": "Specification tests for FE model",
            "file_path": "validation_report_tutorial.html",
        },
        {
            "type": "comparison",
            "title": "Model Comparison",
            "description": "Compare Pooled OLS, FE, and RE models",
            "file_path": "comparison_report_tutorial.html",
        },
        {
            "type": "residuals",
            "title": "Residual Diagnostics",
            "description": "Diagnostic tests for FE model residuals",
            "file_path": "residuals_report_tutorial.html",
        },
    ],
)
print(f"✓ Master report saved: {master_path}")

# Export to JSON
print("\n=== Export to JSON ===")
validation_result.save_json("validation_tutorial.json")
comparison_result.save_json("comparison_tutorial.json")
residual_result.save_json("residuals_tutorial.json")
print("✓ All results exported to JSON")

# Different themes
print("\n=== Testing Different Themes ===")
validation_result.save_html(
    "validation_academic.html",
    test_type="validation",
    theme="academic",
    title="Validation Report - Academic Style",
)
print("✓ Academic theme report saved")

comparison_result.save_html(
    "comparison_presentation.html",
    test_type="comparison",
    theme="presentation",
    title="Model Comparison - Presentation Style",
)
print("✓ Presentation theme report saved")

print("\n" + "=" * 60)
print("✅ TUTORIAL EXECUTION COMPLETE!")
print("=" * 60)
print("\nAll v0.8.0 features tested successfully:")
print("  ✓ PanelExperiment with multiple models")
print("  ✓ ValidationTest runner")
print("  ✓ ComparisonTest runner")
print("  ✓ Residual diagnostics")
print("  ✓ Master report generation")
print("  ✓ Multiple themes (professional, academic, presentation)")
print("  ✓ JSON export")
print("\nGenerated files:")
print("  - master_report_tutorial.html")
print("  - validation_report_tutorial.html")
print("  - comparison_report_tutorial.html")
print("  - residuals_report_tutorial.html")
print("  - validation_academic.html")
print("  - comparison_presentation.html")
print("  - validation_tutorial.json")
print("  - comparison_tutorial.json")
print("  - residuals_tutorial.json")
