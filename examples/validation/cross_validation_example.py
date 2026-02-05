"""
Time-Series Cross-Validation for Panel Data - Example

This example demonstrates how to use TimeSeriesCV to evaluate
out-of-sample predictive performance of panel data models.

Cross-validation respects the temporal structure of the data,
essential for time series applications.
"""

import numpy as np
import pandas as pd

import panelbox as pb

print("=" * 80)
print("Time-Series Cross-Validation Example")
print("=" * 80)
print()

# Generate sample panel data
np.random.seed(42)
n_entities = 20
n_periods = 12

data = []
for entity in range(n_entities):
    # Entity-specific effects
    alpha_i = np.random.normal(0, 1)

    for time in range(n_periods):
        x1 = np.random.normal(0, 1)
        x2 = np.random.normal(0, 1)

        # DGP: y = 2.0 + 1.5*x1 - 1.0*x2 + alpha_i + error
        y = 2.0 + 1.5 * x1 - 1.0 * x2 + alpha_i + np.random.normal(0, 0.5)

        data.append({"entity": entity, "time": time, "y": y, "x1": x1, "x2": x2})

df = pd.DataFrame(data)

print("1. Dataset Summary:")
print("-" * 80)
print(f"   Entities: {df['entity'].nunique()}")
print(f"   Time periods: {df['time'].nunique()}")
print(f"   Total observations: {len(df)}")
print()

# Fit Fixed Effects model
print("2. Fitting Fixed Effects Model:")
print("-" * 80)
fe = pb.FixedEffects("y ~ x1 + x2", df, "entity", "time")
results = fe.fit()
print(results.summary())
print()

# Expanding Window Cross-Validation
print("3. Expanding Window Cross-Validation:")
print("=" * 80)
cv_expanding = pb.TimeSeriesCV(results, method="expanding", min_train_periods=5, verbose=True)

cv_results_expanding = cv_expanding.cross_validate()
print()
print(cv_expanding.summary())
print()

# Rolling Window Cross-Validation
print("\n4. Rolling Window Cross-Validation:")
print("=" * 80)
cv_rolling = pb.TimeSeriesCV(
    results, method="rolling", window_size=6, min_train_periods=5, verbose=True
)

cv_results_rolling = cv_rolling.cross_validate()
print()
print(cv_rolling.summary())
print()

# Compare methods
print("\n5. Comparing CV Methods:")
print("=" * 80)
print(f"{'Method':<20} {'R² (OOS)':<12} {'RMSE':<12} {'MAE':<12}")
print("-" * 80)
print(
    f"{'Expanding':<20} {cv_results_expanding.metrics['r2_oos']:>11.4f} "
    f"{cv_results_expanding.metrics['rmse']:>11.4f} "
    f"{cv_results_expanding.metrics['mae']:>11.4f}"
)
print(
    f"{'Rolling (w=6)':<20} {cv_results_rolling.metrics['r2_oos']:>11.4f} "
    f"{cv_results_rolling.metrics['rmse']:>11.4f} "
    f"{cv_results_rolling.metrics['mae']:>11.4f}"
)
print()

# Interpretation
print("\n6. Interpretation:")
print("=" * 80)
print("Out-of-Sample R²:")
if cv_results_expanding.metrics["r2_oos"] > 0.5:
    print("   ✓ Strong out-of-sample predictive performance (R² > 0.5)")
    print("   → Model generalizes well to unseen data")
elif cv_results_expanding.metrics["r2_oos"] > 0:
    print("   ⚠ Moderate out-of-sample performance (0 < R² < 0.5)")
    print("   → Some predictive power, but room for improvement")
else:
    print("   ✗ Poor out-of-sample performance (R² < 0)")
    print("   → Model performs worse than naïve mean prediction")
print()

print("RMSE:")
in_sample_rmse = np.std(results.resid)
oos_rmse = cv_results_expanding.metrics["rmse"]
rmse_ratio = oos_rmse / in_sample_rmse

print(f"   In-sample RMSE:  {in_sample_rmse:.4f}")
print(f"   Out-of-sample RMSE: {oos_rmse:.4f}")
print(f"   Ratio (OOS/IS):  {rmse_ratio:.2f}")

if rmse_ratio < 1.2:
    print("   ✓ OOS performance close to in-sample (ratio < 1.2)")
    print("   → Model is not overfitting")
else:
    print("   ⚠ OOS performance worse than in-sample (ratio > 1.2)")
    print("   → Possible overfitting")
print()

print("Per-Fold Analysis:")
print(f"   Number of folds: {cv_results_expanding.n_folds}")
print(f"   Mean R²: {cv_results_expanding.fold_metrics['r2_oos'].mean():.4f}")
print(f"   Std R²:  {cv_results_expanding.fold_metrics['r2_oos'].std():.4f}")
print()

# Optional: Plot predictions (requires matplotlib)
try:
    import matplotlib.pyplot as plt

    print("\n7. Plotting Predictions:")
    print("=" * 80)
    cv_expanding.plot_predictions(save_path="cv_predictions.png")
    print("   Plot saved to: cv_predictions.png")
except ImportError:
    print("\n7. Plotting skipped (matplotlib not installed)")
    print("   Install with: pip install matplotlib")

print()
print("=" * 80)
print("Cross-Validation Complete!")
print("=" * 80)
print()
print("Key Takeaways:")
print("- Expanding window CV trains on growing history")
print("- Rolling window CV uses fixed window size")
print("- Out-of-sample R² evaluates predictive performance")
print("- Compare OOS vs in-sample to detect overfitting")
print()
