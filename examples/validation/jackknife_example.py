"""
Jackknife Inference for Panel Data - Example

This example demonstrates how to use PanelJackknife to estimate
bias and variance of panel data model parameters, and identify
influential entities.

The jackknife is computationally lighter than bootstrap and provides
good estimates for small to medium samples.
"""

import numpy as np
import pandas as pd

import panelbox as pb

print("=" * 80)
print("Jackknife Inference Example")
print("=" * 80)
print()

# Generate sample panel data
np.random.seed(42)
n_entities = 20
n_periods = 8

data = []
for entity in range(n_entities):
    # Entity-specific effect
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

# Jackknife Inference
print("\n3. Jackknife Inference:")
print("=" * 80)
jackknife = pb.PanelJackknife(results, verbose=True)
jk_results = jackknife.run()
print()

# Display jackknife summary
print(jackknife.summary())
print()

# Bias-corrected estimates
print("\n4. Bias-Corrected Estimates:")
print("=" * 80)
bias_corrected = jackknife.bias_corrected_estimates()

print(f"{'Parameter':<15} {'Original':>12} {'Bias':>12} {'Corrected':>12}")
print("-" * 80)
for param in results.params.index:
    print(
        f"{param:<15} {results.params[param]:>12.6f} "
        f"{jk_results.jackknife_bias[param]:>12.6f} "
        f"{bias_corrected[param]:>12.6f}"
    )
print()

# Comparison of standard errors
print("\n5. Standard Error Comparison:")
print("=" * 80)
print(f"{'Parameter':<15} {'Asymptotic':>12} {'Jackknife':>12} {'Ratio (JK/AS)':>15}")
print("-" * 80)
for param in results.params.index:
    ratio = jk_results.jackknife_se[param] / results.std_errors[param]
    print(
        f"{param:<15} {results.std_errors[param]:>12.6f} "
        f"{jk_results.jackknife_se[param]:>12.6f} "
        f"{ratio:>15.3f}"
    )
print()

# Interpretation
print("Interpretation:")
avg_ratio = (jk_results.jackknife_se / results.std_errors).mean()
if abs(avg_ratio - 1.0) < 0.1:
    print("   ✓ Jackknife and asymptotic SEs are similar (ratio ≈ 1.0)")
    print("   → Asymptotic approximation appears adequate")
elif avg_ratio > 1.1:
    print("   ⚠ Jackknife SEs larger than asymptotic (ratio > 1.1)")
    print("   → Consider using jackknife SEs for more conservative inference")
else:
    print("   ⚠ Jackknife SEs smaller than asymptotic (ratio < 0.9)")
    print("   → Unusual, investigate potential issues")
print()

# Confidence intervals
print("\n6. Confidence Intervals (95%):")
print("=" * 80)

# Asymptotic CI
ci_asymptotic = results.conf_int(alpha=0.05)

# Jackknife CI (normal approximation)
ci_jackknife = jackknife.confidence_intervals(alpha=0.05, method="normal")

print(f"{'Parameter':<15} {'Method':<15} {'Lower':>12} {'Upper':>12} {'Width':>12}")
print("-" * 80)
for param in results.params.index:
    # Asymptotic
    width_asym = ci_asymptotic.loc[param, "upper"] - ci_asymptotic.loc[param, "lower"]
    print(
        f"{param:<15} {'Asymptotic':<15} "
        f"{ci_asymptotic.loc[param, 'lower']:>12.6f} "
        f"{ci_asymptotic.loc[param, 'upper']:>12.6f} "
        f"{width_asym:>12.6f}"
    )

    # Jackknife
    width_jk = ci_jackknife.loc[param, "upper"] - ci_jackknife.loc[param, "lower"]
    print(
        f"{'':<15} {'Jackknife':<15} "
        f"{ci_jackknife.loc[param, 'lower']:>12.6f} "
        f"{ci_jackknife.loc[param, 'upper']:>12.6f} "
        f"{width_jk:>12.6f}"
    )
    print()

# Influential entities
print("\n7. Influential Entities Analysis:")
print("=" * 80)
influential = jackknife.influential_entities(threshold=2.0, metric="max")

if len(influential) > 0:
    print(f"Found {len(influential)} influential entities (threshold = 2.0 × mean influence)")
    print()
    print(influential.to_string(index=False))
    print()
    print("Note: These entities have disproportionate impact on parameter estimates.")
    print("      Consider robustness checks or investigating these entities.")
else:
    print("No highly influential entities detected.")
    print("✓ Results appear robust to individual entity exclusion")
print()

# Statistical assessment
print("\n8. Overall Assessment:")
print("=" * 80)

# Check bias magnitude
max_rel_bias = (jk_results.jackknife_bias.abs() / jk_results.original_estimates.abs()).max()

print(f"Maximum relative bias: {max_rel_bias:.4f}")
if max_rel_bias < 0.05:
    print("   ✓ Bias is negligible (<5% of estimate)")
elif max_rel_bias < 0.10:
    print("   ⚠ Moderate bias (5-10% of estimate)")
    print("   → Consider using bias-corrected estimates")
else:
    print("   ✗ Substantial bias (>10% of estimate)")
    print("   → Bias correction recommended")
print()

# Check influential entities
if len(influential) == 0:
    print("Influential entities: None detected")
    print("   ✓ Results are robust")
elif len(influential) <= 2:
    print(f"Influential entities: {len(influential)} detected")
    print("   ⚠ Minor concern, check these entities")
else:
    print(f"Influential entities: {len(influential)} detected")
    print("   ✗ Multiple influential entities")
    print("   → Consider robustness checks")
print()

print("=" * 80)
print("Jackknife Analysis Complete!")
print("=" * 80)
print()
print("Key Takeaways:")
print("- Jackknife provides alternative bias and variance estimates")
print("- Less computationally intensive than bootstrap")
print("- Good for identifying influential observations")
print("- Use bias-corrected estimates if bias is substantial")
print()
