"""
PanelBox - Validation Tests Example

This example demonstrates the validation testing system:
1. Estimating a Fixed Effects model
2. Running validation tests
3. Interpreting diagnostic results
"""

import numpy as np
import pandas as pd
import panelbox as pb

# Set random seed
np.random.seed(42)

print("=" * 80)
print("PANELBOX - VALIDATION TESTS EXAMPLE")
print("=" * 80)
print()

# ============================================================================
# 1. CREATE SAMPLE PANEL DATA WITH ISSUES
# ============================================================================
print("1. Creating panel data with diagnostic issues...")
print("-" * 80)

n_firms = 50
n_years = 10
n_obs = n_firms * n_years

# Create panel structure
data = pd.DataFrame({
    'firm': np.repeat(range(1, n_firms + 1), n_years),
    'year': np.tile(range(2010, 2010 + n_years), n_firms),
})

# Add firm-specific fixed effects
firm_effect = {i: np.random.normal(0, 5) for i in range(1, n_firms + 1)}
data['firm_effect'] = data['firm'].map(firm_effect)

# Generate regressors
data['capital'] = np.random.uniform(100, 1000, n_obs)
data['labor'] = np.random.uniform(50, 500, n_obs)

# Generate errors with serial correlation (AR(1) structure)
# This will be detected by Wooldridge test
rho = 0.5  # AR(1) coefficient
errors = np.zeros(n_obs)
for i in range(n_firms):
    start_idx = i * n_years
    end_idx = (i + 1) * n_years
    firm_errors = np.zeros(n_years)
    firm_errors[0] = np.random.normal(0, 10)
    for t in range(1, n_years):
        firm_errors[t] = rho * firm_errors[t-1] + np.random.normal(0, 10)
    errors[start_idx:end_idx] = firm_errors

# Generate dependent variable
data['output'] = (
    10 +
    data['firm_effect'] +
    0.5 * data['capital'] +
    0.3 * data['labor'] +
    errors
)

# Drop the true firm_effect
data = data.drop('firm_effect', axis=1)

print(f"Panel: {n_firms} firms, {n_years} years")
print(f"Introduced AR(1) serial correlation (rho={rho})")
print()

# ============================================================================
# 2. ESTIMATE FIXED EFFECTS MODEL
# ============================================================================
print("2. Estimating Fixed Effects model...")
print("-" * 80)

fe = pb.FixedEffects("output ~ capital + labor", data, "firm", "year")
fe_results = fe.fit(cov_type='clustered')

print(fe_results.summary())
print()

# ============================================================================
# 3. RUN VALIDATION TESTS
# ============================================================================
print("3. Running validation tests...")
print("-" * 80)
print()

# Run all available tests
validation = fe_results.validate(tests='default', alpha=0.05, verbose=True)

print()
print("=" * 80)
print("VALIDATION REPORT")
print("=" * 80)
print()
print(validation.summary(verbose=True))

# ============================================================================
# 4. INTERPRET RESULTS
# ============================================================================
print()
print("4. Interpreting Validation Results...")
print("-" * 80)

# Check for specific issues
failed_tests = validation.get_failed_tests()

if failed_tests:
    print(f"\n⚠️  {len(failed_tests)} test(s) detected potential issues:")
    for test_name in failed_tests:
        print(f"  - {test_name}")
    print()
    print("Recommendations:")
    
    if any('serial' in t for t in failed_tests):
        print("  • Serial correlation detected:")
        print("    - Use clustered standard errors (already using)")
        print("    - Or use HAC standard errors")
        print("    - Consider dynamic panel models (Arellano-Bond)")
    
    if any('het' in t for t in failed_tests):
        print("  • Heteroskedasticity detected:")
        print("    - Use robust standard errors")
        print("    - Or use WLS (weighted least squares)")
    
    if any('cd' in t for t in failed_tests):
        print("  • Cross-sectional dependence detected:")
        print("    - Use Driscoll-Kraay standard errors")
        print("    - Or use spatial/factor models")
else:
    print("✓ No major diagnostic issues detected")

print()

# ============================================================================
# 5. ACCESS INDIVIDUAL TEST RESULTS
# ============================================================================
print()
print("5. Individual Test Details...")
print("-" * 80)
print()

# Wooldridge test
if 'Wooldridge' in validation.serial_tests:
    wooldridge = validation.serial_tests['Wooldridge']
    print(f"Wooldridge AR Test:")
    print(f"  Statistic: {wooldridge.statistic:.4f}")
    print(f"  P-value:   {wooldridge.pvalue:.4f}")
    print(f"  Result:    {'REJECT H0 - Serial correlation detected' if wooldridge.reject_null else 'PASS - No serial correlation'}")
    print()

# Modified Wald test
if 'Modified Wald' in validation.het_tests:
    wald = validation.het_tests['Modified Wald']
    print(f"Modified Wald Test:")
    print(f"  Statistic: {wald.statistic:.4f}")
    print(f"  P-value:   {wald.pvalue:.4f}")
    print(f"  Result:    {'REJECT H0 - Heteroskedasticity detected' if wald.reject_null else 'PASS - Homoskedastic'}")
    print()

# Pesaran CD test
if 'Pesaran CD' in validation.cd_tests:
    pesaran = validation.cd_tests['Pesaran CD']
    print(f"Pesaran CD Test:")
    print(f"  Statistic:  {pesaran.statistic:.4f}")
    print(f"  P-value:    {pesaran.pvalue:.4f}")
    print(f"  Avg |corr|: {pesaran.metadata['avg_abs_correlation']:.4f}")
    print(f"  Result:     {'REJECT H0 - Cross-sectional dependence' if pesaran.reject_null else 'PASS - No cross-sectional dependence'}")
    print()

print("=" * 80)
print("EXAMPLE COMPLETE")
print("=" * 80)
print()
print("Key Takeaways:")
print("1. Validation tests help identify model assumptions violations")
print("2. Wooldridge test detected the AR(1) serial correlation we introduced")
print("3. Use appropriate standard errors based on diagnostic results")
print("4. ValidationSuite provides unified interface to all tests")
print()
print("=" * 80)
