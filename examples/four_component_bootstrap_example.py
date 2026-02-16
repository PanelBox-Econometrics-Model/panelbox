"""Example: Four-Component SFA with Bootstrap Inference

This example demonstrates:
1. Estimating the four-component SFA model
2. Computing bootstrap confidence intervals
3. Comparing with alternative models (Pitt-Lee, True FE/RE)

Author: PanelBox Development Team
"""

import numpy as np
import pandas as pd

from panelbox.frontier.advanced import FourComponentSFA, compare_all_models

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# 1. Generate Synthetic Panel Data
# ============================================================================

N = 50  # Number of entities
T = 10  # Number of time periods
n = N * T  # Total observations

# True parameter values
beta_true = [5.0, 0.3, 0.7]  # Intercept, log(labor), log(capital)
sigma_v = 0.15  # Noise
sigma_u = 0.10  # Transient inefficiency
sigma_mu = 0.20  # Heterogeneity
sigma_eta = 0.12  # Persistent inefficiency

print("=" * 70)
print("FOUR-COMPONENT SFA WITH BOOTSTRAP")
print("=" * 70)
print("\nGenerating synthetic panel data...")
print(f"  Entities (N): {N}")
print(f"  Time periods (T): {T}")
print(f"  Total observations: {n}")
print(f"\nTrue parameters:")
print(f"  β = {beta_true}")
print(f"  σ_v = {sigma_v:.3f} (noise)")
print(f"  σ_u = {sigma_u:.3f} (transient inefficiency)")
print(f"  σ_μ = {sigma_mu:.3f} (heterogeneity)")
print(f"  σ_η = {sigma_eta:.3f} (persistent inefficiency)")

# Generate panel structure
entity_id = np.repeat(np.arange(N), T)
time_id = np.tile(np.arange(T), N)

# Generate inputs
log_labor = np.random.uniform(0, 2, size=n)
log_capital = np.random.uniform(0, 2, size=n)

# Generate the 4 components
mu_i = np.random.normal(0, sigma_mu, size=N)  # Heterogeneity
eta_i = np.abs(np.random.normal(0, sigma_eta, size=N))  # Persistent inefficiency
v_it = np.random.normal(0, sigma_v, size=n)  # Noise
u_it = np.abs(np.random.normal(0, sigma_u, size=n))  # Transient inefficiency

# Generate output
log_output = (
    beta_true[0]
    + beta_true[1] * log_labor
    + beta_true[2] * log_capital
    + mu_i[entity_id]
    - eta_i[entity_id]
    + v_it
    - u_it
)

# Create DataFrame
data = pd.DataFrame(
    {
        "entity": entity_id,
        "time": time_id,
        "log_output": log_output,
        "log_labor": log_labor,
        "log_capital": log_capital,
    }
)

# ============================================================================
# 2. Estimate Four-Component Model
# ============================================================================

print("\n" + "=" * 70)
print("STEP 1: ESTIMATING FOUR-COMPONENT MODEL")
print("=" * 70)

model = FourComponentSFA(
    data=data,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    entity="entity",
    time="time",
    frontier_type="production",
)

result = model.fit(verbose=True)

# ============================================================================
# 3. Bootstrap Inference
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

boot_result = result.bootstrap(
    n_bootstrap=50,  # Use 50 for demonstration (use 500+ in practice)
    confidence_level=0.95,
    random_state=42,
    verbose=True,
)

boot_result.print_summary()

# Show confidence intervals for first 5 entities
print("\nPersistent Efficiency with 95% CI (first 5 entities):")
print(boot_result.persistent_efficiency_ci().head())

# ============================================================================
# 4. Model Comparison
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: MODEL COMPARISON")
print("=" * 70)

comparison = compare_all_models(result)
comparison.print_summary()

# ============================================================================
# 5. Interpretation
# ============================================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

# Get efficiency estimates
eff_df = result.overall_efficiency()

# Aggregate by entity
entity_stats = eff_df.groupby("entity").agg(
    {
        "overall_efficiency": "mean",
        "persistent_efficiency": "first",  # Same across time
        "transient_efficiency": "mean",
    }
)

# Find best and worst performers
best_entity = entity_stats["overall_efficiency"].idxmax()
worst_entity = entity_stats["overall_efficiency"].idxmin()

print(f"\nBest performing entity: {best_entity}")
print(f"  Overall efficiency: {entity_stats.loc[best_entity, 'overall_efficiency']:.4f}")
print(f"  Persistent efficiency: {entity_stats.loc[best_entity, 'persistent_efficiency']:.4f}")
print(f"  Transient efficiency: {entity_stats.loc[best_entity, 'transient_efficiency']:.4f}")

print(f"\nWorst performing entity: {worst_entity}")
print(f"  Overall efficiency: {entity_stats.loc[worst_entity, 'overall_efficiency']:.4f}")
print(f"  Persistent efficiency: {entity_stats.loc[worst_entity, 'persistent_efficiency']:.4f}")
print(f"  Transient efficiency: {entity_stats.loc[worst_entity, 'transient_efficiency']:.4f}")

print("\nPolicy Implications:")
print("  - Low persistent efficiency → Structural reforms needed")
print("    (e.g., infrastructure, technology, location)")
print("  - Low transient efficiency → Managerial improvements needed")
print("    (e.g., training, organization, processes)")

# Compare with true values
print("\n" + "=" * 70)
print("PARAMETER RECOVERY (Estimated vs True)")
print("=" * 70)

print("\nVariance Components:")
print(f"  σ_v:  {result.sigma_v:.4f} vs {sigma_v:.4f} (true)")
print(f"  σ_u:  {result.sigma_u:.4f} vs {sigma_u:.4f} (true)")
print(f"  σ_μ:  {result.sigma_mu:.4f} vs {sigma_mu:.4f} (true)")
print(f"  σ_η:  {result.sigma_eta:.4f} vs {sigma_eta:.4f} (true)")

print("\nCoefficients:")
for i, name in enumerate(["const", "log_labor", "log_capital"]):
    print(f"  β_{name}: {result.beta[i]:.4f} vs {beta_true[i]:.4f} (true)")

print("\n" + "=" * 70)
print("EXAMPLE COMPLETE!")
print("=" * 70)
