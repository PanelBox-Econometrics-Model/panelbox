"""Test optimization with different settings."""

import pandas as pd

from panelbox.frontier import StochasticFrontier

# Load data
data = pd.read_csv("/home/guhaase/projetos/panelbox/tests/validation/sfa/sfa_test_data.csv")
ref = pd.read_csv("/home/guhaase/projetos/panelbox/tests/validation/sfa/sfa_reference_hn.csv")
ref_dict = ref.set_index("parameter")["value"].to_dict()

# Create model
sf = StochasticFrontier(
    data=data,
    depvar="log_output",
    exog=["log_labor", "log_capital"],
    frontier="production",
    dist="half_normal",
)

print("=" * 80)
print("TEST 1: Default settings")
print("=" * 80)
result1 = sf.fit(method="mle", verbose=True)
print(f"Converged: {result1.converged}")
print(f"Log-likelihood: {result1.loglik:.6f}")
print(f"Difference from R: {result1.loglik - ref_dict['loglik']:.6f}")

print("\n" + "=" * 80)
print("TEST 2: Relaxed tolerances")
print("=" * 80)
result2 = sf.fit(method="mle", tol=1e-6, verbose=True)
print(f"Converged: {result2.converged}")
print(f"Log-likelihood: {result2.loglik:.6f}")
print(f"Difference from R: {result2.loglik - ref_dict['loglik']:.6f}")

print("\n" + "=" * 80)
print("TEST 3: More iterations")
print("=" * 80)
result3 = sf.fit(method="mle", maxiter=10000, tol=1e-6, verbose=True)
print(f"Converged: {result3.converged}")
print(f"Log-likelihood: {result3.loglik:.6f}")
print(f"Difference from R: {result3.loglik - ref_dict['loglik']:.6f}")

print("\n" + "=" * 80)
print("TEST 4: BFGS optimizer")
print("=" * 80)
result4 = sf.fit(method="mle", optimizer="BFGS", maxiter=10000, tol=1e-6, verbose=True)
print(f"Converged: {result4.converged}")
print(f"Log-likelihood: {result4.loglik:.6f}")
print(f"Difference from R: {result4.loglik - ref_dict['loglik']:.6f}")

print("\n" + "=" * 80)
print("BEST RESULT COMPARISON")
print("=" * 80)
best_result = max([result1, result2, result3, result4], key=lambda x: x.loglik)

print(f"β₀:              {best_result.params['const']:.6f} (R: {ref_dict['beta_0']:.6f})")
print(f"β₁:              {best_result.params['log_labor']:.6f} (R: {ref_dict['beta_1']:.6f})")
print(f"β₂:              {best_result.params['log_capital']:.6f} (R: {ref_dict['beta_2']:.6f})")
print(f"σ_v:             {best_result.sigma_v:.6f} (R: {ref_dict['sigma_v']:.6f})")
print(f"σ_u:             {best_result.sigma_u:.6f} (R: {ref_dict['sigma_u']:.6f})")
print(f"Log-likelihood:  {best_result.loglik:.6f} (R: {ref_dict['loglik']:.6f})")
print(f"Mean efficiency: {best_result.mean_efficiency:.6f} (R: {ref_dict['mean_eff']:.6f})")
