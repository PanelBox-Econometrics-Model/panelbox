"""Test log-likelihood function directly with R parameters."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from panelbox.frontier.likelihoods import loglik_half_normal
from panelbox.frontier.starting_values import ols_starting_values

HERE = Path(__file__).parent

# Load data
data = pd.read_csv(HERE / "sfa_test_data.csv")
ref = pd.read_csv(HERE / "sfa_reference_hn.csv")
ref_dict = ref.set_index("parameter")["value"].to_dict()

# Prepare data
y = data["log_output"].values
X = np.column_stack(
    [
        np.ones(len(data)),  # constant
        data["log_labor"].values,
        data["log_capital"].values,
    ]
)

# R parameters
beta_r = np.array([ref_dict["beta_0"], ref_dict["beta_1"], ref_dict["beta_2"]])
sigma_v_r = ref_dict["sigma_v"]
sigma_u_r = ref_dict["sigma_u"]

# Convert to parameter vector (with log transformation)
theta_r = np.concatenate([beta_r, [np.log(sigma_v_r**2)], [np.log(sigma_u_r**2)]])

print("=" * 80)
print("TESTING LOG-LIKELIHOOD FUNCTION")
print("=" * 80)
print("R parameters:")
print(f"  β = {beta_r}")
print(f"  σ_v = {sigma_v_r:.6f}")
print(f"  σ_u = {sigma_u_r:.6f}")
print(f"  ln(σ²_v) = {np.log(sigma_v_r**2):.6f}")
print(f"  ln(σ²_u) = {np.log(sigma_u_r**2):.6f}")

# Evaluate log-likelihood at R parameters
ll_r = loglik_half_normal(theta_r, y, X, sign=1)

print("\nLog-likelihood at R parameters:")
print(f"  PanelBox: {ll_r:.6f}")
print(f"  R:        {ref_dict['loglik']:.6f}")
print(f"  Difference: {ll_r - ref_dict['loglik']:.6f}")

# Try with OLS starting values
beta_ols, sigma_v_sq_ols, sigma_u_sq_ols = ols_starting_values(y, X, "half_normal")

theta_ols = np.concatenate([beta_ols, [np.log(sigma_v_sq_ols)], [np.log(sigma_u_sq_ols)]])

ll_ols = loglik_half_normal(theta_ols, y, X, sign=1)

print("\nLog-likelihood at OLS starting values:")
print(f"  PanelBox: {ll_ols:.6f}")
print(f"  β_OLS = {beta_ols}")
print(f"  σ_v_OLS = {np.sqrt(sigma_v_sq_ols):.6f}")
print(f"  σ_u_OLS = {np.sqrt(sigma_u_sq_ols):.6f}")

# Try evaluating at a grid
print("\n" + "=" * 80)
print("GRID SEARCH AROUND R PARAMETERS")
print("=" * 80)

best_ll = -np.inf
best_params = None

for scale_v in [0.5, 0.75, 1.0, 1.25, 1.5]:
    for scale_u in [0.5, 0.75, 1.0, 1.25, 1.5]:
        sigma_v_test = sigma_v_r * scale_v
        sigma_u_test = sigma_u_r * scale_u

        theta_test = np.concatenate([beta_r, [np.log(sigma_v_test**2)], [np.log(sigma_u_test**2)]])

        ll_test = loglik_half_normal(theta_test, y, X, sign=1)

        print(f"σ_v={sigma_v_test:.4f}, σ_u={sigma_u_test:.4f}: ll={ll_test:.4f}")

        if ll_test > best_ll:
            best_ll = ll_test
            best_params = (sigma_v_test, sigma_u_test)

print(f"\nBest log-likelihood from grid: {best_ll:.6f}")
print(f"Best parameters: σ_v={best_params[0]:.6f}, σ_u={best_params[1]:.6f}")

# Test optimization directly
print("\n" + "=" * 80)
print("DIRECT OPTIMIZATION TEST")
print("=" * 80)


def neg_ll(theta):
    ll = loglik_half_normal(theta, y, X, sign=1)
    return -ll


# Start from R parameters
result = minimize(
    neg_ll,
    theta_r,
    method="L-BFGS-B",
    bounds=[(None, None), (None, None), (None, None), (-13.8, 13.8), (-13.8, 13.8)],
)

print("Optimization result:")
print(f"  Converged: {result.success}")
print(f"  Message: {result.message}")
print(f"  Log-likelihood: {-result.fun:.6f}")
print(f"  β = {result.x[:3]}")
print(f"  σ_v = {np.exp(0.5 * result.x[3]):.6f}")
print(f"  σ_u = {np.exp(0.5 * result.x[4]):.6f}")
