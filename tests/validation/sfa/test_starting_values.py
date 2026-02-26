"""Test starting values calculation."""

import numpy as np
import pandas as pd

from panelbox.frontier.starting_values import ols_starting_values

# Load data
data = pd.read_csv("/home/guhaase/projetos/panelbox/tests/validation/sfa/sfa_test_data.csv")
ref = pd.read_csv("/home/guhaase/projetos/panelbox/tests/validation/sfa/sfa_reference_hn.csv")
ref_dict = ref.set_index("parameter")["value"].to_dict()

# Prepare data
y = data["log_output"].values
X = np.column_stack([np.ones(len(data)), data["log_labor"].values, data["log_capital"].values])

print("=" * 80)
print("OLS ESTIMATION")
print("=" * 80)

# OLS
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
residuals = y - X @ beta_ols

print(f"β_OLS = {beta_ols}")
print(f"β_TRUE = {[ref_dict['beta_0'], ref_dict['beta_1'], ref_dict['beta_2']]}")

print("\n" + "=" * 80)
print("RESIDUAL MOMENTS")
print("=" * 80)

m1 = np.mean(residuals)
m2 = np.mean(residuals**2)
m3 = np.mean(residuals**3)

print(f"E[ε]   = {m1:.6f}")
print(f"E[ε²]  = {m2:.6f}")
print(f"E[ε³]  = {m3:.6f}")

# For production frontier: ε = v - u
# E[ε] = -E[u] = -σ_u * sqrt(2/π)
# E[ε²] = σ²_v + (1 - 2/π)σ²_u
# E[ε³] = -σ³_u * sqrt(2/π) * (1 - 4/π)

sqrt_2_pi = np.sqrt(2 / np.pi)
factor3 = sqrt_2_pi * (1 - 4 / np.pi)

print("\n" + "=" * 80)
print("THEORETICAL MOMENTS")
print("=" * 80)

sigma_v_true = ref_dict["sigma_v"]
sigma_u_true = ref_dict["sigma_u"]

m1_theory = -sigma_u_true * sqrt_2_pi
m2_theory = sigma_v_true**2 + (1 - 2 / np.pi) * sigma_u_true**2
m3_theory = -(sigma_u_true**3) * factor3

print(f"E[ε]_theory   = {m1_theory:.6f}")
print(f"E[ε²]_theory  = {m2_theory:.6f}")
print(f"E[ε³]_theory  = {m3_theory:.6f}")

print("\n" + "=" * 80)
print("MOMENT-BASED ESTIMATION")
print("=" * 80)

# From third moment
if abs(m3) < 1e-10:
    sigma_u_mom = 0.0
else:
    # E[ε³] = -σ³_u * factor3
    # σ_u = (-E[ε³] / factor3)^(1/3)
    sigma_u_mom = abs(m3 / (-factor3)) ** (1 / 3)

# From second moment
sigma_v_sq_mom = m2 - (1 - 2 / np.pi) * sigma_u_mom**2

if sigma_v_sq_mom < 0:
    print("WARNING: Negative variance!")
    sigma_v_sq_mom = m2 / 2
    sigma_u_sq_mom = m2 / 2
else:
    sigma_v_sq_mom = sigma_v_sq_mom
    sigma_u_sq_mom = sigma_u_mom**2

sigma_v_mom = np.sqrt(sigma_v_sq_mom)

print(f"σ_u from moments: {sigma_u_mom:.6f}")
print(f"σ_v from moments: {sigma_v_mom:.6f}")
print(f"σ_u TRUE:         {sigma_u_true:.6f}")
print(f"σ_v TRUE:         {sigma_v_true:.6f}")

print("\n" + "=" * 80)
print("USING PANELBOX FUNCTION")
print("=" * 80)

beta_pb, sigma_v_sq_pb, sigma_u_sq_pb = ols_starting_values(y, X, "half_normal")

print(f"β_PB  = {beta_pb}")
print(f"σ_v_PB = {np.sqrt(sigma_v_sq_pb):.6f}")
print(f"σ_u_PB = {np.sqrt(sigma_u_sq_pb):.6f}")

# Check the issue
print("\n" + "=" * 80)
print("INVESTIGATING THE ISSUE")
print("=" * 80)

print(f"\nResidual skewness: {m3 / (m2 ** (3 / 2)):.6f}")
print(f"Sign of m3: {np.sign(m3)}")

# For production frontier, we expect m3 < 0 (negative skew)
# because ε = v - u and u is always positive
print("\nExpected sign for production frontier: negative")
print(f"Actual sign: {'negative' if m3 < 0 else 'positive'}")
