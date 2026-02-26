"""
Simple bootstrap performance test
"""

import time

import numpy as np
import pandas as pd

from panelbox.frontier import StochasticFrontier
from panelbox.frontier.bootstrap import SFABootstrap

# Generate simple data
np.random.seed(42)
n = 100
x = np.random.normal(5, 1, n)
v = np.random.normal(0, 0.1, n)
u = np.abs(np.random.normal(0, 0.2, n))
y = 2 + 0.5 * x + v - u

df = pd.DataFrame({"y": y, "x": x})

print("Estimating SFA...")
sf = StochasticFrontier(data=df, depvar="y", exog=["x"], frontier="production", dist="half_normal")
result = sf.fit(method="mle")

print(f"Log-lik: {result.loglik:.4f}")
print(f"sigma_v: {result.sigma_v:.4f}")
print(f"sigma_u: {result.sigma_u:.4f}")

print("\nBootstrap test (100 reps)...")
start = time.time()
bootstrap = SFABootstrap(result, n_boot=100, method="parametric", n_jobs=2, seed=42)
boot_res = bootstrap.bootstrap_parameters()
elapsed = time.time() - start

print(f"Elapsed: {elapsed:.2f}s")
print(f"Rate: {100 / elapsed:.1f} boot/s")

print("\nBootstrap results:")
print(boot_res)

print("\n✓ Test complete")
