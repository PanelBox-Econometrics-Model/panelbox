---
title: "Panel VECM"
description: "Guide to Panel Vector Error Correction Models in PanelBox: cointegration rank testing, VECM estimation, and long-run equilibrium analysis."
---

# Panel VECM

!!! info "Quick Reference"
    **Class:** `panelbox.var.vecm.PanelVECM`
    **Rank Test:** `panelbox.var.vecm.CointegrationRankTest`
    **Import:** `from panelbox.var.vecm import PanelVECM, CointegrationRankTest`
    **Stata equivalent:** `vec`, `xtcoint`
    **R equivalent:** `vars::ca.jo()`, `urca::ca.jo()`

## Overview

When panel variables are non-stationary but share long-run equilibrium relationships (cointegration), the standard Panel VAR in levels is misspecified and the Panel VAR in differences loses the long-run information. The **Panel Vector Error Correction Model** (Panel VECM) resolves this by decomposing the dynamics into short-run adjustments and long-run equilibrium corrections.

The Panel VECM representation is:

$$
\Delta Y_{it} = \alpha_i + \Pi Y_{i,t-1} + \sum_{j=1}^{p-1} \Gamma_j \Delta Y_{i,t-j} + \varepsilon_{it}
$$

where the **long-run impact matrix** $\Pi = \alpha \beta'$ has reduced rank $r < K$:

- $\beta$ ($K \times r$): **Cointegrating vectors** defining the long-run equilibrium relationships
- $\alpha$ ($K \times r$): **Loading matrix** (adjustment speeds) governing how quickly each variable corrects back to equilibrium
- $\Gamma_j$ ($K \times K$): **Short-run dynamics** matrices capturing transitory effects
- $r$: **Cointegration rank** (number of independent equilibrium relationships)

PanelBox implements the panel extension of the Johansen (1991) procedure following the framework of Larsson, Lyhagen, and Loethgren (2001) for panel cointegration rank testing.

## Quick Example

```python
from panelbox.var import PanelVARData
from panelbox.var.vecm import CointegrationRankTest, PanelVECM

# Prepare data
var_data = PanelVARData(
    data=df,
    endog_vars=["gdp", "consumption", "investment"],
    entity_col="country",
    time_col="year",
    lags=2,
)

# Step 1: Test cointegration rank
rank_test = CointegrationRankTest(var_data, deterministic="c")
rank_result = rank_test.test_rank()
print(rank_result.summary())
print(f"Selected rank: {rank_result.selected_rank}")

# Step 2: Estimate VECM
vecm = PanelVECM(data=var_data, rank=rank_result.selected_rank)
vecm_result = vecm.fit(method="ml")
print(vecm_result.summary())
```

## When to Use

- **Non-stationary variables**: Unit root tests indicate I(1) variables (e.g., log GDP, log consumption, log investment)
- **Long-run equilibrium exists**: Economic theory suggests equilibrium relationships (e.g., consumption-income, money demand)
- **Short-run deviations observed**: Variables temporarily deviate from equilibrium but revert over time
- **VAR in levels is unstable**: Eigenvalues near or above 1 suggest non-stationarity

!!! warning "Key Assumptions"
    - Variables must be integrated of the same order (typically I(1))
    - At least one cointegrating relationship must exist ($0 < r < K$)
    - Continuous time series within each entity (no internal gaps)
    - Sufficient time periods per entity: $T > K \times p + 1$
    - Cross-sectional independence (for standard inference)

## Detailed Guide

### Step 1: Cointegration Rank Testing

Before estimating a VECM, you must determine the cointegration rank $r$ -- the number of independent long-run equilibrium relationships.

```python
from panelbox.var.vecm import CointegrationRankTest

rank_test = CointegrationRankTest(
    data=var_data,
    max_rank=None,            # Default: K-1
    deterministic="c",        # Deterministic specification
)

rank_result = rank_test.test_rank()
print(rank_result.summary())
```

**Deterministic specification options:**

| `deterministic` | Description | Use When |
|-----------------|-------------|----------|
| `"nc"` | No constant | Variables have zero mean in equilibrium (rare) |
| `"c"` | Constant in cointegrating equation | **Default**. Most common case |
| `"ct"` | Constant and linear trend | Variables exhibit deterministic trends |

**Two test types are computed:**

1. **Trace test**: $H_0: \text{rank} \le r$ vs $H_1: \text{rank} > r$
   - Tests whether rank is at most $r$
   - Start from $r=0$ and increase until you fail to reject

2. **Max-eigenvalue test**: $H_0: \text{rank} = r$ vs $H_1: \text{rank} = r + 1$
   - Tests whether rank is exactly $r$ vs $r+1$
   - More specific but less robust than trace

The `RankSelectionResult` provides:

| Attribute | Description |
|-----------|-------------|
| `selected_rank_trace` | Rank selected by trace test |
| `selected_rank_maxeig` | Rank selected by max-eigenvalue test |
| `selected_rank` | Consensus rank (trace test used by default) |
| `trace_tests` | List of `RankTestResult` for each rank |
| `maxeig_tests` | List of `RankTestResult` for each rank |

!!! tip "Interpreting Rank Tests"
    Start with $r = 0$ (no cointegration). If rejected, test $r = 1$, then $r = 2$, etc. The selected rank is the first $r$ where you **fail to reject** $H_0$. If trace and max-eigenvalue disagree, prefer the trace test (more robust in practice).

### Step 2: VECM Estimation

Once the cointegration rank is determined, estimate the VECM:

```python
from panelbox.var.vecm import PanelVECM

vecm = PanelVECM(
    data=var_data,
    rank=1,                   # Cointegration rank (from Step 1)
    deterministic="c",        # Must match rank test specification
)

result = vecm.fit(method="ml")   # Maximum Likelihood (Johansen procedure)
```

**Estimation Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `rank` | `int` or `None` | `None` | Cointegration rank. If `None`, automatically selected via rank test |
| `deterministic` | `str` | `"c"` | `"nc"`, `"c"`, or `"ct"` |
| `method` | `str` | `"ml"` | `"ml"` (Maximum Likelihood) or `"twostep"` |

### Step 3: Interpreting Results

The `PanelVECMResult` provides rich access to all estimated components:

#### Cointegrating Relations ($\beta$)

```python
# Cointegrating vectors (normalized)
beta_df = result.cointegrating_relations()
print(beta_df)
```

Each column of $\beta$ defines a long-run equilibrium. For example, with $K=3$ variables and $r=1$:

$$
\beta' Y_{it} = \beta_1 y_{1,it} + \beta_2 y_{2,it} + \beta_3 y_{3,it} = 0
$$

This equilibrium relationship means that in the long run, deviations from $\beta' Y = 0$ are corrected.

#### Adjustment Speeds ($\alpha$)

```python
# Loading matrix (adjustment speeds)
alpha_df = result.adjustment_speeds()
print(alpha_df)
```

The loading coefficients $\alpha_{k,j}$ tell you how fast variable $k$ adjusts to deviations from equilibrium $j$:

- **Negative $\alpha$**: Variable moves back toward equilibrium (error-correcting behavior)
- **Near zero $\alpha$**: Variable is weakly exogenous with respect to that equilibrium
- **Positive $\alpha$**: Variable moves away from equilibrium (destabilizing -- unusual)

#### Short-Run Dynamics ($\Gamma$)

```python
# Short-run dynamics matrices
gamma_dfs = result.short_run_dynamics()
for lag, gamma_df in enumerate(gamma_dfs, 1):
    print(f"\nGamma_{lag}:")
    print(gamma_df)
```

#### Long-Run Impact Matrix ($\Pi$)

```python
# Pi = alpha @ beta'
print(f"Pi matrix:\n{result.Pi}")
print(f"Rank of Pi: {result.rank}")
```

### Additional Analysis

#### Exogeneity Tests

```python
# Weak exogeneity: alpha[variable, :] = 0
for var in result.var_names:
    weak = result.test_weak_exogeneity(var)
    print(f"{var}: stat={weak['statistic']:.3f}, p={weak['p_value']:.4f}, "
          f"reject={weak['reject']}")

# Strong exogeneity: alpha = 0 AND Gamma(other vars) = 0
for var in result.var_names:
    strong = result.test_strong_exogeneity(var)
    print(f"{var}: stat={strong['statistic']:.3f}, p={strong['p_value']:.4f}")
```

#### Convert to VAR Representation

```python
# Convert VECM to VAR in levels
A_matrices = result.to_var()
for i, A in enumerate(A_matrices, 1):
    print(f"A_{i}:\n{A}")
```

#### IRF and FEVD from VECM

```python
# Impulse Response Functions
irf = result.irf(periods=20, method="cholesky")
irf.plot()

# Forecast Error Variance Decomposition
fevd = result.fevd(periods=20, method="cholesky")
fevd.plot()
```

!!! note "VECM IRFs vs VAR IRFs"
    Unlike stationary VAR models where IRFs converge to zero, VECM IRFs show **permanent effects** due to the presence of unit roots. Cumulative IRFs converge to finite long-run impact levels determined by the cointegrating relationships.

## Configuration Options

### PanelVECMResult Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `alpha` | `np.ndarray` | Loading matrix ($K \times r$) |
| `beta` | `np.ndarray` | Cointegrating vectors ($K \times r$) |
| `Gamma` | `list[np.ndarray]` | Short-run dynamics matrices |
| `Pi` | `np.ndarray` | Long-run impact matrix $\Pi = \alpha\beta'$ |
| `Sigma` | `np.ndarray` | Residual covariance ($K \times K$) |
| `rank` | `int` | Cointegration rank |
| `K` | `int` | Number of variables |
| `p` | `int` | Number of lags (VAR representation) |
| `N` | `int` | Number of entities |

## Complete Workflow Example

```python
import pandas as pd
from panelbox.var import PanelVARData
from panelbox.var.vecm import CointegrationRankTest, PanelVECM

# Load data (non-stationary macroeconomic variables in levels)
df = pd.read_csv("macro_levels.csv")

# Step 1: Create data container
var_data = PanelVARData(
    data=df,
    endog_vars=["log_gdp", "log_consumption", "log_investment"],
    entity_col="country",
    time_col="year",
    lags=2,
)

# Step 2: Test cointegration rank
rank_test = CointegrationRankTest(var_data, deterministic="c")
rank_result = rank_test.test_rank()
print(rank_result.summary())

selected_r = rank_result.selected_rank
print(f"\nSelected cointegration rank: {selected_r}")

if selected_r == 0:
    print("No cointegration detected. Consider VAR in differences.")
else:
    # Step 3: Estimate VECM
    vecm = PanelVECM(data=var_data, rank=selected_r, deterministic="c")
    result = vecm.fit(method="ml")
    print(result.summary())

    # Step 4: Interpret long-run equilibrium
    print("\nCointegrating Relations (beta):")
    print(result.cointegrating_relations())

    print("\nAdjustment Speeds (alpha):")
    print(result.adjustment_speeds())

    # Step 5: Exogeneity tests
    for var in result.var_names:
        weak = result.test_weak_exogeneity(var)
        status = "weakly exogenous" if not weak["reject"] else "endogenous"
        print(f"{var}: {status} (p={weak['p_value']:.3f})")

    # Step 6: IRF analysis
    irf = result.irf(periods=20, method="cholesky", cumulative=True)
    irf.plot()
```

## Tutorials

| Tutorial | Description | Link |
|----------|-------------|------|
| Panel VECM Notebook | Cointegration testing and VECM estimation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/var/notebooks/05_vecm_cointegration.ipynb) |

## See Also

- [Panel VAR Estimation](estimation.md) -- For stationary variables
- [Impulse Response Functions](irf.md) -- Dynamic effects of shocks (permanent in VECM)
- [FEVD](fevd.md) -- Variance decomposition from VECM
- [Granger Causality](granger.md) -- Causality testing in VECM context
- [Forecasting](forecasting.md) -- Multi-step predictions

## References

- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551-1580.
- Larsson, R., Lyhagen, J., & Loethgren, M. (2001). Likelihood-based cointegration tests in heterogeneous panels. *The Econometrics Journal*, 4(1), 109-142.
- Breitung, J., & Pesaran, M. H. (2008). Unit roots and cointegration in panels. In *The Econometrics of Panel Data* (pp. 279-322). Springer.
- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer-Verlag, Chapter 9.
