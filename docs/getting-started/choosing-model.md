---
title: Choosing a Model
description: Decision guide for selecting the right panel data estimator from PanelBox's 13 model families
---

# Choosing a Model

PanelBox provides 13 model families covering virtually every panel data scenario. This guide helps you select the right one.

## Quick Decision Tree

```text
START: What type is your dependent variable?
│
├─ CONTINUOUS (y is real-valued)
│   │
│   ├─ Is there a lagged dependent variable (y_{t-1})?
│   │   ├─ YES → Dynamic models
│   │   │   ├─ Single equation? → GMM (Sec. 2)
│   │   │   └─ Multiple equations? → Panel VAR (Sec. 6)
│   │   │
│   │   └─ NO → Static models
│   │       ├─ Spatial dependence? → Spatial Models (Sec. 3)
│   │       ├─ Endogenous regressor? → Panel IV (Sec. 10)
│   │       ├─ Efficiency/productivity? → SFA (Sec. 4)
│   │       ├─ Distributional effects? → Quantile Regression (Sec. 5)
│   │       └─ Standard panel → Static Models (Sec. 1)
│   │           ├─ Hausman p < 0.05 → Fixed Effects
│   │           └─ Hausman p ≥ 0.05 → Random Effects
│   │
├─ BINARY (0/1)
│   └─ Discrete Choice: Logit / Probit (Sec. 7)
│
├─ COUNT (0, 1, 2, ...)
│   └─ Count Data: Poisson / NegBin / PPML (Sec. 8)
│
├─ ORDERED CATEGORICAL (1 < 2 < 3 ...)
│   └─ Ordered Logit / Ordered Probit (Sec. 7)
│
├─ CENSORED / TRUNCATED
│   └─ Tobit / Honore (Sec. 9)
│
└─ SELECTION BIAS (outcome observed only for a subsample)
    └─ Panel Heckman (Sec. 9)
```

## By Data Type

### Continuous Dependent Variable

When your outcome is a real-valued continuous variable (wages, investment, GDP).

=== "Static (no dynamics)"

    | Model | PanelBox Class | When to Use |
    |-------|---------------|-------------|
    | Pooled OLS | `PooledOLS` | No entity effects (baseline) |
    | Fixed Effects | `FixedEffects` | Entity effects correlated with X |
    | Random Effects | `RandomEffects` | Entity effects uncorrelated with X |
    | Between | `BetweenEstimator` | Cross-sectional variation only |
    | First Difference | `FirstDifferenceEstimator` | Alternative to FE for short T |

    ```python
    from panelbox import FixedEffects
    model = FixedEffects("y ~ x1 + x2", data, "id", "year")
    results = model.fit(cov_type="clustered")
    ```

=== "Dynamic (lagged y)"

    | Model | PanelBox Class | When to Use |
    |-------|---------------|-------------|
    | Difference GMM | `DifferenceGMM` | Arellano-Bond; moderate persistence |
    | System GMM | `SystemGMM` | Blundell-Bond; high persistence |
    | CUE GMM | `CUEGMM` | Continuous updating; robust to weak instruments |
    | Bias-Corrected | `BiasCorrectedGMM` | Small-sample correction |

    ```python
    from panelbox.gmm import SystemGMM
    model = SystemGMM("n ~ L.n + w + k", data, "id", "year",
                       gmm_instruments=["L.n"], iv_instruments=["w", "k"])
    results = model.fit(two_step=True)
    ```

=== "Spatial"

    | Model | PanelBox Class | When to Use |
    |-------|---------------|-------------|
    | Spatial Lag (SAR) | `SpatialLag` | Outcome depends on neighbors' outcomes |
    | Spatial Error (SEM) | `SpatialError` | Errors correlated across neighbors |
    | Spatial Durbin (SDM) | `SpatialDurbin` | Both spatial lag and spatial X effects |
    | Dynamic Spatial | `DynamicSpatialPanel` | Spatial + temporal dynamics |
    | General Nesting | `GeneralNestingSpatial` | Most flexible spatial specification |

    ```python
    from panelbox.models.spatial import SpatialLag
    model = SpatialLag("y ~ x1 + x2", data, "region", "year", W=W)
    results = model.fit()
    ```

=== "Other Continuous"

    | Model | PanelBox Class | When to Use |
    |-------|---------------|-------------|
    | Panel IV (2SLS) | `PanelIV` | Endogenous regressor with instruments |
    | Stochastic Frontier | `StochasticFrontier` | Efficiency/productivity analysis |
    | Four-Component SFA | `FourComponentSFA` | Persistent + transient inefficiency |
    | Quantile Regression | `FixedEffectsQuantile` | Effects at different quantiles |

### Binary Dependent Variable

When your outcome is 0 or 1 (employment status, default, adoption).

| Model | PanelBox Class | When to Use | Stata Equivalent |
|-------|---------------|-------------|-----------------|
| Pooled Logit | `PooledLogit` | No entity effects | `logit` |
| Pooled Probit | `PooledProbit` | No entity effects | `probit` |
| FE Logit | `FixedEffectsLogit` | Conditional logit (entity effects) | `clogit` / `xtlogit, fe` |
| RE Probit | `RandomEffectsProbit` | Random entity effects | `xtprobit, re` |
| Dynamic Discrete | `DynamicDiscreteChoice` | State dependence (lagged y) | `xtdpdml` |

```python
from panelbox.models.discrete.binary import FixedEffectsLogit
model = FixedEffectsLogit("employed ~ age + education", data, "id", "year")
results = model.fit()
```

### Count Dependent Variable

When your outcome is a non-negative integer (number of patents, accidents, visits).

| Model | PanelBox Class | When to Use | Stata Equivalent |
|-------|---------------|-------------|-----------------|
| Poisson FE | `PoissonFE` | Count data with entity effects | `xtpoisson, fe` |
| Poisson RE | `PoissonRE` | Count data, random effects | `xtpoisson, re` |
| PPML | `PPML` | Gravity models, zero-robust | `ppmlhdfe` |
| Negative Binomial | `NegativeBinomialFE` | Overdispersed counts | `xtnbreg` |
| Zero-Inflated Poisson | `ZeroInflatedPoisson` | Excess zeros | `zip` |
| Zero-Inflated NB | `ZeroInflatedNB` | Excess zeros + overdispersion | `zinb` |

```python
from panelbox.models.count.poisson import PoissonFE
model = PoissonFE("patents ~ rd + size", data, "firm", "year")
results = model.fit()
```

### Ordered Categorical Dependent Variable

When your outcome has a natural ordering (satisfaction: 1--5, credit rating: A--D).

| Model | PanelBox Class | Stata Equivalent |
|-------|---------------|-----------------|
| Ordered Logit | `OrderedLogit` | `ologit` |
| Ordered Probit | `OrderedProbit` | `oprobit` |

```python
from panelbox.models.discrete.ordered import OrderedLogit
model = OrderedLogit("rating ~ size + leverage", data, "firm", "year")
results = model.fit()
```

### Censored, Truncated, or Selection

When your outcome is censored (e.g., capped at zero), truncated, or selectively observed.

| Model | PanelBox Class | When to Use | Stata Equivalent |
|-------|---------------|-------------|-----------------|
| Tobit | `Tobit` | Censored at a threshold | `xttobit` |
| Honore | `Honore` | FE Tobit (trimmed LAD) | -- |
| Panel Heckman | `PanelHeckman` | Selection bias correction | `heckman` (panel) |

```python
from panelbox.models.censored import Tobit
model = Tobit("hours ~ wage + children", data, "id", "year", lower=0)
results = model.fit()
```

## By Research Question

### "I need to control for unobserved entity characteristics"

**Use: Fixed Effects vs Random Effects**

Run the Hausman test to decide:

```python
from panelbox import FixedEffects, RandomEffects
from panelbox.validation import HausmanTest

fe = FixedEffects("y ~ x1 + x2", data, "id", "year").fit()
re = RandomEffects("y ~ x1 + x2", data, "id", "year").fit()
print(HausmanTest(fe, re))
```

- **p < 0.05**: Use Fixed Effects (effects correlated with X)
- **p >= 0.05**: Use Random Effects (more efficient)

### "My dependent variable depends on its own lag"

**Use: GMM estimators**

Fixed Effects is biased when the model includes a lagged dependent variable (Nickell bias). GMM uses instrumental variables to obtain consistent estimates.

```python
from panelbox.gmm import SystemGMM
model = SystemGMM("y ~ L.y + x1 + x2", data, "id", "year",
                   gmm_instruments=["L.y"], iv_instruments=["x1", "x2"])
results = model.fit(two_step=True)
```

!!! tip "Difference vs System GMM"
    Use **System GMM** when the dependent variable is highly persistent (autoregressive coefficient > 0.8). Use **Difference GMM** as a more conservative baseline.

### "My entities are spatially connected"

**Use: Spatial panel models**

When outcomes or errors are correlated across neighboring entities (regions, countries):

- **SAR**: Neighbors' outcomes affect yours (trade spillovers, policy diffusion)
- **SEM**: Shared unobserved shocks across neighbors
- **SDM**: Both outcome and regressor spillovers

### "I want to measure efficiency or productivity"

**Use: Stochastic Frontier Analysis (SFA)**

```python
from panelbox.frontier import StochasticFrontier
model = StochasticFrontier("output ~ labor + capital", data, "firm", "year",
                            frontier_type="production", dist_type="half_normal")
results = model.fit()
```

PanelBox also offers `FourComponentSFA` for decomposing persistent vs transient inefficiency -- a model unique to PanelBox among Python libraries.

### "I want effects at different points of the distribution"

**Use: Quantile Regression**

Standard regression estimates the mean effect. Quantile regression estimates effects at the median, 10th percentile, 90th percentile, etc.

| Model | PanelBox Class | Description |
|-------|---------------|-------------|
| Pooled Quantile | `PooledQuantile` | Ignores panel structure |
| FE Quantile | `FixedEffectsQuantile` | With entity fixed effects |
| Canay Two-Step | `CanayTwoStep` | Debiased FE quantile |
| Location-Scale | `LocationScale` | Heterogeneous effects on variance |
| QTE | `QuantileTreatmentEffects` | Treatment effects at quantiles |

### "I have multiple interrelated outcome variables"

**Use: Panel VAR**

Model dynamic interdependencies between multiple variables (e.g., GDP, investment, and consumption jointly):

```python
from panelbox.var import PanelVAR
model = PanelVAR(data, ["gdp", "invest", "consumption"], "country", "year", lags=2)
results = model.fit()

# Impulse response functions
irf = results.irf(periods=10)
irf.plot()
```

## Comprehensive Comparison Table

| Family | Key Models | When to Use | Key Assumption | Stata Equivalent |
|--------|-----------|-------------|----------------|-----------------|
| **Static** | FE, RE, Pooled | Standard panel, no dynamics | Strict exogeneity | `xtreg` |
| **Dynamic GMM** | Diff-GMM, Sys-GMM | Lagged dependent variable | Sequential exogeneity | `xtabond2` |
| **Spatial** | SAR, SEM, SDM | Spatially connected entities | Known weight matrix | `xsmle` |
| **SFA** | SF, 4-Component | Efficiency measurement | Composed error (v + u) | `xtfrontier` |
| **Quantile** | FE Quantile, Canay | Distributional effects | Quantile restrictions | `qregpd` |
| **Panel VAR** | PVAR, PVECM | Multiple endogenous variables | Stationarity (VAR) | `pvar` |
| **Discrete** | Logit, Probit, FE Logit | Binary/multinomial outcome | Latent variable model | `xtlogit`, `xtprobit` |
| **Count** | Poisson, NegBin, PPML | Count/non-negative outcome | Conditional mean spec. | `xtpoisson`, `ppmlhdfe` |
| **Censored** | Tobit, Honore | Censored/truncated outcome | Normality (Tobit) | `xttobit` |
| **Selection** | Panel Heckman | Non-random sample selection | Exclusion restriction | `heckman` |
| **IV** | Panel 2SLS | Endogenous regressors | Instrument validity | `xtivreg` |
| **Standard Errors** | HC, Cluster, DK, PCSE | All models | Varies by type | `vce()` options |
| **Diagnostics** | Hausman, Mundlak, RESET | Model validation | Varies by test | Various |

## Common Research Scenarios

### Scenario 1: Firm Investment Decisions

**Data**: 500 firms, 10 years. Variables: investment, sales, debt, Tobin's Q.

**Question**: What drives firm investment?

**Recommendation**: Start with **Fixed Effects** (firm-specific unobservables like management quality). If investment is persistent, switch to **System GMM** with lagged investment as an endogenous regressor.

### Scenario 2: Effect of Education on Wages

**Data**: 5,000 workers, 5 annual surveys. Variables: wage, education, experience, industry.

**Question**: What is the return to education?

**Recommendation**: **Fixed Effects** to control for unobserved ability. Note that if education doesn't change much over time, FE will have low within-variation and imprecise estimates. Consider **Random Effects** with the Mundlak correction as an alternative.

### Scenario 3: Regional Economic Growth

**Data**: 200 regions, 20 years, with geographic adjacency.

**Question**: Do neighboring regions' growth rates affect local growth?

**Recommendation**: **Spatial Durbin Model (SDM)** to capture both direct effects and spatial spillovers through the weight matrix.

### Scenario 4: Patent Activity

**Data**: 1,000 firms, 15 years. Variable: number of patents (count).

**Question**: How does R&D spending affect patent output?

**Recommendation**: **Poisson FE** for count data with firm fixed effects. If there are excess zeros (many firms with zero patents), use **Zero-Inflated Poisson**. If variance exceeds the mean, use **Negative Binomial**.

### Scenario 5: Bank Default Prediction

**Data**: 800 banks, 12 quarters. Variable: default (0/1).

**Question**: What predicts bank failure?

**Recommendation**: **Fixed Effects Logit** (conditional logit) for binary outcome with bank-specific unobservables. If state dependence matters (past default predicts future default), use **Dynamic Discrete Choice**.

### Scenario 6: Hospital Efficiency

**Data**: 300 hospitals, 8 years. Variables: output (patients treated), inputs (staff, beds, equipment).

**Question**: How efficient are hospitals, and has efficiency changed over time?

**Recommendation**: **Stochastic Frontier Analysis** with a production frontier. Use **Four-Component SFA** to separate persistent inefficiency (structural issues) from transient inefficiency (temporary shocks).

## Testing Your Model Choice

After selecting a model, run diagnostic tests to validate the choice:

### Static Model Diagnostics

```python
from panelbox import FixedEffects, RandomEffects
from panelbox.validation import HausmanTest, MundlakTest

fe_results = FixedEffects("y ~ x1 + x2", data, "id", "year").fit()
re_results = RandomEffects("y ~ x1 + x2", data, "id", "year").fit()

# Hausman test: FE vs RE
print(HausmanTest(fe_results, re_results))

# Mundlak test: alternative to Hausman
print(MundlakTest(re_results))
```

### GMM Diagnostics

```python
# After fitting a GMM model:
# 1. Hansen J-test (overidentification)
print(f"Hansen J p-value: {results.hansen_j.pvalue:.3f}")  # Want p > 0.10

# 2. AR(2) test (no second-order serial correlation)
print(f"AR(2) p-value: {results.ar2_test.pvalue:.3f}")     # Want p > 0.10

# 3. Instrument count
print(f"Instruments: {results.n_instruments}")
print(f"Groups: {results.n_entities}")                      # Want instruments < groups
```

### General Specification Tests

| Test | PanelBox Class | Null Hypothesis | Use When |
|------|---------------|-----------------|----------|
| Hausman | `HausmanTest` | RE is consistent | Choosing FE vs RE |
| Mundlak | `MundlakTest` | No correlation with effects | Alternative to Hausman |
| Breusch-Pagan | `BreuschPaganTest` | Homoskedasticity | Checking error variance |
| Wooldridge AR | `WooldridgeTest` | No serial correlation | Checking autocorrelation |
| Pesaran CD | `PesaranCDTest` | No cross-sectional dependence | Macro panels |
| RESET | `RESETTest` | Correct functional form | Specification check |
| Chow | `ChowTest` | Structural stability | Testing for breaks |

## Further Reading

Each model family has a dedicated section in the User Guide:

- [Static Models](../user-guide/static-models/index.md) -- Pooled OLS, FE, RE, Between, First Difference
- [Dynamic GMM](../user-guide/gmm/index.md) -- Arellano-Bond, Blundell-Bond, CUE, Bias-Corrected
- [Spatial Models](../user-guide/spatial/index.md) -- SAR, SEM, SDM, Dynamic Spatial
- [Stochastic Frontier](../user-guide/frontier/index.md) -- SFA, Four-Component, TFP
- [Quantile Regression](../user-guide/quantile/index.md) -- FE Quantile, Canay, Location-Scale
- [Panel VAR](../user-guide/var/index.md) -- VAR, VECM, IRF, FEVD, Granger Causality
- [Discrete Choice](../user-guide/discrete/index.md) -- Logit, Probit, Ordered, Multinomial, Dynamic
- [Count Data](../user-guide/count/index.md) -- Poisson, NegBin, PPML, Zero-Inflated
- [Censored and Selection](../user-guide/censored/index.md) -- Tobit, Honore, Panel Heckman
- [Panel IV](../user-guide/iv/index.md) -- Two-Stage Least Squares
- [Standard Errors](../inference/index.md) -- Robust, Clustered, Driscoll-Kraay, PCSE
- [Diagnostics](../diagnostics/index.md) -- 50+ specification and misspecification tests
