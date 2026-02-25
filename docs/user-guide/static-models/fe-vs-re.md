---
title: "Fixed Effects vs Random Effects"
description: "Comprehensive guide to choosing between Fixed Effects and Random Effects models, including the Hausman test, Mundlak test, and practical decision rules."
---

# Fixed Effects vs Random Effects

!!! info "Quick Reference"
    **Module:** `panelbox.models.static`
    **Key classes:** `FixedEffects`, `RandomEffects`
    **Stata equivalent:** `xtreg, fe` / `xtreg, re`
    **R equivalent:** `plm::plm(model="within")` / `plm::plm(model="random")`

## Overview

The choice between Fixed Effects (FE) and Random Effects (RE) is one of the most important decisions in panel data analysis. Both models account for unobserved entity-specific heterogeneity, but they differ in a fundamental assumption: **is the unobserved effect correlated with the regressors?**

- **Fixed Effects** allows $\alpha_i$ to be freely correlated with $X_{it}$ -- always consistent, but cannot estimate time-invariant variables
- **Random Effects** requires $E[u_i | X_{it}] = 0$ (uncorrelated) -- more efficient when assumption holds, but biased if it fails

This guide provides the theoretical background, practical tests, and decision workflow to choose correctly.

## The Fundamental Question

The core distinction is the **correlation assumption**:

| Model | Assumption | What It Means |
|-------|------------|---------------|
| **Fixed Effects** | $E[\alpha_i \| X_{it}] \neq 0$ allowed | Entity effects can be correlated with regressors |
| **Random Effects** | $E[u_i \| X_{it}] = 0$ required | Entity effects must be uncorrelated with regressors |

**Example**: Suppose the unobserved entity effect $\alpha_i$ represents "management quality" in a firm investment model.

- **FE allows**: Good managers may choose different investment levels (correlation between management quality and investment drivers)
- **RE requires**: Management quality is independent of all regressors (often unrealistic)

**Consequence**: FE is consistent in both cases. RE is consistent only if the orthogonality assumption holds -- and biased otherwise.

## Mathematical Background

### Fixed Effects Estimation

The within transformation removes entity effects by demeaning:

$$\tilde{y}_{it} = y_{it} - \bar{y}_i = (X_{it} - \bar{X}_i)\beta + (\varepsilon_{it} - \bar{\varepsilon}_i)$$

$$\hat{\beta}_{FE} = \left(\sum_i \sum_t \tilde{X}_{it} \tilde{X}_{it}'\right)^{-1} \left(\sum_i \sum_t \tilde{X}_{it} \tilde{y}_{it}\right)$$

**Properties:**

- Consistent even if $\alpha_i$ is correlated with $X_{it}$
- Uses only **within-entity** variation (over time)
- Cannot estimate coefficients on time-invariant variables

### Random Effects Estimation

The GLS transformation applies quasi-demeaning:

$$y^*_{it} = y_{it} - \theta \bar{y}_i, \quad X^*_{it} = X_{it} - \theta \bar{X}_i$$

where $\theta = 1 - \sqrt{\sigma^2_\varepsilon / (\sigma^2_\varepsilon + T \sigma^2_u)}$.

**Properties:**

- Consistent **only if** $E[u_i | X_{it}] = 0$
- Uses both **within** and **between** entity variation
- More efficient than FE (smaller standard errors) when assumption holds
- Can estimate time-invariant variables

### Interpretation of $\theta$

The parameter $\theta$ determines how much RE quasi-demeans the data:

- $\theta = 0$: No entity effects ($\sigma^2_u = 0$) -- RE reduces to **Pooled OLS**
- $\theta = 1$: All variation is between entities -- RE reduces to **Fixed Effects**
- $0 < \theta < 1$: Partial quasi-demeaning (typical case)

## The Hausman Test

### Purpose

The Hausman test evaluates whether the RE orthogonality assumption holds by comparing FE and RE coefficient estimates. Since FE is always consistent, any systematic difference between FE and RE estimates indicates that RE is inconsistent.

### Hypotheses

- **$H_0$**: RE is consistent and efficient -- $E[u_i | X_{it}] = 0$ holds
- **$H_1$**: RE is inconsistent -- use FE instead

### Test Statistic

$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' [\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE})$$

Under $H_0$, $H \sim \chi^2(K)$ where $K$ is the number of time-varying regressors.

### Implementation in PanelBox

```python
from panelbox import FixedEffects, RandomEffects
from panelbox.validation import HausmanTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Step 1: Estimate both models
fe_results = FixedEffects("invest ~ value + capital", data, "firm", "year").fit()
re_results = RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# Step 2: Run Hausman test (runs automatically on initialization)
hausman = HausmanTest(fe_results, re_results)

# Step 3: Examine results
print(f"Test statistic: {hausman.statistic:.4f}")
print(f"P-value: {hausman.pvalue:.4f}")
print(f"Degrees of freedom: {hausman.df}")
print(f"Recommendation: {hausman.recommendation}")

# Full summary
print(hausman.summary())
```

### Decision Rule

| p-value | Decision | Interpretation |
|---------|----------|----------------|
| p < 0.05 | **Reject $H_0$** | RE is inconsistent. **Use Fixed Effects.** |
| p >= 0.05 | **Fail to reject** | No evidence against RE. **Use Random Effects** (more efficient). |

!!! tip "Practical Guidance"
    The Hausman test is a useful guide, but should not be the sole criterion. Consider the **research context**, **data structure**, and **theoretical justification** alongside the test result.

### When the Hausman Test Is Inconclusive

The Hausman test can fail or be unreliable in several situations:

- **Negative test statistic**: The variance difference $\text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE})$ can be negative semi-definite, producing a negative chi-squared statistic. This typically occurs with small samples.
- **Borderline p-values** (0.03 -- 0.10): The decision is ambiguous. Consider the Mundlak test as an alternative.
- **Heteroskedastic or autocorrelated errors**: The classic Hausman test assumes i.i.d. errors. Use robust versions when possible.
- **Very small samples**: Low power -- the test may fail to reject even when RE is inconsistent.

## The Mundlak Test

### Purpose

The Mundlak test (Mundlak, 1978) provides an alternative to the Hausman test by augmenting the RE model with entity means of the time-varying regressors. If the means are jointly significant, it indicates that entity effects are correlated with the regressors.

### Model

$$y_{it} = \beta_0 + X_{it} \beta + \bar{X}_i \gamma + u_i + \varepsilon_{it}$$

where $\bar{X}_i$ are entity-level means of the time-varying regressors.

- If $\gamma = 0$: No correlation between entity effects and regressors -- standard RE is valid
- If $\gamma \neq 0$: Correlation exists -- use FE

### Implementation in PanelBox

```python
from panelbox import RandomEffects
from panelbox.validation import MundlakTest
from panelbox.datasets import load_grunfeld

data = load_grunfeld()

# Estimate Random Effects
re_results = RandomEffects("invest ~ value + capital", data, "firm", "year").fit()

# Run Mundlak test
mundlak = MundlakTest(re_results)
result = mundlak.run()
print(result.summary())

# Examine results
print(f"Test statistic: {result.statistic:.4f}")
print(f"P-value: {result.pvalue:.4f}")
print(f"Reject null: {result.reject_null}")
```

### Advantages Over Hausman

- Works with **heteroskedastic and clustered** standard errors
- Provides a **regression-based** framework (easier to extend)
- Can include **time-invariant variables** alongside entity means
- Always produces a valid test statistic (no negative chi-squared issue)

## Comparison Table

| Feature | Fixed Effects | Random Effects |
|---------|---------------|----------------|
| **Assumption** | $E[\alpha_i \| X_{it}]$ unrestricted | $E[u_i \| X_{it}] = 0$ required |
| **Consistency** | Always (under strict exogeneity) | Only if orthogonality holds |
| **Efficiency** | Less efficient | More efficient (if consistent) |
| **Time-invariant X** | Cannot estimate | Can estimate |
| **Intercept** | Absorbed | Estimated |
| **Interpretation** | Within-entity effects | Weighted within + between |
| **Typical use** | Micro (firms, individuals) | Surveys, macro (countries) |
| **Sample** | Any (including non-random) | Preferably random |
| **Robustness** | Very robust | Sensitive to violations |

## Decision Workflow

Follow this systematic workflow to choose between FE and RE:

### Step 1: Estimate Both Models

```python
from panelbox import PooledOLS, FixedEffects, RandomEffects

data = load_grunfeld()

pooled = PooledOLS("invest ~ value + capital", data, "firm", "year").fit()
fe = FixedEffects("invest ~ value + capital", data, "firm", "year").fit(cov_type="clustered")
re = RandomEffects("invest ~ value + capital", data, "firm", "year").fit()
```

### Step 2: Test for Entity Effects (FE vs Pooled OLS)

```python
# F-test for entity effects
print(f"F-statistic: {fe.f_statistic:.4f}")
print(f"F-test p-value: {fe.f_pvalue:.4f}")
# If p < 0.05 -> entity effects exist -> panel structure matters
```

If the F-test is not significant (p >= 0.05), entity effects may not exist. Pooled OLS may be adequate.

### Step 3: Run the Hausman Test

```python
from panelbox.validation import HausmanTest

hausman = HausmanTest(fe, re)
print(f"Hausman p-value: {hausman.pvalue:.4f}")
print(f"Recommendation: {hausman.recommendation}")
```

### Step 4: Confirm with Mundlak Test

```python
from panelbox.validation import MundlakTest

mundlak = MundlakTest(re)
result = mundlak.run()
print(f"Mundlak p-value: {result.pvalue:.4f}")
```

### Step 5: Final Decision

```python
# Decision logic
if hausman.pvalue < 0.05:
    print("Use Fixed Effects (RE assumption violated)")
    final_results = fe
else:
    print("Use Random Effects (more efficient, assumption holds)")
    final_results = re

print(final_results.summary())
```

## Practical Considerations

### Prefer Fixed Effects When:

- **Applied microeconomics**: Firms, individuals, households -- unobserved heterogeneity is almost always correlated with regressors
- **Not a random sample**: Selection bias exists (e.g., S&P 500 firms, specific hospitals)
- **Time-invariant variables are not of interest**: Focus is on time-varying effects
- **Conservative approach**: FE is robust to correlation -- the "safe" choice
- **Hausman rejects RE**: The test indicates RE is inconsistent

### Prefer Random Effects When:

- **Random sample from population**: Survey data with random sampling
- **Time-invariant variables are key**: Gender, race, country characteristics
- **Hausman does not reject**: No evidence of correlation
- **Efficiency matters**: Small sample, need tighter confidence intervals
- **Theoretical justification**: Entity effects are plausibly random draws

### Consider the Mundlak Approach When:

- **Hausman test is inconclusive or fails**: Negative test statistic or borderline p-value
- **You want the best of both worlds**: RE efficiency with FE consistency
- **You need time-invariant variables**: The Mundlak specification allows them while controlling for potential correlation

## Common Scenarios

### Scenario 1: Firm Investment

**Model**: $\text{invest}_{it} = \beta_1 \text{value}_{it} + \beta_2 \text{capital}_{it} + \alpha_i + \varepsilon_{it}$

**Unobserved**: Management quality, corporate culture ($\alpha_i$)

**Question**: Is management quality correlated with firm value and capital stock?

**Answer**: Almost certainly yes. Better managers tend to have more valuable firms.

**Recommendation**: **Use Fixed Effects**

### Scenario 2: Cross-Country Growth

**Model**: $\text{growth}_{it} = \beta_1 \text{investment}_{it} + \beta_2 \text{institutions}_i + \alpha_i + \varepsilon_{it}$

**Unobserved**: Geography, culture ($\alpha_i$)

**Question**: Are institutions time-invariant and of primary interest?

**Answer**: Yes, and the sample may be representative of countries.

**Recommendation**: **Use Random Effects** (can estimate institution effects)

### Scenario 3: Wage Determination

**Model**: $\text{wage}_{it} = \beta_1 \text{education}_{it} + \beta_2 \text{experience}_{it} + \alpha_i + \varepsilon_{it}$

**Unobserved**: Innate ability ($\alpha_i$)

**Question**: Is ability correlated with education?

**Answer**: Almost certainly yes (able people get more education).

**Recommendation**: **Use Fixed Effects**

### Scenario 4: School Performance

**Model**: $\text{score}_{it} = \beta_1 \text{class\_size}_{it} + \beta_2 \text{funding}_{it} + \alpha_i + \varepsilon_{it}$

**Unobserved**: School quality, neighborhood ($\alpha_i$)

**Question**: Does school quality affect class size and funding?

**Answer**: Likely (better schools attract more resources).

**Recommendation**: **Use Fixed Effects**, or run Hausman test to confirm

## Summary Decision Rules

!!! note "Quick Decision Guide"
    1. **Do you need time-invariant variables?** Yes -> Consider RE (run Hausman test)
    2. **Is the sample a random draw?** Yes -> Consider RE (run Hausman test)
    3. **Hausman test**: p < 0.05 -> **Use FE**. p >= 0.05 -> **Use RE**
    4. **When in doubt**: **Use Fixed Effects** (always consistent, safer choice)

```text
START: Panel data with entity-specific effects
  |
  v
Q1: Need time-invariant variable estimates?
  YES -> Consider RE (go to Q3)
  NO  -> Continue
  |
  v
Q2: Random sample from population?
  YES -> Consider RE (go to Q3)
  NO  -> Prefer FE
  |
  v
Q3: Run Hausman Test
  p < 0.05  -> Use Fixed Effects
  p >= 0.05 -> Use Random Effects
  |
  v
DECISION MADE
```

## Tutorials

| Tutorial | Level | Colab |
|----------|-------|-------|
| Random Effects and Hausman Test | Beginner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/fundamentals/03_random_effects_hausman.ipynb) |
| Comparison of All Estimators | Advanced | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PanelBox-Econometrics-Model/panelbox/blob/main/examples/static_models/advanced/06_comparison_estimators.ipynb) |

## See Also

- [Fixed Effects](fixed-effects.md) -- Detailed guide to the Fixed Effects estimator
- [Random Effects](random-effects.md) -- Detailed guide to the Random Effects estimator
- [Pooled OLS](pooled-ols.md) -- Baseline model (no entity effects)
- [Between Estimator](between.md) -- Cross-sectional variation component
- [First Difference](first-difference.md) -- Alternative to FE for removing entity effects

## References

- Hausman, J. A. (1978). "Specification Tests in Econometrics." *Econometrica*, 46(6), 1251--1271.
- Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data." *Econometrica*, 46(1), 69--85.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 10.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. Chapters 2--3.
- Cameron, A. C., & Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. Chapter 21.
