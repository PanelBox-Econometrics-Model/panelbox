# Panel Cointegration: Theory and Methods

## 1. Introduction

### 1.1 What is Cointegration?

**Cointegration** describes a long-run equilibrium relationship between non-stationary (I(1)) variables. While individual series may wander without bounds, cointegrated variables move together in a predictable way.

**Economic Intuition:**
- **Purchasing Power Parity (PPP):** Exchange rates and relative price levels
- **Interest Rate Parity:** Domestic and foreign interest rates
- **Consumption-Income:** Long-run relationship despite short-run shocks

### 1.2 Why Panel Cointegration?

**Advantages over Time Series:**
1. **More Power:** Cross-sectional dimension increases test power
2. **Heterogeneity:** Allow different cointegrating relationships across entities
3. **Robustness:** Panel asymptotics (N→∞, T→∞) vs time series (T→∞ only)
4. **Policy Insights:** Identify common vs. heterogeneous long-run relationships

**Example Applications:**
- **Macroeconomics:** PPP, consumption functions across countries
- **Finance:** Asset price relationships, portfolio cointegration
- **Development:** Growth convergence, trade relationships

---

## 2. Theoretical Framework

### 2.1 Notation and Setup

**Panel Structure:**
- **Entities:** i = 1, ..., N (countries, firms, individuals)
- **Time Periods:** t = 1, ..., T
- **Variables:** y_it, x_it (scalars or vectors)

**Integration Order:**
- **I(1):** Integrated of order 1 (unit root, non-stationary)
- **I(0):** Integrated of order 0 (stationary)

### 2.2 Cointegrating Regression

**General Form:**
```
y_it = α_i + β_i' x_it + ε_it
```

Where:
- `y_it` ~ I(1)
- `x_it` ~ I(1) (k-dimensional vector)
- `ε_it` ~ I(0) (cointegrating residual)
- `β_i` is the cointegrating vector

**Cointegration Condition:**
If ε_it ~ I(0), then (y_it, x_it) are cointegrated with vector β_i.

### 2.3 Homogeneous vs. Heterogeneous Cointegration

**Homogeneous (Kao):**
```
β_i = β for all i
```
- Same long-run relationship across all entities
- More restrictive but higher power if true
- Appropriate for: Identical economic structures

**Heterogeneous (Pedroni, Westerlund):**
```
β_i varies across i
```
- Entity-specific long-run relationships
- More flexible, lower power
- Appropriate for: Diverse economies, structural differences

---

## 3. Test Families

### 3.1 Kao (1999) Tests

#### Theory

**Pooled Regression:**
```
y_it = α_i + β' x_it + u_it
```

**Null Hypothesis:** No cointegration
```
H0: u_it ~ I(1) (has unit root)
```

**Test Procedure:**
1. Estimate pooled OLS regression
2. Obtain pooled residuals: û_it
3. Test for unit root in residuals:
   - **DF test:** Δû_it = ρ û_i,t-1 + ν_it
   - **ADF test:** Δû_it = ρ û_i,t-1 + Σ δ_j Δû_i,t-j + ν_it

**Test Statistic:**
```
t_ρ = (ρ̂ - 1) / SE(ρ̂)
```

Adjusted for panel structure with bias corrections.

#### Critical Values

Kao (1999) provides asymptotic critical values:
- 1%: -2.58
- 5%: -1.96
- 10%: -1.64

**Distribution:** Standard normal N(0,1) after adjustments

#### Advantages and Limitations

✅ **Advantages:**
- Simple to compute
- Clear economic interpretation (pooled relationship)
- Good power when homogeneity holds

❌ **Limitations:**
- Assumes β_i = β (homogeneity)
- Over-rejects in finite samples
- Sensitive to cross-sectional dependence

---

### 3.2 Pedroni (1999, 2004) Tests

#### Theory

**Heterogeneous Cointegrating Regression:**
```
y_it = α_i + δ_i t + β_i' x_it + ε_it
```

Where β_i varies across i.

**Null Hypothesis:** No cointegration for any i
```
H0: ε_it ~ I(1) for all i
```

**Test Procedure:**
1. Estimate entity-specific regressions
2. Obtain residuals: ε̂_it for each i
3. Construct panel/group unit root statistics

#### Seven Test Statistics

**Within-Dimension (Panel):**

1. **Panel ν-statistic:**
   ```
   Z_ν = (Σ_i Σ_t L̂²_i ε̂²_i,t-1)^(-1)
   ```
   - Variance ratio statistic
   - Non-parametric

2. **Panel ρ-statistic:**
   ```
   Z_ρ - 1 = (Σ_i Σ_t ε̂²_i,t-1)^(-1) Σ_i Σ_t (ε̂_i,t-1 Δε̂_it - λ̂_i)
   ```
   - Pooled Dickey-Fuller statistic
   - Non-parametric (uses kernel estimators)

3. **Panel PP-statistic:**
   ```
   Z_t = (s²Σ_i Σ_t ε̂²_i,t-1)^(-1/2) Σ_i Σ_t (ε̂_i,t-1 Δε̂_it - λ̂_i)
   ```
   - Phillips-Perron type
   - Non-parametric (heteroskedasticity-robust)

4. **Panel ADF-statistic:**
   ```
   Z_ADF = (s²Σ_i Σ_t ε̂*²_i,t-1)^(-1/2) Σ_i Σ_t (ε̂*_i,t-1 Δε̂*_it)
   ```
   - Augmented Dickey-Fuller type
   - Parametric (includes lags)

**Between-Dimension (Group):**

5. **Group ρ-statistic:**
   ```
   Z̃_ρ - 1 = Σ_i (Σ_t ε̂²_i,t-1)^(-1) Σ_t (ε̂_i,t-1 Δε̂_it - λ̂_i)
   ```

6. **Group PP-statistic:**
   ```
   Z̃_t = Σ_i (σ̂²_i Σ_t ε̂²_i,t-1)^(-1/2) Σ_t (ε̂_i,t-1 Δε̂_it - λ̂_i)
   ```

7. **Group ADF-statistic:**
   ```
   Z̃_ADF = Σ_i (s²_i Σ_t ε̂*²_i,t-1)^(-1/2) Σ_t (ε̂*_i,t-1 Δε̂*_it)
   ```

**Interpretation:**
- **Panel stats:** Pool information across i (common ρ)
- **Group stats:** Average individual statistics (heterogeneous ρ_i)

#### Critical Values

Pedroni (2004) provides Monte Carlo critical values depending on:
- N (number of entities)
- T (number of periods)
- Trend specification ('n', 'c', 'ct')

**Distribution:** Non-standard (tabulated)

#### Advantages and Limitations

✅ **Advantages:**
- Allows heterogeneous β_i
- 7 statistics provide robustness check
- Well-established, widely used

❌ **Limitations:**
- Panel ν has poor finite-sample properties (over-rejects)
- Critical values complex (require interpolation)
- Sensitive to deterministic trend specification

---

### 3.3 Westerlund (2007) Tests

#### Theory

**Error Correction Model (ECM):**
```
Δy_it = α_i d_t + α_i(y_i,t-1 - β_i x_i,t-1)
        + Σ_j γ_ij Δy_i,t-j
        + Σ_j δ_ij Δx_i,t-j
        + ε_it
```

Where:
- `α_i` is error correction parameter
- `d_t` are deterministics (constant, trend)
- `γ_ij, δ_ij` are short-run dynamics

**Null Hypothesis:** No error correction → No cointegration
```
H0: α_i = 0 for all i
```

**Alternative:**
```
H1: α_i < 0 for some/all i
```

#### Four Test Statistics

**Group-Mean Statistics:**

1. **G_t statistic:**
   ```
   G_t = 1/N Σ_i (α̂_i / SE(α̂_i))
   ```
   - Average t-statistic for error correction
   - Tests: α_i < 0 for at least some i

2. **G_a statistic:**
   ```
   G_a = 1/N Σ_i T α̂_i / α̂_i(1)
   ```
   - Average ratio statistic
   - Normalizes by first-order autocorrelation

**Panel Statistics:**

3. **P_t statistic:**
   ```
   P_t = α̂ / SE(α̂)
   ```
   - Pooled t-statistic
   - Tests: α < 0 (pooled)

4. **P_a statistic:**
   ```
   P_a = T α̂
   ```
   - Pooled ratio statistic

**Interpretation:**
- **G statistics:** Allow heterogeneous error correction (α_i)
- **P statistics:** Assume pooled error correction (α)

#### Bootstrap Critical Values

**Procedure:**
1. Generate data under H0 (α = 0)
2. Re-estimate test statistics
3. Repeat B times
4. Empirical distribution → p-values

**Advantages:**
- Finite-sample accuracy
- Robust to heteroskedasticity
- No need for tabulated values

#### Automatic Lag Selection

**AIC/BIC Criteria:**
For each entity i, select lags p_i to minimize:
```
AIC = log(σ̂²) + 2p/T
BIC = log(σ̂²) + p log(T)/T
```

Where σ̂² is residual variance from ECM.

#### Advantages and Limitations

✅ **Advantages:**
- Based on ECM (direct test of error correction)
- Bootstrap option (better finite-sample properties)
- Automatic lag selection
- Robust to heterogeneous dynamics

❌ **Limitations:**
- G_a may have low power in some configurations
- Computationally intensive with bootstrap
- Requires choosing max lags (p_max)

---

## 4. Asymptotic Theory

### 4.1 Panel Asymptotics

**Sequential Asymptotics:**
```
T → ∞, then N → ∞
```

**Joint Asymptotics:**
```
(N,T) → ∞ jointly with N/T → k ∈ (0,∞)
```

**Implications:**
- Different limiting distributions
- Critical values depend on asymptotic assumption
- Finite-sample performance varies

### 4.2 Cross-Sectional Independence

**Assumption (Strong):**
```
E[ε_it ε_js] = 0 for all i ≠ j, t, s
```

**Violation:**
- Common shocks (oil prices, financial crises)
- Spatial dependence
- Trade/financial linkages

**Consequences if violated:**
- Size distortions
- Invalid inference
- Need for cross-sectionally augmented tests

**Testing:**
Use Pesaran (2004) CD test before cointegration testing.

---

## 5. Practical Guidance

### 5.1 Which Test to Use?

**Decision Tree:**

1. **Is β_i likely homogeneous?**
   - Yes → Kao test
   - No → Pedroni or Westerlund

2. **Sample size:**
   - Small (N < 30, T < 50) → Westerlund with bootstrap
   - Medium (N=30-100, T=50-100) → All tests
   - Large (N,T > 100) → All tests fine

3. **Cross-sectional dependence:**
   - High → ⚠️ Use with caution, wait for CS-augmented versions
   - Low → Any test

4. **Computational budget:**
   - Limited → Kao, Pedroni
   - Unlimited → Westerlund with high bootstrap replications

### 5.2 Interpreting Results

**Consensus Approach:**
Run multiple tests and look for agreement:

- **Strong evidence:** ≥2 test families reject H0
- **Moderate evidence:** 1 test family rejects consistently
- **Weak evidence:** Mixed results across tests
- **No evidence:** No tests reject H0

**Example:**
```
Kao:         p = 0.02 ✅ Reject
Pedroni:     5/7 stats reject ✅ Mostly reject
Westerlund:  Pt, Pa reject ✅ Reject

→ Strong evidence of cointegration
```

### 5.3 Robustness Checks

1. **Trend specification:**
   - Try 'n', 'c', 'ct'
   - PPP typically uses 'c'
   - GDP growth may need 'ct'

2. **Lag selection:**
   - Use AIC/BIC
   - Verify robustness to lag length
   - Too few → size distortion
   - Too many → power loss

3. **Sample splits:**
   - Test stability over sub-periods
   - Cointegration may break down (structural breaks)

4. **Cross-sectional dependence:**
   - Always test first (CD test)
   - If significant, interpret with caution

---

## 6. Economic Interpretation

### 6.1 What Does Rejection Mean?

**Reject H0:**
- Long-run equilibrium relationship exists
- Deviations are temporary (mean-reverting)
- Economic forces pull variables back together
- Can estimate VECM to quantify speed of adjustment

**Fail to Reject H0:**
- No evidence of cointegration
- Variables may drift apart permanently
- Estimate in differences (VAR in Δy, Δx)
- Or re-examine variable selection

### 6.2 Example: Purchasing Power Parity (PPP)

**Theory:** Exchange rate adjusts to equalize prices across countries.

**Relationship:**
```
s_it = p_it - p*_it
```
Where:
- s_it = log exchange rate (domestic/foreign)
- p_it = log domestic price level
- p*_it = log foreign price level

**Cointegration Test:**
1. Test s, p, p* for unit roots → All I(1)
2. Test for cointegration (β = [1, -1])
3. If cointegrated → PPP holds in long run

**Economic Meaning:**
- α < 0 → Speed of PPP adjustment
- |α| large → Fast adjustment (competitive markets)
- |α| small → Slow adjustment (trade barriers, nominal rigidities)

### 6.3 Example: Consumption-Income

**Permanent Income Hypothesis:**
```
c_it = β y_it + ε_it
```
Where:
- c = consumption
- y = income
- β = marginal propensity to consume (MPC)

**If cointegrated:**
- β̂ estimates long-run MPC
- ε_it captures transitory consumption shocks
- Can estimate error correction: Δc depends on (c - βy)

---

## 7. Extensions and Future Work

### 7.1 Cross-Sectional Dependence

**Problem:** Global shocks violate independence assumption.

**Solutions:**
- **Pesaran (2007):** Cross-sectionally augmented tests
- **Bai & Ng (2004):** Factor structure
- **Gengenbach et al. (2016):** Panel analysis of common factors

**Status in PanelBox:** Planned for future release.

### 7.2 Structural Breaks

**Problem:** Cointegration relationship may change over time.

**Solutions:**
- **Gregory & Hansen (1996):** Tests with unknown break point
- **Westerlund (2006):** Panel tests with breaks

**Status:** Not yet implemented.

### 7.3 Non-linear Cointegration

**Problem:** Adjustment may be non-linear (threshold effects).

**Solutions:**
- **Hansen & Seo (2002):** Threshold cointegration
- **Panel threshold models**

**Status:** Not yet implemented.

---

## 8. Mathematical Details

### 8.1 Kao Test Standardization

**DF Statistic (pooled):**
```
t_ρ = (ρ̂ - 1) / SE(ρ̂)
```

**Adjustment for panel structure:**
```
t_ρ* = [t_ρ + √(6N σ_v)/(2σ_0v)] / √(σ²_0v/(2σ²_v) + 3σ²_v/(10σ²_0v))
```

Where:
- σ²_v = variance of ν_it
- σ²_0v = long-run variance
- Adjusts for nuisance parameters

**Limiting Distribution:**
```
t_ρ* →^d N(0,1) as (N,T) → ∞
```

### 8.2 Pedroni Panel Statistics

**General Form:**
```
Z = (1/N) Σ_i Z_i
```

Where Z_i is entity-specific statistic.

**Standardization:**
```
Z* = √N (Z - μ) / σ
```

Where μ, σ from Monte Carlo simulations.

**Limiting Distribution:**
```
Z* →^d N(0,1)
```

### 8.3 Westerlund Bootstrap Algorithm

1. **Estimate ECM under H0:**
   ```
   Δy_it = δ_i d_t + Σ γ_ij Δy_i,t-j + Σ δ_ij Δx_i,t-j + ε̂_it
   ```
   (α_i = 0 imposed)

2. **Resample residuals:**
   ```
   ε*_it ~ F̂(ε̂_it)
   ```

3. **Generate bootstrap data:**
   ```
   Δy*_it = δ̂_i d_t + Σ γ̂_ij Δy*_i,t-j + Σ δ̂_ij Δx*_i,t-j + ε*_it
   y*_it = y*_i,t-1 + Δy*_it
   ```

4. **Compute bootstrap statistics:** G*_t, G*_a, P*_t, P*_a

5. **Repeat B times → empirical distribution**

6. **P-value:**
   ```
   p = (1/B) Σ_b I(|G*_b| > |G_obs|)
   ```

---

## 9. References

### Foundational Papers

1. **Kao, C. (1999).** "Spurious Regression and Residual-Based Tests for Cointegration in Panel Data." *Journal of Econometrics*, 90(1), 1-44.

2. **Pedroni, P. (1999).** "Critical Values for Cointegration Tests in Heterogeneous Panels with Multiple Regressors." *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.

3. **Pedroni, P. (2004).** "Panel Cointegration: Asymptotic and Finite Sample Properties of Pooled Time Series Tests with an Application to the PPP Hypothesis." *Econometric Theory*, 20(3), 597-625.

4. **Westerlund, J. (2007).** "Testing for Error Correction in Panel Data." *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.

### Cross-Sectional Dependence

5. **Pesaran, M. H. (2004).** "General Diagnostic Tests for Cross Section Dependence in Panels." *Cambridge Working Papers in Economics*, 0435.

6. **Pesaran, M. H. (2007).** "A Simple Panel Unit Root Test in the Presence of Cross-Section Dependence." *Journal of Applied Econometrics*, 22(2), 265-312.

### Textbooks

7. **Baltagi, B. H. (2021).** *Econometric Analysis of Panel Data* (6th ed.). Springer.

8. **Hsiao, C. (2014).** *Analysis of Panel Data* (3rd ed.). Cambridge University Press.

---

## 10. Summary Comparison

| Feature | Kao (1999) | Pedroni (1999) | Westerlund (2007) |
|---------|------------|----------------|-------------------|
| **Heterogeneity** | Homogeneous β | Heterogeneous β_i | Heterogeneous α_i, β_i |
| **Number of tests** | 2 (DF, ADF) | 7 (panel + group) | 4 (G + P) |
| **Approach** | Pooled residuals | Entity-specific residuals | Error correction |
| **Critical values** | Tabulated (simple) | Tabulated (complex) | Bootstrap option |
| **Finite-sample** | Over-rejects | Panel ν over-rejects | Good with bootstrap |
| **Power** | High (if homogeneous) | Moderate | High |
| **Computation** | Fast | Fast | Slow (with bootstrap) |
| **Cross-section dep** | ❌ Not robust | ❌ Not robust | ❌ Not robust |
| **Recommended for** | Large, homogeneous | Heterogeneous panels | General use |

---

## 11. Glossary

- **I(1):** Integrated of order 1; non-stationary with unit root
- **I(0):** Integrated of order 0; stationary
- **Cointegration:** Long-run equilibrium relationship between I(1) variables
- **Error Correction:** Speed at which system returns to equilibrium
- **ECM:** Error Correction Model
- **VECM:** Vector Error Correction Model
- **Spurious Regression:** False relationship between unrelated I(1) variables
- **Panel:** Cross-sectional (N) and time series (T) data combined
- **Homogeneity:** Same parameters across all entities
- **Heterogeneity:** Different parameters across entities
