# Comparison of Spatial Panel Models

## Overview

This guide compares the main spatial panel models available in PanelBox, helping you choose the right model for your application. We cover the Spatial Lag (SAR), Spatial Error (SEM), Spatial Durbin (SDM), and General Nesting Spatial (GNS) models.

## Model Specifications

### Spatial Lag Model (SAR)

**Specification**:
$$y_{it} = \rho \sum_{j=1}^N w_{ij} y_{jt} + x_{it}'\beta + \mu_i + \varepsilon_{it}$$

**Key Features**:
- Spatial dependence in the dependent variable
- Captures global spillovers with multiplier effects
- Endogenous spatial interaction

**When to Use**:
- Outcome in one region directly affects outcomes in other regions
- Strategic interactions between units
- Policy spillovers are important

**Example Applications**:
- Regional GDP growth
- Housing price diffusion
- Crime rates
- Technology adoption

### Spatial Error Model (SEM)

**Specification**:
$$y_{it} = x_{it}'\beta + \mu_i + u_{it}$$
$$u_{it} = \lambda \sum_{j=1}^N w_{ij} u_{jt} + \varepsilon_{it}$$

**Key Features**:
- Spatial dependence in the error term
- Captures omitted spatially correlated variables
- No multiplier effects
- Exogenous spatial correlation

**When to Use**:
- Spatial clustering due to omitted variables
- Measurement error with spatial pattern
- Common unobserved shocks
- No direct interaction between units

**Example Applications**:
- Agricultural productivity (weather shocks)
- Retail sales (unobserved demographics)
- Environmental quality
- Health outcomes

### Spatial Durbin Model (SDM)

**Specification**:
$$y_{it} = \rho \sum_{j=1}^N w_{ij} y_{jt} + x_{it}'\beta + \sum_{j=1}^N w_{ij} x_{jt}'\theta + \mu_i + \varepsilon_{it}$$

**Key Features**:
- Spatial lags of both dependent and independent variables
- Most flexible specification
- Captures both local and global spillovers
- Allows for complex interaction patterns

**When to Use**:
- Both outcomes and inputs have spatial spillovers
- Complex diffusion processes
- When LM tests suggest both lag and error
- General specification for robustness

**Example Applications**:
- Knowledge spillovers (R&D)
- Environmental regulations
- Fiscal policy interactions
- Labor market dynamics

### General Nesting Spatial Model (GNS)

**Specification**:
$$y_{it} = \rho \sum_{j=1}^N w_{ij} y_{jt} + x_{it}'\beta + \sum_{j=1}^N w_{ij} x_{jt}'\theta + u_{it}$$
$$u_{it} = \lambda \sum_{j=1}^N w_{ij} u_{jt} + \varepsilon_{it}$$

**Key Features**:
- Nests all other spatial models
- Both spatial lag and spatial error
- Most general specification
- Can test restrictions to simpler models

**When to Use**:
- Model uncertainty
- Testing between specifications
- Very complex spatial processes
- Maximum flexibility needed

**Example Applications**:
- Multi-faceted policy analysis
- Complex urban systems
- International trade
- Migration patterns

## Model Comparison Table

| Feature | SAR | SEM | SDM | GNS |
|---------|-----|-----|-----|-----|
| **Spatial lag of y** | ✓ | ✗ | ✓ | ✓ |
| **Spatial lag of X** | ✗ | ✗ | ✓ | ✓ |
| **Spatial error** | ✗ | ✓ | ✗ | ✓ |
| **Global spillovers** | ✓ | ✗ | ✓ | ✓ |
| **Local spillovers** | ✗ | ✓ | ✓ | ✓ |
| **Parameters** | k+2 | k+2 | 2k+2 | 2k+3 |
| **Estimation complexity** | Medium | Low | High | Very High |
| **Interpretation** | Simple | Simple | Complex | Complex |

## Choosing the Right Model

### Decision Tree

```
Start: Estimate OLS/Panel Model
           ↓
    Test for Spatial Autocorrelation (Moran's I)
           ↓
    Significant? → No → Use OLS/Panel Model
           ↓
          Yes
           ↓
    Run LM Tests (LM-Lag and LM-Error)
           ↓
    ┌──────┼──────┐
    ↓      ↓      ↓
Only Lag  Only   Both
    ↓    Error    ↓
   SAR     ↓      ↓
          SEM   Check Robust LM
                  ↓
            ┌─────┼─────┐
            ↓     ↓     ↓
        R-Lag  R-Error Both
            ↓     ↓     ↓
           SAR   SEM   SDM
```

### Practical Guidelines

**1. Start Simple**
- Always begin with non-spatial model
- Test for spatial autocorrelation
- Add spatial structure only if needed

**2. Use Theory**
- Economic theory should guide model choice
- Consider the mechanism of spatial interaction
- Think about direct vs. indirect channels

**3. Test Restrictions**
- SDM can be tested against SAR and SEM
- GNS can be tested against all nested models
- Use likelihood ratio or Wald tests

**4. Check Robustness**
- Try multiple specifications
- Vary the weight matrix
- Compare key parameters across models

## Parameter Interpretation

### SAR Model

**Spatial Parameter ρ**:
- Measures strength of spatial dependence
- Range: typically (-1, 1) for row-normalized W
- ρ = 0.5: 50% of neighbors' outcomes affect own outcome

**Total Effect**:
$$\frac{\partial y}{\partial x_k} = (I - \rho W)^{-1} \beta_k$$

**Multiplier**: $(1 - \rho)^{-1}$ for aggregate effects

### SEM Model

**Spatial Parameter λ**:
- Measures spatial correlation in errors
- No direct interpretation in outcome units
- Indicates strength of omitted spatial factors

**Effects**:
- Direct effect only: $\beta_k$
- No spillovers in covariates
- Spatial pattern in residuals

### SDM Model

**Direct Effects**:
$$\text{Direct}_k = \beta_k + w_{ii}\theta_k$$

**Indirect Effects**:
$$\text{Indirect}_k = \frac{\sum_{i \neq j}[\partial y_i/\partial x_{jk}]}{N(N-1)}$$

**Total Effects**:
$$\text{Total}_k = \text{Direct}_k + \text{Indirect}_k$$

### GNS Model

Combines interpretations from SAR, SEM, and SDM:
- ρ: Spatial lag effect
- λ: Spatial error correlation
- θ: Spatial lag of covariates
- Complex interaction of all three

## Estimation Methods

### Maximum Likelihood (ML)

**Advantages**:
- Efficient under correct specification
- Allows likelihood-based testing
- Well-established theory

**Disadvantages**:
- Computationally intensive for large N
- Requires computation of log-determinant
- Sensitive to distributional assumptions

**Best for**: Small to medium panels (N < 1000)

### Generalized Method of Moments (GMM)

**Advantages**:
- Computationally faster
- Robust to some misspecification
- No log-determinant needed

**Disadvantages**:
- Less efficient than ML
- Choice of instruments matters
- Weaker for hypothesis testing

**Best for**: Large panels (N > 1000)

### Quasi-Maximum Likelihood (QML)

**Advantages**:
- Robust to non-normality
- Consistent under weaker assumptions
- Good finite sample properties

**Disadvantages**:
- Still requires log-determinant
- Computationally intensive
- More complex inference

**Best for**: Medium panels with non-normal errors

## Model Diagnostics

### Post-Estimation Tests

**Spatial Autocorrelation in Residuals**:
```python
# After spatial model estimation
residuals = model_result.residuals
moran_test = MoranIPanelTest(residuals, W)
# Should be insignificant if model adequate
```

**Common Factor Test** (SDM → SAR/SEM):
$$H_0: \theta + \rho\beta = 0$$

**LR Test for Nested Models**:
```python
# Test SDM against SAR
lr_stat = 2 * (llf_sdm - llf_sar)
p_value = chi2.sf(lr_stat, df=k)  # k = number of X variables
```

### Information Criteria

```python
# Model comparison
models = {
    'SAR': sar_result,
    'SEM': sem_result,
    'SDM': sdm_result
}

for name, result in models.items():
    print(f"{name}: AIC={result.aic:.1f}, BIC={result.bic:.1f}")
```

**Selection Rule**:
- Lower AIC/BIC indicates better fit
- BIC penalizes complexity more
- Consider economic interpretation too

## Practical Examples

### Example 1: Regional Unemployment

**Question**: Do unemployment shocks spread across regions?

**Model Choice Process**:
1. Economic theory: Labor mobility suggests SAR
2. Moran's I on OLS residuals: Significant
3. LM-Lag: Significant
4. LM-Error: Not significant
5. **Choice: SAR model**

### Example 2: Housing Prices

**Question**: How do housing prices influence neighboring areas?

**Model Choice Process**:
1. Theory: Both direct spillovers and common amenities
2. Moran's I: Highly significant
3. LM-Lag: Significant
4. LM-Error: Significant
5. Robust tests: Both significant
6. **Choice: SDM model**

### Example 3: Agricultural Productivity

**Question**: Spatial patterns in crop yields?

**Model Choice Process**:
1. Theory: Weather shocks affect neighbors similarly
2. Moran's I: Significant
3. LM-Lag: Not significant
4. LM-Error: Significant
5. **Choice: SEM model**

## Advanced Considerations

### Spatial Weight Matrix Specification

Different models may be sensitive to W:

**SAR**: Very sensitive to W specification
**SEM**: Moderately sensitive
**SDM**: Can accommodate misspecification better
**GNS**: Most robust but hardest to interpret

### Panel Structure

**Fixed Effects**:
- Controls for time-invariant spatial heterogeneity
- May absorb some spatial dependence
- Reduces need for spatial models

**Random Effects**:
- Allows for correlation between effects and regressors
- May conflate spatial and individual effects
- Requires careful specification

### Dynamic Spatial Models

When time dynamics matter:
$$y_{it} = \tau y_{i,t-1} + \rho \sum_j w_{ij} y_{jt} + x_{it}'\beta + \mu_i + \varepsilon_{it}$$

Combines:
- Temporal persistence (τ)
- Spatial dependence (ρ)
- More complex but realistic

## Common Pitfalls

### 1. Over-parameterization
- GNS/SDM may be too complex for small samples
- Check condition numbers and convergence

### 2. Weight Matrix Misspecification
- Results sensitive to W choice
- Always test robustness

### 3. Ignoring Dynamics
- Panel data often has time dependence too
- Consider dynamic spatial models

### 4. Mechanical Model Selection
- Don't rely only on tests
- Use economic theory
- Consider interpretation

## Summary Recommendations

1. **Default Strategy**: Start with OLS → Test spatial autocorrelation → Use LM tests → Choose model

2. **Conservative Approach**: When in doubt, use SDM (nests SAR/SEM)

3. **Pragmatic Approach**:
   - Small effects: Stay with OLS
   - Clear theory: Use theory-implied model
   - Exploration: Try multiple specifications

4. **For Publication**: Report multiple specifications, justify choice, show robustness

## References

1. Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
2. LeSage, J. and Pace, R.K. (2009). *Introduction to Spatial Econometrics*. Chapman & Hall/CRC.
3. Lee, L.F. and Yu, J. (2010). Estimation of spatial autoregressive panel data models with fixed effects. *Journal of Econometrics*, 154(2), 165-185.
4. Manski, C.F. (1993). Identification of endogenous social effects: The reflection problem. *Review of Economic Studies*, 60(3), 531-542.
