# Quantile Treatment Effects (QTE) Analysis

## Overview

This example demonstrates comprehensive estimation and inference for heterogeneous treatment effects using quantile regression methods. It showcases how treatment effects can vary dramatically across the outcome distribution, providing insights beyond average treatment effects.

## Key Features

1. **Unconditional QTE**: Simple comparison of quantiles between treated and control groups
2. **Conditional QTE**: Treatment effects controlling for covariates using quantile regression
3. **Difference-in-Differences QTE**: Panel data methods to control for time-invariant unobservables
4. **Policy Simulation**: Distributional impact assessment of universal treatment

## Economic Context

Average treatment effects (ATE) from OLS or difference-in-differences can mask important heterogeneity:

- **Job training**: May help low-skilled workers more than high-skilled
- **Medical treatment**: Efficacy may vary by disease severity
- **Education policy**: Returns may differ across ability distribution
- **Minimum wage**: Effects differ across wage distribution

QTE methods reveal:
- Who benefits most from treatment?
- Does treatment reduce or increase inequality?
- What is the distributional impact of scaling up the program?

## Dataset

The example uses simulated job training program data with realistic features:

- **Panel structure**: 2,000 individuals × 2 periods (pre/post)
- **Outcome**: Log earnings (Mincer equation)
- **Treatment**: Job training participation (non-random)
- **Covariates**: Education, experience, gender
- **Selection**: Lower-skilled workers more likely to participate
- **Heterogeneous effects**: Larger gains for low-skill workers

### Data Generating Process

```python
# Earnings equation
log_earnings = α + β₁×education + β₂×experience + β₃×female +
               δ(skill)×treatment + ε

# Treatment effect varies by skill level
δ(skill) = 0.15×(1 - skill) + 0.05×skill
```

This generates:
- Larger effects for low-skill workers (up to 15% gain)
- Smaller effects for high-skill workers (~5% gain)
- Average effect masks this heterogeneity

## Key Results

### 1. Unconditional QTE

Simple quantile comparison (no covariates):

| Quantile | Treated | Control | QTE | % Effect |
|----------|---------|---------|-----|----------|
| 0.10 | 2.80 | 2.65 | 0.15 | +16% |
| 0.50 | 3.10 | 3.00 | 0.10 | +11% |
| 0.90 | 3.40 | 3.35 | 0.05 | +5% |

**Interpretation**: Treatment effects decline from 16% at bottom to 5% at top of distribution.

### 2. Conditional QTE

Controlling for education, experience, gender:

| Quantile | QTE | Std Error | t-stat | p-value |
|----------|-----|-----------|--------|---------|
| 0.10 | 0.148 | 0.018 | 8.2 | <0.001 |
| 0.50 | 0.095 | 0.012 | 7.9 | <0.001 |
| 0.90 | 0.052 | 0.015 | 3.5 | <0.001 |

**Heterogeneity Test**: QTE(0.10) - QTE(0.90) = 0.096 (p < 0.001)
→ Strong evidence of heterogeneous effects

### 3. DiD-QTE Results

Panel methods controlling for individual fixed effects:

**Pooled QR DiD**:
- Treats time-invariant characteristics as covariates
- May suffer from bias if unobservables correlated with treatment

**Canay (2011) Two-Step**:
1. Estimate fixed effects via within-transformation
2. Transform outcome by removing FE
3. Run pooled QR on transformed data

Comparison shows Canay estimates are ~15% smaller (less bias).

### 4. Policy Simulation

**Question**: What if we extend training to everyone?

Distributional impact:

| Quantile | Actual | Universal | Change |
|----------|--------|-----------|--------|
| 0.10 | 2.72 | 2.87 | +5.5% |
| 0.50 | 3.05 | 3.14 | +2.9% |
| 0.90 | 3.38 | 3.43 | +1.5% |

**Inequality Impact**:
- Interquartile range: Decreases by 8%
- 90-10 gap: Decreases by 12%
- **Conclusion**: Universal training would reduce earnings inequality

## Usage

### Basic Analysis

```python
from qte_analysis import QuantileTreatmentEffectsAnalysis

# Initialize
analysis = QuantileTreatmentEffectsAnalysis()

# Or load your own data
analysis = QuantileTreatmentEffectsAnalysis(data_path='program_data.csv')

# 1. Unconditional QTE
uncond_qte = analysis.estimate_unconditional_qte()

# 2. Conditional QTE (with covariates)
cond_qte, qr_result = analysis.estimate_conditional_qte()

# 3. DiD-QTE (panel data)
did_results = analysis.estimate_did_qte()

# 4. Policy simulation
analysis.policy_simulation()
```

### Required Data Format

**Cross-sectional analysis**:
- `outcome`: Outcome variable (continuous)
- `treated`: Treatment indicator (0/1)
- Covariates: Additional control variables

**Panel data analysis** (adds):
- `person_id`: Individual identifier
- `period`: Time period (0=pre, 1=post)

### Output

1. **QTE Estimates**: Treatment effects at each quantile
2. **Standard Errors**: Robust or clustered SEs
3. **Heterogeneity Tests**: Statistical tests for varying effects
4. **Visualizations**:
   - QTE across distribution
   - Confidence intervals
   - Comparison plots (DiD methods)
   - Policy simulation results
5. **Publication-ready figures** (300 DPI)

## Methodological Notes

### Unconditional QTE

**Definition**:
```
QTE(τ) = Q_Y1(τ) - Q_Y0(τ)
```
where Q_Y1(τ) is τ-th quantile of treated group.

**Assumption**: Random treatment assignment (or selection on observables)

**Estimation**: Sample quantiles for each group

### Conditional QTE

**Model**:
```
Q_Y(τ|X,D) = X'β(τ) + D×δ(τ)
```

**Interpretation**: δ(τ) is treatment effect at τ-th quantile, controlling for X

**Estimation**: Quantile regression

**Inference**:
- Robust standard errors
- Cluster-robust SEs (for panel data)
- Bootstrap (pairs or wild)

### DiD-QTE

**Pooled specification**:
```
Q_Y(τ|X,D,POST) = β₀(τ) + β₁(τ)×D + β₂(τ)×POST + δ(τ)×D×POST + X'γ(τ)
```

**Canay (2011) two-step**:
1. Estimate αᵢ via within-transformation
2. Compute Ỹᵢₜ = Yᵢₜ - α̂ᵢ
3. Run QR on Ỹᵢₜ

**Assumption**: Fixed effects are pure location shifts (strong!)

**Alternative**: Ponomareva (2011) instrumental variables QR

### Inference Challenges

1. **Bootstrap**:
   - Pairs bootstrap: Resample (i,t) pairs
   - Cluster bootstrap: Resample entire individuals
   - Wild bootstrap: For small number of clusters

2. **Asymptotic**:
   - Powell (1986) kernel-based SEs
   - Clustered SEs for panel data

## Extensions

### Advanced Methods

1. **Distributional DiD**: Callaway, Li, & Oka (2018)
2. **Changes-in-Changes**: Athey & Imbens (2006)
3. **Quantile IV**: Abadie, Angrist, & Imbens (2002)
4. **Synthetic Control QR**: Firpo & Pinto (2016)

### Alternative Estimands

1. **QoTT** (Quantile of Treatment on Treated): Focus on treated group only
2. **QTET** (Quantile Treatment Effect on Treated): Average effect at quantile
3. **Distributional Treatment Effect**: Compare entire distributions

### Robustness Checks

1. **Parallel trends**: Test pre-treatment trend equality
2. **Placebo tests**: Estimate effects in pre-period
3. **Sensitivity**: Vary quantile grid, covariates
4. **Subgroup analysis**: Heterogeneity by demographics

## Applications

### Labor Economics
- **Training programs**: Returns across skill distribution
- **Minimum wage**: Employment effects by wage level
- **Unions**: Wage premium heterogeneity

### Health Economics
- **Medical treatments**: Efficacy by disease severity
- **Insurance**: Moral hazard across spending distribution
- **Public health**: Intervention effects on health outcomes

### Education
- **Class size**: Achievement effects by ability
- **School vouchers**: Distributional impacts
- **Teacher quality**: Effects across student distribution

### Development
- **Microcredit**: Income effects across poverty distribution
- **Cash transfers**: Consumption heterogeneity
- **Infrastructure**: Distributional benefits

## References

### Quantile Treatment Effects
- Doksum, K. A. (1974). Empirical probability plots and statistical inference for nonlinear models. *Annals of Statistics*, 267-277.
- Firpo, S. (2007). Efficient semiparametric estimation of quantile treatment effects. *Econometrica*, 75(1), 259-276.

### Panel Data Methods
- Canay, I. A. (2011). A simple approach to quantile regression for panel data. *Econometrics Journal*, 14(3), 368-386.
- Ponomareva, M. (2011). Quantile regression for panel data models with fixed effects. *Working paper*.

### Difference-in-Differences
- Callaway, B., Li, T., & Oka, T. (2018). Quantile treatment effects in difference in differences models. *Journal of Econometrics*, 206(2), 447-470.
- Athey, S., & Imbens, G. W. (2006). Identification and inference in nonlinear difference‐in‐differences models. *Econometrica*, 74(2), 431-497.

### Applications
- Card, D., Mas, A., & Rothstein, J. (2008). Tipping and the dynamics of segregation. *QJE*, 123(1), 177-218.
- Autor, D., Manning, A., & Smith, C. L. (2016). The contribution of the minimum wage to US wage inequality over three decades. *AEJ: Applied*, 8(1), 58-99.

## Citation

```
PanelBox Development Team (2024). Quantile Treatment Effects Analysis.
PanelBox Examples. https://github.com/panelbox/panelbox
```

## Support

- GitHub: https://github.com/panelbox/panelbox/issues
- Documentation: https://panelbox.readthedocs.io
- Contact: contact@panelbox.org
