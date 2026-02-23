# NLS Women's Work Panel -- Codebook

## Source

Synthetic labour-market panel inspired by the **National Longitudinal Survey of Young Women (NLS-YW)** 1968--1988, commonly distributed with Stata as `nlswork.dta`. All values are computer-generated and do not represent real individuals. The dataset is designed to exhibit correlated individual effects so that Hausman specification tests reject the random-effects assumption.

## Panel Dimensions

| Dimension | Value |
|-----------|-------|
| Cross-sectional units (N) | 4,000 women |
| Time periods (T) | 1968--1996 (biennial: 15 waves) |
| Total observations | ~40,000 (unbalanced due to attrition) |
| Balance | Unbalanced; average ~10 waves per individual |

## Variable Dictionary

| Variable | Type | Description | Typical Range | Unit |
|----------|------|-------------|---------------|------|
| `idcode` | int | Individual identifier | 1--4,000 | -- |
| `year` | int | Survey year (biennial) | 1968, 1970, ..., 1996 | -- |
| `ln_wage` | float | Natural log of hourly wage | 0.5--4.0 | Log USD |
| `experience` | float | Actual labour-market experience | 0--30 | Years |
| `tenure` | float | Job tenure at current employer | 0--25 | Years |
| `education` | int | Highest grade completed | 8--20 | Years |
| `union` | int | Union membership indicator | 0 or 1 | Binary |
| `married` | int | Currently married indicator | 0 or 1 | Binary |
| `hours` | float | Usual hours worked per week | 10--60 | Hours |
| `industry` | int | Industry classification code | 1--12 | Categorical |

## Data Generating Process

### Wage Equation

The log-wage is generated from a Mincer-style equation with correlated individual heterogeneity:

```
ln_wage_{i,t} = alpha_i + beta_1 * experience_{i,t} + beta_2 * experience_{i,t}^2
              + beta_3 * tenure_{i,t} + beta_4 * education_i
              + beta_5 * union_{i,t} + beta_6 * married_{i,t}
              + beta_7 * log(hours_{i,t}) + epsilon_{i,t}
```

**Parameter values:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `beta_1` | 0.035 | Return to experience (linear) |
| `beta_2` | -0.0005 | Experience diminishing returns |
| `beta_3` | 0.012 | Return to tenure |
| `beta_4` | 0.065 | Return to education |
| `beta_5` | 0.10 | Union wage premium |
| `beta_6` | -0.03 | Marriage penalty (women) |
| `beta_7` | -0.15 | Hours elasticity |

### Individual Effect (`alpha_i`) -- Correlated with Regressors

The individual effect captures unobserved ability and is **correlated with education**:

```
alpha_i = 0.06 * education_i + eta_i
```

- `eta_i ~ N(0, 0.20^2)` -- residual unobserved heterogeneity
- The `0.06 * education_i` component creates an endogeneity problem for random effects, because the individual effect is correlated with the regressor `education`

This correlation ensures that:
- **Fixed effects** removes `alpha_i` and produces consistent estimates of time-varying coefficients
- **Random effects** treats `alpha_i` as uncorrelated with regressors, yielding **biased** estimates
- The **Hausman test should reject** the null of RE consistency

### Idiosyncratic Error

```
epsilon_{i,t} ~ N(0, 0.15^2)
```

Errors are i.i.d. across individuals and time (no serial correlation in idiosyncratic component).

### Time-Varying Covariates

| Variable | DGP |
|----------|-----|
| `experience` | Starts at `U(0, 5)` in first wave; increments by 2 each wave (biennial) with small noise |
| `tenure` | Resets on job change (Bernoulli(0.12) per wave); otherwise increments by 2 |
| `union` | Persistent binary: `Bernoulli(p_i)` with `p_i ~ Beta(2, 5)`, transition probability 0.05 |
| `married` | Persistent binary: initial `Bernoulli(0.3)`; transition probability 0.04 per wave |
| `hours` | `35 + 5 * married + N(0, 4^2)`, truncated to [10, 60] |
| `industry` | Time-invariant, drawn from `Categorical(12)` with unequal probabilities |

### Time-Invariant Covariate

| Variable | DGP |
|----------|-----|
| `education` | Drawn once: `max(8, min(20, round(N(13, 2.5^2))))` |

### Attrition

Panel attrition is generated as:

```
P(drop_{i,t}) = logit^{-1}(-4.0 + 0.02 * age_{i,t} - 0.1 * education_i)
```

Once an individual drops out, they do not return. This produces an unbalanced panel with lower-education individuals more likely to attrit, introducing mild selection bias.

## Key Statistical Properties

| Property | Detail |
|----------|--------|
| Hausman test | Should reject H0 (RE consistent) due to `Corr(alpha_i, education_i) != 0` |
| FE vs RE bias | RE overestimates the return to education (~0.125 vs true 0.065) because it conflates ability and schooling |
| Unbalanced panel | Attrition correlated with education (mild selection) |
| Within vs between variation | Education is time-invariant (only between variation); experience has both |
| Cluster structure | Observations within individual are correlated through `alpha_i` |

## Intended Tutorial Use

- Fixed effects vs. random effects estimation
- Hausman specification test (FE vs. RE)
- Mundlak/Chamberlain correlated random effects as alternative
- Demonstrating omitted variable bias when ability is unobserved
- Clustered standard errors at the individual level
- Handling unbalanced panels

## References

- Center for Human Resource Research. National Longitudinal Survey of Young Women, 1968--1988. Ohio State University.
- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251--1271.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69--85.
