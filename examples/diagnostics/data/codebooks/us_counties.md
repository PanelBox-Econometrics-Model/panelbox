# US Counties Panel -- Codebook

## Source

Synthetic US county-level panel inspired by **Bureau of Economic Analysis (BEA)** Local Area Personal Income and **Bureau of Labor Statistics (BLS)** Local Area Unemployment Statistics. All values are computer-generated and do not represent real counties. The dataset is designed to exhibit spatial dependence through a Spatial Autoregressive (SAR) structure for use in spatial econometrics tutorials.

## Panel Dimensions

| Dimension | Value |
|-----------|-------|
| Cross-sectional units (N) | 200 counties |
| Time periods (T) | 2010--2019 (10 years) |
| Total observations (N x T) | 2,000 |
| Balance | Strongly balanced |

## Variable Dictionary

| Variable | Type | Description | Typical Range | Unit |
|----------|------|-------------|---------------|------|
| `county_id` | int | Unique county identifier | 1--200 | -- |
| `state` | str | State abbreviation | 10 states | Categorical |
| `year` | int | Calendar year | 2010--2019 | -- |
| `unemployment` | float | County unemployment rate | 2.0--15.0 | Percent |
| `log_income` | float | Log of per-capita personal income | 9.8--11.5 | Log USD |
| `log_population` | float | Log of county population | 8.5--14.0 | Log persons |
| `manufacturing_share` | float | Manufacturing employment as share of total | 0.02--0.45 | Proportion |
| `education_pct` | float | Percentage of adults with bachelor's degree or higher | 10.0--55.0 | Percent |

### State Distribution

Counties are distributed across 10 synthetic states, with 20 counties per state arranged on a spatial grid.

## Data Generating Process

### Spatial Structure

Counties are arranged on a **14 x 15 rectangular grid** (with 10 unused cells), and spatial relationships are defined by **queen contiguity** (8 nearest neighbours on the grid, fewer at boundaries):

```
W = queen_contiguity_matrix(grid_14x15)
W = row_standardize(W)
```

- Each interior county has 8 neighbours
- Edge counties have 5 neighbours; corner counties have 3
- The weight matrix `W` is row-standardised so each row sums to 1

### Unemployment (Spatial Autoregressive Model)

Unemployment follows a **SAR (Spatial Lag) model**:

```
unemployment_{i,t} = rho * sum_j(W_{ij} * unemployment_{j,t})
                   + X_{i,t} * beta + alpha_i + delta_t + epsilon_{i,t}
```

Equivalently:

```
y_t = (I - rho * W)^{-1} * (X_t * beta + alpha + delta_t * 1 + epsilon_t)
```

**Spatial autoregressive parameter:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `rho` | 0.35 | Moderate positive spatial dependence in unemployment |

**Regression coefficients:**

| Variable | Parameter | Value | Interpretation |
|----------|-----------|-------|----------------|
| `log_income` | `beta_1` | -2.50 | Higher income associated with lower unemployment |
| `manufacturing_share` | `beta_2` | 4.00 | Manufacturing-dependent counties have higher unemployment |
| `education_pct` | `beta_3` | -0.08 | Education reduces unemployment |
| `log_population` | `beta_4` | -0.20 | Larger counties have slightly lower unemployment |

**Fixed effects:**

- `alpha_i ~ N(0, 0.8^2)` -- county fixed effects (persistent local labour market conditions)
- `delta_t` -- time fixed effects (national business cycle)
  - `delta_2010 = 2.5` (post-Great Recession), declining to `delta_2019 = 0.0`

**Idiosyncratic error:**

```
epsilon_{i,t} ~ N(0, 0.6^2)
```

### Covariate DGPs

**Log per-capita income:**
```
log_income_{i,t} = mu_income_i + 0.02 * t + 0.15 * education_pct_i / 100 + nu_income_{i,t}
```
- `mu_income_i ~ N(10.3, 0.3^2)` -- county baseline income
- Trending upward at ~2% per year
- Positively correlated with education
- `nu_income_{i,t} ~ N(0, 0.04^2)`

**Log population:**
```
log_population_{i,t} = log_population_{i,0} + g_i * (t - 2010) + nu_pop_{i,t}
```
- `log_population_{i,0} ~ N(11.0, 1.2^2)` -- initial population (wide range)
- `g_i ~ N(0.005, 0.003^2)` -- county growth rate (some declining)
- `nu_pop_{i,t} ~ N(0, 0.01^2)`

**Manufacturing share:**
```
manufacturing_share_{i,t} = max(0.02, ms_i_0 - 0.005 * (t - 2010) + nu_ms_{i,t})
```
- `ms_i_0 ~ Beta(2, 5) * 0.5` -- initial manufacturing share
- Secular decline of ~0.5 percentage points per year
- `nu_ms_{i,t} ~ N(0, 0.01^2)`
- Truncated below at 0.02

**Education percentage:**
```
education_pct_{i,t} = ed_i_0 + 0.3 * (t - 2010) + nu_ed_{i,t}
```
- `ed_i_0 ~ N(28, 8^2)`, truncated to [10, 50]
- Slowly increasing (~0.3 pp per year)
- `nu_ed_{i,t} ~ N(0, 0.5^2)`

## Spatial Weight Matrix Properties

| Property | Value |
|----------|-------|
| Type | Queen contiguity on rectangular grid |
| Standardisation | Row-standardised |
| Average number of neighbours | ~7.2 |
| Min / Max neighbours | 3 (corners) / 8 (interior) |
| Largest eigenvalue of W | 1.0 (by row-standardisation) |
| Sparsity | ~96% zeros |

## Key Statistical Properties

| Property | Detail |
|----------|--------|
| Spatial lag dependence | `rho = 0.35`; Moran's I on OLS residuals should be significantly positive |
| SAR vs SEM | True DGP is SAR (spatial lag), not SEM (spatial error) |
| OLS bias | OLS ignoring spatial lag produces biased and inconsistent coefficient estimates |
| Endogeneity | Spatial lag `W*y` is endogenous (correlated with `epsilon` through simultaneous determination) |
| County fixed effects | Correlated with manufacturing share and education (compositional differences) |
| Time effects | Capture national unemployment trends (Great Recession recovery) |

## Intended Tutorial Use

- Moran's I test for spatial autocorrelation in OLS residuals
- Spatial lag model (SAR) estimation via maximum likelihood or IV/2SLS
- LM tests for spatial lag vs. spatial error specification
- Comparing OLS, SAR, and SEM estimates
- Direct vs. indirect (spillover) effect decomposition
- Spatial weight matrix construction and sensitivity analysis

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
- LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
