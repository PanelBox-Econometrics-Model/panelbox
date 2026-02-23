# EU NUTS2 Regional Panel -- Codebook

## Source

Synthetic European regional panel inspired by **Eurostat Regional Statistics** at the NUTS2 level. All values are computer-generated and do not represent real regions. The dataset is designed to exhibit spatial error dependence through a Spatial Error Model (SEM) structure for use in spatial econometrics tutorials.

## Panel Dimensions

| Dimension | Value |
|-----------|-------|
| Cross-sectional units (N) | 100 NUTS2 regions |
| Time periods (T) | 2005--2019 (15 years) |
| Total observations (N x T) | 1,500 |
| Balance | Strongly balanced |

## Variable Dictionary

| Variable | Type | Description | Typical Range | Unit |
|----------|------|-------------|---------------|------|
| `region_id` | str | NUTS2 region code (synthetic) | e.g., "DE11", "FR42" | -- |
| `country` | str | Country code | 10 EU member states | Categorical |
| `year` | int | Calendar year | 2005--2019 | -- |
| `gdp_per_capita` | float | Regional GDP per capita | 8,000--80,000 | EUR (2015 prices) |
| `log_gdp_pc` | float | Natural log of GDP per capita | 9.0--11.3 | Log EUR |
| `fdi` | float | Foreign direct investment inflows | 0--5,000 | Millions EUR |
| `rd_expenditure` | float | R&D expenditure (GERD) as percentage of regional GDP | 0.2--5.0 | Percent |
| `infrastructure` | float | Infrastructure quality index | 1.0--10.0 | Index |

### Country Distribution

| Country Code | N Regions | Approximate Inspiration |
|-------------|-----------|------------------------|
| DE | 15 | Germany |
| FR | 15 | France |
| IT | 12 | Italy |
| ES | 12 | Spain |
| PL | 10 | Poland |
| NL | 8 | Netherlands |
| SE | 8 | Sweden |
| CZ | 8 | Czech Republic |
| PT | 6 | Portugal |
| RO | 6 | Romania |

## Data Generating Process

### GDP Per Capita (Spatial Error Model)

The log of GDP per capita follows a **Spatial Error Model (SEM)**:

```
log_gdp_pc_{i,t} = X_{i,t} * beta + alpha_i + delta_t + u_{i,t}
u_{i,t} = lambda * sum_j(W_{ij} * u_{j,t}) + epsilon_{i,t}
```

Equivalently, the error is:

```
u_t = (I - lambda * W)^{-1} * epsilon_t
```

**Spatial error parameter:**

| Parameter | Value | Interpretation |
|-----------|-------|----------------|
| `lambda` | 0.40 | Moderate positive spatial error dependence |

**Regression coefficients:**

| Variable | Parameter | Value | Interpretation |
|----------|-----------|-------|----------------|
| `fdi` (log-transformed) | `beta_1` | 0.025 | FDI inflows raise regional GDP per capita |
| `rd_expenditure` | `beta_2` | 0.060 | R&D intensity boosts productivity |
| `infrastructure` | `beta_3` | 0.040 | Infrastructure quality supports growth |

**Fixed effects:**

- `alpha_i` -- region fixed effects capturing time-invariant characteristics (institutional quality, geography, historical development):
  - Core regions (DE, NL, SE): `alpha_i ~ N(0.3, 0.15^2)`
  - Intermediate (FR, IT-North, ES-Northeast, CZ): `alpha_i ~ N(0.0, 0.15^2)`
  - Periphery (PL, RO, PT, IT-South, ES-South): `alpha_i ~ N(-0.3, 0.15^2)`

- `delta_t` -- time fixed effects:
  - Steady growth 2005--2007 (`delta ~ +0.02/year`)
  - Financial crisis 2008--2009 (`delta_2009 = -0.05`)
  - Recovery 2010--2019 (`delta ~ +0.015/year`)

**Idiosyncratic error:**

```
epsilon_{i,t} ~ N(0, 0.06^2)
```

The spatial error structure means that unobserved shocks (e.g., weather, policy changes) are spatially correlated across neighbouring regions, but the explanatory variables themselves do not have spatial lag effects.

### Spatial Weight Matrix

Regions are arranged on a **10 x 10 grid** with queen contiguity, modified to reflect country borders:

```
W_raw = queen_contiguity(grid_10x10)
W_border = W_raw * (1 - 0.5 * cross_border_{ij})
W = row_standardize(W_border)
```

- Within-country neighbours receive full weight
- Cross-border neighbours receive half weight (reflecting national borders as partial barriers)
- Row-standardised so each row sums to 1

### Covariate DGPs

**FDI (Foreign Direct Investment):**
```
fdi_{i,t} = exp(mu_fdi_i + 0.03 * (t - 2005) + 0.2 * infrastructure_{i,t} + nu_fdi_{i,t})
```
- `mu_fdi_i ~ N(5.0, 1.5^2)` -- region-specific FDI attractiveness
- Core regions attract more FDI
- `nu_fdi_{i,t} ~ N(0, 0.4^2)` -- high volatility (FDI is lumpy)
- Some region-years may have very low FDI (near zero)

**R&D Expenditure (% of GDP):**
```
rd_expenditure_{i,t} = rd_i_0 + 0.05 * (t - 2005) + nu_rd_{i,t}
```
- `rd_i_0` -- country-dependent:
  - High R&D (SE, DE, NL): `~ N(2.5, 0.5^2)`
  - Medium R&D (FR, CZ): `~ N(1.5, 0.4^2)`
  - Low R&D (PL, RO, PT, ES, IT): `~ N(0.8, 0.3^2)`
- Slowly increasing over time (~0.05 pp/year)
- `nu_rd_{i,t} ~ N(0, 0.1^2)`, truncated to [0.2, 5.0]

**Infrastructure Index:**
```
infrastructure_{i,t} = infra_i_0 + 0.1 * (t - 2005) + nu_infra_{i,t}
```
- `infra_i_0` -- country-dependent:
  - High infrastructure (DE, NL, SE): `~ N(7.5, 0.8^2)`
  - Medium (FR, IT-North, ES, CZ): `~ N(5.5, 0.8^2)`
  - Low (PL, RO, PT): `~ N(3.5, 0.8^2)`
- Convergence regions improve faster (cohesion policy effect)
- `nu_infra_{i,t} ~ N(0, 0.2^2)`, truncated to [1.0, 10.0]

## Spatial Weight Matrix Properties

| Property | Value |
|----------|-------|
| Type | Queen contiguity on 10x10 grid, border-adjusted |
| Standardisation | Row-standardised |
| Average number of neighbours | ~7.0 |
| Cross-border discount | 50% weight reduction |
| Largest eigenvalue of W | 1.0 (by row-standardisation) |
| Sparsity | ~93% zeros |

## Key Statistical Properties

| Property | Detail |
|----------|--------|
| Spatial error dependence | `lambda = 0.40`; Moran's I on OLS residuals should be positive and significant |
| SEM vs SAR | True DGP is **SEM** (spatial error), not SAR (spatial lag) |
| OLS consistency | OLS coefficients are consistent but **inefficient** under SEM; standard errors are biased |
| LM test discrimination | Robust LM-Error should be significant; Robust LM-Lag should not (identifying SEM over SAR) |
| Regional convergence | Core-periphery structure with convergence dynamics |
| Country clustering | Regions within the same country share similar levels of R&D and infrastructure |

## Intended Tutorial Use

- Moran's I test for spatial autocorrelation in regression residuals
- Spatial Error Model (SEM) estimation via maximum likelihood
- LM and robust LM tests to discriminate between SAR and SEM specifications
- Comparing OLS (consistent but inefficient) with SEM (efficient) under spatial error dependence
- Demonstrating that OLS standard errors are unreliable under spatial error dependence
- EU regional convergence analysis with spatial effects

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.
- Eurostat. Regional Statistics. https://ec.europa.eu/eurostat/web/regions
- Dall'Erba, S., & Le Gallo, J. (2008). Regional convergence and the impact of European structural funds, 1989--1999. *Papers in Regional Science*, 87(2), 219--244.
