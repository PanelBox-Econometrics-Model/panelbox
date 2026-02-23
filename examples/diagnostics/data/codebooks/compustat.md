# Compustat Firm Productivity Panel -- Codebook

## Source

Synthetic firm-level productivity panel inspired by **Standard & Poor's Compustat North America** database. All values are computer-generated and do not represent real firms. The dataset is designed for production function estimation tutorials, with a known Cobb-Douglas DGP that allows students to benchmark their estimates.

## Panel Dimensions

| Dimension | Value |
|-----------|-------|
| Cross-sectional units (N) | 200 firms |
| Time periods (T) | 2000--2019 (20 years) |
| Total observations (N x T) | 4,000 |
| Balance | Strongly balanced |

## Variable Dictionary

| Variable | Type | Description | Typical Range | Unit |
|----------|------|-------------|---------------|------|
| `firm_id` | int | Unique firm identifier | 1--200 | -- |
| `year` | int | Fiscal year | 2000--2019 | -- |
| `log_output` | float | Log of real gross output (revenue deflated) | 2.0--9.0 | Log millions USD |
| `log_capital` | float | Log of real capital stock (net PPE deflated) | 1.0--8.0 | Log millions USD |
| `log_labor` | float | Log of employment (number of employees) | 1.0--7.0 | Log thousands |
| `log_materials` | float | Log of real materials expenditure (COGS less depreciation) | 1.5--8.5 | Log millions USD |
| `rd_intensity` | float | R&D expenditure as fraction of revenue | 0.00--0.15 | Proportion |
| `sector` | str | Industry sector classification | 5 categories | Categorical |
| `exporter` | int | Firm exports indicator | 0 or 1 | Binary |

### Sector Categories

| Code | Sector | N firms | Typical characteristics |
|------|--------|---------|------------------------|
| `manufacturing` | Manufacturing | 60 | High capital, high materials |
| `technology` | Technology | 40 | High R&D, high labour productivity |
| `services` | Services | 40 | Low capital, high labour share |
| `energy` | Energy | 30 | Very high capital intensity |
| `retail` | Retail | 30 | High materials, low capital |

## Data Generating Process

### Production Function (Cobb-Douglas)

Output is generated from a log-linear Cobb-Douglas production function:

```
log_output_{i,t} = omega_{i,t} + beta_K * log_capital_{i,t}
                 + beta_L * log_labor_{i,t}
                 + beta_M * log_materials_{i,t}
                 + beta_R * rd_intensity_{i,t}
                 + epsilon_{i,t}
```

**True parameter values:**

| Parameter | Symbol | Value | Interpretation |
|-----------|--------|-------|----------------|
| Capital elasticity | `beta_K` | 0.30 | Output elasticity of capital |
| Labour elasticity | `beta_L` | 0.35 | Output elasticity of labour |
| Materials elasticity | `beta_M` | 0.25 | Output elasticity of materials |
| R&D effect | `beta_R` | 0.05 | Productivity effect of R&D intensity |
| Returns to scale | `beta_K + beta_L + beta_M` | 0.90 | Slightly decreasing returns (excluding R&D) |

### Total Factor Productivity (`omega_{i,t}`)

TFP follows a persistent AR(1) process with firm heterogeneity:

```
omega_{i,t} = mu_i + rho_omega * omega_{i,t-1} + xi_{i,t}
```

- `mu_i` -- sector-specific mean productivity:
  - Manufacturing: 0.0, Technology: 0.3, Services: -0.1, Energy: 0.1, Retail: -0.2
- `rho_omega = 0.80` -- persistence of productivity shocks
- `xi_{i,t} ~ N(0, 0.10^2)` -- productivity innovation
- Exporters receive a productivity premium: `mu_i += 0.15` if `exporter = 1`

### Measurement Error

```
epsilon_{i,t} ~ N(0, 0.05^2)
```

Small i.i.d. measurement error in output (revenue measurement noise).

### Input Factor DGPs

**Capital** (slow-moving state variable):
```
log_capital_{i,t} = log_capital_{i,t-1} + delta_K_i + 0.3 * xi_{i,t-1} + nu_K_{i,t}
```
- `delta_K_i ~ U(0.02, 0.06)` -- net investment rate
- Capital responds to lagged productivity shocks (simultaneity source)
- `nu_K_{i,t} ~ N(0, 0.03^2)`

**Labour** (flexible input):
```
log_labor_{i,t} = a_L_i + 0.5 * omega_{i,t} + 0.2 * log_capital_{i,t} + nu_L_{i,t}
```
- Labour is chosen **after observing current productivity** -- creates simultaneity bias
- `a_L_i` -- firm-specific labour demand intercept
- `nu_L_{i,t} ~ N(0, 0.08^2)`

**Materials** (fully flexible):
```
log_materials_{i,t} = a_M_i + 0.6 * omega_{i,t} + 0.15 * log_capital_{i,t} + nu_M_{i,t}
```
- Materials respond most strongly to current productivity (proxy variable candidate)
- `a_M_i` -- firm-specific materials demand intercept
- `nu_M_{i,t} ~ N(0, 0.06^2)`

**R&D intensity:**
```
rd_intensity_{i,t} = r_i + 0.1 * omega_{i,t-1} + nu_R_{i,t}
```
- `r_i ~ sector-specific U(...)`: Technology 0.08, Manufacturing 0.03, others 0.01
- R&D responds to lagged productivity (less endogenous than labour/materials)
- `nu_R_{i,t} ~ N(0, 0.01^2)`, truncated to [0, 0.15]

### Exporter Status

```
exporter_i ~ Bernoulli(p_s)
```
- Sector-dependent: Technology 0.6, Manufacturing 0.5, Energy 0.4, Services 0.2, Retail 0.1
- Time-invariant for simplicity
- Exporters have higher initial TFP (self-selection into exporting)

## Key Statistical Properties

| Property | Detail |
|----------|--------|
| Simultaneity bias | OLS overestimates `beta_L` and `beta_M` because labour and materials respond to `omega_{i,t}` |
| Capital timing | Capital is predetermined (responds to `omega_{i,t-1}`), less biased in OLS |
| Proxy variable | `log_materials` is a valid proxy for `omega_{i,t}` (Levinsohn-Petrin approach) |
| Firm fixed effects | `mu_i` creates unobserved heterogeneity correlated with input choices |
| Returns to scale | True RTS = 0.90 (slightly decreasing); OLS will overestimate due to simultaneity |
| Exporter premium | Exporters are ~15% more productive (selection, not treatment) |

## Intended Tutorial Use

- OLS production function estimation and demonstrating simultaneity bias
- Fixed effects estimation to control for time-invariant heterogeneity
- Olley-Pakes / Levinsohn-Petrin / Ackerberg-Caves-Frazer proxy-variable methods
- Testing for constant returns to scale
- Exporter productivity premia analysis
- Hausman test comparing FE vs RE with firm-level heterogeneity

## References

- Ackerberg, D. A., Caves, K., & Frazer, G. (2015). Identification properties of recent production function estimators. *Econometrica*, 83(6), 2411--2451.
- Levinsohn, J., & Petrin, A. (2003). Estimating production functions using inputs to control for unobservables. *Review of Economic Studies*, 70(2), 317--341.
