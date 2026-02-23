# Penn World Table Panel -- Codebook

## Source

Synthetic macro panel inspired by **Penn World Table 10.0** (Feenstra, Inklaar & Timmer, 2015). All values are computer-generated and do not represent real country data. The dataset is designed to reproduce the stylised statistical properties of PWT variables for use in panel unit-root and cointegration tutorials.

## Panel Dimensions

| Dimension | Value |
|-----------|-------|
| Cross-sectional units (N) | 30 OECD countries |
| Time periods (T) | 1970--2019 (50 years) |
| Total observations (N x T) | 1,500 |
| Balance | Strongly balanced |

## Variable Dictionary

| Variable | Type | Description | Typical Range | Unit |
|----------|------|-------------|---------------|------|
| `countrycode` | str | ISO 3166-1 alpha-3 country code | -- | -- |
| `year` | int | Calendar year | 1970--2019 | -- |
| `rgdpna` | float | Real GDP at constant 2017 national prices | 5,000--20,000,000 | Millions USD |
| `rkna` | float | Capital stock at constant 2017 national prices | 10,000--50,000,000 | Millions USD |
| `emp` | float | Number of persons engaged (employment) | 0.5--150 | Millions |
| `labsh` | float | Share of labour compensation in GDP | 0.40--0.70 | Proportion |
| `pop` | float | Total population | 0.3--350 | Millions |
| `hc` | float | Human capital index (based on years of schooling and returns to education) | 1.0--4.0 | Index |

## Data Generating Process

### GDP (`rgdpna`)

GDP follows a unit-root process with positive drift, matching the well-known non-stationarity of real output:

```
log(rgdpna_{i,t}) = mu_i + log(rgdpna_{i,t-1}) + epsilon_{i,t}
```

- `mu_i ~ U(0.015, 0.035)` -- country-specific trend growth rate
- `epsilon_{i,t} ~ N(0, sigma_i^2)` with `sigma_i ~ U(0.01, 0.04)`
- Initial values calibrated to approximate 1970 OECD GDP levels

GDP is **I(1) with drift** by construction.

### Capital Stock (`rkna`)

Capital stock is generated as a perpetual inventory accumulation:

```
rkna_{i,t} = (1 - delta_i) * rkna_{i,t-1} + I_{i,t}
```

- `delta_i ~ U(0.03, 0.07)` -- depreciation rate
- `I_{i,t}` (investment) is a fraction of GDP: `I_{i,t} = s_i * rgdpna_{i,t}` with `s_i ~ U(0.18, 0.30)`

Capital stock is **I(1)**, cointegrated with GDP through the investment channel.

### Employment (`emp`)

Employment follows a slow-moving I(1) process:

```
log(emp_{i,t}) = log(emp_{i,t-1}) + gamma_i + nu_{i,t}
```

- `gamma_i ~ U(0.002, 0.012)` -- trend employment growth
- `nu_{i,t} ~ N(0, 0.008^2)`

### Labour Share (`labsh`)

Labour share is **stationary (I(0))**, mean-reverting around a country-specific level:

```
labsh_{i,t} = alpha_i + rho * labsh_{i,t-1} + eta_{i,t}
```

- `alpha_i ~ U(0.20, 0.30)` -- country-specific intercept
- `rho = 0.85` -- autoregressive parameter (stationary)
- `eta_{i,t} ~ N(0, 0.015^2)`
- Unconditional mean approximately `alpha_i / (1 - rho) ~ 0.55`

### Population (`pop`)

Population follows a deterministic trend with small stochastic shocks:

```
log(pop_{i,t}) = log(pop_{i,0}) + g_i * t + xi_{i,t}
```

- `g_i ~ U(0.002, 0.015)` -- population growth rate
- `xi_{i,t}` -- cumulated small AR(1) shocks

Population is **I(1)** (trend-stationary plus accumulated shocks).

### Human Capital Index (`hc`)

Human capital is a slow-moving trending variable:

```
hc_{i,t} = hc_{i,t-1} + delta_hc_i + zeta_{i,t}
```

- `delta_hc_i ~ U(0.005, 0.020)` -- trend improvement in schooling
- `zeta_{i,t} ~ N(0, 0.003^2)`
- Bounded below at 1.0

Human capital is **I(1)** with drift.

## Key Statistical Properties

| Property | Detail |
|----------|--------|
| Unit roots | `rgdpna`, `rkna`, `emp`, `pop`, `hc` are all I(1); `labsh` is I(0) |
| Cointegration | `log(rgdpna)` and `log(rkna)` are cointegrated (long-run capital-output ratio) |
| Cross-sectional dependence | Countries share a common global business-cycle factor in GDP innovations |
| Heterogeneous dynamics | Drift rates and volatilities vary across countries |
| No missing values | Panel is complete |

## Intended Tutorial Use

- Panel unit-root tests (IPS, LLC, CIPS) on `log(rgdpna)`, `labsh`
- Panel cointegration tests (Pedroni, Kao, Westerlund) between `log(rgdpna)` and `log(rkna)`
- Distinguishing I(0) variables (`labsh`) from I(1) variables
- Demonstrating cross-sectional dependence and its impact on first-generation tests

## References

- Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The next generation of the Penn World Table. *American Economic Review*, 105(10), 3150--3182.
- PWT 10.0: https://www.rug.nl/ggdc/productivity/pwt/
