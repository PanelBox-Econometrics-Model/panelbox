# OECD Macro Panel -- Codebook

## Source

Synthetic macroeconomic panel inspired by **OECD National Accounts** data. All values are computer-generated and do not represent real country data. The dataset is designed to exhibit consumption-income cointegration and weak investment cointegration for use in panel cointegration tutorials.

## Panel Dimensions

| Dimension | Value |
|-----------|-------|
| Cross-sectional units (N) | 20 OECD countries |
| Time periods (T) | 1980--2019 (40 years) |
| Total observations (N x T) | 800 |
| Balance | Strongly balanced |

## Variable Dictionary

| Variable | Type | Description | Typical Range | Unit |
|----------|------|-------------|---------------|------|
| `country` | str | Country name | -- | -- |
| `year` | int | Calendar year | 1980--2019 | -- |
| `consumption` | float | Real private consumption expenditure | 50,000--15,000,000 | Millions USD (2017 prices) |
| `income` | float | Real GDP (used as income proxy) | 80,000--22,000,000 | Millions USD (2017 prices) |
| `investment` | float | Real gross fixed capital formation | 15,000--5,000,000 | Millions USD (2017 prices) |
| `log_C` | float | Natural log of consumption | 10.8--16.5 | Log millions USD |
| `log_Y` | float | Natural log of income | 11.3--16.9 | Log millions USD |

## Data Generating Process

### Income (`income`, `log_Y`)

Income follows a unit-root process with country-specific drift:

```
log_Y_{i,t} = mu_i + log_Y_{i,t-1} + epsilon_{i,t}
```

- `mu_i ~ U(0.015, 0.030)` -- country-specific trend growth
- `epsilon_{i,t} = phi * f_t + sqrt(1-phi^2) * e_{i,t}` with `phi = 0.4`
- `f_t ~ N(0, 0.015^2)` -- common global factor
- `e_{i,t} ~ N(0, sigma_i^2)` with `sigma_i ~ U(0.010, 0.025)`

Income is **I(1)** with cross-sectional dependence through the common factor `f_t`.

### Consumption (`consumption`, `log_C`)

Consumption is cointegrated with income through a long-run consumption function:

```
log_C_{i,t} = a_i + beta_i * log_Y_{i,t} + u_{i,t}
u_{i,t} = rho_u * u_{i,t-1} + v_{i,t}
```

- `a_i ~ U(-0.25, -0.05)` -- country-specific intercept (log average propensity)
- `beta_i ~ U(0.82, 0.88)` -- long-run marginal propensity to consume (MPC ~ 0.85)
- `rho_u = 0.60` -- error-correction speed (stationary equilibrium error)
- `v_{i,t} ~ N(0, 0.008^2)`

The cointegrating vector is approximately `[1, -0.85]` in `(log_C, log_Y)` space. The equilibrium error `u_{i,t}` is **I(0)** by construction, so `log_C` and `log_Y` are **cointegrated**.

### Investment (`investment`)

Investment is related to income but with a weaker cointegrating relationship:

```
log_I_{i,t} = c_i + gamma_i * log_Y_{i,t} + w_{i,t}
w_{i,t} = rho_w * w_{i,t-1} + xi_{i,t}
```

- `c_i ~ U(-1.5, -0.5)` -- country-specific intercept
- `gamma_i ~ U(0.90, 1.10)` -- long-run investment elasticity
- `rho_w = 0.92` -- near-unit-root error correction (weak cointegration)
- `xi_{i,t} ~ N(0, 0.025^2)`

Investment-income cointegration is **weak**: the equilibrium error is stationary but highly persistent (`rho_w = 0.92`), making it harder to detect with standard panel cointegration tests at moderate T.

## Key Statistical Properties

| Property | Detail |
|----------|--------|
| Unit roots | `log_Y`, `log_C`, `log(investment)` are all I(1) |
| Strong cointegration | `log_C` and `log_Y` with MPC ~ 0.85; equilibrium error has `rho = 0.60` |
| Weak cointegration | `log(investment)` and `log_Y`; equilibrium error has `rho = 0.92` |
| Cross-sectional dependence | Common factor in income innovations (`phi = 0.4`) transmits to consumption |
| Heterogeneous cointegrating vectors | `beta_i` varies across countries (0.82--0.88) |
| No missing values | Panel is complete |

## Intended Tutorial Use

- Panel cointegration tests (Pedroni, Kao, Westerlund) between `log_C` and `log_Y`
- Comparing test power: strong cointegration (consumption-income) vs. weak cointegration (investment-income)
- Estimating cointegrating vectors (FMOLS, DOLS) and recovering MPC
- Demonstrating the effect of cross-sectional dependence on cointegration test size
- Error correction model estimation

## References

- Campbell, J. Y., & Mankiw, N. G. (1989). Consumption, income, and interest rates: Reinterpreting the time series evidence. *NBER Macroeconomics Annual*, 4, 185--216.
- Pedroni, P. (2004). Panel cointegration: Asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis. *Econometric Theory*, 20(3), 597--625.
