# Specification Test Datasets

**Version:** 1.0.0
**Last Updated:** 2026-02-22

This directory contains datasets used in Tutorial 03 (Specification Tests). Each dataset is designed with known DGP properties so that specification tests yield predictable, pedagogically useful results.

---

## 1. nlswork.csv

**Tutorial:** 03 - Specification Tests
**Dimensions:** 4,000 individuals x 15 periods = up to 60,000 observations (unbalanced)
**Type:** Unbalanced panel (attrition present)
**Purpose:** Wage equation with correlated unobserved ability for demonstrating Hausman and Mundlak tests

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `individual_id` | Individual identifier | Integer | 1-4,000 | Entity ID |
| `period` | Time period | Integer | 1-15 | Time ID |
| `log_wage` | Log hourly wage | Log USD | 1.5-4.5 | Dependent variable |
| `experience` | Years of labor market experience | Years | 0-30 | Time-varying |
| `experience_sq` | Experience squared / 100 | Years^2 / 100 | 0-9 | Time-varying |
| `tenure` | Job tenure | Years | 0-15 | Time-varying |
| `education` | Years of schooling | Years | 8-20 | Time-invariant |
| `union` | Union membership | Binary (0/1) | 0-1 | Time-varying |
| `married` | Marital status | Binary (0/1) | 0-1 | Time-varying |
| `urban` | Urban residence | Binary (0/1) | 0-1 | Time-varying |
| `female` | Female indicator | Binary (0/1) | 0-1 | Time-invariant |
| `ability` | Unobserved ability (for verification only) | Standardized | -3-3 | Unobserved in practice |

### Key Features

- **Correlated random effects**: Unobserved ability (`ability`) is positively correlated with `education` (rho ~ 0.6) and `experience` (rho ~ 0.3). This means RE is inconsistent and the Hausman test should reject.
- **Time-invariant variables**: `education` and `female` are time-invariant, dropped by FE but estimable under RE. This illustrates the FE-RE trade-off.
- **Unbalanced panel**: Attrition is non-random (lower-wage individuals more likely to exit), providing a realistic setting for specification testing.
- **Heteroskedasticity**: Error variance is higher for less-educated workers, detectable by modified Wald test.
- **Serial correlation**: AR(1) errors with persistence rho ~ 0.4, detectable by Wooldridge test.

### Data Generating Process

```
log_wage_{i,t} = beta_0 + beta_1 * experience_{i,t} + beta_2 * experience_sq_{i,t}
                + beta_3 * tenure_{i,t} + beta_4 * education_i + beta_5 * union_{i,t}
                + alpha_i + epsilon_{i,t}
```
where `alpha_i = gamma * education_i + eta_i` (correlated with education), `eta_i ~ N(0, sigma_alpha^2)`, and `epsilon_{i,t} = rho * epsilon_{i,t-1} + v_{i,t}`.

### Expected Test Results

| Test | H0 | Expected Result |
|------|-----|-----------------|
| Hausman (FE vs. RE) | RE is consistent | Reject (ability correlated with regressors) |
| Mundlak test | No correlation between alpha_i and X | Reject (group means significant) |
| Breusch-Pagan LM | No random effects (OLS is adequate) | Reject (individual effects present) |
| Wooldridge serial correlation | No first-order serial correlation | Reject (AR(1) errors present) |
| Pesaran CD | No cross-sectional dependence | Fail to reject (individual-level data) |

---

## 2. firm_productivity.csv

**Tutorial:** 03 - Specification Tests
**Dimensions:** 200 firms x 20 years = 4,000 observations
**Type:** Balanced panel
**Purpose:** Cobb-Douglas production function for demonstrating specification tests in a classic econometric setting

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `firm_id` | Firm identifier | Integer | 1-200 | Entity ID |
| `year` | Year | Integer | 2000-2019 | Time ID |
| `log_output` | Log value added | Log million USD | 2.0-8.0 | Dependent variable |
| `log_capital` | Log capital stock | Log million USD | 1.5-7.5 | Time-varying |
| `log_labor` | Log number of employees | Log count | 2.0-7.0 | Time-varying |
| `log_materials` | Log intermediate inputs | Log million USD | 1.0-7.0 | Time-varying |
| `industry` | Industry classification | Integer (1-10) | 1-10 | Time-invariant |
| `age` | Firm age | Years | 1-50 | Time-varying |
| `rd_intensity` | R&D as share of revenue | Percentage | 0-15 | Time-varying |
| `export_share` | Export revenue share | Percentage | 0-80 | Time-varying |
| `productivity_shock` | TFP innovation (for verification) | Standardized | -3-3 | Unobserved |

### Key Features

- **Production function estimation**: Standard Cobb-Douglas with `log_output = beta_K * log_capital + beta_L * log_labor + alpha_i + epsilon_{i,t}`, where alpha_i represents persistent firm-level TFP.
- **Simultaneity concern**: Input choices are correlated with firm-level productivity (alpha_i), motivating FE estimation. The Hausman test should reject RE in favor of FE.
- **Constant returns to scale**: The true DGP has `beta_K + beta_L ~ 1.0`, testable via Wald test after estimation.
- **Heteroskedasticity**: Larger firms have more stable output, so error variance decreases with firm size.
- **No serial correlation**: Idiosyncratic errors are i.i.d., so the Wooldridge test should fail to reject.

### Data Generating Process

```
log_output_{i,t} = beta_K * log_capital_{i,t} + beta_L * log_labor_{i,t}
                  + alpha_i + epsilon_{i,t}
```
where `alpha_i ~ N(mu_industry, sigma_alpha^2)`, `Corr(alpha_i, log_capital_{i,t}) > 0`, and `epsilon_{i,t} ~ N(0, sigma_i^2)` with `sigma_i` inversely related to firm size.

### Expected Test Results

| Test | H0 | Expected Result |
|------|-----|-----------------|
| Hausman (FE vs. RE) | RE is consistent | Reject (productivity correlated with inputs) |
| Mundlak test | No correlation | Reject (capital and labor means significant) |
| Breusch-Pagan LM | No random effects | Reject (firm effects present) |
| Wooldridge serial correlation | No AR(1) | Fail to reject (i.i.d. errors) |
| Modified Wald (heteroskedasticity) | Homoskedasticity | Reject (size-dependent variance) |

---

## 3. trade_panel.csv

**Tutorial:** 03 - Specification Tests
**Dimensions:** 300 country-pairs x 15 years = 4,500 observations
**Type:** Balanced panel
**Purpose:** Gravity model for bilateral trade, demonstrating specification tests with dyadic panel data

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `pair_id` | Country-pair identifier | Integer | 1-300 | Entity ID |
| `year` | Year | Integer | 2005-2019 | Time ID |
| `exporter` | Exporting country | String (ISO-3 code) | 25 countries | Pair component |
| `importer` | Importing country | String (ISO-3 code) | 25 countries | Pair component |
| `log_trade` | Log bilateral trade value | Log million USD | 0.5-12 | Dependent variable |
| `log_gdp_exp` | Log exporter GDP | Log billion USD | 2-10 | Time-varying |
| `log_gdp_imp` | Log importer GDP | Log billion USD | 2-10 | Time-varying |
| `log_distance` | Log geographic distance | Log km | 4.5-9.5 | Time-invariant |
| `contiguous` | Share a border | Binary (0/1) | 0-1 | Time-invariant |
| `common_language` | Share official language | Binary (0/1) | 0-1 | Time-invariant |
| `fta` | Free trade agreement in force | Binary (0/1) | 0-1 | Time-varying |
| `log_reer_exp` | Log real effective exchange rate (exporter) | Log index | 4.2-5.0 | Time-varying |

### Key Features

- **Gravity model**: Classic trade gravity equation with distance, GDP, and trade facilitation variables. Pair-specific fixed effects capture bilateral resistance.
- **Time-invariant variables**: `log_distance`, `contiguous`, `common_language` are time-invariant (absorbed by pair FE), illustrating the FE limitation.
- **RE may be appropriate**: If pair effects are uncorrelated with GDPs (plausible with proper controls), RE may be consistent. The Hausman test provides the definitive check.
- **Cross-sectional dependence**: Common global trade shocks (WTO rounds, financial crises) create cross-sectional dependence across pairs, detectable by Pesaran CD test.
- **Heteroskedasticity**: Trade variance increases with distance (more uncertain for distant partners).

### Data Generating Process

```
log_trade_{i,t} = beta_1 * log_gdp_exp_{i,t} + beta_2 * log_gdp_imp_{i,t}
                + beta_3 * fta_{i,t} + mu_i + lambda_t + e_{i,t}
```
where `mu_i` captures pair-specific trade costs (related to distance, contiguity, language) and `lambda_t` captures common time effects. The error `e_{i,t}` includes a common factor generating cross-sectional dependence.

### Expected Test Results

| Test | H0 | Expected Result |
|------|-----|-----------------|
| Hausman (FE vs. RE) | RE is consistent | Depends on specification (marginal) |
| Breusch-Pagan LM | No random effects | Reject (pair effects present) |
| Wooldridge serial correlation | No AR(1) | Reject (persistent trade shocks) |
| Pesaran CD | No cross-sectional dependence | Reject (common global shocks) |
| F-test for time effects | No time effects | Reject (significant year dummies) |

---

## Data Generation

All datasets can be regenerated using:

```python
from utils.data_generators import (
    generate_nlswork,
    generate_firm_productivity,
    generate_trade_panel,
)

# NLS Work
df_nls = generate_nlswork(n_individuals=4000, n_periods=15, seed=42)
df_nls.to_csv('nlswork.csv', index=False)

# Firm Productivity
df_prod = generate_firm_productivity(n_firms=200, n_years=20, seed=42)
df_prod.to_csv('firm_productivity.csv', index=False)

# Trade Panel
df_trade = generate_trade_panel(n_pairs=300, n_years=15, seed=42)
df_trade.to_csv('trade_panel.csv', index=False)
```

All generation uses `np.random.seed(42)` for reproducibility.

---

## File Format

- **Encoding:** UTF-8
- **Separator:** Comma (`,`)
- **Decimal:** Period (`.`)
- **Header:** First row
- **Missing values:** `nlswork.csv` has missing values due to attrition; `firm_productivity.csv` and `trade_panel.csv` are complete

---

## References

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *Review of Economic Studies*, 47(1), 239-253.
- Wooldridge, J. M. (2002). Econometric analysis of cross section and panel data. MIT Press.
- Levinsohn, J., & Petrin, A. (2003). Estimating production functions using inputs to control for unobservables. *Review of Economic Studies*, 70(2), 317-341.
- Anderson, J. E., & van Wincoop, E. (2003). Gravity with gravitas: A solution to the border puzzle. *American Economic Review*, 93(1), 170-192.
