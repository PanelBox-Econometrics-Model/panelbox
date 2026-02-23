# Cointegration Test Datasets

**Version:** 1.0.0
**Last Updated:** 2026-02-22

This directory contains datasets used in Tutorial 02 (Cointegration Tests). All three datasets feature I(1) variables with known long-run equilibrium relationships, suitable for demonstrating Pedroni, Kao, and Westerlund panel cointegration tests.

---

## 1. oecd_macro.csv

**Tutorial:** 02 - Cointegration Tests
**Dimensions:** 20 countries x 40 years = 800 observations
**Type:** Balanced panel
**Purpose:** Consumption-income cointegrating relationship (permanent income hypothesis)

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `country` | OECD country identifier | String (ISO-3 code) | 20 countries | Entity ID |
| `year` | Year | Integer | 1980-2019 | Time ID |
| `log_consumption` | Log real private consumption per capita | Log USD (constant 2015) | 8.5-10.8 | I(1) |
| `log_income` | Log real GDP per capita | Log USD (constant 2015) | 8.8-11.2 | I(1) |
| `log_wealth` | Log household net wealth per capita | Log USD (constant 2015) | 9.0-12.5 | I(1) |
| `interest_rate` | Real long-term interest rate | Percentage | -2-8 | I(0) |
| `unemployment` | Unemployment rate | Percentage | 2-25 | I(0) |
| `government_debt` | Government debt as share of GDP | Percentage | 15-250 | I(1) |

### Key Features

- **Consumption-income cointegration**: `log_consumption` and `log_income` share a common stochastic trend (cointegrated). The cointegrating vector is approximately (1, -beta) where beta is near 1 (long-run consumption function).
- **Heterogeneous cointegrating vectors**: The long-run propensity to consume (`beta_i`) varies across countries (0.85-1.05), motivating tests that allow heterogeneity (Pedroni).
- **Error correction dynamics**: Deviations from long-run equilibrium are corrected at country-specific speeds (alpha_i ranges from -0.05 to -0.30 per year).
- **Additional I(1) variable**: `log_wealth` provides a second cointegrating relationship opportunity, useful for multivariate extensions.

### Cointegrating Relationship

The true DGP satisfies:
```
log_consumption_{i,t} = a_i + beta_i * log_income_{i,t} + e_{i,t}
```
where `e_{i,t}` is I(0) (stationary equilibrium error) and `beta_i ~ U(0.85, 1.05)`.

### Expected Test Results

| Test | H0 | Consumption-Income | Consumption-Wealth |
|------|-----|--------------------|--------------------|
| Pedroni (panel v) | No cointegration | Reject | Reject |
| Pedroni (group ADF) | No cointegration | Reject | Reject |
| Kao (ADF) | No cointegration | Reject | Reject |
| Westerlund (Gt, Ga) | No cointegration | Reject | Reject |

---

## 2. ppp_data.csv

**Tutorial:** 02 - Cointegration Tests
**Dimensions:** 25 countries x 35 years = 875 observations
**Type:** Balanced panel
**Purpose:** Purchasing power parity (PPP) as a cointegrating relationship

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `country` | Country identifier | String (ISO-3 code) | 25 countries | Entity ID |
| `year` | Year | Integer | 1985-2019 | Time ID |
| `log_exchange_rate` | Log nominal exchange rate (vs. USD) | Log (local currency / USD) | -2.0-8.0 | I(1) |
| `log_price_domestic` | Log domestic price level (CPI) | Log index | 3.5-6.5 | I(1) |
| `log_price_foreign` | Log US price level (CPI) | Log index | 4.0-5.2 | I(1) |
| `log_real_exchange_rate` | Log real exchange rate | Log index | -1.5-2.0 | Near I(0) if PPP holds |
| `trade_balance` | Trade balance as share of GDP | Percentage | -15-20 | I(0) |

### Key Features

- **PPP relationship**: Under absolute PPP, `log_exchange_rate = log_price_domestic - log_price_foreign`. The three I(1) variables are cointegrated with known theoretical vector (1, -1, 1).
- **Slow adjustment**: PPP deviations are persistent (half-life of 3-5 years), consistent with empirical PPP literature. This creates realistic challenges for cointegration tests.
- **Real exchange rate**: `log_real_exchange_rate` = `log_exchange_rate - log_price_domestic + log_price_foreign` should be I(0) if PPP holds, providing a direct stationarity check.
- **Heterogeneous adjustment**: Different countries adjust at different speeds toward PPP, with faster adjustment for open economies.

### Cointegrating Relationship

The PPP condition implies:
```
log_exchange_rate_{i,t} - log_price_domestic_{i,t} + log_price_foreign_{i,t} = mu_i + z_{i,t}
```
where `z_{i,t}` is I(0) with AR(1) coefficient between 0.85 and 0.95 (slow mean reversion).

### Expected Test Results

| Test | H0 | PPP Relationship |
|------|-----|-----------------|
| Pedroni (panel ADF) | No cointegration | Reject (at 5%) |
| Kao (ADF) | No cointegration | Reject (marginal) |
| Westerlund (Pt, Pa) | No cointegration | Reject |

Note: Results may be marginal due to slow adjustment speed, consistent with the PPP puzzle in empirical literature.

---

## 3. interest_rates.csv

**Tutorial:** 02 - Cointegration Tests
**Dimensions:** 15 countries x 30 years = 450 observations
**Type:** Balanced panel
**Purpose:** Interest rate parity (IRP) and international interest rate linkages

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `country` | Country identifier | String (ISO-3 code) | 15 countries | Entity ID |
| `year` | Year | Integer | 1990-2019 | Time ID |
| `long_rate` | Domestic long-term government bond yield | Percentage | 0.5-15 | I(1) |
| `short_rate` | Domestic short-term (3-month) interest rate | Percentage | 0.0-12 | I(1) |
| `us_long_rate` | US 10-year Treasury yield | Percentage | 1.5-8 | I(1) |
| `inflation_diff` | Inflation differential (domestic - US) | Percentage points | -5-10 | I(0) |
| `risk_premium` | Sovereign risk premium (spread over US) | Percentage points | 0-8 | Near I(0) |

### Key Features

- **Term structure cointegration**: `long_rate` and `short_rate` within each country are cointegrated (expectations hypothesis of the term structure). The spread is I(0).
- **International interest rate linkages**: `long_rate` and `us_long_rate` are cointegrated (financial integration), with the spread reflecting country risk and expected exchange rate changes.
- **Smaller panel**: N=15, T=30 provides a setting where asymptotic approximations may be less reliable, useful for discussing finite-sample properties of cointegration tests.
- **Structural breaks**: The global financial crisis (2008-2009) and zero-lower-bound era introduce potential structural breaks, motivating robust testing approaches.

### Cointegrating Relationships

**Term structure (within country):**
```
long_rate_{i,t} = c_i + beta_i * short_rate_{i,t} + e_{i,t}
```
where `e_{i,t}` is I(0) and `beta_i` is near 1.

**International linkage:**
```
long_rate_{i,t} = d_i + gamma_i * us_long_rate_t + v_{i,t}
```
where `v_{i,t}` is I(0) and `gamma_i` reflects financial integration (closer to 1 for more integrated economies).

### Expected Test Results

| Test | H0 | Term Structure | International |
|------|-----|---------------|---------------|
| Pedroni (group ADF) | No cointegration | Reject | Reject |
| Kao (ADF) | No cointegration | Reject | Reject (weak) |
| Westerlund (Gt) | No cointegration | Reject | Reject |

---

## Data Generation

All datasets can be regenerated using:

```python
from utils.data_generators import (
    generate_oecd_macro,
    generate_ppp_data,
    generate_interest_rates,
)

# OECD Macro
df_oecd = generate_oecd_macro(n_countries=20, n_years=40, seed=42)
df_oecd.to_csv('oecd_macro.csv', index=False)

# PPP Data
df_ppp = generate_ppp_data(n_countries=25, n_years=35, seed=42)
df_ppp.to_csv('ppp_data.csv', index=False)

# Interest Rates
df_ir = generate_interest_rates(n_countries=15, n_years=30, seed=42)
df_ir.to_csv('interest_rates.csv', index=False)
```

All generation uses `np.random.seed(42)` for reproducibility.

---

## File Format

- **Encoding:** UTF-8
- **Separator:** Comma (`,`)
- **Decimal:** Period (`.`)
- **Header:** First row
- **Missing values:** None

---

## References

- Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous panels with multiple regressors. *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.
- Pedroni, P. (2004). Panel cointegration: Asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis. *Econometric Theory*, 20(3), 597-625.
- Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. *Journal of Econometrics*, 90(1), 1-44.
- Westerlund, J. (2007). Testing for error correction in panel data. *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.
- Rogoff, K. (1996). The purchasing power parity puzzle. *Journal of Economic Literature*, 34(2), 647-668.
