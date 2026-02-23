# Unit Root Test Datasets

**Version:** 1.0.0
**Last Updated:** 2026-02-22

This directory contains datasets used in Tutorial 01 (Unit Root Tests). Both datasets feature variables with I(1) behavior suitable for demonstrating panel unit root testing procedures.

---

## 1. penn_world_table.csv

**Tutorial:** 01 - Unit Root Tests
**Dimensions:** 30 countries x 50 years = 1,500 observations
**Type:** Balanced panel
**Purpose:** Macroeconomic variables with unit root properties for demonstrating LLC, IPS, Breitung, Hadri, and CIPS tests

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `country` | Country identifier | String (ISO-3 code) | 30 countries | Entity ID |
| `year` | Year | Integer | 1970-2019 | Time ID |
| `log_gdp` | Log real GDP per capita | Log USD (constant 2017) | 6.5-11.5 | I(1), trending |
| `log_investment` | Log gross capital formation share | Log percentage of GDP | 2.0-4.0 | I(1) |
| `log_consumption` | Log household consumption share | Log percentage of GDP | 3.5-4.5 | I(1) |
| `population_growth` | Population growth rate | Percentage | -0.5-3.5 | I(0), stationary |
| `trade_openness` | (Exports + Imports) / GDP | Ratio | 0.1-4.0 | I(1), bounded |
| `government_share` | Government consumption share of GDP | Percentage | 5-35 | Near I(1) |
| `inflation` | CPI inflation rate | Percentage | -2-50 | I(0), stationary |
| `tfp_growth` | Total factor productivity growth | Percentage | -5-8 | I(0), stationary |

### Key Features

- **GDP per capita** (`log_gdp`): Strong I(1) process with country-specific trends. Standard unit root tests (LLC, IPS) should fail to reject the unit root null at conventional levels. First differences are stationary.
- **Investment and consumption shares**: I(1) with weaker trends than GDP. Useful for contrasting test power across variables.
- **Population growth and inflation**: Stationary (I(0)) variables included for comparison. Tests should reject the unit root null for these variables.
- **Cross-sectional dependence**: Common global shocks (oil crises, financial crises) induce cross-sectional dependence, motivating second-generation tests (CIPS).

### Data Generating Process

GDP for country i at time t follows:
```
log_gdp_{i,t} = alpha_i + delta_i * t + log_gdp_{i,t-1} + epsilon_{i,t}
```
where `alpha_i` captures country-specific intercepts, `delta_i` allows heterogeneous trends, and `epsilon_{i,t} = gamma * f_t + u_{i,t}` includes a common factor `f_t` generating cross-sectional dependence.

### Expected Test Results

| Test | H0 | Variable: log_gdp | Variable: inflation |
|------|-----|-------------------|---------------------|
| LLC | Common unit root | Fail to reject | Reject |
| IPS | Individual unit roots | Fail to reject | Reject |
| Breitung | Common unit root | Fail to reject | Reject |
| Hadri | Stationarity | Reject | Fail to reject |
| CIPS | Unit root (CD robust) | Fail to reject | Reject |

---

## 2. prices_panel.csv

**Tutorial:** 01 - Unit Root Tests
**Dimensions:** 40 regions x 30 years = 1,200 observations
**Type:** Balanced panel
**Purpose:** Regional price data with I(1) properties for practicing unit root tests and understanding PPP implications

### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `region` | Region identifier | String | 40 regions | Entity ID |
| `year` | Year | Integer | 1990-2019 | Time ID |
| `log_cpi` | Log consumer price index | Log index (base=100 in 1990) | 4.6-5.8 | I(1), trending |
| `log_food_price` | Log food price index | Log index | 4.5-5.9 | I(1) |
| `log_housing_price` | Log housing price index | Log index | 4.4-6.5 | I(1), heterogeneous trends |
| `log_energy_price` | Log energy price index | Log index | 4.3-6.0 | I(1), volatile |
| `relative_price` | Relative price (region vs. national avg) | Ratio | 0.7-1.4 | Near I(0) if PPP holds |

### Key Features

- **CPI** (`log_cpi`): I(1) process with region-specific inflation rates. Demonstrates unit root in levels, stationarity in first differences.
- **Sectoral prices**: Food, housing, and energy price indices are I(1) with different persistence and volatility patterns, useful for comparing test power across variables.
- **Relative prices** (`relative_price`): If purchasing power parity holds within the panel, relative prices should be I(0) (mean-reverting). This provides a natural test case for stationarity.
- **Heterogeneous dynamics**: Regions have different inflation rates and adjustment speeds, demonstrating the importance of tests that allow heterogeneous autoregressive parameters (IPS vs. LLC).

### Data Generating Process

CPI for region i at time t follows:
```
log_cpi_{i,t} = mu_i + log_cpi_{i,t-1} + v_{i,t}
```
where `mu_i ~ U(0.01, 0.05)` represents region-specific inflation and `v_{i,t} ~ N(0, sigma_i^2)` with heterogeneous variance.

### Expected Test Results

| Test | H0 | Variable: log_cpi | Variable: relative_price |
|------|-----|-------------------|--------------------------|
| LLC | Common unit root | Fail to reject | Reject (weak) |
| IPS | Individual unit roots | Fail to reject | Reject |
| Hadri | Stationarity | Reject | Fail to reject |

---

## Data Generation

Both datasets can be regenerated using:

```python
from utils.data_generators import generate_penn_world_table, generate_prices_panel

# Penn World Table
df_pwt = generate_penn_world_table(n_countries=30, n_years=50, seed=42)
df_pwt.to_csv('penn_world_table.csv', index=False)

# Prices panel
df_prices = generate_prices_panel(n_regions=40, n_years=30, seed=42)
df_prices.to_csv('prices_panel.csv', index=False)
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

- Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). Unit root tests in panel data. *Journal of Econometrics*, 108(1), 1-24.
- Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in heterogeneous panels. *Journal of Econometrics*, 115(1), 53-74.
- Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. *Journal of Applied Econometrics*, 22(2), 265-312.
