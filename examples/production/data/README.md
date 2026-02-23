# Production & Deployment Datasets

Datasets for the Production & Deployment tutorial series.

## Overview

| Dataset | N | T | Rows | Purpose | Used in |
|---------|---|---|------|---------|---------|
| `firm_panel.csv` | 100 firms | 20 years | 2,000 | Static model predict | 01, 02 |
| `bank_lgd.csv` | 200 contracts | 15 months | 3,000 | GMM predict & forecast | 01, 03, 06 |
| `macro_quarterly.csv` | 30 countries | 40 quarters | 1,200 | Multi-step forecasting | 01, 03, 05 |
| `new_firms.csv` | 20 firms | 5 years | 100 | Out-of-sample prediction | 01, 02 |
| `new_bank_data.csv` | 50 contracts | 3 months | 150 | Production prediction | 03, 06 |
| `future_macro.csv` | 30 countries | 4 quarters | 120 | Future exogenous for forecast | 03, 05 |

## Dataset Details

### `firm_panel.csv`

Firm-level panel for demonstrating `predict(newdata)` with Static models (PooledOLS, FE, RE).

| Variable | Type | Description |
|----------|------|-------------|
| `firm_id` | int | Firm identifier (1-100) |
| `year` | int | Year (2000-2019) |
| `investment` | float | Capital investment (log) |
| `value` | float | Market value (log) |
| `capital` | float | Capital stock (log) |
| `sales` | float | Sales revenue (log) |
| `sector` | str | Industry sector (5 unique) |

**DGP**: `investment = alpha_i + 0.3*value + 0.2*capital + 0.15*sales + epsilon`

### `bank_lgd.csv`

Banking LGD dynamic panel for GMM predict and forecast.

| Variable | Type | Description |
|----------|------|-------------|
| `contract_id` | int | Contract identifier (1-200) |
| `month` | int | Month period (1-15) |
| `lgd_logit` | float | LGD in logit scale |
| `saldo_real` | float | Outstanding balance (log R$) |
| `pib_growth` | float | GDP growth (%) |
| `selic` | float | Interest rate (%) |
| `collateral_ratio` | float | Collateral/balance ratio |

**DGP**: `lgd_logit_t = 0.6*lgd_logit_{t-1} + 0.1*saldo_real + 0.05*pib_growth - 0.03*collateral + fe_i + e_it`

### `macro_quarterly.csv`

Macro quarterly panel for multi-step forecasting with exogenous variables.

| Variable | Type | Description |
|----------|------|-------------|
| `country` | str | Country name (30 unique) |
| `quarter` | int | Quarter index (1-40) |
| `gdp_growth` | float | GDP growth (%) |
| `inflation` | float | Inflation (%) |
| `interest_rate` | float | Policy rate (%) |

**DGP**: Panel VAR(1) with country fixed effects.

### `new_firms.csv`

Out-of-sample firm data (10 known firms + 10 new firms) for testing predictions on unseen data.

### `new_bank_data.csv`

New bank observations (50 contracts x 3 months) for production prediction testing.

### `future_macro.csv`

Future exogenous variables only (no `gdp_growth`) for forecast demonstrations.

## Reproducibility

All datasets are generated with `seed=42` using the functions in `utils/data_generators.py`. To regenerate:

```python
from utils.data_generators import generate_firm_panel
df = generate_firm_panel(seed=42)
df.to_csv('data/firm_panel.csv', index=False)
```
