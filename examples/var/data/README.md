# VAR Tutorial Datasets

## Overview

| Dataset | File | N | T | Variables | Used in |
|---------|------|---|---|-----------|---------|
| Macro Panel | `macro_panel.csv` | 30 countries | 40 quarters | gdp_growth, inflation, interest_rate, unemployment, exchange_rate | 01, 02, 03, 04, 06, 07 |
| Energy Panel | `energy_panel.csv` | 25 countries | 60 quarters | oil_price, gas_price, electricity_price | 02 |
| Finance Panel | `finance_panel.csv` | 50 countries | 100 periods | stock_return, bond_return, fx_return, commodity_return | 03 |
| Monetary Policy | `monetary_policy.csv` | 25 OECD | 80 quarters | gdp_growth, inflation, interest_rate, unemployment | 07 |

Additional datasets for notebooks 04-06 are generated inline using `utils/data_generators.py`.

## Generation

All datasets are synthetic, generated using `utils/data_generators.py` with fixed random seeds for reproducibility.

To regenerate:

```python
import sys
sys.path.insert(0, '../utils')
from data_generators import (
    generate_macro_panel,
    generate_energy_panel,
    generate_finance_panel,
    generate_monetary_policy_panel,
)

generate_macro_panel().to_csv('macro_panel.csv', index=False)
generate_energy_panel().to_csv('energy_panel.csv', index=False)
generate_finance_panel().to_csv('finance_panel.csv', index=False)
generate_monetary_policy_panel().to_csv('monetary_policy.csv', index=False)
```

## Format

- All CSV files are in **long format** with entity and time identifier columns.
- Entity column: `country` (string)
- Time column: `quarter` (string "YYYY-QN") or `time` (integer)
- No missing values in any dataset.

## Codebooks

Detailed variable descriptions are available in the `codebooks/` subdirectory:

- `codebooks/macro_panel_codebook.md`
- `codebooks/energy_panel_codebook.md`
- `codebooks/finance_panel_codebook.md`
- `codebooks/monetary_policy_codebook.md`

## Reproduction

All datasets use `seed=42` by default. To reproduce exact values:

```python
df = generate_macro_panel(n_countries=30, n_quarters=40, seed=42)
```
