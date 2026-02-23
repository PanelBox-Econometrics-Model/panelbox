# Diagnostics Tutorial Datasets

**Version:** 1.0.0
**Last Updated:** 2026-02-22

This directory contains all datasets used in the panel data diagnostics tutorials. All data are simulated based on real-world stylized facts from published research.

---

## Overview

All datasets are designed to:
- Demonstrate specific econometric diagnostic concepts
- Exhibit realistic data features (unit roots, cointegration, spatial dependence)
- Be computationally manageable for tutorial purposes
- Be fully reproducible via `utils/data_generators.py` with `np.random.seed(42)`

**Important:** These are simulated data created for pedagogical purposes. They are based on stylized facts from real studies but should not be used for actual empirical research.

---

## Dataset Catalog by Subdirectory

### Unit Root (`unit_root/`)

| Dataset | Dimensions | Type | Purpose | Used in |
|---------|-----------|------|---------|---------|
| `penn_world_table.csv` | 30 countries x 50 years | Balanced panel | GDP, investment, consumption with I(1) behavior | Notebook 01 |
| `prices_panel.csv` | 40 regions x 30 years | Balanced panel | Regional price indices with I(1) dynamics | Notebook 01 |

### Cointegration (`cointegration/`)

| Dataset | Dimensions | Type | Purpose | Used in |
|---------|-----------|------|---------|---------|
| `oecd_macro.csv` | 20 countries x 40 years | Balanced panel | Consumption-income cointegrating relationship | Notebook 02 |
| `ppp_data.csv` | 25 countries x 35 years | Balanced panel | Purchasing power parity long-run equilibrium | Notebook 02 |
| `interest_rates.csv` | 15 countries x 30 years | Balanced panel | Interest rate parity across countries | Notebook 02 |

### Specification (`specification/`)

| Dataset | Dimensions | Type | Purpose | Used in |
|---------|-----------|------|---------|---------|
| `nlswork.csv` | 4,000 individuals x 15 periods | Unbalanced panel | Wage equation with correlated unobserved ability | Notebook 03 |
| `firm_productivity.csv` | 200 firms x 20 years | Balanced panel | Cobb-Douglas production function estimation | Notebook 03 |
| `trade_panel.csv` | 300 country-pairs x 15 years | Balanced panel | Gravity model for bilateral trade | Notebook 03 |

### Spatial (`spatial/`)

| Dataset | Dimensions | Type | Purpose | Used in |
|---------|-----------|------|---------|---------|
| `us_counties.csv` | 500 counties x 10 years | Balanced panel | County-level economic indicators | Notebook 04 |
| `W_counties.npy` | 500 x 500 | Weight matrix | Queen contiguity weights for US counties | Notebook 04 |
| `W_counties_distance.npy` | 500 x 500 | Weight matrix | Inverse-distance weights for US counties | Notebook 04 |
| `eu_regions.csv` | 200 regions x 15 years | Balanced panel | EU NUTS-2 regional economic data | Notebook 04 |
| `W_eu_contiguity.npy` | 200 x 200 | Weight matrix | Contiguity weights for EU regions | Notebook 04 |
| `coordinates_counties.csv` | 500 x 2 | Cross-section | County centroid coordinates (lat, lon) | Notebook 04 |
| `coordinates_eu.csv` | 200 x 2 | Cross-section | EU region centroid coordinates (lat, lon) | Notebook 04 |

---

## Variable Descriptions Overview

### Common Identifiers

All panel datasets include:
- **Entity identifier**: `country`, `region`, `individual_id`, `firm_id`, or `pair_id`
- **Time identifier**: `year` or `period`

### Economic Variables (across datasets)

- GDP, income, consumption, investment (levels and logs)
- Price indices and exchange rates
- Wages, hours worked, experience
- Output, capital, labor inputs
- Bilateral trade flows, distance, trade agreements

### Spatial Variables

- Weight matrices (`.npy` format): row-standardized, symmetric or asymmetric
- Coordinate files: centroid latitude and longitude for distance computation

See individual subdirectory READMEs for complete variable listings.

---

## Data Generation

### Reproducibility

All datasets are generated using fixed random seeds for full reproducibility:

```python
from utils.data_generators import (
    generate_penn_world_table,
    generate_prices_panel,
    generate_oecd_macro,
    generate_ppp_data,
    generate_interest_rates,
    generate_nlswork,
    generate_firm_productivity,
    generate_trade_panel,
    generate_us_counties,
    generate_eu_regions,
)

# All generators use np.random.seed(42) internally
df = generate_penn_world_table()
df.to_csv('unit_root/penn_world_table.csv', index=False)
```

### Methodology

Data generation follows these principles:

1. **Unit root datasets**: Variables are generated as random walks with drift (I(1) processes) using cumulative sums of stationary innovations, with cross-sectional heterogeneity in drift and variance parameters.

2. **Cointegration datasets**: Non-stationary variables are generated jointly so that specific linear combinations are stationary (I(0)), producing genuine cointegrating relationships with heterogeneous adjustment speeds.

3. **Specification datasets**: Panel data with known DGP properties (correlated random effects, heteroskedasticity, serial correlation) so that diagnostic tests yield known correct answers.

4. **Spatial datasets**: Variables are generated with spatial autoregressive processes using known weight matrices, ensuring detectable spatial dependence patterns.

### Seed Convention

All data generation uses `np.random.seed(42)` as the base seed. Individual datasets may use derived seeds for independent streams, but the master seed ensures full reproducibility.

---

## File Formats

### CSV Specifications
- **Encoding:** UTF-8
- **Separator:** Comma (`,`)
- **Decimal:** Period (`.`)
- **Header:** Yes (first row)
- **Missing values:** None (all datasets are complete)

### NumPy Array Specifications (`.npy`)
- **Format:** NumPy binary format (saved with `np.save()`)
- **Loading:** `W = np.load('W_counties.npy')`
- **Content:** Row-standardized spatial weight matrices

---

## Data Quality

### No Missing Values

All datasets are complete (no NaN values) by design:

```python
import pandas as pd

df = pd.read_csv('unit_root/penn_world_table.csv')
assert df.isnull().sum().sum() == 0
print("No missing values")
```

### Verification

```python
# Verify panel structure
n_entities = df['country'].nunique()
n_periods = df['year'].nunique()
print(f"Entities: {n_entities}, Periods: {n_periods}, Obs: {len(df)}")
```

---

## Size Information

| Subdirectory | Total Files | Approx. Total Size |
|-------------|-------------|-------------------|
| unit_root/ | 2 CSV | ~500 KB |
| cointegration/ | 3 CSV | ~600 KB |
| specification/ | 3 CSV | ~800 KB |
| spatial/ | 2 CSV + 3 NPY + 2 CSV | ~1.5 MB |
| **Total** | **15 files** | **~3.4 MB** |

---

## Citation and Usage

### Academic Use

If you use these datasets in presentations or teaching:

> "Data simulated based on stylized facts from [relevant paper]. Generated using PanelBox Diagnostics Tutorials (2026)."

### Restrictions

Do NOT:
- Present as real empirical data
- Use for actual policy analysis
- Publish as original research data

---

For detailed documentation of individual datasets, see the README in each subdirectory.
