# Count Models Tutorial Datasets

**Version:** 1.0.0
**Last Updated:** 2026-02-16

This directory contains all datasets used in the count models tutorials. All data are simulated based on real-world stylized facts from published research.

---

## Overview

All datasets are designed to:
- Demonstrate specific econometric concepts
- Exhibit realistic data features
- Be computationally manageable
- Be fully reproducible (via `utils/data_generators.py`)

**Important:** These are simulated data created for pedagogical purposes. They are based on stylized facts from real studies but should not be used for actual empirical research.

---

## Dataset Catalog

### 1. healthcare_visits.csv

**Tutorial:** 01 - Poisson Introduction
**Observations:** 2,000
**Type:** Cross-sectional
**Purpose:** Introduction to basic Poisson regression

**Variables:**
- `individual_id`: Unique identifier (1-2000)
- `visits`: Doctor visits in a year (count, 0-25)
- `age`: Age in years (18-85)
- `income`: Annual income in $1,000s (10-150)
- `insurance`: Health insurance (0=No, 1=Yes)
- `chronic`: Chronic condition (0=No, 1=Yes)

**Key Features:**
- Mean visits: 4.2
- Variance: 5.5 (mild overdispersion, Var/Mean ≈ 1.3)
- Zero prevalence: 15%
- Suitable for Poisson (equidispersion approximately holds)

**Codebook:** `codebooks/healthcare_visits_codebook.txt`

---

### 2. firm_patents.csv

**Tutorial:** 02 - Negative Binomial
**Observations:** 1,500 firms × 5 years = 7,500
**Type:** Balanced panel
**Purpose:** Demonstrate overdispersion and Negative Binomial models

**Variables:**
- `firm_id`: Firm identifier (1-1500)
- `year`: Year (2015-2019)
- `patents`: Number of patents filed (count, 0-45)
- `rd_expenditure`: R&D spending in millions (0.1-100)
- `firm_size`: Log(employees) (2-9)
- `industry`: Industry code (1-10)
- `region`: Geographic region (1-5)

**Key Features:**
- Mean patents: 2.8
- Variance: 12.4 (severe overdispersion, Var/Mean ≈ 4.4)
- Zero prevalence: 42%
- Right-skewed distribution
- Requires Negative Binomial specification

**Codebook:** `codebooks/firm_patents_codebook.txt`

---

### 3. city_crime.csv

**Tutorial:** 03 - Fixed and Random Effects
**Observations:** 150 cities × 10 years = 1,500
**Type:** Balanced panel
**Purpose:** Panel count models with FE and RE

**Variables:**
- `city_id`: City identifier (1-150)
- `year`: Year (2010-2019)
- `crime_count`: Violent crimes (count, 10-500)
- `unemployment_rate`: Unemployment % (3-15)
- `police_per_capita`: Police per 1,000 residents (1-5)
- `median_income`: Median household income in $1,000s (30-120)
- `population`: Population in 100,000s (0.5-50)
- `temperature`: Average temperature in °F (40-85)

**Key Features:**
- Substantial within-city variation over time
- Time-invariant city characteristics (size, region) matter
- Suitable for FE vs RE comparison
- Hausman test typically favors FE

**Codebook:** `codebooks/city_crime_codebook.txt`

---

### 4. bilateral_trade.csv

**Tutorial:** 04 - PPML and Gravity Models
**Observations:** 50 countries × 50 partners × 15 years = 37,500
**Type:** Unbalanced panel (N=10,000 after filtering)
**Purpose:** PPML for gravity equations with zeros

**Variables:**
- `exporter`: Exporting country code (ISO)
- `importer`: Importing country code (ISO)
- `year`: Year (2005-2019)
- `trade_value`: Export value in millions USD (0-50,000)
- `distance`: Geographic distance in km (100-20,000)
- `contiguous`: Share border (0/1)
- `common_language`: Common official language (0/1)
- `gdp_exporter`: Exporter GDP in billions (10-20,000)
- `gdp_importer`: Importer GDP in billions (10-20,000)
- `trade_agreement`: FTA in force (0/1)

**Key Features:**
- Zero prevalence: 23% (zero trade flows)
- Demonstrates Santos Silva & Tenreyro (2006) PPML
- High-dimensional fixed effects (exporter-year, importer-year)
- Heteroskedasticity typical of trade data

**Codebook:** `codebooks/bilateral_trade_codebook.txt`

---

### 5. healthcare_zinb.csv

**Tutorial:** 05 - Zero-Inflated Models
**Observations:** 3,000
**Type:** Cross-sectional
**Purpose:** Zero-inflated Poisson and ZINB models

**Variables:**
- `individual_id`: Unique identifier (1-3000)
- `visits`: Doctor visits (count, 0-30)
- `age`: Age in years (18-85)
- `income`: Annual income in $1,000s (10-150)
- `insurance`: Health insurance (0/1)
- `chronic`: Chronic condition (0/1)
- `rural`: Rural residence (0/1)
- `health_literacy`: Health literacy score (1-10)

**Key Features:**
- Zero prevalence: 60% (excess zeros)
- Two processes:
  - Structural zeros (never users due to barriers)
  - Count zeros (potential users with zero visits)
- Vuong test strongly favors ZIP/ZINB over standard models

**Codebook:** `codebooks/healthcare_zinb_codebook.txt`

---

### 6. policy_impact.csv

**Tutorial:** 06 - Marginal Effects
**Observations:** 1,200
**Type:** Cross-sectional
**Purpose:** Computing and interpreting marginal effects

**Variables:**
- `individual_id`: Unique identifier (1-1200)
- `outcome_count`: Outcome of interest (count, 0-20)
- `treatment`: Treatment indicator (0/1)
- `age`: Age in years (25-65)
- `education`: Years of education (8-20)
- `income`: Income in $1,000s (15-200)
- `female`: Female (0/1)
- `urban`: Urban residence (0/1)

**Key Features:**
- Clear treatment effect (AME ≈ 1.8 events)
- Heterogeneous effects by education and income
- Demonstrates AME, MEM, MER calculations
- Good for counterfactual policy simulations

**Codebook:** `codebooks/policy_impact_codebook.txt`

---

### 7. firm_innovation_full.csv

**Tutorial:** 07 - Innovation Case Study
**Observations:** 500 firms × 8 years = 4,000
**Type:** Balanced panel
**Purpose:** Comprehensive real-world application

**Variables:**
- `firm_id`: Firm identifier (1-500)
- `year`: Year (2012-2019)
- `patents`: Patents filed (count, 0-35)
- `rd_intensity`: R&D as % of sales (0-25)
- `firm_size`: Log(employees)
- `firm_age`: Years since founding (1-50)
- `industry`: Industry code (1-8)
- `export_share`: % sales from exports (0-100)
- `capital_intensity`: Capital per employee ($1,000s)
- `hhi`: Market concentration index (0-1)
- `subsidy`: R&D subsidy recipient (0/1)

**Key Features:**
- Rich set of controls
- Multiple model comparisons needed
- Zero prevalence: 35%
- Overdispersion present
- Policy-relevant application (subsidy impact)

**Codebook:** `codebooks/firm_innovation_full_codebook.txt`

---

## Data Generation

### Reproducibility

All datasets can be regenerated using:

```python
from utils.data_generators import (
    generate_healthcare_data,
    generate_patent_data,
    generate_crime_data,
    generate_trade_data,
    generate_zinb_healthcare_data,
    generate_policy_impact_data,
    generate_innovation_data
)

# Example: Regenerate healthcare data
df = generate_healthcare_data(n=2000, seed=42)
df.to_csv('healthcare_visits.csv', index=False)
```

### Seeds

All data generation uses fixed seeds for reproducibility:
- Healthcare visits: `seed=42`
- Firm patents: `seed=123`
- City crime: `seed=456`
- Bilateral trade: `seed=789`
- Healthcare ZINB: `seed=101`
- Policy impact: `seed=202`
- Firm innovation: `seed=303`

### Customization

You can create variations by modifying parameters:

```python
# Create larger sample
df_large = generate_healthcare_data(n=5000, seed=42)

# Different overdispersion
df_more_dispersed = generate_patent_data(
    n_firms=1500,
    n_years=5,
    alpha=0.8,  # Higher overdispersion
    seed=123
)
```

---

## Data Quality

### No Missing Values

All datasets are complete (no NaN values) unless explicitly noted for pedagogical purposes.

Verify:
```python
import pandas as pd

df = pd.read_csv('healthcare_visits.csv')
assert df.isnull().sum().sum() == 0
print("✓ No missing values")
```

### Data Types

Recommended dtypes for efficient loading:

```python
# Example for healthcare_visits.csv
dtype_dict = {
    'individual_id': 'int32',
    'visits': 'int32',
    'age': 'int16',
    'income': 'float32',
    'insurance': 'int8',
    'chronic': 'int8'
}

df = pd.read_csv('healthcare_visits.csv', dtype=dtype_dict)
```

See individual codebooks for complete dtype specifications.

---

## File Formats

### CSV Specifications
- **Encoding:** UTF-8
- **Separator:** Comma (`,`)
- **Decimal:** Period (`.`)
- **Line terminator:** `\n`
- **Header:** Yes (first row)
- **Quoting:** Minimal (only when necessary)

### Size Information

| Dataset | Rows | Columns | File Size | Memory Usage |
|---------|------|---------|-----------|--------------|
| healthcare_visits.csv | 2,000 | 6 | ~70 KB | ~94 KB |
| firm_patents.csv | 7,500 | 7 | ~450 KB | ~577 KB |
| city_crime.csv | 1,500 | 8 | ~110 KB | ~141 KB |
| bilateral_trade.csv | 10,000 | 10 | ~800 KB | ~1.0 MB |
| healthcare_zinb.csv | 3,000 | 8 | ~140 KB | ~180 KB |
| policy_impact.csv | 1,200 | 8 | ~80 KB | ~102 KB |
| firm_innovation_full.csv | 4,000 | 11 | ~320 KB | ~410 KB |

Total: ~1.97 MB uncompressed

---

## Citation and Usage

### Academic Use

If you use these datasets in presentations or teaching:

> "Data simulated based on stylized facts from [relevant paper]. Generated using PanelBox Count Models Tutorials (2026)."

### Modification

You are free to:
- Modify data generation parameters
- Create derivative datasets
- Use in your own teaching materials

Please acknowledge the source.

### Restrictions

Do NOT:
- Present as real empirical data
- Use for actual policy analysis
- Publish as original research data

---

## Codebooks

Each dataset has a detailed codebook in `codebooks/` containing:
- Variable descriptions
- Units and scales
- Summary statistics
- Data generation methodology
- Suggested analyses

Always read the codebook before analysis!

---

## Troubleshooting

### Data Won't Load

```python
# Check file exists
from pathlib import Path
data_file = Path('healthcare_visits.csv')
assert data_file.exists(), f"File not found: {data_file}"

# Check working directory
import os
print(f"Current directory: {os.getcwd()}")
```

### Encoding Issues

```python
# Specify encoding explicitly
df = pd.read_csv('healthcare_visits.csv', encoding='utf-8')
```

### Memory Issues (Large Datasets)

```python
# Load with optimized dtypes
df = pd.read_csv(
    'bilateral_trade.csv',
    dtype={
        'exporter': 'category',
        'importer': 'category',
        'year': 'int16',
        'trade_value': 'float32',
        # ... specify all columns
    }
)
```

### Regenerating Corrupted Data

```python
from utils.data_generators import generate_healthcare_data

# Regenerate from scratch
df = generate_healthcare_data(n=2000, seed=42)
df.to_csv('healthcare_visits.csv', index=False)
print("✓ Data regenerated successfully")
```

---

## Version History

**v1.0.0 (2026-02-16)**
- Initial release
- 7 datasets covering all tutorials
- Complete codebooks
- Reproducible generation scripts

---

For questions about data, consult:
1. Relevant codebook in `codebooks/`
2. Data generation code in `../utils/data_generators.py`
3. Tutorial notebooks for usage examples
