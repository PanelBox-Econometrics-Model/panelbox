# Datasets for Censored and Selection Models

This directory contains all datasets used in the censored and selection models tutorial series.

## Dataset Overview

| File | Observations | Type | Used In |
|------|-------------|------|---------|
| `labor_supply.csv` | 500 | Cross-section | 01, 07 |
| `health_expenditure_panel.csv` | 2,000 (N=500, T=4) | Panel | 02, 08 |
| `consumer_durables_panel.csv` | 1,000 (N=200, T=5) | Panel | 03 |
| `mroz_1987.csv` | 753 | Cross-section | 04, 05, 06 |
| `college_wage.csv` | 600 | Cross-section | 06, 07 |

## Dataset Descriptions

### 1. `labor_supply.csv` - Labor Supply Data

Simulated cross-sectional data on labor supply with left-censoring at zero hours.

| Variable | Type | Description |
|----------|------|-------------|
| `hours` | float | Weekly hours worked (censored at 0) |
| `wage` | float | Hourly wage rate |
| `education` | int | Years of education |
| `experience` | float | Years of work experience |
| `experience_sq` | float | Experience squared |
| `age` | int | Age in years |
| `children` | int | Number of children under 6 |
| `married` | int | Marital status (0/1) |
| `non_labor_income` | float | Non-labor household income (thousands) |

**Censoring:** Left-censored at `hours = 0` (non-participants)
**Selection rate:** ~65% with positive hours

### 2. `health_expenditure_panel.csv` - Health Expenditure Panel

Simulated panel data on individual health expenditures with left-censoring.

| Variable | Type | Description |
|----------|------|-------------|
| `id` | int | Individual identifier |
| `time` | int | Time period (1-4) |
| `expenditure` | float | Health expenditure (censored at 0) |
| `income` | float | Annual income (thousands) |
| `age` | int | Age in years |
| `chronic` | int | Number of chronic conditions |
| `insurance` | int | Has health insurance (0/1) |
| `female` | int | Female indicator (0/1) |
| `bmi` | float | Body mass index |

**Censoring:** Left-censored at `expenditure = 0`
**Panel structure:** N=500 individuals, T=4 periods

### 3. `consumer_durables_panel.csv` - Consumer Durables Spending

Simulated panel data on household durable goods purchases.

| Variable | Type | Description |
|----------|------|-------------|
| `id` | int | Household identifier |
| `time` | int | Time period (1-5) |
| `spending` | float | Durable goods spending (censored at 0) |
| `income` | float | Household income (thousands) |
| `wealth` | float | Net household wealth (thousands) |
| `household_size` | int | Number of household members |
| `homeowner` | int | Homeownership indicator (0/1) |
| `urban` | int | Urban residence indicator (0/1) |
| `credit_score` | float | Credit score (standardized) |

**Censoring:** Left-censored at `spending = 0`
**Panel structure:** N=200 households, T=5 periods

### 4. `mroz_1987.csv` - Mroz (1987) Labor Force Participation

Simulated data based on Mroz (1987) labor force participation study.

| Variable | Type | Description |
|----------|------|-------------|
| `lfp` | int | Labor force participation (0/1) |
| `hours` | float | Annual hours worked |
| `wage` | float | Hourly wage (observed if lfp=1) |
| `education` | int | Years of education |
| `experience` | float | Years of work experience |
| `experience_sq` | float | Experience squared |
| `age` | int | Age in years |
| `children_lt6` | int | Number of children under 6 |
| `children_6_18` | int | Number of children 6-18 |
| `husband_income` | float | Husband's income (thousands) |

**Selection:** `lfp = 1` indicates observed wages
**Selection rate:** ~57% participate in labor force

### 5. `college_wage.csv` - College Graduate Wages

Simulated data on college attendance decisions and subsequent wages.

| Variable | Type | Description |
|----------|------|-------------|
| `college` | int | Attended college (0/1) |
| `wage` | float | Log hourly wage (observed if college=1) |
| `ability` | float | Standardized ability measure |
| `parent_education` | int | Average parental education (years) |
| `family_income` | float | Family income during adolescence (thousands) |
| `distance_college` | float | Distance to nearest college (miles) |
| `tuition` | float | Average local tuition (thousands) |
| `urban` | int | Urban origin indicator (0/1) |
| `female` | int | Female indicator (0/1) |

**Selection:** `college = 1` indicates observed post-college wages
**Note:** `distance_college` and `tuition` serve as exclusion restrictions

## Data Generation

All datasets are generated synthetically using functions in `../utils/data_generation.py`. To regenerate:

```python
from utils.data_generation import (
    generate_labor_supply,
    generate_health_panel,
    generate_consumer_durables,
    generate_mroz_data,
    generate_college_wage,
)

# Generate with default seed for reproducibility
labor = generate_labor_supply(n=500, seed=42)
health = generate_health_panel(n=500, t=4, seed=42)
durables = generate_consumer_durables(n=200, t=5, seed=42)
mroz = generate_mroz_data(n=753, seed=42)
college = generate_college_wage(n=600, seed=42)
```

## Summary Statistics

Summary statistics are available in each tutorial notebook where the dataset is first introduced. Use `df.describe()` for quick summaries.
