# Quantile Regression Tutorial Datasets

## Overview

All datasets in this directory are **synthetic** (simulated), designed to exhibit
specific properties relevant to quantile regression analysis. Each dataset is
generated with `np.random.seed(42)` for full reproducibility.

## Main Datasets

### `card_education.csv` - Education and Wage Panel
- **Inspired by**: Card (1995) education-wage studies
- **Dimensions**: 500 individuals x 6 years = 3,000 observations
- **Key feature**: Heterogeneous returns to education across the wage distribution
  (education effect increases at higher quantiles) and glass ceiling effect
  (gender gap widens at top quantiles)
- **Used in**: Notebooks 01, 02, 06, 07
- **Variables**: id, year, lwage, educ, exper, black, south, married, female,
  union, hours, age

### `firm_production.csv` - Firm Production Panel
- **Inspired by**: Cobb-Douglas production function literature
- **Dimensions**: 500 firms x 10 years = 5,000 observations
- **Key feature**: Heterogeneous firm-level productivity fixed effects;
  some firms have quantile-varying FE (location shift assumption violated)
- **Used in**: Notebooks 03, 04, 10
- **Variables**: firm_id, year, log_output, log_capital, log_labor,
  log_materials, profit, size, sector, exporter

### `financial_returns.csv` - Financial Returns Panel
- **Inspired by**: Fama-French factor models
- **Dimensions**: 200 firms x 60 months = 12,000 observations
- **Key feature**: Fat-tailed returns (t-distribution), size-dependent
  volatility, asymmetric firm-specific risk
- **Used in**: Notebooks 04, 05
- **Variables**: firm_id, month, returns, size, book_to_market, momentum,
  volatility, sector

### `labor_program.csv` - Labor Training Program Evaluation
- **Inspired by**: LaLonde (1986) job training evaluation
- **Dimensions**: 1,000 individuals x 2 periods = 2,000 observations
- **Key feature**: Progressive treatment effects (program helps low earners
  more than high earners; ATE hides distributional heterogeneity)
- **Used in**: Notebook 09
- **Variables**: id, period, treatment, earnings, education, age,
  experience, female

## Simulated Datasets (`simulated/`)

### `crossing_example.csv` - Quantile Crossing Data
- **Purpose**: Demonstrate quantile crossing problem
- **Dimensions**: 300 units x 8 periods = 2,400 observations
- **DGP**: Heteroskedasticity depends on x1; linear model is misspecified
- **Variables**: id, t, y, x1, x2

### `location_shift.csv` - Location Shift Test Data
- **Purpose**: Verify Canay (2011) location shift assumption
- **Dimensions**: 400 units x 10 periods = 4,000 observations
- **DGP**: Two groups -- one satisfies location shift, one violates it
- **Variables**: id, t, y, x1, x2, group

### `heteroskedastic.csv` - Heteroskedastic Panel
- **Purpose**: Motivate location-scale models
- **Dimensions**: 500 units x 8 periods = 4,000 observations
- **DGP**: Explicit location-scale structure with known parameters
- **Variables**: id, t, y, x1, x2, x3

### `treatment_effects.csv` - DiD Treatment Effects
- **Purpose**: Contrast with labor_program (regressive instead of progressive)
- **Dimensions**: 800 individuals x 2 periods = 1,600 observations
- **DGP**: Treatment effects increase with quantile (regressive program)
- **Variables**: id, period, treatment, earnings, education, age, female

## Regenerating Datasets

All datasets can be regenerated using the utility functions in `../utils/simulation_helpers.py`:

```python
from utils.simulation_helpers import generate_card_education
df = generate_card_education(n_individuals=500, n_years=6, seed=42)
df.to_csv('data/card_education.csv', index=False)
```
