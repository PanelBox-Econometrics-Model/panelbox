# Wage Panel Dataset Codebook

## Overview
- **Source**: Subset of NLSY (National Longitudinal Survey of Youth) / PSID (Panel Study of Income Dynamics)
- **Sample**: 500 individuals, 7 years, 3500 total observations
- **Time Range**: 1980 - 1986
- **Entity Type**: Young workers (ages 18-25 in 1980)
- **Panel Type**: Balanced panel

## Description

This dataset contains wage and demographic information for a sample of young workers tracked over 7 years. It is commonly used to study wage dynamics, returns to education, and experience effects in labor economics.

## Variables

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| person_id | int | Individual identifier | - |
| year | int | Year of observation | - |
| wage | float | Hourly wage | Dollars per hour (nominal) |
| hours | float | Annual hours worked | Hours |
| experience | float | Years of work experience | Years |
| education | int | Years of schooling completed | Years |
| female | int | Gender indicator (1=female, 0=male) | Binary |
| married | int | Marital status (1=married, 0=single) | Binary |
| union | int | Union membership (1=member, 0=non-member) | Binary |
| industry | str | Industry code (2-digit) | Categorical |

## Missing Data

Minimal missing data (< 1% of observations). Missing values primarily in:
- `hours`: 15 observations (~0.4%)
- `union`: 23 observations (~0.7%)

Missing values are coded as `NaN` in the dataset.

## Typical Usage

Common specifications include:

**Wage equation**:
```
log(wage_it) = β₀ + β₁ experience_it + β₂ education_i + β₃ female_i + α_i + ε_it
```

**Returns to experience**:
```
log(wage_it) = β₀ + β₁ experience_it + β₂ experience²_it + α_i + ε_it
```

## References

- Card, D. (1999). The Causal Effect of Education on Earnings. *Handbook of Labor Economics*, 3, 1801-1863.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Data Quality Notes

- Wages are in nominal dollars (not adjusted for inflation)
- Experience is calculated as (age - education - 6)
- Sample is restricted to individuals with consistent employment
- Industry codes follow SIC (Standard Industrial Classification) system

## Citation

Please cite the original survey source when using this data:
```
National Longitudinal Survey of Youth (NLSY79). Bureau of Labor Statistics,
U.S. Department of Labor.
```
