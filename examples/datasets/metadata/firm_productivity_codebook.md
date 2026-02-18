# Firm Productivity Dataset Codebook

## Overview
- **Source**: Synthetic data based on manufacturing census patterns
- **Sample**: 300 firms, 12 years, 3200 total observations (unbalanced)
- **Time Range**: 2008 - 2019
- **Entity Type**: Manufacturing firms
- **Panel Type**: Unbalanced panel

## Description

This dataset contains productivity and input measures for manufacturing firms. Unlike other datasets in this collection, this is an **unbalanced panel** - some firms enter or exit during the observation period. This makes it suitable for teaching handling of unbalanced panels and attrition issues.

## Variables

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| firm_id | int | Firm identifier (1-300) | - |
| year | int | Year of observation | - |
| output | float | Gross output/revenue | Thousands of dollars |
| labor | float | Number of employees | Persons |
| capital | float | Capital stock | Thousands of dollars |
| materials | float | Intermediate inputs | Thousands of dollars |
| age | int | Firm age | Years |
| size_category | str | Size classification (small/medium/large) | Categorical |
| industry_2digit | int | 2-digit industry code (NAICS) | Categorical |
| export_status | int | Exports indicator (1=yes, 0=no) | Binary |
| tfp | float | Total factor productivity (estimated) | Index |

## Panel Structure

- **Balanced firms**: 180 firms (observed all 12 years) = 2,160 observations
- **Entry**: 60 firms entered during period = ~420 observations
- **Exit**: 40 firms exited before 2019 = ~320 observations
- **Entry & Exit**: 20 firms both entered and exited = ~300 observations
- **Total**: 300 unique firms, 3,200 total observations

## Missing Data

Minimal missing data in core variables (output, labor, capital). Materials and TFP have slightly higher missingness (~5%) due to reporting issues for small firms.

## Typical Usage

**Production function estimation**:
```
log(output_it) = β₀ + β_l log(labor_it) + β_k log(capital_it) + β_m log(materials_it) + α_i + ε_it
```

**TFP determinants**:
```
tfp_it = β₀ + β₁ age_it + β₂ export_status_it + β₃ size_it + α_i + ε_it
```

**Handling attrition**:
Useful for demonstrating selection models and unbalanced panel techniques.

## Size Categories

- **Small**: labor < 50 employees (35% of observations)
- **Medium**: 50 ≤ labor < 250 employees (45% of observations)
- **Large**: labor ≥ 250 employees (20% of observations)

## Industry Distribution

Covers manufacturing industries (NAICS 31-33):
- Food manufacturing (311): 15%
- Textile mills (313): 10%
- Wood products (321): 12%
- Chemical manufacturing (325): 18%
- Fabricated metal products (332): 20%
- Machinery manufacturing (333): 15%
- Other: 10%

## Attrition Patterns

- Exit is correlated with low productivity (creates selection bias)
- Entry is higher in high-growth industries
- Approximately 13% attrition rate per year
- Useful for teaching Sample Selection models and Heckman corrections

## References

- Olley, G. S., & Pakes, A. (1996). The Dynamics of Productivity in the Telecommunications Equipment Industry. *Econometrica*, 64(6), 1263-1297.
- Wooldridge, J. M. (2009). On Estimating Firm-Level Production Functions Using Proxy Variables to Control for Unobservables. *Economics Letters*, 104(3), 112-114.

## Data Quality Notes

- This is synthetic data created for educational purposes
- Output, labor, capital, and materials are realistic but simulated
- TFP estimates computed using standard Olley-Pakes methodology
- Attrition is generated to mimic real manufacturing panel patterns

## Citation

```
PanelBox Educational Datasets (2026). Firm Productivity Dataset.
Synthetic data for panel data methods instruction.
```
