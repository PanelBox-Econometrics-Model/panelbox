# Grunfeld Dataset Codebook

## Overview
- **Source**: Grunfeld, Y. (1958). The Determinants of Corporate Investment
- **Sample**: 10 firms, 20 years (1935-1954), 200 total observations
- **Time Range**: 1935 - 1954
- **Entity Type**: Large US manufacturing firms
- **Panel Type**: Balanced panel

## Description

This is a classic dataset in econometrics, used extensively for teaching and demonstrating panel data methods. The data contains information on investment, market value, and capital stock for 10 major US manufacturing corporations over the period 1935-1954.

## Variables

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| firm | int | Firm identifier (1-10) | - |
| year | int | Year of observation | - |
| invest | float | Gross investment | Millions of 1947 dollars |
| value | float | Market value of the firm | Millions of 1947 dollars |
| capital | float | Stock of plant and equipment | Millions of 1947 dollars |

## Firms Included

1. General Motors
2. US Steel
3. General Electric
4. Chrysler
5. Atlantic Refining
6. IBM
7. Union Oil
8. Westinghouse
9. Goodyear
10. Diamond Match

## Missing Data

No missing values in this dataset. All 200 observations (10 firms × 20 years) are complete.

## Typical Usage

This dataset is commonly used to estimate investment equations of the form:

```
invest_it = β₀ + β₁ value_it + β₂ capital_it + α_i + ε_it
```

where:
- `invest_it` is the dependent variable (gross investment)
- `value_it` represents the firm's market value (proxy for investment opportunities)
- `capital_it` represents the existing capital stock
- `α_i` is the firm-specific fixed effect
- `ε_it` is the error term

## References

- Grunfeld, Y. (1958). *The Determinants of Corporate Investment*. Unpublished Ph.D. dissertation, University of Chicago.
- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Data Quality Notes

- All values are in constant 1947 dollars, controlling for inflation
- The data covers the Great Depression and World War II period, which may affect interpretation
- This is a canonical dataset with well-documented properties
- Known to exhibit strong firm-specific effects (high within-firm correlation)

## Citation

When using this dataset, please cite:

```
Grunfeld, Y. (1958). The Determinants of Corporate Investment.
Unpublished Ph.D. dissertation, University of Chicago.
```
