# Country Growth Dataset Codebook

## Overview
- **Source**: Penn World Tables (PWT) version 10.0
- **Sample**: 80 countries, 30 years (1990-2019), 2400 total observations
- **Time Range**: 1990 - 2019
- **Entity Type**: Countries (global sample)
- **Panel Type**: Balanced panel

## Description

This dataset contains macroeconomic indicators for 80 countries over 30 years, derived from the Penn World Tables. It is designed for studying economic growth, convergence, and cross-country productivity differences.

## Variables

| Variable | Type | Description | Unit |
|----------|------|-------------|------|
| country_code | str | ISO 3-letter country code | - |
| year | int | Year of observation | - |
| gdp_pc | float | Real GDP per capita | 2017 PPP dollars |
| investment | float | Investment share of GDP | Percent (0-100) |
| consumption | float | Consumption share of GDP | Percent (0-100) |
| government | float | Government share of GDP | Percent (0-100) |
| openness | float | Trade openness (exports + imports)/GDP | Percent |
| population | float | Population | Millions |
| education | float | Average years of schooling | Years |
| life_expectancy | float | Life expectancy at birth | Years |
| region | str | Geographic region | Categorical |
| income_group | str | Income classification (World Bank) | Categorical |

## Regions

- **Africa**: 15 countries
- **Americas**: 18 countries
- **Asia**: 22 countries
- **Europe**: 20 countries
- **Oceania**: 5 countries

## Income Groups

- High income: 30 countries
- Upper middle income: 25 countries
- Lower middle income: 20 countries
- Low income: 5 countries

## Missing Data

Education and life expectancy have some missing values for early years (1990-1995) for certain countries. Approximately 3% of observations have missing values in these variables.

## Typical Usage

**Growth regression**:
```
Δlog(gdp_pc_it) = β₀ + β₁ log(gdp_pc_it-1) + β₂ investment_it + β₃ education_it + α_i + ε_it
```

**Convergence analysis**:
```
Δlog(gdp_pc_it) = α + β log(gdp_pc_it-1) + γ X_it + η_i + ε_it
```

where β < 0 indicates conditional convergence.

## References

- Feenstra, R. C., Inklaar, R., & Timmer, M. P. (2015). The Next Generation of the Penn World Table. *American Economic Review*, 105(10), 3150-3182.
- Barro, R. J., & Sala-i-Martin, X. (2004). *Economic Growth* (2nd ed.). MIT Press.

## Data Quality Notes

- All GDP values are in constant 2017 purchasing power parity (PPP) dollars
- Investment, consumption, and government shares may not sum to 100% due to net exports
- Some countries have incomplete data for the early 1990s (transition economies)
- Education data interpolated for some years when census data unavailable

## Citation

```
Feenstra, Robert C., Robert Inklaar and Marcel P. Timmer (2015),
"The Next Generation of the Penn World Table" American Economic Review,
105(10), 3150-3182.
```
