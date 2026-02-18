# US Counties Dataset

## Overview

This dataset contains socioeconomic panel data for US counties with geographic boundaries for spatial analysis.

## Data Description

**Geographic Coverage**: ~3,000 US counties (contiguous United States)
**Time Period**: 2000-2020 (annual observations)
**Panel Structure**: Balanced panel (N×T ≈ 63,000 observations)

## Variables

### Identifiers
- `county_fips`: 5-digit FIPS code (entity identifier)
- `county_name`: County name
- `state_fips`: 2-digit state FIPS code
- `state_name`: State name
- `year`: Year (time identifier)

### Economic Variables
- `income_pc`: Per capita personal income (USD, current)
- `income_pc_real`: Per capita personal income (USD, 2015 constant)
- `gdp_county`: County GDP (millions USD)
- `employment`: Total employment (thousands)
- `unemployment_rate`: Unemployment rate (%)
- `median_household_income`: Median household income (USD)

### Demographic Variables
- `population`: Total population
- `population_density`: Population per square mile
- `pct_urban`: Percentage urban population
- `pct_white`: Percentage white (non-Hispanic)
- `pct_black`: Percentage Black or African American
- `pct_hispanic`: Percentage Hispanic or Latino
- `median_age`: Median age (years)

### Education Variables
- `pct_hs_graduate`: % age 25+ with high school diploma or higher
- `pct_bachelors`: % age 25+ with bachelor's degree or higher
- `pct_graduate`: % age 25+ with graduate or professional degree

### Geographic Variables
- `latitude`: County centroid latitude
- `longitude`: County centroid longitude
- `land_area`: Land area (square miles)
- `water_area`: Water area (square miles)

## Files

### Data Files
- `us_counties.csv`: Main panel data file
- `us_counties.shp`: ESRI Shapefile (geometry)
- `us_counties.shx`: Shapefile index
- `us_counties.dbf`: Shapefile attribute table
- `us_counties.prj`: Coordinate reference system (CRS) definition

### File Format

**CSV Structure**:
```
county_fips,county_name,state_fips,state_name,year,income_pc,population,...
01001,Autauga County,01,Alabama,2000,24571,43671,...
01001,Autauga County,01,Alabama,2001,25432,44234,...
...
```

**Coordinate Reference System**: NAD83 / EPSG:4269

## Data Sources

- **Bureau of Economic Analysis (BEA)**: Income and GDP data
- **US Census Bureau**: Demographic and education data
- **Bureau of Labor Statistics (BLS)**: Employment and unemployment
- **US Census TIGER/Line**: County boundaries

## Usage in Tutorials

This dataset is used in:
- **Notebook 01**: Introduction to spatial autocorrelation
- **Notebook 02**: Building contiguity-based spatial weights
- **Notebook 03**: Estimating spatial lag models for income convergence

## Loading the Data

### Load CSV with Pandas
```python
import pandas as pd
from pathlib import Path

data_path = Path("data/us_counties/us_counties.csv")
df = pd.read_csv(data_path, dtype={"county_fips": str, "state_fips": str})
```

### Load Shapefile with GeoPandas
```python
import geopandas as gpd

shapefile_path = Path("data/us_counties/us_counties.shp")
gdf = gpd.read_file(shapefile_path)
```

### Merge Data and Geography
```python
# Merge CSV data with shapefile
gdf_merged = gdf.merge(df, on="county_fips", how="inner")
```

## Data Cleaning Notes

- Alaska (state FIPS 02) and Hawaii (state FIPS 15) excluded for contiguous US analysis
- Some counties have missing data for certain years (imputed or dropped)
- County boundaries stable over time (2010 Census definitions)
- Income variables adjusted for inflation using CPI-U (base year 2015)

## Citation

If using this data, please cite:

```
US Bureau of Economic Analysis (2023). Regional Economic Accounts.
US Census Bureau (2023). American Community Survey.
Bureau of Labor Statistics (2023). Local Area Unemployment Statistics.
```

## Example Analysis

### Computing Spatial Autocorrelation
```python
from libpysal import weights
from esda import Moran

# Build Queen contiguity weights
w = weights.Queen.from_shapefile("data/us_counties/us_counties.shp")

# Subset to one year
df_2020 = df[df["year"] == 2020]

# Compute Moran's I for income
moran = Moran(df_2020["income_pc_real"], w)
print(f"Moran's I: {moran.I:.4f}")
print(f"p-value: {moran.p_sim:.4f}")
```

## Known Issues

- Some small island counties may not have contiguous neighbors
- Missing data in early years (2000-2005) for some economic variables
- County mergers/splits in Virginia require special handling

## Version

**Data Version**: 1.0
**Last Updated**: 2026-02-16
**Generated for**: PanelBox Spatial Tutorials
