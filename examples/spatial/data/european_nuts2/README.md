# European NUTS-2 Regions Dataset

## Overview

This dataset contains panel data for European NUTS-2 (Nomenclature of Territorial Units for Statistics, level 2) regions with geographic boundaries.

## Data Description

**Geographic Coverage**: ~270 NUTS-2 regions across EU27 + UK + EFTA countries
**Time Period**: 2005-2020 (annual observations)
**Panel Structure**: Balanced panel (N×T ≈ 4,320 observations)

## Variables

### Identifiers
- `nuts2_code`: NUTS-2 region code (e.g., "DE21", "FR10", "IT51")
- `nuts2_name`: Region name (English)
- `nuts1_code`: NUTS-1 parent region code
- `nuts0_code`: Country code (ISO 3166-1 alpha-2)
- `country_name`: Country name
- `year`: Year (time identifier)

### Economic Variables
- `gdp_eur`: Regional GDP (millions EUR, current)
- `gdp_pps`: Regional GDP (millions PPS - Purchasing Power Standard)
- `gdp_pc_eur`: GDP per capita (EUR)
- `gdp_pc_pps`: GDP per capita (PPS)
- `gdp_pc_real`: Real GDP per capita (EUR, 2015 constant)
- `gva_total`: Gross value added (millions EUR)
- `labor_productivity`: Labor productivity (EUR per hour worked)

### Innovation and R&D
- `rd_expenditure`: R&D expenditure (millions EUR)
- `rd_intensity`: R&D as % of GDP
- `rd_personnel`: R&D personnel (full-time equivalent)
- `patent_applications`: Patent applications (EPO)
- `high_tech_employment`: % employment in high-tech sectors

### Labor Market
- `employment_total`: Total employment (thousands)
- `unemployment_rate`: Unemployment rate (%)
- `activity_rate`: Activity rate (% population 15-64)
- `employment_rate`: Employment rate (% population 15-64)
- `youth_unemployment`: Youth unemployment rate (% population 15-24)

### Human Capital
- `tertiary_education`: % population 25-64 with tertiary education
- `early_leavers`: % early leavers from education (age 18-24)
- `lifelong_learning`: % population 25-64 in education/training
- `digital_skills`: % population with basic/above basic digital skills

### Demographic Variables
- `population`: Total population
- `population_density`: Population per km²
- `median_age`: Median age (years)
- `dependency_ratio`: Old-age dependency ratio (%)

### Infrastructure
- `motorway_density`: Km of motorways per 1,000 km²
- `railway_density`: Km of railways per 1,000 km²
- `broadband_coverage`: % households with broadband access

### Regional Characteristics
- `capital_region`: Dummy for capital/metropolitan region (1/0)
- `cohesion_region`: EU cohesion policy target region (1/0)
- `coastal`: Coastal region dummy (1/0)

### Geographic Variables
- `latitude`: Region centroid latitude
- `longitude`: Region centroid longitude
- `area_km2`: Total area (km²)

## Files

### Data Files
- `eu_nuts2.csv`: Main panel data file
- `eu_nuts2.shp`: ESRI Shapefile (geometry)
- `eu_nuts2.shx`: Shapefile index
- `eu_nuts2.dbf`: Shapefile attribute table
- `eu_nuts2.prj`: Coordinate reference system definition

### File Format

**CSV Structure**:
```
nuts2_code,nuts2_name,nuts0_code,country_name,year,gdp_pc_pps,population,...
DE21,Oberbayern,DE,Germany,2005,38542,4234567,...
DE21,Oberbayern,DE,Germany,2006,39876,4289123,...
...
```

**Coordinate Reference System**: ETRS89 / EPSG:4258

## Data Sources

- **Eurostat**: Primary source for all regional statistics
  - Regional economic accounts (nama_10r_*)
  - Regional labor market statistics (lfst_r_*)
  - Regional innovation statistics (rd_e_*)
  - Regional demographic statistics (demo_r_*)
- **European Patent Office (EPO)**: Patent data
- **OECD Regional Database**: Supplementary indicators

## Usage in Tutorials

This dataset is used in:
- **Notebook 05**: Spatial Durbin model for R&D spillovers
- **Notebook 06**: Marginal effects decomposition for innovation policy
- **Notebook 08**: Spatial specification tests and model comparison

## Loading the Data

### Load CSV with Pandas
```python
import pandas as pd
from pathlib import Path

data_path = Path("data/european_nuts2/eu_nuts2.csv")
df = pd.read_csv(data_path, dtype={"nuts2_code": str, "nuts0_code": str})
```

### Load Shapefile with GeoPandas
```python
import geopandas as gpd

shapefile_path = Path("data/european_nuts2/eu_nuts2.shp")
gdf = gpd.read_file(shapefile_path)
```

## Data Cleaning Notes

- NUTS-2 boundaries based on 2016 classification (stable 2015-2020)
- UK regions (UKxx) included through 2020
- GDP deflated using national GDP deflators, base year 2015
- Some missing R&D data for small regions (confidential/unavailable)
- Outermost regions (French overseas, Canary Islands) included
- PPS conversion uses EU28 average (before Brexit)

## Example: R&D Spillovers Analysis

### Estimate Knowledge Spillovers
```python
from libpysal import weights
from panelbox.models.spatial import SpatialDurbin
from panelbox.effects import compute_spatial_effects

# Build distance-based weights (500 km threshold)
coords = df[["longitude", "latitude"]].drop_duplicates()
w = weights.DistanceBand.from_array(coords.values, threshold=500, binary=False)

# Estimate spatial Durbin model
model = SpatialDurbin(
    data=df,
    dependent="labor_productivity",
    exog=["rd_intensity", "tertiary_education", "motorway_density"],
    entity_id="nuts2_code",
    time_id="year",
    W=w
)

results = model.fit()

# Compute direct and indirect effects
effects = compute_spatial_effects(results, w)
print(effects)
```

## Spatial Features

### Regional Patterns
- Core-periphery structure (Blue Banana, Sunbelt)
- Capital regions with high productivity
- Innovation clusters (Bavaria, Île-de-France, Lombardy)
- Convergence clubs (Eastern vs. Western Europe)

### Spatial Dependencies
- Strong R&D spillovers (Moran's I ≈ 0.72)
- Cross-border labor market integration
- Technology diffusion along transport corridors

## Applications

### Research Questions
1. Do R&D investments generate regional spillovers?
2. What is the spatial reach of innovation diffusion?
3. How do infrastructure investments affect neighboring regions?
4. Do EU cohesion policies create spatial dependencies?

## Citation

If using this data, please cite:

```
Eurostat (2023). Regional Statistics by NUTS Classification.
European Commission, Luxembourg.
https://ec.europa.eu/eurostat/web/regions/data/database

European Patent Office (2023). PATSTAT Database.
```

## Additional Resources

- Eurostat Regions: https://ec.europa.eu/eurostat/web/regions
- NUTS Classification: https://ec.europa.eu/eurostat/web/nuts
- OECD Regional Statistics: https://www.oecd.org/regional/

## Known Issues

- NUTS classification revisions (2010, 2013, 2016, 2021)
- Brexit impact on UK regions data post-2020
- Missing data for some small German regions (NUTS-2 level)
- Regional boundary changes require careful handling

## Version

**Data Version**: 1.0
**Last Updated**: 2026-02-16
**Generated for**: PanelBox Spatial Tutorials
