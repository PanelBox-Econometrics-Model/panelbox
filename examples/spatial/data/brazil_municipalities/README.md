# Brazilian Municipalities Dataset

## Overview

This dataset contains panel data for Brazilian municipalities (municípios) with geographic boundaries for spatial econometric analysis.

## Data Description

**Geographic Coverage**: 5,570 Brazilian municipalities
**Time Period**: 2010-2020 (annual observations)
**Panel Structure**: Balanced panel (N×T = 61,270 observations)

## Variables

### Identifiers
- `munic_code`: 7-digit IBGE municipality code (entity identifier)
- `munic_name`: Municipality name (Portuguese)
- `state_code`: 2-digit state code
- `state_name`: State name
- `region`: Geographic region (Norte, Nordeste, Centro-Oeste, Sudeste, Sul)
- `year`: Year (time identifier)

### Economic Variables
- `gdp_total`: Municipal GDP (thousands BRL, current)
- `gdp_pc`: GDP per capita (BRL)
- `gdp_pc_real`: GDP per capita (BRL, 2015 constant)
- `gdp_agriculture`: Agricultural GDP (thousands BRL)
- `gdp_industry`: Industrial GDP (thousands BRL)
- `gdp_services`: Services GDP (thousands BRL)
- `gva_total`: Gross value added (thousands BRL)

### Social Development
- `idhm`: Municipal Human Development Index (IDHM)
- `idhm_income`: IDHM income component
- `idhm_longevity`: IDHM longevity component
- `idhm_education`: IDHM education component
- `gini`: Gini coefficient (income inequality)
- `poverty_rate`: Poverty rate (% population)
- `extreme_poverty_rate`: Extreme poverty rate (%)

### Demographic Variables
- `population`: Total population
- `population_density`: Population per km²
- `urbanization_rate`: % urban population
- `dependency_ratio`: Dependency ratio (%)

### Public Finance
- `revenue_total`: Total municipal revenue (thousands BRL)
- `revenue_own`: Own revenue (thousands BRL)
- `revenue_transfers`: Federal/state transfers (thousands BRL)
- `expenditure_total`: Total expenditure (thousands BRL)
- `expenditure_education`: Education expenditure (thousands BRL)
- `expenditure_health`: Health expenditure (thousands BRL)

### Infrastructure
- `water_access`: % households with piped water
- `sewage_access`: % households with sewage connection
- `electricity_access`: % households with electricity
- `paved_roads`: Km of paved roads

### Geographic Variables
- `latitude`: Municipality centroid latitude
- `longitude`: Municipality centroid longitude
- `area_km2`: Total area (km²)

## Files

### Data Files
- `brazil_munic.csv`: Main panel data file
- `brazil_munic.shp`: ESRI Shapefile (geometry)
- `brazil_munic.shx`: Shapefile index
- `brazil_munic.dbf`: Shapefile attribute table
- `brazil_munic.prj`: Coordinate reference system definition

### File Format

**CSV Structure**:
```
munic_code,munic_name,state_code,state_name,region,year,gdp_total,population,...
1100015,Alta Floresta d'Oeste,11,Rondônia,Norte,2010,245678,23945,...
1100015,Alta Floresta d'Oeste,11,Rondônia,Norte,2011,267234,24312,...
...
```

**Coordinate Reference System**: SIRGAS 2000 / EPSG:4674

## Data Sources

- **IBGE (Instituto Brasileiro de Geografia e Estatística)**:
  - Municipal GDP and demographics
  - Census and population estimates
  - Municipal boundaries (BCIM)
- **Atlas do Desenvolvimento Humano**: IDHM and social indicators
- **IPEA (Instituto de Pesquisa Econômica Aplicada)**: Economic indicators
- **Tesouro Nacional**: Municipal finance data

## Usage in Tutorials

This dataset is used in:
- **Notebook 02**: Constructing spatial weights for large panels
- **Notebook 05**: Spatial Durbin model for public investment spillovers
- **Notebook 07**: Dynamic spatial panel for development convergence

## Loading the Data

### Load CSV with Pandas
```python
import pandas as pd
from pathlib import Path

data_path = Path("data/brazil_municipalities/brazil_munic.csv")
df = pd.read_csv(data_path, dtype={"munic_code": str, "state_code": str})
```

### Load Shapefile with GeoPandas
```python
import geopandas as gpd

shapefile_path = Path("data/brazil_municipalities/brazil_munic.shp")
gdf = gpd.read_file(shapefile_path)
```

## Data Cleaning Notes

- Municipality codes use IBGE 7-digit standard
- Some municipalities created after 2010 excluded for balanced panel
- GDP deflated using IPCA (Brazilian CPI) with base year 2015
- Missing IDHM values interpolated for intercensal years (2011-2019)
- Financial variables in thousands BRL (nominal terms)

## Example: Regional Analysis

### Analyze Development Spillovers by Region
```python
from libpysal import weights
from panelbox.models.spatial import SpatialDurbin

# Build contiguity weights
w = weights.Queen.from_shapefile("data/brazil_municipalities/brazil_munic.shp")

# Filter to Northeast region
df_ne = df[df["region"] == "Nordeste"]

# Estimate spatial Durbin model
model = SpatialDurbin(
    data=df_ne,
    dependent="idhm",
    exog=["gdp_pc_real", "expenditure_education", "urbanization_rate"],
    entity_id="munic_code",
    time_id="year",
    W=w
)

results = model.fit()
print(results.summary())
```

## Spatial Features

### Regional Patterns
- Strong North-South development gradient
- Coastal vs. interior disparities
- Metropolitan area agglomerations

### Spatial Dependencies
- High spatial autocorrelation in GDP (Moran's I ≈ 0.65)
- Public investment spillovers across borders
- Knowledge diffusion in clusters

## Citation

If using this data, please cite:

```
IBGE - Instituto Brasileiro de Geografia e Estatística (2023).
Sistema de Contas Regionais do Brasil.

PNUD, IPEA, FJP (2023). Atlas do Desenvolvimento Humano no Brasil.
```

## Additional Resources

- IBGE Portal: https://www.ibge.gov.br/
- Atlas do Desenvolvimento Humano: http://www.atlasbrasil.org.br/
- IPEA Data: http://www.ipeadata.gov.br/

## Known Issues

- Municipality boundary changes in some states (São Paulo, Tocantins)
- Census years (2010, 2020) have more detailed data
- Some remote Amazon municipalities have missing infrastructure data

## Version

**Data Version**: 1.0
**Last Updated**: 2026-02-16
**Generated for**: PanelBox Spatial Tutorials
