# Agricultural Productivity Dataset

## Overview

This dataset contains panel data for agricultural regions with crop yields, input usage, and environmental factors suitable for spatial production function analysis.

## Data Description

**Geographic Coverage**: Agricultural regions in Midwest US (Iowa, Illinois, Indiana, etc.)
**Time Period**: 2010-2020 (annual observations)
**Panel Structure**: Balanced panel (N ≈ 500-1,000 regions, T = 11 years)
**Unit of Observation**: Agricultural Statistical District (ASD) or county

## Variables

### Identifiers
- `region_id`: Agricultural region identifier
- `region_name`: Region name
- `state_fips`: State FIPS code
- `state_name`: State name
- `year`: Year (time identifier)

### Crop Yields (bushels per acre)
- `corn_yield`: Corn yield
- `soybean_yield`: Soybean yield
- `wheat_yield`: Wheat yield
- `corn_acres`: Acres planted with corn
- `soybean_acres`: Acres planted with soybeans
- `wheat_acres`: Acres planted with wheat

### Input Variables
- `fertilizer_nitrogen`: Nitrogen fertilizer (lbs/acre)
- `fertilizer_phosphorus`: Phosphorus fertilizer (lbs/acre)
- `fertilizer_potassium`: Potassium fertilizer (lbs/acre)
- `pesticide_expenditure`: Pesticide expenditure (USD/acre)
- `seed_expenditure`: Seed expenditure (USD/acre)
- `irrigation_share`: % of cropland irrigated
- `machinery_value`: Machinery and equipment value (USD/acre)

### Environmental and Climate Variables
- `precipitation_growing`: Growing season precipitation (inches)
- `temperature_avg`: Average growing season temperature (°F)
- `temperature_gdd`: Growing degree days (base 50°F)
- `drought_index`: Palmer Drought Severity Index
- `frost_free_days`: Number of frost-free days
- `soil_quality`: Soil productivity index (0-100)
- `erosion_risk`: Soil erosion risk category (1-5)

### Management Practices
- `conservation_tillage`: % acres with conservation tillage
- `cover_crops`: % acres with cover crops
- `precision_agriculture`: Adoption of precision ag technology (1/0)
- `organic_share`: % acres certified organic

### Economic Variables
- `corn_price`: Corn price received (USD/bushel)
- `soybean_price`: Soybean price received (USD/bushel)
- `land_rent`: Average cash rent (USD/acre)
- `labor_cost`: Hired labor cost (USD/acre)
- `farm_size_avg`: Average farm size in region (acres)

### Demographic and Structural
- `farms_count`: Number of farms in region
- `operator_age_avg`: Average operator age
- `education_college`: % operators with college degree
- `off_farm_income`: % operators with off-farm income

### Geographic Variables
- `latitude`: Region centroid latitude
- `longitude`: Region centroid longitude
- `elevation`: Average elevation (feet)
- `area_sq_miles`: Total area (square miles)
- `cropland_share`: % of area in cropland

## Files

### Data Files
- `agricultural_productivity.csv`: Main panel data file

**Note**: This dataset uses region centroids for spatial analysis. Spatial weights can be constructed using distance or shared borders between agricultural regions.

### File Format

**CSV Structure**:
```
region_id,region_name,state_fips,state_name,year,corn_yield,soybean_yield,fertilizer_nitrogen,...
IA010,Northwest Iowa,19,Iowa,2010,165.3,52.1,145.2,...
IA010,Northwest Iowa,19,Iowa,2011,172.8,54.6,148.7,...
...
```

## Data Sources

- **USDA National Agricultural Statistics Service (NASS)**:
  - Crop yields and acreage
  - Prices received by farmers
- **USDA Economic Research Service (ERS)**:
  - Input usage and costs
  - Farm structure data
- **NOAA National Centers for Environmental Information**:
  - Weather and climate data
  - Drought indices
- **USDA Natural Resources Conservation Service**:
  - Soil quality data
  - Conservation practice adoption
- **Census of Agriculture**: Farm demographics and characteristics

## Usage in Tutorials

This dataset is used in:
- **Notebook 04**: Spatial error model for agricultural productivity
  - Testing for spatial error autocorrelation
  - Comparing SAR vs. SEM specifications

## Loading the Data

### Load CSV with Pandas
```python
import pandas as pd
from pathlib import Path

data_path = Path("data/agriculture/agricultural_productivity.csv")
df = pd.read_csv(data_path, dtype={"region_id": str, "state_fips": str})
```

### Create Spatial Weights
```python
from libpysal import weights

# Build contiguity weights for agricultural regions
coords = df[["longitude", "latitude"]].drop_duplicates()
w = weights.KNN.from_array(coords.values, k=8)
```

## Data Cleaning Notes

- Yields normalized to standard moisture content
- Input variables per acre of cropland (not total area)
- Prices are calendar year averages
- Weather data aggregated to growing season (April-September)
- Missing values imputed using adjacent years or spatial interpolation
- Outliers (extreme weather events, disasters) flagged but retained

## Example: Spatial Production Function

### Estimate Spatial Error Model
```python
from panelbox.models.spatial import SpatialError
from panelbox.diagnostics import spatial_lm_tests

# Test for spatial error dependence
lm_tests = spatial_lm_tests(
    data=df,
    dependent="corn_yield",
    exog=["fertilizer_nitrogen", "precipitation_growing", "temperature_gdd", "soil_quality"],
    entity_id="region_id",
    time_id="year",
    W=w
)
print(lm_tests)

# Estimate spatial error model
model = SpatialError(
    data=df,
    dependent="corn_yield",
    exog=["fertilizer_nitrogen", "precipitation_growing", "temperature_gdd", "soil_quality"],
    entity_id="region_id",
    time_id="year",
    W=w
)

results = model.fit()
print(results.summary())
```

## Spatial Features

### Spatial Autocorrelation Sources
- **Weather and climate**: Spatially correlated precipitation and temperature
- **Pest and disease**: Spread across neighboring regions
- **Soil characteristics**: Gradual spatial variation
- **Technology diffusion**: Neighboring farmers adopt similar practices
- **Unobserved management**: Spatial correlation in farming quality

### Why Spatial Error Model?
This dataset is ideal for demonstrating SEM because:
1. Yields affected by spatially correlated weather shocks
2. Pest pressures spread geographically
3. Omitted soil characteristics create spatial error correlation
4. No direct spillovers in production (unlike knowledge or R&D)

## Applications

### Research Questions
1. Do spatially correlated weather shocks affect yield estimates?
2. How does technology diffusion create spatial dependencies?
3. What is the role of unobserved spatial heterogeneity?
4. Do conservation practices generate spatial externalities?

## Production Function Framework

### Cobb-Douglas Specification
```
log(yield) = α + β₁·log(nitrogen) + β₂·log(precipitation) +
             β₃·temperature + β₄·soil_quality + u

where u = λWu + ε  (spatial error process)
```

### Environmental Interactions
- Fertilizer response depends on rainfall
- Temperature effects non-linear (optimal range)
- Soil quality moderates input productivity

## Citation

If using this data, please cite:

```
USDA National Agricultural Statistics Service (2023).
Crop Production and Quick Stats Database.
https://quickstats.nass.usda.gov/

NOAA National Centers for Environmental Information (2023).
Climate Data Online.
https://www.ncdc.noaa.gov/cdo-web/
```

## Policy Relevance

### Applications
- **Precision agriculture**: Spatial targeting of inputs
- **Climate adaptation**: Regional vulnerability assessment
- **Conservation policy**: Targeting environmental programs
- **Crop insurance**: Spatially correlated risk pricing

## Known Issues

- Aggregation to region level masks within-region heterogeneity
- Changing crop mixes over time affect yield comparisons
- Irrigation data incomplete for some regions
- Conservation practice adoption self-reported (measurement error)
- Extreme weather years (2012 drought) create outliers

## Version

**Data Version**: 1.0
**Last Updated**: 2026-02-16
**Generated for**: PanelBox Spatial Tutorials
