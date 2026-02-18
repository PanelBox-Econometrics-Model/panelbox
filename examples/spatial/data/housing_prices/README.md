# Housing Prices Dataset

## Overview

This dataset contains panel data for residential housing sales with geographic coordinates, suitable for spatial hedonic price analysis.

## Data Description

**Geographic Coverage**: Metropolitan area (example: Boston, MA or San Francisco, CA)
**Time Period**: 2015-2020 (quarterly observations)
**Panel Structure**: Repeated cross-sections (N ≈ 50,000 unique properties)
**Unit of Observation**: Individual housing transaction

## Variables

### Identifiers
- `property_id`: Unique property identifier
- `transaction_id`: Unique transaction identifier
- `quarter`: Year-Quarter (e.g., "2015Q1", "2020Q4")
- `year`: Year
- `quarter_num`: Quarter number (1-4)

### Price Variables
- `sale_price`: Sale price (USD, nominal)
- `sale_price_real`: Real sale price (USD, 2015 constant)
- `price_per_sqft`: Price per square foot
- `log_price`: Natural log of sale price

### Property Characteristics
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `sqft_living`: Interior living space (square feet)
- `sqft_lot`: Lot size (square feet)
- `floors`: Number of floors
- `waterfront`: Waterfront property (1/0)
- `view_quality`: View quality (0-4 scale)
- `condition`: Overall condition (1-5 scale)
- `grade`: Construction quality grade (1-13 scale)
- `year_built`: Year property was built
- `year_renovated`: Year of last renovation (0 if never)
- `age`: Property age at sale
- `basement`: Has basement (1/0)
- `garage`: Number of garage spaces

### Location Variables
- `latitude`: Property latitude (decimal degrees)
- `longitude`: Property longitude (decimal degrees)
- `zipcode`: ZIP code
- `neighborhood`: Neighborhood/district name
- `school_district`: School district code

### Neighborhood Characteristics (ZIP-level)
- `median_income`: Median household income in ZIP
- `crime_rate`: Crimes per 1,000 residents
- `school_rating`: Average school rating (1-10 scale)
- `walk_score`: Walkability score (0-100)
- `transit_score`: Transit accessibility score (0-100)
- `park_distance`: Distance to nearest park (miles)
- `cbd_distance`: Distance to central business district (miles)
- `highway_distance`: Distance to nearest highway (miles)

### Market Context
- `days_on_market`: Days property was listed before sale
- `list_price`: Original listing price
- `price_cut`: Indicator for price reduction (1/0)
- `multiple_offers`: Indicator for multiple offers (1/0)

## Files

### Data Files
- `housing_prices.csv`: Main transaction data with coordinates

**Note**: This dataset uses point coordinates rather than polygon boundaries, suitable for distance-based spatial weights.

### File Format

**CSV Structure**:
```
property_id,transaction_id,quarter,year,sale_price,bedrooms,bathrooms,sqft_living,latitude,longitude,...
P001,T0001,2015Q1,2015,450000,3,2,1850,42.3601,-71.0589,...
P002,T0002,2015Q1,2015,625000,4,2.5,2400,42.3656,-71.0623,...
...
```

## Data Sources

- **Public Records**: County assessor and recorder offices
- **Multiple Listing Service (MLS)**: Real estate transaction data
- **Census Bureau**: Neighborhood demographics
- **Walk Score API**: Walkability and transit scores
- **Local Police Departments**: Crime statistics
- **School Districts**: School performance ratings

## Usage in Tutorials

This dataset is used in:
- **Notebook 03**: Spatial lag model for housing price spillovers
- **Notebook 05**: Spatial Durbin model with neighborhood amenities
- **Notebook 06**: Marginal effects of location characteristics

## Loading the Data

### Load CSV with Pandas
```python
import pandas as pd
from pathlib import Path

data_path = Path("data/housing_prices/housing_prices.csv")
df = pd.read_csv(data_path)

# Convert quarter to datetime
df["date"] = pd.PeriodIndex(df["quarter"], freq="Q").to_timestamp()
```

### Create GeoDataFrame
```python
import geopandas as gpd
from shapely.geometry import Point

# Create Point geometries
geometry = [Point(xy) for xy in zip(df["longitude"], df["latitude"])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
```

## Data Cleaning Notes

- Outliers removed (prices < 50k or > 5M)
- Foreclosures and non-arm's-length sales excluded
- Missing values imputed or observations dropped
- Prices deflated using regional CPI (base year 2015)
- Coordinates jittered slightly for privacy (±0.0001 degrees)

## Example: Spatial Hedonic Model

### Estimate Hedonic Price Model with Spatial Lag
```python
from libpysal import weights
from panelbox.models.spatial import SpatialLag

# Build k-nearest neighbors weights (k=10)
coords = df[["longitude", "latitude"]].values
w = weights.KNN.from_array(coords, k=10)

# Estimate spatial lag model
model = SpatialLag(
    data=df,
    dependent="log_price",
    exog=[
        "bedrooms", "bathrooms", "sqft_living", "age",
        "waterfront", "view_quality", "condition",
        "school_rating", "crime_rate", "cbd_distance"
    ],
    W=w
)

results = model.fit()
print(results.summary())
```

## Spatial Features

### Price Patterns
- Strong spatial clustering (Moran's I ≈ 0.55)
- Spillover effects from neighboring sales
- School district boundaries create discontinuities
- Waterfront premium with spatial heterogeneity

### Spatial Dependencies
- **Peer effects**: Comparable sales influence pricing
- **Amenity spillovers**: Parks, schools affect neighboring values
- **Negative spillovers**: Crime, vacancies reduce nearby prices

## Applications

### Research Questions
1. What is the magnitude of spatial price spillovers?
2. How do school quality improvements affect neighboring home values?
3. Do green spaces generate positive spatial externalities?
4. What is the spatial decay of amenity effects?

## Hedonic Variables

### Structural Characteristics
- Size (square footage, rooms)
- Quality (condition, grade)
- Age and depreciation

### Location Attributes
- Accessibility (CBD distance, transit)
- School quality
- Neighborhood safety
- Environmental amenities

### Spatial Interactions
- Neighboring property values
- Proximity to amenities
- Spatial multiplier effects

## Citation

If using this data, please cite data sources and acknowledge:

```
County Assessor's Office, [County Name] (2023).
Real Estate Transaction Records.

Walk Score® (2023). Walkability and Transit Scores.
www.walkscore.com
```

## Privacy Notes

- Exact addresses removed for privacy
- Coordinates slightly randomized (±10 meters)
- Individual property IDs are anonymous
- Aggregated to protect confidentiality

## Known Issues

- Seasonal effects in quarterly data
- Selection bias (only completed transactions observed)
- Missing renovation year for many properties
- School ratings may change over time period
- Crime data aggregated at ZIP level (not property-specific)

## Version

**Data Version**: 1.0
**Last Updated**: 2026-02-16
**Generated for**: PanelBox Spatial Tutorials
