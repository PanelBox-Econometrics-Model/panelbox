# Spatial Diagnostics Datasets

**Version:** 1.0.0
**Last Updated:** 2026-02-22

This directory contains datasets and spatial weight matrices used in Tutorial 04 (Spatial Diagnostics). The datasets include panel data with known spatial dependence structures and pre-built weight matrices for Moran's I, LM tests, and spatial model specification.

---

## Panel Datasets

### 1. us_counties.csv

**Tutorial:** 04 - Spatial Diagnostics
**Dimensions:** 500 counties x 10 years = 5,000 observations
**Type:** Balanced panel
**Purpose:** County-level economic data for demonstrating spatial autocorrelation tests and LM diagnostics

#### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `county_id` | County identifier | Integer | 1-500 | Entity ID |
| `county_name` | County name | String | -- | Label |
| `state` | State abbreviation | String | -- | Grouping |
| `year` | Year | Integer | 2010-2019 | Time ID |
| `log_income_pc` | Log per capita income | Log USD | 9.5-11.5 | Spatially dependent |
| `unemployment_rate` | Unemployment rate | Percentage | 2-18 | Spatially dependent |
| `log_population` | Log population | Log count | 7-15 | Spatially correlated |
| `pct_college` | Percentage with college degree | Percentage | 5-60 | Spatially correlated |
| `poverty_rate` | Poverty rate | Percentage | 3-40 | Spatially dependent |
| `log_house_value` | Log median house value | Log USD | 10.5-14 | Strongly spatially dependent |

#### Key Features

- **Spatial lag dependence**: `log_income_pc` and `log_house_value` are generated with spatial autoregressive (SAR) processes, meaning each county's value depends on its neighbors' values. LM-lag tests should be significant.
- **Spatial error dependence**: `unemployment_rate` and `poverty_rate` include spatially correlated errors (SEM process), where unobserved shocks spill across neighboring counties. LM-error tests should be significant.
- **Mixed dependence**: Some variables exhibit both spatial lag and spatial error dependence, requiring robust LM tests to distinguish.
- **Strong spatial clustering**: `log_house_value` has the strongest spatial dependence (Moran's I ~ 0.6), while `pct_college` has moderate dependence (Moran's I ~ 0.3).

#### Data Generating Process

For spatially lagged variables:
```
y_{t} = rho * W * y_{t} + X_{t} * beta + alpha + epsilon_{t}
```
where `rho` is the spatial autoregressive parameter (0.3-0.5) and `W` is the row-standardized contiguity weight matrix.

For spatially correlated errors:
```
y_{t} = X_{t} * beta + alpha + u_{t},  where  u_{t} = lambda * W * u_{t} + e_{t}
```
where `lambda` is the spatial error parameter (0.2-0.4).

---

### 2. eu_regions.csv

**Tutorial:** 04 - Spatial Diagnostics
**Dimensions:** 200 regions x 15 years = 3,000 observations
**Type:** Balanced panel
**Purpose:** European regional data for spatial weight matrix validation and cross-border spillover analysis

#### Variables

| Variable | Description | Units | Range | Properties |
|----------|-------------|-------|-------|------------|
| `region_id` | NUTS-2 region identifier | Integer | 1-200 | Entity ID |
| `region_name` | Region name | String | -- | Label |
| `country` | Country code | String (ISO-2) | -- | Grouping |
| `year` | Year | Integer | 2005-2019 | Time ID |
| `log_gdp_pc` | Log GDP per capita | Log EUR (PPS) | 9.0-11.0 | Spatially dependent |
| `employment_rate` | Employment rate | Percentage | 50-85 | Spatially correlated |
| `log_rd_intensity` | Log R&D expenditure as % of GDP | Log percentage | -2-2 | Spatially clustered |
| `accessibility` | Transport accessibility index | Index (0-100) | 20-95 | Spatially smooth |
| `human_capital` | Share with tertiary education | Percentage | 10-55 | Spatially correlated |
| `structural_funds` | EU structural funds per capita | EUR per capita | 0-500 | Spatially clustered |

#### Key Features

- **Cross-border spillovers**: Spatial dependence crosses national borders, demonstrating the importance of spatial weights that connect regions in different countries.
- **Core-periphery pattern**: Strong spatial clustering of income levels (high-income core in Western Europe, lower-income periphery), producing high global Moran's I values.
- **Weight matrix comparison**: Designed to show different results under contiguity vs. distance-based weights, motivating weight matrix specification testing.

---

## Spatial Weight Matrices

### W_counties.npy

**Dimensions:** 500 x 500
**Type:** Queen contiguity weight matrix
**Format:** NumPy binary (`.npy`)

#### Properties

- **Construction**: Two counties are neighbors if they share a boundary or vertex (Queen contiguity)
- **Row standardization**: Each row sums to 1 (weights are `1/n_i` where `n_i` is the number of neighbors for county i)
- **Symmetry**: The underlying binary contiguity matrix is symmetric; row standardization makes `W` asymmetric
- **Sparsity**: ~98% zeros (most counties have 4-8 neighbors out of 500)
- **No islands**: Every county has at least one neighbor (minimum degree = 1)
- **Average neighbors**: ~5.5 per county

#### Loading

```python
import numpy as np

W = np.load('W_counties.npy')
print(f"Shape: {W.shape}")           # (500, 500)
print(f"Row sums: {W.sum(axis=1)}")  # All approximately 1.0
print(f"Sparsity: {(W == 0).mean():.2%}")  # ~98%
```

#### Verification

```python
# Check row standardization
assert np.allclose(W.sum(axis=1), 1.0), "Not row-standardized"

# Check no self-neighbors
assert np.allclose(np.diag(W), 0.0), "Self-neighbors detected"

# Check non-negative
assert (W >= 0).all(), "Negative weights detected"

# Check connectivity (no islands)
assert (W.sum(axis=1) > 0).all(), "Island detected"
```

---

### W_counties_distance.npy

**Dimensions:** 500 x 500
**Type:** Inverse-distance weight matrix
**Format:** NumPy binary (`.npy`)

#### Properties

- **Construction**: `w_{ij} = 1/d_{ij}` for counties within a distance threshold, 0 otherwise
- **Distance threshold**: 150 km (counties beyond this distance receive zero weight)
- **Row standardization**: Each row sums to 1
- **Symmetry**: The underlying distance matrix is symmetric; row standardization may introduce mild asymmetry
- **Denser than contiguity**: More non-zero entries than Queen contiguity (counties can be "neighbors" without sharing a border)
- **Distance decay**: Closer counties receive higher weights

#### Loading

```python
W_dist = np.load('W_counties_distance.npy')
print(f"Shape: {W_dist.shape}")               # (500, 500)
print(f"Non-zero: {(W_dist > 0).sum()}")      # More than contiguity
print(f"Row sums: {W_dist.sum(axis=1)}")      # All approximately 1.0
```

#### Use Case

Compare Moran's I and LM test results under contiguity vs. distance-based weights to assess sensitivity of spatial diagnostics to weight matrix specification.

---

### W_eu_contiguity.npy

**Dimensions:** 200 x 200
**Type:** Queen contiguity weight matrix for EU NUTS-2 regions
**Format:** NumPy binary (`.npy`)

#### Properties

- **Construction**: Two regions are neighbors if they share a boundary (including cross-border neighbors)
- **Row standardization**: Each row sums to 1
- **Cross-border connections**: Regions in different countries can be neighbors (e.g., French region neighboring Belgian region)
- **Average neighbors**: ~4.8 per region
- **No islands**: All regions have at least one neighbor (island regions are connected to nearest mainland region)

#### Loading

```python
W_eu = np.load('W_eu_contiguity.npy')
print(f"Shape: {W_eu.shape}")  # (200, 200)
```

---

## Coordinate Files

### coordinates_counties.csv

**Dimensions:** 500 x 3
**Purpose:** County centroid coordinates for constructing custom distance-based weight matrices

| Variable | Description | Units |
|----------|-------------|-------|
| `county_id` | County identifier (matches `us_counties.csv`) | Integer |
| `latitude` | Centroid latitude | Decimal degrees |
| `longitude` | Centroid longitude | Decimal degrees |

### coordinates_eu.csv

**Dimensions:** 200 x 3
**Purpose:** EU region centroid coordinates for constructing custom distance-based weight matrices

| Variable | Description | Units |
|----------|-------------|-------|
| `region_id` | Region identifier (matches `eu_regions.csv`) | Integer |
| `latitude` | Centroid latitude | Decimal degrees |
| `longitude` | Centroid longitude | Decimal degrees |

#### Building Custom Weight Matrices from Coordinates

```python
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

coords = pd.read_csv('coordinates_counties.csv')
points = coords[['latitude', 'longitude']].values

# Haversine or Euclidean distance matrix
D = cdist(points, points, metric='euclidean')

# K-nearest-neighbors weight matrix (k=5)
W_knn = np.zeros_like(D)
for i in range(len(D)):
    neighbors = np.argsort(D[i])[1:6]  # 5 nearest (exclude self)
    W_knn[i, neighbors] = 1.0

# Row standardize
row_sums = W_knn.sum(axis=1, keepdims=True)
W_knn = W_knn / row_sums
```

---

## Weight Matrix Properties Summary

| Matrix | Dimensions | Type | Avg. Neighbors | Sparsity | Cross-border |
|--------|-----------|------|---------------|----------|-------------|
| `W_counties.npy` | 500 x 500 | Queen contiguity | ~5.5 | ~98% | N/A |
| `W_counties_distance.npy` | 500 x 500 | Inverse distance (150km) | ~12 | ~95% | N/A |
| `W_eu_contiguity.npy` | 200 x 200 | Queen contiguity | ~4.8 | ~95% | Yes |

All weight matrices are:
- **Row-standardized**: Each row sums to 1.0
- **Non-negative**: All entries >= 0
- **Zero diagonal**: No self-neighbors (w_{ii} = 0)
- **Connected**: No isolated units (every unit has at least one neighbor)

---

## Data Generation

All datasets and weight matrices can be regenerated using:

```python
from utils.data_generators import (
    generate_us_counties,
    generate_eu_regions,
    generate_contiguity_weights,
    generate_distance_weights,
)

# US Counties
df_counties, W_cont, W_dist, coords = generate_us_counties(
    n_counties=500, n_years=10, seed=42
)
df_counties.to_csv('us_counties.csv', index=False)
np.save('W_counties.npy', W_cont)
np.save('W_counties_distance.npy', W_dist)
coords.to_csv('coordinates_counties.csv', index=False)

# EU Regions
df_eu, W_eu, coords_eu = generate_eu_regions(
    n_regions=200, n_years=15, seed=42
)
df_eu.to_csv('eu_regions.csv', index=False)
np.save('W_eu_contiguity.npy', W_eu)
coords_eu.to_csv('coordinates_eu.csv', index=False)
```

All generation uses `np.random.seed(42)` for reproducibility.

---

## File Formats

### CSV Files
- **Encoding:** UTF-8
- **Separator:** Comma (`,`)
- **Header:** First row
- **Missing values:** None

### NPY Files
- **Format:** NumPy binary format
- **Loading:** `np.load('filename.npy')`
- **Dtype:** float64

---

## References

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
- Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics*, 26(1), 77-104.
- LeSage, J. P., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Getis, A. (2009). Spatial weights matrices. *Geographical Analysis*, 41(4), 404-410.
