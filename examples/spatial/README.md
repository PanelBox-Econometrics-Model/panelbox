# Spatial Econometrics Tutorials

This directory contains comprehensive tutorials on spatial econometrics using the PanelBox library.

## Overview

Spatial econometrics extends traditional panel data analysis by accounting for spatial dependencies and spillover effects across geographic units. These tutorials cover the theoretical foundations, implementation details, and practical applications of spatial panel models.

## Learning Path

The tutorials are designed to be followed sequentially:

1. **Introduction to Spatial Econometrics** - Concepts, motivation, and spatial autocorrelation
2. **Spatial Weights Matrices** - Construction and properties of W matrices
3. **Spatial Lag Model (SAR)** - Modeling spatial dependence in the outcome
4. **Spatial Error Model (SEM)** - Modeling spatial dependence in disturbances
5. **Spatial Durbin Model (SDM)** - Including spatially lagged explanatory variables
6. **Spatial Marginal Effects** - Direct, indirect, and total effects decomposition
7. **Dynamic Spatial Panels** - Combining temporal and spatial dynamics
8. **Specification Tests** - Model selection and diagnostic testing

## Prerequisites

### Statistical Knowledge
- Panel data econometrics
- Maximum likelihood estimation
- Basic spatial statistics concepts
- Hypothesis testing

### Python Skills
- Intermediate Python programming
- NumPy and Pandas for data manipulation
- Experience with Jupyter notebooks
- Basic visualization with Matplotlib

### Required Libraries

```python
# Core dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Spatial analysis
import geopandas as gpd
from libpysal import weights
from esda import Moran, Moran_Local

# PanelBox models
from panelbox.models.spatial import (
    SpatialLag,
    SpatialError,
    SpatialDurbin,
    DynamicSpatialPanel,
    GeneralNestingSpatial
)
from panelbox.effects import compute_spatial_effects
from panelbox.diagnostics import spatial_lm_tests
```

## Directory Structure

```
spatial/
├── README.md                          # This file
├── data/                              # Datasets for tutorials
│   ├── us_counties/                   # US counties socioeconomic data
│   ├── brazil_municipalities/         # Brazilian municipal data
│   ├── european_nuts2/                # European regional data
│   ├── housing_prices/                # Housing market data
│   └── agriculture/                   # Agricultural productivity data
├── notebooks/                         # Tutorial Jupyter notebooks
│   ├── 01_intro_spatial_econometrics.ipynb
│   ├── 02_spatial_weights_matrices.ipynb
│   ├── 03_spatial_lag_model.ipynb
│   ├── 04_spatial_error_model.ipynb
│   ├── 05_spatial_durbin_model.ipynb
│   ├── 06_spatial_marginal_effects.ipynb
│   ├── 07_dynamic_spatial_panels.ipynb
│   └── 08_specification_tests.ipynb
├── scripts/                           # Reusable utility scripts
│   ├── data_preparation.py           # Data loading and cleaning
│   ├── weight_matrix_builder.py      # W matrix construction
│   └── visualization_utils.py        # Plotting functions
└── outputs/                           # Generated outputs
    ├── figures/                       # Plots and maps
    ├── tables/                        # Results tables
    └── reports/                       # HTML/PDF reports
```

## Installation

Ensure you have PanelBox installed:

```bash
pip install panelbox

# Or for development version
git clone https://github.com/yourusername/panelbox.git
cd panelbox
pip install -e .
```

Additional spatial dependencies:

```bash
pip install geopandas libpysal esda splot
```

## Datasets

### Available Datasets

Each dataset includes:
- CSV file with panel data
- Shapefile for geographic visualization (when applicable)
- README with variable descriptions and sources

See individual data subdirectories for detailed documentation:
- `data/us_counties/README.md`
- `data/brazil_municipalities/README.md`
- `data/european_nuts2/README.md`
- `data/housing_prices/README.md`
- `data/agriculture/README.md`

### Data Sources

- US data: Census Bureau, BEA
- Brazilian data: IBGE, IPEA
- European data: Eurostat
- Housing data: Public real estate listings
- Agricultural data: FAO, national agricultural surveys

## Key Concepts Covered

### Spatial Dependence
- Spatial autocorrelation (Moran's I, Geary's C)
- Spatial lag vs spatial error processes
- Spillover effects and feedback loops

### Spatial Weights Matrices
- Contiguity-based (Queen, Rook)
- Distance-based (threshold, k-nearest neighbors)
- Economic distance (trade flows, migration)
- Row-standardization and normalization

### Spatial Panel Models
- **SAR (Spatial Autoregressive)**: y = ρWy + Xβ + ε
- **SEM (Spatial Error Model)**: y = Xβ + u, u = λWu + ε
- **SDM (Spatial Durbin Model)**: y = ρWy + Xβ + WXθ + ε
- **SDEM (Spatial Durbin Error Model)**: y = Xβ + WXθ + u, u = λWu + ε
- **GNS (General Nesting Spatial)**: y = ρWy + Xβ + WXθ + u, u = λWu + ε

### Estimation Methods
- Maximum likelihood (ML)
- Quasi-maximum likelihood (QML)
- Generalized method of moments (GMM)
- Spatial two-stage least squares (S2SLS)

### Effects Decomposition
- **Direct effects**: Impact of Xi on yi
- **Indirect effects**: Impact of Xi on yj (i ≠ j)
- **Total effects**: Direct + Indirect
- Standard errors via delta method or simulation

### Diagnostic Tests
- Lagrange Multiplier (LM) tests
- Robust LM tests
- Likelihood ratio tests
- Wald tests
- Spatial Hausman test

## Usage Example

```python
from pathlib import Path
import pandas as pd
from panelbox.models.spatial import SpatialLag
from spatial.scripts.weight_matrix_builder import build_contiguity_matrix

# Load data
data_path = Path("data/us_counties/us_counties.csv")
df = pd.read_csv(data_path)

# Build spatial weights matrix
W = build_contiguity_matrix(
    shapefile="data/us_counties/us_counties.shp",
    id_variable="county_fips"
)

# Estimate spatial lag model
model = SpatialLag(
    data=df,
    dependent="log_income",
    exog=["education", "unemployment", "population_density"],
    entity_id="county_fips",
    time_id="year",
    W=W
)

results = model.fit()
print(results.summary())

# Compute marginal effects
from panelbox.effects import compute_spatial_effects
effects = compute_spatial_effects(results, W)
print(effects)
```

## Applications Covered

1. **Regional Economics**
   - Income convergence across regions
   - Knowledge spillovers and innovation
   - Regional unemployment dynamics

2. **Urban Economics**
   - Housing price spillovers
   - Neighborhood effects
   - Urban sprawl and land use

3. **Agricultural Economics**
   - Technology diffusion
   - Productivity spillovers
   - Pest and disease spread

4. **Environmental Economics**
   - Pollution spillovers
   - Conservation policies
   - Climate adaptation

5. **Development Economics**
   - Growth spillovers
   - Infrastructure effects
   - Institutional diffusion

## References

### Books
- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Elhorst, J. P. (2014). *Spatial Econometrics: From Cross-Sectional Data to Spatial Panels*. Springer.

### Papers
- Anselin, L., Le Gallo, J., & Jayet, H. (2008). Spatial panel econometrics. In *The Econometrics of Panel Data* (pp. 625-660).
- Elhorst, J. P. (2010). Applied spatial econometrics: Raising the bar. *Spatial Economic Analysis*, 5(1), 9-28.
- LeSage, J. P., & Pace, R. K. (2014). The biggest myth in spatial econometrics. *Econometrics*, 2(4), 217-249.

### Software Documentation
- PanelBox: https://panelbox.readthedocs.io/
- PySAL: https://pysal.org/
- GeoPandas: https://geopandas.org/

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add your tutorial or enhancement
4. Submit a pull request

## License

These tutorials are part of the PanelBox project and are distributed under the same license.

## Support

For questions or issues:
- Open an issue on GitHub
- Check PanelBox documentation
- Consult the PySAL community forums

## Acknowledgments

These tutorials were developed to support applied researchers in economics, geography, regional science, and related fields. We thank the PySAL and spatial econometrics communities for their foundational work.

---

**Last Updated**: 2026-02-16
**Version**: 1.0.0
