# Panel Data Diagnostics Tutorial Series

**Version:** 1.0.0
**Last Updated:** 2026-02-22

## Overview

This tutorial series provides a comprehensive introduction to diagnostic testing for panel data models using PanelBox. Diagnostics are essential for validating model assumptions, detecting data pathologies, and ensuring reliable inference in applied panel data research.

| # | Notebook | Level | Duration | Topics |
|---|----------|-------|----------|--------|
| 01 | Unit Root Tests | Intermediate | 90-120 min | LLC, IPS, Breitung, Hadri, CIPS tests for panel stationarity |
| 02 | Cointegration Tests | Intermediate-Advanced | 90-120 min | Pedroni, Kao, Westerlund tests for panel cointegration |
| 03 | Specification Tests | Intermediate | 75-90 min | Hausman, Mundlak, Breusch-Pagan, serial correlation, cross-sectional dependence |
| 04 | Spatial Diagnostics | Advanced | 90-120 min | Moran's I, LM tests, spatial weights validation, cross-sectional dependence |

## Why Panel Diagnostics?

Reliable panel data inference requires careful validation of modeling assumptions:

- **Unit root tests** determine the order of integration and whether variables need differencing
- **Cointegration tests** detect long-run equilibrium relationships among non-stationary variables
- **Specification tests** verify the choice between fixed and random effects, check for heteroskedasticity, and detect serial correlation
- **Spatial diagnostics** assess the presence and form of spatial dependence across cross-sectional units

Ignoring these diagnostics can lead to spurious regression, inconsistent estimators, and invalid inference.

## Prerequisites

### Required Knowledge

- Basic statistics (regression, hypothesis testing, time series concepts)
- Understanding of panel data structure and fixed/random effects
- Python programming fundamentals
- Pandas and NumPy basics

### Software Requirements

- Python 3.8+
- PanelBox 0.7.0+
- pandas, numpy, scipy, matplotlib, seaborn
- Jupyter Notebook or JupyterLab

Install dependencies:
```bash
pip install panelbox pandas numpy scipy matplotlib seaborn jupyter
```

Or with conda:
```bash
conda install -c conda-forge panelbox pandas numpy scipy matplotlib seaborn jupyter
```

## Learning Path

**Recommended order:** 01 --> 02 --> 03 --> 04

- Notebooks 01 and 02 should be completed sequentially (cointegration builds on unit root concepts)
- Notebooks 03 and 04 are independent of each other and can be completed in any order after 01
- Notebook 04 (Spatial) is standalone but benefits from understanding cross-sectional dependence in 03

```
01 Unit Root Tests
      |
      v
02 Cointegration Tests
      |
      +------+------+
      |             |
      v             v
03 Specification  04 Spatial
   Tests          Diagnostics
```

## Quick Start

1. **Navigate to the directory:**
   ```bash
   cd examples/diagnostics
   ```

2. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Open the first tutorial:**
   - Navigate to `notebooks/01_unit_root_tests.ipynb`
   - Follow the instructions in the notebook

4. **Check your setup:**
   ```python
   import panelbox
   import pandas as pd
   from pathlib import Path

   # Verify PanelBox version
   print(f"PanelBox version: {panelbox.__version__}")

   # Test data access
   data_path = Path('data/unit_root/penn_world_table.csv')
   df = pd.read_csv(data_path)
   print(f"Data loaded: {df.shape}")
   ```

## Folder Structure

```
diagnostics/
├── README.md                          # This file
├── notebooks/                         # Tutorial notebooks
│   ├── README.md
│   ├── 01_unit_root_tests.ipynb
│   ├── 02_cointegration_tests.ipynb
│   ├── 03_specification_tests.ipynb
│   └── 04_spatial_diagnostics.ipynb
├── solutions/                         # Exercise solutions
│   ├── README.md
│   ├── 01_unit_root_tests_solutions.ipynb
│   ├── 02_cointegration_tests_solutions.ipynb
│   ├── 03_specification_tests_solutions.ipynb
│   └── 04_spatial_diagnostics_solutions.ipynb
├── data/                              # Tutorial datasets
│   ├── README.md
│   ├── unit_root/                     # Datasets for notebook 01
│   │   ├── README.md
│   │   ├── penn_world_table.csv
│   │   └── prices_panel.csv
│   ├── cointegration/                 # Datasets for notebook 02
│   │   ├── README.md
│   │   ├── oecd_macro.csv
│   │   ├── ppp_data.csv
│   │   └── interest_rates.csv
│   ├── specification/                 # Datasets for notebook 03
│   │   ├── README.md
│   │   ├── nlswork.csv
│   │   ├── firm_productivity.csv
│   │   └── trade_panel.csv
│   └── spatial/                       # Datasets for notebook 04
│       ├── README.md
│       ├── us_counties.csv
│       ├── W_counties.npy
│       ├── W_counties_distance.npy
│       ├── eu_regions.csv
│       ├── W_eu_contiguity.npy
│       └── coordinates_*.csv
├── outputs/                           # Generated outputs
│   ├── README.md
│   ├── figures/                       # Plots and visualizations
│   ├── tables/                        # Formatted result tables
│   └── results/                       # Saved test results
└── utils/                             # Helper functions
    ├── data_generators.py
    ├── plot_helpers.py
    └── test_helpers.py
```

## Datasets

All datasets are simulated based on real-world stylized facts from published research. They are generated with fixed random seeds (`np.random.seed(42)`) for full reproducibility.

| Subdirectory | Dataset | Dimensions | Purpose |
|-------------|---------|------------|---------|
| unit_root/ | penn_world_table.csv | 30 x 50 | GDP, investment, consumption with I(1) behavior |
| unit_root/ | prices_panel.csv | 40 x 30 | Regional price indices with I(1) dynamics |
| cointegration/ | oecd_macro.csv | 20 x 40 | Consumption-income cointegrating relationship |
| cointegration/ | ppp_data.csv | 25 x 35 | Purchasing power parity long-run equilibrium |
| cointegration/ | interest_rates.csv | 15 x 30 | Interest rate parity across countries |
| specification/ | nlswork.csv | 4000 x 15 | Wage equation with correlated unobserved ability |
| specification/ | firm_productivity.csv | 200 x 20 | Cobb-Douglas production function |
| specification/ | trade_panel.csv | 300 x 15 | Gravity model for bilateral trade |
| spatial/ | us_counties.csv | 500 x 10 | County-level economic indicators |
| spatial/ | eu_regions.csv | 200 x 15 | EU NUTS-2 regional data |

See `data/README.md` and individual subdirectory READMEs for complete documentation.

## Key References

### Unit Root Tests

- Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). Unit root tests in panel data: Asymptotic and finite-sample properties. *Journal of Econometrics*, 108(1), 1-24.
- Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in heterogeneous panels. *Journal of Econometrics*, 115(1), 53-74.
- Breitung, J. (2000). The local power of some unit root tests for panel data. In B. Baltagi (Ed.), *Advances in Econometrics* (Vol. 15, pp. 161-178).
- Hadri, K. (2000). Testing for stationarity in heterogeneous panel data. *Econometrics Journal*, 3(2), 148-161.
- Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. *Journal of Applied Econometrics*, 22(2), 265-312.

### Cointegration Tests

- Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous panels with multiple regressors. *Oxford Bulletin of Economics and Statistics*, 61(S1), 653-670.
- Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. *Journal of Econometrics*, 90(1), 1-44.
- Westerlund, J. (2007). Testing for error correction in panel data. *Oxford Bulletin of Economics and Statistics*, 69(6), 709-748.

### Specification Tests

- Hausman, J. A. (1978). Specification tests in econometrics. *Econometrica*, 46(6), 1251-1271.
- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69-85.
- Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. *Review of Economic Studies*, 47(1), 239-253.
- Pesaran, M. H. (2004). General diagnostic tests for cross section dependence in panels. CESifo Working Paper No. 1229.

### Spatial Diagnostics

- Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer Academic Publishers.
- Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. *Regional Science and Urban Economics*, 26(1), 77-104.
- Baltagi, B. H., Song, S. H., & Koh, W. (2003). Testing panel data regression models with spatial error correlation. *Journal of Econometrics*, 117(1), 123-150.

### Textbooks

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press.

## Common Issues and Troubleshooting

### Installation Issues

**Problem:** `ModuleNotFoundError: No module named 'panelbox'`
**Solution:** Install PanelBox: `pip install panelbox`

**Problem:** Old version of PanelBox
**Solution:** Upgrade: `pip install --upgrade panelbox`

### Data Loading Issues

**Problem:** `FileNotFoundError` when loading data
**Solution:** Check your working directory and use relative paths:
```python
from pathlib import Path
data_path = Path(__file__).parent / 'data' / 'unit_root' / 'penn_world_table.csv'
```

### Runtime Issues

**Problem:** Slow unit root or cointegration tests on large panels
**Solution:** Reduce lag order or use asymptotic critical values instead of bootstrap

**Problem:** Singular weight matrix in spatial diagnostics
**Solution:** Check for islands (units with no neighbors) and handle appropriately

## Getting Help

### Within Tutorials
- Each notebook has extensive comments and explanations
- Exercise solutions are in the `solutions/` directory

### External Resources
- **PanelBox Documentation:** https://panelbox.readthedocs.io
- **Issue Tracker:** https://github.com/panelbox/panelbox/issues

## License

These tutorials are released under the MIT License, same as PanelBox.

## Citation

If you use these tutorials in academic work, please cite:

```bibtex
@misc{panelbox_diagnostics_tutorials,
  title={PanelBox Panel Data Diagnostics Tutorials},
  author={PanelBox Development Team},
  year={2026},
  howpublished={\url{https://github.com/panelbox/panelbox}},
  note={Version 1.0.0}
}
```

---

**Ready to start?** Open `notebooks/01_unit_root_tests.ipynb` and begin with panel unit root testing.
