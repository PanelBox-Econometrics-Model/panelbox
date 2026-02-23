# Diagnostics Tutorial Notebooks

**Version:** 1.0.0
**Last Updated:** 2026-02-22

## Overview

This directory contains the 4 tutorial notebooks for the Panel Data Diagnostics series. Each notebook is self-contained with explanations, worked examples, and exercises.

## Notebook Listing

| # | Notebook | Level | Duration | Description |
|---|----------|-------|----------|-------------|
| 01 | `01_unit_root_tests.ipynb` | Intermediate | 90-120 min | Panel unit root tests for stationarity assessment |
| 02 | `02_cointegration_tests.ipynb` | Intermediate-Advanced | 90-120 min | Panel cointegration tests for long-run relationships |
| 03 | `03_specification_tests.ipynb` | Intermediate | 75-90 min | Model specification and misspecification diagnostics |
| 04 | `04_spatial_diagnostics.ipynb` | Advanced | 90-120 min | Spatial dependence detection and weight matrix validation |

## Recommended Sequence

**Sequential path (01 then 02):**
1. Start with `01_unit_root_tests.ipynb` -- establishes stationarity concepts for panel data
2. Continue with `02_cointegration_tests.ipynb` -- builds directly on unit root results

**Independent path (03 and 04):**
3. `03_specification_tests.ipynb` -- can be completed independently after 01
4. `04_spatial_diagnostics.ipynb` -- can be completed independently after 01

## Notebook Descriptions

### 01 - Unit Root Tests

Tests whether panel data variables are stationary (I(0)) or contain a unit root (I(1)). Covers first-generation tests (LLC, IPS, Breitung, Hadri) that assume cross-sectional independence, and second-generation tests (Pesaran CIPS) that allow for cross-sectional dependence. Uses macroeconomic (Penn World Table) and regional price panel data.

**Key skills:** Stationarity assessment, test selection based on panel characteristics, handling cross-sectional dependence, power vs. size trade-offs.

**Datasets:** `data/unit_root/penn_world_table.csv`, `data/unit_root/prices_panel.csv`

### 02 - Cointegration Tests

Tests for long-run equilibrium relationships among non-stationary I(1) panel variables. Covers residual-based tests (Pedroni, Kao) and error-correction-based tests (Westerlund). Demonstrates applications to consumption-income, purchasing power parity, and interest rate parity.

**Key skills:** Cointegrating regression, residual-based vs. ECM-based testing, handling heterogeneous cointegrating vectors, spurious regression detection.

**Datasets:** `data/cointegration/oecd_macro.csv`, `data/cointegration/ppp_data.csv`, `data/cointegration/interest_rates.csv`

### 03 - Specification Tests

Diagnostic tests for choosing between model specifications and detecting violations of key assumptions. Covers the Hausman test (FE vs. RE), Mundlak formulation, Breusch-Pagan LM test for random effects, Wooldridge test for serial correlation, and Pesaran CD test for cross-sectional dependence.

**Key skills:** FE vs. RE selection, detecting heteroskedasticity, serial correlation testing, robust standard error selection, cross-sectional dependence assessment.

**Datasets:** `data/specification/nlswork.csv`, `data/specification/firm_productivity.csv`, `data/specification/trade_panel.csv`

### 04 - Spatial Diagnostics

Diagnostics for detecting and characterizing spatial dependence in panel data. Covers Moran's I statistic, LM tests for spatial lag vs. spatial error, robust LM tests, spatial weight matrix construction and validation, and panel-specific spatial dependence tests.

**Key skills:** Spatial weight matrix construction, spatial autocorrelation testing, distinguishing spatial lag from spatial error, weight matrix specification testing.

**Datasets:** `data/spatial/us_counties.csv`, `data/spatial/W_counties.npy`, `data/spatial/W_counties_distance.npy`, `data/spatial/eu_regions.csv`, `data/spatial/W_eu_contiguity.npy`

## Running Notebooks

### Option 1: Jupyter Notebook

```bash
cd examples/diagnostics
jupyter notebook notebooks/
```

### Option 2: JupyterLab

```bash
cd examples/diagnostics
jupyter lab notebooks/
```

### Option 3: Command-line Execution

```bash
jupyter nbconvert --to notebook --execute notebooks/01_unit_root_tests.ipynb
```

## Notebook Structure

Each notebook follows a consistent structure:

1. **Introduction** -- Motivation, theory review, and learning objectives
2. **Setup** -- Imports and data loading
3. **Theory** -- Mathematical background with intuitive explanations
4. **Implementation** -- Step-by-step worked examples using PanelBox
5. **Interpretation** -- How to read and report results
6. **Exercises** -- Hands-on problems (solutions in `../solutions/`)
7. **Summary** -- Key takeaways and further reading

## Tips

- Run cells sequentially from top to bottom
- Complete the exercises before consulting the solutions directory
- Each notebook generates outputs saved to `../outputs/`
- If you encounter errors, restart the kernel and re-run from the top
