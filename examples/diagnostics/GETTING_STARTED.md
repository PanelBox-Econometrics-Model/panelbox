# Getting Started with Diagnostics Tutorials

**Version:** 1.0.0
**Last Updated:** 2026-02-22

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Verifying Your Setup](#verifying-your-setup)
4. [Running Your First Tutorial](#running-your-first-tutorial)
5. [Understanding the Structure](#understanding-the-structure)
6. [Common Issues](#common-issues)
7. [Next Steps](#next-steps)

## System Requirements

### Software

- Python: 3.8 or higher (3.9+ recommended)
- Operating System: Windows, macOS, or Linux
- RAM: Minimum 4GB (8GB+ recommended for spatial tests)
- Disk Space: ~200MB for environment and data

### Knowledge Prerequisites

- Basic Python programming (pandas, NumPy)
- Understanding of panel data concepts (entity, time dimensions)
- Familiarity with hypothesis testing (p-values, significance levels)
- Experience with Jupyter Notebooks

## Installation

### Step 1: Set Up Python Environment

#### Using venv (built-in)

```bash
python -m venv panelbox_env
# Activate (Windows)
panelbox_env\Scripts\activate
# Activate (macOS/Linux)
source panelbox_env/bin/activate
```

#### Using conda

```bash
conda create -n panelbox_env python=3.10
conda activate panelbox_env
```

### Step 2: Install Required Packages

#### Option A: Install from PyPI (recommended)

```bash
pip install panelbox pandas numpy matplotlib seaborn jupyter scipy statsmodels
```

#### Option B: Install from source (development)

```bash
cd /path/to/panelbox
pip install -e .
pip install pandas numpy matplotlib seaborn jupyter scipy statsmodels
```

## Verifying Your Setup

### Quick Check

Open a Python console and run:

```python
from panelbox.examples import diagnostics
result = diagnostics.verify_installation()
```

### Manual Check

```python
# Test core imports
import panelbox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Test diagnostic imports
from panelbox.diagnostics.unit_root import hadri_test, breitung_test
from panelbox.diagnostics.cointegration import pedroni_test, kao_test
from panelbox.diagnostics import hausman_test
from panelbox.diagnostics.spatial_tests import lm_lag_test, MoranIPanelTest

print("All imports successful!")
```

## Running Your First Tutorial

1. Navigate to the tutorials directory:

```bash
cd examples/diagnostics/notebooks/
```

2. Launch Jupyter:

```bash
jupyter notebook
```

3. Open `01_unit_root_tests.ipynb` and run the first cell to verify setup.

## Understanding the Structure

```
diagnostics/
├── notebooks/     # Tutorial notebooks (01-04)
├── solutions/     # Exercise solutions
├── data/          # Datasets by test type
│   ├── unit_root/
│   ├── cointegration/
│   ├── specification/
│   └── spatial/
├── utils/         # Helper functions
├── outputs/       # Generated figures and tables
└── tests/         # Data integrity tests
```

### Recommended Learning Order

| Order | Notebook | Duration | Topics |
|-------|----------|----------|--------|
| 1 | 01_unit_root_tests | 90 min | IPS, LLC, Breitung, Hadri |
| 2 | 02_cointegration_tests | 110 min | Pedroni, Westerlund, Kao |
| 3 | 03_specification_tests | 110 min | Hausman, J-test, encompassing |
| 4 | 04_spatial_tests | 120 min | LM tests, Moran's I, LISA |

Notebooks 01 and 02 are sequential (cointegration builds on unit roots).
Notebooks 03 and 04 can be done independently after 01.

## Common Issues

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'panelbox'`

**Solution:** Ensure panelbox is installed in your active environment:

```bash
pip install panelbox
# or for development:
pip install -e /path/to/panelbox
```

### Data Loading Errors

**Problem:** `FileNotFoundError` when loading datasets

**Solution:** Ensure you're running notebooks from the `notebooks/` directory.
The notebooks use relative paths like `../data/unit_root/`.

### Spatial Tests Slow

**Problem:** Westerlund bootstrap or LISA permutations take too long

**Solution:** Reduce `n_bootstrap` or `permutations` parameter:

```python
# Use fewer bootstrap replications
result = westerlund_test(..., n_bootstrap=100)
```

## Next Steps

After completing the tutorials:

1. Apply diagnostic tests to your own panel data
2. Explore the PanelBox documentation for additional tests
3. Check the `solutions/` folder to compare your exercise answers
4. Try combining diagnostics in a full workflow

---

**Ready to start?** Open `notebooks/01_unit_root_tests.ipynb`!
