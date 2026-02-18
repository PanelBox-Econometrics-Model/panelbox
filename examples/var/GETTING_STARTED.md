# Getting Started with Panel VAR Tutorials

## 1. Installation

Install PanelBox and its dependencies:

```bash
pip install panelbox
```

Verify installation:

```python
import panelbox
print(f"PanelBox version: {panelbox.__version__}")
```

## 2. Verify Setup

Run this quick check to ensure all required imports work:

```python
# Core imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# PanelBox VAR module
from panelbox.var import PanelVAR, PanelVARData

print("All imports successful!")
```

## 3. Generate Data

The tutorial datasets are provided as CSV files in the `data/` directory. If you need to regenerate them, use the data generators:

```python
import sys
sys.path.insert(0, 'utils')
from data_generators import generate_macro_panel

# Generate the primary dataset
df = generate_macro_panel()
print(f"Shape: {df.shape}")
print(df.head())
```

Alternatively, load directly from CSV:

```python
import pandas as pd
df = pd.read_csv('data/macro_panel.csv')
```

## 4. First Notebook

Open the first tutorial notebook:

```bash
jupyter notebook notebooks/01_var_introduction.ipynb
```

Or using JupyterLab:

```bash
jupyter lab notebooks/01_var_introduction.ipynb
```

## 5. Troubleshooting

### Import Errors

If you get `ModuleNotFoundError: No module named 'panelbox'`:

```bash
pip install --upgrade panelbox
```

### Missing Data Files

If CSV files are missing from `data/`, regenerate them:

```python
import sys
sys.path.insert(0, 'utils')
from data_generators import (
    generate_macro_panel,
    generate_energy_panel,
    generate_finance_panel,
    generate_monetary_policy_panel,
)

generate_macro_panel().to_csv('data/macro_panel.csv', index=False)
generate_energy_panel().to_csv('data/energy_panel.csv', index=False)
generate_finance_panel().to_csv('data/finance_panel.csv', index=False)
generate_monetary_policy_panel().to_csv('data/monetary_policy.csv', index=False)
```

### Matplotlib Backend Issues

If plots don't display in notebooks, add this to the first cell:

```python
%matplotlib inline
```

### Memory Issues

For large datasets (finance_panel with 5,000 rows), ensure you have at least 2 GB of available RAM.
