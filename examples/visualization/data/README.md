# Data Directory — Visualization Tutorial Series

This directory holds any external datasets used by the Visualization notebooks.

---

## Primary Data Sources

Most notebooks in this series use **no external CSV files** — data comes from:

1. **PanelBox built-in datasets**:

```python
import panelbox as pb

# Grunfeld Investment Data (1935-1954)
data = pb.datasets.load_grunfeld()

# Arellano-Bond Employment Data
data = pb.datasets.load_abdata()
```

2. **Synthetic data generators** (`utils/data_generators.py`):

```python
from visualization.utils.data_generators import (
    generate_panel_data,
    generate_heteroskedastic_panel,
    generate_autocorrelated_panel,
    generate_spatial_panel,
)

# Basic balanced panel
df = generate_panel_data(n_individuals=200, n_periods=10, seed=42)

# Panel with heteroskedasticity (used in Notebook 02)
df = generate_heteroskedastic_panel(n_individuals=200, n_periods=10, seed=42)
```

---

## External Datasets (if present)

| File | Description | Notebooks | Source |
|---|---|---|---|
| *(none yet)* | — | — | — |

Large CSV files (> 1 MB) are listed in `.gitignore` and must be downloaded separately.
See individual notebook cells for download instructions if applicable.

---

## Loading Pattern Used in Notebooks

```python
from pathlib import Path
import pandas as pd

DATA_DIR = Path("..") / "data"

# Example (if external CSV is needed):
df = pd.read_csv(DATA_DIR / "some_dataset.csv")
```

---

## Adding New Datasets

1. Place the CSV file in this directory.
2. Add a row to the table above.
3. If the file is > 1 MB, add it to `.gitignore` and document how to obtain it.
4. Update the relevant notebook's data-loading cell.
