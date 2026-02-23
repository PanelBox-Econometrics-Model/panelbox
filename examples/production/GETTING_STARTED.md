# Getting Started with Production & Deployment Tutorials

## System Requirements

- Python 3.8+
- PanelBox installed (`pip install panelbox` or development mode)
- Jupyter Notebook or JupyterLab

## Verify Installation

Run this in a Python console or Jupyter cell to verify everything works:

```python
# 1. Check PanelBox imports
from panelbox.models.static.pooled_ols import PooledOLS
from panelbox.models.static.fixed_effects import FixedEffects
from panelbox.models.static.random_effects import RandomEffects
from panelbox.gmm import DifferenceGMM, SystemGMM
from panelbox.production import PanelPipeline, ModelValidator, ModelRegistry
from panelbox import load_model
print("All imports OK!")

# 2. Check datasets exist
import pandas as pd
from pathlib import Path

data_dir = Path('data')  # adjust path if needed
for f in ['firm_panel.csv', 'bank_lgd.csv', 'macro_quarterly.csv']:
    df = pd.read_csv(data_dir / f)
    print(f"{f}: {df.shape[0]} rows, {df.shape[1]} columns")

print("\nReady to start!")
```

## Quick Test: Predict with PooledOLS

```python
import pandas as pd
from panelbox.models.static.pooled_ols import PooledOLS

# Load data
df = pd.read_csv('data/firm_panel.csv')

# Fit model
model = PooledOLS('investment ~ value + capital + sales', df,
                   entity_col='firm_id', time_col='year')
results = model.fit()
print(results.summary())

# Predict on new data
new_df = pd.read_csv('data/new_firms.csv')
predictions = results.predict(new_df)
print(f"Predictions shape: {predictions.shape}")
print(f"First 5: {predictions[:5]}")
```

## Running the Tutorials

1. Navigate to `notebooks/` directory
2. Start with `01_predict_fundamentals.ipynb`
3. Follow the learning pathway in README.md

## Troubleshooting

**Import errors**: Make sure PanelBox is installed in your Python environment:
```bash
pip install -e .  # from the panelbox root directory
```

**File not found**: Ensure you're running notebooks from the `notebooks/` directory, or adjust the `BASE_DIR` path at the top of each notebook.

**Memory issues**: The datasets are small (max 3,000 rows), so memory should not be an issue.
