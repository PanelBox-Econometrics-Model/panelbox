# Tutorial Utility Scripts

This directory contains helper scripts for the PanelBox tutorials.

## Available Scripts

### setup_environment.py
Verifies that your Python environment has all required packages for the tutorials.

**Usage:**
```bash
cd scripts
python setup_environment.py
```

**Checks:**
- Python version (3.8+)
- NumPy, Pandas, SciPy
- Matplotlib, Seaborn
- Statsmodels
- PanelBox
- Jupyter (optional but recommended)

### utils.py
Reusable utility functions for tutorial notebooks.

**Usage in notebooks:**
```python
import sys
sys.path.append('../../scripts')
from utils import plot_panel_structure, summary_stats
```

**Available functions:**
- `plot_panel_structure()`: Visualize entity × time coverage
- `summary_stats()`: Generate comprehensive summary statistics
- `plot_residual_diagnostics()`: Create diagnostic plots
- `compare_models()`: Compare fit statistics across models
- `export_results_table()`: Export results to LaTeX/Markdown/HTML
- `validate_panel_data()`: Check panel structure validity
- `set_tutorial_style()`: Apply consistent plotting style

**Examples:**
```python
# Visualize panel structure
plot_panel_structure(df, entity_col='firm', time_col='year')

# Get summary statistics
stats = summary_stats(df, variables=['invest', 'value', 'capital'])

# Validate panel data
report = validate_panel_data(df, 'firm', 'year')
print(report['summary'])

# Compare models
comparison = compare_models(model1, model2, model3,
                           model_names=['Pooled', 'FE', 'RE'])
```

## Future Scripts

### download_datasets.py (Coming Soon)
Script to download external datasets that are too large to include in the repository.

---

## Contributing

To add new utility functions:
1. Add function to `utils.py` with complete docstring
2. Include usage example in docstring
3. Update this README with description
4. Test function in at least one tutorial notebook

---

**Last Updated**: 2026-02-16
