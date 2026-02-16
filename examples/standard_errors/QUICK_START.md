# Quick Start Guide - Standard Errors Tutorials

**For Developers and Content Creators**

---

## Directory Overview

```
standard_errors/
├── data/              # Add datasets here (CSV format)
├── notebooks/         # Add tutorial notebooks here (.ipynb)
├── outputs/           # Auto-generated (excluded from Git)
│   ├── figures/       # PNG/SVG plots by notebook (01-07)
│   └── reports/html/  # HTML reports
├── utils/             # Python utilities package
│   ├── plotting.py           # Visualization helpers
│   ├── diagnostics.py        # Diagnostic tests
│   ├── data_generators.py    # Synthetic data
│   └── __init__.py           # Package init
├── README.md                  # Main overview
├── CHANGELOG.md               # Version history
├── IMPLEMENTATION_STATUS.md   # Progress tracker
└── .gitignore                 # Git exclusions
```

---

## For Tutorial Developers

### Creating a New Notebook

1. **Location**: Save in `notebooks/`
2. **Naming**: Use `{number}_{short_name}.ipynb` (e.g., `01_robust_fundamentals.ipynb`)
3. **Template Structure**:

```python
# Cell 1: Title (Markdown)
"""
# {Number}. {Title}
**Author**: Your Name
**Date**: YYYY-MM-DD
**Duration**: ~XX minutes
**Prerequisites**: [List prerequisites]
"""

# Cell 2: Table of Contents (Markdown)
"""
## Contents
1. [Introduction](#introduction)
2. [Theory](#theory)
...
"""

# Cell 3: Learning Objectives (Markdown)
"""
## Learning Objectives
- Objective 1
- Objective 2
...
"""

# Cell 4: Setup (Code)
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# PanelBox imports
import panelbox as pb
from panelbox.models.static import FixedEffects, PooledOLS

# Local utilities
import sys
sys.path.append('../utils')
from plotting import plot_se_comparison
from diagnostics import test_heteroskedasticity

# Configuration
np.random.seed(42)
sns.set_style('whitegrid')
pd.set_option('display.precision', 4)

# Cell 5+: Content cells
# Alternate Markdown explanations with Code cells
```

### Data Loading Pattern

```python
# Define data path
DATA_PATH = '../data/'

# Load dataset
data = pd.read_csv(DATA_PATH + 'grunfeld.csv')

# Display info
print(f"Shape: {data.shape}")
print(f"Entities: {data['firm_id'].nunique()}")
print(f"Time periods: {data['year'].nunique()}")
data.head()
```

### Output Saving Pattern

```python
# Define paths
FIG_PATH = '../outputs/figures/01_robust/'

# Save plot
plt.savefig(FIG_PATH + 'plot_name.png', dpi=300, bbox_inches='tight')
```

### Important Rules

- ✅ Use **relative paths** only (`../data/`, not absolute paths)
- ✅ Set **random seeds** for reproducibility (`np.random.seed(42)`)
- ✅ Clear all **outputs** before committing to Git
- ✅ Test notebook **end-to-end** in fresh kernel before committing
- ✅ Include **exercises** at the end
- ✅ Add **references** in final cell
- ❌ Don't use absolute paths
- ❌ Don't commit notebooks with outputs
- ❌ Don't import from parent directories (use `utils/` package)

---

## For Utility Developers

### Adding a New Function

1. **Choose module**: `plotting.py`, `diagnostics.py`, or `data_generators.py`
2. **Write function with NumPy-style docstring**:

```python
def plot_se_comparison(results_dict, method_names=None):
    """
    Plot comparison of standard errors across methods.

    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys, PanelBox results as values
    method_names : list of str, optional
        Custom names for methods (default: use dict keys)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object

    Examples
    --------
    >>> results = {'OLS': res1, 'Robust': res2, 'Clustered': res3}
    >>> fig, ax = plot_se_comparison(results)
    >>> plt.show()

    Notes
    -----
    Uses seaborn color palette for consistency across tutorials.
    """
    # Implementation here
    pass
```

3. **Add to `__init__.py`** if needed
4. **Write unit test** (optional but recommended)
5. **Test in notebook** before committing

### Recommended Libraries

- **Plotting**: `matplotlib`, `seaborn`
- **Diagnostics**: `scipy.stats`, `statsmodels.stats.diagnostic`
- **Data generation**: `numpy`, `pandas`

---

## For Dataset Curators

### Adding a New Dataset

1. **Save to**: `data/`
2. **Format**: CSV with headers
3. **Required columns**:
   - Entity ID (e.g., `firm_id`, `country_id`)
   - Time ID (e.g., `year`, `month`)
   - Dependent variable
   - Independent variables
   - Spatial coordinates if applicable (`latitude`, `longitude`)

4. **Naming**: Use lowercase with underscores (e.g., `macro_growth.csv`)

5. **Create data dictionary** (in notebook or separate file):

```markdown
### Data Dictionary: `grunfeld.csv`

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| `firm_id` | int | Firm identifier | 1-10 |
| `year` | int | Year | 1935-1954 |
| `invest` | float | Gross investment | >0 |
| `value` | float | Market value | >0 |
| `capital` | float | Capital stock | >0 |

**Source**: Grunfeld (1958)
**N**: 10 firms, **T**: 20 years
```

### Data Quality Checklist

- [ ] No missing values (or explicitly handled)
- [ ] Reasonable ranges (no outliers unless intentional)
- [ ] Proper data types (int for IDs, float for continuous)
- [ ] Balanced panel (or document if unbalanced)
- [ ] Entity and time IDs are clearly labeled

---

## Git Workflow

### Before Committing Notebooks

```bash
# 1. Test notebook end-to-end
# In Jupyter: Kernel -> Restart & Run All

# 2. Clear all outputs
# In Jupyter: Kernel -> Restart & Clear Output

# 3. Save notebook
# Ctrl+S or Cmd+S

# 4. Check git status
cd /path/to/panelbox/examples/standard_errors
git status

# 5. Add files
git add notebooks/01_robust_fundamentals.ipynb
git add data/grunfeld.csv

# 6. Commit with descriptive message
git commit -m "Add tutorial 01: Robust fundamentals with Grunfeld data"

# 7. Push to remote
git push origin feature/notebook-01
```

### Excluded from Git (automatic via `.gitignore`)

- Output figures: `outputs/figures/**/*.png`
- HTML reports: `outputs/reports/**/*.html`
- Jupyter checkpoints: `.ipynb_checkpoints/`
- Python cache: `__pycache__/`

---

## Testing Checklist

### Before Releasing a Notebook

- [ ] All cells execute without errors (Restart & Run All)
- [ ] Random seeds are set (`np.random.seed(42)`)
- [ ] File paths are relative (no absolute paths)
- [ ] Datasets are in `../data/`
- [ ] Plots use consistent style (seaborn theme)
- [ ] Code follows PEP 8 style
- [ ] All sections have explanatory Markdown
- [ ] Exercises are included
- [ ] References are complete
- [ ] Estimated duration is accurate
- [ ] All outputs are cleared before committing
- [ ] Tested in fresh environment (if possible)

---

## Common Commands

### Navigate to tutorial directory

```bash
cd /home/guhaase/projetos/panelbox/examples/standard_errors
```

### Launch Jupyter Notebook

```bash
cd notebooks/
jupyter notebook
```

### Check structure

```bash
tree -L 2
```

### Count files

```bash
find . -type f | wc -l
```

### Search for absolute paths in notebooks (should return nothing)

```bash
grep -r "/home/" notebooks/
```

---

## Learning Path

### For Students

1. **Beginner**: 01 → 02 → 07
2. **Intermediate**: 01 → 02 → 03 → 07
3. **Advanced**: 01 → 02 → 03 → 05 → 04 → 06 → 07

### For Developers

1. Read `README.md` (overview)
2. Read `IMPLEMENTATION_STATUS.md` (current progress)
3. Check which notebooks/utilities are pending
4. Pick a task and start coding
5. Follow template and checklist above
6. Test thoroughly
7. Submit for review

---

## Dependencies

### Required

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
panelbox>=0.8.0
```

### Optional

```python
statsmodels>=0.13.0  # For some diagnostic tests
jupyter>=1.0.0       # For running notebooks
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn scipy panelbox
pip install statsmodels jupyter  # Optional
```

---

## Getting Help

### Documentation

- **PanelBox Docs**: [https://panelbox.readthedocs.io](https://panelbox.readthedocs.io)
- **Main README**: `README.md`
- **Implementation Status**: `IMPLEMENTATION_STATUS.md`
- **Changelog**: `CHANGELOG.md`

### Issues

- **GitHub Issues**: For bugs, feature requests
- **GitHub Discussions**: For questions, sharing applications

### Contact

- **Email**: [your-email@example.com]
- **GitHub**: [@yourusername]

---

## Version Info

- **Structure Version**: 1.0.0
- **Date Created**: 2026-02-16
- **PanelBox Version**: 0.8.0+
- **Python Version**: 3.9+

---

## Quick Reference: Notebook Numbers

| # | Notebook | Topic | Difficulty |
|---|----------|-------|------------|
| 01 | `robust_fundamentals.ipynb` | HC0-HC3 robust SEs | Beginner |
| 02 | `clustering_panels.ipynb` | Clustering | Intermediate |
| 03 | `hac_autocorrelation.ipynb` | HAC estimators | Intermediate |
| 04 | `spatial_errors.ipynb` | Spatial correlation | Advanced |
| 05 | `mle_inference.ipynb` | MLE sandwich | Advanced |
| 06 | `bootstrap_quantile.ipynb` | Bootstrap | Advanced |
| 07 | `methods_comparison.ipynb` | Comparison | Intermediate |

---

**Ready to contribute?** Pick a task from `IMPLEMENTATION_STATUS.md` and get started! 🚀
