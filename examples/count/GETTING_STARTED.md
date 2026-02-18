# Getting Started with Count Models Tutorials

**Version:** 1.0.0
**Last Updated:** 2026-02-16

This guide will help you get started with the PanelBox count models tutorials. Follow these steps to ensure your environment is properly configured.

---

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Verifying Your Setup](#verifying-your-setup)
4. [Running Your First Tutorial](#running-your-first-tutorial)
5. [Understanding the Structure](#understanding-the-structure)
6. [Common Issues](#common-issues)
7. [Next Steps](#next-steps)

---

## System Requirements

### Software
- **Python:** 3.8 or higher (3.9+ recommended)
- **Operating System:** Windows, macOS, or Linux
- **RAM:** Minimum 4GB (8GB+ recommended)
- **Disk Space:** ~500MB for environment and data

### Knowledge Prerequisites
- Basic Python programming
- Familiarity with pandas and NumPy
- Understanding of regression analysis
- Experience with Jupyter Notebooks

---

## Installation

### Step 1: Set Up Python Environment

We recommend using a virtual environment to avoid dependency conflicts.

#### Using venv (built-in)

```bash
# Create virtual environment
python -m venv panelbox_env

# Activate (Windows)
panelbox_env\Scripts\activate

# Activate (macOS/Linux)
source panelbox_env/bin/activate
```

#### Using conda

```bash
# Create environment
conda create -n panelbox_env python=3.10

# Activate
conda activate panelbox_env
```

### Step 2: Install Required Packages

#### Option A: Install from PyPI (recommended)

```bash
pip install panelbox pandas numpy matplotlib seaborn jupyter scipy statsmodels
```

#### Option B: Install from requirements file

If a `requirements.txt` is provided:

```bash
pip install -r requirements.txt
```

Create your own `requirements.txt`:
```text
panelbox>=0.7.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
scipy>=1.7.0
statsmodels>=0.13.0
```

#### Option C: Using conda

```bash
conda install -c conda-forge panelbox pandas numpy matplotlib seaborn jupyter scipy statsmodels
```

### Step 3: Verify Installation

```python
# Test imports
import panelbox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check versions
print(f"PanelBox: {panelbox.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
```

Expected output:
```
PanelBox: 0.7.0 (or higher)
Pandas: 1.3.0 (or higher)
NumPy: 1.21.0 (or higher)
```

---

## Verifying Your Setup

### Automatic Verification

Run the built-in verification function:

```python
from panelbox.examples import count

# Run verification
results = count.verify_installation()

# Check results
if results['all_passed']:
    print("✓ All checks passed! You're ready to start.")
else:
    print("✗ Some components are missing:")
    for category, items in results.items():
        if category != 'all_passed':
            for name, status in items.items():
                if not status:
                    print(f"  Missing: {category}/{name}")
```

### Manual Verification Checklist

Run these checks in a Python console:

```python
from pathlib import Path
import pandas as pd

# 1. Check module structure
count_dir = Path('examples/count')
assert count_dir.exists(), "Count directory not found"
print("✓ Directory structure OK")

# 2. Check data access
data_file = count_dir / 'data' / 'healthcare_visits.csv'
df = pd.read_csv(data_file)
assert df.shape[0] > 0, "Data file empty"
print(f"✓ Data loaded: {df.shape}")

# 3. Check PanelBox models
from panelbox.models.count import PooledPoisson
print("✓ PanelBox models accessible")

# 4. Test a simple model
from panelbox.datasets import generate_count_data
test_data = generate_count_data(n=100, seed=42)
model = PooledPoisson.from_formula('y ~ x1 + x2', data=test_data)
result = model.fit()
print(f"✓ Test model fitted: converged={result.converged}")
```

If all checks pass, you're ready to go!

---

## Running Your First Tutorial

### Step 1: Navigate to Tutorial Directory

```bash
cd examples/count
```

### Step 2: Start Jupyter

```bash
jupyter notebook
```

Or use JupyterLab:
```bash
jupyter lab
```

Your browser should open automatically to the Jupyter interface.

### Step 3: Open Tutorial 01

1. In Jupyter, navigate to `notebooks/`
2. Click on `01_poisson_introduction.ipynb`
3. The notebook will open in a new tab

### Step 4: Run the Notebook

#### Run All Cells
- Menu: `Cell` → `Run All`
- Or use keyboard shortcut: `Shift + Enter` to run cells one by one

#### Expected Runtime
- Tutorial 01 takes approximately 5-10 minutes to run completely
- Most computation is fast; time is for reading and understanding

### Step 5: Verify Output

You should see:
- Data loaded successfully
- Model summary tables displayed
- Diagnostic plots rendered
- No error messages (warnings are OK)

---

## Understanding the Structure

### File Organization

```
count/
├── notebooks/          # Start here - tutorial notebooks
├── data/               # Datasets used in tutorials
│   └── codebooks/      # Variable documentation
├── outputs/            # Generated figures and tables
│   ├── figures/        # All plots (by notebook)
│   ├── tables/         # Summary tables
│   └── results/        # Saved model objects
├── solutions/          # Exercise solutions (use after trying!)
└── utils/              # Helper functions (imported in notebooks)
```

### Path Configuration in Notebooks

All notebooks use relative paths from the notebook location:

```python
from pathlib import Path

# Paths relative to notebook
BASE_DIR = Path('..')              # count/
DATA_DIR = BASE_DIR / 'data'       # count/data/
OUTPUT_DIR = BASE_DIR / 'outputs'  # count/outputs/
```

This ensures notebooks work regardless of where PanelBox is installed.

### Importing Utilities

Notebooks import helper functions from `utils/`:

```python
import sys
sys.path.insert(0, str(BASE_DIR / 'utils'))

from data_generators import generate_healthcare_data
from visualization_helpers import plot_rootogram
from diagnostics_helpers import compute_overdispersion_index
```

---

## Common Issues

### Issue 1: Module Not Found

**Symptom:**
```
ModuleNotFoundError: No module named 'panelbox'
```

**Solution:**
```bash
# Ensure you're in the correct environment
pip install panelbox

# Or upgrade if already installed
pip install --upgrade panelbox
```

### Issue 2: Data Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '../data/healthcare_visits.csv'
```

**Solution:**
Check your working directory and paths:
```python
import os
print(f"Current directory: {os.getcwd()}")

# If data is missing, generate it
from utils.data_generators import generate_healthcare_data
df = generate_healthcare_data()
df.to_csv('../data/healthcare_visits.csv', index=False)
```

### Issue 3: Import Errors from Utils

**Symptom:**
```
ModuleNotFoundError: No module named 'visualization_helpers'
```

**Solution:**
Ensure utils directory is in Python path:
```python
import sys
from pathlib import Path

# Add utils to path
utils_path = Path('..') / 'utils'
if str(utils_path) not in sys.path:
    sys.path.insert(0, str(utils_path))

# Now import should work
from visualization_helpers import plot_rootogram
```

### Issue 4: Convergence Warnings

**Symptom:**
```
ConvergenceWarning: Maximum number of iterations reached
```

**Solution:**
This is often OK, but to investigate:
```python
# Check if model converged
print(f"Converged: {result.converged}")
print(f"Iterations: {result.nit}")

# Try with different options
result = model.fit(maxiter=1000, gtol=1e-6)
```

### Issue 5: Plotting Issues

**Symptom:**
Plots don't display or look wrong.

**Solution:**
```python
# Ensure matplotlib backend is set
import matplotlib
matplotlib.use('Agg')  # For saving only
# or
%matplotlib inline     # In Jupyter

# Reset style if plots look wrong
import matplotlib.pyplot as plt
plt.style.use('default')
```

### Issue 6: Jupyter Kernel Dies

**Symptom:**
Kernel crashes when running cells.

**Solution:**
1. Restart kernel: `Kernel` → `Restart`
2. Clear outputs: `Cell` → `All Output` → `Clear`
3. Check memory usage (may need more RAM)
4. Try running cells individually instead of all at once

---

## Working with Datasets

### Loading Data

Standard approach in notebooks:

```python
import pandas as pd
from pathlib import Path

# Define path
DATA_PATH = Path('../data')
data_file = DATA_PATH / 'healthcare_visits.csv'

# Load with explicit types (faster, more reliable)
dtype_dict = {
    'individual_id': 'int32',
    'visits': 'int32',
    'age': 'int16',
    'income': 'float32',
    'insurance': 'int8',
    'chronic': 'int8'
}

df = pd.read_csv(data_file, dtype=dtype_dict)

# Verify
print(f"Shape: {df.shape}")
print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
```

### Reading Codebooks

Before analyzing, understand your data:

```python
codebook_path = Path('../data/codebooks/healthcare_visits_codebook.txt')

with open(codebook_path, 'r') as f:
    print(f.read())
```

---

## Best Practices

### 1. Follow the Learning Sequence

Don't skip ahead unless you're confident in prerequisites:
- Tutorial 01 → 02 → 06 → 07 (Beginner path)
- See README.md for other pathways

### 2. Try Exercises Before Solutions

Each tutorial includes exercises. Attempt them before checking solutions:
1. Read exercise prompt carefully
2. Try coding the solution
3. Run and debug
4. Only then check `solutions/`

### 3. Keep Notebooks Clean

- Restart kernel periodically: `Kernel` → `Restart & Clear Output`
- Don't modify original notebooks; make copies for experiments
- Save frequently

### 4. Use Version Control

If you're making modifications:
```bash
git init
git add examples/count/notebooks/*.ipynb
git commit -m "Initial notebooks"
```

### 5. Document Your Learning

Add markdown cells with notes:
- Questions that arose
- Insights gained
- Connections to your research

---

## Next Steps

### After Tutorial 01

You should understand:
- ✓ When to use Poisson regression
- ✓ How to interpret IRRs
- ✓ Basic diagnostics (overdispersion)
- ✓ Comparison with OLS

**Next:**
- If overdispersion is an issue → Tutorial 02 (Negative Binomial)
- If you have panel data → Tutorial 03 (FE/RE)
- If comfortable with basics → Tutorial 06 (Marginal Effects)

### Building Your Own Analysis

1. **Start with exploratory analysis:**
   - Load your data
   - Check distribution of outcome
   - Assess zero prevalence
   - Look for overdispersion

2. **Choose appropriate model:**
   - Equidispersed → Poisson
   - Overdispersed → Negative Binomial
   - Excess zeros → ZIP/ZINB
   - Panel structure → FE/RE

3. **Use tutorial code as template:**
   - Copy relevant sections
   - Adapt to your variable names
   - Follow diagnostic steps

4. **Check robustness:**
   - Alternative specifications
   - Different standard error types
   - Sensitivity analyses

---

## Getting Help

### Debugging Checklist

Before asking for help:

1. ✓ Read the error message carefully
2. ✓ Check the [Common Issues](#common-issues) section
3. ✓ Verify your environment setup
4. ✓ Try restarting the kernel
5. ✓ Check if the problem persists with fresh data

### Resources

**Documentation:**
- PanelBox docs: https://panelbox.readthedocs.io
- Pandas docs: https://pandas.pydata.org/docs/
- Jupyter docs: https://jupyter-notebook.readthedocs.io/

**Community:**
- PanelBox GitHub Issues: https://github.com/panelbox/panelbox/issues
- Stack Overflow: Tag with `panelbox` and `count-data`

**Academic References:**
See the References section in main README.md

---

## Tips for Success

1. **Take Your Time**
   - Each tutorial is 60-90 minutes
   - Don't rush through content
   - Understand before moving on

2. **Experiment**
   - Modify parameters
   - Try different specifications
   - Break things (in a copy!)

3. **Connect to Theory**
   - Review cited papers
   - Understand the econometric foundations
   - Think about assumptions

4. **Practice Regularly**
   - Revisit tutorials
   - Apply to different datasets
   - Teach concepts to others

5. **Keep Learning**
   - Read recent papers using these methods
   - Follow PanelBox updates
   - Explore related topics (duration models, panel count, etc.)

---

## Quick Reference Card

### Essential Commands

```python
# Load data
import pandas as pd
df = pd.read_csv('../data/healthcare_visits.csv')

# Fit Poisson
from panelbox.models.count import PooledPoisson
model = PooledPoisson.from_formula('visits ~ age + income', data=df)
result = model.fit()

# Summary
print(result.summary())

# Marginal effects
from panelbox.marginal_effects import count_me
me = count_me(result, at='mean')
print(me.summary())
```

### Keyboard Shortcuts (Jupyter)

- `Shift + Enter`: Run cell and advance
- `Ctrl + Enter`: Run cell in place
- `Esc`: Enter command mode
- `M`: Change cell to Markdown
- `Y`: Change cell to Code
- `A`: Insert cell above
- `B`: Insert cell below
- `DD`: Delete cell

---

## Checklist Before Starting

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] PanelBox and dependencies installed
- [ ] Installation verified (all checks pass)
- [ ] Jupyter running successfully
- [ ] Tutorial 01 opens without errors
- [ ] Can load healthcare_visits.csv
- [ ] Understand basic notebook navigation

If all checked, you're ready! Open `notebooks/01_poisson_introduction.ipynb` and begin your journey into count data econometrics.

---

**Questions?** Review the main README.md or open an issue on GitHub.

**Ready to start?** Proceed to Tutorial 01: Poisson Introduction.

Good luck and enjoy learning!
