---
title: Installation
description: Install PanelBox and its dependencies in any Python environment
---

# Installation

## Quick Install

```bash
pip install panelbox
```

That's it. PanelBox and all core dependencies will be installed automatically.

## Requirements

**Python**: >= 3.9 (3.9, 3.10, 3.11, 3.12 supported)

**Core dependencies** (installed automatically):

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| NumPy | >= 1.24.0 | Array operations |
| Pandas | >= 2.0.0 | Data handling |
| SciPy | >= 1.10.0 | Statistical functions |
| statsmodels | >= 0.14.0 | Econometric foundations |
| patsy | >= 0.5.3 | Formula parsing |

**Optional dependencies**:

| Package | Minimum Version | Purpose |
|---------|----------------|---------|
| matplotlib | >= 3.5.0 | Static plotting |
| plotly | >= 5.0.0 | Interactive charts |
| numba | >= 0.56.0 | Performance optimization |

## Installation Options

=== "pip (Recommended)"

    ```bash
    pip install panelbox
    ```

    Upgrade to the latest version:

    ```bash
    pip install --upgrade panelbox
    ```

=== "From Source"

    ```bash
    git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
    cd panelbox
    pip install -e .
    ```

=== "Development Mode"

    ```bash
    git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
    cd panelbox
    pip install -e ".[dev]"
    ```

### Optional Extras

```bash
# Plotting support
pip install panelbox[plots]

# Performance optimization
pip install panelbox[performance]

# Development tools (testing, linting, formatting)
pip install panelbox[dev]

# Documentation tools
pip install panelbox[docs]

# Everything
pip install panelbox[all]
```

## Verification

Verify that PanelBox is installed correctly:

```python
import panelbox as pb

# Check version
print(f"PanelBox version: {pb.__version__}")

# Test with built-in dataset
data = pb.load_grunfeld()
print(f"Loaded {len(data)} observations from Grunfeld dataset")

# Quick model test
from panelbox import FixedEffects
model = FixedEffects("invest ~ value + capital", data, "firm", "year")
results = model.fit()
print(f"R-squared (within): {results.rsquared_within:.4f}")
```

Expected output:

```text
PanelBox version: 1.x.x
Loaded 200 observations from Grunfeld dataset
R-squared (within): 0.7668
```

## Jupyter and Colab Setup

### Local Jupyter

```bash
# Install in your Jupyter environment
pip install panelbox

# If using a virtual environment, register the kernel
python -m ipykernel install --user --name=panelbox_env
```

### Google Colab

In the first cell of your notebook:

```python
!pip install panelbox -q
import panelbox as pb
print(pb.__version__)
```

!!! tip "Colab Notebooks"
    PanelBox tutorials are available as ready-to-run Colab notebooks.
    See the [User Guide](../user-guide/index.md) sections for links.

## Virtual Environments

We recommend using a virtual environment to avoid dependency conflicts:

=== "venv"

    ```bash
    python -m venv panelbox_env
    source panelbox_env/bin/activate   # Linux/macOS
    panelbox_env\Scripts\activate      # Windows
    pip install panelbox
    ```

=== "conda"

    ```bash
    conda create -n panelbox_env python=3.11
    conda activate panelbox_env
    pip install panelbox
    ```

## Troubleshooting

### `ModuleNotFoundError: No module named 'panelbox'`

Ensure PanelBox is installed in the active Python environment:

```bash
pip list | grep panelbox
```

If not found, install it. If using Jupyter, make sure the notebook kernel matches the environment where PanelBox is installed.

### Dependency Conflicts

Create a fresh virtual environment:

```bash
python -m venv fresh_env
source fresh_env/bin/activate
pip install panelbox
```

### Windows: DLL Load Failed

Install the Microsoft Visual C++ Redistributable from
[Microsoft's download page](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### macOS Apple Silicon (M1/M2/M3)

Use conda with native ARM64 support:

```bash
conda create -n panelbox_env python=3.11
conda activate panelbox_env
pip install panelbox
```

### Performance Warning: Numba Not Available

PanelBox works without Numba, but large datasets benefit from it:

```bash
pip install numba
```

### System Information for Bug Reports

```python
import sys, platform
import panelbox as pb

print(f"Python:    {sys.version}")
print(f"Platform:  {platform.platform()}")
print(f"PanelBox:  {pb.__version__}")
```

Include this output when [reporting issues](https://github.com/PanelBox-Econometrics-Model/panelbox/issues).

## Next Steps

- **[Quick Start](quickstart.md)** -- Run your first panel model in 5 minutes
- **[Core Concepts](core-concepts.md)** -- Understand panel data fundamentals
- **[Choosing a Model](choosing-model.md)** -- Find the right estimator for your research
