# How to Install PanelBox

> Quick guide to installing PanelBox in various environments.

## Quick Install (Recommended)

Install the latest stable version from PyPI:

```bash
pip install panelbox
```

That's it! PanelBox and all its dependencies will be installed.

## Requirements

**Python Version:**
- Python ≥ 3.9 (3.9, 3.10, 3.11, 3.12 supported)

**Core Dependencies** (installed automatically):
- NumPy ≥ 1.24.0
- Pandas ≥ 2.0.0
- SciPy ≥ 1.10.0
- statsmodels ≥ 0.14.0
- patsy ≥ 0.5.3

**Optional Dependencies:**
- matplotlib ≥ 3.5.0 (for plotting)
- numba ≥ 0.56.0 (for performance optimization)

## Installation Methods

### 1. Using pip (Recommended)

**Standard installation:**
```bash
pip install panelbox
```

**With optional dependencies:**
```bash
# With plotting support
pip install panelbox[plots]

# With performance optimization
pip install panelbox[performance]

# With development tools
pip install panelbox[dev]

# Everything
pip install panelbox[all]
```

**Upgrade to latest version:**
```bash
pip install --upgrade panelbox
```

### 2. Using conda

PanelBox is also available via conda-forge:

```bash
conda install -c conda-forge panelbox
```

### 3. From Source (Development)

Install the latest development version from GitHub:

```bash
# Clone repository
git clone https://github.com/PanelBox-Econometrics-Model/panelbox.git
cd panelbox

# Install in editable mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Virtual Environments

**Recommended:** Always use a virtual environment to avoid dependency conflicts.

### Using venv (Built-in)

```bash
# Create virtual environment
python -m venv panelbox_env

# Activate (Linux/Mac)
source panelbox_env/bin/activate

# Activate (Windows)
panelbox_env\Scripts\activate

# Install PanelBox
pip install panelbox
```

### Using conda

```bash
# Create environment
conda create -n panelbox_env python=3.11

# Activate
conda activate panelbox_env

# Install
conda install -c conda-forge panelbox
# or: pip install panelbox
```

## Verify Installation

Check that PanelBox is installed correctly:

```python
import panelbox as pb
print(f"PanelBox version: {pb.__version__}")

# Test basic functionality
data = pb.load_grunfeld()
print(f"Loaded {len(data)} observations")
```

**Expected output:**
```
PanelBox version: 1.0.0
Loaded 200 observations
```

## Platform-Specific Notes

### Windows

**Issue:** Some users report installation errors related to NumPy/SciPy.

**Solution:** Install via Anaconda/Miniconda:
```bash
conda install -c conda-forge panelbox
```

Or use pre-built wheels:
```bash
pip install --only-binary :all: panelbox
```

### macOS (Apple Silicon M1/M2)

**Issue:** Native ARM64 wheels may not be available for all dependencies.

**Solution 1 - Rosetta (stable):**
```bash
arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
arch -x86_64 pip install panelbox
```

**Solution 2 - Native ARM64 (recommended):**
```bash
# Use miniforge (ARM64-native conda)
conda install -c conda-forge panelbox
```

### Linux

Should work out-of-the-box on all major distributions.

**If build tools are missing:**
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# RHEL/CentOS/Fedora
sudo yum install python3-devel gcc gcc-c++

# Then install
pip install panelbox
```

## Troubleshooting

### Import Error: No module named 'panelbox'

**Cause:** PanelBox not installed or wrong Python environment

**Solution:**
```bash
# Check if installed
pip list | grep panelbox

# If not found, install
pip install panelbox

# Check Python executable
which python  # Linux/Mac
where python  # Windows
```

### Dependency Conflicts

**Cause:** Conflicting package versions

**Solution 1 - Fresh virtual environment:**
```bash
python -m venv fresh_env
source fresh_env/bin/activate  # Linux/Mac
pip install panelbox
```

**Solution 2 - Update dependencies:**
```bash
pip install --upgrade numpy pandas scipy statsmodels
pip install panelbox
```

### ImportError: DLL load failed (Windows)

**Cause:** Missing Microsoft Visual C++ Redistributable

**Solution:** Download and install from:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Performance Warning: Numba not available

**Not critical:** PanelBox works without Numba, just slower for large datasets

**Solution (optional):**
```bash
pip install numba
```

## Jupyter Notebook Setup

Install PanelBox in your Jupyter environment:

```bash
# Activate your Jupyter environment
source ~/jupyter_env/bin/activate

# Install PanelBox
pip install panelbox

# Install Jupyter kernel
python -m ipykernel install --user --name=panelbox_env
```

**In notebook:**
```python
import panelbox as pb
data = pb.load_grunfeld()
```

## Upgrading

**Check current version:**
```bash
pip show panelbox
```

**Upgrade to latest:**
```bash
pip install --upgrade panelbox
```

**Upgrade with dependencies:**
```bash
pip install --upgrade --upgrade-strategy eager panelbox
```

## Uninstalling

```bash
pip uninstall panelbox
```

## Getting Help

**Installation issues:**
- GitHub Issues: https://github.com/PanelBox-Econometrics-Model/panelbox/issues
- Check existing issues first
- Provide your system info: Python version, OS, error message

**System information for bug reports:**
```python
import sys
import platform
import panelbox as pb

print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PanelBox: {pb.__version__}")
```

## Next Steps

✅ **Installation complete!** Now:

1. [**Getting Started Tutorial**](../tutorials/01_getting_started.md): Your first panel model
2. [**API Reference**](../api/index.md): Complete documentation
3. [**Examples**](../../examples/): Real-world use cases

---

**Need help?** Open an issue on [GitHub](https://github.com/PanelBox-Econometrics-Model/panelbox/issues).
