# Getting Started — Visualization and Reports

This guide walks you through everything needed to run the Visualization tutorial series.

---

## Prerequisites

| Requirement | Minimum Version |
|---|---|
| Python | 3.8+ |
| Jupyter Notebook / JupyterLab | 6.0+ / 3.0+ |
| PanelBox | 0.8.0+ |
| NumPy | 1.21+ |
| pandas | 1.3+ |
| matplotlib | 3.5+ |
| Plotly | 5.0+ |
| kaleido | 0.2+ (for static chart export) |

---

## Step 1: Install PanelBox

**Option A — from PyPI** (stable release):

```bash
pip install panelbox
```

**Option B — development version** (from source):

```bash
git clone https://github.com/panelbox/panelbox.git
cd panelbox
pip install -e .
```

---

## Step 2: Install Tutorial Dependencies

```bash
pip install jupyter pandas numpy matplotlib seaborn plotly kaleido scipy
```

For JupyterLab with interactive Plotly support:

```bash
pip install jupyterlab-plotly
```

---

## Step 3: Verify Your Environment

Run the following in a Python shell or notebook cell:

```python
import sys
print("Python:", sys.version)

import numpy as np
print("NumPy:", np.__version__)

import pandas as pd
print("pandas:", pd.__version__)

import matplotlib
print("matplotlib:", matplotlib.__version__)

import plotly
print("Plotly:", plotly.__version__)

import panelbox as pb
print("PanelBox:", pb.__version__)

# Test chart export
import plotly.io as pio
import plotly.graph_objects as go

fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
pio.write_image(fig, "/tmp/test_kaleido.png")
print("Chart export: OK")
```

All imports should succeed and `/tmp/test_kaleido.png` should be created.

---

## Step 4: Launch Jupyter

```bash
cd /home/guhaase/projetos/panelbox/examples/visualization/notebooks
jupyter notebook
# or
jupyter lab
```

Open `01_visualization_introduction.ipynb` to begin.

---

## Step 5: Configure Path (if not installed via pip)

If PanelBox is not installed system-wide, all notebooks include this path setup at the top:

```python
import sys
sys.path.insert(0, '../../../')   # points to /path/to/panelbox/
import panelbox as pb
```

This is already present in each notebook — no action needed.

---

## Recommended Order

| Notebook | Topic | Time |
|---|---|---|
| `01_visualization_introduction.ipynb` | Core plotting API, basic charts | ~45 min |
| `02_visual_diagnostics.ipynb` | Residual and influence diagnostics | ~60 min |
| `03_advanced_visualizations.ipynb` | Themes, interactive Plotly, layouts | ~75 min |
| `04_automated_reports.ipynb` | HTML and LaTeX report generation | ~60 min |

**Total estimated time**: 4 hours

---

## Output Directories

Notebooks write all generated files to:

```
outputs/
├── charts/
│   ├── png/     ← PNG exports (web, email)
│   ├── svg/     ← SVG exports (editable vector)
│   └── pdf/     ← PDF exports (LaTeX insertion)
└── reports/
    ├── html/    ← Self-contained HTML reports
    └── latex/   ← .tex files and compiled PDFs
```

These directories are already created. Contents are git-ignored (only `.gitkeep` files are tracked).

---

## Quick Sanity Check

Run the test suite to verify all example code works:

```bash
cd /home/guhaase/projetos/panelbox/examples/visualization
pytest tests/test_visualization_examples.py -v
```

---

## Common Issues

| Issue | Fix |
|---|---|
| `ModuleNotFoundError: panelbox` | `pip install panelbox` or add path with `sys.path.insert` |
| Plotly charts show as blank | Set `pio.renderers.default = 'notebook'` |
| `kaleido` export fails | `pip install kaleido` |
| LaTeX reports fail | Install TeX Live or MiKTeX |
| Jupyter not found | `pip install jupyter` |

---

## Next Steps

After completing the tutorial series:

- Explore other PanelBox example modules: `discrete/`, `count/`, `spatial/`, `standard_errors/`
- Read the [PanelBox Visualization API docs](https://panelbox.readthedocs.io/visualization)
- Check `solutions/` for fully executed reference notebooks
