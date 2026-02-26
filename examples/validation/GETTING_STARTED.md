# Getting Started

This guide walks you through the one-time setup required before running the
validation tutorial notebooks.

---

## Step 1 — Install panelbox

```bash
pip install panelbox
```

Or install from source (from the repository root):

```bash
pip install -e .
```

---

## Step 2 — Install notebook dependencies

```bash
pip install numpy pandas scipy matplotlib plotly jupyterlab
```

---

## Step 3 — Generate the datasets

Run the data-generator script once to create all CSV files in `data/`:

```bash
cd examples/validation
python utils/data_generators.py
```

Expected output:

```
Generating validation datasets...
  Wrote firmdata.csv  (1,000 rows × 6 cols)
  Wrote macro_panel.csv  (600 rows × 6 cols)
  Wrote small_panel.csv  (200 rows × 5 cols)
  Wrote sales_panel.csv  (1,200 rows × 6 cols)
  Wrote macro_ts_panel.csv  (600 rows × 5 cols)
  Wrote panel_with_outliers.csv  (640 rows × 6 cols)
  Wrote real_firms.csv  (600 rows × 6 cols)
  Wrote panel_comprehensive.csv  (1,200 rows × 12 cols)
  Wrote panel_unbalanced.csv  (~1,050 rows × 6 cols)
Done.
```

---

## Step 4 — Verify imports

```python
from panelbox.validation import ValidationSuite, ValidationReport
from panelbox.validation.heteroskedasticity import ModifiedWaldTest
from panelbox.validation.serial_correlation import WooldridgeARTest
from panelbox.validation.cross_sectional_dependence import PesaranCDTest
from panelbox.validation.robustness import PanelBootstrap, TimeSeriesCV
print("All imports OK")
```

---

## Step 5 — Launch Jupyter

```bash
cd examples/validation
jupyter lab notebooks/
```

Run notebooks in order: **01 → 02 → 03 → 04**.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ImportError: panelbox` | Ensure `pip install -e` completed or `PYTHONPATH` is set |
| `FileNotFoundError: firmdata.csv` | Run `python utils/data_generators.py` again |
| Slow bootstrap | Reduce `n_bootstrap` parameter in notebook cells |

---

## Output files

Notebooks write results to `outputs/`:

- `01_assumption_summary.json`
- `02_bootstrap_results.json`
- `03_influence_report.html`
- `04_model_comparison.html`
