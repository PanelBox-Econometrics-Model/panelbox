# API Reference

Complete API documentation for PanelBox.

## Overview

PanelBox provides a comprehensive API for panel data econometrics organized into the following modules:

### 📊 [Static Models](models.md)

Panel models without dynamics:

- **PooledOLS**: Pooled Ordinary Least Squares
- **FixedEffects**: Fixed Effects (Within) estimator
- **RandomEffects**: Random Effects (GLS) estimator
- **Between**: Between estimator (entity means)
- **FirstDifferences**: First Differences estimator

### 🔄 [GMM Models](gmm.md)

Dynamic panel GMM estimators:

- **DifferenceGMM**: Arellano-Bond Difference GMM (1991)
- **SystemGMM**: Blundell-Bond System GMM (1998)

### 📈 [Results](results.md)

Results container class:

- **PanelResults**: Estimation results with summary, tests, export

### ✅ [Validation](validation.md)

Diagnostic and specification tests:

- **HausmanTest**: Fixed Effects vs Random Effects
- **BreuschPaganLM**: Random effects test
- **BreuschPaganTest**: Heteroskedasticity test
- **WooldridgeTest**: Serial correlation test
- **Hansen J, Sargan**: GMM overidentification tests
- **AR tests**: GMM serial correlation tests

### 📦 [Datasets](datasets.md)

Example datasets for learning and testing:

- **load_grunfeld()**: Grunfeld investment data
- **load_abdata()**: Arellano-Bond employment data
- **list_datasets()**: List available datasets
- **get_dataset_info()**: Dataset information

### 📋 [Report](report.md)

Reporting and export utilities:

- **to_latex()**: Export to LaTeX tables
- **summary()**: Formatted summary tables
- **compare_models()**: Side-by-side comparison

## Quick Links

| Topic | API Documentation |
|-------|-------------------|
| Estimate Fixed Effects | [FixedEffects](models.md#fixedeffects) |
| Estimate Random Effects | [RandomEffects](models.md#randomeffects) |
| Run Hausman Test | [HausmanTest](validation.md#hausmantest) |
| Estimate Difference GMM | [DifferenceGMM](gmm.md#differencegmm) |
| Estimate System GMM | [SystemGMM](gmm.md#systemgmm) |
| Check Hansen J Test | [Results](results.md#hansen-j-test) |
| Load Example Data | [load_grunfeld](datasets.md#load_grunfeld) |
| Export to LaTeX | [to_latex](report.md#to_latex) |

## Usage Patterns

### Basic Workflow

```python
import panelbox as pb

# 1. Load data
data = pb.load_grunfeld()

# 2. Create model
model = pb.FixedEffects(
    formula="invest ~ value + capital",
    data=data,
    entity_col="firm",
    time_col="year"
)

# 3. Fit model
results = model.fit(cov_type='clustered')

# 4. View results
print(results.summary())

# 5. Export
results.to_latex("table1.tex")
```

### Advanced Workflow (GMM)

```python
# 1. Load data
data = pb.load_grunfeld()

# 2. Create GMM model
gmm = pb.SystemGMM(
    data=data,
    dep_var='invest',
    lags=1,
    exog_vars=['value', 'capital'],
    id_var='firm',
    time_var='year',
    collapse=True,
    robust=True
)

# 3. Fit
results = gmm.fit()

# 4. Check diagnostics
print(f"Hansen J: {results.hansen_j.pvalue:.3f}")
print(f"AR(2): {results.ar2_test.pvalue:.3f}")

# 5. If tests pass, view results
if results.hansen_j.pvalue > 0.10 and results.ar2_test.pvalue > 0.10:
    print(results.summary())
```

## Module Organization

```
panelbox/
├── models/
│   ├── static/
│   │   ├── pooled_ols.py      → PooledOLS
│   │   ├── fixed_effects.py   → FixedEffects
│   │   ├── random_effects.py  → RandomEffects
│   │   ├── between.py         → Between
│   │   └── first_differences.py → FirstDifferences
│   └── base.py                → PanelModel (base class)
├── gmm/
│   ├── difference_gmm.py      → DifferenceGMM
│   └── system_gmm.py          → SystemGMM
├── core/
│   └── results.py             → PanelResults
├── validation/
│   ├── specification/
│   │   ├── hausman.py         → HausmanTest
│   │   └── breusch_pagan_lm.py → BreuschPaganLM
│   └── ...
├── datasets/
│   └── load.py                → load_grunfeld, etc.
└── report/
    └── latex.py               → LaTeX export
```

## Next Steps

- Browse specific API documentation in the navigation
- See [Tutorials](../tutorials/01_getting_started.md) for hands-on examples
- Check [How-To Guides](../how-to/choose_model.md) for task-oriented help
