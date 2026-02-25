---
title: "CLI API"
description: "API reference for the panelbox command-line interface — estimate models and inspect data from the terminal"
---

# CLI API Reference

!!! info "Entry Point"
    **Command**: `panelbox`
    **Source**: `panelbox/cli/`

## Overview

PanelBox provides a command-line interface for quick model estimation and data inspection without writing Python code. The CLI supports all static panel models, GMM estimators, and multiple covariance types.

```bash
panelbox <command> [options]
```

| Command | Description |
|---------|-------------|
| `panelbox estimate` | Estimate a panel model from a data file |
| `panelbox info` | Display data or results information |

---

## `panelbox estimate`

Estimate a panel model from a CSV data file.

```bash
panelbox estimate --data FILE --model TYPE --formula FORMULA --entity COL --time COL [options]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--data FILE` | Path to CSV data file | `--data panel.csv` |
| `--model TYPE` | Model type | `--model fe` |
| `--formula FORMULA` | R-style formula | `--formula "invest ~ value + capital"` |
| `--entity COL` | Entity column name | `--entity firm` |
| `--time COL` | Time column name | `--time year` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output FILE`, `-o` | — | Save results to file |
| `--cov-type TYPE` | `nonrobust` | Covariance type (see table below) |
| `--format FORMAT` | `pickle` | Output format: `pickle` or `json` |
| `--verbose`, `-v` | — | Print verbose output |
| `--no-summary` | — | Skip printing the summary table |

### Model Types

| Value | Model |
|-------|-------|
| `pooled` | Pooled OLS |
| `fe` | Fixed Effects |
| `re` | Random Effects |
| `between` | Between Estimator |
| `fd` | First Difference |
| `diff_gmm` | Difference GMM |
| `sys_gmm` | System GMM |

### Covariance Types

| Value | Description |
|-------|-------------|
| `nonrobust` | Standard (default) |
| `robust` | HC1 robust |
| `hc0`, `hc1`, `hc2`, `hc3` | Heteroskedasticity-consistent |
| `clustered` | Clustered by entity |
| `twoway` | Two-way clustered |
| `driscoll_kraay` | Driscoll-Kraay |
| `newey_west` | Newey-West HAC |
| `pcse` | Panel-corrected SE |

### Examples

```bash
# Basic fixed effects estimation
panelbox estimate \
  --data investment.csv \
  --model fe \
  --formula "invest ~ value + capital" \
  --entity firm \
  --time year

# With robust standard errors, save results
panelbox estimate \
  --data investment.csv \
  --model fe \
  --formula "invest ~ value + capital" \
  --entity firm \
  --time year \
  --cov-type robust \
  --output results.pkl

# Random effects with clustered SE
panelbox estimate \
  --data panel.csv \
  --model re \
  --formula "y ~ x1 + x2 + x3" \
  --entity id \
  --time period \
  --cov-type clustered \
  --verbose

# Difference GMM
panelbox estimate \
  --data employment.csv \
  --model diff_gmm \
  --formula "n ~ w + k | L.n" \
  --entity id \
  --time year
```

---

## `panelbox info`

Display information about a dataset or saved model results.

```bash
panelbox info [--data FILE | --results FILE] [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--data FILE` | Path to CSV data file (mutually exclusive with `--results`) |
| `--results FILE` | Path to saved results file (.pkl) (mutually exclusive with `--data`) |
| `--entity COL` | Entity column (for `--data` mode) |
| `--time COL` | Time column (for `--data` mode) |
| `--verbose`, `-v` | Print verbose output |

### Examples

```bash
# Inspect a data file
panelbox info --data panel.csv --entity firm --time year

# Inspect saved results
panelbox info --results results.pkl --verbose
```

### Data Info Output

When using `--data`, the output includes:

- Number of observations, entities, and time periods
- Panel balance (balanced/unbalanced)
- Variable statistics (mean, std, min, max)
- Missing values per column

### Results Info Output

When using `--results`, the output includes:

- Model type and formula
- Coefficient estimates and standard errors
- Fit statistics (R², AIC, BIC)
- Diagnostic test results (if available)

---

## Integration with Python API

The CLI is designed for quick analyses. For full control, use the Python API:

=== "CLI"

    ```bash
    panelbox estimate \
      --data data.csv \
      --model fe \
      --formula "y ~ x1 + x2" \
      --entity id --time year \
      --cov-type robust
    ```

=== "Python"

    ```python
    import pandas as pd
    from panelbox.models.static import FixedEffects

    data = pd.read_csv("data.csv")
    model = FixedEffects(data, formula="y ~ x1 + x2",
                         entity_col="id", time_col="year")
    result = model.fit(cov_type="robust")
    print(result.summary())
    ```

The Python API provides access to:

- All model types (70+)
- Validation and diagnostic tests
- Visualization and report generation
- Result serialization and loading

---

## See Also

- [Getting Started](../getting-started/index.md) — installation and first steps
- [Static Models API](static-models.md) — full Python model API
- [Datasets API](datasets.md) — built-in datasets
