# GMM Tutorial Data

## Overview

This directory contains all datasets used in the GMM tutorial series, including both real-world panel data and simulated datasets for pedagogical purposes.

## Real Datasets

### `abdata.csv` — Arellano-Bond Employment Data

| Field | Description |
|-------|-------------|
| **Source** | Arellano & Bond (1991) |
| **Dimensions** | 140 firms, 7–9 years (unbalanced) |
| **Used in** | Notebooks 01, 02, 04 |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `firm` | Firm identifier | int |
| `year` | Year | int |
| `n` | Log employment | float |
| `w` | Log real wage | float |
| `k` | Log gross capital | float |
| `ys` | Log industry output | float |

**Citation:**
> Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. *Review of Economic Studies*, 58(2), 277–297.

---

### `growth.csv` — Country Growth Data

| Field | Description |
|-------|-------------|
| **Source** | Penn World Tables (adapted) |
| **Dimensions** | 100 countries, 20 years |
| **Used in** | Notebook 02 (persistent series) |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `country` | Country identifier | int |
| `year` | Year | int |
| `lgdp` | Log GDP per capita | float |
| `inv` | Investment share of GDP | float |
| `school` | Schooling (years) | float |
| `popgrowth` | Population growth rate | float |

---

### `firm_investment.csv` — Corporate Investment Data

| Field | Description |
|-------|-------------|
| **Source** | Compustat-style (adapted) |
| **Dimensions** | 500 firms, 10 years |
| **Used in** | Notebooks 03, 06 |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `firm` | Firm identifier | int |
| `year` | Year | int |
| `ik` | Investment-to-capital ratio (I/K) | float |
| `q` | Tobin's Q | float |
| `cashflow` | Cash flow / assets | float |
| `sales` | Log sales | float |
| `debt` | Debt / assets | float |

---

## Simulated Datasets

All simulated datasets are generated using `utils/data_generation.py` with fixed random seeds for reproducibility.

### `dgp_nickell_bias.csv` — Nickell Bias Demonstration

| Field | Description |
|-------|-------------|
| **DGP** | y_{it} = ρ y_{i,t-1} + μ_i + ε_{it} |
| **Parameters** | ρ ∈ {0.3, 0.5, 0.8}, T ∈ {5, 10, 20} |
| **Used in** | Notebook 01 |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `entity` | Entity identifier | int |
| `time` | Time period | int |
| `y` | Dependent variable | float |
| `rho` | True autoregressive parameter | float |
| `T` | Panel length | int |

---

### `weak_instruments.csv` — Weak Instruments Demonstration

| Field | Description |
|-------|-------------|
| **DGP** | Near unit root process (ρ → 1) |
| **Used in** | Notebook 02 |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `entity` | Entity identifier | int |
| `time` | Time period | int |
| `y` | Dependent variable (near random walk) | float |
| `x` | Exogenous regressor | float |

---

### `bad_specification.csv` — Diagnostic Pedagogy

| Field | Description |
|-------|-------------|
| **DGP** | Model with correlated omitted variable |
| **Purpose** | Hansen J-test rejects due to misspecification |
| **Used in** | Notebook 04 |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `entity` | Entity identifier | int |
| `time` | Time period | int |
| `y` | Dependent variable | float |
| `x1` | Included regressor | float |
| `x2_omitted` | Omitted variable (correlated with instruments) | float |

---

### `medium_panel_bias.csv` — Bias Correction

| Field | Description |
|-------|-------------|
| **DGP** | N=200, T=15, ρ=0.7 |
| **Purpose** | Compare two-step vs CUE bias correction |
| **Used in** | Notebook 05 |

**Variables:**

| Variable | Description | Type |
|----------|-------------|------|
| `entity` | Entity identifier | int |
| `time` | Time period | int |
| `y` | Dependent variable | float |
| `x` | Exogenous regressor | float |

---

## Descriptive Statistics

Run the following to generate summary statistics for all datasets:

```python
import pandas as pd
from pathlib import Path

data_dir = Path(".")
for csv_file in sorted(data_dir.glob("*.csv")):
    df = pd.read_csv(csv_file)
    print(f"\n{'='*60}")
    print(f"Dataset: {csv_file.name}")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"{'='*60}")
    print(df.describe().round(3))
```

## Data Generation

Simulated datasets can be regenerated using:

```python
from utils.data_generation import (
    generate_nickell_bias_data,
    generate_weak_instruments_data,
    generate_bad_specification_data,
    generate_medium_panel_data,
)

# Each function uses a fixed seed for reproducibility
generate_nickell_bias_data(output_path="data/dgp_nickell_bias.csv", seed=42)
generate_weak_instruments_data(output_path="data/weak_instruments.csv", seed=123)
generate_bad_specification_data(output_path="data/bad_specification.csv", seed=456)
generate_medium_panel_data(output_path="data/medium_panel_bias.csv", seed=789)
```
