---
title: "Datasets API"
description: "API reference for panelbox.datasets — built-in panel datasets for examples and testing"
---

# Datasets API Reference

!!! info "Module"
    **Import**: `from panelbox.datasets import ...`
    **Source**: `panelbox/datasets/`

## Overview

PanelBox includes built-in panel datasets for examples, testing, and learning. All datasets are returned as pandas DataFrames ready for immediate use with PanelBox models.

| Function | Dataset | N | T | Observations |
|----------|---------|---|---|-------------|
| `load_grunfeld()` | Grunfeld (1958) Investment | 10 firms | 20 years | 200 |
| `load_abdata()` | Arellano-Bond (1991) Employment | ~140 firms | ~9 years | ~1,031 |

---

## `load_grunfeld()`

Load the Grunfeld (1958) investment data — the classic balanced panel dataset.

```python
def load_grunfeld() -> pd.DataFrame
```

**Returns:** `pd.DataFrame` with 200 rows and 5 columns.

**Variables:**

| Column | Type | Description |
|--------|------|-------------|
| `invest` | `float` | Gross investment (dependent variable) |
| `value` | `float` | Market value of the firm |
| `capital` | `float` | Stock of plant and equipment |
| `firm` | `str/int` | Firm identifier (10 firms) |
| `year` | `int` | Year (1935–1954) |

**Firms:** General Motors, Chrysler, General Electric, Westinghouse, US Steel, and 5 others.

**Example:**

```python
from panelbox.datasets import load_grunfeld
from panelbox.models.static import FixedEffects

data = load_grunfeld()
print(data.head())
#    invest   value  capital  firm  year
# 0  317.6  3078.5    2.8     1   1935
# 1  391.8  4661.7   52.6     1   1936
# ...

model = FixedEffects(data, formula="invest ~ value + capital",
                     entity_col="firm", time_col="year")
result = model.fit()
print(result.summary())
```

---

## `load_abdata()`

Load the Arellano-Bond (1991) employment data — commonly used for dynamic panel (GMM) examples.

```python
def load_abdata() -> pd.DataFrame
```

**Returns:** `pd.DataFrame` (unbalanced panel).

**Variables:**

| Column | Type | Description |
|--------|------|-------------|
| `n` | `float` | Log employment (dependent variable) |
| `w` | `float` | Log real wage |
| `k` | `float` | Log capital stock |
| `ys` | `float` | Log industry output |
| `yr` | `int` | Year identifier |
| `id` | `int` | Firm identifier |
| `year` | `int` | Year |

**Example:**

```python
from panelbox.datasets import load_abdata
from panelbox.gmm import DifferenceGMM

data = load_abdata()

model = DifferenceGMM(
    data,
    formula="n ~ w + k | L.n",
    entity_col="id",
    time_col="year",
)
result = model.fit()
print(result.summary())
```

---

## `list_datasets()`

List all available built-in datasets.

```python
def list_datasets() -> list[str]
```

**Returns:** List of dataset names.

```python
from panelbox.datasets import list_datasets

print(list_datasets())
# ['grunfeld', 'abdata']
```

---

## `get_dataset_info()`

Get metadata about a built-in dataset.

```python
def get_dataset_info(dataset_name: str) -> dict[str, Any]
```

**Returns:** Dictionary with keys: `description`, `variables`, `n_observations`, `n_entities`, `n_periods`, `source`, `balanced`.

```python
from panelbox.datasets import get_dataset_info

info = get_dataset_info("grunfeld")
print(info["description"])
print(f"Variables: {info['variables']}")
print(f"Observations: {info['n_observations']}")
```

---

## See Also

- [Getting Started](../getting-started/index.md) — quick start with built-in data
- [Tutorials: Fundamentals](../tutorials/fundamentals.md) — panel data basics
- [GMM API](gmm.md) — GMM estimation with abdata
