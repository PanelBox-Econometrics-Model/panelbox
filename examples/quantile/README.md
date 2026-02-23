# Quantile Regression Tutorials - PanelBox

Comprehensive tutorial series on quantile regression methods for panel data
using the PanelBox library.

## Overview

| # | Notebook | Level | Duration | Topics |
|---|----------|-------|----------|--------|
| 1 | Fundamentals | Introductory | 90 min | Check loss, PooledQuantile, coefficient paths |
| 2 | Multiple Quantiles | Intermediate | 120 min | Quantile process, inter-quantile tests, glass ceiling |
| 3 | FE Canay | Intermediate | 120 min | Location shift, Canay two-step, FE comparison |
| 4 | FE Penalty | Advanced | 150 min | Koenker penalty, lambda selection, alpha_i(tau) |
| 5 | Location-Scale | Advanced | 150 min | MSS model, heteroskedasticity, crossing prevention |
| 6 | Diagnostics | Advanced | 120 min | Khmaladze, He-Zhu, Cook's D, health score |
| 7 | Bootstrap | Inter-Advanced | 120 min | Cluster bootstrap, BCa, parallel computation |
| 8 | Monotonicity | Advanced | 150 min | Rearrangement, isotonic, constrained QR |
| 9 | Treatment Effects | Very Advanced | 180 min | CQTE, UQTE/RIF, DiD-QR, CiC |
| 10 | Dynamic Models | Very Advanced | 180 min | QAR, IV-QR, Galvao bias correction |

## Prerequisites

- Python 3.8+
- panelbox >= 0.7.0
- pandas, numpy, scipy, matplotlib, seaborn

## Learning Pathways

**Beginner**: 01 -> 02 -> 07

**Fixed Effects Focus**: 01 -> 02 -> 03 -> 04

**Advanced Time Series**: 01 -> 02 -> 03 -> 04 -> 10

**Causal Inference**: 01 -> 02 -> 03 -> 07 -> 09

**Complete Course**: 01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> 09 -> 10

## Datasets

All datasets are in `data/`. See `data/README.md` for details.

## Running the Tutorials

1. Navigate to the `notebooks/` directory
2. Start Jupyter: `jupyter notebook`
3. Open notebooks in order (or follow a learning pathway)
4. Exercises are provided at the end of each notebook
5. Solutions are in `solutions/`

## Running Tests

```bash
# Test data integrity
pytest tests/test_data_integrity.py -v

# Test utility functions
pytest tests/test_utils.py -v

# Test notebooks execute (slow)
pytest tests/test_notebooks_run.py -v -m slow
```
