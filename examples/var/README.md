# Panel VAR Tutorial Series

Comprehensive tutorial series on Panel Vector Autoregression (VAR) models
using the PanelBox library.

## Overview

This series of 7 Jupyter notebooks covers Panel VAR from fundamentals to
advanced applications:

| # | Notebook | Level | Duration | Topics |
|---|----------|-------|----------|--------|
| 1 | VAR Introduction | Beginner | 60-90 min | Estimation, lag selection, stability |
| 2 | IRF Analysis | Intermediate | 90-120 min | Cholesky, Generalized, Bootstrap CI |
| 3 | FEVD Decomposition | Intermediate | 60-90 min | Variance decomposition, interpretation |
| 4 | Granger Causality | Inter-Advanced | 90-120 min | Wald tests, DH test, networks |
| 5 | VECM Cointegration | Advanced | 120-150 min | I(1) variables, rank tests, VECM |
| 6 | Dynamic GMM | Advanced | 120-150 min | Nickell bias, Arellano-Bond, Blundell-Bond |
| 7 | Case Study | Capstone | 180-240 min | Monetary policy transmission |

## Prerequisites

- Python 3.8+
- panelbox >= 0.7.0
- pandas, numpy, scipy, matplotlib, seaborn

## Getting Started

See [GETTING_STARTED.md](GETTING_STARTED.md) for installation and first steps.

## Learning Path

**Recommended order**: 01 → 02 → 03 → 04 → 05 → 06 → 07

- Notebooks 01-04 can be completed independently after 01
- Notebook 05 (VECM) requires understanding of 01-02
- Notebook 06 (GMM) is self-contained but benefits from 01
- Notebook 07 (Case Study) integrates all concepts from 01-06

## Folder Structure

- `notebooks/`: Tutorial notebooks (01-07)
- `solutions/`: Complete solutions for all exercises
- `data/`: Datasets (synthetic panel data)
- `utils/`: Utility functions for data generation and visualization
- `outputs/`: Generated figures, tables, and reports
- `tests/`: Validation tests
