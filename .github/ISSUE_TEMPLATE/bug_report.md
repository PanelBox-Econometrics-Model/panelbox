---
name: Bug Report
about: Report a bug in PanelBox quantile regression module
title: '[QR] '
labels: 'bug, quantile'
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## To Reproduce

Minimal code example to reproduce the issue:

```python
from panelbox.models.quantile import PooledQuantile
import pandas as pd

# Your code here
```

## Expected Behavior

What you expected to happen.

## Actual Behavior

What actually happened. Include any error messages:

```
Paste error traceback here
```

## Environment

- **PanelBox version**: [e.g., 0.3.0]
- **Python version**: [e.g., 3.11.5]
- **Operating System**: [e.g., Ubuntu 22.04, Windows 11, macOS 13]
- **NumPy version**: [e.g., 1.24.3]
- **SciPy version**: [e.g., 1.11.1]

## Additional Context

Add any other context about the problem here, such as:
- Data characteristics (size, missingness, etc.)
- Whether the issue is reproducible
- Any workarounds you've found

## Validation Against R (if applicable)

If you've compared with R quantreg/rqpd and found discrepancies:

- **R version**: [e.g., 4.3.1]
- **quantreg version**: [e.g., 5.94]
- **Difference magnitude**: [e.g., coefficients differ by 0.01]

Please attach:
- [ ] Minimal reproducible example
- [ ] Sample data (if not too large)
- [ ] R comparison code (if applicable)
