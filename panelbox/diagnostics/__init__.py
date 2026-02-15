"""
Panel data diagnostics and tests.

This module provides diagnostic tests for panel data models including:
- Hausman test for fixed vs random effects
- Panel unit root tests (Hadri, Breitung)
- Panel cointegration tests (Kao, Pedroni, Westerlund)
- Quantile regression diagnostics
"""

from . import cointegration, unit_root

__all__ = [
    "cointegration",
    "unit_root",
]
