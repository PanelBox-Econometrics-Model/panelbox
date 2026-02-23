"""
Panel data diagnostics and tests.

This module provides diagnostic tests for panel data models including:
- Hausman test for fixed vs random effects
- Panel unit root tests (Hadri, Breitung)
- Panel cointegration tests (Kao, Pedroni, Westerlund)
- Quantile regression diagnostics
- Spatial dependence tests (LM-Lag, LM-Error)
"""

from __future__ import annotations

from . import cointegration, unit_root
from .spatial_tests import (
    LISAResult,
    LMTestResult,
    LocalMoranI,
    MoranIPanelTest,
    MoranIResult,
    lm_error_test,
    lm_lag_test,
    robust_lm_error_test,
    robust_lm_lag_test,
    run_lm_tests,
)

__all__ = [
    "LISAResult",
    "LMTestResult",
    # LISA
    "LocalMoranI",
    # Moran's I
    "MoranIPanelTest",
    "MoranIResult",
    "cointegration",
    "lm_error_test",
    # Spatial LM tests
    "lm_lag_test",
    "robust_lm_error_test",
    "robust_lm_lag_test",
    "run_lm_tests",
    "unit_root",
]
