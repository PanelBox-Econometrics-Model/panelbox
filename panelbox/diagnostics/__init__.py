"""
Panel data diagnostics and tests.

This module provides diagnostic tests for panel data models including:
- Hausman test for fixed vs random effects
- Panel unit root tests (Hadri, Breitung)
- Panel cointegration tests (Kao, Pedroni, Westerlund)
- Quantile regression diagnostics
- Spatial dependence tests (LM-Lag, LM-Error)
"""

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
    "cointegration",
    "unit_root",
    # Spatial LM tests
    "lm_lag_test",
    "lm_error_test",
    "robust_lm_lag_test",
    "robust_lm_error_test",
    "run_lm_tests",
    "LMTestResult",
    # Moran's I
    "MoranIPanelTest",
    "MoranIResult",
    # LISA
    "LocalMoranI",
    "LISAResult",
]
