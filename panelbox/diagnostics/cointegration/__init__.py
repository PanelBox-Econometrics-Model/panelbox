"""
Panel Cointegration Tests

This module provides state-of-the-art cointegration tests for panel data,
including Westerlund (2007) ECM-based tests, Pedroni (1999) residual-based
tests, and Kao (1999) DF-based tests.

References
----------
Westerlund, J. (2007). "Testing for Error Correction in Panel Data."
    Oxford Bulletin of Economics and Statistics, 69(6), 709-748.

Pedroni, P. (1999). "Critical Values for Cointegration Tests in
    Heterogeneous Panels with Multiple Regressors." Oxford Bulletin of
    Economics and Statistics, 61(S1), 653-670.

Kao, C. (1999). "Spurious Regression and Residual-Based Tests for
    Cointegration in Panel Data." Journal of Econometrics, 90(1), 1-44.
"""

from .kao import KaoResult, kao_test
from .pedroni import PedroniResult, pedroni_test
from .westerlund import WesterlundResult, westerlund_test

__all__ = [
    "westerlund_test",
    "WesterlundResult",
    "pedroni_test",
    "PedroniResult",
    "kao_test",
    "KaoResult",
]
