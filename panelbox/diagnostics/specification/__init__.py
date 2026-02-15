"""
Specification testing for panel data models.

This module provides tools for testing model specification including:
- Davidson-MacKinnon J-test for non-nested models
- Encompassing tests (Cox, Wald, Likelihood Ratio)
"""

from .davidson_mackinnon import JTestResult, j_test
from .encompassing import (
    EncompassingResult,
    cox_test,
    likelihood_ratio_test,
    wald_encompassing_test,
)

__all__ = [
    "j_test",
    "JTestResult",
    "cox_test",
    "wald_encompassing_test",
    "likelihood_ratio_test",
    "EncompassingResult",
]
