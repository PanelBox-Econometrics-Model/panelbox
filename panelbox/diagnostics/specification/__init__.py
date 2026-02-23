"""
Specification testing for panel data models.

This module provides tools for testing model specification including:
- Davidson-MacKinnon J-test for non-nested models
- Encompassing tests (Cox, Wald, Likelihood Ratio)
"""

from __future__ import annotations

from .davidson_mackinnon import JTestResult, j_test
from .encompassing import (
    EncompassingResult,
    cox_test,
    likelihood_ratio_test,
    wald_encompassing_test,
)

__all__ = [
    "EncompassingResult",
    "JTestResult",
    "cox_test",
    "j_test",
    "likelihood_ratio_test",
    "wald_encompassing_test",
]
