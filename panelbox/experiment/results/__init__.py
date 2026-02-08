"""
Result containers for panel data tests.

This module provides result container classes for various panel data tests
and analyses. All result containers inherit from BaseResult.
"""

from panelbox.experiment.results.base import BaseResult
from panelbox.experiment.results.comparison_result import ComparisonResult
from panelbox.experiment.results.residual_result import ResidualResult
from panelbox.experiment.results.validation_result import ValidationResult

__all__ = ["BaseResult", "ValidationResult", "ComparisonResult", "ResidualResult"]
