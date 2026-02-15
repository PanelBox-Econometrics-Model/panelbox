"""
Sample selection models for panel data.

This module provides models for handling sample selection bias:
- Heckman two-step correction (Wooldridge 1995)
- Maximum likelihood estimation
- Inverse Mills Ratio computation
- Murphy-Topel variance correction
- Selection diagnostics
"""

from .heckman import PanelHeckman, PanelHeckmanResult
from .inverse_mills import compute_imr, imr_derivative, imr_diagnostics, test_selection_effect

__all__ = [
    "PanelHeckman",
    "PanelHeckmanResult",
    "compute_imr",
    "imr_derivative",
    "imr_diagnostics",
    "test_selection_effect",
]
