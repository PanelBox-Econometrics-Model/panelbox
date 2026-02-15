"""
Sample selection models for panel data.

This module provides models for handling sample selection bias:
- Heckman two-step correction
- Maximum likelihood estimation
"""

from .heckman import PanelHeckman, PanelHeckmanResult

__all__ = ["PanelHeckman", "PanelHeckmanResult"]
