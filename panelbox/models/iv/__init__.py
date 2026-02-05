"""
Instrumental Variables models for panel data.

This module provides IV/2SLS estimators for panel data with endogenous
regressors and instrumental variables.
"""

from panelbox.models.iv.panel_iv import PanelIV

__all__ = ["PanelIV"]
