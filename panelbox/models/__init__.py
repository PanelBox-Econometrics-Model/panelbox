"""
Panel econometric models.

This package contains implementations of various panel data estimators.
"""

from __future__ import annotations

from panelbox.models.static.pooled_ols import PooledOLS

__all__ = [
    "PooledOLS",
]
