"""
Utility functions for stochastic frontier analysis.

This module provides tools for:
- TFP decomposition (Malmquist productivity index)
- Marginal effects on inefficiency determinants
- Other diagnostic and analysis utilities

Main Classes
------------
TFPDecomposition
    Decompose productivity growth into technical change, efficiency change,
    and scale effects.

Main Functions
--------------
marginal_effects
    Compute marginal effects of covariates on inefficiency or efficiency.

marginal_effects_summary
    Format marginal effects results as text table.

Example
-------
>>> from panelbox.frontier.utils import TFPDecomposition, marginal_effects
>>>
>>> # TFP decomposition
>>> tfp = TFPDecomposition(result)
>>> decomp = tfp.decompose()
>>> print(tfp.summary())
>>>
>>> # Marginal effects
>>> me = marginal_effects(result, method='mean')
>>> print(me)
"""

from .decomposition import TFPDecomposition
from .marginal_effects import (
    marginal_effects,
    marginal_effects_bc95,
    marginal_effects_summary,
    marginal_effects_wang_2002,
)

__all__ = [
    "TFPDecomposition",
    "marginal_effects",
    "marginal_effects_wang_2002",
    "marginal_effects_bc95",
    "marginal_effects_summary",
]
