"""
Marginal effects for discrete choice models.

This module implements:
- Average Marginal Effects (AME)
- Marginal Effects at Means (MEM)
- Marginal Effects at Representative values (MER)
- Marginal effects for ordered choice models
"""

from panelbox.marginal_effects.delta_method import delta_method_se, numerical_gradient
from panelbox.marginal_effects.discrete_me import (
    MarginalEffectsResult,
    compute_ame,
    compute_mem,
    compute_mer,
)
from panelbox.marginal_effects.interactions import (
    InteractionEffectsResult,
    compute_interaction_effects,
    test_interaction_significance,
)
from panelbox.marginal_effects.ordered_me import (
    OrderedMarginalEffectsResult,
    compute_ordered_ame,
    compute_ordered_mem,
)

__all__ = [
    "compute_ame",
    "compute_mem",
    "compute_mer",
    "MarginalEffectsResult",
    "compute_ordered_ame",
    "compute_ordered_mem",
    "OrderedMarginalEffectsResult",
    "delta_method_se",
    "numerical_gradient",
    "compute_interaction_effects",
    "InteractionEffectsResult",
    "test_interaction_significance",
]
