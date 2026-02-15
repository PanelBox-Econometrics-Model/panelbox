"""
Quantile Regression for Panel Data.

Complete implementation of quantile regression methods including:
- Pooled QR with cluster-robust inference
- Fixed Effects QR (Koenker 2004)
- Canay (2011) two-step estimator
- Location-Scale models (MSS 2019)
- Dynamic panel QR
- Quantile treatment effects
- Non-crossing constraints and monotonicity

All models integrate seamlessly with PanelBox ecosystem.
"""

from .base import QuantilePanelModel, QuantilePanelResult
from .canay import CanayTwoStep
from .comparison import FEQuantileComparison
from .dynamic import DynamicQuantile
from .fixed_effects import FixedEffectsQuantile
from .location_scale import LocationScale
from .monotonicity import QuantileMonotonicity
from .pooled import PooledQuantile
from .treatment_effects import QuantileTreatmentEffects

__all__ = [
    "QuantilePanelModel",
    "QuantilePanelResult",
    "PooledQuantile",
    "FixedEffectsQuantile",
    "CanayTwoStep",
    "LocationScale",
    "DynamicQuantile",
    "QuantileTreatmentEffects",
    "QuantileMonotonicity",
    "FEQuantileComparison",
]
