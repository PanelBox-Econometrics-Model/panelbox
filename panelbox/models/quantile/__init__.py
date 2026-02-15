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
from .pooled import PooledQuantile, PooledQuantileResults

# Temporary: Comment out imports with missing dependencies
try:
    from .canay import CanayTwoStep
except ImportError:
    CanayTwoStep = None
try:
    from .comparison import FEQuantileComparison
except ImportError:
    FEQuantileComparison = None
try:
    from .dynamic import DynamicQuantile
except ImportError:
    DynamicQuantile = None
try:
    from .fixed_effects import FixedEffectsQuantile
except ImportError:
    FixedEffectsQuantile = None
try:
    from .location_scale import LocationScale
except ImportError:
    LocationScale = None
try:
    from .monotonicity import QuantileMonotonicity
except ImportError:
    QuantileMonotonicity = None
try:
    from .treatment_effects import QuantileTreatmentEffects
except ImportError:
    QuantileTreatmentEffects = None

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
