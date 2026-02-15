"""
Panel data count models.

This module implements various count data models for panel data including:

1. Poisson Models:
   - PooledPoisson: Pooled Poisson with cluster-robust SEs
   - PoissonFixedEffects: Conditional MLE (Hausman, Hall, Griliches 1984)
   - RandomEffectsPoisson: Random effects with Gamma or Normal distribution
   - PoissonQML: Quasi-Maximum Likelihood (Wooldridge 1999)

2. Negative Binomial Models:
   - NegativeBinomial: NB2 model for overdispersed count data
   - NegativeBinomialFixedEffects: Fixed effects NB (Allison & Waterman 2002)

Key Features:
- Handles overdispersion through Negative Binomial models
- Efficient algorithms for conditional MLE (Fixed Effects)
- Quasi-ML estimation for robustness
- Automatic detection of overdispersion with warnings
- Comprehensive marginal effects calculation

References:
- Hausman, J., Hall, B. H., & Griliches, Z. (1984). "Econometric models for count
  data with an application to the patents-R&D relationship."
- Wooldridge, J. M. (1999). "Distribution-free estimation of some nonlinear
  panel data models."
- Cameron, A. C., & Trivedi, P. K. (2013). "Regression analysis of count data."
"""

from .negbin import FixedEffectsNegativeBinomial, NegativeBinomial
from .poisson import PoissonFixedEffects, PoissonQML, PooledPoisson, RandomEffectsPoisson
from .zero_inflated import (
    ZeroInflatedNegativeBinomial,
    ZeroInflatedNegativeBinomialResult,
    ZeroInflatedPoisson,
    ZeroInflatedPoissonResult,
)

__all__ = [
    # Poisson models
    "PooledPoisson",
    "PoissonFixedEffects",
    "RandomEffectsPoisson",
    "PoissonQML",
    # Negative Binomial models
    "NegativeBinomial",
    "FixedEffectsNegativeBinomial",
    # Zero-Inflated models
    "ZeroInflatedPoisson",
    "ZeroInflatedPoissonResult",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedNegativeBinomialResult",
]
