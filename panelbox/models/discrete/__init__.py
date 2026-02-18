"""
Discrete choice and limited dependent variable models for panel data.

This package contains implementations of nonlinear panel models including:
- Binary choice models (Logit, Probit)
- Fixed Effects Logit (Chamberlain 1980)
- Ordered choice models
- Count data models (Poisson, Negative Binomial)

Examples
--------
>>> import panelbox as pb
>>> data = pb.load_mroz()
>>>
>>> # Pooled Logit
>>> logit = pb.PooledLogit("lfp ~ age + educ + kids", data, "id", "year")
>>> results = logit.fit(cov_type='cluster')
>>> print(results.summary())
>>>
>>> # Fixed Effects Logit
>>> fe_logit = pb.FixedEffectsLogit("lfp ~ exper + kidslt6", data, "id", "year")
>>> fe_results = fe_logit.fit()
>>> print(fe_results.summary())
"""

from panelbox.models.discrete.base import NonlinearPanelModel
from panelbox.models.discrete.binary import (
    FixedEffectsLogit,
    PooledLogit,
    PooledProbit,
    RandomEffectsProbit,
)
from panelbox.models.discrete.multinomial import (
    ConditionalLogit,
    ConditionalLogitResult,
    MultinomialLogit,
    MultinomialLogitResult,
)
from panelbox.models.discrete.ordered import OrderedLogit, OrderedProbit, RandomEffectsOrderedLogit

__all__ = [
    "NonlinearPanelModel",
    "PooledLogit",
    "PooledProbit",
    "FixedEffectsLogit",
    "RandomEffectsProbit",
    "OrderedLogit",
    "OrderedProbit",
    "RandomEffectsOrderedLogit",
    "MultinomialLogit",
    "MultinomialLogitResult",
    "ConditionalLogit",
    "ConditionalLogitResult",
]
