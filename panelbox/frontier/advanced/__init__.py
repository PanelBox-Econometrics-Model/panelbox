"""
Advanced stochastic frontier models.

This module implements cutting-edge SFA models from the frontier literature,
including models that decompose inefficiency into persistent and transient components.

Models:
    FourComponentSFA: Kumbhakar et al. (2014) four-component model
        - Separates persistent inefficiency from transient inefficiency
        - Separates random heterogeneity from persistent inefficiency
        - Essential for policy-making: identify structural vs managerial inefficiency

References:
    Kumbhakar, S. C., Lien, G., & Hardaker, J. B. (2014).
        Technical efficiency in competing panel data models: a study of
        Norwegian grain farming. Journal of Productivity Analysis, 41(2), 321-337.

    Colombi, R., Kumbhakar, S. C., Martini, G., & Vittadini, G. (2014).
        Closed-skew normality in stochastic frontiers with individual effects
        and long/short-run efficiency. Journal of Productivity Analysis, 42, 123-136.

    Filippini, M., & Greene, W. H. (2016).
        Persistent and transient productive inefficiency: A maximum simulated
        likelihood approach. Journal of Productivity Analysis, 45(2), 187-196.
"""

from .four_component import FourComponentResult, FourComponentSFA

__all__ = [
    "FourComponentSFA",
    "FourComponentResult",
]
