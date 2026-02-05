"""
Standard errors and covariance matrix estimation for panel data.

This module provides various robust standard error estimators commonly
used in panel data econometrics:

- Heteroskedasticity-robust (HC0, HC1, HC2, HC3)
- Cluster-robust (one-way and two-way)
- Driscoll-Kraay (spatial and temporal dependence)
- Newey-West HAC (heteroskedasticity and autocorrelation consistent)

Examples
--------
>>> from panelbox.standard_errors import robust_covariance, cluster_by_entity
>>>
>>> # HC1 robust standard errors
>>> result = robust_covariance(X, resid, method='HC1')
>>> print(result.std_errors)
>>>
>>> # Cluster by entity
>>> result = cluster_by_entity(X, resid, entity_ids)
>>> print(result.std_errors)
"""

# Robust (HC) standard errors
from .robust import (
    RobustStandardErrors,
    RobustCovarianceResult,
    robust_covariance
)

# Clustered standard errors
from .clustered import (
    ClusteredStandardErrors,
    ClusteredCovarianceResult,
    cluster_by_entity,
    cluster_by_time,
    twoway_cluster
)

# Driscoll-Kraay standard errors
from .driscoll_kraay import (
    DriscollKraayStandardErrors,
    DriscollKraayResult,
    driscoll_kraay
)

# Newey-West HAC standard errors
from .newey_west import (
    NeweyWestStandardErrors,
    NeweyWestResult,
    newey_west
)

# Panel-Corrected Standard Errors (PCSE)
from .pcse import (
    PanelCorrectedStandardErrors,
    PCSEResult,
    pcse
)

# Standard Error Comparison
from .comparison import (
    StandardErrorComparison,
    ComparisonResult
)

# Utilities
from .utils import (
    compute_leverage,
    compute_bread,
    compute_meat_hc,
    sandwich_covariance,
    hc_covariance,
    clustered_covariance,
    twoway_clustered_covariance
)

__all__ = [
    # Robust (HC) SE
    'RobustStandardErrors',
    'RobustCovarianceResult',
    'robust_covariance',

    # Clustered SE
    'ClusteredStandardErrors',
    'ClusteredCovarianceResult',
    'cluster_by_entity',
    'cluster_by_time',
    'twoway_cluster',

    # Driscoll-Kraay SE
    'DriscollKraayStandardErrors',
    'DriscollKraayResult',
    'driscoll_kraay',

    # Newey-West HAC SE
    'NeweyWestStandardErrors',
    'NeweyWestResult',
    'newey_west',

    # Panel-Corrected SE (PCSE)
    'PanelCorrectedStandardErrors',
    'PCSEResult',
    'pcse',

    # Comparison
    'StandardErrorComparison',
    'ComparisonResult',

    # Utilities
    'compute_leverage',
    'compute_bread',
    'compute_meat_hc',
    'sandwich_covariance',
    'hc_covariance',
    'clustered_covariance',
    'twoway_clustered_covariance',
]
