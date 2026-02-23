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
>>> result = robust_covariance(X, resid, method="HC1")
>>> print(result.std_errors)
>>>
>>> # Cluster by entity
>>> result = cluster_by_entity(X, resid, entity_ids)
>>> print(result.std_errors)
"""

from __future__ import annotations

# Clustered standard errors
from .clustered import (
    ClusteredCovarianceResult,
    ClusteredStandardErrors,
    cluster_by_entity,
    cluster_by_time,
    twoway_cluster,
)

# Standard Error Comparison
from .comparison import ComparisonResult, StandardErrorComparison

# Driscoll-Kraay standard errors
from .driscoll_kraay import DriscollKraayResult, DriscollKraayStandardErrors, driscoll_kraay

# Newey-West HAC standard errors
from .newey_west import NeweyWestResult, NeweyWestStandardErrors, newey_west

# Panel-Corrected Standard Errors (PCSE)
from .pcse import PanelCorrectedStandardErrors, PCSEResult, pcse

# Robust (HC) standard errors
from .robust import RobustCovarianceResult, RobustStandardErrors, robust_covariance

# Spatial HAC standard errors
from .spatial_hac import DriscollKraayComparison, SpatialHAC

# Utilities
from .utils import (
    clustered_covariance,
    compute_bread,
    compute_leverage,
    compute_meat_hc,
    hc_covariance,
    sandwich_covariance,
    twoway_clustered_covariance,
)

__all__ = [
    "ClusteredCovarianceResult",
    # Clustered SE
    "ClusteredStandardErrors",
    "ComparisonResult",
    "DriscollKraayComparison",
    "DriscollKraayResult",
    # Driscoll-Kraay SE
    "DriscollKraayStandardErrors",
    "NeweyWestResult",
    # Newey-West HAC SE
    "NeweyWestStandardErrors",
    "PCSEResult",
    # Panel-Corrected SE (PCSE)
    "PanelCorrectedStandardErrors",
    "RobustCovarianceResult",
    # Robust (HC) SE
    "RobustStandardErrors",
    # Spatial HAC SE
    "SpatialHAC",
    # Comparison
    "StandardErrorComparison",
    "cluster_by_entity",
    "cluster_by_time",
    "clustered_covariance",
    "compute_bread",
    # Utilities
    "compute_leverage",
    "compute_meat_hc",
    "driscoll_kraay",
    "hc_covariance",
    "newey_west",
    "pcse",
    "robust_covariance",
    "sandwich_covariance",
    "twoway_cluster",
    "twoway_clustered_covariance",
]
from typing import Annotated, Callable, ClassVar

MutantDict = Annotated[dict[str, Callable], "Mutant"]  # type: ignore


def _mutmut_trampoline(orig, mutants, call_args, call_kwargs, self_arg=None):  # type: ignore
    """Forward call to original or mutated function, depending on the environment."""
    import os  # type: ignore

    mutant_under_test = os.environ["MUTANT_UNDER_TEST"]  # type: ignore
    if mutant_under_test == "fail":  # type: ignore
        from mutmut.__main__ import MutmutProgrammaticFailException  # type: ignore

        raise MutmutProgrammaticFailException("Failed programmatically")  # type: ignore
    elif mutant_under_test == "stats":  # type: ignore
        from mutmut.__main__ import record_trampoline_hit  # type: ignore

        record_trampoline_hit(orig.__module__ + "." + orig.__name__)  # type: ignore
        # (for class methods, orig is bound and thus does not need the explicit self argument)
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    prefix = orig.__module__ + "." + orig.__name__ + "__mutmut_"  # type: ignore
    if not mutant_under_test.startswith(prefix):  # type: ignore
        result = orig(*call_args, **call_kwargs)  # type: ignore
        return result  # type: ignore
    mutant_name = mutant_under_test.rpartition(".")[-1]  # type: ignore
    if self_arg is not None:  # type: ignore
        # call to a class method where self is not bound
        result = mutants[mutant_name](self_arg, *call_args, **call_kwargs)  # type: ignore
    else:
        result = mutants[mutant_name](*call_args, **call_kwargs)  # type: ignore
    return result  # type: ignore
